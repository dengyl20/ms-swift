# extract_features.py
from __future__ import annotations

import os
import io
import json
import time
import yaml
import math
import socket
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, Sampler

# -----------------------------
# 你提供的 Frozen encoders 依赖
# -----------------------------
from swift.llm.model.point_cloud.point_bert import PointBERTConfig, PointBERTEncoder
from transformers import AutoTokenizer, Qwen3OmniMoeThinkerForConditionalGeneration
import torch.nn as nn


# =============================
# YAML
# =============================
def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        return {}
    return cfg


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _parse_torch_dtype(s: str) -> torch.dtype:
    s = (s or "fp16").lower()
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp32", "float32", "float"):
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {s}")


def _parse_np_dtype_save(s: str) -> np.dtype:
    s = (s or "fp16").lower()
    if s in ("fp16", "float16", "half"):
        return np.float16
    if s in ("fp32", "float32", "float"):
        return np.float32
    # bfloat16 的 numpy memmap 原生不友好，这里不直接支持（可扩展成 uint16 自定义）
    raise ValueError(f"Unsupported save_dtype for memmap: {s} (recommend fp16/fp32)")


def set_seed(seed: int, rank: int) -> None:
    seed = int(seed) + int(rank)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================
# Manifest 构建
# =============================
def has_point_indicator(conversations: List[Dict[str, Any]], indicator: str) -> bool:
    for turn in conversations:
        v = str(turn.get("value", ""))
        if indicator in v:
            return True
    return False


def extract_gpt_text(conversations: List[Dict[str, Any]], strategy: str = "concat") -> str:
    gpts = []
    for turn in conversations:
        if str(turn.get("from", "")).lower() == "gpt":
            gpts.append(str(turn.get("value", "")))
    if not gpts:
        return ""
    strategy = (strategy or "concat").lower()
    if strategy == "first":
        return gpts[0]
    if strategy == "last":
        return gpts[-1]
    if strategy == "concat":
        return "\n".join(gpts)
    raise ValueError(f"Unknown gpt_text_strategy: {strategy}")


def build_manifest(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    dcfg = cfg["dataset"]
    anno_path = dcfg["anno_path"]
    point_root = dcfg["point_root"]
    pointnum = int(dcfg.get("pointnum", 8192))
    require_point = bool(dcfg.get("require_point_indicator", True))
    indicator = str(dcfg.get("point_indicator", "<point>"))
    conv_types = dcfg.get("conversation_types", []) or []
    conv_types = set([str(x) for x in conv_types]) if len(conv_types) > 0 else None
    gpt_strategy = str(dcfg.get("gpt_text_strategy", "concat"))
    verify_files = bool(dcfg.get("verify_point_files", True))
    max_samples = int(dcfg.get("max_samples", -1))

    with open(anno_path, "r") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"anno_path must be a JSON list, got: {type(raw)}")

    manifest: List[Dict[str, Any]] = []
    missing = 0
    filtered = 0

    for i, item in enumerate(raw):
        if max_samples > 0 and len(manifest) >= max_samples:
            break

        obj_id = str(item.get("object_id", ""))
        conv = item.get("conversations", [])
        if not obj_id or not isinstance(conv, list):
            filtered += 1
            continue

        # conversation_type filter (optional)
        if conv_types is not None:
            ctype = str(item.get("conversation_type", "simple_description"))
            if ctype not in conv_types:
                filtered += 1
                continue

        if require_point and (not has_point_indicator(conv, indicator)):
            filtered += 1
            continue

        point_path = os.path.join(point_root, f"{obj_id}_{pointnum}.npy")
        if verify_files and (not os.path.isfile(point_path)):
            missing += 1
            continue

        gpt_text = extract_gpt_text(conv, gpt_strategy)

        manifest.append(
            {
                "global_index": len(manifest),  # 重新编号，保证连续 [0..N-1]
                "object_id": obj_id,
                "point_path": point_path,
                "gpt_text": gpt_text,
                "conversation_type": item.get("conversation_type", "simple_description"),
            }
        )

    print(
        f"[manifest] loaded={len(raw)} kept={len(manifest)} "
        f"filtered={filtered} missing_point_files={missing}"
    )
    return manifest


# =============================
# 新增：按 tokenizer 的 token 长度过滤 manifest（丢弃 > max_text_len 的样本）
# =============================
def filter_manifest_by_text_token_len(
    manifest: List[Dict[str, Any]],
    text_cfg: Dict[str, Any],
    *,
    batch_size: int = 4096,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    丢弃 token_len > max_text_len 的样本（不截断保留）。

    注意：
    - token_len 的计算与训练/抽特征保持一致：同一个 tokenizer + add_special_tokens 设置。
    - 为了高效判断是否超过阈值，这里用 truncation=True, max_length=max_len+1，只需要知道是否 > max_len。
    """
    model_name = text_cfg["model_name_or_path"]
    tok_name = text_cfg.get("tokenizer_name_or_path", model_name)
    trust_remote_code = bool(text_cfg.get("trust_remote_code", True))

    max_len = int(text_cfg.get("max_text_len", 128))
    add_special_tokens = bool(text_cfg.get("add_special_tokens", False))

    if max_len <= 0:
        raise ValueError(f"encoders.text.max_text_len must be > 0, got {max_len}")

    print(
        f"[rank0] filtering manifest by text token length: "
        f"max_len={max_len}, add_special_tokens={add_special_tokens}, tokenizer={tok_name}"
    )

    tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=trust_remote_code)

    kept: List[Dict[str, Any]] = []
    dropped = 0

    n = len(manifest)
    t0 = time.time()

    # 用于过滤的编码上限：max_len+1
    # - 长度 <= max_len：编码长度就是实际长度
    # - 长度 >  max_len：编码会被截到 max_len+1（>= max_len+1），据此丢弃即可
    detect_len = max_len + 1

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        texts = [str(manifest[i].get("gpt_text", "")) for i in range(start, end)]

        enc = tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=detect_len,
            add_special_tokens=add_special_tokens,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_length=True,
        )

        lens = enc.get("length", None)
        if lens is None:
            # 兼容极少数 tokenizer 不返回 length 的情况
            input_ids = enc["input_ids"]
            lens = [len(x) for x in input_ids]

        for i, l in enumerate(lens):
            if int(l) <= max_len:
                m = dict(manifest[start + i])
                m["text_token_len"] = int(l)
                kept.append(m)
            else:
                dropped += 1

        # 简单进度打印
        if (end == n) or ((end // batch_size) % 50 == 0):
            elapsed = time.time() - t0
            print(
                f"[rank0] text_len_filter progress: {end}/{n} "
                f"kept={len(kept)} dropped={dropped} elapsed={elapsed:.1f}s"
            )

    # 重新连续编号 global_index，并保留过滤前编号用于追溯
    for new_i, m in enumerate(kept):
        m["global_index_before_text_filter"] = int(m.get("global_index", new_i))
        m["global_index"] = int(new_i)

    print(
        f"[rank0] text_len_filter done: before={n} after={len(kept)} dropped={dropped} "
        f"drop_ratio={dropped / max(n, 1):.6f}"
    )
    stats = {"before": n, "after": len(kept), "dropped": dropped, "max_len": max_len}
    return kept, stats


# =============================
# Dataset: raw (load npy + pc_norm)
# =============================
def pc_norm_np(pc: np.ndarray) -> np.ndarray:
    """
    pc: (N,C), C>=3
    对 xyz 做中心化 + scale 到 unit sphere; 其余 feature 不变
    """
    pc = pc.astype(np.float32, copy=False)
    xyz = pc[:, :3]
    other = pc[:, 3:] if pc.shape[1] > 3 else None

    centroid = xyz.mean(axis=0, dtype=np.float32)
    xyz = xyz - centroid[None, :]

    dist2 = (xyz * xyz).sum(axis=1, dtype=np.float32)
    m = np.sqrt(dist2).max()
    if not np.isfinite(m) or m < 1e-6:
        m = 1.0
    xyz = xyz / m

    if other is not None and other.size > 0:
        out = np.concatenate([xyz, other], axis=1)
    else:
        out = xyz
    return out


class RawPointTextPairDataset(Dataset):
    def __init__(
        self,
        manifest: List[Dict[str, Any]],
        use_color: bool = True,
        normalize_pc: bool = True,
        on_error: str = "zero",  # zero | raise
    ):
        super().__init__()
        self.manifest = manifest
        self.use_color = bool(use_color)
        self.normalize_pc = bool(normalize_pc)
        self.on_error = str(on_error).lower()

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        m = self.manifest[idx]
        obj_id = m["object_id"]
        point_path = m["point_path"]
        gpt_text = m.get("gpt_text", "")
        valid = True
        err_msg = ""

        try:
            pc = np.load(point_path)  # expected (8192,6)
            if pc.ndim != 2 or pc.shape[0] <= 0:
                raise ValueError(f"bad pc shape: {pc.shape}")

            if not self.use_color:
                pc = pc[:, :3]

            if self.normalize_pc:
                pc = pc_norm_np(pc)

            # 防 NaN/Inf
            if not np.isfinite(pc).all():
                pc = np.nan_to_num(pc, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        except Exception as e:
            valid = False
            err_msg = f"{type(e).__name__}: {e}"
            if self.on_error == "raise":
                raise
            # zero fallback
            # 默认按 (8192,6) 给零；真实维度由上层配置决定，这里尽量保守
            pc = np.zeros((8192, 6), dtype=np.float32)

        return {
            "global_index": int(m["global_index"]),
            "object_id": obj_id,
            "gpt_text": gpt_text,
            "point_cloud": torch.from_numpy(pc.astype(np.float32, copy=False)),  # CPU tensor
            "valid": bool(valid),
            "error": err_msg,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # 默认保持顺序
    point_clouds = torch.stack([b["point_cloud"] for b in batch], dim=0)  # (B,N,C)
    texts = [b["gpt_text"] for b in batch]
    object_ids = [b["object_id"] for b in batch]
    global_indices = torch.tensor([b["global_index"] for b in batch], dtype=torch.long)
    valid = torch.tensor([1 if b["valid"] else 0 for b in batch], dtype=torch.uint8)
    errors = [b.get("error", "") for b in batch]
    return {
        "point_clouds": point_clouds,
        "texts": texts,
        "object_ids": object_ids,
        "global_indices": global_indices,
        "valid": valid,
        "errors": errors,
    }


# =============================
# Sampler: 不 padding，不重复
# =============================
class IndexSampler(Sampler[int]):
    def __init__(self, indices: List[int]):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def split_indices_no_pad(n: int, rank: int, world_size: int, mode: str = "strided") -> List[int]:
    """
    mode:
      - strided: rank, rank+world_size, ...
      - contiguous: 按块切
    """
    if world_size <= 1:
        return list(range(n))
    mode = (mode or "strided").lower()
    if mode == "strided":
        return list(range(rank, n, world_size))
    if mode == "contiguous":
        # 近似均分
        per = (n + world_size - 1) // world_size
        start = rank * per
        end = min(n, start + per)
        return list(range(start, end))
    raise ValueError(f"Unknown split mode: {mode}")


# =============================
# Frozen Encoders (PointBERT)
# =============================
class FrozenPointBERTTokens(nn.Module):
    """
    输入 raw points: (B,8192,6)
    输出 tokens:
      - drop_cls=True:  (B, num_group, trans_dim)
      - drop_cls=False: (B, num_group+1, trans_dim)
    """

    def __init__(self, cfg: Dict[str, Any], device: torch.device):
        super().__init__()
        self.device = device

        ckpt_path = cfg["ckpt_path"]
        drop_cls = bool(cfg.get("drop_cls", True))
        self.drop_cls = drop_cls

        input_dtype = (cfg.get("input_dtype", "fp32") or "fp32").lower()
        if input_dtype not in ("fp16", "fp32", "bf16"):
            raise ValueError("point_bert.input_dtype must be one of: fp16, bf16, fp32")
        self.input_dtype = input_dtype

        cfg_dict = cfg.get("config", {})
        pb_cfg = PointBERTConfig(**cfg_dict)

        self.encoder = PointBERTEncoder(pb_cfg, use_max_pool=False)
        self.encoder.load_checkpoint(ckpt_path, strict=True, map_location="cuda", verbose=True)

        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.encoder.to(device)

        self.trans_dim = int(pb_cfg.trans_dim)
        self.num_group = int(pb_cfg.num_group)

    @torch.inference_mode()
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        points = points.to(self.device, non_blocking=True)

        if self.input_dtype == "fp16":
            points = points.half()
        elif self.input_dtype == "bf16":
            points = points.bfloat16()
        else:
            points = points.float()

        tokens = self.encoder(points, return_tokens=True)  # (B,G+1,trans_dim)
        if self.drop_cls:
            tokens = tokens[:, 1:, :]  # (B,G,trans_dim)
        return tokens


# =============================
# Frozen Text Embedding (Qwen) - 推荐缓存 weight 避免重复加载整模型
# =============================
def extract_and_cache_qwen_embedding_weight(text_cfg: Dict[str, Any], save_path: str) -> None:
    """
    rank0 执行：加载一次 Qwen3 全模型（CPU），抽取 embedding weight，保存到 save_path。
    """
    ensure_dir(os.path.dirname(save_path))
    model_name = text_cfg["model_name_or_path"]
    trust_remote_code = bool(text_cfg.get("trust_remote_code", True))
    torch_dtype = _parse_torch_dtype(text_cfg.get("torch_dtype", "fp16"))

    print(f"[rank0] extracting Qwen embedding weight from: {model_name}")
    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    emb = model.get_input_embeddings().weight.detach().cpu()
    # 仅保存 weight，避免保存大模型
    payload = {
        "weight": emb,
        "dtype": str(emb.dtype),
        "vocab_size": int(emb.shape[0]),
        "hidden_size": int(emb.shape[1]),
        "model_name_or_path": model_name,
    }
    torch.save(payload, save_path)
    del model
    print(f"[rank0] saved embedding weight to: {save_path}")


class FrozenQwenEmbeddingTableFromWeight(nn.Module):
    """
    给定 texts(list[str]) -> (embeddings, mask)
      embeddings: (B, max_len, hidden)
      mask:       (B, max_len) bool
    从缓存的 embedding weight 加载，不重复加载整模型。
    """

    def __init__(self, cfg: Dict[str, Any], device: torch.device):
        super().__init__()
        self.device = device

        model_name = cfg["model_name_or_path"]
        tok_name = cfg.get("tokenizer_name_or_path", model_name)
        trust_remote_code = bool(cfg.get("trust_remote_code", True))

        self.max_text_len = int(cfg.get("max_text_len", 128))
        self.add_special_tokens = bool(cfg.get("add_special_tokens", False))

        self.tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # 兜底
                self.tokenizer.pad_token = self.tokenizer.convert_ids_to_tokens(0)

        weight_cache = cfg.get("embedding_weight_cache", None)
        if not weight_cache or (not os.path.isfile(weight_cache)):
            raise FileNotFoundError(
                f"embedding_weight_cache not found: {weight_cache}. "
                "Please let rank0 extract it first (script will do it if configured)."
            )

        payload = torch.load(weight_cache, map_location="cpu")
        weight = payload["weight"]
        self.hidden_size = int(weight.shape[1])

        self.embed = nn.Embedding.from_pretrained(weight, freeze=True).to(self.device)
        self.embed.eval()
        for p in self.embed.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def forward(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        # 关键修改：
        # - truncation=False，确保不会对超长文本“悄悄截断并保留”
        # - 超长文本应当在 rank0 manifest 过滤阶段被丢弃；如果仍出现，这里会直接报错，便于定位问题。
        enc = self.tokenizer(
            texts,
            padding="max_length",
            truncation=False,
            max_length=self.max_text_len,
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device, non_blocking=True)
        attn_mask = enc["attention_mask"].to(self.device, non_blocking=True).bool()

        emb = self.embed(input_ids)  # (B, max_len, hidden)
        return emb, attn_mask


# =============================
# Memmap Writer
# =============================
@dataclass
class ShardPaths:
    text_embeds: str
    text_mask: str
    point_tokens: str
    object_ids: str
    global_indices: str
    valid: str


class MemmapShardWriter:
    def __init__(
        self,
        out_dir: str,
        rank: int,
        num_samples: int,
        text_shape: Tuple[int, int],        # (max_len, hidden)
        point_shape: Tuple[int, int],       # (G, trans_dim)
        save_dtype: np.dtype,
    ):
        self.rank = rank
        self.num_samples = int(num_samples)
        self.save_dtype = save_dtype

        shard_dir = os.path.join(out_dir, "shards")
        ensure_dir(shard_dir)

        self.paths = ShardPaths(
            text_embeds=os.path.join(shard_dir, f"text_embeds_rank{rank:02d}.mmap"),
            text_mask=os.path.join(shard_dir, f"text_mask_rank{rank:02d}.mmap"),
            point_tokens=os.path.join(shard_dir, f"point_tokens_rank{rank:02d}.mmap"),
            object_ids=os.path.join(shard_dir, f"object_ids_rank{rank:02d}.mmap"),
            global_indices=os.path.join(shard_dir, f"global_indices_rank{rank:02d}.mmap"),
            valid=os.path.join(shard_dir, f"valid_rank{rank:02d}.mmap"),
        )

        max_len, hidden = text_shape
        G, trans_dim = point_shape

        # 注意：mode='w+' 会覆盖旧文件（上层需用 overwrite 控制）
        self.text_embeds = np.memmap(self.paths.text_embeds, mode="w+", dtype=save_dtype,
                                     shape=(self.num_samples, max_len, hidden))
        self.text_mask = np.memmap(self.paths.text_mask, mode="w+", dtype=np.uint8,
                                   shape=(self.num_samples, max_len))
        self.point_tokens = np.memmap(self.paths.point_tokens, mode="w+", dtype=save_dtype,
                                      shape=(self.num_samples, G, trans_dim))

        # object_id 固定 32 字符 hex，存 bytes (S32)
        self.object_ids = np.memmap(self.paths.object_ids, mode="w+", dtype="S32",
                                    shape=(self.num_samples,))
        self.global_indices = np.memmap(self.paths.global_indices, mode="w+", dtype=np.int64,
                                        shape=(self.num_samples,))
        self.valid = np.memmap(self.paths.valid, mode="w+", dtype=np.uint8,
                               shape=(self.num_samples,))

        self._ptr = 0

    def write_batch(
        self,
        text_emb: torch.Tensor,     # (B, L, H) on GPU
        text_mask: torch.Tensor,    # (B, L) bool on GPU
        point_tokens: torch.Tensor, # (B, G, D) on GPU
        object_ids: List[str],
        global_indices: torch.Tensor,  # (B,)
        valid: torch.Tensor,           # (B,) uint8
    ) -> None:
        b = text_emb.shape[0]
        s = self._ptr
        e = s + b
        if e > self.num_samples:
            raise RuntimeError(f"ShardWriter overflow: ptr={s}, batch={b}, num_samples={self.num_samples}")

        # GPU -> CPU -> numpy
        te = text_emb.detach().cpu().numpy()
        tm = text_mask.detach().cpu().numpy().astype(np.uint8)
        pt = point_tokens.detach().cpu().numpy()
        gi = global_indices.detach().cpu().numpy().astype(np.int64)
        va = valid.detach().cpu().numpy().astype(np.uint8)

        self.text_embeds[s:e] = te
        self.text_mask[s:e] = tm
        self.point_tokens[s:e] = pt
        self.global_indices[s:e] = gi
        self.valid[s:e] = va

        # object_ids
        for i, oid in enumerate(object_ids):
            self.object_ids[s + i] = np.bytes_(oid.encode("utf-8"))

        self._ptr = e

    def flush(self) -> None:
        self.text_embeds.flush()
        self.text_mask.flush()
        self.point_tokens.flush()
        self.object_ids.flush()
        self.global_indices.flush()
        self.valid.flush()

    def close(self) -> None:
        self.flush()


# =============================
# distributed helpers
# =============================
def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def is_torchrun_env() -> bool:
    return ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ)


def ddp_init_if_needed(rank: int, world_size: int, backend: str, init_method: Optional[str]) -> None:
    if world_size <= 1:
        return
    if dist.is_initialized():
        return
    dist.init_process_group(
        backend=backend,
        init_method=init_method or "env://",
        world_size=world_size,
        rank=rank,
    )


def ddp_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


# =============================
# main worker
# =============================
def worker_main(
    rank: int,
    world_size: int,
    cfg: Dict[str, Any],
    init_method: Optional[str],
) -> None:
    # 环境建议：tokenizers 并行会在多进程下导致线程爆炸，通常建议关掉
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    backend = str(cfg.get("distributed", {}).get("backend", "nccl"))
    seed = int(cfg.get("seed", 1234))

    # local_rank：torchrun 时用 LOCAL_RANK；spawn 时 rank==local_rank
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    ddp_init_if_needed(rank, world_size, backend, init_method)
    set_seed(seed, rank)

    dcfg = cfg["dataset"]
    rcfg = cfg["runtime"]
    ecfg = cfg["encoders"]
    out_dir = rcfg["output_dir"]
    overwrite = bool(rcfg.get("overwrite", False))

    ensure_dir(out_dir)
    cache_dir = os.path.join(out_dir, "cache")
    ensure_dir(cache_dir)

    manifest_pt = os.path.join(out_dir, "manifest.pt")

    # ---- rank0: build manifest + (新增)按 token 长度过滤 + cache qwen embedding weight ----
    text_len_filter_stats: Optional[Dict[str, int]] = None

    if rank == 0:
        if (not overwrite) and (os.path.exists(os.path.join(out_dir, "dataset_info.yaml"))):
            raise FileExistsError(
                f"Output already exists: {out_dir}. "
                "Set runtime.overwrite=true to overwrite."
            )

        manifest = build_manifest(cfg)

        # ===== 新增：丢弃 token_len > max_text_len 的样本 =====
        # 你想把 max_length 设为 24 并丢弃超长样本，那么必须在这里过滤，
        # 否则后续 tokenizer 会 truncation=True 截断并保留（或者即使你改成 truncation=False 也会直接报错）。
        text_cfg_rank0 = ecfg["text"]
        manifest, text_len_filter_stats = filter_manifest_by_text_token_len(
            manifest,
            text_cfg_rank0,
            batch_size=int(text_cfg_rank0.get("length_filter_batch_size", 4096)),
        )

        torch.save(manifest, manifest_pt)
        print(f"[rank0] manifest saved to: {manifest_pt}  (N={len(manifest)})")

        # cache qwen embedding weight
        text_cfg = ecfg["text"]
        weight_cache = text_cfg.get("embedding_weight_cache", os.path.join(cache_dir, "qwen_embed_weight.pt"))
        text_cfg["embedding_weight_cache"] = weight_cache  # 回写，供其它 rank 使用
        if not os.path.isfile(weight_cache):
            extract_and_cache_qwen_embedding_weight(text_cfg, weight_cache)
        else:
            print(f"[rank0] found cached embedding weight: {weight_cache}")

    ddp_barrier()

    # ---- all ranks: load manifest ----
    manifest: List[Dict[str, Any]] = torch.load(manifest_pt, map_location="cpu")
    n_total = len(manifest)

    # ---- split indices ----
    split_mode = str(cfg.get("distributed", {}).get("split_mode", "strided"))
    indices = split_indices_no_pad(n_total, rank, world_size, mode=split_mode)
    n_local = len(indices)
    if rank == 0:
        print(f"[split] mode={split_mode}, total={n_total}, world_size={world_size}")

    print(f"[rank{rank}] local samples: {n_local}")

    # ---- dataset / dataloader ----
    dataset = RawPointTextPairDataset(
        manifest=manifest,
        use_color=bool(dcfg.get("use_color", True)),
        normalize_pc=bool(dcfg.get("normalize_pc", True)),
        on_error=str(dcfg.get("on_error", "zero")),
    )
    sampler = IndexSampler(indices)

    loader = DataLoader(
        dataset,
        batch_size=int(rcfg.get("batch_size", 32)),
        sampler=sampler,
        num_workers=int(rcfg.get("num_workers", 8)),
        collate_fn=collate_fn,
        pin_memory=bool(rcfg.get("pin_memory", True)),
        persistent_workers=bool(rcfg.get("persistent_workers", True)),
        prefetch_factor=int(rcfg.get("prefetch_factor", 4)),
        drop_last=False,
    )

    # ---- encoders ----
    # point
    point_encoder = FrozenPointBERTTokens(ecfg["point"], device=device)

    # text (from cached weight)
    text_cfg = ecfg["text"]
    weight_cache = text_cfg.get("embedding_weight_cache", os.path.join(cache_dir, "qwen_embed_weight.pt"))
    text_cfg["embedding_weight_cache"] = weight_cache
    text_encoder = FrozenQwenEmbeddingTableFromWeight(text_cfg, device=device)

    max_len = int(text_encoder.max_text_len)
    hidden = int(text_encoder.hidden_size)
    G = int(point_encoder.num_group) if point_encoder.drop_cls else int(point_encoder.num_group + 1)
    trans_dim = int(point_encoder.trans_dim)

    save_dtype = _parse_np_dtype_save(rcfg.get("save_dtype", "fp16"))

    # ---- writer ----
    # overwrite 控制：如果文件存在且不 overwrite，直接报错
    shard_dir = os.path.join(out_dir, "shards")
    ensure_dir(shard_dir)
    test_path = os.path.join(shard_dir, f"text_embeds_rank{rank:02d}.mmap")
    if (not overwrite) and os.path.exists(test_path):
        raise FileExistsError(f"Shard already exists: {test_path}. Set runtime.overwrite=true to overwrite.")

    writer = MemmapShardWriter(
        out_dir=out_dir,
        rank=rank,
        num_samples=n_local,
        text_shape=(max_len, hidden),
        point_shape=(G, trans_dim),
        save_dtype=save_dtype,
    )

    # ---- inference loop ----
    log_every = int(rcfg.get("log_every", 20))
    flush_every = int(rcfg.get("flush_every", 50))

    torch.backends.cudnn.benchmark = False

    t0 = time.time()
    num_done = 0

    for step, batch in enumerate(loader):
        point_clouds = batch["point_clouds"]        # CPU (B,N,C)
        texts = batch["texts"]
        object_ids = batch["object_ids"]
        global_indices = batch["global_indices"]    # CPU tensor
        valid = batch["valid"]                      # CPU uint8

        # 推理
        with torch.inference_mode():
            point_tokens = point_encoder(point_clouds)      # on GPU
            text_emb, text_mask = text_encoder(texts)       # on GPU

            # 保存精度（memmap 支持 fp16/fp32）
            if save_dtype == np.float16:
                point_tokens = point_tokens.to(dtype=torch.float16)
                text_emb = text_emb.to(dtype=torch.float16)
            elif save_dtype == np.float32:
                point_tokens = point_tokens.to(dtype=torch.float32)
                text_emb = text_emb.to(dtype=torch.float32)

        writer.write_batch(
            text_emb=text_emb,
            text_mask=text_mask,
            point_tokens=point_tokens,
            object_ids=object_ids,
            global_indices=global_indices,
            valid=valid,
        )

        num_done += len(object_ids)

        if flush_every > 0 and (step + 1) % flush_every == 0:
            writer.flush()

        if log_every > 0 and (step + 1) % log_every == 0:
            elapsed = time.time() - t0
            speed = num_done / max(elapsed, 1e-9)
            print(f"[rank{rank}] step={step+1} done={num_done}/{n_local} speed={speed:.2f} samples/s")

    writer.close()

    elapsed = time.time() - t0
    speed = num_done / max(elapsed, 1e-9)
    print(f"[rank{rank}] finished. local={n_local} time={elapsed:.1f}s speed={speed:.2f} samples/s")

    # ---- write shard info ----
    shard_info = {
        "rank": rank,
        "num_samples": n_local,
        "text": {"max_len": max_len, "hidden": hidden, "dtype": str(save_dtype)},
        "point": {"num_tokens": G, "trans_dim": trans_dim, "dtype": str(save_dtype)},
        "paths": {
            "text_embeds": writer.paths.text_embeds,
            "text_mask": writer.paths.text_mask,
            "point_tokens": writer.paths.point_tokens,
            "object_ids": writer.paths.object_ids,
            "global_indices": writer.paths.global_indices,
            "valid": writer.paths.valid,
        },
    }
    shard_info_path = os.path.join(out_dir, f"shard_info_rank{rank:02d}.yaml")
    with open(shard_info_path, "w") as f:
        yaml.safe_dump(shard_info, f, sort_keys=False)

    ddp_barrier()

    # ---- rank0: merge dataset_info.yaml ----
    if rank == 0:
        shards = []
        for r in range(world_size):
            p = os.path.join(out_dir, f"shard_info_rank{r:02d}.yaml")
            with open(p, "r") as f:
                shards.append(yaml.safe_load(f))

        dataset_info = {
            "version": 1,
            "num_samples_total": int(n_total),
            "world_size": int(world_size),
            "split_mode": split_mode,
            "dataset": {
                "anno_path": dcfg["anno_path"],
                "point_root": dcfg["point_root"],
                "pointnum": int(dcfg.get("pointnum", 8192)),
                "use_color": bool(dcfg.get("use_color", True)),
                "normalize_pc": bool(dcfg.get("normalize_pc", True)),
                "require_point_indicator": bool(dcfg.get("require_point_indicator", True)),
                "point_indicator": str(dcfg.get("point_indicator", "<point>")),
                "gpt_text_strategy": str(dcfg.get("gpt_text_strategy", "concat")),
            },
            "features": {
                "text": {"max_len": max_len, "hidden": hidden, "dtype": str(save_dtype)},
                "point": {"num_tokens": G, "trans_dim": trans_dim, "dtype": str(save_dtype)},
            },
            # 新增：记录文本长度过滤统计（如果需要追溯丢了多少样本）
            "text_length_filter": text_len_filter_stats,
            "shards": shards,
        }
        info_path = os.path.join(out_dir, "dataset_info.yaml")
        with open(info_path, "w") as f:
            yaml.safe_dump(dataset_info, f, sort_keys=False)
        print(f"[rank0] dataset_info saved to: {info_path}")

    ddp_barrier()

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    # 不使用命令行参数：默认读 configs/extract_features.yaml
    cfg_path = os.environ.get("MM_CFG", os.path.join("configs", "extract_features.yaml"))
    cfg = load_yaml(cfg_path)
    if not cfg:
        raise ValueError(f"Empty config: {cfg_path}")

    # torchrun 环境
    if is_torchrun_env():
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        # torchrun 用 env://
        worker_main(rank=rank, world_size=world_size, cfg=cfg, init_method="env://")
        return

    # 否则：脚本内 spawn 单机多卡
    num_gpus = int(cfg.get("distributed", {}).get("num_gpus", torch.cuda.device_count()))
    if num_gpus <= 1:
        worker_main(rank=0, world_size=1, cfg=cfg, init_method=None)
        return

    port = _find_free_port()
    init_method = f"tcp://127.0.0.1:{port}"
    mp.spawn(
        fn=worker_main,
        args=(num_gpus, cfg, init_method),
        nprocs=num_gpus,
        join=True,
    )


if __name__ == "__main__":
    main()
