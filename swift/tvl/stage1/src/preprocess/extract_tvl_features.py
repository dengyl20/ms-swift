from __future__ import annotations

import json
import os
import socket
import sys
import time
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from rich.pretty import pprint
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoTokenizer, Qwen3OmniMoeThinkerForConditionalGeneration
import traceback
# -----------------------------------------------------------------------------
# Import TVL touch preprocessing (the file you created in ms-swift)
# -----------------------------------------------------------------------------
from swift.tvl.stage1.src.preprocess.tvl_touch_encoder import (  # noqa: E402
        TVLTouchPreprocessConfig,
        TVLTouchPreprocessor,
    )


class FrozenQwenEmbeddingTableFromWeight(torch.nn.Module):
    def __init__(self, cfg: Dict[str, Any], device: torch.device):
        super().__init__()
        model_name = cfg["model_name_or_path"]
        tok_name = cfg.get("tokenizer_name_or_path", model_name)
        self.max_text_len = int(cfg.get("max_text_len", 128))
        self.add_special_tokens = bool(cfg.get("add_special_tokens", False))
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=bool(cfg.get("trust_remote_code", True)))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.convert_ids_to_tokens(0)

        cache = cfg.get("embedding_weight_cache")
        if not cache or (not os.path.isfile(cache)):
            raise FileNotFoundError(f"embedding_weight_cache not found: {cache}")
        payload = torch.load(cache, map_location="cpu")
        self.hidden_size = int(payload["hidden_size"])
        self.embed = torch.nn.Embedding.from_pretrained(payload["weight"], freeze=True).to(self.device)

    @torch.inference_mode()
    def forward(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt",
        )
        ids = enc["input_ids"].to(self.device, non_blocking=True)
        mask = enc["attention_mask"].to(self.device, non_blocking=True).bool()
        emb = self.embed(ids)
        return emb, mask


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def ensure_dir(path: str) -> None:
    if path:
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
    raise ValueError(f"Unsupported save_dtype: {s}")


def set_seed(seed: int, rank: int) -> None:
    base = int(seed) + int(rank)
    random.seed(base)
    np.random.seed(base)
    torch.manual_seed(base)
    torch.cuda.manual_seed_all(base)


def _first_present(d: Dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for k in keys:
        v = d.get(k, None)
        if v is not None and str(v) != "":
            return v
    return None


def _resolve_path(raw: Optional[str], roots: Sequence[str], base_dir: str) -> Optional[str]:
    if raw is None:
        return None
    p = str(raw)
    if os.path.isabs(p) and os.path.exists(p):
        return p
    cand = os.path.join(base_dir, p)
    if os.path.exists(cand):
        return cand
    for r in roots:
        cand = os.path.join(r, p)
        if os.path.exists(cand):
            return cand
    return None


# -----------------------------------------------------------------------------
# Text extraction (keep your logic)
# -----------------------------------------------------------------------------
def _extract_text(item: Dict[str, Any], text_keys: Sequence[str], gpt_text_strategy: str) -> str:
    direct = _first_present(item, text_keys)
    if isinstance(direct, str) and direct.strip():
        return direct

    conv = item.get("conversations", None)
    if isinstance(conv, list):
        gpt_list: List[str] = []
        for turn in conv:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("from", "")).lower()
            if role in ("gpt", "assistant"):
                gpt_list.append(str(turn.get("value", "")))
        if not gpt_list:
            return ""
        strategy = (gpt_text_strategy or "first").lower()
        if strategy == "first":
            return gpt_list[0]
        if strategy == "last":
            return gpt_list[-1]
        if strategy == "concat":
            return "\n".join(gpt_list)
    return ""


# -----------------------------------------------------------------------------
# Manifest (touch + text + OPTIONAL tactile_background)
# -----------------------------------------------------------------------------
def build_manifest(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    dcfg = cfg["dataset"]
    meta_paths = dcfg.get("meta_paths", [])
    if not meta_paths:
        raise ValueError("dataset.meta_paths is required")

    touch_key_candidates = dcfg.get("touch_key_candidates", ["touch_path", "touch", "tactile_path", "tactile"])
    touch_bg_key_candidates = dcfg.get(
        "touch_bg_key_candidates",
        ["tactile_background", "touch_background", "background_tactile", "tactile_bg", "tactile_background_path"],
    )
    text_key_candidates = dcfg.get("text_key_candidates", ["text", "caption", "description"])
    id_key_candidates = dcfg.get("id_key_candidates", ["id", "sample_id", "uid", "object_id"])

    touch_roots = dcfg.get("touch_roots", [])
    verify_files = bool(dcfg.get("verify_files", True))
    gpt_text_strategy = str(dcfg.get("gpt_text_strategy", "first"))
    max_samples = int(dcfg.get("max_samples", -1))

    manifest: List[Dict[str, Any]] = []
    total = 0
    filtered = 0
    missing = 0
    missing_bg = 0

    pprint(f"len(meta_paths)={len(meta_paths)}")
    for meta_path in meta_paths:
        with open(meta_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            raise ValueError(f"{meta_path} should be list json")

        base_dir = os.path.dirname(meta_path)
        for item in raw:
            total += 1
            if max_samples > 0 and len(manifest) >= max_samples:
                break
            if not isinstance(item, dict):
                filtered += 1
                continue

            sample_id = str(_first_present(item, id_key_candidates) or f"sample_{len(manifest)}")
            text = _extract_text(item, text_key_candidates, gpt_text_strategy)

            touch_path = _resolve_path(_first_present(item, touch_key_candidates), touch_roots, base_dir)
            touch_bg_path = _resolve_path(_first_present(item, touch_bg_key_candidates), touch_roots, base_dir)

            # require touch + text
            if (not text) or (not touch_path):
                filtered += 1
                continue

            if verify_files:
                if touch_path is None or (not os.path.isfile(touch_path)):
                    missing += 1
                    continue
                # bg 文件不是所有数据都有：这里只统计，不强制过滤
                if touch_bg_path is not None and (not os.path.isfile(touch_bg_path)):
                    missing_bg += 1
                    touch_bg_path = None

            manifest.append(
                {
                    "global_index": len(manifest),
                    "sample_id": sample_id,
                    "touch_path": touch_path,
                    "touch_bg_path": touch_bg_path,  # can be None
                    "text": text,
                    "source_meta": meta_path,
                }
            )

    pprint(
        f"[manifest] loaded={total} kept={len(manifest)} "
        f"filtered={filtered} missing_touch_files={missing} missing_bg_files={missing_bg}"
    )
    return manifest


# -----------------------------------------------------------------------------
# Dataset / Collate
#   IMPORTANT: touch tensor is produced using TVLTouchPreprocessor (tacvis.load_tactile_data etc.)
# -----------------------------------------------------------------------------
class TVLTouchTextDataset(Dataset):
    """
    返回 tactile tensor (3,224,224) + text
    - worker 内 lazy 初始化 preprocessor，避免多进程/多worker下 import/path 问题
    """

    def __init__(self, manifest: List[Dict[str, Any]], touch_preprocess_cfg: Dict[str, Any], on_error: str = "zero"):
        super().__init__()
        self.manifest = manifest
        self.on_error = (on_error or "zero").lower()
        self.touch_preprocess_cfg = dict(touch_preprocess_cfg)
        self._pre: Optional[TVLTouchPreprocessor] = None

    def _get_pre(self) -> TVLTouchPreprocessor:
        if self._pre is None:
            cfg = TVLTouchPreprocessConfig(**self.touch_preprocess_cfg)
            self._pre = TVLTouchPreprocessor(cfg)
        return self._pre

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        m = self.manifest[idx]
        valid = True
        err = ""
        tactile: Optional[torch.Tensor] = None
        text = str(m.get("text", ""))

        try:
            pre = self._get_pre()
            dataset_hint = pre.infer_dataset_hint(m["touch_path"], m.get("touch_bg_path") or "")
            tactile = pre.load_tactile(
                tactile_path=m["touch_path"],
                dataset_hint=dataset_hint,
                tactile_background_path=m.get("touch_bg_path"),
            )

            # tactile: (3,224,224), float tensor
            if not isinstance(tactile, torch.Tensor):
                raise TypeError(f"tactile must be torch.Tensor, got {type(tactile)}")
            if tactile.dim() != 3:
                raise ValueError(f"tactile must be (3,H,W), got shape={tuple(tactile.shape)}")

            # touch-only random_drop：对齐 TVL-LLaMA 的随机丢模态逻辑（这里仅支持丢 tactile）
            if bool(self.touch_preprocess_cfg.get("random_drop", False)):
                drop = random.choice([0, 1, 2])
                if drop == 1:
                    tactile = torch.zeros_like(tactile)

        except Exception as e:
            valid = False
            err = f"{type(e).__name__}: {e}"
            if self.on_error == "raise":
                raise

        return {
            "global_index": int(m["global_index"]),
            "sample_id": m["sample_id"],
            "tactile": tactile,
            "text": text,
            "valid": bool(valid),
            "error": err,
        }


def collate_raw(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "global_indices": torch.tensor([b["global_index"] for b in batch], dtype=torch.long),
        "sample_ids": [b["sample_id"] for b in batch],
        "tactiles": [b["tactile"] for b in batch],  # list[Tensor|None]
        "texts": [b["text"] for b in batch],
        "valid": torch.tensor([1 if b["valid"] else 0 for b in batch], dtype=torch.uint8),
        "errors": [b.get("error", "") for b in batch],
    }


class IndexSampler(Sampler[int]):
    def __init__(self, indices: List[int]):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def split_indices_no_pad(n: int, rank: int, world_size: int, mode: str = "strided") -> List[int]:
    if world_size <= 1:
        return list(range(n))
    mode = (mode or "strided").lower()
    if mode == "strided":
        return list(range(rank, n, world_size))
    if mode == "contiguous":
        per = (n + world_size - 1) // world_size
        start = rank * per
        end = min(n, start + per)
        return list(range(start, end))
    raise ValueError(f"Unknown split mode: {mode}")


# -----------------------------------------------------------------------------
# Touch encoder (timm tactile_model + checkpoint) -> (B, 1, D)
# -----------------------------------------------------------------------------
def _extract_state_dict_from_ckpt(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        # fallback: keep only tensor-like entries
        out = {k: v for k, v in ckpt.items() if torch.is_tensor(v)}
        if out:
            return out
    raise ValueError("Cannot extract a usable state_dict from checkpoint.")


def _select_tactile_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    兼容多种前缀：
      - tactile_encoder.*
      - image_bind.tactile_encoder.*
      - module.tactile_encoder.*
      - image_bind.module.tactile_encoder.*
    返回剥离前缀后的 tactile encoder state_dict
    """
    # strip 'module.' first
    sd2 = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        sd2[k] = v
    sd = sd2

    prefixes = [
        "tactile_encoder.",
        "image_bind.tactile_encoder.",
        "image_bind.module.tactile_encoder.",
        "model.tactile_encoder.",
        "image_bind.model.tactile_encoder.",
    ]
    for pref in prefixes:
        if any(k.startswith(pref) for k in sd.keys()):
            out = {}
            for k, v in sd.items():
                if k.startswith(pref):
                    out[k[len(pref) :]] = v
            return out

    # already tactile-only
    return sd


class FrozenTVLTactileEncoder(torch.nn.Module):
    """
    输出 ViT token 序列:
      input : (B,3,224,224)
      output: (B, N, D)  其中 N=197(=1+14*14), D=768(对于 vit_base_patch16_224)
    """

    def __init__(self, cfg: Dict[str, Any], device: torch.device):
        super().__init__()
        try:
            import timm
        except Exception as e:
            raise ImportError("timm is required for FrozenTVLTactileEncoder") from e

        self.device = device
        self.tactile_model = str(cfg.get("tactile_model", "vit_base_patch16_224"))
        self.l2_normalize = bool(cfg.get("l2_normalize", True))
        self.input_dtype = str(cfg.get("input_dtype", "fp32")).lower()

        self.drop_rate = float(cfg.get("drop_rate", 0.0))
        self.drop_path_rate = float(cfg.get("drop_path_rate", 0.0))

        # 关键点1：不要 num_classes=out_dim（那是 head 输出维度）
        # 关键点2：num_classes=0 会让 head 变成 Identity，更适合只抽 tokens，也更不容易加载 ckpt 时 shape mismatch
        self.model = timm.create_model(
            self.tactile_model,
            pretrained=False,
            num_classes=0,       # <--- disable classification head
            global_pool="",      # <--- no pooling
            drop_rate=self.drop_rate,
            drop_path_rate=self.drop_path_rate,
        )

        # 加载 checkpoint（保持你原来的逻辑）
        ckpt_path = cfg.get("checkpoint_path", None)
        if ckpt_path:
            import argparse
            try:
                from torch.serialization import add_safe_globals
                add_safe_globals([argparse.Namespace])
            except Exception:
                pass

            ckpt = torch.load(str(ckpt_path), map_location="cpu")
            sd = _extract_state_dict_from_ckpt(ckpt)
            sd = _select_tactile_state_dict(sd)

            # 1) drop head/classifier keys (更稳，避免一些 shape mismatch / 干扰)
            drop_prefix = ("head.", "fc.", "classifier.")
            sd = {k: v for k, v in sd.items() if not k.startswith(drop_prefix)}

            # 2) norm name compatibility: fc_norm <-> norm
            model_keys = set(self.model.state_dict().keys())
            model_has_norm = ("norm.weight" in model_keys and "norm.bias" in model_keys)
            model_has_fc_norm = ("fc_norm.weight" in model_keys and "fc_norm.bias" in model_keys)

            ckpt_has_norm = ("norm.weight" in sd and "norm.bias" in sd)
            ckpt_has_fc_norm = ("fc_norm.weight" in sd and "fc_norm.bias" in sd)

            # ckpt 用 fc_norm，但模型期望 norm -> 重命名 fc_norm -> norm
            if model_has_norm and (not ckpt_has_norm) and ckpt_has_fc_norm:
                sd["norm.weight"] = sd.pop("fc_norm.weight")
                sd["norm.bias"] = sd.pop("fc_norm.bias")

            # ckpt 用 norm，但模型期望 fc_norm -> 重命名 norm -> fc_norm
            elif model_has_fc_norm and (not ckpt_has_fc_norm) and ckpt_has_norm:
                sd["fc_norm.weight"] = sd.pop("norm.weight")
                sd["fc_norm.bias"] = sd.pop("norm.bias")

            missing, unexpected = self.model.load_state_dict(sd, strict=False)

            # 对“完全一致”，我们希望 norm/fc_norm 不再 missing/unexpected
            if missing or unexpected:
                # 仍然打印，但你可以把它改成 raise 来强制严格一致
                raise RuntimeError(f"Checkpoint keys did not fully match model keys. missing={missing} unexpected={unexpected}")
                        

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.to(device)

        # 记录一下 token hidden dim（ViT 一般是 embed_dim=768/1024...）
        self.embed_dim = getattr(self.model, "embed_dim", None)
        if self.embed_dim is None:
            # timm 有些模型用 num_features
            self.embed_dim = int(getattr(self.model, "num_features", 0)) or None

    @torch.inference_mode()
    def forward(self, tactile: torch.Tensor) -> torch.Tensor:
        # tactile: (B,3,224,224)
        if tactile.dim() != 4:
            raise ValueError(f"Expected tactile (B,3,H,W), got {tuple(tactile.shape)}")

        if self.input_dtype in ("fp16", "float16", "half"):
            tactile = tactile.half()
        elif self.input_dtype in ("bf16", "bfloat16"):
            tactile = tactile.bfloat16()
        else:
            tactile = tactile.float()

        tactile = tactile.to(self.device, non_blocking=True)

        # 关键点3：用 forward_features 拿 token 序列，而不是 model(x) 的 pooled 向量
        if not hasattr(self.model, "forward_features"):
            raise RuntimeError(f"Model {self.tactile_model} has no forward_features(); not a ViT-like timm model?")
        tokens = self.model.forward_features(tactile)

        # 有的模型 forward_features 可能返回 tuple/list
        # if isinstance(tokens, (tuple, list)):
        #     tokens = tokens[0]

        # # ViT: (B, N, D)
        # # ConvNet: (B, C, H, W) -> 展平为 tokens (B, H*W, C)
        # if tokens.dim() == 4:
        #     tokens = tokens.flatten(2).transpose(1, 2).contiguous()

        # if tokens.dim() != 3:
        #     raise RuntimeError(f"Unexpected tokens shape from forward_features: {tuple(tokens.shape)}")

        # 你如果只想要 patch tokens (196,768)，不要 cls token，就用 tokens[:,1:,:]
        # tokens = tokens[:, 1:, :]

        if self.l2_normalize:
            tokens = tokens / tokens.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        return tokens  # (B, N, D)



# -----------------------------------------------------------------------------
# Text embedding table (Qwen tokenizer + frozen embedding weights)
# -----------------------------------------------------------------------------
def extract_and_cache_qwen_embedding_weight(text_cfg: Dict[str, Any], save_path: str) -> None:
    ensure_dir(os.path.dirname(save_path))
    model_name = text_cfg["model_name_or_path"]
    trust_remote_code = bool(text_cfg.get("trust_remote_code", True))
    torch_dtype = _parse_torch_dtype(text_cfg.get("torch_dtype", "fp16"))

    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    emb = model.get_input_embeddings().weight.detach().cpu()
    torch.save({"weight": emb, "hidden_size": int(emb.shape[1]), "vocab_size": int(emb.shape[0])}, save_path)
    del model


# -----------------------------------------------------------------------------
# Memmap writer (touch + text)
# -----------------------------------------------------------------------------
@dataclass
class ShardPaths:
    touch_tokens: str
    text_embeds: str
    text_mask: str
    sample_ids: str
    global_indices: str
    valid: str


class MemmapShardWriter:
    def __init__(
        self,
        out_dir: str,
        rank: int,
        num_samples: int,
        touch_shape: Tuple[int, int],
        text_shape: Tuple[int, int],
        save_dtype: np.dtype,
    ):
        self.rank = rank
        self.num_samples = int(num_samples)

        shard_dir = os.path.join(out_dir, "shards")
        ensure_dir(shard_dir)

        self.paths = ShardPaths(
            touch_tokens=os.path.join(shard_dir, f"touch_tokens_rank{rank:02d}.mmap"),
            text_embeds=os.path.join(shard_dir, f"text_embeds_rank{rank:02d}.mmap"),
            text_mask=os.path.join(shard_dir, f"text_mask_rank{rank:02d}.mmap"),
            sample_ids=os.path.join(shard_dir, f"sample_ids_rank{rank:02d}.mmap"),
            global_indices=os.path.join(shard_dir, f"global_indices_rank{rank:02d}.mmap"),
            valid=os.path.join(shard_dir, f"valid_rank{rank:02d}.mmap"),
        )

        t_tok, t_dim = touch_shape
        l_text, h_text = text_shape

        self.touch_tokens = np.memmap(self.paths.touch_tokens, mode="w+", dtype=save_dtype, shape=(self.num_samples, t_tok, t_dim))
        self.text_embeds = np.memmap(self.paths.text_embeds, mode="w+", dtype=save_dtype, shape=(self.num_samples, l_text, h_text))
        self.text_mask = np.memmap(self.paths.text_mask, mode="w+", dtype=np.uint8, shape=(self.num_samples, l_text))
        self.sample_ids = np.memmap(self.paths.sample_ids, mode="w+", dtype="S64", shape=(self.num_samples,))
        self.global_indices = np.memmap(self.paths.global_indices, mode="w+", dtype=np.int64, shape=(self.num_samples,))
        self.valid = np.memmap(self.paths.valid, mode="w+", dtype=np.uint8, shape=(self.num_samples,))
        self._ptr = 0

    def write_batch(
        self,
        touch_tokens: torch.Tensor,
        text_embeds: torch.Tensor,
        text_mask: torch.Tensor,
        sample_ids: List[str],
        global_indices: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        b = int(touch_tokens.shape[0])
        s = self._ptr
        e = s + b
        if e > self.num_samples:
            raise RuntimeError(f"Shard overflow: ptr={s} batch={b} total={self.num_samples}")

        self.touch_tokens[s:e] = touch_tokens.detach().cpu().numpy()
        self.text_embeds[s:e] = text_embeds.detach().cpu().numpy()
        self.text_mask[s:e] = text_mask.detach().cpu().numpy().astype(np.uint8)
        self.global_indices[s:e] = global_indices.detach().cpu().numpy().astype(np.int64)
        self.valid[s:e] = valid.detach().cpu().numpy().astype(np.uint8)

        for i, sid in enumerate(sample_ids):
            self.sample_ids[s + i] = np.bytes_(sid.encode("utf-8")[:64])

        self._ptr = e

    def flush(self) -> None:
        self.touch_tokens.flush()
        self.text_embeds.flush()
        self.text_mask.flush()
        self.sample_ids.flush()
        self.global_indices.flush()
        self.valid.flush()

    def close(self) -> None:
        self.flush()


# -----------------------------------------------------------------------------
# DDP helpers (Fix warnings by binding device_id + barrier device_ids)
# -----------------------------------------------------------------------------
def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def is_torchrun_env() -> bool:
    return ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ)


def get_rank_info() -> Tuple[int, int, int]:
    if is_torchrun_env():
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return rank, world_size, local_rank
    return 0, 1, 0


def ddp_init_if_needed(rank: int, world_size: int, local_rank: int, backend: str, init_method: Optional[str]) -> torch.device:
    if world_size <= 1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(0)
        return device

    backend = str(backend or ("nccl" if torch.cuda.is_available() else "gloo")).lower()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    kwargs = dict(backend=backend, init_method=init_method, rank=rank, world_size=world_size)
    if backend == "nccl" and device.type == "cuda":
        kwargs["device_id"] = device

    try:
        dist.init_process_group(**kwargs)
    except TypeError:
        kwargs.pop("device_id", None)
        dist.init_process_group(**kwargs)

    return device


def ddp_barrier(device: Optional[torch.device]) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    if device is not None and device.type == "cuda":
        try:
            dist.barrier(device_ids=[int(device.index)])
            return
        except TypeError:
            pass
    dist.barrier()


# -----------------------------------------------------------------------------
# Worker
# -----------------------------------------------------------------------------
def worker_main(rank: int, world_size: int, local_rank: int, cfg: Dict[str, Any], init_method: Optional[str]) -> None:
    dcfg = cfg["dataset"]
    rcfg = cfg["runtime"]
    ecfg = cfg["encoders"]

    ddp_cfg = cfg.get("distributed", {})
    backend = str(ddp_cfg.get("backend", "nccl" if torch.cuda.is_available() else "gloo"))
    split_mode = str(ddp_cfg.get("split_mode", "strided"))

    device = ddp_init_if_needed(rank, world_size, local_rank, backend, init_method)
    set_seed(int(cfg.get("seed", 1234)), rank)

    out_dir = rcfg["output_dir"]
    overwrite = bool(rcfg.get("overwrite", True))
    ensure_dir(out_dir)

    # Build manifest once (rank0)
    if rank == 0:
        manifest = build_manifest(cfg)
        man_path = os.path.join(out_dir, "manifest_rank0.json")
        with open(man_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False)
    ddp_barrier(device)

    man_path = os.path.join(out_dir, "manifest_rank0.json")
    with open(man_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    n_total = len(manifest)
    local_indices = split_indices_no_pad(n_total, rank, world_size, split_mode)
    local_manifest = [manifest[i] for i in local_indices]
    n_local = len(local_manifest)
    # pprint(f"rank={rank} world_size={world_size} n_total={n_total} n_local={n_local} local_indices={local_indices[:10]}...")
    if n_local == 0:
        print(f"[rank{rank}] no samples assigned, skip.")
        ddp_barrier(device)
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        return

    # Rank0 prepares Qwen embedding cache
    if rank == 0:
        text_cfg = ecfg["text"]
        if bool(text_cfg.get("extract_embedding_and_discard_model", True)):
            cache = text_cfg.get("embedding_weight_cache", os.path.join(out_dir, "qwen_embedding_weight.pt"))
            if overwrite or (not os.path.isfile(cache)):
                extract_and_cache_qwen_embedding_weight(text_cfg, cache)
            text_cfg["embedding_weight_cache"] = cache
    ddp_barrier(device)

    if "embedding_weight_cache" not in ecfg["text"]:
        ecfg["text"]["embedding_weight_cache"] = os.path.join(out_dir, "qwen_embedding_weight.pt")

    # touch preprocess config (map keys for compatibility)
    touch_cfg = dict(ecfg.get("touch", {}))
    tvl_repo_path = touch_cfg.get("tvl_repo_path") or touch_cfg.get("repo_path")  # allow old key

    touch_pre_cfg = dict(
        tvl_repo_path=tvl_repo_path,
        crop_tacvis=bool(touch_cfg.get("crop_tacvis", False)),
        subtract_background=touch_cfg.get("subtract_background", None),  # None or "background"
        augment_rgb=bool(touch_cfg.get("augment_rgb", False)),           # not used (we don't load vision)
        augment_tactile=bool(touch_cfg.get("augment_tactile", False)),
        random_drop=bool(touch_cfg.get("random_drop", False)),
        image_size=int(touch_cfg.get("image_size", 224)),
    )

    # Encoders
    touch_encoder = FrozenTVLTactileEncoder(touch_cfg, device=device)
    text_encoder = FrozenQwenEmbeddingTableFromWeight(ecfg["text"], device=device)

    # Probe shapes using the first valid sample
    probe_touch = None
    probe_text = None
    probe_mask = None
    for m in local_manifest:
     
        pre = TVLTouchPreprocessor(TVLTouchPreprocessConfig(**touch_pre_cfg))
        
        dataset_hint = pre.infer_dataset_hint(m["touch_path"], m.get("touch_bg_path") or "")
        tactile = pre.load_tactile(
            tactile_path=m["touch_path"],
            dataset_hint=dataset_hint,
            tactile_background_path=m.get("touch_bg_path"),
        )
        txt = str(m["text"])
        with torch.inference_mode():
            probe_touch = touch_encoder(tactile.unsqueeze(0))         # (1, T, D)
            probe_text, probe_mask = text_encoder([txt])              # (1, L, H), (1, L)
        break
    pprint(f"[rank{rank}] probe_touch_shape={tuple(probe_touch.shape) if probe_touch is not None else None} "
           f"probe_text_shape={tuple(probe_text.shape) if probe_text is not None else None} "
           f"probe_mask_shape={tuple(probe_mask.shape) if probe_mask is not None else None}")

    if probe_touch is None or probe_text is None or probe_mask is None:
        raise RuntimeError(f"[rank{rank}] cannot find any valid sample to probe shapes!")

    touch_shape = (int(probe_touch.shape[1]), int(probe_touch.shape[2]))
    text_shape = (int(probe_text.shape[1]), int(probe_text.shape[2]))

    save_dtype = _parse_np_dtype_save(rcfg.get("save_dtype", "fp16"))

    shard_dir = os.path.join(out_dir, "shards")
    ensure_dir(shard_dir)
    test_path = os.path.join(shard_dir, f"touch_tokens_rank{rank:02d}.mmap")
    if (not overwrite) and os.path.exists(test_path):
        raise FileExistsError(f"Shard already exists: {test_path}. set runtime.overwrite=true")
    pprint(f"[rank{rank}] touch_shape={touch_shape} text_shape={text_shape} ")
    writer = MemmapShardWriter(
        out_dir=out_dir,
        rank=rank,
        num_samples=n_local,
        touch_shape=touch_shape,
        text_shape=text_shape,
        save_dtype=save_dtype,
    )

    # Loader
    ds = TVLTouchTextDataset(
        local_manifest,
        touch_preprocess_cfg=touch_pre_cfg,
        on_error=str(dcfg.get("on_error", "zero")).lower(),
    )
    batch_size = int(rcfg.get("batch_size", 8))
    nw = int(rcfg.get("num_workers", 4))
    prefetch_factor = int(rcfg.get("prefetch_factor", 2))
    pin_memory = bool(rcfg.get("pin_memory", True))
    persistent_workers = bool(rcfg.get("persistent_workers", True)) and nw > 0

    dl_kwargs = dict(
        dataset=ds,
        batch_size=batch_size,
        num_workers=nw,
        sampler=IndexSampler(list(range(n_local))),
        collate_fn=collate_raw,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )
    if nw > 0:
        dl_kwargs["prefetch_factor"] = prefetch_factor

    loader = DataLoader(**dl_kwargs)

    log_every = int(rcfg.get("log_every", 20))
    flush_every = int(rcfg.get("flush_every", 50))

    t0 = time.time()
    num_done = 0

    t_tok, t_dim = touch_shape
    l_text, h_text = text_shape

    for step, batch in enumerate(loader):
        valid = batch["valid"]  # (B,)
        tactiles_list = batch["tactiles"]  # list[Tensor|None]
        texts = batch["texts"]
        sample_ids = batch["sample_ids"]
        global_indices = batch["global_indices"]
        # pprint(f"size of one tactile is {tactiles_list[0].shape if tactiles_list[0] is not None else None}")
        # pprint(f"rank={rank} step={step} batch_size={len(sample_ids)} first 5 text is texts[:5]={texts[:5]} valid={valid[:5]} sample_ids[:5]={sample_ids[:5]} global_indices[:5]={global_indices[:5]}")

        b = len(sample_ids)

        torch_out_dtype = torch.float16 if save_dtype == np.float16 else torch.float32
        touch_tokens_out = torch.zeros((b, t_tok, t_dim), device=device, dtype=torch_out_dtype)
        text_embeds_out = torch.zeros((b, l_text, h_text), device=device, dtype=torch_out_dtype)
        text_mask_out = torch.zeros((b, l_text), device=device, dtype=torch.bool)

        valid_idx = (valid == 1).nonzero(as_tuple=False).squeeze(1).tolist()
        if len(valid_idx) > 0:
            tactile_valid = [tactiles_list[i] for i in valid_idx]
            text_valid = [texts[i] for i in valid_idx]

            # stack tactile tensor: (bv,3,224,224)
            tactile_batch = torch.stack(tactile_valid, dim=0)
            with torch.inference_mode():
                touch_tokens_v = touch_encoder(tactile_batch).to(dtype=torch_out_dtype)  # (bv,T,D)
                text_embeds_v, text_mask_v = text_encoder(text_valid)                   # (bv,L,H), (bv,L)
                # pprint(f"rank={rank} step={step} touch_tokens_v_shape={tuple(touch_tokens_v.shape)} text_embeds_v_shape={tuple(text_embeds_v.shape)} text_mask_v_shape={tuple(text_mask_v.shape)},text_valid ={text_valid} first text_mask_v[0]={text_mask_v[0]}")
                # exit(0)
                text_embeds_v = text_embeds_v.to(dtype=torch_out_dtype)

            touch_tokens_out[valid_idx] = touch_tokens_v
            text_embeds_out[valid_idx] = text_embeds_v
            text_mask_out[valid_idx] = text_mask_v

        writer.write_batch(
            touch_tokens=touch_tokens_out,
            text_embeds=text_embeds_out,
            text_mask=text_mask_out,
            sample_ids=sample_ids,
            global_indices=global_indices,
            valid=valid,
        )

        num_done += b
        if flush_every > 0 and (step + 1) % flush_every == 0:
            writer.flush()

        if log_every > 0 and (step + 1) % log_every == 0:
            elapsed = time.time() - t0
            speed = num_done / max(elapsed, 1e-9)
            print(f"[rank{rank}] step={step+1} done={num_done}/{n_local} speed={speed:.2f} samples/s")

    writer.close()

    shard_info = {
        "rank": rank,
        "num_samples": n_local,
        "touch": {"num_tokens": touch_shape[0], "hidden": touch_shape[1], "dtype": str(save_dtype)},
        "text": {"max_len": text_shape[0], "hidden": text_shape[1], "dtype": str(save_dtype)},
        "touch_preprocess": touch_pre_cfg,
        "touch_encoder": {
            "tactile_model": touch_cfg.get("tactile_model", "vit_tiny_patch16_224"),
            "checkpoint_path": touch_cfg.get("checkpoint_path", None),
            "out_dim": int(touch_cfg.get("out_dim", 768)),
            "l2_normalize": bool(touch_cfg.get("l2_normalize", True)),
        },
        "paths": {
            "touch_tokens": writer.paths.touch_tokens,
            "text_embeds": writer.paths.text_embeds,
            "text_mask": writer.paths.text_mask,
            "sample_ids": writer.paths.sample_ids,
            "global_indices": writer.paths.global_indices,
            "valid": writer.paths.valid,
        },
    }

    shard_info_path = os.path.join(out_dir, f"shard_info_rank{rank:02d}.yaml")
    with open(shard_info_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(shard_info, f, sort_keys=False)

    ddp_barrier(device)

    if rank == 0:
        shards = []
        for r in range(world_size):
            p = os.path.join(out_dir, f"shard_info_rank{r:02d}.yaml")
            with open(p, "r", encoding="utf-8") as f:
                shards.append(yaml.safe_load(f))

        dataset_info = {
            "version": 1,
            "num_samples_total": int(n_total),
            "world_size": int(world_size),
            "split_mode": split_mode,
            "dataset": {
                "meta_paths": dcfg.get("meta_paths", []),
                "touch_key_candidates": dcfg.get("touch_key_candidates", []),
                "touch_bg_key_candidates": dcfg.get("touch_bg_key_candidates", []),
                "text_key_candidates": dcfg.get("text_key_candidates", []),
                "gpt_text_strategy": str(dcfg.get("gpt_text_strategy", "first")),
            },
            "features": {
                "touch": {"num_tokens": touch_shape[0], "hidden": touch_shape[1], "dtype": str(save_dtype)},
                "text": {"max_len": text_shape[0], "hidden": text_shape[1], "dtype": str(save_dtype)},
            },
            "pairs": ["touch-text", "text-touch"],
            "shards": shards,
        }
        out_path = os.path.join(out_dir, "dataset_info.yaml")
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(dataset_info, f, sort_keys=False)
        print(f"[rank0] dataset_info saved to: {out_path}")

    ddp_barrier(device)
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    cfg_path = os.environ.get(
        "MM_CFG",
        os.path.join("swift", "point_cloud", "stage1", "configs", "extract_tvl_features.yaml"),
    )
    cfg = load_yaml(cfg_path)
    if not cfg:
        raise ValueError(f"Empty config: {cfg_path}")

    # torchrun multi-proc env
    if is_torchrun_env():
        rank, world_size, local_rank = get_rank_info()
        worker_main(rank=rank, world_size=world_size, local_rank=local_rank, cfg=cfg, init_method="env://")
        return

    # Single-node mp.spawn mode
    num_gpus = int(cfg.get("distributed", {}).get("num_gpus", torch.cuda.device_count()))
    if num_gpus <= 1:
        worker_main(rank=0, world_size=1, local_rank=0, cfg=cfg, init_method=None)
        return

    port = _find_free_port()
    init_method = f"tcp://127.0.0.1:{port}"

    def _wrapped(rank_: int, world_size_: int, cfg_: Dict[str, Any], init_method_: str) -> None:
        worker_main(rank=rank_, world_size=world_size_, local_rank=rank_, cfg=cfg_, init_method=init_method_)

    mp.spawn(_wrapped, args=(num_gpus, cfg, init_method), nprocs=num_gpus, join=True)


if __name__ == "__main__":
    main()


