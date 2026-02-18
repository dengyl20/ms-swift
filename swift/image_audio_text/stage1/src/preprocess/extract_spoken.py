# extract_spoken_features.py
from __future__ import annotations

import os
import time
import socket
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler, ConcatDataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    Qwen3OmniMoeProcessor,  # 只用来拿 image_processor / feature_extractor，不走 processor.__call__
    Qwen3OmniMoeThinkerForConditionalGeneration,
)

from swift.image_audio_text.stage1.src.data.spokencoco import SpokenCoCoTripletDataset, collate_triplets
from rich.pretty import pprint

# accelerate: slim meta-model loading
try:
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
except Exception:
    init_empty_weights = None
    set_module_tensor_to_device = None


# ============================================================
# YAML / IO
# ============================================================
def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return {} if cfg is None else cfg


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ============================================================
# dtype helpers
# ============================================================
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
    raise ValueError(f"Unsupported save_dtype: {s} (fp16/fp32)")


def _unwrap_first_output(x: Any) -> Any:
    """
    兼容多种 HF 输出：
    - tuple/list: (hidden_states, extra) -> hidden_states
    - ModelOutput / dataclass: 有 last_hidden_state -> 取它
    - dict: 有 last_hidden_state -> 取它
    - tensor: 原样返回
    """
    if isinstance(x, (tuple, list)):
        return x[0]
    if hasattr(x, "last_hidden_state"):
        return x.last_hidden_state
    if isinstance(x, dict) and "last_hidden_state" in x:
        return x["last_hidden_state"]
    return x


def _pad_or_trunc_3d(x: torch.Tensor, target_tokens: int) -> torch.Tensor:
    # x: (B,N,D)
    b, n, d = x.shape
    if n == target_tokens:
        return x
    if n > target_tokens:
        return x[:, :target_tokens, :]
    pad = x.new_zeros((b, target_tokens - n, d))
    return torch.cat([x, pad], dim=1)


def _split_flat_tokens_by_grid_thw(
    flat: torch.Tensor,  # (total_tokens, D)
    grid_thw: torch.Tensor,  # (B,3) each: (t,h,w)
    spatial_merge_size: int,
) -> List[torch.Tensor]:
    """
    把 vision 的 flat tokens (sum over batch) split 回每个样本一段。

    注意：不同 transformers/Qwen3 版本里，grid_thw 的 h/w 可能表示 merge 前或 merge 后。
    所以这里做一个“自动判别”：
      - 优先尝试 after-merge: t*(h//m)*(w//m)
      - 不匹配则尝试 pre-merge: t*h*w
      - 都不匹配再做 fallback（平均切分 or 直接报错）
    """
    if flat.ndim != 2:
        raise ValueError(f"expected flat 2D tokens, got shape={tuple(flat.shape)}")

    B = int(grid_thw.shape[0])
    total = int(flat.shape[0])

    g = grid_thw.detach().to("cpu")
    t = g[:, 0].to(torch.long)
    h = g[:, 1].to(torch.long)
    w = g[:, 2].to(torch.long)

    m = int(spatial_merge_size) if spatial_merge_size is not None else 1
    m = max(m, 1)

    # candidate 1: after-merge lens
    h1 = (h // m).clamp(min=1)
    w1 = (w // m).clamp(min=1)
    lens1 = (t * h1 * w1).tolist()
    sum1 = int(sum(lens1))

    # candidate 2: pre-merge lens
    lens2 = (t * h * w).tolist()
    sum2 = int(sum(lens2))

    if sum1 == total:
        lens = lens1
    elif sum2 == total:
        lens = lens2
    else:
        # fallback：如果 flat 能整除 batch，就均分；否则直接报错（避免 silent 错分）
        if total % B == 0:
            per = total // B
            lens = [per] * B
        else:
            raise RuntimeError(
                f"cannot split vision tokens: total={total}, "
                f"sum(after_merge)={sum1}, sum(pre_merge)={sum2}, B={B}"
            )

    return list(torch.split(flat, lens, dim=0))


def as_vision_tokens_3d(
    vision_out: Any,
    *,
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    target_tokens: int,
) -> torch.Tensor:
    """
    把 get_image_features() 的输出（可能是 tuple）标准化成 (B, target_tokens, D)
    """
    x = _unwrap_first_output(vision_out)

    if not torch.is_tensor(x):
        raise TypeError(f"vision output is not tensor after unwrap, got {type(x)}")

    if x.ndim == 3:
        # (B,N,D) -> pad/trunc
        return _pad_or_trunc_3d(x, target_tokens)

    if x.ndim == 2:
        # (total_tokens,D) -> split -> pad/trunc -> stack
        chunks = _split_flat_tokens_by_grid_thw(x, grid_thw=grid_thw, spatial_merge_size=spatial_merge_size)
        padded = [_pad_or_trunc_tokens(c, target_tokens) for c in chunks]  # each (target_tokens,D)
        return torch.stack(padded, dim=0)

    raise RuntimeError(f"unexpected vision token shape: {tuple(x.shape)}")



# ============================================================
# seed
# ============================================================
def set_seed(seed: int, rank: int) -> None:
    seed = int(seed) + int(rank)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# ddp helpers
# ============================================================
def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def is_torchrun_env() -> bool:
    return ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ)


def ddp_init_if_needed(
    rank: int,
    world_size: int,
    backend: str,
    init_method: Optional[str],
    device_id: Optional[torch.device],
) -> None:
    if world_size <= 1:
        return
    if dist.is_initialized():
        return

    try:
        dist.init_process_group(
            backend=backend,
            init_method=init_method or "env://",
            world_size=world_size,
            rank=rank,
            device_id=device_id,
        )
    except TypeError:
        dist.init_process_group(
            backend=backend,
            init_method=init_method or "env://",
            world_size=world_size,
            rank=rank,
        )


def ddp_barrier(local_rank: Optional[int] = None) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    if torch.cuda.is_available() and local_rank is not None:
        try:
            dist.barrier(device_ids=[int(local_rank)])
        except TypeError:
            dist.barrier()
    else:
        dist.barrier()


# ============================================================
# split indices (no padding, no repeat)
# ============================================================
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


# ============================================================
# preprocessors (NO processor.__call__)
# ============================================================
def load_image_audio_preprocessors(
    model_name: str,
    trust_remote_code: bool,
    *,
    use_fast: Optional[bool] = None,
):
    """
    仅用于拿 image_processor / feature_extractor / sampling_rate。
    不走 processor.__call__（不会要求 text，也不会触发 multi-modal packing）。
    """
    kwargs = dict(trust_remote_code=bool(trust_remote_code))
    if use_fast is not None:
        kwargs["use_fast"] = bool(use_fast)

    proc = Qwen3OmniMoeProcessor.from_pretrained(model_name, **kwargs)
    image_processor = proc.image_processor
    audio_fe = proc.feature_extractor
    sr = int(getattr(audio_fe, "sampling_rate", 16000))
    pprint(f"Loaded image/audio preprocessors from {model_name}: sampling_rate={sr}")
    return image_processor, audio_fe, sr


def prepare_image_inputs(image_processor, images: List[Image.Image]) -> Dict[str, torch.Tensor]:
    out = image_processor(images=images, return_tensors="pt")
    if "pixel_values" not in out:
        raise RuntimeError(f"image_processor output has no pixel_values, keys={list(out.keys())}")
    return out


def _ensure_audio_1d_float32(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 2 and a.shape[0] == 1:
        a = a[0]  # (1,T) -> (T,)
    elif a.ndim == 2 and a.shape[1] == 1:
        a = a[:, 0]  # (T,1) -> (T,)
    elif a.ndim == 2:
        # 多通道 -> mono，兼容 (C,T) 或 (T,C)
        if a.shape[0] <= 8 and a.shape[1] > a.shape[0]:
            a = a.mean(axis=0, dtype=np.float32)  # (C,T) -> (T,)
        else:
            a = a.mean(axis=1, dtype=np.float32)  # (T,C) -> (T,)
    elif a.ndim != 1:
        raise ValueError(f"audio must be 1D waveform, got shape={a.shape}")

    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return np.ascontiguousarray(a)


def prepare_audio_inputs(audio_fe, audios_1d: List[np.ndarray], sampling_rate: int) -> Dict[str, torch.Tensor]:
    audios_1d = [_ensure_audio_1d_float32(a) for a in audios_1d]
    # WhisperFeatureExtractor 的标准接口
    try:
        out = audio_fe(
            audios_1d,
            sampling_rate=int(sampling_rate),
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
    except TypeError:
        out = audio_fe(
            audios_1d,
            sampling_rate=int(sampling_rate),
            padding=True,
            return_tensors="pt",
        )

    if "input_features" not in out:
        raise RuntimeError(f"audio_fe output has no input_features, keys={list(out.keys())}")

    # 统一 key
    if "feature_attention_mask" not in out:
        if "attention_mask" in out:
            out["feature_attention_mask"] = out["attention_mask"]
        else:
            B = out["input_features"].shape[0]
            T = out["input_features"].shape[-1]
            out["feature_attention_mask"] = torch.ones((B, T), dtype=torch.long)

    if "audio_feature_lengths" not in out:
        out["audio_feature_lengths"] = out["feature_attention_mask"].sum(-1)

    return out


# ============================================================
# Audio loading: returns 1D waveform (T,) float32
# ============================================================
def _linear_resample_np(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return x
    if x.size == 0:
        return x
    new_len = int(round(x.shape[0] * float(target_sr) / float(orig_sr)))
    new_len = max(new_len, 1)
    xp = np.linspace(0.0, 1.0, num=x.shape[0], dtype=np.float32)
    xq = np.linspace(0.0, 1.0, num=new_len, dtype=np.float32)
    return np.interp(xq, xp, x.astype(np.float32, copy=False)).astype(np.float32, copy=False)


def load_audio_1d_float32(wav_path: str, target_sr: int, *, max_audio_seconds: Optional[float] = None) -> np.ndarray:
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(wav_path)

    audio = None
    sr = None

    try:
        import soundfile as sf
        audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    except Exception:
        audio = None

    if audio is None:
        import librosa
        y, sr = librosa.load(wav_path, sr=None, mono=False)
        audio = y.astype(np.float32, copy=False)

    # to mono 1D
    if audio.ndim == 1:
        mono = audio
    elif audio.ndim == 2:
        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            mono = audio.mean(axis=0, dtype=np.float32)  # (C,T) -> (T,)
        else:
            mono = audio.mean(axis=1, dtype=np.float32)  # (T,C) -> (T,)
    else:
        raise ValueError(f"bad audio ndim: {audio.ndim} for {wav_path}")

    mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    if int(sr) != int(target_sr):
        done = False
        try:
            import torchaudio
            x = torch.from_numpy(mono).unsqueeze(0)  # (1,T)
            y = torchaudio.functional.resample(x, orig_freq=int(sr), new_freq=int(target_sr))
            mono = y.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
            done = True
        except Exception:
            done = False

        if not done:
            try:
                import librosa
                mono = librosa.resample(mono, orig_sr=int(sr), target_sr=int(target_sr)).astype(np.float32, copy=False)
                done = True
            except Exception:
                done = False

        if not done:
            mono = _linear_resample_np(mono, orig_sr=int(sr), target_sr=int(target_sr))

    mono = _ensure_audio_1d_float32(mono)

    if max_audio_seconds is not None and max_audio_seconds > 0:
        max_len = int(float(max_audio_seconds) * float(target_sr))
        if mono.shape[0] > max_len:
            mono = mono[:max_len]

    return mono


# ============================================================
# rank0: build manifest from SpokenCoCoTripletDataset
# ============================================================
def build_manifest(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    dcfg = cfg["dataset"]
    max_samples = int(dcfg.get("max_samples", -1))
    verify_files = bool(dcfg.get("verify_files", True))
    skip_missing = bool(dcfg.get("skip_missing", True))

    train_ds = SpokenCoCoTripletDataset(
        json_path=dcfg["train_json"],
        coco_root=dcfg["coco_root"],
        spokencoco_root=dcfg["spokencoco_root"],
        split="train",
        verify_files=verify_files,
        skip_missing=skip_missing,
        max_samples=max_samples,
    )
    val_ds = SpokenCoCoTripletDataset(
        json_path=dcfg["val_json"],
        coco_root=dcfg["coco_root"],
        spokencoco_root=dcfg["spokencoco_root"],
        split="val",
        verify_files=verify_files,
        skip_missing=skip_missing,
        max_samples=max_samples,
    )
    ds = ConcatDataset([train_ds, val_ds])
    pprint(f"train_ds={len(train_ds)} val_ds={len(val_ds)} total={len(ds)},finish constructing dataset")

    bs = int(dcfg.get("manifest_batch_size", 1024))
    nw = int(dcfg.get("manifest_num_workers", 0))
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate_triplets)

    manifest: List[Dict[str, Any]] = []
    missing = 0
    kept = 0

    for batch in dl:
        texts = batch["text"]
        image_paths = batch["image_path"]
        audio_paths = batch["audio_path"]
        image_rel = batch.get("image_rel", image_paths)
        wav_rel = batch.get("wav_rel", audio_paths)

        bsz = len(texts)
        for i in range(bsz):
            t = "" if texts[i] is None else str(texts[i])
            ip = str(image_paths[i])
            ap = str(audio_paths[i])

            if verify_files:
                ok = os.path.isfile(ip) and os.path.isfile(ap)
                if (not ok) and skip_missing:
                    missing += 1
                    continue

            manifest.append(
                dict(
                    global_index=len(manifest),
                    text=t,
                    image_path=ip,
                    audio_path=ap,
                    image_rel=str(image_rel[i]),
                    wav_rel=str(wav_rel[i]),
                )
            )
            kept += 1

    print(f"[manifest] kept={kept} missing_files_skipped={missing}")
    return manifest


def _force_module_all_tensors_to_device_(module: nn.Module, device: torch.device) -> None:
    """
    强制把 module 内所有参数/registered buffers 搬到 device。
    重点：registered buffers 包含 persistent=False 的 buffer（不会出现在 state_dict 里），
    但它们往往正是 rotary cos/sin 的来源。
    """
    # 1) parameters
    for m in module.modules():
        for pn, p in list(m._parameters.items()):
            if p is None:
                continue
            # meta parameter 跳过（理论上 visual/audio 不该有 meta，如果有说明 cache 不完整）
            if getattr(p, "is_meta", False):
                continue
            if p.device != device:
                m._parameters[pn] = nn.Parameter(p.to(device=device, non_blocking=True), requires_grad=False)

        # 2) registered buffers (包含 persistent=False)
        for bn, b in list(m._buffers.items()):
            if b is None:
                continue
            if torch.is_tensor(b) and b.device != device:
                m._buffers[bn] = b.to(device=device, non_blocking=True)

    # 3) 兜底：有些实现会把 cos_cached/sin_cached 放在 attribute 里而不是 buffer
    # （如果你的版本确实这样，就会再次出现 cpu/cuda mismatch）
    for m in module.modules():
        for k, v in list(vars(m).items()):
            if torch.is_tensor(v):
                # 跳过已经是 parameter/buffer 的（避免重复）
                if k in m._parameters or k in m._buffers:
                    continue
                if getattr(v, "is_meta", False):
                    continue
                if v.device != device:
                    try:
                        setattr(m, k, v.to(device=device, non_blocking=True))
                    except Exception:
                        pass



# ============================================================
# rank0: filter manifest by token length (完全 follow point_text 逻辑)
# ============================================================
def filter_manifest_by_text_token_len(
    manifest: List[Dict[str, Any]],
    text_cfg: Dict[str, Any],
    *,
    batch_size: int = 4096,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    model_name = text_cfg["model_name_or_path"]
    tok_name = text_cfg.get("tokenizer_name_or_path", model_name)
    trust_remote_code = bool(text_cfg.get("trust_remote_code", True))

    max_len = int(text_cfg.get("max_text_len", 64))
    add_special_tokens = bool(text_cfg.get("add_special_tokens", False))

    if max_len <= 0:
        raise ValueError(f"encoders.text.max_text_len must be > 0, got {max_len}")

    print(
        f"[rank0] filtering manifest by token length: "
        f"max_len={max_len}, add_special_tokens={add_special_tokens}, tokenizer={tok_name}"
    )
    tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=trust_remote_code)

    kept: List[Dict[str, Any]] = []
    dropped = 0
    n = len(manifest)
    t0 = time.time()
    detect_len = max_len + 1

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        texts = [str(manifest[i].get("text", "")) for i in range(start, end)]

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
            lens = [len(x) for x in enc["input_ids"]]

        for i, l in enumerate(lens):
            if int(l) <= max_len:
                m = dict(manifest[start + i])
                m["text_token_len"] = int(l)
                kept.append(m)
            else:
                dropped += 1

        if (end == n) or ((end // batch_size) % 50 == 0):
            elapsed = time.time() - t0
            print(f"[rank0] text_len_filter: {end}/{n} kept={len(kept)} dropped={dropped} elapsed={elapsed:.1f}s")

    for new_i, m in enumerate(kept):
        m["global_index_before_text_filter"] = int(m.get("global_index", new_i))
        m["global_index"] = int(new_i)

    stats = {"before": n, "after": len(kept), "dropped": dropped, "max_len": max_len}
    print(f"[rank0] text_len_filter done: {stats}")
    return kept, stats


# ============================================================
# rank0: cache TEXT embedding weight（完全 follow point_text）
# ============================================================
def extract_and_cache_qwen_embedding_weight(text_cfg: Dict[str, Any], save_path: str) -> None:
    ensure_dir(os.path.dirname(save_path))
    model_name = text_cfg["model_name_or_path"]
    trust_remote_code = bool(text_cfg.get("trust_remote_code", True))
    dtype = _parse_torch_dtype(text_cfg.get("torch_dtype", "fp16"))

    print(f"[rank0] extracting Qwen embedding weight from: {model_name}")

    try:
        model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="cpu",
        )
    except TypeError:
        model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="cpu",
        )

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    emb = model.get_input_embeddings().weight.detach().cpu()
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
    完全复刻 point_text 的逻辑：
      - truncation=False
      - 超长样本在 rank0 阶段已丢弃
    """

    def __init__(self, cfg: Dict[str, Any], device: torch.device):
        super().__init__()
        self.device = device

        model_name = cfg["model_name_or_path"]
        tok_name = cfg.get("tokenizer_name_or_path", model_name)
        trust_remote_code = bool(cfg.get("trust_remote_code", True))

        self.max_text_len = int(cfg.get("max_text_len", 64))
        self.add_special_tokens = bool(cfg.get("add_special_tokens", False))

        self.tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.pad_token = self.tokenizer.convert_ids_to_tokens(0)

        weight_cache = cfg.get("embedding_weight_cache", None)
        if not weight_cache or (not os.path.isfile(weight_cache)):
            raise FileNotFoundError(f"embedding_weight_cache not found: {weight_cache}")

        payload = torch.load(weight_cache, map_location="cpu")
        weight = payload["weight"]
        self.hidden_size = int(weight.shape[1])

        self.embed = nn.Embedding.from_pretrained(weight, freeze=True).to(self.device)
        self.embed.eval()
        for p in self.embed.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def forward(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
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
        emb = self.embed(input_ids)
        return emb, attn_mask


# ============================================================
# NEW: cache image/audio weights by PREFIX (rank0 CPU)
# ============================================================
@torch.inference_mode()
def extract_and_cache_qwen_image_audio_weights_by_prefix(
    qwen_cfg: Dict[str, Any],
    *,
    save_path: str,
) -> None:
    """
    rank0:
      - CPU load full Thinker once
      - 直接从 state_dict 抽出 visual.* 与 audio_tower.* 前缀权重（不 forward，不依赖 get_image_features 签名）
      - 保存为一个小 cache，供每个 rank slim-load 到 GPU
    """
    ensure_dir(os.path.dirname(save_path))
    model_name = qwen_cfg["model_name_or_path"]
    trust_remote_code = bool(qwen_cfg.get("trust_remote_code", True))
    cache_dtype = _parse_torch_dtype(qwen_cfg.get("cache_dtype", "fp16"))

    print(f"[rank0] extracting image/audio weights by prefix (CPU) from: {model_name} -> {save_path}")


    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        torch_dtype=cache_dtype,
        low_cpu_mem_usage=True,
        device_map="cpu",)


    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    sd = model.state_dict()

    # 优先用 "visual." / "audio_tower."；若不存在则尝试其它前缀（极少数包装情况）
    candidates = [
        ("visual.", "audio_tower."),
    ]
    v_prefix = None
    a_prefix = None
    for vp, ap in candidates:
        has_v = any(k.startswith(vp) for k in sd.keys())
        has_a = any(k.startswith(ap) for k in sd.keys())
        if has_v and has_a:
            v_prefix, a_prefix = vp, ap
            break
    if v_prefix is None or a_prefix is None:
        # 至少打印一些提示
        sample_keys = list(sd.keys())[:50]
        raise RuntimeError(
            f"Cannot find visual/audio_tower prefixes in state_dict. "
            f"Tried={candidates}. sample_keys={sample_keys}"
        )

    small_sd: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith(v_prefix) or k.startswith(a_prefix):
            t = v.detach().cpu()
            if t.is_floating_point():
                t = t.to(dtype=cache_dtype)
            # 关键：保存时把前缀保持原样（后续 meta model 必须能匹配这些 key）
            small_sd[k] = t

    payload = {
        "model_name_or_path": model_name,
        "cache_dtype": str(cache_dtype),
        "visual_prefix": v_prefix,
        "audio_prefix": a_prefix,
        "num_tensors": int(len(small_sd)),
        "state_dict": small_sd,
    }
    torch.save(payload, save_path)
    del model
    pprint(f"[rank0] saved image/audio cache: {save_path} tensors={len(small_sd)}")


def load_slim_qwen_thinker_from_image_audio_cache(
    qwen_cfg: Dict[str, Any],
    *,
    cache_path: str,
    device: torch.device,
) -> Qwen3OmniMoeThinkerForConditionalGeneration:
    if init_empty_weights is None or set_module_tensor_to_device is None:
        raise RuntimeError("accelerate is required for slim image/audio model loading. Please install accelerate.")

    model_name = qwen_cfg["model_name_or_path"]
    trust_remote_code = bool(qwen_cfg.get("trust_remote_code", True))
    compute_dtype = _parse_torch_dtype(qwen_cfg.get("compute_dtype", "fp16"))

    payload = torch.load(cache_path, map_location="cpu")
    sd_small: Dict[str, torch.Tensor] = payload["state_dict"]

    cfg = Qwen3OmniMoeThinkerForConditionalGeneration.config_class.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )

    with init_empty_weights():
        model = Qwen3OmniMoeThinkerForConditionalGeneration(cfg)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # materialize cached tensors only
    for k, v in sd_small.items():
        if not torch.is_tensor(v):
            continue
        t = v
        if t.is_floating_point():
            t = t.to(dtype=compute_dtype)
        set_module_tensor_to_device(model, k, device=device, value=t.to(device=device, non_blocking=True))

        # --- 关键：把 visual/audio_tower 内所有 buffers/attrs 强制搬到目标 GPU ---
    _force_module_all_tensors_to_device_(model.visual, device=device)
    _force_module_all_tensors_to_device_(model.audio_tower, device=device)

    return model




def _as_tokens_3d(x: torch.Tensor) -> torch.Tensor:
    """
    把不同形状的输出统一成 (B, N, D)
    """
    if x.ndim == 3:
        return x
    if x.ndim == 4:
        # (B, H, W, D) -> (B, H*W, D)
        b, h, w, d = x.shape
        return x.reshape(b, h * w, d)
    if x.ndim == 2:
        # (B, D) -> (B, 1, D)
        return x.unsqueeze(1)
    raise RuntimeError(f"unexpected feature ndim={x.ndim} shape={tuple(x.shape)}")


def _pad_or_trunc_tokens(x: torch.Tensor, target_tokens: int) -> torch.Tensor:
    n, d = x.shape
    if n == target_tokens:
        return x
    if n > target_tokens:
        return x[:target_tokens]
    pad = torch.zeros((target_tokens - n, d), dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=0)


class FrozenQwenImageAudioTokens(nn.Module):
    """
    只用 visual / audio_tower 提 tower feature（不走 processor.__call__，不走 MoE/LLM）。
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        *,
        image_processor,
        audio_fe,
        sampling_rate: int,
        device: torch.device,
    ):
        super().__init__()
        self.cfg = cfg
        self.image_processor = image_processor
        self.audio_fe = audio_fe
        self.sampling_rate = int(sampling_rate)
        self.device = device

        self.image_tokens_target = int(cfg.get("image_tokens", 256))
        self.audio_tokens_target = int(cfg.get("audio_tokens", 128))
        self.max_audio_seconds = cfg.get("max_audio_seconds", None)
        self.max_audio_seconds = float(self.max_audio_seconds) if self.max_audio_seconds is not None else None

        cache_path = cfg.get("image_audio_weight_cache", None)
        if not cache_path or (not os.path.isfile(cache_path)):
            raise FileNotFoundError(f"encoders.qwen.image_audio_weight_cache not found: {cache_path}")

        self.model = load_slim_qwen_thinker_from_image_audio_cache(cfg, cache_path=cache_path, device=self.device)

    @torch.inference_mode()
    def forward(self, image_paths: List[str], audio_paths: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = len(image_paths)
        assert len(audio_paths) == bsz

        images: List[Image.Image] = []
        audios: List[np.ndarray] = []
        kept: List[int] = []
        valid = torch.zeros((bsz,), dtype=torch.uint8, device=self.device)

        for i in range(bsz):
            try:
                ip = image_paths[i]
                ap = audio_paths[i]
                if (not os.path.isfile(ip)) or (not os.path.isfile(ap)):
                    raise FileNotFoundError("missing image/audio")

                img = Image.open(ip).convert("RGB")
                wav = load_audio_1d_float32(ap, target_sr=self.sampling_rate, max_audio_seconds=self.max_audio_seconds)
                images.append(img)
                audios.append(wav)
                kept.append(i)
                valid[i] = 1
            except Exception:
                valid[i] = 0

        # 全坏 batch：返回占位
        if len(kept) == 0:
            hidden = int(getattr(self.model.config, "hidden_size", 4096))
            image_tokens = torch.zeros((bsz, self.image_tokens_target, hidden), dtype=torch.float16, device=self.device)
            audio_tokens = torch.zeros((bsz, self.audio_tokens_target, hidden), dtype=torch.float16, device=self.device)
            return image_tokens, audio_tokens, valid

        # preprocess -> tensors
        img_in = prepare_image_inputs(self.image_processor, images)
        aud_in = prepare_audio_inputs(self.audio_fe, audios, sampling_rate=self.sampling_rate)

        compute_dtype = _parse_torch_dtype(self.cfg.get("compute_dtype", "fp16"))

        # move to device
        for k, v in img_in.items():
            if torch.is_tensor(v):
                if v.is_floating_point():
                    img_in[k] = v.to(device=self.device, dtype=compute_dtype, non_blocking=True)
                else:
                    img_in[k] = v.to(device=self.device, non_blocking=True)

        for k, v in aud_in.items():
            if torch.is_tensor(v):
                if v.is_floating_point():
                    aud_in[k] = v.to(device=self.device, dtype=compute_dtype, non_blocking=True)
                else:
                    aud_in[k] = v.to(device=self.device, non_blocking=True)

        # ===== image tower =====
        pixel_values = img_in["pixel_values"]
        image_grid_thw = img_in.get("image_grid_thw", None)

        # 直接用 get_image_features（你贴的签名里没有 return_dict）
        try:
            img_feat = self.model.get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        except TypeError:
            # 极少数版本 keyword 不同：尝试 positional
            img_feat = self.model.get_image_features(pixel_values, image_grid_thw)

        img_tok_kept = _as_tokens_3d(img_feat)  # (K, Ni, D)

        # ===== audio tower =====
        # 注意：不同版本的 get_audio_features 参数名可能不同，所以做多路 fallback
        input_features = aud_in["input_features"]
        fam = aud_in.get("feature_attention_mask", None)
        afl = aud_in.get("audio_feature_lengths", None)

        aud_feat = None
        if hasattr(self.model, "get_audio_features"):
            try:
                aud_feat = self.model.get_audio_features(
                    input_features=input_features,
                    feature_attention_mask=fam,
                    audio_feature_lengths=afl,
                )
            except TypeError:
                try:
                    aud_feat = self.model.get_audio_features(input_features, fam, afl)
                except TypeError:
                    aud_feat = self.model.get_audio_features(input_features)
        else:
            # fallback: 直接调用 audio_tower（如果你的版本没有 get_audio_features）
            if fam is not None and afl is not None:
                try:
                    aud_feat = self.model.audio_tower(input_features, attention_mask=fam, audio_feature_lengths=afl)
                except TypeError:
                    aud_feat = self.model.audio_tower(input_features)
            else:
                aud_feat = self.model.audio_tower(input_features)

        if isinstance(aud_feat, (tuple, list)):
            aud_feat = aud_feat[0]
        if hasattr(aud_feat, "last_hidden_state"):
            aud_feat = aud_feat.last_hidden_state
        if isinstance(aud_feat, dict) and "last_hidden_state" in aud_feat:
            aud_feat = aud_feat["last_hidden_state"]
        if not torch.is_tensor(aud_feat):
            raise RuntimeError(f"unexpected audio feature type: {type(aud_feat)}")

        aud_tok_kept = _as_tokens_3d(aud_feat)  # (K, Na, D)

        # scatter 回原 batch，并 pad/trunc
        K, _, d_img = img_tok_kept.shape
        _, _, d_aud = aud_tok_kept.shape

        image_tokens = torch.zeros((bsz, self.image_tokens_target, int(d_img)), device=self.device, dtype=img_tok_kept.dtype)
        audio_tokens = torch.zeros((bsz, self.audio_tokens_target, int(d_aud)), device=self.device, dtype=aud_tok_kept.dtype)

        for j, bi in enumerate(kept):
            image_tokens[bi] = _pad_or_trunc_tokens(img_tok_kept[j], self.image_tokens_target)
            audio_tokens[bi] = _pad_or_trunc_tokens(aud_tok_kept[j], self.audio_tokens_target)

        return image_tokens, audio_tokens, valid


# ============================================================
# Manifest Dataset
# ============================================================
class ManifestTripletDataset(Dataset):
    def __init__(self, manifest: List[Dict[str, Any]]):
        self.manifest = manifest

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        m = self.manifest[idx]
        return {
            "global_index": int(m["global_index"]),
            "text": str(m.get("text", "")),
            "image_path": str(m["image_path"]),
            "audio_path": str(m["audio_path"]),
            "image_rel": str(m.get("image_rel", m["image_path"])),
            "wav_rel": str(m.get("wav_rel", m["audio_path"])),
        }


def collate_manifest_triplets(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "global_indices": torch.tensor([b["global_index"] for b in batch], dtype=torch.long),
        "texts": [b["text"] for b in batch],
        "image_paths": [b["image_path"] for b in batch],
        "audio_paths": [b["audio_path"] for b in batch],
        "image_rel": [b["image_rel"] for b in batch],
        "wav_rel": [b["wav_rel"] for b in batch],
    }


# ============================================================
# Memmap writer (per-rank shard)
# ============================================================
@dataclass
class ShardPaths:
    text_embeds: str
    text_mask: str
    image_tokens: str
    audio_tokens: str
    image_rel: str
    wav_rel: str
    global_indices: str
    valid: str


class MemmapShardWriter:
    def __init__(
        self,
        out_dir: str,
        rank: int,
        num_samples: int,
        text_shape: Tuple[int, int],
        image_shape: Tuple[int, int],
        audio_shape: Tuple[int, int],
        save_dtype: np.dtype,
    ):
        self.rank = int(rank)
        self.num_samples = int(num_samples)
        self.save_dtype = save_dtype

        shard_dir = os.path.join(out_dir, "shards")
        ensure_dir(shard_dir)

        self.paths = ShardPaths(
            text_embeds=os.path.join(shard_dir, f"text_embeds_rank{rank:02d}.mmap"),
            text_mask=os.path.join(shard_dir, f"text_mask_rank{rank:02d}.mmap"),
            image_tokens=os.path.join(shard_dir, f"image_tokens_rank{rank:02d}.mmap"),
            audio_tokens=os.path.join(shard_dir, f"audio_tokens_rank{rank:02d}.mmap"),
            image_rel=os.path.join(shard_dir, f"image_rel_rank{rank:02d}.mmap"),
            wav_rel=os.path.join(shard_dir, f"wav_rel_rank{rank:02d}.mmap"),
            global_indices=os.path.join(shard_dir, f"global_indices_rank{rank:02d}.mmap"),
            valid=os.path.join(shard_dir, f"valid_rank{rank:02d}.mmap"),
        )

        L, Dt = text_shape
        Ti, Di = image_shape
        Ta, Da = audio_shape

        self.text_embeds = np.memmap(self.paths.text_embeds, mode="w+", dtype=save_dtype, shape=(self.num_samples, L, Dt))
        self.text_mask = np.memmap(self.paths.text_mask, mode="w+", dtype=np.uint8, shape=(self.num_samples, L))
        self.image_tokens = np.memmap(self.paths.image_tokens, mode="w+", dtype=save_dtype, shape=(self.num_samples, Ti, Di))
        self.audio_tokens = np.memmap(self.paths.audio_tokens, mode="w+", dtype=save_dtype, shape=(self.num_samples, Ta, Da))

        self.image_rel = np.memmap(self.paths.image_rel, mode="w+", dtype="S256", shape=(self.num_samples,))
        self.wav_rel = np.memmap(self.paths.wav_rel, mode="w+", dtype="S256", shape=(self.num_samples,))
        self.global_indices = np.memmap(self.paths.global_indices, mode="w+", dtype=np.int64, shape=(self.num_samples,))
        self.valid = np.memmap(self.paths.valid, mode="w+", dtype=np.uint8, shape=(self.num_samples,))

        self._ptr = 0

    def write_batch(
        self,
        *,
        text_emb: torch.Tensor,
        text_mask: torch.Tensor,
        image_tokens: torch.Tensor,
        audio_tokens: torch.Tensor,
        image_rel: List[str],
        wav_rel: List[str],
        global_indices: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        b = int(text_emb.shape[0])
        s = self._ptr
        e = s + b
        if e > self.num_samples:
            raise RuntimeError(f"memmap overflow: ptr={s} batch={b} total={self.num_samples}")

        te = text_emb.detach().cpu().numpy()
        tm = text_mask.detach().cpu().numpy().astype(np.uint8)
        it = image_tokens.detach().cpu().numpy()
        at = audio_tokens.detach().cpu().numpy()
        gi = global_indices.detach().cpu().numpy().astype(np.int64)
        va = valid.detach().cpu().numpy().astype(np.uint8)

        self.text_embeds[s:e] = te
        self.text_mask[s:e] = tm
        self.image_tokens[s:e] = it
        self.audio_tokens[s:e] = at
        self.global_indices[s:e] = gi
        self.valid[s:e] = va

        for i in range(b):
            self.image_rel[s + i] = np.bytes_(image_rel[i].encode("utf-8")[:256])
            self.wav_rel[s + i] = np.bytes_(wav_rel[i].encode("utf-8")[:256])

        self._ptr = e

    def flush(self) -> None:
        self.text_embeds.flush()
        self.text_mask.flush()
        self.image_tokens.flush()
        self.audio_tokens.flush()
        self.image_rel.flush()
        self.wav_rel.flush()
        self.global_indices.flush()
        self.valid.flush()

    def close(self) -> None:
        self.flush()


# ============================================================
# worker main
# ============================================================
def worker_main(rank: int, world_size: int, cfg: Dict[str, Any], init_method: Optional[str]) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # 让 torch / triton / inductor 的缓存写到可控目录（避免 ~/.cache 写失败）
    os.environ.setdefault("XDG_CACHE_HOME", os.path.join(cfg["runtime"]["output_dir"], "cache"))
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.join(cfg["runtime"]["output_dir"], "cache", "torch_extensions"))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(cfg["runtime"]["output_dir"], "cache", "torchinductor"))

    backend = str(cfg.get("distributed", {}).get("backend", "nccl"))
    seed = int(cfg.get("seed", 1234))

    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device_id = torch.device(f"cuda:{local_rank}")
    else:
        device_id = None

    ddp_init_if_needed(rank, world_size, backend, init_method, device_id=device_id)
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
    text_len_filter_stats: Optional[Dict[str, int]] = None

    # ---------------- rank0: build + filter + cache weights ----------------
    if rank == 0:
        info_path = os.path.join(out_dir, "dataset_info.yaml")
        pprint(f"rank0 will write dataset info to: {info_path}")
        if (not overwrite) and os.path.exists(info_path):
            raise FileExistsError(f"{info_path} exists. Set runtime.overwrite=true to overwrite.")

        manifest = build_manifest(cfg)

        # text length filter (完全 follow point_text)
        text_cfg_rank0 = ecfg["text"]
        manifest, text_len_filter_stats = filter_manifest_by_text_token_len(
            manifest,
            text_cfg_rank0,
            batch_size=int(text_cfg_rank0.get("length_filter_batch_size", 4096)),
        )

        torch.save(manifest, manifest_pt)
        print(f"[rank0] manifest saved: {manifest_pt} (N={len(manifest)})")

        # text embedding cache
        weight_cache = text_cfg_rank0.get("embedding_weight_cache", os.path.join(cache_dir, "qwen_embed_weight.pt"))
        text_cfg_rank0["embedding_weight_cache"] = weight_cache
        if not os.path.isfile(weight_cache):
            extract_and_cache_qwen_embedding_weight(text_cfg_rank0, weight_cache)
        else:
            print(f"[rank0] found cached embedding weight: {weight_cache}")

        # image/audio cache (prefix extract, no forward)
        qwen_cfg = ecfg["qwen"]
        ia_cache = qwen_cfg.get("image_audio_weight_cache", os.path.join(cache_dir, "qwen_image_audio_weights.pt"))
        qwen_cfg["image_audio_weight_cache"] = ia_cache
        if not os.path.isfile(ia_cache):
            pprint(f"[rank0] no cached image/audio weights found, extracting by prefix and saving to: {ia_cache}")
            extract_and_cache_qwen_image_audio_weights_by_prefix(qwen_cfg, save_path=ia_cache)
        else:
            pprint(f"[rank0] found cached image/audio weights: {ia_cache}")

    ddp_barrier(local_rank=local_rank)
    # pprint(f"[rank{rank}] passed barrier after manifest/caches ready, now loading manifest and encoders")

    # ---------------- all ranks: load manifest ----------------
    manifest: List[Dict[str, Any]] = torch.load(manifest_pt, map_location="cpu")
    n_total = len(manifest)

    split_mode = str(cfg.get("distributed", {}).get("split_mode", "strided"))
    indices = split_indices_no_pad(n_total, rank, world_size, mode=split_mode)
    n_local = len(indices)

    if rank == 0:
        print(f"[split] mode={split_mode} total={n_total} world_size={world_size}")
    print(f"[rank{rank}] local samples: {n_local}")

    local_manifest = [manifest[i] for i in indices]
    ds = ManifestTripletDataset(local_manifest)

    loader = DataLoader(
        ds,
        batch_size=int(rcfg.get("batch_size", 4)),
        shuffle=False,
        num_workers=int(rcfg.get("num_workers", 4)),
        collate_fn=collate_manifest_triplets,
        pin_memory=bool(rcfg.get("pin_memory", True)),
        drop_last=False,
    )

    # ---------------- text encoder ----------------
    text_embed_device_str = str(ecfg["text"].get("embed_device", "cpu"))
    text_embed_device = torch.device(text_embed_device_str if torch.cuda.is_available() else "cpu")

    weight_cache = ecfg["text"].get("embedding_weight_cache", os.path.join(cache_dir, "qwen_embed_weight.pt"))
    ecfg["text"]["embedding_weight_cache"] = weight_cache
    text_encoder = FrozenQwenEmbeddingTableFromWeight(ecfg["text"], device=text_embed_device)

    # ---------------- image/audio encoder (slim) ----------------
    qwen_cfg = ecfg["qwen"]
    model_name = qwen_cfg["model_name_or_path"]
    trust_remote_code = bool(qwen_cfg.get("trust_remote_code", True))
    use_fast = qwen_cfg.get("processor_use_fast", None)

    image_processor, audio_fe, sr = load_image_audio_preprocessors(
        model_name=model_name,
        trust_remote_code=trust_remote_code,
        use_fast=use_fast,
    )
    # pprint(f"[rank{rank}] loaded image/audio preprocessors with sampling_rate={sr}")
    ia_device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")

    img_aud_encoder = FrozenQwenImageAudioTokens(
        qwen_cfg,
        image_processor=image_processor,
        audio_fe=audio_fe,
        sampling_rate=sr,
        device=ia_device,
    )
    # pprint(f"[rank{rank}] loaded slim Qwen image/audio encoder on device: {ia_device}")
    # shapes
    L = int(text_encoder.max_text_len)
    Dt = int(text_encoder.hidden_size)
    Ti = int(img_aud_encoder.image_tokens_target)
    Ta = int(img_aud_encoder.audio_tokens_target)

    # probe hidden dims
    with torch.inference_mode():
        dummy_img = Image.new("RGB", (224, 224), color=(0, 0, 0))
        dummy_wav = np.zeros((sr * 2,), dtype=np.float32)

        img_in = prepare_image_inputs(image_processor, [dummy_img])
        aud_in = prepare_audio_inputs(audio_fe, [dummy_wav], sampling_rate=sr)

        compute_dtype = _parse_torch_dtype(qwen_cfg.get("compute_dtype", "fp16"))
        for k, v in img_in.items():
            if torch.is_tensor(v):
                img_in[k] = v.to(device=ia_device, dtype=compute_dtype) if v.is_floating_point() else v.to(device=ia_device)
        for k, v in aud_in.items():
            if torch.is_tensor(v):
                aud_in[k] = v.to(device=ia_device, dtype=compute_dtype) if v.is_floating_point() else v.to(device=ia_device)

        image_embeds, image_embeds_multiscale = img_aud_encoder.model.get_image_features(
            pixel_values=img_in["pixel_values"],
            image_grid_thw=img_in.get("image_grid_thw", None),
        )
        img_tok = _as_tokens_3d(img_feat)
        Di = int(img_tok.shape[-1])

        # audio
        aud_feat = None
        if hasattr(img_aud_encoder.model, "get_audio_features"):
            try:
                aud_feat = img_aud_encoder.model.get_audio_features(
                    input_features=aud_in["input_features"],
                    feature_attention_mask=aud_in.get("feature_attention_mask", None),
                    audio_feature_lengths=aud_in.get("audio_feature_lengths", None),
                )
            except TypeError:
                aud_feat = img_aud_encoder.model.get_audio_features(aud_in["input_features"])
        else:
            aud_feat = img_aud_encoder.model.audio_tower(aud_in["input_features"])

        if isinstance(aud_feat, (tuple, list)):
            aud_feat = aud_feat[0]
        if hasattr(aud_feat, "last_hidden_state"):
            aud_feat = aud_feat.last_hidden_state
        if isinstance(aud_feat, dict) and "last_hidden_state" in aud_feat:
            aud_feat = aud_feat["last_hidden_state"]
        aud_tok = _as_tokens_3d(aud_feat)
        Da = int(aud_tok.shape[-1])

    save_dtype = _parse_np_dtype_save(rcfg.get("save_dtype", "fp16"))

    # ---------------- writer ----------------
    shard_dir = os.path.join(out_dir, "shards")
    ensure_dir(shard_dir)
    test_path = os.path.join(shard_dir, f"text_embeds_rank{rank:02d}.mmap")
    if (not overwrite) and os.path.exists(test_path):
        raise FileExistsError(f"Shard exists: {test_path}. Set runtime.overwrite=true")

    writer = MemmapShardWriter(
        out_dir=out_dir,
        rank=rank,
        num_samples=n_local,
        text_shape=(L, Dt),
        image_shape=(Ti, Di),
        audio_shape=(Ta, Da),
        save_dtype=save_dtype,
    )

    log_every = int(rcfg.get("log_every", 20))
    flush_every = int(rcfg.get("flush_every", 50))

    t0 = time.time()
    done = 0

    for step, batch in enumerate(loader):
        texts: List[str] = batch["texts"]
        image_paths: List[str] = batch["image_paths"]
        audio_paths: List[str] = batch["audio_paths"]
        image_rel: List[str] = batch["image_rel"]
        wav_rel: List[str] = batch["wav_rel"]
        global_indices: torch.Tensor = batch["global_indices"]

        # text embeddings (完全 follow point_text)
        try:
            text_emb, text_mask = text_encoder(texts)
            text_valid = torch.ones((len(texts),), dtype=torch.uint8)
        except Exception:
            bsz = len(texts)
            text_emb = torch.zeros((bsz, L, Dt), dtype=torch.float32)
            text_mask = torch.zeros((bsz, L), dtype=torch.bool)
            text_valid = torch.zeros((bsz,), dtype=torch.uint8)

        # image/audio tokens
        try:
            image_tokens, audio_tokens, ia_valid = img_aud_encoder(image_paths, audio_paths)
        except Exception:
            bsz = len(texts)
            image_tokens = torch.zeros((bsz, Ti, Di), dtype=torch.float16)
            audio_tokens = torch.zeros((bsz, Ta, Da), dtype=torch.float16)
            ia_valid = torch.zeros((bsz,), dtype=torch.uint8)

        valid = (text_valid & ia_valid.detach().cpu().to(torch.uint8)).to(torch.uint8)

        # save dtype
        if save_dtype == np.float16:
            text_emb = text_emb.to(dtype=torch.float16)
            image_tokens = image_tokens.to(dtype=torch.float16)
            audio_tokens = audio_tokens.to(dtype=torch.float16)
        elif save_dtype == np.float32:
            text_emb = text_emb.to(dtype=torch.float32)
            image_tokens = image_tokens.to(dtype=torch.float32)
            audio_tokens = audio_tokens.to(dtype=torch.float32)

        writer.write_batch(
            text_emb=text_emb,
            text_mask=text_mask,
            image_tokens=image_tokens,
            audio_tokens=audio_tokens,
            image_rel=image_rel,
            wav_rel=wav_rel,
            global_indices=global_indices,
            valid=valid,
        )

        done += len(texts)
        if flush_every > 0 and (step + 1) % flush_every == 0:
            writer.flush()

        if log_every > 0 and (step + 1) % log_every == 0:
            elapsed = time.time() - t0
            speed = done / max(elapsed, 1e-9)
            print(f"[rank{rank}] step={step+1} done={done}/{n_local} speed={speed:.2f} samples/s")

    writer.close()
    elapsed = time.time() - t0
    print(f"[rank{rank}] finished. local={n_local} time={elapsed:.1f}s")

    # shard info
    shard_info = {
        "rank": int(rank),
        "num_samples": int(n_local),
        "text": {"max_len": int(L), "hidden": int(Dt), "dtype": str(save_dtype)},
        "image": {"num_tokens": int(Ti), "hidden": int(Di), "dtype": str(save_dtype)},
        "audio": {"num_tokens": int(Ta), "hidden": int(Da), "dtype": str(save_dtype)},
        "paths": {
            "text_embeds": writer.paths.text_embeds,
            "text_mask": writer.paths.text_mask,
            "image_tokens": writer.paths.image_tokens,
            "audio_tokens": writer.paths.audio_tokens,
            "image_rel": writer.paths.image_rel,
            "wav_rel": writer.paths.wav_rel,
            "global_indices": writer.paths.global_indices,
            "valid": writer.paths.valid,
        },
    }
    shard_info_path = os.path.join(out_dir, f"shard_info_rank{rank:02d}.yaml")
    with open(shard_info_path, "w") as f:
        yaml.safe_dump(shard_info, f, sort_keys=False)

    ddp_barrier(local_rank=local_rank)

    # rank0 merge dataset_info
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
                "train_json": dcfg["train_json"],
                "val_json": dcfg["val_json"],
                "coco_root": dcfg["coco_root"],
                "spokencoco_root": dcfg["spokencoco_root"],
                "verify_files": bool(dcfg.get("verify_files", True)),
                "skip_missing": bool(dcfg.get("skip_missing", True)),
            },
            "features": {
                "text": {"max_len": int(L), "hidden": int(Dt), "dtype": str(save_dtype)},
                "image": {"num_tokens": int(Ti), "hidden": int(Di), "dtype": str(save_dtype)},
                "audio": {"num_tokens": int(Ta), "hidden": int(Da), "dtype": str(save_dtype)},
            },
            "text_length_filter": text_len_filter_stats,
            "shards": shards,
        }
        info_path = os.path.join(out_dir, "dataset_info.yaml")
        with open(info_path, "w") as f:
            yaml.safe_dump(dataset_info, f, sort_keys=False)
        print(f"[rank0] dataset_info saved to: {info_path}")

    ddp_barrier(local_rank=local_rank)

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# ============================================================
# main entry
# ============================================================
def main() -> None:
    cfg_path = os.environ.get("MM_CFG", os.path.join("configs", "extract_spoken_features.yaml"))
    cfg = load_yaml(cfg_path)
    if not cfg:
        raise ValueError(f"Empty config: {cfg_path}")

    if is_torchrun_env():
        print("Detected torchrun environment variables. Using torchrun for distributed execution.")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        worker_main(rank=rank, world_size=world_size, cfg=cfg, init_method="env://")
        return

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





