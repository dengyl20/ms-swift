from __future__ import annotations

import json
import os
import socket
import time
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from rich.pretty import pprint
from torch.utils.data import DataLoader, Dataset, Sampler

from PIL import Image, ImageFile
from transformers import AutoProcessor, Qwen3OmniMoeThinkerForConditionalGeneration

# -----------------------------------------------------------------------------
# accelerate (slim load)
# -----------------------------------------------------------------------------
try:
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
except Exception:
    init_empty_weights = None
    set_module_tensor_to_device = None

# -----------------------------------------------------------------------------
# Import TVL touch preprocessing
# -----------------------------------------------------------------------------
from swift.image_touch.stage1.src.preprocess.tvl_touch_encoder import (  # noqa: E402
    TVLTouchPreprocessConfig,
    TVLTouchPreprocessor,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


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
# Manifest (touch + image + optional tactile_background)
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
    image_key_candidates = dcfg.get(
        "image_key_candidates",
        ["image_path", "image", "rgb_path", "img_path", "image_file", "vision_path"],
    )
    id_key_candidates = dcfg.get("id_key_candidates", ["id", "sample_id", "uid", "object_id"])

    touch_roots = dcfg.get("touch_roots", [])
    image_roots = dcfg.get("image_roots", [])
    verify_files = bool(dcfg.get("verify_files", True))
    max_samples = int(dcfg.get("max_samples", -1))

    manifest: List[Dict[str, Any]] = []
    total = 0
    filtered = 0
    missing_touch = 0
    missing_image = 0
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

            touch_path = _resolve_path(_first_present(item, touch_key_candidates), touch_roots, base_dir)
            touch_bg_path = _resolve_path(_first_present(item, touch_bg_key_candidates), touch_roots, base_dir)
            image_path = _resolve_path(_first_present(item, image_key_candidates), image_roots, base_dir)

            # require touch + image
            if (not touch_path) or (not image_path):
                filtered += 1
                continue

            if verify_files:
                if (touch_path is None) or (not os.path.isfile(touch_path)):
                    missing_touch += 1
                    continue
                if (image_path is None) or (not os.path.isfile(image_path)):
                    missing_image += 1
                    continue
                if touch_bg_path is not None and (not os.path.isfile(touch_bg_path)):
                    missing_bg += 1
                    touch_bg_path = None

            manifest.append(
                {
                    "global_index": len(manifest),
                    "sample_id": sample_id,
                    "touch_path": touch_path,
                    "touch_bg_path": touch_bg_path,
                    "image_path": image_path,
                    "source_meta": meta_path,
                }
            )

    pprint(
        f"[manifest] loaded={total} kept={len(manifest)} filtered={filtered} "
        f"missing_touch_files={missing_touch} missing_image_files={missing_image} missing_bg_files={missing_bg}"
    )
    return manifest


# -----------------------------------------------------------------------------
# Dataset / Collate
# -----------------------------------------------------------------------------
class TVLTouchImageDataset(Dataset):
    """
    返回 tactile tensor (3,224,224) + image_path
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

        try:
            pre = self._get_pre()
            dataset_hint = pre.infer_dataset_hint(m["touch_path"], m.get("touch_bg_path") or "")
            tactile = pre.load_tactile(
                tactile_path=m["touch_path"],
                dataset_hint=dataset_hint,
                tactile_background_path=m.get("touch_bg_path"),
            )

            if not isinstance(tactile, torch.Tensor):
                raise TypeError(f"tactile must be torch.Tensor, got {type(tactile)}")
            if tactile.dim() != 3:
                raise ValueError(f"tactile must be (3,H,W), got shape={tuple(tactile.shape)}")

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
            "image_path": str(m.get("image_path", "")),
            "valid": bool(valid),
            "error": err,
        }


def collate_raw(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "global_indices": torch.tensor([b["global_index"] for b in batch], dtype=torch.long),
        "sample_ids": [b["sample_id"] for b in batch],
        "tactiles": [b["tactile"] for b in batch],  # list[Tensor|None]
        "image_paths": [b["image_path"] for b in batch],  # list[str]
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
# Touch encoder (保持你原来 timm + ckpt 逻辑)
# -----------------------------------------------------------------------------
def _extract_state_dict_from_ckpt(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        out = {k: v for k, v in ckpt.items() if torch.is_tensor(v)}
        if out:
            return out
    raise ValueError("Cannot extract a usable state_dict from checkpoint.")


def _select_tactile_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
    return sd


class FrozenTVLTactileEncoder(torch.nn.Module):
    """
    input : (B,3,224,224)
    output: (B, N, D)
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

        drop_rate = float(cfg.get("drop_rate", 0.0))
        drop_path_rate = float(cfg.get("drop_path_rate", 0.0))

        self.model = timm.create_model(
            self.tactile_model,
            pretrained=False,
            num_classes=0,
            global_pool="",
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

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

            drop_prefix = ("head.", "fc.", "classifier.")
            sd = {k: v for k, v in sd.items() if not k.startswith(drop_prefix)}

            model_keys = set(self.model.state_dict().keys())
            model_has_norm = ("norm.weight" in model_keys and "norm.bias" in model_keys)
            model_has_fc_norm = ("fc_norm.weight" in model_keys and "fc_norm.bias" in model_keys)

            ckpt_has_norm = ("norm.weight" in sd and "norm.bias" in sd)
            ckpt_has_fc_norm = ("fc_norm.weight" in sd and "fc_norm.bias" in sd)

            if model_has_norm and (not ckpt_has_norm) and ckpt_has_fc_norm:
                sd["norm.weight"] = sd.pop("fc_norm.weight")
                sd["norm.bias"] = sd.pop("fc_norm.bias")
            elif model_has_fc_norm and (not ckpt_has_fc_norm) and ckpt_has_norm:
                sd["fc_norm.weight"] = sd.pop("norm.weight")
                sd["fc_norm.bias"] = sd.pop("norm.bias")

            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"Checkpoint keys did not fully match model keys. missing={missing} unexpected={unexpected}"
                )

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.to(device)

    @torch.inference_mode()
    def forward(self, tactile: torch.Tensor) -> torch.Tensor:
        if tactile.dim() != 4:
            raise ValueError(f"Expected tactile (B,3,H,W), got {tuple(tactile.shape)}")

        if self.input_dtype in ("fp16", "float16", "half"):
            tactile = tactile.half()
        elif self.input_dtype in ("bf16", "bfloat16"):
            tactile = tactile.bfloat16()
        else:
            tactile = tactile.float()

        tactile = tactile.to(self.device, non_blocking=True)

        if not hasattr(self.model, "forward_features"):
            raise RuntimeError(f"Model {self.tactile_model} has no forward_features()")
        tokens = self.model.forward_features(tactile)

        if self.l2_normalize:
            tokens = tokens / tokens.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return tokens


# -----------------------------------------------------------------------------
# Qwen image cache + slim load (只做 image，兼容你 YAML 的 image_audio_weight_cache)
# -----------------------------------------------------------------------------
@torch.inference_mode()
def extract_and_cache_qwen_image_weights_by_prefix(
    qwen_cfg: Dict[str, Any],
    *,
    save_path: str,
) -> None:
    """
    rank0:
      - CPU load full Thinker once
      - extract a reduced state_dict containing visual (+ possible projectors)
      - save to cache for slim meta-load on each rank
    """
    ensure_dir(os.path.dirname(save_path))
    model_name = qwen_cfg["model_name_or_path"]
    trust_remote_code = bool(qwen_cfg.get("trust_remote_code", True))
    cache_dtype = _parse_torch_dtype(qwen_cfg.get("cache_dtype", "fp16"))

    pprint(f"[rank0] extracting Qwen image weights (CPU) from: {model_name} -> {save_path}")

    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        torch_dtype=cache_dtype,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # collect names first (avoid huge model.state_dict() materialization)
    param_names = [n for n, _ in model.named_parameters()]
    buffer_names = [n for n, _ in model.named_buffers()]
    all_names = param_names + buffer_names

    prefixes = ["visual."]

    # possible projectors
    candidate_prefixes = [
        "multi_modal_projector.",
        "mm_projector.",
        "vision_projector.",
        "vision_proj.",
        "vision_proj_layer.",
    ]
    for cp in candidate_prefixes:
        if any(n.startswith(cp) for n in all_names):
            prefixes.append(cp)

    # heuristic: any non-language_model.* name that contains project/proj and vision/visual
    for n in all_names:
        if n.startswith("language_model."):
            continue
        if ("project" in n or "proj" in n) and ("vision" in n or "visual" in n):
            head = n.split(".")[0] + "."
            if head not in prefixes:
                prefixes.append(head)

    small_sd: Dict[str, torch.Tensor] = {}

    for n, p in model.named_parameters():
        if any(n.startswith(pref) for pref in prefixes):
            t = p.detach().cpu()
            if t.is_floating_point():
                t = t.to(dtype=cache_dtype)
            small_sd[n] = t

    for n, b in model.named_buffers():
        if any(n.startswith(pref) for pref in prefixes):
            t = b.detach().cpu()
            if t.is_floating_point():
                t = t.to(dtype=cache_dtype)
            small_sd[n] = t

    payload = {
        "model_name_or_path": model_name,
        "cache_dtype": str(cache_dtype),
        "prefixes": prefixes,
        "num_tensors": int(len(small_sd)),
        "state_dict": small_sd,
    }
    torch.save(payload, save_path)
    del model
    pprint(f"[rank0] saved image cache: {save_path} tensors={len(small_sd)} prefixes={prefixes}")


def _force_module_all_tensors_to_device_(m: nn.Module, device: torch.device) -> None:
    for _, p in m.named_parameters(recurse=True):
        if not torch.is_tensor(p):
            continue
        if p.device.type == "meta":
            continue
        if p.device != device:
            p.data = p.data.to(device=device, non_blocking=True)
    for _, b in m.named_buffers(recurse=True):
        if not torch.is_tensor(b):
            continue
        if b.device.type == "meta":
            continue
        if b.device != device:
            b.data = b.data.to(device=device, non_blocking=True)


def load_slim_qwen_thinker_from_image_cache(
    qwen_cfg: Dict[str, Any],
    *,
    cache_path: str,
    device: torch.device,
) -> Qwen3OmniMoeThinkerForConditionalGeneration:
    if init_empty_weights is None or set_module_tensor_to_device is None:
        raise RuntimeError("accelerate is required for slim model loading. Please install accelerate.")

    model_name = qwen_cfg["model_name_or_path"]
    trust_remote_code = bool(qwen_cfg.get("trust_remote_code", True))
    compute_dtype = _parse_torch_dtype(qwen_cfg.get("compute_dtype", qwen_cfg.get("dtype", "fp16")))

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

    # materialize only cached tensors
    for k, v in sd_small.items():
        if not torch.is_tensor(v):
            continue
        t = v
        if t.is_floating_point():
            t = t.to(dtype=compute_dtype)
        set_module_tensor_to_device(model, k, device=device, value=t.to(device=device, non_blocking=True))

    if hasattr(model, "visual") and isinstance(model.visual, nn.Module):
        _force_module_all_tensors_to_device_(model.visual, device=device)

    return model


def prepare_image_inputs(image_processor, images: List[Image.Image], **kwargs) -> Dict[str, torch.Tensor]:
    out = image_processor(images=images, return_tensors="pt", **kwargs)
    if "pixel_values" not in out:
        raise RuntimeError(f"image_processor missing 'pixel_values', got keys={list(out.keys())}")
    if "image_grid_thw" not in out:
        raise RuntimeError(f"image_processor missing 'image_grid_thw', got keys={list(out.keys())}")
    return {"pixel_values": out["pixel_values"], "image_grid_thw": out["image_grid_thw"]}


def _infer_token_len_from_grid_thw(grid_thw_row: torch.Tensor, merge_size: int, mode: str) -> int:
    t = int(grid_thw_row[0].item())
    h = int(grid_thw_row[1].item())
    w = int(grid_thw_row[2].item())
    if mode == "div_merge":
        ms = max(int(merge_size), 1)
        return int(t * ((h * w) // (ms * ms)))
    if mode == "no_merge":
        return int(t * h * w)
    raise ValueError(mode)


def _split_visual_flat_by_grid_thw(
    visual_embeds_flat: torch.Tensor,  # (total, D)
    *,
    grid_thw: torch.Tensor,  # (K,3)
    merge_size: int,
) -> List[torch.Tensor]:
    assert visual_embeds_flat.ndim == 2
    total = int(visual_embeds_flat.shape[0])
    k = int(grid_thw.shape[0])

    lens = [_infer_token_len_from_grid_thw(grid_thw[i], merge_size, "div_merge") for i in range(k)]
    if sum(lens) != total:
        lens2 = [_infer_token_len_from_grid_thw(grid_thw[i], merge_size, "no_merge") for i in range(k)]
        if sum(lens2) == total:
            lens = lens2
        else:
            diff = total - sum(lens)
            lens[-1] = max(lens[-1] + diff, 0)
            if sum(lens) != total:
                base = total // k
                lens = [base] * k
                lens[-1] += total - sum(lens)

    chunks: List[torch.Tensor] = []
    start = 0
    for li in lens:
        end = start + int(li)
        chunks.append(visual_embeds_flat[start:end])
        start = end
    return chunks


def _pad_or_trunc_tokens_with_mask(x: torch.Tensor, target: int) -> Tuple[torch.Tensor, torch.Tensor]:
    n = int(x.shape[0])
    h = int(x.shape[1])
    if n >= target:
        y = x[:target]
        mask = torch.ones((target,), device=x.device, dtype=torch.bool)
        return y, mask
    pad = torch.zeros((target - n, h), device=x.device, dtype=x.dtype)
    y = torch.cat([x, pad], dim=0)
    mask = torch.cat(
        [
            torch.ones((n,), device=x.device, dtype=torch.bool),
            torch.zeros((target - n,), device=x.device, dtype=torch.bool),
        ],
        dim=0,
    )
    return y, mask


class FrozenQwenImageTokens(nn.Module):
    """
    只做 image tokens：
      encode_image(image_paths) -> (B, Ti, H), (B, Ti)mask, (B,)valid
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        *,
        image_processor,
        device: torch.device,
    ):
        super().__init__()
        self.cfg = dict(cfg)
        self.image_processor = image_processor
        self.device = device

        self.image_tokens_target = int(cfg.get("image_tokens", 256))
        self.l2_normalize = bool(cfg.get("l2_normalize", False))
        self.image_processor_kwargs = dict(cfg.get("image_processor_kwargs", {}))

        cache_path = cfg.get("image_weight_cache", None) or cfg.get("image_audio_weight_cache", None)
        if not cache_path or (not os.path.isfile(cache_path)):
            raise FileNotFoundError(f"qwen image cache not found: {cache_path}")
        self.cache_path = cache_path

        self.model = load_slim_qwen_thinker_from_image_cache(cfg, cache_path=cache_path, device=self.device)

        # merge_size mirrors ms-swift qwen.py
        self.merge_size = int(getattr(self.image_processor, "merge_size", 1))

        # hidden_size will be finalized on first successful forward (avoid mismatch surprises)
        self.hidden_size = int(cfg.get("hidden_size", 0))  # 0 -> infer dynamically

    def _zeros_image(self, bsz: int, hidden: int, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros((bsz, self.image_tokens_target, hidden), device=self.device, dtype=dtype)

    def _zeros_mask(self, bsz: int) -> torch.Tensor:
        return torch.zeros((bsz, self.image_tokens_target), device=self.device, dtype=torch.bool)

    def _move_batch_(self, batch: Dict[str, torch.Tensor], compute_dtype: torch.dtype) -> None:
        for k, v in list(batch.items()):
            if torch.is_tensor(v):
                if v.is_floating_point():
                    batch[k] = v.to(device=self.device, dtype=compute_dtype, non_blocking=True)
                else:
                    batch[k] = v.to(device=self.device, non_blocking=True)

    @torch.inference_mode()
    def encode_image(self, image_paths: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = len(image_paths)
        compute_dtype = _parse_torch_dtype(self.cfg.get("compute_dtype", self.cfg.get("dtype", "fp16")))

        images: List[Image.Image] = []
        kept: List[int] = []
        valid = torch.zeros((bsz,), dtype=torch.uint8, device=self.device)

        for i, ip in enumerate(image_paths):
            try:
                if not os.path.isfile(ip):
                    raise FileNotFoundError(ip)
                with Image.open(ip) as im:
                    img = im.convert("RGB")
                    img.load()
                images.append(img)
                kept.append(i)
                valid[i] = 1
            except Exception:
                valid[i] = 0

        if len(kept) == 0:
            hidden = self.hidden_size if self.hidden_size > 0 else 1
            return self._zeros_image(bsz, hidden, dtype=compute_dtype), self._zeros_mask(bsz), valid

        img_in = prepare_image_inputs(self.image_processor, images, **self.image_processor_kwargs)
        self._move_batch_(img_in, compute_dtype)

        pixel_values = img_in["pixel_values"]
        image_grid_thw = img_in["image_grid_thw"].long()

        out = self.model.visual(pixel_values, grid_thw=image_grid_thw)
        visual_embeds = out[0] if isinstance(out, (tuple, list)) else out

        if visual_embeds.ndim not in (2, 3):
            raise RuntimeError(f"unexpected visual_embeds shape={tuple(visual_embeds.shape)}")

        hidden_now = int(visual_embeds.shape[-1])
        if self.hidden_size <= 0:
            self.hidden_size = hidden_now
        elif self.hidden_size != hidden_now:
            # 直接以实际为准（否则后续写 mmap 会 shape mismatch）
            self.hidden_size = hidden_now

        if visual_embeds.ndim == 3:
            chunks = [visual_embeds[i] for i in range(visual_embeds.shape[0])]
        else:
            chunks = _split_visual_flat_by_grid_thw(visual_embeds, grid_thw=image_grid_thw, merge_size=self.merge_size)

        padded_tokens: List[torch.Tensor] = []
        padded_masks: List[torch.Tensor] = []
        for c in chunks:
            tok, m = _pad_or_trunc_tokens_with_mask(c, self.image_tokens_target)
            padded_tokens.append(tok)
            padded_masks.append(m)

        tok_kept = torch.stack(padded_tokens, dim=0)  # (K,Ti,H)
        msk_kept = torch.stack(padded_masks, dim=0)   # (K,Ti)

        if self.l2_normalize:
            tok_kept = tok_kept / tok_kept.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        out_tok = self._zeros_image(bsz, self.hidden_size, dtype=tok_kept.dtype)
        out_msk = self._zeros_mask(bsz)

        for j, bi in enumerate(kept):
            out_tok[bi] = tok_kept[j]
            out_msk[bi] = msk_kept[j]

        return out_tok, out_msk, valid


# -----------------------------------------------------------------------------
# Memmap writer (touch + image)
# -----------------------------------------------------------------------------
@dataclass
class ShardPaths:
    touch_tokens: str
    image_tokens: str
    image_mask: str
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
        image_shape: Tuple[int, int],
        save_dtype: np.dtype,
    ):
        self.rank = rank
        self.num_samples = int(num_samples)

        shard_dir = os.path.join(out_dir, "shards")
        ensure_dir(shard_dir)

        self.paths = ShardPaths(
            touch_tokens=os.path.join(shard_dir, f"touch_tokens_rank{rank:02d}.mmap"),
            image_tokens=os.path.join(shard_dir, f"image_tokens_rank{rank:02d}.mmap"),
            image_mask=os.path.join(shard_dir, f"image_mask_rank{rank:02d}.mmap"),
            sample_ids=os.path.join(shard_dir, f"sample_ids_rank{rank:02d}.mmap"),
            global_indices=os.path.join(shard_dir, f"global_indices_rank{rank:02d}.mmap"),
            valid=os.path.join(shard_dir, f"valid_rank{rank:02d}.mmap"),
        )

        t_tok, t_dim = touch_shape
        i_tok, i_dim = image_shape

        self.touch_tokens = np.memmap(self.paths.touch_tokens, mode="w+", dtype=save_dtype, shape=(self.num_samples, t_tok, t_dim))
        self.image_tokens = np.memmap(self.paths.image_tokens, mode="w+", dtype=save_dtype, shape=(self.num_samples, i_tok, i_dim))
        self.image_mask = np.memmap(self.paths.image_mask, mode="w+", dtype=np.uint8, shape=(self.num_samples, i_tok))

        self.sample_ids = np.memmap(self.paths.sample_ids, mode="w+", dtype="S64", shape=(self.num_samples,))
        self.global_indices = np.memmap(self.paths.global_indices, mode="w+", dtype=np.int64, shape=(self.num_samples,))
        self.valid = np.memmap(self.paths.valid, mode="w+", dtype=np.uint8, shape=(self.num_samples,))
        self._ptr = 0

    def write_batch(
        self,
        touch_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
        image_mask: torch.Tensor,
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
        self.image_tokens[s:e] = image_tokens.detach().cpu().numpy()
        self.image_mask[s:e] = image_mask.detach().cpu().numpy().astype(np.uint8)
        self.global_indices[s:e] = global_indices.detach().cpu().numpy().astype(np.int64)
        self.valid[s:e] = valid.detach().cpu().numpy().astype(np.uint8)

        for i, sid in enumerate(sample_ids):
            self.sample_ids[s + i] = np.bytes_(sid.encode("utf-8")[:64])

        self._ptr = e

    def flush(self) -> None:
        self.touch_tokens.flush()
        self.image_tokens.flush()
        self.image_mask.flush()
        self.sample_ids.flush()
        self.global_indices.flush()
        self.valid.flush()

    def close(self) -> None:
        self.flush()


# -----------------------------------------------------------------------------
# DDP helpers
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
def _get_qwen_cfg_robust(encoders_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    你的报错核心修复点：保证能拿到 encoders.qwen
    - 优先 encoders.qwen
    - 如果用户没写 qwen，但写了 text（同一个 Qwen 模型），就 fallback 用 text 里的 model_name_or_path
    """
    qwen_cfg = encoders_cfg.get("qwen", None)
    if isinstance(qwen_cfg, dict) and len(qwen_cfg) > 0:
        return dict(qwen_cfg)

    # fallback: encoders.text
    text_cfg = encoders_cfg.get("text", None)
    if isinstance(text_cfg, dict) and "model_name_or_path" in text_cfg:
        fallback = {
            "model_name_or_path": text_cfg["model_name_or_path"],
            "trust_remote_code": bool(text_cfg.get("trust_remote_code", True)),
            # sane defaults
            "cache_dtype": "fp16",
            "compute_dtype": text_cfg.get("torch_dtype", "fp16"),
            "image_tokens": 256,
        }
        pprint("[warn] encoders.qwen not found, fallback to encoders.text for model_name_or_path")
        return fallback

    return {}  # caller will raise


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

    if n_local == 0:
        print(f"[rank{rank}] no samples assigned, skip.")
        ddp_barrier(device)
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        return

    # ---------------------------
    # Qwen image weights cache (rank0)  —— 关键修复：兼容 image_audio_weight_cache + 不存在就先缓存
    # ---------------------------
    qwen_cfg = _get_qwen_cfg_robust(ecfg)
    if not qwen_cfg:
        pprint(f"[debug] encoders keys = {list(ecfg.keys())}")
        raise ValueError("encoders.qwen is required for image extraction (or provide encoders.text as fallback).")

    # 兼容你 YAML 的 key：image_audio_weight_cache
    cache_path = (
        qwen_cfg.get("image_weight_cache", None)
        or qwen_cfg.get("image_audio_weight_cache", None)
        or os.path.join(out_dir, "qwen_image_weight_cache.pt")
    )

    if rank == 0:
        if overwrite or (not os.path.isfile(cache_path)):
            extract_and_cache_qwen_image_weights_by_prefix(qwen_cfg, save_path=cache_path)
        # 写回 cfg（两种 key 都写一份，避免你其它代码继续用 image_audio_weight_cache）
        qwen_cfg["image_weight_cache"] = cache_path
        qwen_cfg["image_audio_weight_cache"] = cache_path
    ddp_barrier(device)

    # all ranks set
    qwen_cfg["image_weight_cache"] = cache_path
    qwen_cfg["image_audio_weight_cache"] = cache_path

    # ---------------------------
    # Touch preprocess config (保持兼容你的 touch.repo_path)
    # ---------------------------
    touch_cfg = dict(ecfg.get("touch", {}))
    tvl_repo_path = touch_cfg.get("tvl_repo_path") or touch_cfg.get("repo_path")
    touch_pre_cfg = dict(
        tvl_repo_path=tvl_repo_path,
        crop_tacvis=bool(touch_cfg.get("crop_tacvis", False)),
        subtract_background=touch_cfg.get("subtract_background", None),
        augment_rgb=bool(touch_cfg.get("augment_rgb", False)),
        augment_tactile=bool(touch_cfg.get("augment_tactile", False)),
        random_drop=bool(touch_cfg.get("random_drop", False)),
        image_size=int(touch_cfg.get("image_size", 224)),
    )

    # Encoders
    touch_encoder = FrozenTVLTactileEncoder(touch_cfg, device=device)

    # processor（只需要 image_processor）
    proc_name = qwen_cfg.get("processor_name_or_path", qwen_cfg["model_name_or_path"])
    trust_remote_code = bool(qwen_cfg.get("trust_remote_code", True))
    use_fast = qwen_cfg.get("processor_use_fast", None)

    try:
        if use_fast is None:
            processor = AutoProcessor.from_pretrained(proc_name, trust_remote_code=trust_remote_code)
        else:
            processor = AutoProcessor.from_pretrained(proc_name, trust_remote_code=trust_remote_code, use_fast=bool(use_fast))
    except TypeError:
        # 某些版本 AutoProcessor 不接受 use_fast
        processor = AutoProcessor.from_pretrained(proc_name, trust_remote_code=trust_remote_code)

    if not hasattr(processor, "image_processor"):
        raise RuntimeError(f"AutoProcessor({proc_name}) has no image_processor")

    image_encoder = FrozenQwenImageTokens(
        qwen_cfg,
        image_processor=processor.image_processor,
        device=device,
    )

    # ---------------------------
    # Probe shapes
    # ---------------------------
    # touch probe
    probe_touch = None
    for m in local_manifest:
        pre = TVLTouchPreprocessor(TVLTouchPreprocessConfig(**touch_pre_cfg))
        dataset_hint = pre.infer_dataset_hint(m["touch_path"], m.get("touch_bg_path") or "")
        tactile = pre.load_tactile(
            tactile_path=m["touch_path"],
            dataset_hint=dataset_hint,
            tactile_background_path=m.get("touch_bg_path"),
        )
        with torch.inference_mode():
            probe_touch = touch_encoder(tactile.unsqueeze(0))
        break
    if probe_touch is None:
        raise RuntimeError(f"[rank{rank}] cannot probe touch shape!")

    # image probe (so hidden_size is finalized)
    probe_img_tok, probe_img_msk, probe_img_valid = image_encoder.encode_image([local_manifest[0]["image_path"]])
    if int(probe_img_valid[0].item()) != 1:
        raise RuntimeError(f"[rank{rank}] image probe failed: {local_manifest[0]['image_path']}")

    touch_shape = (int(probe_touch.shape[1]), int(probe_touch.shape[2]))
    image_shape = (int(image_encoder.image_tokens_target), int(image_encoder.hidden_size))

    pprint(f"[rank{rank}] touch_shape={touch_shape} image_shape={image_shape} cache_path={cache_path}")

    # ---------------------------
    # Writer
    # ---------------------------
    save_dtype = _parse_np_dtype_save(rcfg.get("save_dtype", "fp16"))

    shard_dir = os.path.join(out_dir, "shards")
    ensure_dir(shard_dir)
    test_path = os.path.join(shard_dir, f"touch_tokens_rank{rank:02d}.mmap")
    if (not overwrite) and os.path.exists(test_path):
        raise FileExistsError(f"Shard already exists: {test_path}. set runtime.overwrite=true")

    writer = MemmapShardWriter(
        out_dir=out_dir,
        rank=rank,
        num_samples=n_local,
        touch_shape=touch_shape,
        image_shape=image_shape,
        save_dtype=save_dtype,
    )

    # ---------------------------
    # Loader
    # ---------------------------
    ds = TVLTouchImageDataset(
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
    i_tok, i_dim = image_shape
    torch_out_dtype = torch.float16 if save_dtype == np.float16 else torch.float32

    for step, batch in enumerate(loader):
        valid_touch = batch["valid"].clone()  # (B,)
        tactiles_list = batch["tactiles"]
        image_paths = batch["image_paths"]
        sample_ids = batch["sample_ids"]
        global_indices = batch["global_indices"]

        b = len(sample_ids)
        touch_tokens_out = torch.zeros((b, t_tok, t_dim), device=device, dtype=torch_out_dtype)
        image_tokens_out = torch.zeros((b, i_tok, i_dim), device=device, dtype=torch_out_dtype)
        image_mask_out = torch.zeros((b, i_tok), device=device, dtype=torch.bool)

        valid_final = valid_touch.clone()

        idx = (valid_touch == 1).nonzero(as_tuple=False).squeeze(1).tolist()
        if len(idx) > 0:
            tactile_valid = [tactiles_list[i] for i in idx]
            img_valid_paths = [image_paths[i] for i in idx]

            tactile_batch = torch.stack(tactile_valid, dim=0)
            with torch.inference_mode():
                touch_tokens_v = touch_encoder(tactile_batch).to(dtype=torch_out_dtype)  # (K,T,D)

            with torch.inference_mode():
                img_tokens_v, img_mask_v, img_valid_v = image_encoder.encode_image(img_valid_paths)
                img_tokens_v = img_tokens_v.to(dtype=torch_out_dtype)

            ok_pos = (img_valid_v == 1).nonzero(as_tuple=False).squeeze(1).tolist()
            bad_pos = (img_valid_v == 0).nonzero(as_tuple=False).squeeze(1).tolist()

            for p in bad_pos:
                bi = idx[p]
                valid_final[bi] = 0  # paired validity: image failed => invalid

            if len(ok_pos) > 0:
                ok_bi = [idx[p] for p in ok_pos]
                touch_tokens_out[ok_bi] = touch_tokens_v[ok_pos]
                image_tokens_out[ok_bi] = img_tokens_v[ok_pos]
                image_mask_out[ok_bi] = img_mask_v[ok_pos]

        writer.write_batch(
            touch_tokens=touch_tokens_out,
            image_tokens=image_tokens_out,
            image_mask=image_mask_out,
            sample_ids=sample_ids,
            global_indices=global_indices,
            valid=valid_final,
        )

        num_done += b
        if flush_every > 0 and (step + 1) % flush_every == 0:
            writer.flush()

        if log_every > 0 and (step + 1) % log_every == 0:
            elapsed = time.time() - t0
            speed = num_done / max(elapsed, 1e-9)
            print(f"[rank{rank}] step={step+1} done={num_done}/{n_local} speed={speed:.2f} samples/s")

    writer.close()

    # save shard info
    shard_info = {
        "rank": rank,
        "num_samples": n_local,
        "touch": {"num_tokens": touch_shape[0], "hidden": touch_shape[1], "dtype": str(save_dtype)},
        "image": {"num_tokens": image_shape[0], "hidden": image_shape[1], "dtype": str(save_dtype)},
        "paths": {
            "touch_tokens": writer.paths.touch_tokens,
            "image_tokens": writer.paths.image_tokens,
            "image_mask": writer.paths.image_mask,
            "sample_ids": writer.paths.sample_ids,
            "global_indices": writer.paths.global_indices,
            "valid": writer.paths.valid,
        },
        "qwen": {
            "model_name_or_path": qwen_cfg.get("model_name_or_path"),
            "cache_path": cache_path,
            "cache_dtype": qwen_cfg.get("cache_dtype", "fp16"),
            "compute_dtype": qwen_cfg.get("compute_dtype", qwen_cfg.get("dtype", "fp16")),
            "image_tokens": int(image_encoder.image_tokens_target),
            "merge_size": int(image_encoder.merge_size),
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
            "features": {
                "touch": {"num_tokens": touch_shape[0], "hidden": touch_shape[1], "dtype": str(save_dtype)},
                "image": {"num_tokens": image_shape[0], "hidden": image_shape[1], "dtype": str(save_dtype)},
            },
            "pairs": ["touch-image", "image-touch"],
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
    cfg_path = os.environ.get("MM_CFG", "")
    if not cfg_path:
        raise ValueError("Please set env MM_CFG=/path/to/your.yaml")

    cfg = load_yaml(cfg_path)
    if not cfg:
        raise ValueError(f"Empty config: {cfg_path}")

    if is_torchrun_env():
        rank, world_size, local_rank = get_rank_info()
        worker_main(rank=rank, world_size=world_size, local_rank=local_rank, cfg=cfg, init_method="env://")
        return

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


