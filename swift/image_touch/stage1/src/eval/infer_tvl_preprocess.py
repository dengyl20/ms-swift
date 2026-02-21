# -*- coding: utf-8 -*-
"""
infer_tvl_preprocess.py

【功能】
- 把 TVL 测试集的 SSVTP + HCT(data1/2/3) 分别组织成 meta json
- 分别抽取 tactile/touch tokens，并保存成 memmap + dataset_info.yaml
- 输出格式兼容 swift.tvl.stage1.src.data.feature_dataset.ProcessedTouchTextFeatureDataset

【输出目录结构（重点：两份 yaml 分开存）】
OUT_DIR/
  ssvtp/
    meta_test.json
    features/
      dataset_info.yaml
      shards/
        touch_tokens_rank00.mmap
        text_embeds_rank00.mmap
        text_mask_rank00.mmap
        sample_ids_rank00.mmap
        global_indices_rank00.mmap
        valid_rank00.mmap

  hct/
    meta_test.json
    features/
      dataset_info.yaml
      shards/...

【用法】
(1) 直接改文件开头 DEFAULT_* 配置，然后：
    CUDA_VISIBLE_DEVICES=0 python infer_tvl_preprocess.py

(2) 或命令行覆盖（推荐你现在的方式）：
    CUDA_VISIBLE_DEVICES=0 python infer_tvl_preprocess.py \
      --out_dir /vast/.../tvl_test_features_all \
      --include_ssvtp --include_hct --overwrite
"""

from __future__ import annotations

import os
import re
import csv
import json
import time
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

from transformers import AutoTokenizer

# --------- torch safe globals (你需要的) ---------
from torch.serialization import add_safe_globals
add_safe_globals([argparse.Namespace])

# --------- OFFLINE ---------
os.environ["HF_HUB_OFFLINE"] = "1"

# --------- ms-swift TVL tactile preprocess ---------
from swift.tvl.stage1.src.preprocess.tvl_touch_encoder import (
    TVLTouchPreprocessConfig,
    TVLTouchPreprocessor,
)

# =========================================================
# ✅ 默认设置（你要的：写在文件开头）
# =========================================================

DEFAULT_OUT_DIR = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/tvl_test_features_all"

DEFAULT_INCLUDE_SSVTP = True
DEFAULT_INCLUDE_HCT = True

DEFAULT_DATASET_ROOT = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/dataset/TVL/Touch-Vision-Language-Dataset/tvl_dataset"
DEFAULT_SSVTP_DIR = os.path.join(DEFAULT_DATASET_ROOT, "ssvtp")
DEFAULT_HCT_DIR = os.path.join(DEFAULT_DATASET_ROOT, "hct")

DEFAULT_SSVTP_TEST_CSV = os.path.join(DEFAULT_SSVTP_DIR, "test.csv")
DEFAULT_SSVTP_PREFIX_TXT = os.path.join(DEFAULT_SSVTP_DIR, "text_prefix.txt")

DEFAULT_HCT_TEST_CSVS = [
    os.path.join(DEFAULT_HCT_DIR, "data1", "test.csv"),
    os.path.join(DEFAULT_HCT_DIR, "data2", "test.csv"),
    os.path.join(DEFAULT_HCT_DIR, "data3", "test.csv"),
]

DEFAULT_TVL_REPO_PATH = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/TVL/tvl"
DEFAULT_TACTILE_ENCODER_CKPT = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/dataset/TVL/checkpoints/tvl_enc_vitb.pth"
DEFAULT_TACTILE_MODEL = "vit_base_patch16_224"

DEFAULT_QWEN_TOKENIZER = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 16

# 是否做背景相减（HCT 有背景图；SSVTP 没有）
# "none" 或 "background"
DEFAULT_SUBTRACT_BACKGROUND = "none"

# prompt 生成策略（只写 meta json 用，不影响抽特征）
DEFAULT_SSVTP_PROMPT_MODE = "random_prefix"  # random_prefix | fixed
DEFAULT_HCT_PROMPT_MODE = "fixed"           # fixed | random_prefix
DEFAULT_FIXED_PROMPT = "This image gives tactile feelings of?"

# 你训练 features 的 spec
TOUCH_NUM_TOKENS = 197
TOUCH_HIDDEN = 768
TEXT_MAX_LEN = 24
TEXT_HIDDEN = 2048


# =========================================================
# helper
# =========================================================
def ensure_dir(p: str) -> None:
    if p:
        os.makedirs(p, exist_ok=True)


def sniff_csv_delimiter(path: str) -> str:
    with open(path, "r", encoding="utf-8", newline="") as f:
        head = f.readline()
    return "\t" if "\t" in head else ","


def read_lines(path: str) -> List[str]:
    if not path or (not os.path.exists(path)):
        return []
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(s)
    return out


def make_question_from_prefix(prefix: str) -> str:
    p = (prefix or "").strip()
    if not p:
        p = "This image gives tactile feelings of"
    if not p.endswith("?"):
        p = p.rstrip() + "?"
    return p


def normalize_id_from_ssvtp_rgb_rel(rgb_rel: str) -> str:
    # images_rgb/image_437_rgb.jpg -> 000000000437
    base = os.path.basename(rgb_rel)
    m = re.search(r"image_(\d+)_rgb", base)
    if m:
        return str(int(m.group(1))).zfill(12)
    return os.path.splitext(base)[0]


def ssvtp_rgb_rel_to_tactile_rel(rgb_rel: str) -> str:
    # images_rgb/image_106_rgb.jpg -> images_tac/image_106_tac.jpg
    p = rgb_rel.strip().lstrip("/")
    p = p.replace("images_rgb/", "images_tac/")
    p = p.replace("_rgb.", "_tac.")
    p = p.replace("_rgb", "_tac")
    return p


def abs_join(base: str, rel_or_abs: str) -> str:
    p = (rel_or_abs or "").strip()
    if not p:
        return ""
    if os.path.isabs(p):
        return p
    return os.path.join(base, p)


def short_sample_id(s: str) -> str:
    s = (s or "").strip()
    b = s.encode("utf-8", errors="ignore")
    if len(b) <= 64:
        return s
    return b[:64].decode("utf-8", errors="ignore")


# =========================================================
# tactile encoder (timm ViT)
# =========================================================
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
    output: (B, 197, 768) for vit_base_patch16_224
    """

    def __init__(
        self,
        tactile_model: str,
        checkpoint_path: str,
        device: torch.device,
        l2_normalize: bool = True,
        strict_load: bool = False,
    ):
        super().__init__()
        try:
            import timm
        except Exception as e:
            raise ImportError("timm is required. Please `pip install timm`.") from e

        self.device = device
        self.l2_normalize = bool(l2_normalize)

        self.model = timm.create_model(
            tactile_model,
            pretrained=False,
            num_classes=0,
            global_pool="",
        )

        if checkpoint_path and os.path.isfile(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            sd = _extract_state_dict_from_ckpt(ckpt)
            sd = _select_tactile_state_dict(sd)
            sd = {k: v for k, v in sd.items() if not k.startswith(("head.", "fc.", "classifier."))}

            mk = set(self.model.state_dict().keys())
            model_has_norm = ("norm.weight" in mk and "norm.bias" in mk)
            model_has_fc_norm = ("fc_norm.weight" in mk and "fc_norm.bias" in mk)
            ckpt_has_norm = ("norm.weight" in sd and "norm.bias" in sd)
            ckpt_has_fc_norm = ("fc_norm.weight" in sd and "fc_norm.bias" in sd)

            if model_has_norm and (not ckpt_has_norm) and ckpt_has_fc_norm:
                sd["norm.weight"] = sd.pop("fc_norm.weight")
                sd["norm.bias"] = sd.pop("fc_norm.bias")
            elif model_has_fc_norm and (not ckpt_has_fc_norm) and ckpt_has_norm:
                sd["fc_norm.weight"] = sd.pop("norm.weight")
                sd["fc_norm.bias"] = sd.pop("norm.bias")

            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            if (missing or unexpected) and strict_load:
                raise RuntimeError(f"tactile ckpt not fully matched: missing={missing} unexpected={unexpected}")
            if missing or unexpected:
                print(f"[WARN] tactile encoder strict=False load: missing={len(missing)} unexpected={len(unexpected)}")

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.to(device)

    @torch.inference_mode()
    def forward(self, tactile: torch.Tensor) -> torch.Tensor:
        tactile = tactile.to(self.device, non_blocking=True).float()
        if not hasattr(self.model, "forward_features"):
            raise RuntimeError("timm model has no forward_features()")
        tokens = self.model.forward_features(tactile)
        if isinstance(tokens, (tuple, list)):
            tokens = tokens[0]
        if tokens.dim() == 4:
            tokens = tokens.flatten(2).transpose(1, 2).contiguous()
        if tokens.dim() != 3:
            raise RuntimeError(f"Unexpected tokens shape: {tuple(tokens.shape)}")
        if self.l2_normalize:
            tokens = tokens / tokens.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return tokens


# =========================================================
# memmap paths
# =========================================================
@dataclass
class FeaturePaths:
    out_dir: str

    @property
    def shard_dir(self) -> str:
        return os.path.join(self.out_dir, "shards")

    @property
    def dataset_info_yaml(self) -> str:
        return os.path.join(self.out_dir, "dataset_info.yaml")

    @property
    def touch_tokens_mmap(self) -> str:
        return os.path.join(self.shard_dir, "touch_tokens_rank00.mmap")

    @property
    def text_embeds_mmap(self) -> str:
        return os.path.join(self.shard_dir, "text_embeds_rank00.mmap")

    @property
    def text_mask_mmap(self) -> str:
        return os.path.join(self.shard_dir, "text_mask_rank00.mmap")

    @property
    def sample_ids_mmap(self) -> str:
        return os.path.join(self.shard_dir, "sample_ids_rank00.mmap")

    @property
    def global_indices_mmap(self) -> str:
        return os.path.join(self.shard_dir, "global_indices_rank00.mmap")

    @property
    def valid_mmap(self) -> str:
        return os.path.join(self.shard_dir, "valid_rank00.mmap")


# =========================================================
# build meta: split by dataset
# =========================================================
def build_meta_ssvtp(
    *,
    ssvtp_dir: str,
    ssvtp_test_csv: str,
    ssvtp_prefix_txt: str,
    seed: int,
    prompt_mode: str,
    fixed_prompt: str,
) -> List[Dict[str, Any]]:
    import random
    rng = random.Random(seed)
    prefixes = read_lines(ssvtp_prefix_txt) or ["This image gives tactile feelings of"]

    def pick_question() -> str:
        if (prompt_mode or "fixed").lower() == "random_prefix":
            return make_question_from_prefix(rng.choice(prefixes))
        q = (fixed_prompt or "This image gives tactile feelings of?").strip()
        if not q.endswith("?"):
            q += "?"
        return q

    if not os.path.exists(ssvtp_test_csv):
        raise FileNotFoundError(f"SSVTP test.csv not found: {ssvtp_test_csv}")

    delim = sniff_csv_delimiter(ssvtp_test_csv)
    items: List[Dict[str, Any]] = []
    gidx = 0

    with open(ssvtp_test_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        for row in reader:
            url = (row.get("url") or "").strip()
            caption = (row.get("caption") or "").strip()
            if not url or not caption:
                continue

            num_id = normalize_id_from_ssvtp_rgb_rel(url)
            sid = short_sample_id(f"ssvtp_{num_id}")

            rgb_abs = abs_join(ssvtp_dir, url)
            tac_abs = abs_join(ssvtp_dir, ssvtp_rgb_rel_to_tactile_rel(url))

            q = pick_question()

            items.append(
                {
                    "global_index": gidx,
                    "id": sid,
                    "dataset": "ssvtp",
                    "source_csv": ssvtp_test_csv,
                    "rgb": rgb_abs,
                    "tactile": tac_abs,
                    "tactile_background": None,
                    "caption": caption,
                    "conversations": [
                        {"from": "human", "value": "<image>\n" + q},
                        {"from": "gpt", "value": caption},
                    ],
                }
            )
            gidx += 1

    return items


def build_meta_hct(
    *,
    hct_test_csvs: List[str],
    seed: int,
    prompt_mode: str,
    fixed_prompt: str,
    ssvtp_prefix_txt: str,  # 复用 prefix（可选）
) -> List[Dict[str, Any]]:
    import random
    rng = random.Random(seed)
    prefixes = read_lines(ssvtp_prefix_txt) or ["This image gives tactile feelings of"]

    def pick_question() -> str:
        if (prompt_mode or "fixed").lower() == "random_prefix":
            return make_question_from_prefix(rng.choice(prefixes))
        q = (fixed_prompt or "This image gives tactile feelings of?").strip()
        if not q.endswith("?"):
            q += "?"
        return q

    items: List[Dict[str, Any]] = []
    gidx = 0

    for csv_path in hct_test_csvs:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"HCT test.csv not found: {csv_path}")

        base_dir = os.path.dirname(csv_path)       # .../data1
        subset = os.path.basename(base_dir)        # data1/data2/data3
        delim = sniff_csv_delimiter(csv_path)

        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=delim)
            for row_idx, row in enumerate(reader):
                url = (row.get("url") or "").strip()
                tactile = (row.get("tactile") or "").strip()
                tactile_bg = (row.get("tactile_background") or "").strip()
                caption = (row.get("caption") or "").strip()
                if (not tactile) or (not caption):
                    continue

                sid = short_sample_id(f"hct_{subset}_{row_idx:06d}")
                rgb_abs = abs_join(base_dir, url) if url else ""
                tac_abs = abs_join(base_dir, tactile)
                bg_abs = abs_join(base_dir, tactile_bg) if tactile_bg else ""

                q = pick_question()

                items.append(
                    {
                        "global_index": gidx,
                        "id": sid,
                        "dataset": "hct",
                        "subset": subset,
                        "source_csv": csv_path,
                        "rgb": rgb_abs,
                        "tactile": tac_abs,
                        "tactile_background": (bg_abs if bg_abs else None),
                        "caption": caption,
                        "conversations": [
                            {"from": "human", "value": "<image>\n" + q},
                            {"from": "gpt", "value": caption},
                        ],
                    }
                )
                gidx += 1

    return items


# =========================================================
# feature extraction (single dataset -> single yaml)
# =========================================================
def extract_features_one_dataset(
    *,
    items: List[Dict[str, Any]],
    out_feature_dir: str,                 # .../features
    meta_json_path: str,                  # .../meta_test.json (写入 dataset_info.yaml 的 meta_paths)
    tvl_repo_path: str,
    subtract_background: Optional[str],   # None or "background"
    tactile_model: str,
    tactile_ckpt: str,
    qwen_tokenizer_name_or_path: str,
    batch_size: int,
    overwrite: bool,
    strict_tactile_ckpt: bool,
) -> str:
    ensure_dir(out_feature_dir)
    paths = FeaturePaths(out_feature_dir)
    ensure_dir(paths.shard_dir)

    if (not overwrite) and os.path.exists(paths.dataset_info_yaml):
        print(f"[INFO] dataset_info.yaml exists, skip extract: {paths.dataset_info_yaml}")
        return paths.dataset_info_yaml

    n = len(items)
    if n <= 0:
        raise RuntimeError("No items to extract.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] extracting features: out={out_feature_dir} device={device} n={n}")

    # preprocessor
    pre_cfg = TVLTouchPreprocessConfig(
        tvl_repo_path=tvl_repo_path,
        crop_tacvis=False,
        subtract_background=subtract_background,
        augment_rgb=False,
        augment_tactile=False,
        random_drop=False,
        image_size=224,
    )
    pre = TVLTouchPreprocessor(pre_cfg)

    # tactile encoder
    tac_enc = FrozenTVLTactileEncoder(
        tactile_model=tactile_model,
        checkpoint_path=tactile_ckpt,
        device=device,
        l2_normalize=True,
        strict_load=strict_tactile_ckpt,
    )

    # tokenizer (only for text_mask)
    tokenizer = AutoTokenizer.from_pretrained(qwen_tokenizer_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.convert_ids_to_tokens(0)

    # memmaps
    touch_mm = np.memmap(paths.touch_tokens_mmap, mode="w+", dtype=np.float16, shape=(n, TOUCH_NUM_TOKENS, TOUCH_HIDDEN))
    text_mm = np.memmap(paths.text_embeds_mmap, mode="w+", dtype=np.float16, shape=(n, TEXT_MAX_LEN, TEXT_HIDDEN))
    mask_mm = np.memmap(paths.text_mask_mmap, mode="w+", dtype=np.uint8, shape=(n, TEXT_MAX_LEN))
    sid_mm = np.memmap(paths.sample_ids_mmap, mode="w+", dtype="S64", shape=(n,))
    gidx_mm = np.memmap(paths.global_indices_mmap, mode="w+", dtype=np.int64, shape=(n,))
    valid_mm = np.memmap(paths.valid_mmap, mode="w+", dtype=np.uint8, shape=(n,))

    # text_embeds 全0（避免泄漏）
    text_mm[:] = 0

    # text_mask 来自 caption tokenizer attention_mask（max_len=24）
    captions = [str(it.get("caption", "")) for it in items]
    enc = tokenizer(
        captions,
        padding="max_length",
        truncation=True,
        max_length=TEXT_MAX_LEN,
        add_special_tokens=False,
        return_tensors="pt",
    )
    mask_mm[:] = enc["attention_mask"].to(torch.uint8).cpu().numpy()

    # ids / indices
    for i, it in enumerate(items):
        sid = short_sample_id(str(it.get("id", "")))
        gidx = int(it.get("global_index", i))
        sid_mm[i] = np.bytes_(sid.encode("utf-8")[:64])
        gidx_mm[i] = gidx

    # tactile -> tokens batching
    t0 = time.time()
    ok = 0
    pending_tensors: List[torch.Tensor] = []
    pending_indices: List[int] = []

    def flush_batch():
        nonlocal ok
        if not pending_indices:
            return
        tactile_batch = torch.stack(pending_tensors, dim=0)
        with torch.inference_mode():
            tokens = tac_enc(tactile_batch)  # (B,197,768)
        tokens = tokens.detach().cpu().to(torch.float16).numpy()

        for bi, idx in enumerate(pending_indices):
            touch_mm[idx] = tokens[bi]
            valid_mm[idx] = 1
            ok += 1

        pending_tensors.clear()
        pending_indices.clear()

    for i, it in enumerate(items):
        tac_path = str(it.get("tactile", "")).strip()
        bg_path = it.get("tactile_background", None)
        bg_path = str(bg_path).strip() if bg_path else ""

        try:
            if not tac_path or (not os.path.isfile(tac_path)):
                raise FileNotFoundError(f"tactile missing: {tac_path}")
            if bg_path and (not os.path.isfile(bg_path)):
                bg_path = ""  # bg 不强制

            dataset_hint = pre.infer_dataset_hint(tac_path, bg_path)
            tactile = pre.load_tactile(
                tactile_path=tac_path,
                dataset_hint=dataset_hint,
                tactile_background_path=(bg_path if bg_path else None),
            )
            if not isinstance(tactile, torch.Tensor) or tactile.dim() != 3:
                raise RuntimeError(f"bad tactile tensor: type={type(tactile)} shape={getattr(tactile,'shape',None)}")

            pending_tensors.append(tactile)
            pending_indices.append(i)

            if len(pending_indices) >= max(1, int(batch_size)):
                flush_batch()

        except Exception as e:
            touch_mm[i] = 0
            valid_mm[i] = 0
            print(f"[WARN] extract failed i={i} id={it.get('id')} : {repr(e)}")

        if (i + 1) % 50 == 0 or (i + 1) == n:
            flush_batch()
            elapsed = time.time() - t0
            print(f"[INFO] progress {i+1}/{n} ok={ok} elapsed={elapsed:.1f}s")

    flush_batch()

    # flush
    touch_mm.flush(); text_mm.flush(); mask_mm.flush()
    sid_mm.flush(); gidx_mm.flush(); valid_mm.flush()

    # dataset_info.yaml（单 shard）
    shard = {
        "rank": 0,
        "num_samples": int(n),
        "touch": {"num_tokens": TOUCH_NUM_TOKENS, "hidden": TOUCH_HIDDEN, "dtype": "float16"},
        "text": {"max_len": TEXT_MAX_LEN, "hidden": TEXT_HIDDEN, "dtype": "float16"},
        "touch_preprocess": {
            "tvl_repo_path": tvl_repo_path,
            "crop_tacvis": False,
            "subtract_background": subtract_background,
            "augment_rgb": False,
            "augment_tactile": False,
            "random_drop": False,
            "image_size": 224,
        },
        "touch_encoder": {
            "tactile_model": tactile_model,
            "checkpoint_path": tactile_ckpt,
            "out_dim": TOUCH_HIDDEN,
            "l2_normalize": True,
        },
        "paths": {
            "touch_tokens": paths.touch_tokens_mmap,
            "text_embeds": paths.text_embeds_mmap,
            "text_mask": paths.text_mask_mmap,
            "sample_ids": paths.sample_ids_mmap,
            "global_indices": paths.global_indices_mmap,
            "valid": paths.valid_mmap,
        },
    }

    dataset_info = {
        "version": 1,
        "num_samples_total": int(n),
        "world_size": 1,
        "split_mode": "contiguous",
        "dataset": {
            "meta_paths": [meta_json_path],
            "touch_key_candidates": ["tactile"],
            "touch_bg_key_candidates": ["tactile_background"],
            "text_key_candidates": ["caption"],
            "gpt_text_strategy": "first",
        },
        "features": {
            "touch": {"num_tokens": TOUCH_NUM_TOKENS, "hidden": TOUCH_HIDDEN, "dtype": "float16"},
            "text": {"max_len": TEXT_MAX_LEN, "hidden": TEXT_HIDDEN, "dtype": "float16"},
        },
        "pairs": ["touch-text", "text-touch"],
        "shards": [shard],
    }

    with open(paths.dataset_info_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_info, f, sort_keys=False)

    print(f"[INFO] dataset_info.yaml saved: {paths.dataset_info_yaml}")
    return paths.dataset_info_yaml


# =========================================================
# main
# =========================================================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)

    ap.add_argument("--include_ssvtp", action="store_true", default=DEFAULT_INCLUDE_SSVTP)
    ap.add_argument("--include_hct", action="store_true", default=DEFAULT_INCLUDE_HCT)

    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--overwrite", action="store_true", default=False)

    ap.add_argument("--ssvtp_dir", type=str, default=DEFAULT_SSVTP_DIR)
    ap.add_argument("--ssvtp_test_csv", type=str, default=DEFAULT_SSVTP_TEST_CSV)
    ap.add_argument("--ssvtp_prefix_txt", type=str, default=DEFAULT_SSVTP_PREFIX_TXT)

    ap.add_argument("--hct_test_csvs", type=str, nargs="*", default=DEFAULT_HCT_TEST_CSVS)

    ap.add_argument("--tvl_repo_path", type=str, default=DEFAULT_TVL_REPO_PATH)
    ap.add_argument("--subtract_background", type=str, default=DEFAULT_SUBTRACT_BACKGROUND, choices=["none", "background"])
    ap.add_argument("--tactile_model", type=str, default=DEFAULT_TACTILE_MODEL)
    ap.add_argument("--tactile_ckpt", type=str, default=DEFAULT_TACTILE_ENCODER_CKPT)
    ap.add_argument("--strict_tactile_ckpt", action="store_true", default=False)

    ap.add_argument("--qwen_tokenizer", type=str, default=DEFAULT_QWEN_TOKENIZER)

    ap.add_argument("--ssvtp_prompt_mode", type=str, default=DEFAULT_SSVTP_PROMPT_MODE, choices=["random_prefix", "fixed"])
    ap.add_argument("--hct_prompt_mode", type=str, default=DEFAULT_HCT_PROMPT_MODE, choices=["fixed", "random_prefix"])
    ap.add_argument("--fixed_prompt", type=str, default=DEFAULT_FIXED_PROMPT)

    return ap.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    if (not args.include_ssvtp) and (not args.include_hct):
        raise ValueError("Need at least one of --include_ssvtp / --include_hct")

    subtract_bg = None if args.subtract_background == "none" else "background"

    # ---------------- SSVTP ----------------
    if args.include_ssvtp:
        out_ssvtp = os.path.join(args.out_dir, "ssvtp")
        ensure_dir(out_ssvtp)
        meta_json = os.path.join(out_ssvtp, "meta_test.json")
        feat_dir = os.path.join(out_ssvtp, "features")

        items = build_meta_ssvtp(
            ssvtp_dir=args.ssvtp_dir,
            ssvtp_test_csv=args.ssvtp_test_csv,
            ssvtp_prefix_txt=args.ssvtp_prefix_txt,
            seed=int(args.seed),
            prompt_mode=args.ssvtp_prompt_mode,
            fixed_prompt=args.fixed_prompt,
        )

        if (not args.overwrite) and os.path.exists(meta_json):
            print(f"[INFO] meta exists, skip write: {meta_json}")
        else:
            with open(meta_json, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
            print(f"[INFO] SSVTP meta saved: {meta_json} n={len(items)}")

        yaml_path = extract_features_one_dataset(
            items=items,
            out_feature_dir=feat_dir,
            meta_json_path=meta_json,
            tvl_repo_path=args.tvl_repo_path,
            subtract_background=None,  # SSVTP 没 bg，固定 None 更合理
            tactile_model=args.tactile_model,
            tactile_ckpt=args.tactile_ckpt,
            qwen_tokenizer_name_or_path=args.qwen_tokenizer,
            batch_size=int(args.batch_size),
            overwrite=bool(args.overwrite),
            strict_tactile_ckpt=bool(args.strict_tactile_ckpt),
        )
        print(f"[INFO] SSVTP features yaml: {yaml_path}")

    # ---------------- HCT ----------------
    if args.include_hct:
        out_hct = os.path.join(args.out_dir, "hct")
        ensure_dir(out_hct)
        meta_json = os.path.join(out_hct, "meta_test.json")
        feat_dir = os.path.join(out_hct, "features")

        items = build_meta_hct(
            hct_test_csvs=list(args.hct_test_csvs),
            seed=int(args.seed),
            prompt_mode=args.hct_prompt_mode,
            fixed_prompt=args.fixed_prompt,
            ssvtp_prefix_txt=args.ssvtp_prefix_txt,
        )

        if (not args.overwrite) and os.path.exists(meta_json):
            print(f"[INFO] meta exists, skip write: {meta_json}")
        else:
            with open(meta_json, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
            print(f"[INFO] HCT meta saved: {meta_json} n={len(items)}")

        yaml_path = extract_features_one_dataset(
            items=items,
            out_feature_dir=feat_dir,
            meta_json_path=meta_json,
            tvl_repo_path=args.tvl_repo_path,
            subtract_background=subtract_bg,  # HCT 才考虑 bgsub
            tactile_model=args.tactile_model,
            tactile_ckpt=args.tactile_ckpt,
            qwen_tokenizer_name_or_path=args.qwen_tokenizer,
            batch_size=int(args.batch_size),
            overwrite=bool(args.overwrite),
            strict_tactile_ckpt=bool(args.strict_tactile_ckpt),
        )
        print(f"[INFO] HCT features yaml: {yaml_path}")

    print("\n[INFO] Done.")
    print(f"OUT_DIR = {args.out_dir}")
    if args.include_ssvtp:
        print(f"- SSVTP yaml: {os.path.join(args.out_dir, 'ssvtp', 'features', 'dataset_info.yaml')}")
    if args.include_hct:
        print(f"- HCT   yaml: {os.path.join(args.out_dir, 'hct',   'features', 'dataset_info.yaml')}")


if __name__ == "__main__":
    main()