# -*- coding: utf-8 -*-
"""
infer_ssvtp_test_touch_ae_qwen3.py

SSVTP 测试集（42条）touch-only inference pipeline（单文件）：
1) 从 ssvtp/test.csv + ssvtp/text_prefix.txt 组织 meta json（含 conversations）
2) 抽取 touch tokens 特征并保存为 memmap + dataset_info.yaml（兼容 ProcessedTouchTextFeatureDataset）
3) 用 UnifiedTouchTextAE(point/touch AE) 输出 text token embeddings，注入到 Qwen3-Omni prompt 的 <touch> token span，生成回答
4) 打印 Pred vs GT（GT=caption）

重要说明：
- 为避免信息泄漏：本脚本抽特征时 text_embeds 默认写全 0；text_mask 来自 tokenizer(caption)。
  inference 时 AE 的 text_feat 也用全 0（只用 shape + mask）。
- 你不做 baseline（GT text_embeds 注入）的话，完全不需要真正的 text_embeds。
- system prompt 中不会出现字面 "<touch>"，避免 tokenizer 特殊处理影响注入定位。
"""

from __future__ import annotations

import os
import re
import csv
import json
import glob
import time
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# ====== 你的模块（按你工程路径）======
from swift.tvl.stage1.src.data.touch_fea_dataset import ProcessedTouchTextFeatureDataset
from swift.tvl.stage1.src.models.unified_touch import UnifiedTouchTextAE
from swift.tvl.stage1.src.preprocess.tvl_touch_encoder import (
    TVLTouchPreprocessConfig,
    TVLTouchPreprocessor,
)

from transformers import (
    AutoTokenizer,
    AddedToken,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeProcessor,
)

os.environ["HF_HUB_OFFLINE"] = "1"


# =========================
# 默认路径（按你提供的）
# =========================

DEFAULT_SSVTP_DIR = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/dataset/TVL/Touch-Vision-Language-Dataset/tvl_dataset/ssvtp"
DEFAULT_TEST_CSV = os.path.join(DEFAULT_SSVTP_DIR, "test.csv")
DEFAULT_PREFIX_TXT = os.path.join(DEFAULT_SSVTP_DIR, "text_prefix.txt")

DEFAULT_TVL_REPO_PATH = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/TVL/tvl"
DEFAULT_TACTILE_ENCODER_CKPT = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/dataset/TVL/checkpoints/tvl_enc_vitb.pth"
DEFAULT_TACTILE_MODEL = "vit_base_patch16_224"

DEFAULT_AE_CKPT_PATH = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/swift/checkpoints/tvl/stage1/1/best.pt"
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

DEFAULT_OUT_DIR = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/tvl_ssvtp_test_eval_outputs"

# text feature spec（与你训练 feature 一致）
TEXT_MAX_LEN = 24
TEXT_HIDDEN = 2048

# touch feature spec（与你训练 feature 一致）
TOUCH_NUM_TOKENS = 197
TOUCH_HIDDEN = 768

TOUCH_PLACEHOLDER = "<touch>"

SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group.\n\n"
    "Task setting:\n"
    "- You will answer questions using tactile/touch context.\n"
    "- The user may include multimodal tags like <image>, but you do NOT receive the image.\n"
    "- A section named 'TOUCH_EMBEDDING' may appear in the user message. The tokens in that section are placeholders.\n"
    "  Their embeddings are injected at inference time to carry tactile information.\n\n"
    "Instructions:\n"
    "- Use the 'TOUCH_EMBEDDING' section as the primary context.\n"
    "- Answer the QUESTION concisely.\n"
    "- Output only the final answer text.\n"
)


# =========================
# utils
# =========================

def ensure_dir(p: str) -> None:
    if p:
        os.makedirs(p, exist_ok=True)


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_ckpt_file(path: str) -> str:
    """支持给目录：优先 best*.pt，否则最新 .pt。"""
    path = os.path.abspath(path)
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise FileNotFoundError(f"AE_CKPT_PATH not found: {path}")

    pts = sorted(glob.glob(os.path.join(path, "*.pt")))
    if not pts:
        raise FileNotFoundError(f"No .pt files found under ckpt dir: {path}")

    best = [p for p in pts if os.path.basename(p).lower().startswith("best")]
    if best:
        return best[0]

    pts.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return pts[0]


def normalize_id_from_filename(rgb_rel: str) -> str:
    """
    例如 images_rgb/image_437_rgb.jpg -> "000000000437"
    如果解析不到数字，则用 basename 去扩展名。
    """
    base = os.path.basename(rgb_rel)
    m = re.search(r"image_(\d+)_rgb", base)
    if m:
        n = m.group(1)
        return str(int(n)).zfill(12)
    # fallback
    stem = os.path.splitext(base)[0]
    return stem


def rgb_rel_to_abs(ssvtp_dir: str, rgb_rel: str) -> str:
    p = rgb_rel.strip().lstrip("/")
    return os.path.join(ssvtp_dir, p)


def tactile_rel_from_rgb_rel(rgb_rel: str) -> str:
    """
    SSVTP 规则（与你示例一致）：
      images_rgb/image_106_rgb.jpg -> images_tac/image_106_tac.jpg
    """
    p = rgb_rel.strip().lstrip("/")
    p = p.replace("images_rgb/", "images_tac/")
    p = p.replace("_rgb.", "_tac.")
    p = p.replace("_rgb", "_tac")
    return p


def clean_human_to_question(human_raw: str) -> str:
    t = "" if human_raw is None else str(human_raw)
    for tag in ["<image>", "<tactile>", "<touch>"]:
        t = t.replace(tag, " ")
    t = "\n".join([" ".join(line.split()) for line in t.splitlines() if " ".join(line.split())])
    return t.strip()


def read_prefixes(prefix_txt: str) -> List[str]:
    if not os.path.exists(prefix_txt):
        return ["This image gives tactile feelings of"]
    out = []
    with open(prefix_txt, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(s)
    return out or ["This image gives tactile feelings of"]


def make_question(prefix: str) -> str:
    p = (prefix or "").strip()
    if not p:
        p = "This image gives tactile feelings of"
    # 统一成问句
    if not p.endswith("?"):
        p = p.rstrip() + "?"
    return p


def build_user_prompt_with_touch(question_text: str, k: int, placeholder: str = TOUCH_PLACEHOLDER) -> str:
    k = max(1, int(k))
    q = (question_text or "").strip()
    if q == "":
        q = "This image gives tactile feelings of?"
    block = " ".join([placeholder] * k)
    return (
        "TOUCH_EMBEDDING:\n"
        f"{block}\n\n"
        "QUESTION:\n"
        f"{q}\n\n"
        "ANSWER:"
    )


def resolve_safe_pad_token_id(tokenizer, model=None) -> int:
    eos = getattr(tokenizer, "eos_token_id", None)
    pad = getattr(tokenizer, "pad_token_id", None)

    if pad is None:
        try:
            cand = tokenizer.convert_tokens_to_ids("<|endoftext|>")
            if isinstance(cand, int) and cand >= 0:
                pad = cand
        except Exception:
            pad = None

    if pad is None and model is not None:
        pad = getattr(getattr(model, "generation_config", None), "pad_token_id", None)

    if pad is None:
        pad = 0 if eos != 0 else 1

    if eos is not None and pad == eos:
        pad = 0 if eos != 0 else 1

    return int(pad)


# =========================
# 1) 组织 meta json（含 conversations）
# =========================

def sniff_csv_delimiter(path: str) -> str:
    with open(path, "r", encoding="utf-8", newline="") as f:
        head = f.readline()
    return "\t" if "\t" in head else ","


def build_ssvtp_test_meta_json(
    ssvtp_dir: str,
    test_csv: str,
    prefix_txt: str,
    out_json: str,
    seed: int,
    prompt_mode: str = "random_prefix",  # "random_prefix" | "fixed"
    fixed_prompt: str = "This image gives tactile feelings of?",
    overwrite: bool = True,
) -> str:
    ensure_dir(os.path.dirname(out_json))
    if (not overwrite) and os.path.exists(out_json):
        return out_json

    prefixes = read_prefixes(prefix_txt)
    import random
    rng = random.Random(seed)

    delim = sniff_csv_delimiter(test_csv)
    items: List[Dict[str, Any]] = []

    with open(test_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        # 期望字段：url, caption
        for row in reader:
            rgb_rel = (row.get("url") or "").strip()
            caption = (row.get("caption") or "").strip()
            if not rgb_rel or not caption:
                continue

            sid = normalize_id_from_filename(rgb_rel)

            rgb_abs = rgb_rel_to_abs(ssvtp_dir, rgb_rel)
            tac_rel = tactile_rel_from_rgb_rel(rgb_rel)
            tac_abs = rgb_rel_to_abs(ssvtp_dir, tac_rel)

            if prompt_mode == "fixed":
                q = fixed_prompt.strip()
                if not q.endswith("?"):
                    q = q + "?"
            else:
                q = make_question(rng.choice(prefixes))

            items.append(
                {
                    "id": sid,
                    "image": rgb_abs,
                    "tactile": tac_abs,
                    "caption": caption,  # 给 extractor 用；也作为 GT
                    "conversations": [
                        {"from": "human", "value": "<image>\n" + q},
                        {"from": "gpt", "value": caption},
                    ],
                }
            )

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"[INFO] meta json saved: {out_json} (num_items={len(items)})")
    return out_json


def load_meta_json_map(meta_json: str) -> Dict[str, Dict[str, Any]]:
    with open(meta_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for it in data:
        sid = str(it.get("id", "")).strip()
        if sid:
            out[sid] = it
    return out


def extract_first_round(conv_list: Any) -> Optional[Tuple[str, str]]:
    if not isinstance(conv_list, list):
        return None
    human = None
    gpt = None
    hi = None
    for i, msg in enumerate(conv_list):
        if isinstance(msg, dict) and msg.get("from") == "human":
            human = msg.get("value", "")
            hi = i
            break
    if hi is None:
        return None
    for j in range(hi + 1, len(conv_list)):
        msg = conv_list[j]
        if isinstance(msg, dict) and msg.get("from") == "gpt":
            gpt = msg.get("value", "")
            break
    if human is None or gpt is None:
        return None
    return str(human), str(gpt)


# =========================
# 2) 特征抽取（小测试集：单进程直接抽 + memmap 保存）
# =========================

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
    # strip module.
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
    timm ViT tactile encoder -> tokens (B,197,768) for vit_base_patch16_224
    """
    def __init__(
        self,
        tactile_model: str,
        checkpoint_path: str,
        device: torch.device,
        l2_normalize: bool = True,
    ):
        super().__init__()
        try:
            import timm
        except Exception as e:
            raise ImportError("timm is required for tactile encoder. Please `pip install timm`.") from e

        self.device = device
        self.l2_normalize = bool(l2_normalize)
        self.model = timm.create_model(
            tactile_model,
            pretrained=False,
            num_classes=0,
            global_pool="",
        )
        from torch.serialization import add_safe_globals
        add_safe_globals([argparse.Namespace])
        if checkpoint_path and os.path.isfile(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            sd = _extract_state_dict_from_ckpt(ckpt)
            sd = _select_tactile_state_dict(sd)
            # drop head
            drop_prefix = ("head.", "fc.", "classifier.")
            sd = {k: v for k, v in sd.items() if not k.startswith(drop_prefix)}

            # norm / fc_norm compatibility
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
            if missing or unexpected:
                print(f"[WARN] tactile encoder load_state_dict strict=False: missing={len(missing)} unexpected={len(unexpected)}")

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.model.to(device)

    @torch.inference_mode()
    def forward(self, tactile: torch.Tensor) -> torch.Tensor:
        # tactile: (B,3,224,224)
        tactile = tactile.to(self.device, non_blocking=True).float()
        if not hasattr(self.model, "forward_features"):
            raise RuntimeError("timm model has no forward_features()")
        tokens = self.model.forward_features(tactile)  # (B,N,D) in most timm ViTs
        if tokens.dim() != 3:
            raise RuntimeError(f"Unexpected tokens shape: {tuple(tokens.shape)}")
        if self.l2_normalize:
            tokens = tokens / tokens.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return tokens


@dataclass
class FeaturePaths:
    dataset_info_yaml: str
    touch_tokens_mmap: str
    text_embeds_mmap: str
    text_mask_mmap: str
    sample_ids_mmap: str
    global_indices_mmap: str
    valid_mmap: str


def make_feature_paths(out_dir: str) -> FeaturePaths:
    shard_dir = os.path.join(out_dir, "shards")
    ensure_dir(shard_dir)
    return FeaturePaths(
        dataset_info_yaml=os.path.join(out_dir, "dataset_info.yaml"),
        touch_tokens_mmap=os.path.join(shard_dir, "touch_tokens_rank00.mmap"),
        text_embeds_mmap=os.path.join(shard_dir, "text_embeds_rank00.mmap"),
        text_mask_mmap=os.path.join(shard_dir, "text_mask_rank00.mmap"),
        sample_ids_mmap=os.path.join(shard_dir, "sample_ids_rank00.mmap"),
        global_indices_mmap=os.path.join(shard_dir, "global_indices_rank00.mmap"),
        valid_mmap=os.path.join(shard_dir, "valid_rank00.mmap"),
    )


def extract_features_for_meta_json(
    meta_json: str,
    out_dir: str,
    tvl_repo_path: str,
    tactile_model: str,
    tactile_ckpt: str,
    qwen_tokenizer_name_or_path: str,
    overwrite: bool = True,
) -> str:
    """
    生成 memmap + dataset_info.yaml（rank0 单 shard）
    text_embeds 默认写全0；text_mask 用 tokenizer(caption) 得到 attention_mask。
    """
    ensure_dir(out_dir)
    paths = make_feature_paths(out_dir)

    if (not overwrite) and os.path.exists(paths.dataset_info_yaml):
        print(f"[INFO] features already exist, skip extract: {paths.dataset_info_yaml}")
        return paths.dataset_info_yaml

    with open(meta_json, "r", encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list) or len(items) == 0:
        raise RuntimeError(f"meta_json is empty or not list: {meta_json}")

    n = len(items)
    print(f"[INFO] extracting features for {n} test samples...")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # preprocessor
    pre_cfg = TVLTouchPreprocessConfig(
        tvl_repo_path=tvl_repo_path,
        crop_tacvis=False,
        subtract_background=None,
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
    )

    # tokenizer（只用来做 mask）
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

    # fill text_embeds zeros
    text_mm[:] = 0

    t0 = time.time()
    ok = 0
    for i, it in enumerate(items):
        sid = str(it.get("id", "")).strip()
        tactile_path = str(it.get("tactile", "")).strip()
        caption = str(it.get("caption", "")).strip()

        gidx_mm[i] = i
        sid_mm[i] = np.bytes_(sid.encode("utf-8")[:64])

        # mask from caption tokenization
        enc = tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=TEXT_MAX_LEN,
            add_special_tokens=False,
            return_tensors="pt",
        )
        attn = enc["attention_mask"][0].to(torch.uint8).cpu().numpy()  # (L,)
        mask_mm[i] = attn

        try:
            if (not tactile_path) or (not os.path.isfile(tactile_path)):
                raise FileNotFoundError(f"tactile file missing: {tactile_path}")

            dataset_hint = pre.infer_dataset_hint(tactile_path, "")
            tactile = pre.load_tactile(
                tactile_path=tactile_path,
                dataset_hint=dataset_hint,
                tactile_background_path=None,
            )  # (3,224,224)
            if not isinstance(tactile, torch.Tensor) or tactile.dim() != 3:
                raise RuntimeError(f"bad tactile tensor: {type(tactile)} shape={getattr(tactile,'shape',None)}")

            with torch.inference_mode():
                tokens = tac_enc(tactile.unsqueeze(0))  # (1,197,768)
            tokens = tokens[0].detach().cpu().to(torch.float16).numpy()
            if tokens.shape != (TOUCH_NUM_TOKENS, TOUCH_HIDDEN):
                raise RuntimeError(f"unexpected touch token shape: {tokens.shape}")

            touch_mm[i] = tokens
            valid_mm[i] = 1
            ok += 1

        except Exception as e:
            print(f"[WARN] feature extract failed at i={i} id={sid}: {repr(e)}")
            touch_mm[i] = 0
            valid_mm[i] = 0

        if (i + 1) % 10 == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            print(f"[INFO] progress {i+1}/{n} ok={ok} elapsed={elapsed:.1f}s")

    # flush
    touch_mm.flush(); text_mm.flush(); mask_mm.flush()
    sid_mm.flush(); gidx_mm.flush(); valid_mm.flush()

    # write dataset_info.yaml (single shard rank0)
    dataset_info = {
        "version": 1,
        "num_samples_total": int(n),
        "world_size": 1,
        "split_mode": "contiguous",
        "dataset": {
            "meta_paths": [meta_json],
            "touch_key_candidates": ["tactile"],
            "touch_bg_key_candidates": [],
            "text_key_candidates": ["caption", "text", "description"],
            "gpt_text_strategy": "first",
        },
        "features": {
            "touch": {"num_tokens": TOUCH_NUM_TOKENS, "hidden": TOUCH_HIDDEN, "dtype": "float16"},
            "text": {"max_len": TEXT_MAX_LEN, "hidden": TEXT_HIDDEN, "dtype": "float16"},
        },
        "pairs": ["touch-text", "text-touch"],
        "shards": [
            {
                "rank": 0,
                "num_samples": int(n),
                "touch": {"num_tokens": TOUCH_NUM_TOKENS, "hidden": TOUCH_HIDDEN, "dtype": "float16"},
                "text": {"max_len": TEXT_MAX_LEN, "hidden": TEXT_HIDDEN, "dtype": "float16"},
                "touch_preprocess": {
                    "tvl_repo_path": tvl_repo_path,
                    "crop_tacvis": False,
                    "subtract_background": None,
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
        ],
    }

    import yaml
    with open(paths.dataset_info_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_info, f, sort_keys=False)

    print(f"[INFO] dataset_info.yaml saved: {paths.dataset_info_yaml}")
    return paths.dataset_info_yaml


# =========================
# 3) inference：AE -> Qwen 注入
# =========================

def load_ae_from_ckpt(ckpt_path: str, device: torch.device, dtype: torch.dtype) -> Tuple[UnifiedTouchTextAE, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", None)
    if cfg is None:
        raise RuntimeError(f"Checkpoint missing 'cfg': {ckpt_path}")
    model_cfg = cfg.get("model", None)
    if model_cfg is None:
        raise RuntimeError(f"Checkpoint cfg missing 'model': {ckpt_path}")
    ae = UnifiedTouchTextAE(model_cfg)
    ae.load_state_dict(ckpt["model"], strict=True)
    ae.eval()
    ae.to(device=device, dtype=dtype)
    return ae, cfg


@torch.no_grad()
def ae_touch_to_text_token_embeddings(
    ae: UnifiedTouchTextAE,
    touch_tokens: torch.Tensor,     # (G,D)
    text_shape_like: torch.Tensor,  # (L,H)
    text_mask: torch.Tensor,        # (L,) bool
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    返回 pred_text_tokens: (L,H), mask: (L,)
    关键：text_feat 用全0，避免泄漏。
    """
    L = int(text_shape_like.shape[0])
    H = int(text_shape_like.shape[1])
    te = torch.zeros((1, L, H), device=device, dtype=dtype)
    tm = text_mask.unsqueeze(0).to(device=device)

    tt = touch_tokens.unsqueeze(0).to(device=device, dtype=dtype)

    out = None
    last_err = None
    for touch_key in ["touch_feat", "touch_tokens", "touch", "tactile_feat", "tactile_tokens", "tactile", "sensor_feat", "point_feat"]:
        try:
            out = ae(**{touch_key: tt, "text_feat": te, "text_mask": tm})
            break
        except TypeError as e:
            last_err = e
            continue
    if out is None or (not isinstance(out, dict)):
        raise RuntimeError(f"AE forward failed. last_err={repr(last_err)}")

    if "text_recon_from_touch" in out:
        pred = out["text_recon_from_touch"][0]
    elif "text_recon_from_tactile" in out:
        pred = out["text_recon_from_tactile"][0]
    else:
        cand = [k for k in out.keys() if str(k).startswith("text_recon")]
        if not cand:
            raise RuntimeError(f"AE output missing text recon key. out.keys()={list(out.keys())}")
        pred = out[cand[0]][0]
    return pred, text_mask.to(device=device)


def build_qwen_inputs(processor: Qwen3OmniMoeProcessor, user_text: str) -> Dict[str, torch.Tensor]:
    conversations = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]
    inputs = processor.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    return inputs


@torch.no_grad()
def inject_embeddings_into_inputs_embeds(
    *,
    tokenizer,
    input_ids: torch.Tensor,        # (1,S)
    inputs_embeds: torch.Tensor,    # (1,S,H)
    payload: torch.Tensor,          # (H,) or (K,H)
    placeholder: str,
) -> Tuple[torch.Tensor, List[int]]:
    new_embeds = inputs_embeds.clone()
    H = new_embeds.shape[-1]

    if payload.dim() == 1:
        K = None
        if payload.numel() != H:
            raise RuntimeError(f"payload dim mismatch: {payload.numel()} vs H={H}")
    elif payload.dim() == 2:
        K = int(payload.shape[0])
        if payload.shape[1] != H:
            raise RuntimeError(f"payload dim mismatch: {payload.shape[1]} vs H={H}")
    else:
        raise RuntimeError(f"payload must be (H,) or (K,H), got {tuple(payload.shape)}")

    pid = tokenizer.convert_tokens_to_ids(placeholder)
    if pid is None or int(pid) < 0:
        raise RuntimeError(f"Placeholder token '{placeholder}' not in vocab. Did you add_special_tokens?")
    pid = int(pid)

    pos = torch.where(input_ids[0] == pid)[0].tolist()
    if len(pos) == 0:
        raise RuntimeError(f"Cannot find placeholder '{placeholder}' in input_ids. Prompt may not contain it.")

    if K is None:
        vec = payload.to(device=new_embeds.device, dtype=new_embeds.dtype).view(1, 1, H)
        for p in pos:
            new_embeds[:, p:p+1, :] = vec
        return new_embeds, pos

    if len(pos) < K:
        raise RuntimeError(f"Need K={K} placeholders, but only found {len(pos)} in prompt.")

    for i in range(K):
        p = int(pos[i])
        vec = payload[i].to(device=new_embeds.device, dtype=new_embeds.dtype).view(1, 1, H)
        new_embeds[:, p:p+1, :] = vec

    return new_embeds, pos[:K]


@torch.no_grad()
def generate_with_inputs_embeds(
    *,
    model: Qwen3OmniMoeThinkerForConditionalGeneration,
    processor: Qwen3OmniMoeProcessor,
    inputs: Dict[str, torch.Tensor],
    inputs_embeds: torch.Tensor,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    tokenizer = processor.tokenizer
    gen_kwargs = {k: v for k, v in inputs.items() if k != "input_ids"}
    gen_kwargs["inputs_embeds"] = inputs_embeds

    gen_kwargs["pad_token_id"] = resolve_safe_pad_token_id(tokenizer, model)
    if getattr(tokenizer, "eos_token_id", None) is not None:
        gen_kwargs["eos_token_id"] = tokenizer.eos_token_id

    extra = {}
    if do_sample:
        extra["temperature"] = float(temperature)
        extra["top_p"] = float(top_p)

    out = model.generate(
        **gen_kwargs,
        max_new_tokens=int(max_new_tokens),
        do_sample=bool(do_sample),
        **extra,
    )
    prompt_len = inputs_embeds.shape[1]
    seq = out[0]
    new_tokens = seq[prompt_len:] if seq.shape[0] > prompt_len else seq
    return tokenizer.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()


def run_inference_on_features(
    dataset_info_yaml: str,
    meta_json: str,
    ae_ckpt_path: str,
    qwen_model_name_or_path: str,
    inject_mode: str = "sequence",
    max_touch_tokens: int = 24,          # 建议 <= TEXT_MAX_LEN
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    print_prompt: bool = False,
    save_results_json: Optional[str] = None,
) -> None:
    # load conv map
    meta_map = load_meta_json_map(meta_json)

    # load Qwen
    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        qwen_model_name_or_path,
        dtype="auto",
        device_map="auto",
    )
    model.eval()
    processor = Qwen3OmniMoeProcessor.from_pretrained(qwen_model_name_or_path)
    tokenizer = processor.tokenizer

    # register <touch> as single token
    old_ids = tokenizer.encode(TOUCH_PLACEHOLDER, add_special_tokens=False)
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": [AddedToken(TOUCH_PLACEHOLDER, lstrip=False, rstrip=False)]}
    )
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        # init (optional)
        try:
            new_id = int(tokenizer.convert_tokens_to_ids(TOUCH_PLACEHOLDER))
            emb = model.get_input_embeddings()
            if isinstance(old_ids, list) and len(old_ids) > 0:
                ids_t = torch.tensor(old_ids, device=emb.weight.device, dtype=torch.long)
                with torch.no_grad():
                    init_vec = emb.weight.data.index_select(0, ids_t).mean(dim=0)
                    emb.weight.data[new_id].copy_(init_vec)
        except Exception:
            pass

    pid = tokenizer.convert_tokens_to_ids(TOUCH_PLACEHOLDER)
    print(f"[INFO] registered '{TOUCH_PLACEHOLDER}' as single token id={pid}, added={num_added}, vocab={len(tokenizer)}")

    emb_layer = model.get_input_embeddings()
    emb_device = emb_layer.weight.device
    emb_dtype = emb_layer.weight.dtype
    llm_hidden = int(emb_layer.weight.shape[1])
    print(f"[INFO] Qwen emb device={emb_device}, dtype={emb_dtype}, hidden={llm_hidden}")

    # load AE
    ckpt_file = resolve_ckpt_file(ae_ckpt_path)
    print(f"[INFO] AE ckpt file = {ckpt_file}")
    ae, ae_cfg = load_ae_from_ckpt(ckpt_file, device=emb_device, dtype=emb_dtype)

    try:
        ae_d_text_in = int(ae_cfg["model"].get("d_text_in", -1))
        if ae_d_text_in != -1 and ae_d_text_in != llm_hidden:
            print(f"[WARN] AE d_text_in={ae_d_text_in} != Qwen hidden={llm_hidden} -> 注入将维度不匹配（需要 projection）")
    except Exception:
        pass

    # load feature dataset
    ds = ProcessedTouchTextFeatureDataset(dataset_info_yaml, require_valid=True, return_ids=True)
    print(f"[INFO] feature dataset loaded, len={len(ds)}  yaml={dataset_info_yaml}")

    results: List[Dict[str, Any]] = []

    for i in range(len(ds)):
        sample = ds[i]
        sid = str(sample.get("sample_id", "")).strip()
        gidx = sample.get("global_index", None)

        meta = meta_map.get(sid, None)
        if meta is None:
            # 如果 id 对不上（很少见），直接跳过
            print(f"[WARN] cannot find meta for sample_id={sid}, skip.")
            continue

        # gt & question
        pair = extract_first_round(meta.get("conversations", []))
        if pair is None:
            human_raw = ""
            gt = str(meta.get("caption", ""))
        else:
            human_raw, gt = pair

        question = clean_human_to_question(human_raw)

        # AE: touch -> pred text tokens
        pred_tokens, mask = ae_touch_to_text_token_embeddings(
            ae=ae,
            touch_tokens=sample["touch"],
            text_shape_like=sample["text"],  # (L,H) 全0也行，只要 shape 对
            text_mask=sample["mask"],
            device=emb_device,
            dtype=emb_dtype,
        )

        # build payload
        if inject_mode == "sequence":
            seq = pred_tokens[mask] if mask.any() else pred_tokens[:1]
            if seq.shape[0] > max_touch_tokens:
                seq = seq[:max_touch_tokens]
            K = int(seq.shape[0])
            payload = seq
        elif inject_mode == "pooled":
            # mean pool
            m = mask.to(pred_tokens.device).to(pred_tokens.dtype)
            denom = m.sum().clamp_min(1.0)
            vec = (pred_tokens * m.unsqueeze(-1)).sum(dim=0) / denom
            K = 1
            payload = vec
        else:
            raise ValueError(f"Unknown inject_mode: {inject_mode}")

        user_text = build_user_prompt_with_touch(question, K, TOUCH_PLACEHOLDER)
        inputs = build_qwen_inputs(processor, user_text)
        inputs = {k: (v.to(emb_device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            base_embeds = emb_layer(input_ids)

        injected_embeds, pos = inject_embeddings_into_inputs_embeds(
            tokenizer=tokenizer,
            input_ids=input_ids,
            inputs_embeds=base_embeds,
            payload=payload,
            placeholder=TOUCH_PLACEHOLDER,
        )

        pred = generate_with_inputs_embeds(
            model=model,
            processor=processor,
            inputs=inputs,
            inputs_embeds=injected_embeds,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

        print("\n" + "=" * 120)
        print(f"[{i}] sample_id={sid} global_index={gidx} K_injected={K} pos(first3)={pos[:3]}")
        print("- QUESTION -")
        print(question)
        if print_prompt:
            print("- PROMPT USED -")
            print(user_text)
        print("- PRED -")
        print(pred)
        print("- GT (caption) -")
        print(gt)

        results.append(
            {
                "sample_id": sid,
                "global_index": gidx,
                "question": question,
                "pred": pred,
                "gt": gt,
                "K_injected": K,
            }
        )

    if save_results_json:
        ensure_dir(os.path.dirname(save_results_json))
        with open(save_results_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] saved results json: {save_results_json}")


# =========================
# main
# =========================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssvtp_dir", type=str, default=DEFAULT_SSVTP_DIR)
    ap.add_argument("--test_csv", type=str, default=DEFAULT_TEST_CSV)
    ap.add_argument("--prefix_txt", type=str, default=DEFAULT_PREFIX_TXT)

    ap.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    ap.add_argument("--meta_json", type=str, default="")  # 默认 out_dir/ssvtp_test_meta.json
    ap.add_argument("--features_dir", type=str, default="")  # 默认 out_dir/ssvtp_test_features

    ap.add_argument("--tvl_repo_path", type=str, default=DEFAULT_TVL_REPO_PATH)
    ap.add_argument("--tactile_model", type=str, default=DEFAULT_TACTILE_MODEL)
    ap.add_argument("--tactile_ckpt", type=str, default=DEFAULT_TACTILE_ENCODER_CKPT)
    ap.add_argument("--qwen_tokenizer", type=str, default=DEFAULT_QWEN_MODEL)

    ap.add_argument("--ae_ckpt", type=str, default=DEFAULT_AE_CKPT_PATH)
    ap.add_argument("--qwen_model", type=str, default=DEFAULT_QWEN_MODEL)

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--prompt_mode", type=str, default="random_prefix", choices=["random_prefix", "fixed"])
    ap.add_argument("--fixed_prompt", type=str, default="This image gives tactile feelings of?")

    ap.add_argument("--overwrite_json", action="store_true")
    ap.add_argument("--overwrite_features", action="store_true")

    ap.add_argument("--inject_mode", type=str, default="sequence", choices=["sequence", "pooled"])
    ap.add_argument("--max_touch_tokens", type=int, default=24)  # <= 24 最合理
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--print_prompt", action="store_true")

    ap.add_argument("--save_results_json", type=str, default="")
    ap.add_argument("--skip_extract", action="store_true")
    ap.add_argument("--skip_infer", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    ensure_dir(args.out_dir)

    meta_json = args.meta_json.strip() or os.path.join(args.out_dir, "ssvtp_test_meta.json")
    features_dir = args.features_dir.strip() or os.path.join(args.out_dir, "ssvtp_test_features")

    # 1) build meta json
    build_ssvtp_test_meta_json(
        ssvtp_dir=args.ssvtp_dir,
        test_csv=args.test_csv,
        prefix_txt=args.prefix_txt,
        out_json=meta_json,
        seed=args.seed,
        prompt_mode=args.prompt_mode,
        fixed_prompt=args.fixed_prompt,
        overwrite=bool(args.overwrite_json),
    )

    # 2) extract features
    dataset_info_yaml = os.path.join(features_dir, "dataset_info.yaml")
    if not args.skip_extract:
        extract_features_for_meta_json(
            meta_json=meta_json,
            out_dir=features_dir,
            tvl_repo_path=args.tvl_repo_path,
            tactile_model=args.tactile_model,
            tactile_ckpt=args.tactile_ckpt,
            qwen_tokenizer_name_or_path=args.qwen_tokenizer,
            overwrite=bool(args.overwrite_features),
        )
        # 释放一些显存/缓存（特别是如果你后面要 load Qwen 30B）
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    else:
        if not os.path.exists(dataset_info_yaml):
            raise FileNotFoundError(f"--skip_extract set but dataset_info.yaml not found: {dataset_info_yaml}")

    # 3) inference
    if not args.skip_infer:
        run_inference_on_features(
            dataset_info_yaml=dataset_info_yaml,
            meta_json=meta_json,
            ae_ckpt_path=args.ae_ckpt,
            qwen_model_name_or_path=args.qwen_model,
            inject_mode=args.inject_mode,
            max_touch_tokens=int(args.max_touch_tokens),
            max_new_tokens=int(args.max_new_tokens),
            do_sample=bool(args.do_sample),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            print_prompt=bool(args.print_prompt),
            save_results_json=(args.save_results_json.strip() or None),
        )


if __name__ == "__main__":
    main()