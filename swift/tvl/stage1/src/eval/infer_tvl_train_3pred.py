# -*- coding: utf-8 -*-
"""
infer_train_ssvtp_3pred.py

训练集 SSVTP subset inference：
- 读取已抽好的训练集 features: tvl_features_stage1/dataset_info.yaml（8 shards）
- 只选择 sample_id 以 "ssvtp-" 开头的样本做 inference
- 3 个 prediction：
  (1) baseline: 直接把 GT text embeddings(sample["text"]) 注入 Qwen3（用 human 问题做 prompt）
  (2) touch_plain: touch->AE->text token embeds 注入 Qwen3（用 human 问题做 prompt）
  (3) touch_vocab: touch->AE->inject + human 问题 + vocab 约束 prompt

用法（不带参数直接跑默认配置）：
python infer_train_ssvtp_3pred.py

可选参数：
python infer_train_ssvtp_3pred.py --max_samples 500 --seed 42
"""

from __future__ import annotations

import os
import csv
import json
import glob
import argparse
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from transformers import (
    AddedToken,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeProcessor,
)

# --------- torch safe globals ---------
from torch.serialization import add_safe_globals
add_safe_globals([argparse.Namespace])

# --------- OFFLINE ---------
os.environ["HF_HUB_OFFLINE"] = "1"

from swift.tvl.stage1.src.data.touch_fea_dataset import ProcessedTouchTextFeatureDataset
from swift.tvl.stage1.src.models.unified_touch import UnifiedTouchTextAE


# =========================================================
# ✅ 默认设置（你要的：写在文件开头，直接 python 即可运行）
# =========================================================

# 训练集 features
DEFAULT_FEATURE_YAML = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/tvl_features_stage1/dataset_info.yaml"

# 训练集 meta json（也可以从 feature_yaml 里读 meta_paths；默认会优先用 yaml 里的 meta_paths[0]）
DEFAULT_META_JSON_FALLBACK = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/dataset/TVL/Touch-Vision-Language-Dataset/tvl_dataset/finetune_merged.json"

# 输出目录
DEFAULT_OUT_DIR = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/tvl_features_stage1/infer_train_ssvtp_3pred"

# AE ckpt
DEFAULT_AE_CKPT = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/swift/checkpoints/tvl/stage1/1/best.pt"

# Qwen3
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

# 只做 ssvtp subset
SSVTP_PREFIX = "ssvtp-"

# 最大样本数：默认 200；设为 0 表示跑完全部 ssvtp 样本
DEFAULT_MAX_SAMPLES = 40
DEFAULT_SEED = 42

# 注入模式
DEFAULT_INJECT_MODE = "sequence"   # sequence | pooled
DEFAULT_MAX_TOUCH_TOKENS = 24      # <= text_max_len(24) 更安全
DEFAULT_MAX_NEW_TOKENS = 128

DEFAULT_DO_SAMPLE = False
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9

# 三种模式都跑
DEFAULT_RUN_BASELINE_TEXTEMB = True
DEFAULT_RUN_TOUCH_PLAIN = True
DEFAULT_RUN_TOUCH_VOCAB = True

# 词表路径
DEFAULT_VOCAB_PATH = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/dataset/TVL/Touch-Vision-Language-Dataset/tvl_dataset/vocab_merged/canon.words.txt"
VOCAB_MAX_WORDS = 4000
VOCAB_MAX_CHARS = 12000
VOCAB_WORDS_PER_LINE = 40

# placeholder
TOUCH_PLACEHOLDER = "<touch>"

SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group.\n\n"
    "Task setting:\n"
    "- You will answer questions using tactile/touch context.\n"
    "- A section named 'TOUCH_EMBEDDING' may appear. The tokens there are placeholders.\n"
    "  Their embeddings are injected at inference time to carry tactile information.\n\n"
    "Instructions:\n"
    "- Use the 'TOUCH_EMBEDDING' section as the primary context.\n"
    "- Answer the QUESTION concisely.\n"
    "- Output only the final answer text.\n"
)


# =========================================================
# utils
# =========================================================
def ensure_dir(p: str) -> None:
    if p:
        os.makedirs(p, exist_ok=True)


def resolve_ckpt_file(path: str) -> str:
    path = os.path.abspath(path)
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise FileNotFoundError(f"AE ckpt path not found: {path}")
    pts = sorted(glob.glob(os.path.join(path, "*.pt")))
    if not pts:
        raise FileNotFoundError(f"No .pt found in ckpt dir: {path}")
    best = [p for p in pts if os.path.basename(p).lower().startswith("best")]
    if best:
        return best[0]
    pts.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return pts[0]


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


def clean_question(human_raw: str) -> str:
    t = "" if human_raw is None else str(human_raw)
    for tag in ["<image>", "<tactile>", "<touch>"]:
        t = t.replace(tag, " ")
    t = "\n".join([" ".join(line.split()) for line in t.splitlines() if " ".join(line.split())])
    t = t.strip()
    # 保持训练集原问题风格；如果最后没有? 也不强行加（避免改语气）
    return t


# =========================================================
# vocab prompt helpers
# =========================================================
def load_vocab_words(path: str, max_words: int = 0) -> List[str]:
    if not path or (not os.path.exists(path)):
        raise FileNotFoundError(f"vocab file not found: {path}")
    words: List[str] = []
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if not w or w.startswith("#"):
                continue
            if w in seen:
                continue
            seen.add(w)
            words.append(w)
            if max_words and len(words) >= int(max_words):
                break
    return words


def format_vocab_block(words: List[str], words_per_line: int = 40, max_chars: int = 0) -> str:
    if not words:
        return ""
    lines: List[str] = []
    cur_len = 0
    step = max(1, int(words_per_line))
    for i in range(0, len(words), step):
        chunk = words[i:i + step]
        line = ", ".join(chunk)
        new_len = cur_len + len(line) + (1 if lines else 0)
        if max_chars and new_len > int(max_chars):
            break
        lines.append(line)
        cur_len = new_len
    return "\n".join(lines)


# =========================================================
# Prompt builders
# =========================================================
def build_user_prompt_plain(k: int, question: str) -> str:
    k = max(1, int(k))
    q = (question or "").strip() or "This image gives tactile feelings of?"
    block = " ".join([TOUCH_PLACEHOLDER] * k)
    return (
        "TOUCH_EMBEDDING:\n"
        f"{block}\n\n"
        "QUESTION:\n"
        f"{q}\n\n"
        "ANSWER:"
    )


def build_user_prompt_vocab(k: int, question: str, vocab_block: str) -> str:
    k = max(1, int(k))
    q = (question or "").strip() or "This image gives tactile feelings of?"
    block = " ".join([TOUCH_PLACEHOLDER] * k)
    return (
        "TOUCH_EMBEDDING:\n"
        f"{block}\n\n"
        "QUESTION:\n"
        f"{q}\n\n"
        "INSTRUCTION:\n"
        "- Answer by selecting ONLY words/phrases from the VOCABULARY below.\n"
        "- Output a comma-separated list of selected vocabulary entries.\n"
        "- Do NOT add any other words, explanations, or punctuation beyond commas.\n\n"
        "VOCABULARY:\n"
        f"{vocab_block}\n\n"
        "ANSWER:"
    )


# =========================================================
# Meta json streaming loader (only target ids)
# =========================================================
def load_first_round_for_ids(meta_json: str, target_ids: List[str]) -> Dict[str, Dict[str, str]]:
    """
    只加载 target_ids 的 human+gpt 第一轮。
    优先 ijson 流式，失败则退化 json.load（可能占内存）。
    """
    target_set = set(str(x) for x in target_ids if str(x).strip())
    out: Dict[str, Dict[str, str]] = {}
    if not target_set:
        return out

    try:
        import ijson  # type: ignore
        with open(meta_json, "rb") as f:
            for item in ijson.items(f, "item"):
                sid = str(item.get("id", "")).strip()
                if sid in target_set:
                    pair = extract_first_round(item.get("conversations", []))
                    if pair is not None:
                        human, gpt = pair
                        out[sid] = {"human": human, "gt": gpt}
                    else:
                        # fallback：没有 conv 就用空
                        out[sid] = {"human": "", "gt": str(item.get("caption", "")).strip()}
                    if len(out) >= len(target_set):
                        break
        return out
    except Exception:
        pass

    # fallback (可能比较大)
    with open(meta_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        sid = str(item.get("id", "")).strip()
        if sid in target_set:
            pair = extract_first_round(item.get("conversations", []))
            if pair is not None:
                human, gpt = pair
                out[sid] = {"human": human, "gt": gpt}
            else:
                out[sid] = {"human": "", "gt": str(item.get("caption", "")).strip()}
            if len(out) >= len(target_set):
                break
    return out


# =========================================================
# Selection (fast): scan only sample_ids + valid memmaps
# =========================================================
def _infer_1d_memmap_dtype(path: str, n: int) -> Any:
    size = os.path.getsize(path)
    if n <= 0:
        raise ValueError(f"num_samples must be > 0, got {n}")
    if size % n != 0:
        raise RuntimeError(f"Cannot infer dtype for {path}: file_size={size} not divisible by n={n}")
    rec_bytes = size // n
    if rec_bytes == 8:
        return np.int64
    if rec_bytes == 4:
        return np.int32
    if rec_bytes == 2:
        return np.int16
    if rec_bytes == 1:
        return np.uint8
    return f"S{rec_bytes}"


def _decode_memmap_id(raw: Any, dt: Any) -> str:
    if isinstance(dt, str) and dt.startswith("S"):
        # raw is np.void/bytes-like
        try:
            b = raw.tobytes()
        except Exception:
            b = bytes(raw)
        return b.split(b"\x00", 1)[0].decode("utf-8", errors="ignore").strip()
    # numeric
    try:
        return str(int(raw.item()))
    except Exception:
        return str(raw)


def select_ssvtp_samples_from_feature_yaml(
    feature_yaml: str,
    prefix: str,
    max_samples: int,
    seed: int,
    require_valid: bool = True,
) -> Tuple[List[Tuple[int, str]], int]:
    """
    返回：
      selected: List[(dataset_idx, sample_id)]
      total_matched: 满足 prefix 的总数（不受 max_samples 限制）
    采用 reservoir sampling，避免加载完整样本内容。
    """
    with open(feature_yaml, "r", encoding="utf-8") as f:
        info = yaml.safe_load(f)

    shards = list(info.get("shards", []))
    if not shards:
        raise RuntimeError(f"No shards in {feature_yaml}")

    # prefix sums for concat indexing
    shard_sizes = [int(s["num_samples"]) for s in shards]
    prefix_sum = [0]
    for n in shard_sizes:
        prefix_sum.append(prefix_sum[-1] + n)

    rng = random.Random(int(seed))
    selected: List[Tuple[int, str]] = []

    total_matched = 0
    seen = 0  # for reservoir sampling count

    for si, s in enumerate(shards):
        n = int(s["num_samples"])
        paths = s["paths"]
        sid_path = paths.get("sample_ids", None)
        valid_path = paths.get("valid", None)
        if sid_path is None or valid_path is None:
            continue
        if not os.path.exists(sid_path) or not os.path.exists(valid_path):
            continue

        sid_dt = _infer_1d_memmap_dtype(sid_path, n)
        sid_mm = np.memmap(sid_path, mode="r", dtype=sid_dt, shape=(n,))
        valid_mm = np.memmap(valid_path, mode="r", dtype=np.uint8, shape=(n,))

        base_idx = int(prefix_sum[si])

        for li in range(n):
            if require_valid and int(valid_mm[li]) == 0:
                continue
            sid = _decode_memmap_id(sid_mm[li], sid_dt)
            if not sid.startswith(prefix):
                continue

            total_matched += 1

            dataset_idx = base_idx + li
            seen += 1
            item = (dataset_idx, sid)

            if max_samples <= 0:
                selected.append(item)
                continue

            if len(selected) < int(max_samples):
                selected.append(item)
            else:
                j = rng.randrange(seen)
                if j < int(max_samples):
                    selected[j] = item

    return selected, total_matched


# =========================================================
# AE
# =========================================================
def load_ae_from_ckpt(ckpt_file: str, device: torch.device, dtype: torch.dtype):
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", None)
    if cfg is None:
        raise RuntimeError(f"Checkpoint missing 'cfg': {ckpt_file}")
    model_cfg = cfg.get("model", None)
    if model_cfg is None:
        raise RuntimeError(f"Checkpoint cfg missing 'model': {ckpt_file}")

    ae = UnifiedTouchTextAE(model_cfg)
    ae.load_state_dict(ckpt["model"], strict=True)
    ae.eval()
    ae.to(device=device, dtype=dtype)
    return ae, cfg


@torch.no_grad()
def ae_touch_to_text_token_embeddings(
    ae: UnifiedTouchTextAE,
    touch_tokens: torch.Tensor,     # (197,768)
    text_shape_like: torch.Tensor,  # (24,2048) - only for shape
    text_mask: torch.Tensor,        # (24,)
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    touch -> AE -> text token embeddings (L,H)
    关键：text_feat 用全 0，避免泄漏
    """
    L = int(text_shape_like.shape[0])
    H = int(text_shape_like.shape[1])

    te = torch.zeros((1, L, H), device=device, dtype=dtype)
    tm = text_mask.unsqueeze(0).to(device=device)
    tt = touch_tokens.unsqueeze(0).to(device=device, dtype=dtype)

    out = None
    last_err = None
    for touch_key in [
        "touch_feat", "touch_tokens", "touch",
        "tactile_feat", "tactile_tokens", "tactile",
        "sensor_feat", "point_feat",
    ]:
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
            raise RuntimeError(f"AE output missing recon key. out.keys()={list(out.keys())}")
        pred = out[cand[0]][0]

    return pred, text_mask.to(device=device)


# =========================================================
# Qwen inject/generate
# =========================================================
def build_qwen_inputs(processor: Qwen3OmniMoeProcessor, user_text: str) -> Dict[str, torch.Tensor]:
    conversations = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]
    return processor.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )


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
        raise RuntimeError(f"Placeholder '{placeholder}' not in vocab. Did you add_special_tokens?")
    pid = int(pid)

    pos = torch.where(input_ids[0] == pid)[0].tolist()
    if len(pos) == 0:
        raise RuntimeError(f"Cannot find placeholder '{placeholder}' in input_ids.")

    if K is None:
        vec = payload.to(device=new_embeds.device, dtype=new_embeds.dtype).view(1, 1, H)
        for p in pos:
            new_embeds[:, p:p+1, :] = vec
        return new_embeds, pos

    if len(pos) < K:
        raise RuntimeError(f"Need K={K} placeholders, but only found {len(pos)}.")

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


@torch.no_grad()
def predict_once(
    *,
    model,
    processor,
    tokenizer,
    emb_layer,
    emb_device: torch.device,
    user_text: str,
    payload: torch.Tensor,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    inputs = build_qwen_inputs(processor, user_text)
    inputs = {k: (v.to(emb_device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]

    base_embeds = emb_layer(input_ids)

    injected_embeds, _ = inject_embeddings_into_inputs_embeds(
        tokenizer=tokenizer,
        input_ids=input_ids,
        inputs_embeds=base_embeds,
        payload=payload,
        placeholder=TOUCH_PLACEHOLDER,
    )

    return generate_with_inputs_embeds(
        model=model,
        processor=processor,
        inputs=inputs,
        inputs_embeds=injected_embeds,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )


# =========================================================
# save outputs
# =========================================================
def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    cols = [
        "sample_id",
        "dataset_idx",
        "global_index",
        "question",
        "gt",
        "pred_textemb_plain",
        "pred_touch_plain",
        "pred_touch_vocab",
        "K_textemb",
        "K_touch",
        "status_textemb",
        "status_touch_plain",
        "status_touch_vocab",
        "error_textemb",
        "error_touch_plain",
        "error_touch_vocab",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


# =========================================================
# main
# =========================================================
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--feature_yaml", type=str, default=DEFAULT_FEATURE_YAML)
    ap.add_argument("--meta_json", type=str, default="", help="optional override; default uses feature_yaml.dataset.meta_paths[0]")
    ap.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)

    ap.add_argument("--max_samples", type=int, default=DEFAULT_MAX_SAMPLES, help="0 means all matched samples")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)

    ap.add_argument("--ae_ckpt", type=str, default=DEFAULT_AE_CKPT)
    ap.add_argument("--qwen_model", type=str, default=DEFAULT_QWEN_MODEL)

    ap.add_argument("--inject_mode", type=str, default=DEFAULT_INJECT_MODE, choices=["sequence", "pooled"])
    ap.add_argument("--max_touch_tokens", type=int, default=DEFAULT_MAX_TOUCH_TOKENS)
    ap.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)

    ap.add_argument("--do_sample", action="store_true", default=DEFAULT_DO_SAMPLE)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)

    ap.add_argument("--run_baseline_textemb", action="store_true", default=DEFAULT_RUN_BASELINE_TEXTEMB)
    ap.add_argument("--run_touch_plain", action="store_true", default=DEFAULT_RUN_TOUCH_PLAIN)
    ap.add_argument("--run_touch_vocab", action="store_true", default=DEFAULT_RUN_TOUCH_VOCAB)

    ap.add_argument("--vocab_path", type=str, default=DEFAULT_VOCAB_PATH)

    return ap.parse_args()


def main():
    torch.set_grad_enabled(False)

    args = parse_args()
    ensure_dir(args.out_dir)

    # ---- resolve meta json (prefer yaml meta_paths[0]) ----
    with open(args.feature_yaml, "r", encoding="utf-8") as f:
        finfo = yaml.safe_load(f)
    meta_paths = finfo.get("dataset", {}).get("meta_paths", []) or []
    meta_json = args.meta_json.strip() if args.meta_json.strip() else (meta_paths[0] if meta_paths else DEFAULT_META_JSON_FALLBACK)
    if not os.path.exists(meta_json):
        raise FileNotFoundError(f"meta_json not found: {meta_json}")

    # ---- load vocab once ----
    vocab_words = load_vocab_words(args.vocab_path, max_words=int(VOCAB_MAX_WORDS) if VOCAB_MAX_WORDS else 0)
    vocab_block = format_vocab_block(
        vocab_words,
        words_per_line=int(VOCAB_WORDS_PER_LINE),
        max_chars=int(VOCAB_MAX_CHARS) if VOCAB_MAX_CHARS else 0,
    )
    print(f"[INFO] vocab loaded: {len(vocab_words)} words, vocab_block_chars={len(vocab_block)}")

    # ---- select ssvtp samples fast ----
    selected, total_matched = select_ssvtp_samples_from_feature_yaml(
        feature_yaml=args.feature_yaml,
        prefix=SSVTP_PREFIX,
        max_samples=int(args.max_samples),
        seed=int(args.seed),
        require_valid=True,
    )
    selected.sort(key=lambda x: x[0])  # sort by dataset_idx for nicer IO
    sel_ids = [sid for _, sid in selected]

    print(f"[INFO] meta_json = {meta_json}")
    print(f"[INFO] total matched ssvtp samples (valid=1): {total_matched}")
    print(f"[INFO] selected for inference: {len(selected)} (max_samples={args.max_samples})")

    # ---- load only needed conversations ----
    conv_map = load_first_round_for_ids(meta_json, sel_ids)
    print(f"[INFO] loaded conversations: {len(conv_map)}/{len(sel_ids)}")

    # ---- load Qwen once ----
    try:
        model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            args.qwen_model,
            dtype="auto",
            device_map="auto",
        )
    except TypeError:
        model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            args.qwen_model,
            torch_dtype="auto",
            device_map="auto",
        )
    model.eval()

    processor = Qwen3OmniMoeProcessor.from_pretrained(args.qwen_model)
    tokenizer = processor.tokenizer

    # register <touch>
    old_ids = tokenizer.encode(TOUCH_PLACEHOLDER, add_special_tokens=False)
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": [AddedToken(TOUCH_PLACEHOLDER, lstrip=False, rstrip=False)]}
    )
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
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

    emb_layer = model.get_input_embeddings()
    emb_device = emb_layer.weight.device
    emb_dtype = emb_layer.weight.dtype
    llm_hidden = int(emb_layer.weight.shape[1])

    print(f"[INFO] Qwen emb device={emb_device} dtype={emb_dtype} hidden={llm_hidden}")
    print(f"[INFO] <touch> id={tokenizer.convert_tokens_to_ids(TOUCH_PLACEHOLDER)} added={num_added}")

    # ---- load AE once ----
    ckpt_file = resolve_ckpt_file(args.ae_ckpt)
    print(f"[INFO] AE ckpt file: {ckpt_file}")
    ae, ae_cfg = load_ae_from_ckpt(ckpt_file, device=emb_device, dtype=emb_dtype)
    try:
        ae_d_text_in = int(ae_cfg["model"].get("d_text_in", -1))
        if ae_d_text_in != -1 and ae_d_text_in != llm_hidden:
            print(f"[WARN] AE d_text_in={ae_d_text_in} != Qwen hidden={llm_hidden} -> 可能需要 projection")
    except Exception:
        pass

    # ---- load feature dataset (full, but only fetch selected indices later) ----
    ds = ProcessedTouchTextFeatureDataset(args.feature_yaml, require_valid=False, return_ids=True)
    print(f"[INFO] feature dataset loaded, len={len(ds)}")

    # ---- output ----
    split_out = os.path.join(args.out_dir, "train_ssvtp")
    ensure_dir(split_out)
    out_jsonl = os.path.join(split_out, "results.jsonl")
    out_csv = os.path.join(split_out, "results.csv")
    out_summary = os.path.join(split_out, "summary.json")

    rows: List[Dict[str, Any]] = []
    num_total = 0
    num_ok_any = 0
    num_err_any = 0
    num_missing_conv = 0

    for dataset_idx, sid in selected:
        num_total += 1

        row = {
            "sample_id": sid,
            "dataset_idx": int(dataset_idx),
            "global_index": "",
            "question": "",
            "gt": "",
            "pred_textemb_plain": "",
            "pred_touch_plain": "",
            "pred_touch_vocab": "",
            "K_textemb": 0,
            "K_touch": 0,
            "status_textemb": "skip",
            "status_touch_plain": "skip",
            "status_touch_vocab": "skip",
            "error_textemb": "",
            "error_touch_plain": "",
            "error_touch_vocab": "",
        }

        conv = conv_map.get(sid, None)
        if conv is None:
            num_missing_conv += 1
            row["status_textemb"] = "skip_missing_conv"
            row["status_touch_plain"] = "skip_missing_conv"
            row["status_touch_vocab"] = "skip_missing_conv"
            row["error_textemb"] = "not found in meta_json(conv_map)"
            row["error_touch_plain"] = "not found in meta_json(conv_map)"
            row["error_touch_vocab"] = "not found in meta_json(conv_map)"
            rows.append(row)
            continue

        human_raw = conv.get("human", "")
        gt = conv.get("gt", "")
        question = clean_question(human_raw)

        row["question"] = question
        row["gt"] = gt

        # fetch feature sample
        sample = ds[int(dataset_idx)]
        row["global_index"] = sample.get("global_index", "")

        valid = bool(sample.get("valid", True))
        if not valid:
            row["status_textemb"] = "skip_invalid_feature"
            row["status_touch_plain"] = "skip_invalid_feature"
            row["status_touch_vocab"] = "skip_invalid_feature"
            row["error_textemb"] = "valid=0"
            row["error_touch_plain"] = "valid=0"
            row["error_touch_vocab"] = "valid=0"
            rows.append(row)
            continue

        # ---------- (1) baseline: inject GT text embeddings ----------
        if args.run_baseline_textemb:
            try:
                text_tokens = sample["text"].to(device=emb_device, dtype=emb_dtype)  # (L,H)
                mask = sample["mask"].to(device=emb_device)                          # (L,) bool

                if args.inject_mode == "sequence":
                    seq = text_tokens[mask] if mask.any() else text_tokens[:1]
                    if seq.shape[0] > int(args.max_touch_tokens):
                        seq = seq[: int(args.max_touch_tokens)]
                    K = int(seq.shape[0])
                    payload = seq
                else:
                    m = mask.to(text_tokens.dtype)
                    denom = m.sum().clamp_min(1.0)
                    vec = (text_tokens * m.unsqueeze(-1)).sum(dim=0) / denom
                    K = 1
                    payload = vec

                row["K_textemb"] = K
                user_text = build_user_prompt_plain(K, question)

                row["pred_textemb_plain"] = predict_once(
                    model=model,
                    processor=processor,
                    tokenizer=tokenizer,
                    emb_layer=emb_layer,
                    emb_device=emb_device,
                    user_text=user_text,
                    payload=payload,
                    max_new_tokens=int(args.max_new_tokens),
                    do_sample=bool(args.do_sample),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                )
                row["status_textemb"] = "ok"
            except Exception as e:
                row["status_textemb"] = "error"
                row["error_textemb"] = repr(e)

        # ---------- (2)(3) touch -> AE payload ----------
        touch_payload: Optional[torch.Tensor] = None
        K_touch: int = 0
        touch_err: Optional[str] = None

        if args.run_touch_plain or args.run_touch_vocab:
            try:
                pred_tokens, mask2 = ae_touch_to_text_token_embeddings(
                    ae=ae,
                    touch_tokens=sample["touch"],
                    text_shape_like=sample["text"],
                    text_mask=sample["mask"],
                    device=emb_device,
                    dtype=emb_dtype,
                )

                if args.inject_mode == "sequence":
                    seq2 = pred_tokens[mask2] if mask2.any() else pred_tokens[:1]
                    if seq2.shape[0] > int(args.max_touch_tokens):
                        seq2 = seq2[: int(args.max_touch_tokens)]
                    K_touch = int(seq2.shape[0])
                    touch_payload = seq2
                else:
                    m2 = mask2.to(pred_tokens.dtype)
                    denom2 = m2.sum().clamp_min(1.0)
                    vec2 = (pred_tokens * m2.unsqueeze(-1)).sum(dim=0) / denom2
                    K_touch = 1
                    touch_payload = vec2

                row["K_touch"] = K_touch

            except Exception as e:
                touch_err = repr(e)

        # ---------- (2) touch_plain ----------
        if args.run_touch_plain:
            if touch_payload is None:
                row["status_touch_plain"] = "error"
                row["error_touch_plain"] = touch_err or "touch_payload is None"
            else:
                try:
                    user_text2 = build_user_prompt_plain(K_touch, question)
                    row["pred_touch_plain"] = predict_once(
                        model=model,
                        processor=processor,
                        tokenizer=tokenizer,
                        emb_layer=emb_layer,
                        emb_device=emb_device,
                        user_text=user_text2,
                        payload=touch_payload,
                        max_new_tokens=int(args.max_new_tokens),
                        do_sample=bool(args.do_sample),
                        temperature=float(args.temperature),
                        top_p=float(args.top_p),
                    )
                    row["status_touch_plain"] = "ok"
                except Exception as e:
                    row["status_touch_plain"] = "error"
                    row["error_touch_plain"] = repr(e)

        # ---------- (3) touch_vocab ----------
        if args.run_touch_vocab:
            if touch_payload is None:
                row["status_touch_vocab"] = "error"
                row["error_touch_vocab"] = touch_err or "touch_payload is None"
            else:
                try:
                    user_text3 = build_user_prompt_vocab(K_touch, question, vocab_block=vocab_block)
                    row["pred_touch_vocab"] = predict_once(
                        model=model,
                        processor=processor,
                        tokenizer=tokenizer,
                        emb_layer=emb_layer,
                        emb_device=emb_device,
                        user_text=user_text3,
                        payload=touch_payload,
                        max_new_tokens=int(args.max_new_tokens),
                        do_sample=bool(args.do_sample),
                        temperature=float(args.temperature),
                        top_p=float(args.top_p),
                    )
                    row["status_touch_vocab"] = "ok"
                except Exception as e:
                    row["status_touch_vocab"] = "error"
                    row["error_touch_vocab"] = repr(e)

        ok_any = any([
            (row["status_textemb"] == "ok") if args.run_baseline_textemb else False,
            (row["status_touch_plain"] == "ok") if args.run_touch_plain else False,
            (row["status_touch_vocab"] == "ok") if args.run_touch_vocab else False,
        ])
        err_any = any([
            (row["status_textemb"] == "error") if args.run_baseline_textemb else False,
            (row["status_touch_plain"] == "error") if args.run_touch_plain else False,
            (row["status_touch_vocab"] == "error") if args.run_touch_vocab else False,
        ])
        if ok_any:
            num_ok_any += 1
        if err_any:
            num_err_any += 1
        print(row)
        rows.append(row)

        if num_total % 20 == 0:
            print(f"[INFO] progress {num_total}/{len(selected)} ok_any={num_ok_any} err_any={num_err_any} missing_conv={num_missing_conv}")

    # ---- write outputs ----
    write_jsonl(out_jsonl, rows)
    write_csv(out_csv, rows)

    summary = {
        "split": "train_ssvtp",
        "feature_yaml": os.path.abspath(args.feature_yaml),
        "meta_json": os.path.abspath(meta_json),
        "selected": {
            "prefix": SSVTP_PREFIX,
            "total_matched_valid": int(total_matched),
            "max_samples": int(args.max_samples),
            "selected_count": int(len(selected)),
            "missing_conv": int(num_missing_conv),
            "seed": int(args.seed),
        },
        "counts": {
            "num_total": int(num_total),
            "num_ok_any": int(num_ok_any),
            "num_err_any": int(num_err_any),
        },
        "outputs": {
            "jsonl": os.path.abspath(out_jsonl),
            "csv": os.path.abspath(out_csv),
        },
        "config": {
            "inject_mode": args.inject_mode,
            "max_touch_tokens": int(args.max_touch_tokens),
            "max_new_tokens": int(args.max_new_tokens),
            "do_sample": bool(args.do_sample),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "run_baseline_textemb": bool(args.run_baseline_textemb),
            "run_touch_plain": bool(args.run_touch_plain),
            "run_touch_vocab": bool(args.run_touch_vocab),
            "vocab_path": os.path.abspath(args.vocab_path),
            "vocab_max_words": int(VOCAB_MAX_WORDS),
            "vocab_max_chars": int(VOCAB_MAX_CHARS),
        },
    }
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    overall = {
        "out_dir": os.path.abspath(args.out_dir),
        "split_summary": summary,
    }
    with open(os.path.join(args.out_dir, "summary_all.json"), "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    print("\n[INFO] Done.")
    print(f"[INFO] outputs: {split_out}")
    print(f"[INFO] results.csv: {out_csv}")
    print(f"[INFO] results.jsonl: {out_jsonl}")
    print(f"[INFO] summary.json: {out_summary}")
    print(f"[INFO] summary_all.json: {os.path.join(args.out_dir, 'summary_all.json')}")


if __name__ == "__main__":
    main()