# -*- coding: utf-8 -*-
"""
infer_tvl_test.py

基于 infer_tvl_preprocess.py 生成的两份 features（ssvtp/hct）分别做推理，并分别保存结果。

输出结构：
INFER_OUT_DIR/
  ssvtp/
    results.jsonl
    results.csv
    summary.json
  hct/
    results.jsonl
    results.csv
    summary.json
"""

from __future__ import annotations

import os
import csv
import json
import glob
import argparse
from typing import Any, Dict, List, Optional, Tuple

import torch
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
# ✅ 默认设置写在文件开头（你要的）
# =========================================================
DEFAULT_PREPROC_ROOT = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/tvl_test/outfeatures"
DEFAULT_INFER_OUT_DIR = os.path.join(DEFAULT_PREPROC_ROOT, "infer_results")

DEFAULT_AE_CKPT = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/swift/checkpoints/tvl/stage1/1/best.pt"
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

DEFAULT_RUN_SSVTP = True
DEFAULT_RUN_HCT = True

DEFAULT_INJECT_MODE = "sequence"   # sequence | pooled
DEFAULT_MAX_TOUCH_TOKENS = 24      # 一般<=TEXT_MAX_LEN=24
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_DO_SAMPLE = False
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9

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
# helpers
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
    return t.strip()


def build_user_prompt_with_touch(question_text: str, k: int) -> str:
    k = max(1, int(k))
    q = (question_text or "").strip() or "This image gives tactile feelings of?"
    block = " ".join([TOUCH_PLACEHOLDER] * k)
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


def load_meta_map(meta_json: str) -> Dict[str, Dict[str, Any]]:
    with open(meta_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, Dict[str, Any]] = {}
    for it in data:
        sid = str(it.get("id", "")).strip()
        if sid:
            out[sid] = it
    return out


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    cols = [
        "sample_id", "dataset", "subset", "source_csv",
        "tactile", "tactile_background",
        "question", "pred", "gt", "K_injected", "status", "error",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


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
    text_shape_like: torch.Tensor,  # (24,2048) - 只用 shape
    text_mask: torch.Tensor,        # (24,)
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
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


# =========================================================
# run one split
# =========================================================
def run_one_dataset(
    *,
    name: str,
    feature_yaml: str,
    meta_json: str,
    out_dir: str,
    model,
    processor,
    tokenizer,
    emb_layer,
    emb_device: torch.device,
    emb_dtype: torch.dtype,
    ae: UnifiedTouchTextAE,
    inject_mode: str,
    max_touch_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> Dict[str, Any]:
    ensure_dir(out_dir)

    out_jsonl = os.path.join(out_dir, "results.jsonl")
    out_csv = os.path.join(out_dir, "results.csv")
    out_summary = os.path.join(out_dir, "summary.json")

    meta_map = load_meta_map(meta_json)
    ds = ProcessedTouchTextFeatureDataset(feature_yaml, require_valid=False, return_ids=True)

    rows: List[Dict[str, Any]] = []
    num_total = 0
    num_valid = 0
    num_ok = 0
    num_err = 0

    print(f"\n[INFO] ===== Running split={name} =====")
    print(f"[INFO] feature_yaml={feature_yaml}")
    print(f"[INFO] meta_json={meta_json}")
    print(f"[INFO] len(ds)={len(ds)}")

    for i in range(len(ds)):
        num_total += 1
        sample = ds[i]
        sid = str(sample.get("sample_id", "")).strip()
        valid = bool(sample.get("valid", True))

        meta = meta_map.get(sid, None)

        base = {
            "sample_id": sid,
            "global_index": sample.get("global_index", None),
            "valid": valid,
            "dataset": meta.get("dataset") if meta else name,
            "subset": meta.get("subset", "") if meta else "",
            "source_csv": meta.get("source_csv", "") if meta else "",
            "tactile": meta.get("tactile", "") if meta else "",
            "tactile_background": meta.get("tactile_background", "") if meta else "",
        }

        if not valid:
            rows.append({**base, "status": "skip_invalid_feature", "error": "valid=0"})
            continue
        num_valid += 1

        if meta is None:
            rows.append({**base, "status": "skip_missing_meta", "error": "sample_id not found in meta_json"})
            continue

        pair = extract_first_round(meta.get("conversations", []))
        if pair is None:
            human_raw = ""
            gt = str(meta.get("caption", ""))
        else:
            human_raw, gt = pair
        question = clean_question(human_raw)

        try:
            pred_tokens, mask = ae_touch_to_text_token_embeddings(
                ae=ae,
                touch_tokens=sample["touch"],
                text_shape_like=sample["text"],
                text_mask=sample["mask"],
                device=emb_device,
                dtype=emb_dtype,
            )

            if inject_mode == "sequence":
                seq = pred_tokens[mask] if mask.any() else pred_tokens[:1]
                if seq.shape[0] > int(max_touch_tokens):
                    seq = seq[: int(max_touch_tokens)]
                K = int(seq.shape[0])
                payload = seq
            else:
                m = mask.to(pred_tokens.device).to(pred_tokens.dtype)
                denom = m.sum().clamp_min(1.0)
                vec = (pred_tokens * m.unsqueeze(-1)).sum(dim=0) / denom
                K = 1
                payload = vec

            user_text = build_user_prompt_with_touch(question, K)

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

            rows.append(
                {
                    **base,
                    "question": question,
                    "pred": pred,
                    "gt": gt,
                    "K_injected": K,
                    "status": "ok",
                    "error": "",
                }
            )
            num_ok += 1

        except Exception as e:
            rows.append(
                {
                    **base,
                    "question": question,
                    "pred": "",
                    "gt": gt,
                    "K_injected": 0,
                    "status": "error",
                    "error": repr(e),
                }
            )
            num_err += 1

    write_jsonl(out_jsonl, rows)
    write_csv(out_csv, rows)

    summary = {
        "split": name,
        "feature_yaml": os.path.abspath(feature_yaml),
        "meta_json": os.path.abspath(meta_json),
        "counts": {
            "num_total": num_total,
            "num_valid": num_valid,
            "num_ok": num_ok,
            "num_err": num_err,
        },
        "outputs": {
            "jsonl": os.path.abspath(out_jsonl),
            "csv": os.path.abspath(out_csv),
        },
        "config": {
            "inject_mode": inject_mode,
            "max_touch_tokens": int(max_touch_tokens),
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(do_sample),
            "temperature": float(temperature),
            "top_p": float(top_p),
        },
    }

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[INFO] split={name} done -> {out_dir}")
    return summary


# =========================================================
# main
# =========================================================
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--preproc_root", type=str, default=DEFAULT_PREPROC_ROOT)
    ap.add_argument("--out_dir", type=str, default=DEFAULT_INFER_OUT_DIR)

    ap.add_argument("--run_ssvtp", action="store_true", default=DEFAULT_RUN_SSVTP)
    ap.add_argument("--run_hct", action="store_true", default=DEFAULT_RUN_HCT)

    ap.add_argument("--ae_ckpt", type=str, default=DEFAULT_AE_CKPT)
    ap.add_argument("--qwen_model", type=str, default=DEFAULT_QWEN_MODEL)

    ap.add_argument("--inject_mode", type=str, default=DEFAULT_INJECT_MODE, choices=["sequence", "pooled"])
    ap.add_argument("--max_touch_tokens", type=int, default=DEFAULT_MAX_TOUCH_TOKENS)
    ap.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)

    ap.add_argument("--do_sample", action="store_true", default=DEFAULT_DO_SAMPLE)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)

    return ap.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    # -------- load Qwen once --------
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

    # register <touch> as single token
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

    # -------- load AE once --------
    ckpt_file = resolve_ckpt_file(args.ae_ckpt)
    print(f"[INFO] AE ckpt file: {ckpt_file}")
    ae, ae_cfg = load_ae_from_ckpt(ckpt_file, device=emb_device, dtype=emb_dtype)

    try:
        ae_d_text_in = int(ae_cfg["model"].get("d_text_in", -1))
        if ae_d_text_in != -1 and ae_d_text_in != llm_hidden:
            print(f"[WARN] AE d_text_in={ae_d_text_in} != Qwen hidden={llm_hidden} -> 注入可能失败（需要 projection）")
    except Exception:
        pass

    summaries: List[Dict[str, Any]] = []

    # -------- run ssvtp --------
    if args.run_ssvtp:
        feature_yaml = os.path.join(args.preproc_root, "ssvtp", "features", "dataset_info.yaml")
        meta_json = os.path.join(args.preproc_root, "ssvtp", "meta_test.json")
        out_dir = os.path.join(args.out_dir, "ssvtp")
        summaries.append(
            run_one_dataset(
                name="ssvtp",
                feature_yaml=feature_yaml,
                meta_json=meta_json,
                out_dir=out_dir,
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                emb_layer=emb_layer,
                emb_device=emb_device,
                emb_dtype=emb_dtype,
                ae=ae,
                inject_mode=args.inject_mode,
                max_touch_tokens=args.max_touch_tokens,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        )

    # -------- run hct --------
    if args.run_hct:
        feature_yaml = os.path.join(args.preproc_root, "hct", "features", "dataset_info.yaml")
        meta_json = os.path.join(args.preproc_root, "hct", "meta_test.json")
        out_dir = os.path.join(args.out_dir, "hct")
        summaries.append(
            run_one_dataset(
                name="hct",
                feature_yaml=feature_yaml,
                meta_json=meta_json,
                out_dir=out_dir,
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                emb_layer=emb_layer,
                emb_device=emb_device,
                emb_dtype=emb_dtype,
                ae=ae,
                inject_mode=args.inject_mode,
                max_touch_tokens=args.max_touch_tokens,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        )

    # write overall summary
    overall = {
        "preproc_root": os.path.abspath(args.preproc_root),
        "infer_out_dir": os.path.abspath(args.out_dir),
        "ae_ckpt": os.path.abspath(ckpt_file),
        "qwen_model": args.qwen_model,
        "splits": summaries,
    }
    with open(os.path.join(args.out_dir, "summary_all.json"), "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    print("\n[INFO] Done.")
    print(f"[INFO] infer out: {args.out_dir}")
    print(f"[INFO] summary_all.json: {os.path.join(args.out_dir, 'summary_all.json')}")


if __name__ == "__main__":
    main()