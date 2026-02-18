# -*- coding: utf-8 -*-
"""
infer_modelnet40_point_ae_qwen3_omni.py

用途：
- 在 ModelNet40 特征数据集（只含 point_tokens + object_labels）上做推理
- 用你训练好的 UnifiedPointTextAE 将 point_tokens 映射为“文本模态 token embedding”
- 将 embedding 注入到 Qwen3-Omni 的 <point> 占位 token 位置
- 任务：ModelNet40 40 分类（输出一个 label）
- baseline：仍用旧版 caption prompt（看看会输出什么），结果另存一份文件

符合你的改动要求：
1) 仅两种取样方式：顺序前 N 个，或全量
2) 不处理 text_embeds/text_mask（新数据集没有）
3) 不比较旧 baseline（只跑“我的方法” + 新增的“旧 prompt baseline”）
4) 注入 token 数量由一个超参数固定（FIXED_NUM_POINT_TOKENS）
5) 保留 Top-K token proxy（重构 token 最像哪些词）
6) prompt 修改为 40 分类；另做 baseline：旧 prompt（caption）并保存到另一文件
7) 不使用 argparse，全部全局变量
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ====== OFFLINE MODE（按你旧脚本保持）======
os.environ["HF_HUB_OFFLINE"] = "1"

# ====== 你的模块（保持与训练脚本一致的 import 路径）======
from swift.point_cloud.stage1.src.models.unified_ae import UnifiedPointTextAE
from swift.point_cloud.stage1.src.eval.modelnet40_dataset import ModelNet40PointTokenDataset


# ====== Qwen3-Omni (Transformers) ======
from transformers import (
    AddedToken,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeProcessor,
)

# ====== rich（可选，和旧脚本一致）======
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.pretty import Pretty
    from rich import box

    _RICH_AVAILABLE = True
    _console = Console()
except Exception:
    _RICH_AVAILABLE = False
    _console = None
    Console = None
    Table = None
    Panel = None
    Pretty = None
    box = None


def _p(obj: Any = "", **kwargs: Any) -> None:
    if _console is not None:
        _console.print(obj, **kwargs)
    else:
        print(obj)


def _rule(title: str = "") -> None:
    if _console is not None:
        _console.rule(title)
    else:
        if title:
            print("=" * 100 + " " + title)
        else:
            print("=" * 100)


# ============================================================
# 0) 全局配置（按需直接改这里；不使用 argparse）
# ============================================================

# ---------- 必改：你的 AE checkpoint ----------
AE_CKPT_PATH = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/checkpoints/v100-20260215-045054/point_ae_finetuned_checkpoint-1006.pt"

# ---------- 必改：ModelNet40 特征 pt ----------
MODELNET40_FEATURE_PT_PATH = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/PointLLM/PointLLM/modelnet40_test_point_tokens.pt"

# ---------- Qwen3-Omni ----------
QWEN_MODEL_NAME_OR_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

# ---------- 取样方式（仅两种） ----------
# "first_n": 顺序取前 N 个样本（从 START_INDEX 开始）
# "all":     全量样本（从 START_INDEX 到末尾）
SAMPLE_MODE = "first_n"  # "first_n" | "all"
START_INDEX = 0
NUM_SAMPLES = 20  # SAMPLE_MODE="first_n" 时生效

# ---------- 数据集 valid 过滤 ----------
REQUIRE_VALID = False

# ---------- 注入相关 ----------
POINT_PLACEHOLDER = "<point>"

# 你要求的：固定注入 token 数（不再依赖 text_mask）
FIXED_NUM_POINT_TOKENS = 24

# AE forward 需要一个 text_feat + text_mask 的“dummy 形状”，这里决定 dummy 的总长度
# - 若 None：尝试从 AE ckpt cfg 推断；推断不到则用 max(FIXED_NUM_POINT_TOKENS, 16)
# - 若你运行报 shape 相关错误，可手动设为训练时的 text token 总长度（例如 256/512）
AE_DUMMY_TEXT_TOTAL_LEN: Optional[int] = None

# ---------- 生成超参数 ----------
MAX_NEW_TOKENS = 64
DO_SAMPLE = False
TEMPERATURE = 0.7
TOP_P = 0.9

# ---------- 输出文件 ----------
OUTPUT_DIR = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/swift/point_cloud/stage1/src/eval/modelnet40_infer_outputs"
OUT_MAIN_JSONL = os.path.join(OUTPUT_DIR, "predictions_modelnet40_cls.jsonl")
OUT_BASELINE_JSONL = os.path.join(OUTPUT_DIR, "predictions_baseline_caption_prompt.jsonl")
OVERWRITE_OUTPUT_FILES = True  # True: 覆盖写；False: 追加

# ---------- 打印/调试 ----------
VERBOSE_PRINT_PER_SAMPLE = True
PRINT_EVERY = 1  # 每隔多少条打印一次（全量推理时建议调大）

DEBUG_SHOW_INJECTION_DEBUG = True
DEBUG_MAX_SHOW_SPANS = 12
DEBUG_SHOW_TOPK_TOKEN_PROXIES = True
DEBUG_TOPK_TOKENS = 6

# 为了节省算力：只对前多少条样本做 token proxy（全量推理建议设小点；设 None 表示全做）
DEBUG_ONLY_FIRST_N_FOR_TOKEN_PROXY: Optional[int] = 20

# ---------- 随机种子 ----------
SEED = 42

# ============================================================
# 1) Prompt（主任务：40 分类；baseline：旧 caption prompt）
# 注意：system prompt 中不要出现字面 "<point>"（避免 tokenizer 生成额外占位 token）
# ============================================================

SYSTEM_PROMPT_CLS = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group.\n\n"
    "Task setting:\n"
    "- You will classify an object represented by a 3D point cloud into one of the ModelNet40 categories.\n"
    "- The user message may contain a section named '3D_POINT_CLOUD_EMBEDDING'. The tokens inside that section are "
    "placeholders whose embeddings are injected at inference time to carry semantic information about the 3D object.\n\n"
    "Instructions:\n"
    "- Use the embedding section as the ONLY object context.\n"
    "- Output exactly ONE category label from the provided list.\n"
    "- Output only the label string, no extra words, no punctuation, no JSON.\n"
)

# baseline：沿用你旧脚本的 system prompt（caption/QA 风格）
SYSTEM_PROMPT_OLD_CAPTION = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of understanding text inputs "
    "and generating helpful responses.\n\n"
    "Task setting:\n"
    "- You will answer questions about an object represented by a 3D point cloud.\n"
    "- In some requests, the user message will contain a section named '3D_POINT_CLOUD_EMBEDDING'. "
    "The tokens in that section are placeholders whose embeddings are injected at inference time to carry semantic "
    "information about the 3D object.\n\n"
    "Instructions:\n"
    "- Use the '3D_POINT_CLOUD_EMBEDDING' section as object context to answer the question.\n"
    "- If the embedding section is absent, answer based only on the text question and be explicit about uncertainty.\n"
    "- Output only the final answer text (no role labels such as user/assistant, no extra dialogue markers).\n"
)


# ============================================================
# 2) 工具函数（随机种子 / device / prompt / parse label）
# ============================================================

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device_dtype(x: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if x.dtype in (torch.bool, torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8):
        return x.to(device=device)
    return x.to(device=device, dtype=dtype)


def build_user_prompt_with_points_cls(labels: List[str], k: int, placeholder: str = "<point>") -> str:
    k = max(1, int(k))
    point_block = " ".join([placeholder] * k)

    # 把 label list 写成稳定格式，尽量减少模型“自造类别名”
    label_lines = "\n".join([f"- {x}" for x in labels])

    return (
        "3D_POINT_CLOUD_EMBEDDING:\n"
        f"{point_block}\n\n"
        "TASK:\n"
        "Classify the object into exactly one of the following ModelNet40 categories:\n"
        f"{label_lines}\n\n"
        "ANSWER:\n"
    )


def build_user_prompt_with_points_old_caption(k: int, placeholder: str = "<point>") -> str:
    # 尽量保持你旧 prompt 结构（3D_POINT_CLOUD_EMBEDDING / QUESTION / ANSWER）
    k = max(1, int(k))
    point_block = " ".join([placeholder] * k)
    q = "Describe the object represented by the 3D point cloud in one short sentence."
    return (
        "3D_POINT_CLOUD_EMBEDDING:\n"
        f"{point_block}\n\n"
        "QUESTION:\n"
        f"{q}\n\n"
        "ANSWER:"
    )


def parse_modelnet40_label(text: str, label_set: set) -> Optional[str]:
    """
    从模型输出中解析 label（更鲁棒一点）：
    - 优先：输出整体等于某个 label
    - 否则：在输出中找最先出现的 label 子串（按 word boundary-ish 匹配）
    """
    if text is None:
        return None
    t = str(text).strip().lower()
    t = t.strip().strip('"').strip("'").strip("`").strip()
    if t in label_set:
        return t

    # 常见格式：Label: chair / The object is chair / chair.
    # 用“非 [a-z0-9_]”做边界，兼容下划线 label
    best = None
    best_pos = 10**9
    for lab in label_set:
        m = re.search(rf"(?<![a-z0-9_]){re.escape(lab)}(?![a-z0-9_])", t)
        if m:
            if m.start() < best_pos:
                best_pos = m.start()
                best = lab
    return best


# ============================================================
# 3) 加载 AE
# ============================================================

def load_ae_from_ckpt(ckpt_path: str, device: torch.device, dtype: torch.dtype) -> Tuple[UnifiedPointTextAE, Dict[str, Any]]:
    """
    ckpt 结构来自你的训练脚本保存：
      ckpt = {"cfg": cfg, "model": state_dict, ...}
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", None)
    if cfg is None:
        raise RuntimeError(f"Checkpoint missing 'cfg': {ckpt_path}")
    model_cfg = cfg.get("model", None)
    if model_cfg is None:
        raise RuntimeError(f"Checkpoint cfg missing 'model': {ckpt_path}")

    ae = UnifiedPointTextAE(model_cfg)
    ae.load_state_dict(ckpt["model"], strict=True)
    ae.eval()
    ae.to(device=device, dtype=dtype)
    return ae, cfg


def infer_ae_text_total_len_from_cfg(ae_cfg: Dict[str, Any], default_len: int) -> int:
    """
    尝试从 ae_cfg["model"] 推断 text 序列总长度（dummy text_feat 的 L）。
    推断失败就用 default_len。
    """
    model_cfg = ae_cfg.get("model", {})
    # 常见字段名做一圈尝试（不保证都有）
    candidates = [
        "max_text_len",
        "text_max_len",
        "max_len",
        "text_len",
        "n_text_tokens",
        "num_text_tokens",
        "L_text",
        "l_text",
    ]
    for k in candidates:
        if k in model_cfg:
            try:
                v = int(model_cfg[k])
                if v > 0:
                    return v
            except Exception:
                pass
    return int(default_len)


@torch.no_grad()
def ae_point_to_text_token_embeddings_fixed(
    *,
    ae: UnifiedPointTextAE,
    point_tokens: torch.Tensor,   # (T,D)
    llm_hidden: int,              # H
    fixed_k: int,                 # K
    dummy_total_len: int,         # L_total
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    在没有 text_embeds/text_mask 的情况下：
    - 构造 dummy text_feat: (1, L_total, H) 全 0
    - 构造 text_mask: 前 K 个 True，其余 False
    - 得到 out["text_recon_from_point"]: (1, L_total, H)
    - 返回：
        pred_seq: (K, H)   （用于注入）
        pred_all: (L_total, H) （调试备用）
        mask_all: (L_total,) bool
    """
    fixed_k = int(fixed_k)
    dummy_total_len = int(dummy_total_len)
    if fixed_k <= 0:
        raise ValueError(f"fixed_k must be > 0, got {fixed_k}")
    if dummy_total_len < fixed_k:
        raise ValueError(f"dummy_total_len ({dummy_total_len}) must be >= fixed_k ({fixed_k})")

    pt = to_device_dtype(point_tokens.unsqueeze(0), device, dtype)  # (1,T,D)

    # dummy text tokens: (1, L_total, H)
    te = torch.zeros((1, dummy_total_len, llm_hidden), device=device, dtype=dtype)

    mask = torch.zeros((dummy_total_len,), device=device, dtype=torch.bool)
    mask[:fixed_k] = True
    tm = mask.unsqueeze(0)  # (1,L_total)

    out = ae(point_feat=pt, text_feat=te, text_mask=tm)
    if "text_recon_from_point" not in out:
        raise RuntimeError("AE forward output missing key: 'text_recon_from_point'")

    pred_all = out["text_recon_from_point"][0]  # (L_total, H)
    pred_seq = pred_all[mask]                  # (K, H)
    return pred_seq, pred_all, mask


# ============================================================
# 4) 构造 Qwen 输入、注入 embedding、Top-K token proxy、生成
# ============================================================

def build_qwen_inputs(
    processor: Qwen3OmniMoeProcessor,
    system_prompt: str,
    user_text: str,
) -> Dict[str, torch.Tensor]:
    conversations = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
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


@torch.no_grad()
def inject_point_embeddings_into_inputs_embeds(
    *,
    tokenizer,
    input_ids: torch.Tensor,         # (1, S)
    inputs_embeds: torch.Tensor,     # (1, S, H)
    point_embeddings: torch.Tensor,  # (K, H)
    point_placeholder: str = "<point>",
) -> Tuple[torch.Tensor, List[Tuple[int, int]], List[Tuple[str, List[int]]]]:
    """
    基于“<point> 作为一个单独 special token（1 token）”的注入逻辑：
    - 找到 input_ids 中所有 <point> token_id 的位置
    - 用 point_embeddings[i] 替换对应位置 embedding（每个 <point> 替换 1 个 token）
    """
    new_embeds = inputs_embeds.clone()
    H = new_embeds.shape[-1]

    if point_embeddings.dim() != 2:
        raise RuntimeError(f"point_embeddings must be (K,H), got {tuple(point_embeddings.shape)}")
    if point_embeddings.shape[1] != H:
        raise RuntimeError(f"point_embeddings hidden mismatch: {point_embeddings.shape[1]} vs {H}")
    K = int(point_embeddings.shape[0])
    if K <= 0:
        raise RuntimeError("point_embeddings has zero length (K=0).")

    point_token_id = tokenizer.convert_tokens_to_ids(point_placeholder)
    if point_token_id is None or int(point_token_id) < 0:
        raise RuntimeError(
            f"Placeholder token '{point_placeholder}' is not in tokenizer vocab. "
            f"Make sure you add it as an additional special token before inference."
        )
    point_token_id = int(point_token_id)

    pos = torch.where(input_ids[0] == point_token_id)[0].tolist()
    if len(pos) < K:
        raise RuntimeError(f"Found {len(pos)} placeholder tokens, but need K={K}. Check prompt expansion.")

    spans = [(int(p), int(p) + 1) for p in pos[:K]]
    patterns = [(point_placeholder, [point_token_id])]

    for i in range(K):
        p = int(pos[i])
        vec = point_embeddings[i].to(device=new_embeds.device, dtype=new_embeds.dtype).view(1, 1, H)
        new_embeds[:, p:p+1, :] = vec

    return new_embeds, spans, patterns


@torch.no_grad()
def _topk_token_proxies(
    *,
    tokenizer,
    emb_weight: torch.Tensor,                 # (V,H)
    emb_weight_norm: Optional[torch.Tensor],  # (V,)
    vec: torch.Tensor,                        # (H,)
    k: int,
) -> List[Tuple[int, str, float]]:
    if k <= 0:
        return []

    w = emb_weight
    v = vec.to(device=w.device, dtype=w.dtype)

    dot = torch.matmul(w, v)  # (V,)
    v_norm = v.norm().clamp_min(1e-6)

    if emb_weight_norm is None:
        scores = dot / v_norm
    else:
        denom = (emb_weight_norm * v_norm).clamp_min(1e-6)
        scores = dot / denom

    kk = min(int(k), int(scores.numel()))
    topv, topi = torch.topk(scores, k=kk, largest=True)

    ids = topi.detach().cpu().tolist()
    scs = topv.detach().float().cpu().tolist()
    toks = tokenizer.convert_ids_to_tokens(ids)
    return [(int(i), str(t), float(s)) for i, t, s in zip(ids, toks, scs)]


@torch.no_grad()
def render_injection_debug(
    *,
    label: str,
    tokenizer,
    input_ids: torch.Tensor,        # (1,S)
    base_embeds: torch.Tensor,      # (1,S,H)
    injected_embeds: torch.Tensor,  # (1,S,H)
    spans: List[Tuple[int, int]],
    payload: torch.Tensor,          # (K,H)
    emb_weight: Optional[torch.Tensor] = None,
    emb_weight_norm: Optional[torch.Tensor] = None,
    max_show_spans: int = 8,
    show_topk: bool = True,
    topk: int = 6,
) -> None:
    if not DEBUG_SHOW_INJECTION_DEBUG:
        return

    show_n = min(len(spans), int(max_show_spans))

    if _console is not None:
        tb = Table(
            title=f"[bold]Injection Debug[/bold] - {label}",
            box=box.SIMPLE_HEAVY,
            show_lines=False,
        )
        tb.add_column("#", justify="right", style="bold")
        tb.add_column("span", justify="left")
        tb.add_column("tokens", justify="left", overflow="fold")
        tb.add_column("max|inj-base|", justify="right")
        tb.add_column("max|inj-payload|", justify="right")
        tb.add_column("cos(base,inj)", justify="right")
        tb.add_column("||payload||", justify="right")
        if show_topk and (emb_weight is not None):
            tb.add_column(f"Top-{topk} token proxy", justify="left", overflow="fold")

        for i in range(show_n):
            st, ed = spans[i]
            ids = input_ids[0, st:ed].tolist()
            toks = tokenizer.convert_ids_to_tokens(ids)

            base_seg = base_embeds[0, st:ed, :]
            inj_seg = injected_embeds[0, st:ed, :]

            base_vec = base_seg[0]
            inj_vec = inj_seg[0]
            payload_vec = payload[i]

            max_diff_payload = (inj_seg - payload_vec).abs().max().item()
            max_diff_base = (inj_seg - base_seg).abs().max().item()

            cos_bi = float(F.cosine_similarity(base_vec.view(1, -1), inj_vec.view(1, -1), dim=-1).item())
            payload_norm = float(payload_vec.norm().item())

            topk_str = ""
            if show_topk and (emb_weight is not None):
                proxies = _topk_token_proxies(
                    tokenizer=tokenizer,
                    emb_weight=emb_weight,
                    emb_weight_norm=emb_weight_norm,
                    vec=payload_vec,
                    k=topk,
                )
                def _esc(s: str) -> str:
                    return s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
                topk_str = ", ".join([f"{_esc(t)}({s:.3f})" for _, t, s in proxies])

            tb.add_row(
                str(i),
                f"[{st},{ed})",
                " ".join(toks),
                f"{max_diff_base:.3e}",
                f"{max_diff_payload:.3e}",
                f"{cos_bi:.3f}",
                f"{payload_norm:.3f}",
                topk_str if (show_topk and (emb_weight is not None)) else "",
            )

        _console.print(tb)
    else:
        print(f"[DEBUG] Injection Debug - {label}")
        for i in range(show_n):
            st, ed = spans[i]
            toks = tokenizer.convert_ids_to_tokens(input_ids[0, st:ed].tolist())
            base_seg = base_embeds[0, st:ed, :]
            inj_seg = injected_embeds[0, st:ed, :]
            payload_vec = payload[i]
            max_diff_payload = (inj_seg - payload_vec).abs().max().item()
            max_diff_base = (inj_seg - base_seg).abs().max().item()
            print(f"  [{i}] span=[{st},{ed}) toks={toks} max|inj-base|={max_diff_base:.3e} max|inj-payload|={max_diff_payload:.3e}")


_PAD_DEBUG_PRINTED = False


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

    pad_id = resolve_safe_pad_token_id(tokenizer, model)
    gen_kwargs["pad_token_id"] = pad_id
    if getattr(tokenizer, "eos_token_id", None) is not None:
        gen_kwargs["eos_token_id"] = tokenizer.eos_token_id

    global _PAD_DEBUG_PRINTED
    if not _PAD_DEBUG_PRINTED:
        _p(
            f"[DEBUG] tokenizer.pad_token_id={tokenizer.pad_token_id}, eos_token_id={tokenizer.eos_token_id}, resolved_pad_id={pad_id}",
            style="dim" if _console is not None else None,
        )
        _PAD_DEBUG_PRINTED = True

    extra = {}
    if do_sample:
        extra["temperature"] = float(temperature)
        extra["top_p"] = float(top_p)

    gen_out = model.generate(
        **gen_kwargs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        **extra,
    )

    prompt_len = inputs_embeds.shape[1]
    seq = gen_out[0]
    new_tokens = seq[prompt_len:] if seq.shape[0] > prompt_len else seq

    text = tokenizer.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    return text


@torch.no_grad()
def greedy_fallback_generate(
    *,
    model: Qwen3OmniMoeThinkerForConditionalGeneration,
    tokenizer,
    inputs: Dict[str, torch.Tensor],
    inputs_embeds: torch.Tensor,
    max_new_tokens: int,
) -> str:
    attention_mask = inputs.get("attention_mask", None)
    position_ids = inputs.get("position_ids", None)
    padding_mask = inputs.get("padding_mask", None)

    out = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        padding_mask=padding_mask,
        use_cache=True,
    )
    logits = out.logits
    past = out.past_key_values

    generated: List[int] = []
    eos = tokenizer.eos_token_id

    def _extend_attention_mask(am: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if am is None:
            return None
        one = torch.ones((am.shape[0], 1), device=am.device, dtype=am.dtype)
        return torch.cat([am, one], dim=1)

    def _extend_padding_mask(pm: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if pm is None:
            return None
        one = torch.zeros((pm.shape[0], 1), device=pm.device, dtype=pm.dtype)
        return torch.cat([pm, one], dim=1)

    def _extend_position_ids(pid: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if pid is None:
            return None
        if pid.dim() == 2:
            last = pid[:, -1:]
            nxt = last + 1
            return torch.cat([pid, nxt], dim=1)
        if pid.dim() == 3:
            last = pid[:, :, -1:]
            nxt = last + 1
            return torch.cat([pid, nxt], dim=2)
        return pid

    for _ in range(max_new_tokens):
        next_id = torch.argmax(logits[:, -1, :], dim=-1)
        tid = int(next_id.item())
        generated.append(tid)
        if eos is not None and tid == eos:
            break

        attention_mask = _extend_attention_mask(attention_mask)
        padding_mask = _extend_padding_mask(padding_mask)
        position_ids = _extend_position_ids(position_ids)

        out = model(
            input_ids=next_id.unsqueeze(0),
            attention_mask=attention_mask,
            position_ids=position_ids,
            padding_mask=padding_mask,
            past_key_values=past,
            use_cache=True,
        )
        logits = out.logits
        past = out.past_key_values

    text = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text.strip()


# ============================================================
# 5) I/O：写 JSONL
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def open_jsonl(path: str, overwrite: bool):
    mode = "w" if overwrite else "a"
    return open(path, mode, encoding="utf-8")


def write_jsonl_line(f, obj: Dict[str, Any]) -> None:
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    f.flush()


# ============================================================
# 6) main
# ============================================================

def main() -> None:
    set_global_seed(SEED)
    torch.set_grad_enabled(False)

    # ---- dataset ----
    ds = ModelNet40PointTokenDataset(MODELNET40_FEATURE_PT_PATH, require_valid=REQUIRE_VALID)
    n_total = len(ds)

    # 仅两种取样方式
    if SAMPLE_MODE == "first_n":
        # end = min(n_total, START_INDEX + int(NUM_SAMPLES))
        # indices = list(range(int(START_INDEX), int(end)))
        # 改为：从 [START_INDEX, n_total) 中随机无放回采样 NUM_SAMPLES 个
        candidate_indices = list(range(int(START_INDEX), int(n_total)))
        indices = random.sample(candidate_indices, int(NUM_SAMPLES))
    elif SAMPLE_MODE == "all":
        indices = list(range(int(START_INDEX), int(n_total)))
    else:
        raise ValueError(f"Unknown SAMPLE_MODE={SAMPLE_MODE}, must be 'first_n' or 'all'.")

    # labels 列表（尽量用数据集中的实际 label 拼写）
    labels = sorted({ds[i]["object_labels"] for i in range(n_total)})
    label_set = set(labels)

    _p(f"[INFO] Dataset len={n_total}, running {len(indices)} samples (SAMPLE_MODE={SAMPLE_MODE})",
       style="green" if _console is not None else None)
    _p(f"[INFO] Unique labels in dataset: {len(labels)}", style="green" if _console is not None else None)
    if len(labels) != 40:
        _p(f"[WARN] Unique label count != 40 (got {len(labels)}). Prompt will still use these labels.",
           style="yellow" if _console is not None else None)

    # ---- Qwen model & processor ----
    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        QWEN_MODEL_NAME_OR_PATH,
        dtype="auto",
        device_map="auto",
    )
    model.eval()
    processor = Qwen3OmniMoeProcessor.from_pretrained(QWEN_MODEL_NAME_OR_PATH)
    tokenizer = processor.tokenizer

    # 注册 <point> 为“单独 1 token”的 special token，并 resize embedding
    old_point_ids = tokenizer.encode(POINT_PLACEHOLDER, add_special_tokens=False)
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": [AddedToken(POINT_PLACEHOLDER, lstrip=False, rstrip=False)]}
    )
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        # 初始化新 token embedding：用旧分词的平均 embedding 做个稳定初始化
        try:
            point_id_new = int(tokenizer.convert_tokens_to_ids(POINT_PLACEHOLDER))
            emb_tmp = model.get_input_embeddings()
            if isinstance(old_point_ids, list) and len(old_point_ids) > 0:
                ids_t = torch.tensor(old_point_ids, device=emb_tmp.weight.device, dtype=torch.long)
                with torch.no_grad():
                    init_vec = emb_tmp.weight.data.index_select(0, ids_t).mean(dim=0)
                    emb_tmp.weight.data[point_id_new].copy_(init_vec)
        except Exception:
            pass

    point_token_id = tokenizer.convert_tokens_to_ids(POINT_PLACEHOLDER)
    _p(
        f"[INFO] Registered placeholder token. token='{POINT_PLACEHOLDER}', id={point_token_id}, added={num_added}, vocab={len(tokenizer)}",
        style="green" if _console is not None else None,
    )

    emb_layer = model.get_input_embeddings()
    emb_device = emb_layer.weight.device
    emb_dtype = emb_layer.weight.dtype
    llm_hidden = int(emb_layer.weight.shape[1])
    _p(f"[INFO] LLM embedding device={emb_device}, dtype={emb_dtype}, hidden={llm_hidden}",
       style="green" if _console is not None else None)

    # ---- AE ----
    ae, ae_cfg = load_ae_from_ckpt(AE_CKPT_PATH, device=emb_device, dtype=emb_dtype)

    ae_d_text_in = int(ae_cfg.get("model", {}).get("d_text_in", -1))
    if ae_d_text_in != -1 and ae_d_text_in != llm_hidden:
        _p(
            f"[WARN] Dimension mismatch: AE d_text_in={ae_d_text_in} vs LLM hidden={llm_hidden}. "
            f"Injection may fail unless you have a projection.",
            style="yellow" if _console is not None else None,
        )

    # dummy total len 决策
    default_len = max(int(FIXED_NUM_POINT_TOKENS), 16)
    inferred_len = infer_ae_text_total_len_from_cfg(ae_cfg, default_len=default_len)
    dummy_total_len = int(AE_DUMMY_TEXT_TOTAL_LEN) if AE_DUMMY_TEXT_TOTAL_LEN is not None else int(inferred_len)
    if dummy_total_len < FIXED_NUM_POINT_TOKENS:
        dummy_total_len = int(FIXED_NUM_POINT_TOKENS)

    _p(f"[INFO] FIXED_NUM_POINT_TOKENS={FIXED_NUM_POINT_TOKENS}, AE_DUMMY_TEXT_TOTAL_LEN={dummy_total_len}",
       style="green" if _console is not None else None)

    # ---- token proxy 预计算 ----
    emb_weight = emb_layer.weight
    emb_weight_norm = None
    if DEBUG_SHOW_INJECTION_DEBUG and DEBUG_SHOW_TOPK_TOKEN_PROXIES:
        try:
            emb_weight_norm = emb_weight.norm(dim=1).clamp_min(1e-6)
        except Exception:
            emb_weight_norm = None

    # ---- 输出文件 ----
    ensure_dir(OUTPUT_DIR)
    f_main = open_jsonl(OUT_MAIN_JSONL, overwrite=OVERWRITE_OUTPUT_FILES)
    f_base = open_jsonl(OUT_BASELINE_JSONL, overwrite=OVERWRITE_OUTPUT_FILES)
    _p(f"[INFO] Writing main predictions to: {OUT_MAIN_JSONL}", style="green" if _console is not None else None)
    _p(f"[INFO] Writing baseline (old caption prompt) to: {OUT_BASELINE_JSONL}", style="green" if _console is not None else None)

    correct = 0
    parsed_count = 0

    for run_i, ds_idx in enumerate(indices):
        item = ds[ds_idx]
        gt_label = str(item["object_labels"])
        point_tokens = item["point_tokens"]  # (T,D) on CPU

        # 1) AE: point -> text token embeddings (K,H)
        try:
            pred_seq, pred_all, mask_all = ae_point_to_text_token_embeddings_fixed(
                ae=ae,
                point_tokens=point_tokens,
                llm_hidden=llm_hidden,
                fixed_k=int(FIXED_NUM_POINT_TOKENS),
                dummy_total_len=int(dummy_total_len),
                device=emb_device,
                dtype=emb_dtype,
            )
        except Exception as e:
            # 写入失败记录，继续
            rec_fail = {
                "ds_idx": int(ds_idx),
                "run_i": int(run_i),
                "gt_label": gt_label,
                "error": f"AE_forward_failed: {repr(e)}",
            }
            write_jsonl_line(f_main, rec_fail)
            write_jsonl_line(f_base, {**rec_fail, "error": f"AE_forward_failed: {repr(e)}"})
            continue

        K = int(pred_seq.shape[0])

        # 2) 主任务 prompt（分类）
        user_cls = build_user_prompt_with_points_cls(labels=labels, k=K, placeholder=POINT_PLACEHOLDER)
        inputs_cls = build_qwen_inputs(processor, SYSTEM_PROMPT_CLS, user_cls)
        inputs_cls = {k: v.to(emb_device) if isinstance(v, torch.Tensor) else v for k, v in inputs_cls.items()}
        input_ids_cls = inputs_cls["input_ids"]

        with torch.no_grad():
            base_embeds_cls = emb_layer(input_ids_cls)

        # 注入
        try:
            injected_embeds_cls, spans_cls, patterns_cls = inject_point_embeddings_into_inputs_embeds(
                tokenizer=tokenizer,
                input_ids=input_ids_cls,
                inputs_embeds=base_embeds_cls,
                point_embeddings=pred_seq,
                point_placeholder=POINT_PLACEHOLDER,
            )
        except Exception as e:
            rec_fail = {
                "ds_idx": int(ds_idx),
                "run_i": int(run_i),
                "gt_label": gt_label,
                "error": f"injection_failed: {repr(e)}",
            }
            write_jsonl_line(f_main, rec_fail)
            # baseline 也写一下失败原因（因为同一套注入，极可能也失败）
            write_jsonl_line(f_base, {**rec_fail, "error": f"injection_failed: {repr(e)}"})
            continue

        # 生成（分类输出）
        try:
            pred_text = generate_with_inputs_embeds(
                model=model,
                processor=processor,
                inputs=inputs_cls,
                inputs_embeds=injected_embeds_cls,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
        except Exception as e:
            # fallback
            try:
                pred_text = greedy_fallback_generate(
                    model=model,
                    tokenizer=tokenizer,
                    inputs=inputs_cls,
                    inputs_embeds=injected_embeds_cls,
                    max_new_tokens=MAX_NEW_TOKENS,
                )
                pred_text = pred_text + f"  [fallback_used_due_to_generate_error={repr(e)}]"
            except Exception as e2:
                pred_text = f"[generation_failed: {repr(e)}; fallback_failed: {repr(e2)}]"

        pred_label = parse_modelnet40_label(pred_text, label_set=label_set)
        is_correct = None
        if pred_label is not None:
            parsed_count += 1
            is_correct = bool(pred_label == gt_label)
            if is_correct:
                correct += 1

        # 写主结果
        rec_main = {
            "ds_idx": int(ds_idx),
            "run_i": int(run_i),
            "gt_label": gt_label,
            "pred_text": pred_text,
            "pred_label_parsed": pred_label,
            "correct": is_correct,
            "fixed_num_point_tokens": int(FIXED_NUM_POINT_TOKENS),
        }
        write_jsonl_line(f_main, rec_main)

        # 3) baseline：旧 caption prompt（同样注入 pred_seq）
        user_cap = build_user_prompt_with_points_old_caption(k=K, placeholder=POINT_PLACEHOLDER)
        inputs_cap = build_qwen_inputs(processor, SYSTEM_PROMPT_OLD_CAPTION, user_cap)
        inputs_cap = {k: v.to(emb_device) if isinstance(v, torch.Tensor) else v for k, v in inputs_cap.items()}
        input_ids_cap = inputs_cap["input_ids"]
        with torch.no_grad():
            base_embeds_cap = emb_layer(input_ids_cap)

        try:
            injected_embeds_cap, spans_cap, patterns_cap = inject_point_embeddings_into_inputs_embeds(
                tokenizer=tokenizer,
                input_ids=input_ids_cap,
                inputs_embeds=base_embeds_cap,
                point_embeddings=pred_seq,
                point_placeholder=POINT_PLACEHOLDER,
            )
            cap_text = generate_with_inputs_embeds(
                model=model,
                processor=processor,
                inputs=inputs_cap,
                inputs_embeds=injected_embeds_cap,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
        except Exception as e:
            # baseline 不强求 fallback，简单记录
            cap_text = f"[baseline_caption_failed: {repr(e)}]"
            spans_cap, patterns_cap = [], []

        rec_base = {
            "ds_idx": int(ds_idx),
            "run_i": int(run_i),
            "gt_label": gt_label,
            "baseline_old_prompt_output": cap_text,
            "fixed_num_point_tokens": int(FIXED_NUM_POINT_TOKENS),
        }
        write_jsonl_line(f_base, rec_base)

        # 4) 打印（按类似旧脚本风格）
        do_print = VERBOSE_PRINT_PER_SAMPLE and (run_i % int(PRINT_EVERY) == 0)
        if do_print:
            _rule(f"Sample run_i={run_i}  ds_idx={ds_idx}  gt={gt_label}")

            if _console is not None:
                meta = Table(title="Meta", box=box.SIMPLE, show_header=False)
                meta.add_row("SAMPLE_MODE", str(SAMPLE_MODE))
                meta.add_row("ds_idx", str(ds_idx))
                meta.add_row("gt_label", gt_label)
                meta.add_row("FIXED_NUM_POINT_TOKENS", str(FIXED_NUM_POINT_TOKENS))
                meta.add_row("parsed_pred_label", str(pred_label))
                meta.add_row("correct", str(is_correct))
                if parsed_count > 0:
                    meta.add_row("running_acc(parsed_only)", f"{correct}/{parsed_count} = {correct/parsed_count:.4f}")
                _console.print(meta)
            else:
                _p(f"[INFO] gt={gt_label}  pred_label={pred_label}  correct={is_correct}")

            # 注入定位 & token proxy（只用分类 prompt 的 spans_cls）
            if DEBUG_SHOW_INJECTION_DEBUG:
                if _console is not None:
                    dbg = Table(title="Injection Locate Result (CLS prompt)", box=box.SIMPLE, show_header=False)
                    dbg.add_row("spans(len)", f"{len(spans_cls)}  {spans_cls[:10]}{' ...' if len(spans_cls) > 10 else ''}")
                    dbg.add_row("patterns", Pretty(patterns_cls))
                    _console.print(dbg)
                else:
                    _p(f"[DEBUG] spans(len={len(spans_cls)}): {spans_cls[:10]}{' ...' if len(spans_cls) > 10 else ''}")
                    _p(f"[DEBUG] patterns: {patterns_cls}")

                # 只对前 N 条做 token proxy（节省算力）
                allow_proxy = DEBUG_SHOW_TOPK_TOKEN_PROXIES
                if DEBUG_ONLY_FIRST_N_FOR_TOKEN_PROXY is not None and run_i >= int(DEBUG_ONLY_FIRST_N_FOR_TOKEN_PROXY):
                    allow_proxy = False

                render_injection_debug(
                    label="AE(text_recon_from_point) -> CLS prompt",
                    tokenizer=tokenizer,
                    input_ids=input_ids_cls,
                    base_embeds=base_embeds_cls,
                    injected_embeds=injected_embeds_cls,
                    spans=spans_cls,
                    payload=pred_seq,
                    emb_weight=emb_weight if allow_proxy else None,
                    emb_weight_norm=emb_weight_norm if allow_proxy else None,
                    max_show_spans=DEBUG_MAX_SHOW_SPANS,
                    show_topk=allow_proxy,
                    topk=DEBUG_TOPK_TOKENS,
                )

            # prompts
            if _console is not None:
                _console.print(Panel(user_cls, title="User Prompt (CLS)", border_style="cyan", expand=False))
                _console.print(Panel(user_cap, title="User Prompt (Baseline old caption)", border_style="magenta", expand=False))
            else:
                _p("User Prompt (CLS):\n" + user_cls)
                _p("User Prompt (Baseline old caption):\n" + user_cap)

            # outputs
            if _console is not None:
                out_tb = Table(title="Outputs", box=box.SIMPLE_HEAVY)
                out_tb.add_column("Variant", style="bold", justify="left")
                out_tb.add_column("Text", overflow="fold", justify="left")
                out_tb.add_row("Pred (CLS)", pred_text)
                out_tb.add_row("Baseline (old caption prompt)", cap_text)
                out_tb.add_row("GT label", gt_label)
                _console.print(out_tb)
            else:
                _p("Pred (CLS):\n" + pred_text)
                _p("Baseline (old caption prompt):\n" + cap_text)
                _p("GT label:\n" + gt_label)

    # ---- done ----
    f_main.close()
    f_base.close()

    _rule("Done")
    if parsed_count > 0:
        _p(f"[INFO] Parsed predictions: {parsed_count}, correct: {correct}, acc: {correct/parsed_count:.6f}",
           style="green" if _console is not None else None)
    else:
        _p("[INFO] No parsed labels (model outputs may not match label list).", style="yellow" if _console is not None else None)
    _p(f"[INFO] Saved:\n  - {OUT_MAIN_JSONL}\n  - {OUT_BASELINE_JSONL}",
       style="green" if _console is not None else None)


if __name__ == "__main__":
    main()
