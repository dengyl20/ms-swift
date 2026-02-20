# -*- coding: utf-8 -*-
"""
infer_modelnet40_point_mlp_qwen3_omni.py

用途：
- 在 ModelNet40 特征数据集（只含 point_tokens + object_labels）上做推理
- 用你从 baseline LLM checkpoint 提取出来的 point_projector（mlp2x_gelu）将 point_tokens 映射为 “文本 token embedding”
- 将 embedding 注入到 Qwen3-Omni 的 <point> 占位 token 位置
- 任务：ModelNet40 40 分类（输出一个 label）
- 同样额外跑一个“旧 caption prompt baseline”（可选，输出另存一份文件）——逻辑与 AE 推理脚本一致

多进程/多 rank 推理：
- 单卡放不下模型，2 卡放一个推理副本
- 单节点 8 卡 => 4 个 rank 并行推理（每 rank 2 卡）

推荐启动方式（单节点）：
  GPUS_PER_RANK=2 torchrun --nproc_per_node=4 infer_modelnet40_point_mlp_qwen3_omni.py

注意：
- 本脚本不使用 argparse，全部全局变量配置（对齐你 AE 脚本）。
"""

from __future__ import annotations

# ============================================================
# 重要：torchrun 多进程并行时，每个 rank 只“看到”自己的 2 张卡，
#      这样 Transformers 的 device_map="auto" 才会在这 2 张卡内做模型并行。
#      必须在 import torch/transformers 之前设置 CUDA_VISIBLE_DEVICES。
# ============================================================

import os


def _auto_set_cuda_visible_devices_for_mp() -> None:
    """
    torchrun 会设置 RANK/WORLD_SIZE/LOCAL_RANK。
    - 默认每个 rank 使用 2 张卡（可用环境变量 GPUS_PER_RANK 覆盖）。
    - 若外部已设置 CUDA_VISIBLE_DEVICES（例如调度器做了 GPU 过滤），则在该列表内再做切片。
    """
    if ("RANK" not in os.environ) or ("WORLD_SIZE" not in os.environ):
        return

    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except Exception:
        world_size = 1

    if world_size <= 1:
        return

    try:
        gpus_per_rank = int(os.environ.get("GPUS_PER_RANK", "2"))
    except Exception:
        gpus_per_rank = 2
    if gpus_per_rank <= 0:
        gpus_per_rank = 2

    try:
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    except Exception:
        local_rank = 0

    start = local_rank * gpus_per_rank
    end = start + gpus_per_rank

    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    vis_list = [x.strip() for x in vis.split(",") if x.strip() != ""]
    if len(vis_list) >= end:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(vis_list[start:end])
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(start, end))


_auto_set_cuda_visible_devices_for_mp()

import json
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

# ====== OFFLINE MODE（按你旧脚本保持）======
os.environ["HF_HUB_OFFLINE"] = "1"

# ====== 你的数据集（保持与 AE 推理脚本一致的 import 路径）======
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


def _get_dist_env_rank_world() -> Tuple[int, int]:
    try:
        r = int(os.environ.get("RANK", "0"))
    except Exception:
        r = 0
    try:
        w = int(os.environ.get("WORLD_SIZE", "1"))
    except Exception:
        w = 1
    return r, w


def _is_main_process() -> bool:
    r, w = _get_dist_env_rank_world()
    return (w <= 1) or (r == 0)


def _p(obj: Any = "", **kwargs: Any) -> None:
    # 多进程时默认只让 rank0 打印（避免刷屏）；你也可以把 PRINT_ONLY_RANK0 设为 False
    if globals().get("PRINT_ONLY_RANK0", True):
        if not _is_main_process():
            return

    if _console is not None:
        _console.print(obj, **kwargs)
    else:
        print(obj)


def _rule(title: str = "") -> None:
    if globals().get("PRINT_ONLY_RANK0", True):
        if not _is_main_process():
            return

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

# ---------- Multi-process / distributed ----------
GPUS_PER_RANK = int(os.environ.get("GPUS_PER_RANK", "2"))
PRINT_ONLY_RANK0 = True
MERGE_RANK_SHARDS_TO_SINGLE_JSONL = True

# ---------- 必改：你提取出来的 projector checkpoint ----------
# 由 extract_point_projector_from_swift_ckpt.py 输出
PROJECTOR_CKPT_PATH = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/checkpoints_mlp/v1-20260218-055502/point_projector_finetuned_checkpoint.pt"

# ---------- 必改：ModelNet40 特征 pt ----------
MODELNET40_FEATURE_PT_PATH = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/PointLLM/PointLLM/modelnet40_gray_color.pt"

# ---------- Qwen3-Omni ----------
QWEN_MODEL_NAME_OR_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

# ---------- 取样方式（仅两种） ----------
SAMPLE_MODE = "all"  # "first_n" | "all"
START_INDEX = 0
NUM_SAMPLES = 10  # SAMPLE_MODE="first_n" 时生效

# ---------- 数据集 valid 过滤 ----------
REQUIRE_VALID = False

# ---------- 注入相关 ----------
POINT_PLACEHOLDER = "<point>"

# 固定注入 token 数（与 AE 脚本一致）
FIXED_NUM_POINT_TOKENS = 8

# ---------- 生成超参数 ----------
MAX_NEW_TOKENS = 64
DO_SAMPLE = False
TEMPERATURE = 0.7
TOP_P = 0.9

# ---------- 输出文件 ----------
OUTPUT_DIR = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/swift/point_cloud/stage1/src/eval/results/modelnet40_mlp_projector"
OUT_MAIN_JSONL = os.path.join(OUTPUT_DIR, "predictions_modelnet40_cls.jsonl")
OUT_BASELINE_JSONL = os.path.join(OUTPUT_DIR, "predictions_baseline_caption_prompt.jsonl")
OUT_METRICS_JSON = os.path.join(OUTPUT_DIR, "metrics_modelnet40_cls_summary.json")
OVERWRITE_OUTPUT_FILES = True  # True: 覆盖写；False: 追加

# ---------- 打印/调试 ----------
VERBOSE_PRINT_PER_SAMPLE = True
PRINT_EVERY = 1

DEBUG_SHOW_INJECTION_DEBUG = True
DEBUG_MAX_SHOW_SPANS = 12
DEBUG_SHOW_TOPK_TOKEN_PROXIES = True
DEBUG_TOPK_TOKENS = 6
DEBUG_ONLY_FIRST_N_FOR_TOKEN_PROXY: Optional[int] = 10

# ---------- 随机种子 ----------
SEED = 42


# ============================================================
# 1) Prompt（主任务：40 分类；baseline：旧 caption prompt）
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
# 2) 工具函数（随机种子 / device / prompt / parse label / distributed）
# ============================================================

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed_if_needed() -> Tuple[int, int, bool]:
    """
    返回 (rank, world_size, is_distributed)。
    若由 torchrun 启动，则初始化 process group（用于 barrier / all_reduce）。
    注意：这里的分布式仅用于并行推理与统计汇总，不做 DDP 参数同步。
    """
    rank, world_size = _get_dist_env_rank_world()
    is_distributed = world_size > 1

    if is_distributed and (not dist.is_initialized()):
        backend = "nccl" if torch.cuda.is_available() else "gloo"

        # 每个进程只“看到”自己的 2 张卡后，device index 会从 0 开始重映射；
        # 这里统一用 cuda:0 做通信设备即可（对应每个 rank 的第一张可见卡）。
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

        dist.init_process_group(backend=backend, init_method="env://")

    return rank, world_size, is_distributed


def dist_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def to_device_dtype(x: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if x.dtype in (torch.bool, torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8):
        return x.to(device=device)
    return x.to(device=device, dtype=dtype)


def build_user_prompt_with_points_cls(labels: List[str], k: int, placeholder: str = "<point>") -> str:
    k = max(1, int(k))
    point_block = " ".join([placeholder] * k)
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


def _normalize_pred_text_for_label(text: str) -> str:
    t = str(text).strip().lower()
    t = t.strip().strip('"').strip("'").strip("`").strip()
    return t


def is_strict_one_label_answer(text: str, label_set: set) -> Tuple[bool, Optional[str]]:
    if text is None:
        return False, None
    t = _normalize_pred_text_for_label(text)
    if t in label_set:
        return True, t
    return False, None


def parse_modelnet40_label(text: str, label_set: set) -> Optional[str]:
    if text is None:
        return None
    t = _normalize_pred_text_for_label(text)
    if t in label_set:
        return t

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
# 3) 加载 projector ckpt
# ============================================================

def _build_mlp2x_gelu(in_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim, bias=True),
        nn.GELU(),
        nn.Linear(out_dim, out_dim, bias=True),
    )


def _infer_in_out_dim_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    # 兼容你抽取脚本的默认格式（'0.weight'）
    if "0.weight" in state_dict:
        w0 = state_dict["0.weight"]
        if w0.dim() != 2:
            raise RuntimeError(f"Expected '0.weight' to be 2D, got shape={tuple(w0.shape)}")
        out_dim = int(w0.shape[0])
        in_dim = int(w0.shape[1])
        return in_dim, out_dim

    # fallback：找一个二维 weight
    cand = [(k, v) for k, v in state_dict.items() if isinstance(v, torch.Tensor) and v.dim() == 2 and k.endswith("weight")]
    if not cand:
        raise RuntimeError("Cannot infer dims from projector state_dict: no 2D weight found.")
    k, w = cand[0]
    return int(w.shape[1]), int(w.shape[0])


def load_projector_from_ckpt(
    ckpt_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    读取 extract_point_projector_from_swift_ckpt.py 输出的 ckpt：
      ckpt = {"cfg": {"model": {...}}, "model": state_dict, ...}
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", {}) or {}
    model_cfg = cfg.get("model", {}) or {}

    state_dict = ckpt.get("model", None)
    if state_dict is None:
        # 兼容其它命名
        state_dict = ckpt.get("state_dict", None)
    if state_dict is None or not isinstance(state_dict, dict):
        raise RuntimeError(f"Projector ckpt missing 'model' (state_dict dict): {ckpt_path}")

    arch = str(model_cfg.get("arch", "mlp2x_gelu"))
    in_dim = model_cfg.get("in_dim", None)
    out_dim = model_cfg.get("out_dim", None)

    if in_dim is None or out_dim is None:
        in_dim_i, out_dim_i = _infer_in_out_dim_from_state_dict(state_dict)
        in_dim = in_dim_i
        out_dim = out_dim_i

    in_dim = int(in_dim)
    out_dim = int(out_dim)

    if arch != "mlp2x_gelu":
        # 目前 baseline 的 pc_model.py 明确是 mlp2x_gelu；若你未来改结构，可在此扩展
        _p(f"[WARN] cfg.model.arch={arch} (expected 'mlp2x_gelu'). Will still build mlp2x_gelu and try to load.",
           style="yellow" if _console is not None else None)

    proj = _build_mlp2x_gelu(in_dim, out_dim)
    missing, unexpected = proj.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        # 推理阶段更倾向于直接报错（避免 silent wrong）
        raise RuntimeError(
            "Projector state_dict mismatch when loading.\n"
            f"missing: {missing}\n"
            f"unexpected: {unexpected}\n"
            f"ckpt={ckpt_path}\n"
            "If you changed projector architecture, update loader accordingly."
        )

    proj.eval()
    proj.to(device=device, dtype=dtype)
    return proj, cfg


@torch.no_grad()
def projector_point_to_text_token_embeddings_fixed(
    *,
    projector: nn.Module,
    point_tokens: torch.Tensor,   # (T,D)
    fixed_k: int,                 # K
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    baseline projector 推理：
      (T,D) -> (T,H) -> 取前 K；若 T < K 则 pad（复制最后一个 token，保持与 template 防御一致）
    返回：pred_seq (K,H)
    """
    fixed_k = int(fixed_k)
    if fixed_k <= 0:
        raise ValueError(f"fixed_k must be > 0, got {fixed_k}")

    pt = to_device_dtype(point_tokens.unsqueeze(0), device, dtype)  # (1,T,D)
    projected = projector(pt)  # (1,T,H) (期望)

    if projected.dim() != 3:
        raise RuntimeError(f"projector output must be (B,T,H), got shape={tuple(projected.shape)}")

    seq = projected[0]  # (T,H)
    T = int(seq.shape[0])
    H = int(seq.shape[1])

    if T < fixed_k:
        if T == 0:
            seq = torch.zeros((fixed_k, H), device=device, dtype=dtype)
        else:
            pad = seq[-1:].expand(fixed_k - T, -1)
            seq = torch.cat([seq, pad], dim=0)

    return seq[:fixed_k]


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
# 5) I/O：写 JSONL（与 AE 脚本一致）
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def open_jsonl(path: str, overwrite: bool):
    mode = "w" if overwrite else "a"
    return open(path, mode, encoding="utf-8")


def write_jsonl_line(f, obj: Dict[str, Any]) -> None:
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    f.flush()


def add_rank_suffix(path: str, rank: int) -> str:
    base, ext = os.path.splitext(path)
    return f"{base}.rank{rank}{ext}"


def merge_jsonl_shards(
    *,
    shard_paths: List[str],
    out_path: str,
    overwrite: bool,
    sort_key: Tuple[str, str] = ("run_i", "ds_idx"),
) -> None:
    records: List[Dict[str, Any]] = []
    for sp in shard_paths:
        if not os.path.exists(sp):
            continue
        with open(sp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    obj = {"_raw": line}
                records.append(obj)

    k1, k2 = sort_key

    def _key(o: Dict[str, Any]) -> Tuple[int, int]:
        a = o.get(k1, 10**18)
        b = o.get(k2, 10**18)
        try:
            a = int(a)
        except Exception:
            a = 10**18
        try:
            b = int(b)
        except Exception:
            b = 10**18
        return a, b

    try:
        records.sort(key=_key)
    except Exception:
        pass

    mode = "w" if overwrite else "a"
    with open(out_path, mode, encoding="utf-8") as fo:
        for obj in records:
            fo.write(json.dumps(obj, ensure_ascii=False) + "\n")
        fo.flush()


# ============================================================
# 6) main
# ============================================================

def main() -> None:
    rank, world_size, is_distributed = init_distributed_if_needed()

    set_global_seed(SEED)
    torch.set_grad_enabled(False)

    if is_distributed and _is_main_process():
        _p(
            f"[INFO] Distributed enabled: rank={rank}, world_size={world_size}, "
            f"GPUS_PER_RANK(env)={os.environ.get('GPUS_PER_RANK', '2')}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
            style="green" if _console is not None else None,
        )

    # ---- dataset ----
    ds = ModelNet40PointTokenDataset(MODELNET40_FEATURE_PT_PATH, require_valid=REQUIRE_VALID)
    n_total = len(ds)

    if SAMPLE_MODE == "first_n":
        end = min(n_total, START_INDEX + int(NUM_SAMPLES))
        indices = list(range(int(START_INDEX), int(end)))
    elif SAMPLE_MODE == "all":
        indices = list(range(int(START_INDEX), int(n_total)))
    else:
        raise ValueError(f"Unknown SAMPLE_MODE={SAMPLE_MODE}, must be 'first_n' or 'all'.")

    labels = sorted({ds[i]["object_labels"] for i in range(n_total)})
    label_set = set(labels)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    if _is_main_process():
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
    if _is_main_process():
        _p(
            f"[INFO] Registered placeholder token. token='{POINT_PLACEHOLDER}', id={point_token_id}, added={num_added}, vocab={len(tokenizer)}",
            style="green" if _console is not None else None,
        )

    emb_layer = model.get_input_embeddings()
    emb_device = emb_layer.weight.device
    emb_dtype = emb_layer.weight.dtype
    llm_hidden = int(emb_layer.weight.shape[1])
    if _is_main_process():
        _p(f"[INFO] LLM embedding device={emb_device}, dtype={emb_dtype}, hidden={llm_hidden}",
           style="green" if _console is not None else None)

    # ---- projector ----
    projector, proj_cfg = load_projector_from_ckpt(PROJECTOR_CKPT_PATH, device=emb_device, dtype=emb_dtype)

    proj_model_cfg = (proj_cfg.get("model", {}) if isinstance(proj_cfg, dict) else {}) or {}
    proj_out_dim = int(proj_model_cfg.get("out_dim", llm_hidden))
    if proj_out_dim != llm_hidden and _is_main_process():
        _p(
            f"[WARN] Dimension mismatch: projector out_dim={proj_out_dim} vs LLM hidden={llm_hidden}. "
            f"Injection will likely fail unless you have another projection.",
            style="yellow" if _console is not None else None,
        )

    if _is_main_process():
        _p(f"[INFO] FIXED_NUM_POINT_TOKENS={FIXED_NUM_POINT_TOKENS}",
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

    out_main_rank = add_rank_suffix(OUT_MAIN_JSONL, rank) if is_distributed else OUT_MAIN_JSONL
    out_base_rank = add_rank_suffix(OUT_BASELINE_JSONL, rank) if is_distributed else OUT_BASELINE_JSONL

    f_main = open_jsonl(out_main_rank, overwrite=OVERWRITE_OUTPUT_FILES)
    f_base = open_jsonl(out_base_rank, overwrite=OVERWRITE_OUTPUT_FILES)

    if _is_main_process():
        if is_distributed:
            _p(f"[INFO] Writing main predictions shard to: {out_main_rank}", style="green" if _console is not None else None)
            _p(f"[INFO] Writing baseline shard to: {out_base_rank}", style="green" if _console is not None else None)
            _p(f"[INFO] (rank0) will merge shards to:\n  - {OUT_MAIN_JSONL}\n  - {OUT_BASELINE_JSONL}", style="green" if _console is not None else None)
        else:
            _p(f"[INFO] Writing main predictions to: {OUT_MAIN_JSONL}", style="green" if _console is not None else None)
            _p(f"[INFO] Writing baseline (old caption prompt) to: {OUT_BASELINE_JSONL}", style="green" if _console is not None else None)

    # ---- 统计（本 rank）----
    total_local = 0
    parsed_local = 0
    correct_local = 0
    format_ok_local = 0

    per_label_total = [0 for _ in labels]
    per_label_correct = [0 for _ in labels]
    per_label_parsed = [0 for _ in labels]
    per_label_format_ok = [0 for _ in labels]

    correct = 0
    parsed_count = 0

    for run_i, ds_idx in enumerate(indices):
        if is_distributed and ((run_i % world_size) != rank):
            continue

        total_local += 1

        item = ds[ds_idx]
        gt_label = str(item["object_labels"])
        point_tokens = item["point_tokens"]  # (T,D) on CPU

        gt_li = label_to_idx.get(gt_label, None)
        if gt_li is not None:
            per_label_total[gt_li] += 1

        # 1) projector: point -> token embeddings (K,H)
        try:
            pred_seq = projector_point_to_text_token_embeddings_fixed(
                projector=projector,
                point_tokens=point_tokens,
                fixed_k=int(FIXED_NUM_POINT_TOKENS),
                device=emb_device,
                dtype=emb_dtype,
            )
        except Exception as e:
            rec_fail = {
                "ds_idx": int(ds_idx),
                "run_i": int(run_i),
                "gt_label": gt_label,
                "error": f"projector_forward_failed: {repr(e)}",
            }
            write_jsonl_line(f_main, rec_fail)
            write_jsonl_line(f_base, {**rec_fail, "error": f"projector_forward_failed: {repr(e)}"})
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

        # ---- 统计：format / parsed / correct ----
        fmt_ok, strict_label = is_strict_one_label_answer(pred_text, label_set=label_set)
        if fmt_ok:
            format_ok_local += 1
            if gt_li is not None:
                per_label_format_ok[gt_li] += 1

        pred_label = parse_modelnet40_label(pred_text, label_set=label_set)
        is_correct = None
        if pred_label is not None:
            parsed_local += 1
            if gt_li is not None:
                per_label_parsed[gt_li] += 1

            parsed_count += 1
            is_correct = bool(pred_label == gt_label)
            if is_correct:
                correct_local += 1
                if gt_li is not None:
                    per_label_correct[gt_li] += 1
                correct += 1

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
                meta.add_row("format_ok", str(fmt_ok))
                if parsed_count > 0:
                    meta.add_row("running_acc(parsed_only)", f"{correct}/{parsed_count} = {correct/parsed_count:.4f}")
                _console.print(meta)
            else:
                _p(f"[INFO] gt={gt_label}  pred_label={pred_label}  correct={is_correct}  format_ok={fmt_ok}")

            if DEBUG_SHOW_INJECTION_DEBUG:
                if _console is not None:
                    dbg = Table(title="Injection Locate Result (CLS prompt)", box=box.SIMPLE, show_header=False)
                    dbg.add_row("spans(len)", f"{len(spans_cls)}  {spans_cls[:10]}{' ...' if len(spans_cls) > 10 else ''}")
                    dbg.add_row("patterns", Pretty(patterns_cls))
                    _console.print(dbg)
                else:
                    _p(f"[DEBUG] spans(len={len(spans_cls)}): {spans_cls[:10]}{' ...' if len(spans_cls) > 10 else ''}")
                    _p(f"[DEBUG] patterns: {patterns_cls}")

                allow_proxy = DEBUG_SHOW_TOPK_TOKEN_PROXIES
                if DEBUG_ONLY_FIRST_N_FOR_TOKEN_PROXY is not None and run_i >= int(DEBUG_ONLY_FIRST_N_FOR_TOKEN_PROXY):
                    allow_proxy = False

                render_injection_debug(
                    label="Projector(mlp2x_gelu) -> CLS prompt",
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

            if _console is not None:
                _console.print(Panel(user_cls, title="User Prompt (CLS)", border_style="cyan", expand=False))
                _console.print(Panel(user_cap, title="User Prompt (Baseline old caption)", border_style="magenta", expand=False))
            else:
                _p("User Prompt (CLS):\n" + user_cls)
                _p("User Prompt (Baseline old caption):\n" + user_cap)

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

    f_main.close()
    f_base.close()

    # ---- 汇总统计（跨 rank）----
    total_all = total_local
    parsed_all = parsed_local
    correct_all = correct_local
    format_ok_all = format_ok_local

    per_label_total_all = per_label_total
    per_label_parsed_all = per_label_parsed
    per_label_correct_all = per_label_correct
    per_label_format_ok_all = per_label_format_ok

    if is_distributed and dist.is_initialized():
        stat_device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")

        vec = torch.tensor(
            [total_local, parsed_local, correct_local, format_ok_local],
            device=stat_device,
            dtype=torch.long,
        )
        dist.all_reduce(vec, op=dist.ReduceOp.SUM)
        total_all, parsed_all, correct_all, format_ok_all = [int(x.item()) for x in vec.detach().cpu()]

        def _all_reduce_list(xs: List[int]) -> List[int]:
            t = torch.tensor(xs, device=stat_device, dtype=torch.long)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            return [int(x) for x in t.detach().cpu().tolist()]

        per_label_total_all = _all_reduce_list(per_label_total)
        per_label_parsed_all = _all_reduce_list(per_label_parsed)
        per_label_correct_all = _all_reduce_list(per_label_correct)
        per_label_format_ok_all = _all_reduce_list(per_label_format_ok)

    dist_barrier()

    # ---- rank0 合并 shard ----
    if is_distributed and MERGE_RANK_SHARDS_TO_SINGLE_JSONL and _is_main_process():
        main_shards = [add_rank_suffix(OUT_MAIN_JSONL, r) for r in range(world_size)]
        base_shards = [add_rank_suffix(OUT_BASELINE_JSONL, r) for r in range(world_size)]

        merge_jsonl_shards(
            shard_paths=main_shards,
            out_path=OUT_MAIN_JSONL,
            overwrite=OVERWRITE_OUTPUT_FILES,
        )
        merge_jsonl_shards(
            shard_paths=base_shards,
            out_path=OUT_BASELINE_JSONL,
            overwrite=OVERWRITE_OUTPUT_FILES,
        )

        _p(f"[INFO] Merged main predictions to: {OUT_MAIN_JSONL}", style="green" if _console is not None else None)
        _p(f"[INFO] Merged baseline predictions to: {OUT_BASELINE_JSONL}", style="green" if _console is not None else None)

    # ---- rank0 输出统计 ----
    if _is_main_process():
        _rule("Done")

        total = int(total_all)
        parsed = int(parsed_all)
        correct_n = int(correct_all)
        fmt_n = int(format_ok_all)

        acc = (correct_n / total) if total > 0 else 0.0
        parsed_ratio = (parsed / total) if total > 0 else 0.0
        fmt_ratio = (fmt_n / total) if total > 0 else 0.0
        acc_parsed_only = (correct_n / parsed) if parsed > 0 else 0.0

        metrics: Dict[str, Any] = {
            "total_samples": total,
            "correct": correct_n,
            "accuracy": acc,
            "parsed": parsed,
            "parsed_ratio": parsed_ratio,
            "accuracy_parsed_only": acc_parsed_only,
            "format_ok": fmt_n,
            "format_ok_ratio": fmt_ratio,
            "labels": labels,
            "per_label": {},
        }

        for lab in labels:
            i = label_to_idx[lab]
            tot = int(per_label_total_all[i])
            cor = int(per_label_correct_all[i])
            par = int(per_label_parsed_all[i])
            fok = int(per_label_format_ok_all[i])

            metrics["per_label"][lab] = {
                "total": tot,
                "correct": cor,
                "accuracy": (cor / tot) if tot > 0 else None,
                "parsed": par,
                "parsed_ratio": (par / tot) if tot > 0 else None,
                "format_ok": fok,
                "format_ok_ratio": (fok / tot) if tot > 0 else None,
            }

        try:
            with open(OUT_METRICS_JSON, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            _p(f"[INFO] Saved metrics to: {OUT_METRICS_JSON}", style="green" if _console is not None else None)
        except Exception as e:
            _p(f"[WARN] Failed to write metrics json: {repr(e)}", style="yellow" if _console is not None else None)

        _p(
            f"[INFO] Total={total} | Correct={correct_n} | Acc={acc:.6f} | "
            f"FormatOK={fmt_n} ({fmt_ratio:.6f}) | Parsed={parsed} ({parsed_ratio:.6f}) | "
            f"Acc(parsed_only)={acc_parsed_only:.6f}",
            style="green" if _console is not None else None,
        )

        if _console is not None:
            tb = Table(title="Per-label Accuracy (counted over ALL samples; unparsed/errors => incorrect)", box=box.SIMPLE_HEAVY)
            tb.add_column("label", justify="left", style="bold")
            tb.add_column("total", justify="right")
            tb.add_column("correct", justify="right")
            tb.add_column("acc", justify="right")
            tb.add_column("format_ok", justify="right")
            tb.add_column("fmt%", justify="right")
            tb.add_column("parsed", justify="right")
            tb.add_column("parsed%", justify="right")

            for lab in labels:
                m = metrics["per_label"][lab]
                tot = m["total"]
                cor = m["correct"]
                acc_l = m["accuracy"]
                fok = m["format_ok"]
                fmt_l = m["format_ok_ratio"]
                par = m["parsed"]
                pr_l = m["parsed_ratio"]

                tb.add_row(
                    lab,
                    str(tot),
                    str(cor),
                    f"{acc_l:.4f}" if acc_l is not None else "n/a",
                    str(fok),
                    f"{fmt_l:.4f}" if fmt_l is not None else "n/a",
                    str(par),
                    f"{pr_l:.4f}" if pr_l is not None else "n/a",
                )
            _console.print(tb)
        else:
            _p("[INFO] Per-label accuracy:")
            for lab in labels:
                m = metrics["per_label"][lab]
                _p(
                    f"  {lab}: total={m['total']} correct={m['correct']} "
                    f"acc={m['accuracy'] if m['accuracy'] is not None else 'n/a'} "
                    f"format_ok={m['format_ok_ratio'] if m['format_ok_ratio'] is not None else 'n/a'}"
                )

        if is_distributed and MERGE_RANK_SHARDS_TO_SINGLE_JSONL:
            _p(f"[INFO] Saved:\n  - {OUT_MAIN_JSONL}\n  - {OUT_BASELINE_JSONL}\n  - {OUT_METRICS_JSON}",
               style="green" if _console is not None else None)
        else:
            _p(f"[INFO] Saved:\n  - {out_main_rank}\n  - {out_base_rank}\n  - {OUT_METRICS_JSON}",
               style="green" if _console is not None else None)

    if dist.is_available() and dist.is_initialized():
        dist_barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
