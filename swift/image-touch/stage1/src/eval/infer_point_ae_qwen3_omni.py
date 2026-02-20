# -*- coding: utf-8 -*-
"""
infer_point_ae_qwen3_omni.py

目标：
- 用你训练好的 UnifiedPointTextAE：point_tokens -> text feature（embedding）
- 将该 embedding 注入到 Qwen3-Omni 的输入中，替换 prompt 里 <point> 对应 token span 的 inputs_embeds
- 对若干条样本进行推理，打印模型输出与 GT（对话 JSON 中的 gpt value）

本版新增（只改与 baseline/调试输出相关的部分，其它逻辑保持不动）：
1) 新增 baseline：直接注入 Ground Truth text embedding（来自 feature dataset 的 text_embeds/text_mask）
   - 用于验证：LLM 是否能“理解”GT embedding 并用它回答问题（从而分离“注入是否有效”与“点云->embedding 是否有效”）
2) 新增调试：可视化检查 text_recon_from_point 是否真正替换了 <point> span 的 inputs_embeds，并展示“变成了什么样”
   - 展示 span 对应 token、max|diff|、与 payload 的一致性、并可选做 embedding->token 的最近邻 proxy（Top-K）
3) 调试输出美化：优先使用 rich（若环境未安装 rich，会自动 fallback 到 print）
4) 其它不需要改动的无关部分不改动

【修改点（本轮）】
1) 新增 AE_TEXT_MASK_MODE：
   - "dataset"：使用 dataset 提供的 text_mask（变长）
   - "fixed_prefix"：固定 mask 前 N 位为 True（定长），用于生成定长的 AE 重构 token embedding
2) 移除 baseline：Baseline(no inject, prompt w/o <point>)
3) 新增定量评测：Pred(inject AE embedding) vs GT(answer) 的语义相似度（cosine over sentence embedding）
4) 将评测结果保存到指定路径文件（包含 baseline / pred / gt / metric）
5) 同时评测训练集子集 + 验证集（两者均由 dataset_info.yaml + 对话 json 定义，格式相同）

【注意】
- system prompt 中避免出现字面字符串 "<point>"，否则会被 tokenizer 识别为 special token 并影响注入定位。
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ====== 你的模块（保持与训练脚本一致的 import 路径）======
from swift.point_cloud.stage1.src.data.feature_dataset import ProcessedPointTextFeatureDataset
from swift.point_cloud.stage1.src.models.unified_ae import UnifiedPointTextAE

# ====== Qwen3-Omni (Transformers) ======
from transformers import (
    AddedToken,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeProcessor,
)

# ===== OFFLINE MODE =====
import os
os.environ["HF_HUB_OFFLINE"] = "1"

# ====== rich（可选）======
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.pretty import Pretty
    from rich.text import Text
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
    Text = None
    box = None


def _p(obj: Any = "", **kwargs: Any) -> None:
    """rich.print 的轻量封装；rich 不可用时 fallback 到 print。"""
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


# =========================
# 0) 超参数 & 路径（直接改这里）
# =========================

# 你训练好的 AE checkpoint（best.pt 或某个 epoch_xxx.pt）
AE_CKPT_PATH = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/checkpoints/v100-20260215-045054/point_ae_finetuned_checkpoint-1006.pt"

# ===== Train split =====
# stage1 提取 feature 的 dataset_info.yaml（里面记录 shards 路径、shape、dtype 等）
FEATURE_DATASET_INFO_YAML = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/data_features_cleaned_24/dataset_info.yaml"

# 你指定的原始对话 JSON（用于取 prompt 与 GT）
CONV_JSON_PATH = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/PointLLM/PointLLM/PointLLM_brief_description_660K_filtered.json"

# ===== Val split（需你填入验证集对应路径；格式与训练集相同）=====
# NOTE: 若你希望强制评测验证集，请确保这两个路径存在且可读
RUN_EVAL_VALIDATION = True
FEATURE_DATASET_INFO_YAML_VAL = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/data_features_val_200/dataset_info.yaml"  # e.g. "/path/to/val/dataset_info.yaml"
CONV_JSON_PATH_VAL = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/PointLLM/PointLLM/PointLLM_brief_description_val_200_GT.json"             # e.g. "/path/to/val/conversations.json"

# Qwen3-Omni 模型（用 Instruct 权重加载 Thinker text-only，省显存）
QWEN_MODEL_NAME_OR_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

# 只做定性：训练集跑多少条
NUM_SAMPLES = 100
# 验证集跑多少条（可单独设置）
NUM_SAMPLES_VAL = 100

# 从 feature dataset 扫描样本的起点与最大扫描量（防止一直找不到对应 object_id）
DATASET_SCAN_START = 0
MAX_DATASET_SCAN = 200000  # 你可按需调大/调小

# 随机种子（影响抽样）
SEED = 42

# prompt 里的占位符文本
POINT_PLACEHOLDER = "<point>"

# ===== AE text mask 策略（新增）=====
# "dataset": 使用 dataset 提供的 text_mask（变长，当前默认行为）
# "fixed_prefix": 固定前 N 个 token 为 True（定长），用于生成定长重构 embedding
AE_TEXT_MASK_MODE = "dataset"  # "dataset" | "fixed_prefix"
AE_FIXED_PREFIX_LEN = 4       # 仅当 AE_TEXT_MASK_MODE="fixed_prefix" 生效

# ===== 注入策略 =====
# "sequence": 不压缩，取 mask=True 的所有 token embedding，逐 token 注入（推荐与你问题对应）
# "pooled":   压缩成单向量，再覆盖 <point> 的 span（旧逻辑保留）
POINT_INJECT_MODE = "sequence"  # "sequence" | "pooled"

# sequence 模式下最多注入多少个 token（强烈建议限制，避免 K 过大导致推理很慢/爆显存）
MAX_POINT_TOKENS = 128

# pooled 模式下的 pooling（仅当 POINT_INJECT_MODE="pooled" 才用）
POINT_POOLING = "mean"  # "mean" | "first"

# ===== 生成超参数 =====
MAX_NEW_TOKENS = 256
DO_SAMPLE = False
TEMPERATURE = 0.7   # DO_SAMPLE=False 时 temperature 不生效
TOP_P = 0.9         # DO_SAMPLE=False 时 top_p 不生效

# ===== baseline：注入 Ground Truth text embedding =====
RUN_BASELINE_GT_TEXT_EMBED_INJECT = True

# ===== 调试与输出（新增）=====
# 是否显示更详细的注入 debug（spans/patterns + embed 替换校验）
DEBUG_SHOW_INJECTION_DEBUG = True

# 在 debug 表格里最多展示前多少个 span（避免输出过长）
DEBUG_MAX_SHOW_SPANS = 20

# 是否对注入向量做一个“最近邻 token proxy”（Top-K），帮助你直观看 embedding “像什么词”
DEBUG_SHOW_TOPK_TOKEN_PROXIES = True
DEBUG_TOPK_TOKENS = 6

# ===== 新增：语义相似度（定量评测）=====
# 用 Qwen 自身 encoder(hidden states) 做 mean pooling 后 cosine 作为语义相似度
SIM_MAX_LENGTH = 256

# ===== 新增：结果保存 =====
# 结果将保存为 JSONL（每行一个样本记录），并额外保存一个 summary JSON
EVAL_OUTPUT_DIR = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/swift/point_cloud/stage1/src/eval/outputs/objaverse"
EVAL_OUTPUT_JSONL = os.path.join(EVAL_OUTPUT_DIR, "infer_point_ae_qwen3_omni_eval.jsonl")
EVAL_OUTPUT_SUMMARY_JSON = os.path.join(EVAL_OUTPUT_DIR, "infer_point_ae_qwen3_omni_eval_summary.json")
EVAL_OUTPUT_OVERWRITE = True  # True: 覆盖；False: 追加

# system prompt（重新设计：避免出现字面 "<point>"）
SYSTEM_PROMPT = (
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

# =========================
# 1) 通用工具
# =========================

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device_dtype(x: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # 对 bool / long 不强转 dtype
    if x.dtype in (torch.bool, torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8):
        return x.to(device=device)
    return x.to(device=device, dtype=dtype)


def find_all_subsequence_positions(haystack: List[int], needle: List[int]) -> List[int]:
    """
    在 haystack 中查找 needle 子序列出现的所有起始位置（允许多次出现）。
    """
    if len(needle) == 0 or len(haystack) < len(needle):
        return []
    out = []
    n = len(needle)
    # 朴素匹配足够（prompt 很短）
    for i in range(0, len(haystack) - n + 1):
        if haystack[i:i+n] == needle:
            out.append(i)
    return out


def expand_point_placeholders(user_text: str, k: int, placeholder: str = "<point>") -> str:
    """
    将 user_text 中首次出现的 <point> 替换为 k 个 <point>（用空格分隔）。
    如果 user_text 不含 <point>，则在开头追加一段。
    """
    k = int(k)
    if k <= 1:
        return user_text

    block = " ".join([placeholder] * k)
    if placeholder in user_text:
        return user_text.replace(placeholder, block, 1)
    # fallback：没有 placeholder 的情况
    return block + "\n" + user_text


# ===== 新增：移除文本中所有 <point>（用于抽取“纯问题”）=====
def strip_all_point_placeholders(text: str, placeholder: str = "<point>") -> str:
    """
    删除 text 中所有 placeholder 字面串，并做轻量清洗：
    - 去掉仅包含 placeholder 的行
    - 合并多余空白
    """
    lines: List[str] = []
    for line in str(text).splitlines():
        line2 = line.replace(placeholder, " ")
        line2 = " ".join(line2.split())
        if line2.strip() != "":
            lines.append(line2)
    return "\n".join(lines).strip()


# ===== 新增：构造更清晰的 user prompt（注入）=====
def build_user_prompt_with_points(question_text: str, k: int, placeholder: str = "<point>") -> str:
    k = max(1, int(k))
    q = str(question_text).strip()
    if q == "":
        q = "Describe the object represented by the 3D point cloud."
    point_block = " ".join([placeholder] * k)
    return (
        "3D_POINT_CLOUD_EMBEDDING:\n"
        f"{point_block}\n\n"
        "QUESTION:\n"
        f"{q}\n\n"
        "ANSWER:"
    )


# =========================
# 2) 读取对话 JSON：只取第一轮
# =========================

def _extract_first_round(conv_list: List[Dict[str, Any]]) -> Optional[Tuple[str, str]]:
    """
    conv_list: [{"from":"human","value":"..."}, {"from":"gpt","value":"..."}, ...]
    只取第一轮：第一个 human + 其后的第一个 gpt
    """
    if not isinstance(conv_list, list) or len(conv_list) == 0:
        return None

    human_text = None
    gpt_text = None

    # 找第一个 human
    human_idx = None
    for i, msg in enumerate(conv_list):
        frm = msg.get("from", None)
        if frm == "human":
            human_text = msg.get("value", "")
            human_idx = i
            break

    if human_idx is None:
        return None

    # 找 human 后第一个 gpt
    for j in range(human_idx + 1, len(conv_list)):
        frm = conv_list[j].get("from", None)
        if frm == "gpt":
            gpt_text = conv_list[j].get("value", "")
            break

    if human_text is None or gpt_text is None:
        return None
    return str(human_text), str(gpt_text)


def load_conversations_for_object_ids(
    json_path: str,
    target_object_ids: Iterable[str],
) -> Dict[str, Dict[str, str]]:
    """
    只为 target_object_ids 读取对话（human/gpt 第一轮）。
    优先使用 ijson 做流式解析（若环境中可用），否则退化为 json.load（可能占内存）。

    返回：
      {object_id: {"human": ..., "gpt": ...}, ...}
    """
    target_set = set(target_object_ids)
    out: Dict[str, Dict[str, str]] = {}
    if len(target_set) == 0:
        return out

    # 尝试 ijson 流式
    try:
        import ijson  # type: ignore
        with open(json_path, "rb") as f:
            # JSON 顶层是 list，所以 items(f, "item") 逐条读
            for item in ijson.items(f, "item"):
                obj_id = item.get("object_id", None)
                if obj_id in target_set:
                    conv = item.get("conversations", [])
                    pair = _extract_first_round(conv)
                    if pair is not None:
                        human, gpt = pair
                        out[obj_id] = {"human": human, "gpt": gpt}
                    if len(out) >= len(target_set):
                        break
        return out
    except Exception:
        pass

    # fallback：整文件 load（若文件很大可能占内存）
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        obj_id = item.get("object_id", None)
        if obj_id in target_set:
            pair = _extract_first_round(item.get("conversations", []))
            if pair is None:
                continue
            human, gpt = pair
            out[obj_id] = {"human": human, "gpt": gpt}
            if len(out) >= len(target_set):
                break

    return out


# =========================
# 3) 加载 AE
# =========================

def load_ae_from_ckpt(ckpt_path: str, device: torch.device, dtype: torch.dtype) -> Tuple[UnifiedPointTextAE, Dict[str, Any]]:
    """
    ckpt 结构来自你训练脚本保存：
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


def resolve_safe_pad_token_id(tokenizer, model=None) -> int:
    """
    为 inputs_embeds + generate 场景准备一个安全的 pad_token_id：
    - 优先 tokenizer.pad_token_id（Qwen3 一般是 <|endoftext|>）
    - 若 tokenizer 没设，尝试从词表里找 <|endoftext|>
    - 最重要：保证 pad_token_id != eos_token_id
    """
    eos = getattr(tokenizer, "eos_token_id", None)

    pad = getattr(tokenizer, "pad_token_id", None)

    if pad is None:
        # Qwen3 tokenizer_config.json 里 pad_token 通常就是 <|endoftext|>
        try:
            cand = tokenizer.convert_tokens_to_ids("<|endoftext|>")
            if isinstance(cand, int) and cand >= 0:
                pad = cand
        except Exception:
            pad = None

    if pad is None and model is not None:
        pad = getattr(getattr(model, "generation_config", None), "pad_token_id", None)

    if pad is None:
        # 最后兜底：选一个非 eos 的 id（仅用于 generate 内部伪造 prompt ids，不参与 forward embed）
        pad = 0 if eos != 0 else 1

    if eos is not None and pad == eos:
        # 关键：inputs_embeds 生成时 pad==eos 会导致“立即结束”
        pad = 0 if eos != 0 else 1

    return int(pad)


@torch.no_grad()
def ae_point_to_text_token_embeddings(
    ae: UnifiedPointTextAE,
    point_tokens: torch.Tensor,  # (G,D)
    text_embeds: torch.Tensor,   # (L,H) 仅用于 forward 兼容；我们最终用 text_recon_from_point
    text_mask: torch.Tensor,     # (L,)
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    返回：
      pred_text_tokens: (L,H)  = out["text_recon_from_point"][0]
      mask:            (L,)    = 根据 AE_TEXT_MASK_MODE 返回 dataset mask 或 fixed prefix mask
    """
    pt = to_device_dtype(point_tokens.unsqueeze(0), device, dtype)  # (1,G,D)
    te = to_device_dtype(text_embeds.unsqueeze(0), device, dtype)   # (1,L,H)

    # ===== 新增：通过全局变量控制 mask 策略（dataset / fixed_prefix）=====
    L = int(text_embeds.shape[0])

    if AE_TEXT_MASK_MODE == "dataset":
        tm = text_mask.unsqueeze(0).to(device=device)  # (1,L) bool
        used_mask = text_mask.to(device=device)
    elif AE_TEXT_MASK_MODE == "fixed_prefix":
        fixed_len = int(AE_FIXED_PREFIX_LEN)
        if fixed_len <= 0:
            raise ValueError(f"AE_FIXED_PREFIX_LEN must be > 0 when AE_TEXT_MASK_MODE='fixed_prefix', got {fixed_len}")
        fixed_mask = torch.zeros(L, dtype=torch.bool, device=device)
        fixed_mask[: min(fixed_len, L)] = True
        tm = fixed_mask.unsqueeze(0)  # (1,L)
        used_mask = fixed_mask
    else:
        raise ValueError(f"Unknown AE_TEXT_MASK_MODE: {AE_TEXT_MASK_MODE}")

    out = ae(point_feat=pt, text_feat=te, text_mask=tm)

    if "text_recon_from_point" not in out:
        raise RuntimeError("AE forward output missing key: 'text_recon_from_point'")

    pred = out["text_recon_from_point"][0]  # (L,H)
    return pred, used_mask


@torch.no_grad()
def pool_text_tokens_to_single_embedding(
    pred_text_tokens: torch.Tensor,  # (L,H)
    mask: torch.Tensor,              # (L,) bool
    mode: str,
) -> torch.Tensor:
    """
    把 (L,H) 压成 (H,)
    """
    if mode == "mean":
        m = mask.to(pred_text_tokens.device).to(pred_text_tokens.dtype)  # (L,)
        denom = m.sum().clamp_min(1.0)
        v = (pred_text_tokens * m.unsqueeze(-1)).sum(dim=0) / denom
        return v
    elif mode == "first":
        idx = torch.where(mask.to(pred_text_tokens.device))[0]
        if idx.numel() == 0:
            return pred_text_tokens[0]
        return pred_text_tokens[int(idx[0].item())]
    else:
        raise ValueError(f"Unknown pooling mode: {mode}")


# =========================
# 4) 构造 Qwen3-Omni 输入并注入 embedding
# =========================

def build_qwen_inputs(processor: Qwen3OmniMoeProcessor, user_text: str) -> Dict[str, torch.Tensor]:
    """
    使用 Qwen3-Omni 的 processor chat template 构造输入。
    """
    conversations = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_text}],
        },
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
def inject_point_embeddings_into_inputs_embeds(
    *,
    tokenizer,
    input_ids: torch.Tensor,       # (1, S)
    inputs_embeds: torch.Tensor,   # (1, S, H)
    point_embeddings: torch.Tensor,# (H,) 或 (K, H)
    point_placeholder: str = "<point>",
) -> Tuple[torch.Tensor, List[Tuple[int, int]], List[Tuple[str, List[int]]]]:
    """
    基于“<point> 是单独 special token（1 token）”的注入逻辑：

    - 先用 tokenizer.convert_tokens_to_ids(<point>) 得到该 special token 的 token_id
    - 在 input_ids 中直接查找该 token_id 出现的位置
    - 每个 <point> 只替换 1 个 token 的 embedding（span 恒为 [pos, pos+1)）

    返回：
      new_inputs_embeds: (1,S,H)
      spans: [(pos, pos+1), ...]   # token span, end exclusive
      patterns: [(pattern_text, pattern_ids)]  # 调试用：这里只有一个 pattern：<point>
    """
    new_embeds = inputs_embeds.clone()
    H = new_embeds.shape[-1]

    # ---- normalize point_embeddings shape ----
    if point_embeddings.dim() == 1:
        if point_embeddings.numel() != H:
            raise RuntimeError(f"point_embeddings dim mismatch: got {point_embeddings.numel()} vs hidden {H}")
        K = None  # replace all matched <point> tokens with the same vector
    elif point_embeddings.dim() == 2:
        if point_embeddings.shape[1] != H:
            raise RuntimeError(f"point_embeddings dim mismatch: got {point_embeddings.shape[1]} vs hidden {H}")
        K = int(point_embeddings.shape[0])
        if K <= 0:
            raise RuntimeError("point_embeddings has zero length (K=0).")
    else:
        raise RuntimeError(f"point_embeddings must be (H,) or (K,H), got shape={tuple(point_embeddings.shape)}")

    point_token_id = tokenizer.convert_tokens_to_ids(point_placeholder)
    if point_token_id is None or int(point_token_id) < 0:
        raise RuntimeError(
            f"Placeholder token '{point_placeholder}' is not in tokenizer vocab. "
            f"Make sure you have added it as a special token before inference."
        )
    point_token_id = int(point_token_id)

    # positions of <point> special token
    pos = torch.where(input_ids[0] == point_token_id)[0].tolist()
    if len(pos) == 0:
        raise RuntimeError(
            f"Cannot find <point> token_id={point_token_id} in prompt input_ids. "
            f"Check whether the chat template escaped/altered '{point_placeholder}', "
            f"or whether the prompt actually contains it."
        )

    spans_all: List[Tuple[int, int]] = [(int(p), int(p) + 1) for p in pos]
    patterns: List[Tuple[str, List[int]]] = [(point_placeholder, [point_token_id])]

    # ---- replace embeddings (1 token per <point>) ----
    if K is None:
        rep0 = point_embeddings.to(device=new_embeds.device, dtype=new_embeds.dtype).view(1, 1, H)
        for p in pos:
            new_embeds[:, p:p+1, :] = rep0
        return new_embeds, spans_all, patterns

    if len(pos) < K:
        raise RuntimeError(
            f"Found only {len(pos)} occurrences of '{point_placeholder}' in prompt, but need K={K}. "
            f"Check expand_point_placeholders() / MAX_POINT_TOKENS."
        )

    for i in range(K):
        p = int(pos[i])
        vec = point_embeddings[i].to(device=new_embeds.device, dtype=new_embeds.dtype).view(1, 1, H)
        new_embeds[:, p:p+1, :] = vec

    return new_embeds, spans_all[:K], patterns


# =========================
# 4.5) 新增：注入 debug（embedding 替换校验 + token proxy）
# =========================

@torch.no_grad()
def _topk_token_proxies(
    *,
    tokenizer,
    emb_weight: torch.Tensor,              # (V,H)
    emb_weight_norm: Optional[torch.Tensor],  # (V,)
    vec: torch.Tensor,                     # (H,)
    k: int,
) -> List[Tuple[int, str, float]]:
    """
    用输入 embedding 矩阵做一个最近邻 token proxy（cosine）。
    返回 [(token_id, token_str, score), ...]
    """
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
    payload: torch.Tensor,          # (H,) or (K,H)
    emb_weight: Optional[torch.Tensor] = None,
    emb_weight_norm: Optional[torch.Tensor] = None,
    max_show_spans: int = 8,
    show_topk: bool = True,
    topk: int = 6,
) -> None:
    """
    用于验证：payload（尤其是 text_recon_from_point）是否真的替换到了 inputs_embeds 中，
    并展示注入向量的大致 token 语义（Top-K 最近邻 proxy）。
    """
    if (not DEBUG_SHOW_INJECTION_DEBUG) and (_console is None):
        return

    scalar_mode = payload.dim() == 1
    show_n = min(len(spans), int(max_show_spans))

    # ---- rich table ----
    if _console is not None:
        tb = Table(
            title=f"[bold]Injection Debug[/bold] - {label}",
            box=box.SIMPLE_HEAVY,
            show_lines=False,
        )
        tb.add_column("#", justify="right", style="bold")
        tb.add_column("span", justify="left")
        tb.add_column("token_ids", justify="left")
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

            base_seg = base_embeds[0, st:ed, :]      # (len,H)
            inj_seg = injected_embeds[0, st:ed, :]   # (len,H)

            base_vec = base_seg[0]
            inj_vec = inj_seg[0]

            payload_vec = payload if scalar_mode else payload[i]
            # 校验：注入后的 span 是否等于 payload（broadcast）
            max_diff_payload = (inj_seg - payload_vec).abs().max().item()
            # 校验：注入后是否真的改变了原始 embedding
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
                # token 里可能有换行/特殊字符，这里做轻量 escape，避免表格错位
                def _esc(s: str) -> str:
                    return s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")

                topk_str = ", ".join([f"{_esc(t)}({s:.3f})" for _, t, s in proxies])

            tb.add_row(
                str(i),
                f"[{st},{ed})",
                ",".join(map(str, ids)),
                " ".join(toks),
                f"{max_diff_base:.3e}",
                f"{max_diff_payload:.3e}",
                f"{cos_bi:.3f}",
                f"{payload_norm:.3f}",
                topk_str if (show_topk and (emb_weight is not None)) else "",
            )

        _console.print(tb)
    else:
        # fallback 简化输出
        print(f"[DEBUG] Injection Debug - {label}")
        for i in range(show_n):
            st, ed = spans[i]
            ids = input_ids[0, st:ed].tolist()
            toks = tokenizer.convert_ids_to_tokens(ids)
            base_seg = base_embeds[0, st:ed, :]
            inj_seg = injected_embeds[0, st:ed, :]
            payload_vec = payload if scalar_mode else payload[i]
            max_diff_payload = (inj_seg - payload_vec).abs().max().item()
            max_diff_base = (inj_seg - base_seg).abs().max().item()
            print(f"  [{i}] span=[{st},{ed}) ids={ids} toks={toks} "
                  f"max|inj-base|={max_diff_base:.3e} max|inj-payload|={max_diff_payload:.3e}")


# =========================
# 4.6) 新增：定量评测（语义相似度）
# =========================

@torch.no_grad()
def _mean_pool_hidden_states(last_hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    last_hidden: (B, L, H)
    attention_mask: (B, L) in {0,1} or bool
    """
    if attention_mask is None:
        # 全 1
        return last_hidden.mean(dim=1)

    m = attention_mask
    if m.dtype != last_hidden.dtype:
        m = m.to(dtype=last_hidden.dtype)
    denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
    pooled = (last_hidden * m.unsqueeze(-1)).sum(dim=1) / denom
    return pooled


@torch.no_grad()
def sentence_embedding_with_qwen(
    *,
    model: Qwen3OmniMoeThinkerForConditionalGeneration,
    tokenizer,
    text: str,
    device: torch.device,
    max_length: int = 256,
) -> torch.Tensor:
    """
    用 Qwen 模型自身的 hidden states 生成句向量（mean pooling + float32）。
    返回：(H,) on device
    """
    t = "" if text is None else str(text)
    t = t.strip()
    if t == "":
        # 空文本：返回零向量（避免 NaN）
        H = int(model.get_input_embeddings().weight.shape[1])
        return torch.zeros(H, device=device, dtype=torch.float32)

    enc = tokenizer(
        t,
        return_tensors="pt",
        truncation=True,
        max_length=int(max_length),
        padding=False,
        add_special_tokens=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    # 兼容不同 forward 签名：尽量关闭 cache 减少显存
    try:
        out = model(**enc, output_hidden_states=True, return_dict=True, use_cache=False)
    except TypeError:
        out = model(**enc, output_hidden_states=True, return_dict=True)

    last_hidden = None

    # transformers 常见输出：hidden_states[-1]
    hs = getattr(out, "hidden_states", None)
    if hs is not None and isinstance(hs, (tuple, list)) and len(hs) > 0:
        last_hidden = hs[-1]

    # 有些模型会给 last_hidden_state
    if last_hidden is None:
        last_hidden = getattr(out, "last_hidden_state", None)

    # 最后兜底：用 input embedding（无上下文）
    if last_hidden is None:
        emb_layer = model.get_input_embeddings()
        last_hidden = emb_layer(enc["input_ids"])

    attn = enc.get("attention_mask", None)
    pooled = _mean_pool_hidden_states(last_hidden, attn)  # (1,H)
    return pooled[0].to(dtype=torch.float32)


@torch.no_grad()
def semantic_similarity_pred_gt(
    *,
    model: Qwen3OmniMoeThinkerForConditionalGeneration,
    tokenizer,
    pred_text: str,
    gt_text: str,
    device: torch.device,
    max_length: int = 256,
) -> float:
    """
    语义相似度：cosine(emb(pred), emb(gt))
    """
    v1 = sentence_embedding_with_qwen(model=model, tokenizer=tokenizer, text=pred_text, device=device, max_length=max_length)
    v2 = sentence_embedding_with_qwen(model=model, tokenizer=tokenizer, text=gt_text, device=device, max_length=max_length)
    sim = float(F.cosine_similarity(v1.view(1, -1), v2.view(1, -1), dim=-1).item())
    return sim


# =========================
# 5) 生成（优先 generate，失败则 fallback greedy）
# =========================

_PAD_DEBUG_PRINTED = False  # 新增：避免每次 generate 都刷屏


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

    # 复制 inputs，移除 input_ids，注入 inputs_embeds
    gen_kwargs = {k: v for k, v in inputs.items() if k != "input_ids"}
    gen_kwargs["inputs_embeds"] = inputs_embeds

    # --- 关键修复：显式指定一个“安全的 pad_token_id”，并保证 != eos ---
    pad_id = resolve_safe_pad_token_id(tokenizer, model)
    gen_kwargs["pad_token_id"] = pad_id

    global _PAD_DEBUG_PRINTED
    if not _PAD_DEBUG_PRINTED:
        _p(
            f"[DEBUG] tokenizer.pad_token_id={tokenizer.pad_token_id}, "
            f"eos_token_id={tokenizer.eos_token_id}, resolved_pad_id={pad_id}",
            style="dim" if _console is not None else None,
        )
        _PAD_DEBUG_PRINTED = True

    # eos 也建议显式给（对 Qwen 系列通常是 <|im_end|>）
    if getattr(tokenizer, "eos_token_id", None) is not None:
        gen_kwargs["eos_token_id"] = tokenizer.eos_token_id

    # 只在 do_sample 时传 temperature/top_p，避免 None 进入 generate（更稳）
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

    # ---- 解码：对 inputs_embeds 场景做鲁棒处理 ----
    # 理论上 gen_out 是 (prompt + new)；但不同版本/实现可能只返回 new。
    prompt_len = inputs_embeds.shape[1]
    seq = gen_out[0]

    if seq.shape[0] > prompt_len:
        new_tokens = seq[prompt_len:]
    else:
        # 没有包含 prompt 的情况（或根本没生成任何 token）
        new_tokens = seq

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
    """
    简易 greedy fallback（仅兜底用）：
    - step0 用 inputs_embeds 喂入
    - 后续 step 用 input_ids + past_key_values
    - 尝试维护 attention_mask；若 inputs 里有 position_ids，也做简单递增扩展
    """
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


# =========================
# 6) 从 feature dataset 选一些样本
# =========================

def collect_samples_from_feature_dataset(
    ds: ProcessedPointTextFeatureDataset,
    num_samples: int,
    start: int,
    max_scan: int,
) -> List[Dict[str, Any]]:
    """
    顺序扫描 ds，从 start 起拿 num_samples 条 valid 样本（ds.require_valid=True 时 invalid 会抛错）
    """
    out = []
    end = min(len(ds), start + max_scan)
    for idx in range(start, end):
        try:
            item = ds[idx]
        except Exception:
            continue
        out.append(item)
        if len(out) >= num_samples:
            break
    return out


# =========================
# 7) 结果保存工具
# =========================

def _ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and (not os.path.exists(d)):
        os.makedirs(d, exist_ok=True)


def write_jsonl_line(f, obj: Dict[str, Any]) -> None:
    s = json.dumps(obj, ensure_ascii=False)
    f.write(s + "\n")
    f.flush()


# =========================
# 8) split eval
# =========================

def run_eval_on_split(
    *,
    split_name: str,
    feature_dataset_info_yaml: str,
    conv_json_path: str,
    num_samples: int,
    model: Qwen3OmniMoeThinkerForConditionalGeneration,
    processor: Qwen3OmniMoeProcessor,
    tokenizer,
    emb_layer: torch.nn.Module,
    emb_device: torch.device,
    emb_dtype: torch.dtype,
    llm_hidden: int,
    ae: UnifiedPointTextAE,
    ae_cfg: Dict[str, Any],
    emb_weight: torch.Tensor,
    emb_weight_norm: Optional[torch.Tensor],
    results_f,
) -> Dict[str, Any]:
    """
    在指定 split 上评测并写入 JSONL。
    返回 split summary（计数 + mean similarity 等）。
    """
    if num_samples <= 0:
        _p(f"[INFO] Skip split={split_name} because num_samples={num_samples}", style="yellow" if _console is not None else None)
        return {"split": split_name, "num_requested": num_samples, "num_run": 0}

    # -------- 1) load feature dataset --------
    feat_ds = ProcessedPointTextFeatureDataset(feature_dataset_info_yaml, require_valid=True)

    # 先多取一些候选 object_id，再去 JSON 里找对应对话（避免 JSON 全量 load）
    candidate = collect_samples_from_feature_dataset(
        feat_ds,
        num_samples=max(num_samples * 5, num_samples),
        start=DATASET_SCAN_START,
        max_scan=MAX_DATASET_SCAN,
    )
    if len(candidate) == 0:
        raise RuntimeError(f"No valid samples found in feature dataset for split={split_name}. Check dataset_info_yaml / require_valid.")

    cand_ids = [c["object_id"] for c in candidate]
    conv_map = load_conversations_for_object_ids(conv_json_path, cand_ids)

    # 过滤出 JSON 里确实有对话的样本
    samples = [c for c in candidate if c["object_id"] in conv_map]

    # # 更符合“随机选取”的语义：在 matched pool 内 shuffle 后取前 num_samples
    # random.shuffle(samples)
    samples = samples[:num_samples]

    if len(samples) == 0:
        raise RuntimeError(
            f"No samples matched between feature dataset and conversation JSON for split={split_name}. "
            "Please check object_id consistency / scan range."
        )

    _p(f"[INFO] ===== Split={split_name} =====", style="bold green" if _console is not None else None)
    _p(f"[INFO] Feature dataset total={len(feat_ds)}", style="green" if _console is not None else None)
    _p(f"[INFO] Candidate={len(candidate)} matched_in_json={len(samples)} (will run {len(samples)})", style="green" if _console is not None else None)
    _p(f"[INFO] POINT_INJECT_MODE={POINT_INJECT_MODE}, MAX_POINT_TOKENS={MAX_POINT_TOKENS}", style="green" if _console is not None else None)
    _p(f"[INFO] AE_TEXT_MASK_MODE={AE_TEXT_MASK_MODE}, AE_FIXED_PREFIX_LEN={AE_FIXED_PREFIX_LEN}", style="green" if _console is not None else None)

    # -------- 2) sanity check：AE 输出维度应等于 LLM hidden --------
    ae_d_text_in = int(ae_cfg["model"]["d_text_in"])
    if ae_d_text_in != llm_hidden:
        _p(
            f"[WARN] Dimension mismatch: AE d_text_in={ae_d_text_in} vs Qwen hidden={llm_hidden}. "
            f"Injection will fail unless you add a trained projection.",
            style="yellow" if _console is not None else None,
        )

    # -------- 3) run inference --------
    sims: List[float] = []
    num_ok = 0
    num_total = 0

    for si, sample in enumerate(samples):
        num_total += 1

        obj_id = sample["object_id"]
        human_raw = conv_map[obj_id]["human"]
        gt = conv_map[obj_id]["gpt"]

        # 为 prompt 结构化准备：去掉原始 human 里的所有 <point>，得到“纯问题文本”
        question_text = strip_all_point_placeholders(human_raw, POINT_PLACEHOLDER)

        # 4.1 AE: point -> pred text tokens
        pred_tokens, mask = ae_point_to_text_token_embeddings(
            ae=ae,
            point_tokens=sample["point_tokens"],     # (G,D) CPU
            text_embeds=sample["text_embeds"],       # (L,H) CPU
            text_mask=sample["text_mask"],           # (L,) CPU bool
            device=emb_device,
            dtype=emb_dtype,
        )

        # ===== baseline：准备 GT text embedding（来自 feature dataset）=====
        gt_text_tokens = to_device_dtype(sample["text_embeds"], emb_device, emb_dtype)  # (L,H)

        # 4.2 构造 point 注入向量（sequence or pooled）
        if POINT_INJECT_MODE == "sequence":
            # ---- AE payload ----
            if mask.any():
                ae_seq = pred_tokens[mask]        # (Kfull,H)
                gt_seq = gt_text_tokens[mask]     # (Kfull,H)
            else:
                # 极端兜底：mask 全 False
                ae_seq = pred_tokens[:1]
                gt_seq = gt_text_tokens[:1]

            # cap
            if ae_seq.shape[0] > MAX_POINT_TOKENS:
                ae_seq = ae_seq[:MAX_POINT_TOKENS]
                gt_seq = gt_seq[:MAX_POINT_TOKENS]

            K = int(ae_seq.shape[0])

            # ===== 重新组织 prompt（注入版本）：明确分块 =====
            human = build_user_prompt_with_points(question_text, K, POINT_PLACEHOLDER)

            # 两个注入版本
            point_payload_ae = ae_seq  # (K,H)
            point_payload_gt = gt_seq  # (K,H)

        elif POINT_INJECT_MODE == "pooled":
            point_vec_ae = pool_text_tokens_to_single_embedding(pred_tokens, mask, mode=POINT_POOLING)      # (H,)
            point_vec_gt = pool_text_tokens_to_single_embedding(gt_text_tokens, mask, mode=POINT_POOLING)  # (H,)
            K = 1

            # ===== 重新组织 prompt（注入版本）：明确分块 =====
            human = build_user_prompt_with_points(question_text, K, POINT_PLACEHOLDER)

            point_payload_ae = point_vec_ae
            point_payload_gt = point_vec_gt
        else:
            raise ValueError(f"Unknown POINT_INJECT_MODE: {POINT_INJECT_MODE}")

        # 4.3 build Qwen inputs（注入用 inputs）
        inputs = build_qwen_inputs(processor, human)
        inputs = {k: v.to(emb_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        input_ids = inputs["input_ids"]  # (1,S)

        # 4.4 compute embeds & inject（base embeds for inject prompt）
        with torch.no_grad():
            base_embeds = emb_layer(input_ids)  # (1,S,H)

        # ===== baseline：注入 GT text embedding =====
        gt_inject_text = None
        gt_injected_embeds = None
        gt_spans: List[Tuple[int, int]] = []
        gt_patterns: List[Tuple[str, List[int]]] = []
        if RUN_BASELINE_GT_TEXT_EMBED_INJECT:
            try:
                gt_injected_embeds, gt_spans, gt_patterns = inject_point_embeddings_into_inputs_embeds(
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    inputs_embeds=base_embeds,
                    point_embeddings=point_payload_gt,
                    point_placeholder=POINT_PLACEHOLDER,
                )
                gt_inject_text = generate_with_inputs_embeds(
                    model=model,
                    processor=processor,
                    inputs=inputs,
                    inputs_embeds=gt_injected_embeds,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=DO_SAMPLE,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                )
            except Exception as e:
                gt_inject_text = f"[GT-embed inject failed: {repr(e)}]"

        # ===== 原逻辑：注入 AE(text_recon_from_point) embedding =====
        pred_text = None
        injected_embeds = None
        spans: List[Tuple[int, int]] = []
        patterns: List[Tuple[str, List[int]]] = []

        inject_error = None
        try:
            injected_embeds, spans, patterns = inject_point_embeddings_into_inputs_embeds(
                tokenizer=tokenizer,
                input_ids=input_ids,
                inputs_embeds=base_embeds,
                point_embeddings=point_payload_ae,
                point_placeholder=POINT_PLACEHOLDER,
            )
        except Exception as e:
            inject_error = repr(e)

        if inject_error is not None:
            _rule()
            _p(f"[{split_name}][{si}] object_id={obj_id}", style="bold red" if _console is not None else None)
            _p("[ERROR] injection failed: " + inject_error, style="bold red" if _console is not None else None)
            _p("Human(raw):")
            _p(human_raw)
            _p("Human(used for inject variants):")
            _p(human)
            _p("GT:")
            _p(gt)

            # 写入结果（失败也记录）
            rec = {
                "split": split_name,
                "object_id": obj_id,
                "global_index": sample.get("global_index", None),
                "question": question_text,
                "baseline_inject_gt_text_embeds": gt_inject_text,
                "pred_inject_ae_embedding": None,
                "gt_answer": gt,
                "semantic_sim_pred_gt": None,
                "status": "inject_failed",
                "error": inject_error,
                "meta": {
                    "POINT_INJECT_MODE": POINT_INJECT_MODE,
                    "MAX_POINT_TOKENS": int(MAX_POINT_TOKENS),
                    "K_injected": int(K),
                    "AE_TEXT_MASK_MODE": AE_TEXT_MASK_MODE,
                    "AE_FIXED_PREFIX_LEN": int(AE_FIXED_PREFIX_LEN),
                },
            }
            write_jsonl_line(results_f, rec)
            continue

        # 4.5 generate with injected embeds
        gen_error = None
        try:
            pred_text = generate_with_inputs_embeds(
                model=model,
                processor=processor,
                inputs=inputs,
                inputs_embeds=injected_embeds,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
        except Exception as e:
            gen_error = repr(e)
            # fallback
            try:
                pred_text = greedy_fallback_generate(
                    model=model,
                    tokenizer=tokenizer,
                    inputs=inputs,
                    inputs_embeds=injected_embeds,
                    max_new_tokens=MAX_NEW_TOKENS,
                )
                pred_text = pred_text + f"  [fallback_used_due_to_generate_error={gen_error}]"
                gen_error = None
            except Exception as e2:
                pred_text = f"[inject generation failed: {repr(e)}; fallback also failed: {repr(e2)}]"

        # 4.6 定量评测：Pred vs GT 语义相似度
        sim_val = None
        sim_error = None
        try:
            sim_val = semantic_similarity_pred_gt(
                model=model,
                tokenizer=tokenizer,
                pred_text=pred_text if pred_text is not None else "",
                gt_text=gt,
                device=emb_device,
                max_length=SIM_MAX_LENGTH,
            )
        except Exception as e:
            sim_error = repr(e)
            sim_val = None

        if isinstance(sim_val, float):
            sims.append(sim_val)
            num_ok += 1

        # 4.7 pretty print result（rich）
        _rule(f"[bold cyan]{split_name} Sample {si}[/bold cyan]  object_id={obj_id}" if _console is not None else f"{split_name} Sample {si} object_id={obj_id}")

        # meta info
        if _console is not None:
            meta = Table(title="Meta", box=box.SIMPLE, show_header=False)
            meta.add_row("split", str(split_name))
            meta.add_row("object_id", str(obj_id))
            meta.add_row("global_index", str(sample.get("global_index", "N/A")))
            meta.add_row("POINT_INJECT_MODE", str(POINT_INJECT_MODE))
            meta.add_row("AE_TEXT_MASK_MODE", str(AE_TEXT_MASK_MODE))
            if AE_TEXT_MASK_MODE == "fixed_prefix":
                meta.add_row("AE_FIXED_PREFIX_LEN", str(AE_FIXED_PREFIX_LEN))
            if POINT_INJECT_MODE == "sequence":
                meta.add_row("K(injected)", f"{K} (capped by MAX_POINT_TOKENS={MAX_POINT_TOKENS})")
            else:
                meta.add_row("pooled_mode", str(POINT_POOLING))
            meta.add_row("RUN_BASELINE_GT_TEXT_EMBED_INJECT", str(RUN_BASELINE_GT_TEXT_EMBED_INJECT))
            meta.add_row("semantic_sim(Pred,GT)", f"{sim_val:.4f}" if isinstance(sim_val, float) else "N/A")
            _console.print(meta)
        else:
            if POINT_INJECT_MODE == "sequence":
                _p(f"[INFO] injected K={K} point tokens (capped by MAX_POINT_TOKENS={MAX_POINT_TOKENS})")
            else:
                _p(f"[INFO] pooled injection mode={POINT_POOLING}")
            _p(f"[INFO] semantic_sim(Pred,GT)={sim_val}" if sim_val is not None else "[INFO] semantic_sim(Pred,GT)=N/A")

        # debug: spans/patterns
        if DEBUG_SHOW_INJECTION_DEBUG:
            if _console is not None:
                dbg = Table(title="Injection Locate Result", box=box.SIMPLE, show_header=False)
                dbg.add_row("spans(len)", f"{len(spans)}  {spans[:10]}{' ...' if len(spans) > 10 else ''}")
                dbg.add_row("patterns_tried", str(len(patterns)))
                dbg.add_row("patterns", Pretty(patterns) if len(patterns) <= 16 else Pretty(patterns[:16] + [("...truncated...", [])]))
                _console.print(dbg)
            else:
                _p(f"[DEBUG] spans(len={len(spans)}): {spans[:10]}{' ...' if len(spans) > 10 else ''}")
                _p(f"[DEBUG] patterns_tried={len(patterns)}")
                _p(f"[DEBUG] patterns: {patterns}")

            # embedding 替换校验（AE）
            render_injection_debug(
                label="AE(text_recon_from_point)",
                tokenizer=tokenizer,
                input_ids=input_ids,
                base_embeds=base_embeds,
                injected_embeds=injected_embeds,
                spans=spans,
                payload=point_payload_ae,
                emb_weight=emb_weight if DEBUG_SHOW_TOPK_TOKEN_PROXIES else None,
                emb_weight_norm=emb_weight_norm if DEBUG_SHOW_TOPK_TOKEN_PROXIES else None,
                max_show_spans=DEBUG_MAX_SHOW_SPANS,
                show_topk=DEBUG_SHOW_TOPK_TOKEN_PROXIES,
                topk=DEBUG_TOPK_TOKENS,
            )

        # prompts
        if _console is not None:
            _console.print(Panel(human, title="Human(prompt used for inject variants)", border_style="cyan", expand=False))
        else:
            _p("Human(prompt used for inject variants):")
            _p(human)

        # outputs
        if _console is not None:
            out_tb = Table(title="Outputs", box=box.SIMPLE_HEAVY)
            out_tb.add_column("Variant", style="bold", justify="left")
            out_tb.add_column("Text", overflow="fold", justify="left")

            if RUN_BASELINE_GT_TEXT_EMBED_INJECT:
                out_tb.add_row("Baseline(inject GT text_embeds)", gt_inject_text if gt_inject_text is not None else "")

            out_tb.add_row("Pred(inject AE embedding)", pred_text if pred_text is not None else "")
            out_tb.add_row("GT(answer)", gt)
            out_tb.add_row("SemanticSim(Pred,GT)", f"{sim_val:.6f}" if isinstance(sim_val, float) else f"N/A ({sim_error})" if sim_error else "N/A")

            _console.print(out_tb)
        else:
            _p("-" * 80)
            if RUN_BASELINE_GT_TEXT_EMBED_INJECT:
                _p("Baseline(inject GT text_embeds):")
                _p(gt_inject_text)
                _p("-" * 80)
            _p("Pred(inject AE embedding):")
            _p(pred_text)
            _p("-" * 80)
            _p("GT(answer):")
            _p(gt)
            _p("-" * 80)
            _p(f"SemanticSim(Pred,GT): {sim_val}" if sim_val is not None else f"SemanticSim(Pred,GT): N/A ({sim_error})")

        # 写入 JSONL 结果
        rec = {
            "split": split_name,
            "object_id": obj_id,
            "global_index": sample.get("global_index", None),
            "question": question_text,
            "baseline_inject_gt_text_embeds": gt_inject_text,
            "pred_inject_ae_embedding": pred_text,
            "gt_answer": gt,
            "semantic_sim_pred_gt": sim_val,
            "status": "ok" if gen_error is None else "gen_error_fallback_or_failed",
            "error": gen_error,
            "metric_error": sim_error,
            "meta": {
                "POINT_INJECT_MODE": POINT_INJECT_MODE,
                "MAX_POINT_TOKENS": int(MAX_POINT_TOKENS),
                "K_injected": int(K),
                "AE_TEXT_MASK_MODE": AE_TEXT_MASK_MODE,
                "AE_FIXED_PREFIX_LEN": int(AE_FIXED_PREFIX_LEN),
                "DO_SAMPLE": bool(DO_SAMPLE),
                "MAX_NEW_TOKENS": int(MAX_NEW_TOKENS),
            },
        }
        write_jsonl_line(results_f, rec)

    # split summary
    mean_sim = float(np.mean(sims)) if len(sims) > 0 else None
    summary = {
        "split": split_name,
        "num_requested": int(num_samples),
        "num_run": int(num_total),
        "num_metric_ok": int(num_ok),
        "mean_semantic_sim_pred_gt": mean_sim,
    }
    _p(f"[INFO] Split={split_name} done. num_run={num_total}, mean_semantic_sim={mean_sim}", style="green" if _console is not None else None)
    return summary


# =========================
# 9) main
# =========================

def main() -> None:
    set_global_seed(SEED)
    torch.set_grad_enabled(False)

    # -------- 0) output files --------
    _ensure_parent_dir(EVAL_OUTPUT_JSONL)
    _ensure_parent_dir(EVAL_OUTPUT_SUMMARY_JSON)
    mode = "w" if EVAL_OUTPUT_OVERWRITE else "a"
    results_f = open(EVAL_OUTPUT_JSONL, mode, encoding="utf-8")

    # -------- 1) load Qwen3-Omni Thinker + processor --------
    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        QWEN_MODEL_NAME_OR_PATH,
        dtype="auto",
        device_map="auto",
    )
    model.eval()
    processor = Qwen3OmniMoeProcessor.from_pretrained(QWEN_MODEL_NAME_OR_PATH)
    tokenizer = processor.tokenizer

    # ===== 关键修改：把 <point> 加为 tokenizer 的单独 special token（1 token）=====
    # - 必须在后续构造 input_ids 前完成
    # - 若新增 token，需要 resize 模型词嵌入
    old_point_ids = tokenizer.encode(POINT_PLACEHOLDER, add_special_tokens=False)
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": [AddedToken(POINT_PLACEHOLDER, lstrip=False, rstrip=False)]}
    )
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        # 为了让“baseline 更稳定”：用旧分词的平均 embedding 初始化新 token embedding
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
        f"[INFO] Registered '{POINT_PLACEHOLDER}' as a single special token: id={point_token_id}, added={num_added}, tokenizer_len={len(tokenizer)}",
        style="green" if _console is not None else None,
    )

    # embedding device/dtype：用于注入向量对齐
    emb_layer = model.get_input_embeddings()
    emb_device = emb_layer.weight.device
    emb_dtype = emb_layer.weight.dtype
    llm_hidden = emb_layer.weight.shape[1]

    _p(f"[INFO] Qwen embedding device={emb_device}, dtype={emb_dtype}, hidden={llm_hidden}", style="green" if _console is not None else None)

    # -------- 2) load AE（放到同一个 device/dtype 更省拷贝）--------
    ae, ae_cfg = load_ae_from_ckpt(AE_CKPT_PATH, device=emb_device, dtype=emb_dtype)

    # ---- 新增：为 token proxy（Top-K）准备 embedding norm（只在需要时计算一次）----
    emb_weight = emb_layer.weight
    emb_weight_norm = None
    if DEBUG_SHOW_INJECTION_DEBUG and DEBUG_SHOW_TOPK_TOKEN_PROXIES:
        try:
            emb_weight_norm = emb_weight.norm(dim=1).clamp_min(1e-6)
        except Exception:
            emb_weight_norm = None

    # -------- 3) eval train + val --------
    summaries: List[Dict[str, Any]] = []

    # Train
    summaries.append(
        run_eval_on_split(
            split_name="train",
            feature_dataset_info_yaml=FEATURE_DATASET_INFO_YAML,
            conv_json_path=CONV_JSON_PATH,
            num_samples=NUM_SAMPLES,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            emb_layer=emb_layer,
            emb_device=emb_device,
            emb_dtype=emb_dtype,
            llm_hidden=llm_hidden,
            ae=ae,
            ae_cfg=ae_cfg,
            emb_weight=emb_weight,
            emb_weight_norm=emb_weight_norm,
            results_f=results_f,
        )
    )

    # Val
    if RUN_EVAL_VALIDATION:
        if (not isinstance(FEATURE_DATASET_INFO_YAML_VAL, str)) or (FEATURE_DATASET_INFO_YAML_VAL.strip() == ""):
            raise RuntimeError("RUN_EVAL_VALIDATION=True but FEATURE_DATASET_INFO_YAML_VAL is empty. Please set it.")
        if (not isinstance(CONV_JSON_PATH_VAL, str)) or (CONV_JSON_PATH_VAL.strip() == ""):
            raise RuntimeError("RUN_EVAL_VALIDATION=True but CONV_JSON_PATH_VAL is empty. Please set it.")

        summaries.append(
            run_eval_on_split(
                split_name="val",
                feature_dataset_info_yaml=FEATURE_DATASET_INFO_YAML_VAL,
                conv_json_path=CONV_JSON_PATH_VAL,
                num_samples=NUM_SAMPLES_VAL,
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                emb_layer=emb_layer,
                emb_device=emb_device,
                emb_dtype=emb_dtype,
                llm_hidden=llm_hidden,
                ae=ae,
                ae_cfg=ae_cfg,
                emb_weight=emb_weight,
                emb_weight_norm=emb_weight_norm,
                results_f=results_f,
            )
        )

    results_f.close()

    # -------- 4) save summary --------
    summary_obj = {
        "config": {
            "AE_CKPT_PATH": AE_CKPT_PATH,
            "QWEN_MODEL_NAME_OR_PATH": QWEN_MODEL_NAME_OR_PATH,
            "FEATURE_DATASET_INFO_YAML": FEATURE_DATASET_INFO_YAML,
            "CONV_JSON_PATH": CONV_JSON_PATH,
            "FEATURE_DATASET_INFO_YAML_VAL": FEATURE_DATASET_INFO_YAML_VAL,
            "CONV_JSON_PATH_VAL": CONV_JSON_PATH_VAL,
            "RUN_EVAL_VALIDATION": bool(RUN_EVAL_VALIDATION),
            "NUM_SAMPLES_TRAIN": int(NUM_SAMPLES),
            "NUM_SAMPLES_VAL": int(NUM_SAMPLES_VAL),
            "DATASET_SCAN_START": int(DATASET_SCAN_START),
            "MAX_DATASET_SCAN": int(MAX_DATASET_SCAN),
            "SEED": int(SEED),
            "POINT_PLACEHOLDER": POINT_PLACEHOLDER,
            "AE_TEXT_MASK_MODE": AE_TEXT_MASK_MODE,
            "AE_FIXED_PREFIX_LEN": int(AE_FIXED_PREFIX_LEN),
            "POINT_INJECT_MODE": POINT_INJECT_MODE,
            "MAX_POINT_TOKENS": int(MAX_POINT_TOKENS),
            "POINT_POOLING": POINT_POOLING,
            "MAX_NEW_TOKENS": int(MAX_NEW_TOKENS),
            "DO_SAMPLE": bool(DO_SAMPLE),
            "TEMPERATURE": float(TEMPERATURE),
            "TOP_P": float(TOP_P),
            "RUN_BASELINE_GT_TEXT_EMBED_INJECT": bool(RUN_BASELINE_GT_TEXT_EMBED_INJECT),
            "SIM_MAX_LENGTH": int(SIM_MAX_LENGTH),
        },
        "splits": summaries,
        "outputs": {
            "jsonl": os.path.abspath(EVAL_OUTPUT_JSONL),
            "summary_json": os.path.abspath(EVAL_OUTPUT_SUMMARY_JSON),
        },
    }

    with open(EVAL_OUTPUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_obj, f, ensure_ascii=False, indent=2)

    _p(f"\n[INFO] Done. Results saved to:\n- {os.path.abspath(EVAL_OUTPUT_JSONL)}\n- {os.path.abspath(EVAL_OUTPUT_SUMMARY_JSON)}",
       style="green" if _console is not None else None)


if __name__ == "__main__":
    main()
