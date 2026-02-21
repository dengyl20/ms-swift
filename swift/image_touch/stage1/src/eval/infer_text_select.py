# -*- coding: utf-8 -*-
"""
infer_tvl_vocab_constrained.py

TVL touch-only 定性推理脚本（带词表约束输出）：
- 随机抽 N 条样本
- touch -> UnifiedTouchTextAE -> text token embeddings
- 注入到 Qwen3-Omni prompt 的 <touch> token spans（inputs_embeds 替换）
- 生成回答，并将回答强制规整为 “仅由词表词组成的逗号分隔列表”

特点：
- 不做 baseline
- 不做指标
- 不写文件
- 打印 Pred(raw)（可选） + Pred(vocab_only) + GT

关键新增：
1) 加载词表 raw.words.txt
2) 生成时在 user prompt 里提供 “候选词子集”（按 touch embedding 与词表 embedding 相似度 Top-K）
3) 对输出做 vocab 过滤/格式化，确保最终只包含词表词

注意：
- system prompt 中避免出现字面 "<touch>"
- 会把 "<touch>" 注册为 tokenizer 的 additional_special_tokens（确保它是 1 token）
"""

from __future__ import annotations

import os
import json
import glob
import random
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

# ====== 你的模块（按你当前跑通的工程路径）======
from swift.tvl.stage1.src.data.touch_fea_dataset import ProcessedTouchTextFeatureDataset
from swift.tvl.stage1.src.models.unified_touch import UnifiedTouchTextAE

# ====== Qwen3-Omni (Transformers) ======
from transformers import (
    AddedToken,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeProcessor,
)

# ===== OFFLINE MODE =====
os.environ["HF_HUB_OFFLINE"] = "1"

# =========================
# 0) 路径 & 超参数（你只需要改这里）
# =========================

AE_CKPT_PATH = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/swift/checkpoints/tvl/stage1/1/best.pt"

FEATURE_DATASET_INFO_YAML_CANDIDATES = [
    "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/tvl_ssvtp_test_eval_outputs/ssvtp_test_features/dataset_info.yaml",
]

CONV_JSON_PATH = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/tvl_ssvtp_test_eval_outputs/ssvtp_test_meta.json"

# ===== 词表 =====
VOCAB_PATH = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/dataset/TVL/Touch-Vision-Language-Dataset/tvl_dataset/vocab_merged/raw.words.txt"

QWEN_MODEL_NAME_OR_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

NUM_SAMPLES = 30
SEED = 42

TOUCH_PLACEHOLDER = "<touch>"

INJECT_MODE = "sequence"   # "sequence" | "pooled"
MAX_TOUCH_PLACEHOLDER_TOKENS = 128  # 你的 text max_len=24，实际 K<=24

POOLING = "mean"  # pooled 模式用： "mean" | "first"

# ===== 生成参数 =====
MAX_NEW_TOKENS = 48   # 只输出几个词，别太长
DO_SAMPLE = False
TEMPERATURE = 0.7
TOP_P = 0.9

# ===== 输出控制 =====
PRINT_PROMPT = False
PRINT_RAW_PRED = False  # True 可同时打印 Qwen 原始输出（调试用）

# ===== 词表约束参数 =====
# 给模型看的候选词 Top-K（从全词表按相似度选一小部分，避免把整个词表塞进 prompt）
VOCAB_CANDIDATE_TOPK = 80

# 最终输出词数量（过滤后过多会截断；为空则用候选兜底）
OUT_MAX_WORDS = 6
OUT_FALLBACK_WORDS = 4  # 如果过滤后一个词都没匹配上，用 top-4 兜底


SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group.\n\n"
    "Task setting:\n"
    "- You will answer questions using tactile/touch context.\n"
    "- The user text may contain multimodal tags like <image>. You do NOT receive the image.\n"
    "- Some requests contain a section named 'TOUCH_EMBEDDING'. The tokens in that section are placeholders.\n"
    "  Their embeddings are injected at inference time to carry tactile information.\n\n"
    "Output rules (STRICT):\n"
    "- Output MUST be a comma-separated list of tactile descriptor words.\n"
    "- Output ONLY the words list. No explanations, no full sentences.\n"
    "- Use lowercase words.\n"
    "- Format example: smooth, textured, hard.\n"
)


# =========================
# 1) 通用工具
# =========================

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_first_existing_path(cands: List[str]) -> str:
    for p in cands:
        if isinstance(p, str) and p.strip() and os.path.exists(p):
            return p
    return cands[0] if cands else ""


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


def normalize_id(x: Any) -> str:
    """保持你现在已跑通的 id 规则：带前缀的不动，纯数字才 zfill(12)。"""
    if x is None:
        return ""
    if isinstance(x, bytes):
        try:
            x = x.decode("utf-8", errors="ignore")
        except Exception:
            x = str(x)
    s = str(x).strip()
    if s.isdigit() and len(s) < 12:
        s = s.zfill(12)
    return s


def clean_human_to_question(human_raw: str) -> str:
    """清理 <image> 等 tag，避免模型以为有图。"""
    t = "" if human_raw is None else str(human_raw)
    for tag in ["<image>", "<tactile>", "<touch>"]:
        t = t.replace(tag, " ")
    t = "\n".join([" ".join(line.split()) for line in t.splitlines() if " ".join(line.split())])
    return t.strip()


def _extract_first_round(conv_list: List[Dict[str, Any]]) -> Optional[Tuple[str, str]]:
    if not isinstance(conv_list, list) or len(conv_list) == 0:
        return None
    human_text, gpt_text, human_idx = None, None, None
    for i, msg in enumerate(conv_list):
        if msg.get("from") == "human":
            human_text = msg.get("value", "")
            human_idx = i
            break
    if human_idx is None:
        return None
    for j in range(human_idx + 1, len(conv_list)):
        if conv_list[j].get("from") == "gpt":
            gpt_text = conv_list[j].get("value", "")
            break
    if human_text is None or gpt_text is None:
        return None
    return str(human_text), str(gpt_text)


def load_conversations_for_ids(
    json_path: str,
    target_ids: Iterable[str],
    id_key: str = "id",
) -> Dict[str, Dict[str, str]]:
    target_set = set(normalize_id(x) for x in target_ids if normalize_id(x))
    out: Dict[str, Dict[str, str]] = {}
    if len(target_set) == 0:
        return out

    try:
        import ijson  # type: ignore
        with open(json_path, "rb") as f:
            for item in ijson.items(f, "item"):
                sid = normalize_id(item.get(id_key, None))
                if sid in target_set:
                    pair = _extract_first_round(item.get("conversations", []))
                    if pair is not None:
                        human, gpt = pair
                        out[sid] = {"human": human, "gpt": gpt}
                    if len(out) >= len(target_set):
                        break
        return out
    except Exception:
        pass

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        sid = normalize_id(item.get(id_key, None))
        if sid in target_set:
            pair = _extract_first_round(item.get("conversations", []))
            if pair is not None:
                human, gpt = pair
                out[sid] = {"human": human, "gpt": gpt}
            if len(out) >= len(target_set):
                break

    return out


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
def pool_text_tokens_to_single_embedding(pred_text_tokens: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "mean":
        m = mask.to(pred_text_tokens.device).to(pred_text_tokens.dtype)
        denom = m.sum().clamp_min(1.0)
        return (pred_text_tokens * m.unsqueeze(-1)).sum(dim=0) / denom
    elif mode == "first":
        idx = torch.where(mask.to(pred_text_tokens.device))[0]
        return pred_text_tokens[int(idx[0].item())] if idx.numel() > 0 else pred_text_tokens[0]
    else:
        raise ValueError(f"Unknown pooling mode: {mode}")


# =========================
# 2) 词表 & 输出规整
# =========================

def load_vocab_words(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"VOCAB_PATH not found: {path}")
    words: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip().lower()
            if not w:
                continue
            # 允许 a-z 和连字符
            if re.fullmatch(r"[a-z][a-z\-]*", w) is None:
                # 如果你的词表里有其它字符，可以在这里放宽规则
                # 这里先保守一点，但仍然收录（只要不是空）
                pass
            words.append(w)
    # 去重保持顺序
    seen = set()
    out = []
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def parse_and_filter_to_vocab(text: str, vocab_set: set[str], max_words: int) -> List[str]:
    """
    从模型输出里提取词表词：
    - 支持逗号/分号分隔
    - 支持句子输出（会从中抓取 vocab 里的词）
    """
    if text is None:
        text = ""
    s = str(text).lower()

    # 统一分隔符
    s = s.replace("；", ",").replace(";", ",")
    s = s.replace("\n", " ")

    # 用逗号先切，再用空格切
    parts = re.split(r"[,\u3001]+", s)  # \u3001 是顿号
    out: List[str] = []
    seen = set()

    for part in parts:
        part = part.strip()
        if not part:
            continue
        for token in re.split(r"\s+", part):
            token = token.strip()
            if not token:
                continue
            # 去掉两端标点，保留 a-z 和连字符
            token = re.sub(r"^[^a-z\-]+|[^a-z\-]+$", "", token)
            token = token.strip("-")
            if not token:
                continue
            if token in vocab_set and token not in seen:
                out.append(token)
                seen.add(token)
                if len(out) >= max_words:
                    return out
    return out


def format_vocab_list(words: List[str], add_period: bool = True) -> str:
    if not words:
        return ""
    s = ", ".join(words)
    if add_period:
        return s + "."
    return s


# =========================
# 3) “候选词子集”选择：用 embedding 相似度从全词表里选 Top-K
# =========================

@torch.no_grad()
def build_vocab_embedding_matrix(
    *,
    tokenizer,
    emb_weight: torch.Tensor,   # (V,H) in bf16
    vocab_words: List[str],
    device: torch.device,
) -> torch.Tensor:
    """
    为每个 vocab word 计算一个向量（平均 token embedding），并做 L2 normalize。
    返回: vocab_vecs_normed (N,H) float32 on device
    """
    H = int(emb_weight.shape[1])
    out = torch.empty((len(vocab_words), H), device=device, dtype=torch.float32)

    for i, w in enumerate(vocab_words):
        # 尽量用前导空格版本（更贴近生成时 tokenization）
        ids = tokenizer.encode(" " + w, add_special_tokens=False)
        if not ids:
            ids = tokenizer.encode(w, add_special_tokens=False)
        if not ids:
            # 极端：tokenizer 无法编码
            out[i].zero_()
            continue
        ids_t = torch.tensor(ids, device=device, dtype=torch.long)
        vec = emb_weight.index_select(0, ids_t).mean(dim=0).to(torch.float32)
        out[i] = vec

    out = F.normalize(out, dim=1, eps=1e-6)
    return out


@torch.no_grad()
def topk_vocab_candidates(
    pooled_vec: torch.Tensor,          # (H,) float32
    vocab_words: List[str],
    vocab_vecs_normed: torch.Tensor,   # (N,H) float32
    k: int,
) -> List[str]:
    if k <= 0:
        return []
    v = pooled_vec.to(dtype=torch.float32)
    v = F.normalize(v, dim=0, eps=1e-6)
    scores = torch.matmul(vocab_vecs_normed, v)  # (N,)
    kk = min(int(k), int(scores.numel()))
    topi = torch.topk(scores, k=kk, largest=True).indices.detach().cpu().tolist()
    return [vocab_words[j] for j in topi]


# =========================
# 4) AE：touch -> text token embeddings
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
    touch_tokens: torch.Tensor,          # (G,D)
    text_shape_like: torch.Tensor,       # (L,H) 仅提供形状
    text_mask: torch.Tensor,             # (L,) bool
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    返回：
      pred_text_tokens: (L,H)
      mask: (L,)
    关键：text_feat 用全零，避免把 GT text embeddings 喂进 AE 造成信息泄漏。
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

    if out is None:
        raise RuntimeError(
            "AE forward failed for all candidate touch input names. "
            f"Last error={repr(last_err)}. "
            "Please check UnifiedTouchTextAE.forward signature."
        )

    if not isinstance(out, dict):
        raise RuntimeError(f"AE output is not dict, got type={type(out)}")

    if "text_recon_from_touch" in out:
        pred = out["text_recon_from_touch"][0]
    elif "text_recon_from_tactile" in out:
        pred = out["text_recon_from_tactile"][0]
    elif "text_recon_from_point" in out:
        pred = out["text_recon_from_point"][0]
    else:
        cand = [k for k in out.keys() if str(k).startswith("text_recon")]
        if not cand:
            raise RuntimeError(f"AE output missing text recon key. out.keys()={list(out.keys())}")
        pred = out[cand[0]][0]

    return pred, text_mask.to(device=device)


# =========================
# 5) Qwen 输入与注入
# =========================

def build_user_prompt_with_touch(
    question_text: str,
    k: int,
    placeholder: str,
    allowed_words: Optional[List[str]] = None,
) -> str:
    """
    这是你要“修改 question/prompt”的核心：
    - 明确要求输出必须是逗号分隔词列表
    - 并给出 allowed_words（候选词子集），强约束模型只能从里面选
    """
    k = max(1, int(k))
    q = (question_text or "").strip()
    if q == "":
        q = "What tactile/touch qualities are conveyed?"

    block = " ".join([placeholder] * k)

    allow_block = ""
    if allowed_words:
        allow_block = (
            "ALLOWED_WORDS (choose ONLY from this list):\n"
            + ", ".join(allowed_words)
            + "\n\n"
        )

    return (
        "TOUCH_EMBEDDING:\n"
        f"{block}\n\n"
        "INSTRUCTIONS (STRICT):\n"
        "- Answer with 1 to 6 lowercase words.\n"
        "- Words MUST be chosen from ALLOWED_WORDS.\n"
        "- Output ONLY the comma-separated list. No extra text.\n"
        "- Format example: smooth, textured, hard.\n\n"
        f"{allow_block}"
        "QUESTION:\n"
        f"{q}\n\n"
        "ANSWER:"
    )


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
            raise RuntimeError(f"payload dim mismatch: got {payload.numel()} vs H={H}")
    elif payload.dim() == 2:
        K = int(payload.shape[0])
        if payload.shape[1] != H:
            raise RuntimeError(f"payload dim mismatch: got {payload.shape[1]} vs H={H}")
    else:
        raise RuntimeError(f"payload must be (H,) or (K,H), got shape={tuple(payload.shape)}")

    pid = tokenizer.convert_tokens_to_ids(placeholder)
    if pid is None or int(pid) < 0:
        raise RuntimeError(f"Placeholder token '{placeholder}' not in vocab. Did you add_special_tokens?")
    pid = int(pid)

    pos = torch.where(input_ids[0] == pid)[0].tolist()
    if len(pos) == 0:
        raise RuntimeError(f"Cannot find placeholder token '{placeholder}' (id={pid}) in input_ids.")

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
) -> str:
    tokenizer = processor.tokenizer

    gen_kwargs = {k: v for k, v in inputs.items() if k != "input_ids"}
    gen_kwargs["inputs_embeds"] = inputs_embeds

    gen_kwargs["pad_token_id"] = resolve_safe_pad_token_id(tokenizer, model)
    if getattr(tokenizer, "eos_token_id", None) is not None:
        gen_kwargs["eos_token_id"] = tokenizer.eos_token_id

    extra = {}
    if DO_SAMPLE:
        extra["temperature"] = float(TEMPERATURE)
        extra["top_p"] = float(TOP_P)

    out = model.generate(
        **gen_kwargs,
        max_new_tokens=int(MAX_NEW_TOKENS),
        do_sample=bool(DO_SAMPLE),
        **extra,
    )

    prompt_len = inputs_embeds.shape[1]
    seq = out[0]
    new_tokens = seq[prompt_len:] if seq.shape[0] > prompt_len else seq
    return tokenizer.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()


# =========================
# 6) 随机抽样：从 feature dataset 拿 sample_id，去 json 找 GT
# =========================

def random_pick_matched_samples(
    ds: ProcessedTouchTextFeatureDataset,
    conv_json_path: str,
    num_samples: int,
    seed: int,
    candidate_multiplier: int = 8,
    max_tries_factor: int = 200,
) -> List[Tuple[int, str, str, str]]:
    rng = random.Random(seed)

    cand_n = max(num_samples * candidate_multiplier, num_samples)
    cand: List[Tuple[int, str]] = []
    seen_idx = set()
    seen_id = set()

    tries = 0
    max_tries = cand_n * max_tries_factor

    while len(cand) < cand_n and tries < max_tries:
        tries += 1
        idx = rng.randrange(len(ds))
        if idx in seen_idx:
            continue
        seen_idx.add(idx)

        try:
            it = ds[idx]
        except Exception:
            continue

        sid = normalize_id(it.get("sample_id", None))
        if (not sid) or (sid in seen_id):
            continue
        seen_id.add(sid)
        cand.append((idx, sid))

    conv_map = load_conversations_for_ids(conv_json_path, [sid for _, sid in cand], id_key="id")

    matched = []
    for idx, sid in cand:
        if sid in conv_map:
            matched.append((idx, sid, conv_map[sid]["human"], conv_map[sid]["gpt"]))

    rng.shuffle(matched)
    return matched[:num_samples]


# =========================
# 7) main
# =========================

def main() -> None:
    set_global_seed(SEED)
    torch.set_grad_enabled(False)

    # -------- 1) dataset_info.yaml --------
    dataset_info_yaml = pick_first_existing_path(FEATURE_DATASET_INFO_YAML_CANDIDATES)
    if not os.path.exists(dataset_info_yaml):
        raise FileNotFoundError(f"Cannot find dataset_info.yaml. Tried: {FEATURE_DATASET_INFO_YAML_CANDIDATES}")
    print(f"[INFO] dataset_info_yaml = {dataset_info_yaml}")

    # -------- 2) load Qwen --------
    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        QWEN_MODEL_NAME_OR_PATH,
        dtype="auto",
        device_map="auto",
    )
    model.eval()

    processor = Qwen3OmniMoeProcessor.from_pretrained(QWEN_MODEL_NAME_OR_PATH)
    tokenizer = processor.tokenizer

    # -------- 3) register <touch> --------
    old_ids = tokenizer.encode(TOUCH_PLACEHOLDER, add_special_tokens=False)
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": [AddedToken(TOUCH_PLACEHOLDER, lstrip=False, rstrip=False)]}
    )
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        # 用旧分词均值初始化新 token embedding（更稳）
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

    touch_token_id = tokenizer.convert_tokens_to_ids(TOUCH_PLACEHOLDER)
    print(f"[INFO] registered '{TOUCH_PLACEHOLDER}' as single token: id={touch_token_id}, added={num_added}, vocab={len(tokenizer)}")

    emb_layer = model.get_input_embeddings()
    emb_weight = emb_layer.weight
    emb_device = emb_weight.device
    emb_dtype = emb_weight.dtype
    llm_hidden = int(emb_weight.shape[1])
    print(f"[INFO] Qwen emb device={emb_device}, dtype={emb_dtype}, hidden={llm_hidden}")

    # -------- 4) load AE --------
    ckpt_file = resolve_ckpt_file(AE_CKPT_PATH)
    print(f"[INFO] AE ckpt file = {ckpt_file}")
    ae, ae_cfg = load_ae_from_ckpt(ckpt_file, device=emb_device, dtype=emb_dtype)

    # -------- 5) load dataset --------
    ds = ProcessedTouchTextFeatureDataset(dataset_info_yaml, require_valid=True, return_ids=True)
    print(f"[INFO] feature dataset loaded, len={len(ds)}")
    s0 = ds[0]
    print(f"[DEBUG] sample[0].keys() = {list(s0.keys())}")
    print(f"[DEBUG] sample[0].sample_id = {s0.get('sample_id')} (normalized={normalize_id(s0.get('sample_id'))})")

    # -------- 6) load vocab + precompute vocab embeddings --------
    vocab_words = load_vocab_words(VOCAB_PATH)
    vocab_set = set(vocab_words)
    print(f"[INFO] vocab loaded: {len(vocab_words)} words from {VOCAB_PATH}")

    print("[INFO] building vocab embedding matrix (may take a bit)...")
    vocab_vecs_normed = build_vocab_embedding_matrix(
        tokenizer=tokenizer,
        emb_weight=emb_weight,
        vocab_words=vocab_words,
        device=emb_device,
    )
    print("[INFO] vocab embedding matrix ready.")

    # -------- 7) pick samples --------
    picked = random_pick_matched_samples(
        ds=ds,
        conv_json_path=CONV_JSON_PATH,
        num_samples=NUM_SAMPLES,
        seed=SEED,
    )
    if len(picked) == 0:
        raise RuntimeError("No matched samples between feature dataset and conversation json.")
    print(f"[INFO] matched samples: {len(picked)} / requested {NUM_SAMPLES}")

    # -------- 8) inference loop --------
    for i, (ds_idx, sid, human_raw, gt) in enumerate(picked):
        sample = ds[ds_idx]
        question = clean_human_to_question(human_raw)

        touch_tokens = sample["touch"]     # (G,D)
        text_shape_like = sample["text"]   # (L,H)
        text_mask = sample["mask"]         # (L,) bool

        pred_tokens, mask = ae_touch_to_text_token_embeddings(
            ae=ae,
            touch_tokens=touch_tokens,
            text_shape_like=text_shape_like,
            text_mask=text_mask,
            device=emb_device,
            dtype=emb_dtype,
        )

        # ===== 计算 pooled embedding -> 选候选词 Top-K（用于 prompt 约束）=====
        if mask.any():
            pooled = pred_tokens[mask].mean(dim=0).to(torch.float32)
        else:
            pooled = pred_tokens[:1].mean(dim=0).to(torch.float32)

        cand_words = topk_vocab_candidates(
            pooled_vec=pooled,
            vocab_words=vocab_words,
            vocab_vecs_normed=vocab_vecs_normed,
            k=VOCAB_CANDIDATE_TOPK,
        )

        # ===== 构造注入 payload =====
        if INJECT_MODE == "sequence":
            seq = pred_tokens[mask] if mask.any() else pred_tokens[:1]
            if seq.shape[0] > MAX_TOUCH_PLACEHOLDER_TOKENS:
                seq = seq[:MAX_TOUCH_PLACEHOLDER_TOKENS]
            K = int(seq.shape[0])
            payload = seq
        elif INJECT_MODE == "pooled":
            vec = pool_text_tokens_to_single_embedding(pred_tokens, mask, mode=POOLING)
            K = 1
            payload = vec
        else:
            raise ValueError(f"Unknown INJECT_MODE: {INJECT_MODE}")

        # ===== 关键：修改 user prompt / question（严格输出词表词 + 给候选词）=====
        user_text = build_user_prompt_with_touch(
            question_text=question,
            k=K,
            placeholder=TOUCH_PLACEHOLDER,
            allowed_words=cand_words,
        )

        if PRINT_PROMPT:
            print("\n[PROMPT]\n" + user_text + "\n")

        # build inputs
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

        raw_pred = generate_with_inputs_embeds(
            model=model,
            processor=processor,
            inputs=inputs,
            inputs_embeds=injected_embeds,
        )

        # ===== 双保险：把输出规整到“词表词列表” =====
        # filtered = parse_and_filter_to_vocab(raw_pred, vocab_set=vocab_set, max_words=OUT_MAX_WORDS)

        # if len(filtered) == 0:
        #     # 如果一个词都没匹配上：用候选词兜底（top-N）
        #     filtered = cand_words[:OUT_FALLBACK_WORDS]

        # pred_vocab_only = format_vocab_list(filtered, add_period=True)
        pred_vocab_only = raw_pred  # for now, just use raw pred
        print("\n" + "=" * 120)
        print(f"[{i}] sample_id={sid}  ds_idx={ds_idx}  K_injected={K}  placeholder_pos(first3)={pos[:3]}")
        print("- Question (cleaned) -")
        print(question)

        if PRINT_RAW_PRED:
            print("- Pred (raw from Qwen) -")
            print(raw_pred)

        print("- Pred (vocab-only, comma-separated) -")
        print(pred_vocab_only)

        print("- GT (json first-round gpt) -")
        print(gt)

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()