# -*- coding: utf-8 -*-
"""
infer_touch_ae_qwen3_simple.py

TVL touch-only 定性推理脚本：
- 从 ProcessedTouchTextFeatureDataset 随机抽 N 条样本
- 用训练好的 UnifiedTouchTextAE：touch tokens -> text token embeddings（text_recon_from_touch）
- 将该 embeddings 注入 Qwen3-Omni 的输入中，替换 prompt 里 <touch> 对应 token span 的 inputs_embeds
- 打印模型输出 Pred 与 GT（conversations 里第一轮 gpt value）

特点：
- 不做 baseline
- 不做指标
- 不写文件
- 只打印 Pred / GT

注意：
- system prompt 中避免出现字面 "<touch>"，否则 tokenizer 可能当作 special token 干扰定位
- 会把 "<touch>" 注册为 tokenizer 的 additional_special_tokens（确保它是 1 token）
"""

from __future__ import annotations

import os
import json
import glob
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

# ====== 你的模块（按你给出的工程路径）======
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

# 你的 touch AE checkpoint：可以给 .pt 文件，也可以给目录（会自动找 best*.pt 或最新 .pt）
AE_CKPT_PATH = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/swift/checkpoints/tvl/stage1/1/best.pt"

# 你的 memmap feature dataset_info.yaml（脚本会自动在候选路径里找存在的那个）
FEATURE_DATASET_INFO_YAML_CANDIDATES = [
    "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/tvl_features_stage1/dataset_info.yaml",
    "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/tvl_features_stage1/dataset_info.yaml",
    "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/tvl_features_stage1/dataset_info.yaml",
]

# 你的 GT 对话 JSON（你明确给的）
CONV_JSON_PATH = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/dataset/TVL/Touch-Vision-Language-Dataset/tvl_dataset/finetune_merged.json"

# Qwen3-Omni 模型
QWEN_MODEL_NAME_OR_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

# 只做定性：随机抽多少条
NUM_SAMPLES = 30
SEED = 42

# placeholder（用于注入定位）
TOUCH_PLACEHOLDER = "<touch>"

# 注入策略：建议 sequence（逐 token 注入）
INJECT_MODE = "sequence"   # "sequence" | "pooled"
MAX_TOUCH_PLACEHOLDER_TOKENS = 128  # 保险上限（你这里 text max_len=24，实际 K<=24）

# pooled 模式下池化策略
POOLING = "mean"  # "mean" | "first"

# 生成参数
MAX_NEW_TOKENS = 128
DO_SAMPLE = False
TEMPERATURE = 0.7
TOP_P = 0.9

# 是否打印完整 prompt（包含很多 <touch>，会很长）
PRINT_PROMPT = False


SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group.\n\n"
    "Task setting:\n"
    "- You will answer questions using tactile/touch context.\n"
    "- The user text may contain multimodal tags like <image>. You do NOT receive the image.\n"
    "- Some requests contain a section named 'TOUCH_EMBEDDING'. The tokens in that section are placeholders.\n"
    "  Their embeddings are injected at inference time to carry tactile information.\n\n"
    "Instructions:\n"
    "- Use the 'TOUCH_EMBEDDING' section as the primary context.\n"
    "- Answer the QUESTION concisely.\n"
    "- Output only the final answer text.\n"
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
    # 都不存在就返回第一个，后续自然报错，方便你定位
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
    """
    conv json: 常见 id="000000000106" (12位补零)
    dataset: sample_id 可能是 int(106) 或 "106" 或 bytes
    这里统一：如果是纯数字且长度<12，则 zfill(12)
    """
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
    """
    human 文本里常见 <image>，对 touch-only 推理会干扰模型；
    这里把常见 tag 全部清理掉。
    """
    t = "" if human_raw is None else str(human_raw)
    for tag in ["<image>", "<tactile>", "<touch>"]:
        t = t.replace(tag, " ")
    # 合并多余空白/空行
    t = "\n".join([" ".join(line.split()) for line in t.splitlines() if " ".join(line.split())])
    return t.strip()


def build_user_prompt_with_touch(question_text: str, k: int, placeholder: str) -> str:
    k = max(1, int(k))
    q = (question_text or "").strip()
    if q == "":
        q = "What tactile/touch qualities are conveyed?"
    block = " ".join([placeholder] * k)
    return (
        "TOUCH_EMBEDDING:\n"
        f"{block}\n\n"
        "QUESTION:\n"
        f"{q}\n\n"
        "ANSWER:"
    )


def _extract_first_round(conv_list: List[Dict[str, Any]]) -> Optional[Tuple[str, str]]:
    """只取第一轮：第一个 human + 其后的第一个 gpt"""
    if not isinstance(conv_list, list) or len(conv_list) == 0:
        return None

    human_text = None
    gpt_text = None
    human_idx = None

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
    """
    只为 target_ids 读取对话（human/gpt 第一轮）。
    优先 ijson 流式（省内存）；没有 ijson 就 json.load。
    """
    target_set = set(normalize_id(x) for x in target_ids if normalize_id(x))
    out: Dict[str, Dict[str, str]] = {}
    if len(target_set) == 0:
        return out

    # ijson 流式
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

    # fallback：整文件 load
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
    """
    inputs_embeds + generate 的 pad_token_id：
    - pad_token_id 不能等于 eos_token_id，否则可能直接结束
    """
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
    """(L,H) -> (H,)"""
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
# 2) AE：touch -> text token embeddings
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

    # 兼容不同 forward 参数名：优先 touch_feat / touch_tokens / touch
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

    # 解析输出 key：优先 text_recon_from_touch
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
# 3) Qwen 输入与注入
# =========================

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
    """
    基于 <touch> 是单独 special token（1 token）的注入逻辑。
    """
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
        raise RuntimeError(f"Cannot find placeholder token '{placeholder}' (id={pid}) in input_ids. Prompt may not contain it.")

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

    # 复制 inputs，去掉 input_ids，用 inputs_embeds
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

    # 兼容：generate 返回 (prompt+new) 或仅 new
    prompt_len = inputs_embeds.shape[1]
    seq = out[0]
    new_tokens = seq[prompt_len:] if seq.shape[0] > prompt_len else seq
    return tokenizer.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()


# =========================
# 4) 随机抽样：从 feature dataset 拿 sample_id，去 json 找 GT
# =========================

def random_pick_matched_samples(
    ds: ProcessedTouchTextFeatureDataset,
    conv_json_path: str,
    num_samples: int,
    seed: int,
    candidate_multiplier: int = 8,
    max_tries_factor: int = 200,
) -> List[Tuple[int, str, str, str]]:
    """
    返回 list: (ds_idx, norm_id, human_raw, gt_text)
    """
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
# 5) main
# =========================

def main() -> None:
    set_global_seed(SEED)
    torch.set_grad_enabled(False)

    # -------- 1) resolve dataset_info.yaml path --------
    dataset_info_yaml = pick_first_existing_path(FEATURE_DATASET_INFO_YAML_CANDIDATES)
    if not os.path.exists(dataset_info_yaml):
        raise FileNotFoundError(
            f"Cannot find dataset_info.yaml. Tried candidates={FEATURE_DATASET_INFO_YAML_CANDIDATES}"
        )
    print(f"[INFO] dataset_info_yaml = {dataset_info_yaml}")

    # -------- 2) load Qwen model + processor --------
    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        QWEN_MODEL_NAME_OR_PATH,
        dtype="auto",
        device_map="auto",
    )
    model.eval()

    processor = Qwen3OmniMoeProcessor.from_pretrained(QWEN_MODEL_NAME_OR_PATH)
    tokenizer = processor.tokenizer

    # -------- 3) register <touch> as a single special token --------
    old_ids = tokenizer.encode(TOUCH_PLACEHOLDER, add_special_tokens=False)
    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": [AddedToken(TOUCH_PLACEHOLDER, lstrip=False, rstrip=False)]}
    )
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        # 可选：用旧分词均值初始化新 token embedding（更稳定）
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
    emb_device = emb_layer.weight.device
    emb_dtype = emb_layer.weight.dtype
    llm_hidden = int(emb_layer.weight.shape[1])
    print(f"[INFO] Qwen emb device={emb_device}, dtype={emb_dtype}, hidden={llm_hidden}")

    # -------- 4) load AE --------
    ckpt_file = resolve_ckpt_file(AE_CKPT_PATH)
    print(f"[INFO] AE ckpt file = {ckpt_file}")
    ae, ae_cfg = load_ae_from_ckpt(ckpt_file, device=emb_device, dtype=emb_dtype)

    # 维度检查（可选提示）
    try:
        ae_d_text_in = int(ae_cfg["model"].get("d_text_in", -1))
        if ae_d_text_in != -1 and ae_d_text_in != llm_hidden:
            print(f"[WARN] AE d_text_in={ae_d_text_in} != Qwen hidden={llm_hidden} -> 注入将维度不匹配（需要 projection）")
    except Exception:
        pass

    # -------- 5) load feature dataset --------
    ds = ProcessedTouchTextFeatureDataset(dataset_info_yaml, require_valid=True, return_ids=True)
    print(f"[INFO] feature dataset loaded, len={len(ds)}")

    # 打印 sample keys（帮助你确认字段）
    s0 = ds[0]
    print(f"[DEBUG] sample[0].keys() = {list(s0.keys())}")
    print(f"[DEBUG] sample[0].sample_id = {s0.get('sample_id')} (normalized={normalize_id(s0.get('sample_id'))})")

    # -------- 6) random pick matched samples --------
    picked = random_pick_matched_samples(
        ds=ds,
        conv_json_path=CONV_JSON_PATH,
        num_samples=NUM_SAMPLES,
        seed=SEED,
    )
    if len(picked) == 0:
        raise RuntimeError(
            "No matched samples between feature dataset and conversation json.\n"
            f"- conv_json={CONV_JSON_PATH}\n"
            "Please check that ds['sample_id'] matches json['id'] (maybe need different normalize rule)."
        )
    print(f"[INFO] matched samples: {len(picked)} / requested {NUM_SAMPLES}")

    # -------- 7) inference loop --------
    for i, (ds_idx, sid, human_raw, gt) in enumerate(picked):
        sample = ds[ds_idx]

        # 清理 human 文本（去 <image> 等）
        question = clean_human_to_question(human_raw)

        # touch -> AE -> pred text token embeddings
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

        # build payload + prompt
        if INJECT_MODE == "sequence":
            seq = pred_tokens[mask] if mask.any() else pred_tokens[:1]
            if seq.shape[0] > MAX_TOUCH_PLACEHOLDER_TOKENS:
                seq = seq[:MAX_TOUCH_PLACEHOLDER_TOKENS]
            K = int(seq.shape[0])
            user_text = build_user_prompt_with_touch(question, K, TOUCH_PLACEHOLDER)
            payload = seq  # (K,H)
        elif INJECT_MODE == "pooled":
            vec = pool_text_tokens_to_single_embedding(pred_tokens, mask, mode=POOLING)
            K = 1
            user_text = build_user_prompt_with_touch(question, K, TOUCH_PLACEHOLDER)
            payload = vec  # (H,)
        else:
            raise ValueError(f"Unknown INJECT_MODE: {INJECT_MODE}")
        print(f"payload is {payload.shape} tensor, K={K}, question='{question}'")
        # build qwen inputs
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
        )

        print("\n" + "=" * 120)
        print(f"[{i}] sample_id={sid}  ds_idx={ds_idx}  K_injected={K}  placeholder_pos(first3)={pos[:3]}")
        print("- Question (cleaned) -")
        print(question)
        if PRINT_PROMPT:
            print("- Prompt (used) -")
            print(user_text)
        print("- Pred (Qwen + injected TOUCH embedding) -")
        print(pred)
        print("- GT (json first-round gpt) -")
        print(gt)

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()