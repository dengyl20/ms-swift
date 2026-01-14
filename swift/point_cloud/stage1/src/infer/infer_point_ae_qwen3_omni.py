# -*- coding: utf-8 -*-
"""
infer_point_ae_qwen3_omni.py

目标：
- 用你训练好的 UnifiedPointTextAE：point_tokens -> text feature（embedding）
- 将该 embedding 注入到 Qwen3-Omni 的输入中，替换 prompt 里 <point> 对应 token span 的 inputs_embeds
- 对若干条样本进行推理，打印模型输出与 GT（对话 JSON 中的 gpt value）

依赖（建议）：
- transformers==4.57.3（Qwen3-Omni repo 推荐版本）
- accelerate
- torch

你自己的工程依赖：
- swift.point_cloud.stage1.src.data.feature_dataset.ProcessedPointTextFeatureDataset
- swift.point_cloud.stage1.src.models.unified_ae.UnifiedPointTextAE

注意：
- 本脚本不使用命令行参数；超参数/路径都在文件顶部配置。
"""

from __future__ import annotations

import os
import json
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

# ====== 你的模块（保持与训练脚本一致的 import 路径）======
from swift.point_cloud.stage1.src.data.feature_dataset import ProcessedPointTextFeatureDataset
from swift.point_cloud.stage1.src.models.unified_ae import UnifiedPointTextAE


# ====== Qwen3-Omni (Transformers) ======
from transformers import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeProcessor,
)

# =========================
# 0) 超参数 & 路径（直接改这里）
# =========================

# 你训练好的 AE checkpoint（best.pt 或某个 epoch_xxx.pt）
AE_CKPT_PATH = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/checkpoints/unified_ae_features/best.pt"

# stage1 提取 feature 的 dataset_info.yaml（里面记录 shards 路径、shape、dtype 等）
FEATURE_DATASET_INFO_YAML = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/data_features/dataset_info.yaml"

# 你指定的原始对话 JSON（用于取 prompt 与 GT）
CONV_JSON_PATH = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/PointLLM/PointLLM/PointLLM_brief_description_660K.json"

# Qwen3-Omni 模型（用 Instruct 权重加载 Thinker text-only，省显存）
QWEN_MODEL_NAME_OR_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

# 只做定性：跑多少条
NUM_SAMPLES = 8

# 从 feature dataset 扫描样本的起点与最大扫描量（防止一直找不到对应 object_id）
DATASET_SCAN_START = 0
MAX_DATASET_SCAN = 200000  # 你可按需调大/调小

# 随机种子（影响抽样）
SEED = 42

# prompt 里的占位符文本
POINT_PLACEHOLDER = "<point>"

# ===== 用 AE 产生的 text_recon_from_point (L,H) 怎么压成一个向量 (H) =====
# "mean": 按 text_mask 对有效 token 平均池化
# "first": 取第一个有效 token 的向量
POINT_POOLING = "mean"  # "mean" | "first"

# ===== 生成超参数 =====
MAX_NEW_TOKENS = 64
DO_SAMPLE = False
TEMPERATURE = 0.7   # DO_SAMPLE=False 时 temperature 不生效
TOP_P = 0.9         # DO_SAMPLE=False 时 top_p 不生效

# 是否也跑一个 baseline（不注入 embedding）方便你对比
RUN_BASELINE_NO_INJECT = True

# system prompt（可按需要修改）
SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of understanding text inputs and generating helpful responses."
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
      mask:            (L,)    = text_mask（原样返回）
    """
    pt = to_device_dtype(point_tokens.unsqueeze(0), device, dtype)  # (1,G,D)
    te = to_device_dtype(text_embeds.unsqueeze(0), device, dtype)   # (1,L,H)
    tm = text_mask.unsqueeze(0).to(device=device)                   # (1,L) bool

    out = ae(point_feat=pt, text_feat=te, text_mask=tm)

    if "text_recon_from_point" not in out:
        raise RuntimeError("AE forward output missing key: 'text_recon_from_point'")

    pred = out["text_recon_from_point"][0]  # (L,H)
    return pred, text_mask.to(device=device)


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
def inject_point_embedding_into_inputs_embeds(
    *,
    model: Qwen3OmniMoeThinkerForConditionalGeneration,
    tokenizer,
    input_ids: torch.Tensor,      # (1, S)
    inputs_embeds: torch.Tensor,  # (1, S, H)
    point_embedding: torch.Tensor,# (H,)
    point_placeholder: str = "<point>",
) -> Tuple[torch.Tensor, List[Tuple[int, int]], List[int]]:
    """
    在 inputs_embeds 中，将 prompt 里 "<point>" 对应 token span 的 embedding 替换为 point_embedding。
    由于 "<point>" 可能被分成多个 token，因此用子序列匹配方式找到 span。

    返回：
      new_inputs_embeds: (1,S,H)
      spans: [(start, end), ...]   # end is exclusive
      needle_ids: tokenizer.encode(point_placeholder, add_special_tokens=False)
    """
    # 1) needle token ids
    needle_ids = tokenizer.encode(point_placeholder, add_special_tokens=False)
    if len(needle_ids) == 0:
        raise RuntimeError(f"Tokenizer encodes placeholder into empty ids: {point_placeholder}")

    # 2) find spans in prompt
    prompt_ids = input_ids[0].tolist()
    starts = find_all_subsequence_positions(prompt_ids, needle_ids)

    # 某些情况下 prompt 里是 "<point>\n"（紧跟换行），尝试备用 needle
    if len(starts) == 0:
        for variant in [point_placeholder + "\n", point_placeholder + "\r\n", " " + point_placeholder, " " + point_placeholder + "\n"]:
            var_ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(var_ids) == 0:
                continue
            starts = find_all_subsequence_positions(prompt_ids, var_ids)
            if len(starts) > 0:
                needle_ids = var_ids
                break

    if len(starts) == 0:
        raise RuntimeError(
            f"Cannot find placeholder token sequence in prompt. "
            f"placeholder='{point_placeholder}', needle_ids={needle_ids[:10]}..., prompt_len={len(prompt_ids)}"
        )

    # 3) replace each span
    new_embeds = inputs_embeds.clone()
    H = new_embeds.shape[-1]
    if point_embedding.numel() != H:
        raise RuntimeError(f"point_embedding dim mismatch: got {point_embedding.numel()} vs hidden {H}")

    spans: List[Tuple[int, int]] = []
    for st in starts:
        ed = st + len(needle_ids)
        spans.append((st, ed))

        # 用同一个向量覆盖整个 span（span_len 可能 >1）
        rep = point_embedding.to(device=new_embeds.device, dtype=new_embeds.dtype).view(1, 1, H)
        rep = rep.expand(1, ed - st, H)  # (1, span_len, H)
        new_embeds[:, st:ed, :] = rep

    return new_embeds, spans, needle_ids


# =========================
# 5) 生成（优先 generate，失败则 fallback greedy）
# =========================

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
    """
    用 inputs_embeds + (attention_mask/position_ids/...) 做 generation。
    """
    tokenizer = processor.tokenizer

    # 复制 inputs，移除 input_ids，注入 inputs_embeds
    gen_kwargs = {k: v for k, v in inputs.items() if k != "input_ids"}
    gen_kwargs["inputs_embeds"] = inputs_embeds

    # 一些模型需要显式 pad_token_id
    if getattr(tokenizer, "pad_token_id", None) is None:
        # 兜底：用 eos 作为 pad
        gen_kwargs["pad_token_id"] = tokenizer.eos_token_id

    # generation config
    gen_out = model.generate(
        **gen_kwargs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
    )

    # gen_out: (1, prompt+new)
    prompt_len = inputs_embeds.shape[1]
    new_tokens = gen_out[0, prompt_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text.strip()


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
    简易 greedy fallback：
    - step0 用 inputs_embeds 喂入
    - 后续 step 用 input_ids + past_key_values
    - 尝试维护 attention_mask；若 inputs 里有 position_ids，也做简单递增扩展

    该 fallback 对 Qwen3-Omni 的 MRoPE/多模态 position 机制不一定完全稳，
    仅作为 generate 不可用时的兜底。
    """
    # 基础字段
    attention_mask = inputs.get("attention_mask", None)
    position_ids = inputs.get("position_ids", None)
    padding_mask = inputs.get("padding_mask", None)

    # step0
    out = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        padding_mask=padding_mask,
        use_cache=True,
    )
    logits = out.logits  # (1, S, V)
    past = out.past_key_values

    generated: List[int] = []
    eos = tokenizer.eos_token_id

    # helper: extend masks/pos
    def _extend_attention_mask(am: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if am is None:
            return None
        one = torch.ones((am.shape[0], 1), device=am.device, dtype=am.dtype)
        return torch.cat([am, one], dim=1)

    def _extend_padding_mask(pm: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if pm is None:
            return None
        one = torch.zeros((pm.shape[0], 1), device=pm.device, dtype=pm.dtype)  # padding_mask: 0 means not padded?
        # 这里不确定 padding_mask 语义；保守起见：新增 token 设为非 padding（0）
        return torch.cat([pm, one], dim=1)

    def _extend_position_ids(pid: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if pid is None:
            return None
        # pid 可能是 (B, S) 或 (B, 3, S)
        if pid.dim() == 2:
            last = pid[:, -1:]  # (B,1)
            nxt = last + 1
            return torch.cat([pid, nxt], dim=1)
        if pid.dim() == 3:
            last = pid[:, :, -1:]  # (B,3,1)
            nxt = last + 1
            return torch.cat([pid, nxt], dim=2)
        return pid

    for _ in range(max_new_tokens):
        next_id = torch.argmax(logits[:, -1, :], dim=-1)  # (1,)
        tid = int(next_id.item())
        generated.append(tid)
        if eos is not None and tid == eos:
            break

        # extend masks/pos for next step
        attention_mask = _extend_attention_mask(attention_mask)
        padding_mask = _extend_padding_mask(padding_mask)
        position_ids = _extend_position_ids(position_ids)

        out = model(
            input_ids=next_id.unsqueeze(0),  # (1,1)
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
# 7) main
# =========================

def main() -> None:
    set_global_seed(SEED)
    torch.set_grad_enabled(False)

    # -------- 1) load feature dataset --------
    feat_ds = ProcessedPointTextFeatureDataset(FEATURE_DATASET_INFO_YAML, require_valid=True)

    # 先多取一些候选 object_id，再去 JSON 里找对应对话（避免 JSON 全量 load）
    candidate = collect_samples_from_feature_dataset(
        feat_ds,
        num_samples=max(NUM_SAMPLES * 5, NUM_SAMPLES),
        start=DATASET_SCAN_START,
        max_scan=MAX_DATASET_SCAN,
    )
    if len(candidate) == 0:
        raise RuntimeError("No valid samples found in feature dataset. Check dataset_info_yaml / require_valid.")

    cand_ids = [c["object_id"] for c in candidate]
    conv_map = load_conversations_for_object_ids(CONV_JSON_PATH, cand_ids)

    # 过滤出 JSON 里确实有对话的样本
    samples = [c for c in candidate if c["object_id"] in conv_map]
    samples = samples[:NUM_SAMPLES]

    if len(samples) == 0:
        raise RuntimeError(
            "No samples matched between feature dataset and conversation JSON. "
            "Please check object_id consistency / scan range."
        )

    print(f"[INFO] Feature dataset total={len(feat_ds)}")
    print(f"[INFO] Candidate={len(candidate)} matched_in_json={len(samples)} (will run {len(samples)})")

    # -------- 2) load Qwen3-Omni Thinker + processor --------
    # dtype="auto"：让 HF 自己选 bf16/fp16
    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        QWEN_MODEL_NAME_OR_PATH,
        dtype="auto",
        device_map="auto",
    )
    model.eval()
    processor = Qwen3OmniMoeProcessor.from_pretrained(QWEN_MODEL_NAME_OR_PATH)
    tokenizer = processor.tokenizer

    # embedding device/dtype：用于注入向量对齐
    emb_layer = model.get_input_embeddings()
    emb_device = emb_layer.weight.device
    emb_dtype = emb_layer.weight.dtype

    print(f"[INFO] Qwen embedding device={emb_device}, dtype={emb_dtype}, hidden={emb_layer.weight.shape[1]}")

    # -------- 3) load AE（放到同一个 device/dtype 更省拷贝）--------
    ae, ae_cfg = load_ae_from_ckpt(AE_CKPT_PATH, device=emb_device, dtype=emb_dtype)

    # 简单 sanity check：AE 输出维度应等于 LLM hidden
    ae_d_text_in = int(ae_cfg["model"]["d_text_in"])
    if ae_d_text_in != emb_layer.weight.shape[1]:
        print(
            f"[WARN] Dimension mismatch: AE d_text_in={ae_d_text_in} vs Qwen hidden={emb_layer.weight.shape[1]}. "
            f"Injection may fail."
        )

    # -------- 4) run inference --------
    for si, sample in enumerate(samples):
        obj_id = sample["object_id"]
        human = conv_map[obj_id]["human"]
        gt = conv_map[obj_id]["gpt"]

        # 4.1 AE: point -> pred text tokens -> pooled vector
        pred_tokens, mask = ae_point_to_text_token_embeddings(
            ae=ae,
            point_tokens=sample["point_tokens"],     # (G,D) CPU
            text_embeds=sample["text_embeds"],       # (L,H) CPU
            text_mask=sample["text_mask"],           # (L,) CPU bool
            device=emb_device,
            dtype=emb_dtype,
        )
        point_vec = pool_text_tokens_to_single_embedding(pred_tokens, mask, mode=POINT_POOLING)  # (H,)

        # 4.2 build Qwen inputs
        inputs = build_qwen_inputs(processor, human)
        # move to model device (按官方示例)
        inputs = {k: v.to(emb_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        input_ids = inputs["input_ids"]  # (1,S)

        # baseline (no inject)
        baseline_text = None
        if RUN_BASELINE_NO_INJECT:
            try:
                base_out = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=DO_SAMPLE,
                    temperature=TEMPERATURE if DO_SAMPLE else None,
                    top_p=TOP_P if DO_SAMPLE else None,
                )
                prompt_len = input_ids.shape[1]
                baseline_text = tokenizer.decode(
                    base_out[0, prompt_len:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()
            except Exception as e:
                baseline_text = f"[baseline generation failed: {repr(e)}]"

        # 4.3 compute embeds & inject
        with torch.no_grad():
            base_embeds = emb_layer(input_ids)  # (1,S,H)

        try:
            injected_embeds, spans, needle_ids = inject_point_embedding_into_inputs_embeds(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                inputs_embeds=base_embeds,
                point_embedding=point_vec,
                point_placeholder=POINT_PLACEHOLDER,
            )
        except Exception as e:
            print("=" * 100)
            print(f"[{si}] object_id={obj_id}")
            print("[ERROR] injection failed:", repr(e))
            print("Human:", human)
            print("GT   :", gt)
            continue

        # 4.4 generate with injected embeds
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
            # fallback
            try:
                pred_text = greedy_fallback_generate(
                    model=model,
                    tokenizer=tokenizer,
                    inputs=inputs,
                    inputs_embeds=injected_embeds,
                    max_new_tokens=MAX_NEW_TOKENS,
                )
                pred_text = pred_text + f"  [fallback_used_due_to_generate_error={repr(e)}]"
            except Exception as e2:
                pred_text = f"[inject generation failed: {repr(e)}; fallback also failed: {repr(e2)}]"

        # 4.5 print result
        print("=" * 100)
        print(f"[{si}] object_id: {obj_id}")
        print(f"needle_ids(len={len(needle_ids)}): {needle_ids}")
        print(f"matched spans: {spans}")
        print("-" * 80)
        print("Human(prompt):")
        print(human)
        print("-" * 80)
        if RUN_BASELINE_NO_INJECT:
            print("Baseline(no inject):")
            print(baseline_text)
            print("-" * 80)
        print("Pred(inject AE embedding):")
        print(pred_text)
        print("-" * 80)
        print("GT(answer):")
        print(gt)

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()
