# pc_template.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from swift.point_cloud.stage2.src.pc_constants import DEFAULT_SYSTEM_PROMPT, POINT_TOKEN

# ===== 兼容不同 ms-swift 版本导出路径 =====
from swift.llm import Template, TemplateMeta, register_template


def _as_torch(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    # list -> tensor
    return torch.tensor(x)


def _get_underlying_model(model: nn.Module) -> nn.Module:
    # 兼容 DDP / wrapper
    if hasattr(model, "module"):
        return model.module  # type: ignore
    if hasattr(model, "model"):
        # 注意：不能盲目一直 .model 下去；但大多数 wrapper 是这样
        try:
            return model.model  # type: ignore
        except Exception:
            pass
    return model


class Qwen3OmniPointTemplate(Template):
    """
    - 用 messages 做标准 SFT tokenization
    - 额外把 point_tokens / text_mask / inject_len collate 出来
    - 在 _post_encode 内：point_tokens -> AE -> token embeds -> 注入 inputs_embeds
    """
    use_model = True  # 关键：让 ms-swift 在 forward 前调用 _post_encode

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch, padding_to)

        # collate point_tokens/text_mask/inject_len
        # batch[i] 来自 HF dataset row（我们在 loader 里 yield 的 dict）
        if "point_tokens" in batch[0]:
            pts = [_as_torch(b["point_tokens"]) for b in batch]  # each (G,D)
            res["point_tokens"] = torch.stack(pts, dim=0)

        if "text_mask" in batch[0]:
            tms = [_as_torch(b["text_mask"]).bool() for b in batch]  # each (L,)
            res["text_mask"] = torch.stack(tms, dim=0)

        if "inject_len" in batch[0]:
            ks = torch.tensor([int(b["inject_len"]) for b in batch], dtype=torch.long)
            res["inject_len"] = ks

        # 可选：debug 字段（不会用于 forward，但你也可以在 _post_encode pop 掉）
        if "object_id" in batch[0]:
            res["object_id"] = [str(b["object_id"]) for b in batch]

        return res

    def _post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        在 model.forward 前被调用。
        必须返回可微的 inputs_embeds 注入结果。
        """
        base_model = _get_underlying_model(model)

        point_tokens = inputs.pop("point_tokens", None)
        text_mask = inputs.pop("text_mask", None)
        inject_len = inputs.pop("inject_len", None)

        # 如果 batch 没有点云（不应发生），直接不注入
        if point_tokens is None:
            return {}

        if text_mask is None or inject_len is None:
            raise RuntimeError("Missing text_mask/inject_len in batch. Check dataset loader & collator.")

        # 找 AE
        if not hasattr(base_model, "point_ae"):
            raise RuntimeError("Model has no attribute 'point_ae'. Please use model_type=qwen3_omni_point.")
        point_ae = getattr(base_model, "point_ae")

        # tokenizer & point_token_id
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Template.processor.tokenizer is missing.")
        point_token_id = tokenizer.convert_tokens_to_ids(POINT_TOKEN)
        if point_token_id is None or int(point_token_id) < 0:
            raise RuntimeError(f"POINT_TOKEN '{POINT_TOKEN}' not found in tokenizer vocab. Check ModelLoader.")

        # 设备 / dtype 对齐到 LLM embedding
        emb_layer = base_model.get_input_embeddings()
        device = emb_layer.weight.device
        dtype = emb_layer.weight.dtype

        point_tokens = point_tokens.to(device=device, dtype=dtype)
        text_mask = text_mask.to(device=device)

        # input_ids 必须在这里用来定位 <point> token
        input_ids = inputs.get("input_ids", None)
        if input_ids is None:
            raise RuntimeError("input_ids missing from inputs.")
        input_ids = input_ids.to(device=device)

        # 1) 跑 AE：point_tokens -> (B, max_text_len, H)
        # 关键：为了更快，我们只走 point->text 路径，不跑 text_encoder
        point_latents = point_ae.point_encoder(point_tokens)
        pred_full = point_ae.shared_text_decoder(point_latents, target_mask=text_mask)  # (B, L, H)

        # 2) 计算每个样本实际注入序列，并拼成 flat source（用于 masked_scatter）
        B, S = input_ids.shape
        H = int(pred_full.shape[-1])

        seq_list: List[torch.Tensor] = []
        expected_counts: List[int] = []

        for b in range(B):
            k = int(inject_len[b].item())
            k = max(1, k)

            # 取变长 token embeds：pred_full[b][mask]
            mb = text_mask[b]
            seq = pred_full[b][mb]  # (k_full, H)
            if seq.shape[0] < k:
                # 防御：避免越界
                k = int(seq.shape[0])
            seq = seq[:k]  # (k, H)

            # prompt 里必须有 k 个 <point>
            pos = (input_ids[b] == int(point_token_id)).nonzero(as_tuple=False).view(-1)
            if pos.numel() != k:
                raise RuntimeError(
                    f"<point> count mismatch in sample b={b}: "
                    f"prompt_has={pos.numel()} vs inject_len={k}. "
                    f"Fix dataset prompt construction."
                )

            seq_list.append(seq)
            expected_counts.append(k)

        # 拼成 (N, H)
        source = torch.cat(seq_list, dim=0)  # (sum_k, H)

        # 3) 构造 base embeds + 可微 scatter 注入
        with torch.no_grad():
            base_embeds = emb_layer(input_ids)  # (B,S,H), 不需要对 embedding 权重求梯度
        base_embeds = base_embeds.to(dtype=dtype)

        # mask: (B,S,H)
        point_mask_2d = (input_ids == int(point_token_id))  # (B,S)
        point_mask_3d = point_mask_2d.unsqueeze(-1).expand(B, S, H)

        # masked_scatter 可微（梯度回到 source，从而回到 AE）
        inputs_embeds = base_embeds.masked_scatter(point_mask_3d, source)

        # 4) 避免某些模型 forward 同时给 input_ids + inputs_embeds
        inputs.pop("input_ids", None)

        # 只返回需要 update 的字段（ms-swift 会 merge）
        return {"inputs_embeds": inputs_embeds}


def register_qwen3_omni_point_template(exists_ok: bool = True) -> None:
    register_template(
        TemplateMeta(
            template_type="qwen3_omni_point",
            # Qwen 系列通用格式（和 ms-swift 内置 qwen 模板一致的风格）
            prefix=[],
            prompt=["<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n"],
            chat_sep=["<|im_end|>\n"],
            suffix=["<|im_end|>"],
            system_prefix=["<|im_start|>system\n{{SYSTEM}}<|im_end|>\n"],
            default_system=DEFAULT_SYSTEM_PROMPT,
            auto_add_bos=True,
        ),
        template_cls=Qwen3OmniPointTemplate,
        exists_ok=exists_ok,
    )
