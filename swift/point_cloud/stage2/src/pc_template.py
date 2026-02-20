# pc_template.py
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from swift.point_cloud.stage2.src.pc_constants import DEFAULT_SYSTEM_PROMPT, POINT_TOKEN

from swift.template import Template, TemplateMeta, register_template
from swift.utils import get_logger

logger = get_logger()


def _as_torch(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return torch.tensor(x)


def _get_underlying_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "module"):
        return model.module  # type: ignore
    return model


def _stack_if_list(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)

    if isinstance(x, list):
        if len(x) == 0:
            return torch.empty(0)
        if isinstance(x[0], (torch.Tensor, np.ndarray, list)):
            return torch.stack([_as_torch(v) for v in x], dim=0)
        return torch.tensor(x)

    return torch.tensor(x)


class Qwen3OmniPointTemplate(Template):
    """
    你的原 AE 注入模板（不改逻辑）
    """
    use_model = True

    @contextmanager
    def forward_context(self, model: nn.Module, inputs: Dict[str, Any]):
        with super().forward_context(model, inputs):
            if "point_tokens" in inputs:
                updates = self._post_encode(model, inputs)
                if updates:
                    inputs.update(updates)
            yield

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)

        if "point_tokens" in batch[0]:
            pts = [_as_torch(b["point_tokens"]) for b in batch]
            res["point_tokens"] = torch.stack(pts, dim=0)

        if "text_mask" in batch[0]:
            tms = [_as_torch(b["text_mask"]).bool() for b in batch]
            res["text_mask"] = torch.stack(tms, dim=0)

        if "inject_len" in batch[0]:
            ks = torch.tensor([int(b["inject_len"]) for b in batch], dtype=torch.long)
            res["inject_len"] = ks

        if "object_id" in batch[0]:
            res["object_id"] = [str(b["object_id"]) for b in batch]

        return res

    def _post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        base_model = _get_underlying_model(model)

        point_tokens = inputs.pop("point_tokens", None)
        text_mask = inputs.pop("text_mask", None)
        inject_len = inputs.pop("inject_len", None)

        if point_tokens is None:
            raise RuntimeError("point_tokens is missing in inputs.")
        if text_mask is None or inject_len is None:
            raise RuntimeError("Missing text_mask/inject_len in batch. Check dataset loader & collator.")
        if not hasattr(base_model, "point_ae"):
            raise RuntimeError("Model has no attribute 'point_ae'. Please use model_type=qwen3_omni_point.")

        point_ae = getattr(base_model, "point_ae")

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Template.processor.tokenizer is missing.")
        point_token_id = tokenizer.convert_tokens_to_ids(POINT_TOKEN)
        if point_token_id is None or int(point_token_id) < 0:
            raise RuntimeError(f"POINT_TOKEN '{POINT_TOKEN}' not found in tokenizer vocab. Check ModelLoader.")

        emb_layer = base_model.get_input_embeddings()
        device = emb_layer.weight.device
        dtype = emb_layer.weight.dtype

        point_tokens = _stack_if_list(point_tokens).to(device=device, dtype=dtype)
        text_mask = _stack_if_list(text_mask).to(device=device).bool()
        inject_len = _stack_if_list(inject_len).to(device=device, dtype=torch.long)

        input_ids = inputs.get("input_ids", None)
        if input_ids is None:
            raise RuntimeError("input_ids missing from inputs.")
        input_ids = input_ids.to(device=device)

        point_latents = point_ae.point_encoder(point_tokens)
        pred_full = point_ae.shared_text_decoder(point_latents, target_mask=text_mask)  # (B, L, H)

        B, S = input_ids.shape
        H = int(pred_full.shape[-1])

        seq_list: List[torch.Tensor] = []
        for b in range(B):
            k = int(inject_len[b].item())
            k = max(1, k)

            mb = text_mask[b]
            seq = pred_full[b][mb]
            if seq.shape[0] < k:
                k = int(seq.shape[0])
            seq = seq[:k]

            pos = (input_ids[b] == int(point_token_id)).nonzero(as_tuple=False).view(-1)
            if pos.numel() != k:
                raise RuntimeError(
                    f"<point> count mismatch in sample b={b}: prompt_has={pos.numel()} vs inject_len={k}."
                )
            seq_list.append(seq)

        source = torch.cat(seq_list, dim=0)  # (sum_k, H)

        with torch.no_grad():
            base_embeds = emb_layer(input_ids)  # (B,S,H)
        base_embeds = base_embeds.to(dtype=dtype)

        point_mask_2d = (input_ids == int(point_token_id))
        point_mask_3d = point_mask_2d.unsqueeze(-1).expand(B, S, H)

        inputs_embeds = base_embeds.masked_scatter(point_mask_3d, source)

        inputs.pop("input_ids", None)
        return {"inputs_embeds": inputs_embeds}


class Qwen3OmniPointMLPTemplate(Template):
    """
    新 baseline 模板：point_tokens -> 2-layer MLP projector -> 注入 <point> 的 inputs_embeds
    """
    use_model = True

    @contextmanager
    def forward_context(self, model: nn.Module, inputs: Dict[str, Any]):
        with super().forward_context(model, inputs):
            if "point_tokens" in inputs:
                updates = self._post_encode(model, inputs)
                if updates:
                    inputs.update(updates)
            yield

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator_mm_data(batch)

        if "point_tokens" in batch[0]:
            pts = [_as_torch(b["point_tokens"]) for b in batch]
            res["point_tokens"] = torch.stack(pts, dim=0)

        # 仍然 collate 并在 _post_encode pop 掉，避免传进 model.forward 导致 unexpected kwarg
        if "text_mask" in batch[0]:
            tms = [_as_torch(b["text_mask"]).bool() for b in batch]
            res["text_mask"] = torch.stack(tms, dim=0)

        if "inject_len" in batch[0]:
            ks = torch.tensor([int(b["inject_len"]) for b in batch], dtype=torch.long)
            res["inject_len"] = ks

        if "object_id" in batch[0]:
            res["object_id"] = [str(b["object_id"]) for b in batch]

        return res

    def _post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        base_model = _get_underlying_model(model)

        point_tokens = inputs.pop("point_tokens", None)
        inputs.pop("text_mask", None)  # baseline 不用，但必须 pop 掉
        inject_len = inputs.pop("inject_len", None)

        if point_tokens is None:
            raise RuntimeError("point_tokens is missing in inputs.")
        if inject_len is None:
            raise RuntimeError("inject_len is missing in inputs. Check dataset loader & collator.")

        if hasattr(base_model, "point_projector"):
            point_projector = getattr(base_model, "point_projector")
        elif hasattr(base_model, "mm_projector"):
            point_projector = getattr(base_model, "mm_projector")
        else:
            raise RuntimeError(
                "Model has no attribute 'point_projector' (or 'mm_projector'). "
                "Please use model_type=qwen3_omni_point_mlp."
            )

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Template.processor.tokenizer is missing.")
        point_token_id = tokenizer.convert_tokens_to_ids(POINT_TOKEN)
        if point_token_id is None or int(point_token_id) < 0:
            raise RuntimeError(f"POINT_TOKEN '{POINT_TOKEN}' not found in tokenizer vocab. Check ModelLoader.")

        emb_layer = base_model.get_input_embeddings()
        device = emb_layer.weight.device
        dtype = emb_layer.weight.dtype

        point_tokens = _stack_if_list(point_tokens).to(device=device, dtype=dtype)
        inject_len = _stack_if_list(inject_len).to(device=device, dtype=torch.long)

        input_ids = inputs.get("input_ids", None)
        if input_ids is None:
            raise RuntimeError("input_ids missing from inputs.")
        input_ids = input_ids.to(device=device)

        # (B,G,D) -> (B,G,H)
        projected = point_projector(point_tokens)
        if projected.dim() != 3:
            raise RuntimeError(f"point_projector output must be (B,G,H), got shape={tuple(projected.shape)}")

        B, S = input_ids.shape
        _, G, H = projected.shape

        seq_list: List[torch.Tensor] = []
        for b in range(B):
            k = int(inject_len[b].item())
            k = max(1, k)

            pos = (input_ids[b] == int(point_token_id)).nonzero(as_tuple=False).view(-1)
            if pos.numel() != k:
                raise RuntimeError(
                    f"<point> count mismatch in sample b={b}: prompt_has={pos.numel()} vs inject_len={k}."
                )

            seq = projected[b]  # (G,H)
            if seq.shape[0] < k:
                # 理论上不应发生（你 max_inject_tokens=24，G 通常远大于 24），但做个防御
                if seq.shape[0] == 0:
                    seq = torch.zeros((k, H), device=device, dtype=dtype)
                else:
                    pad = seq[-1:].expand(k - seq.shape[0], -1)
                    seq = torch.cat([seq, pad], dim=0)

            seq_list.append(seq[:k])

        source = torch.cat(seq_list, dim=0)  # (sum_k, H)

        with torch.no_grad():
            base_embeds = emb_layer(input_ids)  # (B,S,H)
        base_embeds = base_embeds.to(dtype=dtype)

        point_mask_2d = (input_ids == int(point_token_id))
        point_mask_3d = point_mask_2d.unsqueeze(-1).expand(B, S, H)

        inputs_embeds = base_embeds.masked_scatter(point_mask_3d, source)

        inputs.pop("input_ids", None)
        return {"inputs_embeds": inputs_embeds}


def register_qwen3_omni_point_template(exists_ok: bool = True) -> None:
    register_template(
        TemplateMeta(
            template_type="qwen3_omni_point_cloud",
            prefix=[],
            prompt=["<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n"],
            chat_sep=["<|im_end|>\n"],
            suffix=["<|im_end|>"],
            system_prefix=["<|im_start|>system\n{{SYSTEM}}<|im_end|>\n"],
            default_system=DEFAULT_SYSTEM_PROMPT,
            auto_add_bos=True,
            template_cls=Qwen3OmniPointTemplate,
        ),
        exist_ok=exists_ok,
    )


def register_qwen3_omni_point_mlp_template(exists_ok: bool = True) -> None:
    register_template(
        TemplateMeta(
            template_type="qwen3_omni_point_cloud_mlp",
            prefix=[],
            prompt=["<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n"],
            chat_sep=["<|im_end|>\n"],
            suffix=["<|im_end|>"],
            system_prefix=["<|im_start|>system\n{{SYSTEM}}<|im_end|>\n"],
            default_system=DEFAULT_SYSTEM_PROMPT,
            auto_add_bos=True,
            template_cls=Qwen3OmniPointMLPTemplate,
        ),
        exist_ok=exists_ok,
    )
