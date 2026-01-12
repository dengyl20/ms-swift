# -*- coding: utf-8 -*-
"""
Qwen3-Omni Text Encoder Utilities
================================

Goals:
1) Load Qwen/Qwen3-Omni-30B-A3B-Instruct "thinker" (text-only) part via Transformers.
2) Convert text -> token ids -> embeddings:
   - token input embeddings (inputs_embeds) from model's embedding table
   - (optional) contextual last_hidden_state and pooled sentence embedding
3) Provide a sanity test:
   logits(input_ids=...) == logits(inputs_embeds=...) within tolerance.

Notes:
- Official Qwen3-Omni repo recommends transformers==4.57.3. (See README) 
- HF docs confirm thinker forward supports `inputs_embeds`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch

# These are available in transformers>=4.57.3 for Qwen3-Omni (per official README).
from transformers import Qwen3OmniMoeProcessor, Qwen3OmniMoeThinkerForConditionalGeneration


PoolingStrategy = Literal["last_token", "mean"]


@dataclass(frozen=True)
class TextEmbeddingOutput:
    """
    Container for text encoder outputs.

    input_ids / attention_mask:
        Tokenized model inputs.

    inputs_embeds:
        Token embeddings produced by model.get_input_embeddings()(input_ids).
        Shape: [B, S, H]

    last_hidden_state (optional):
        Contextualized hidden states from the last layer.
        Shape: [B, S, H]

    pooled (optional):
        Sentence-level embedding pooled from last_hidden_state.
        Shape: [B, H]
    """
    input_ids: torch.LongTensor
    attention_mask: Optional[torch.LongTensor]
    inputs_embeds: torch.FloatTensor
    last_hidden_state: Optional[torch.FloatTensor] = None
    pooled: Optional[torch.FloatTensor] = None


class Qwen3OmniTextEncoder:
    """
    A lightweight wrapper around Qwen3-Omni Thinker model for text embedding extraction.
    """

    def __init__(
        self,
        model: Qwen3OmniMoeThinkerForConditionalGeneration,
        processor: Qwen3OmniMoeProcessor,
    ) -> None:
        self.model = model.eval()
        self.processor = processor

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        device_map: Union[str, Dict[str, Union[int, str]]] = "auto",
        dtype: Union[str, torch.dtype] = "auto",
        attn_implementation: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> "Qwen3OmniTextEncoder":
        """
        Load the text-only thinker model + processor.

        Why thinker?
        - HF docs: use ThinkerForConditionalGeneration for text-only to avoid loading audio model. 
        - For embeddings, we only need the text stack.

        Parameters
        ----------
        model_name_or_path:
            HF repo id or local path.
        device_map:
            "auto" recommended for large models.
        dtype:
            "auto" / torch.float16 / torch.bfloat16, etc.
        attn_implementation:
            e.g. "flash_attention_2" if available.
        trust_remote_code:
            Typically False for official merged implementations.
        """
        # The official Qwen3-Omni README uses argument name `dtype="auto"`.
        # Some Transformers versions also accept torch_dtype=...; we do a small compatibility fallback.
        kwargs: Dict[str, Any] = dict(
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation

        try:
            model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
                model_name_or_path,
                dtype=dtype,
                **kwargs,
            )
        except TypeError:
            # Fallback for older signatures
            model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
                model_name_or_path,
                torch_dtype=dtype,
                **kwargs,
            )

        processor = Qwen3OmniMoeProcessor.from_pretrained(model_name_or_path)
        return cls(model=model, processor=processor)

    def _input_device(self) -> torch.device:
        """
        Determine which device the token embedding table lives on.
        Inputs (input_ids / inputs_embeds) must be placed on that device.
        """
        emb = self.model.get_input_embeddings()
        return emb.weight.device

    def build_conversations(
        self,
        texts: Sequence[str],
        system_prompt: Optional[str] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Build Qwen3-Omni style conversations for pure-text input.
        We keep the multimodal-friendly structure:
            [{"role": "...", "content": [{"type": "text", "text": "..."}]}]

        Returns
        -------
        conversations: list of conversations (batch)
        """
        conversations: List[List[Dict[str, Any]]] = []
        for t in texts:
            conv: List[Dict[str, Any]] = []
            if system_prompt:
                conv.append(
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt}],
                    }
                )
            conv.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": t}],
                }
            )
            conversations.append(conv)
        return conversations

    def tokenize(
        self,
        texts: Sequence[str],
        system_prompt: Optional[str] = None,
        add_generation_prompt: bool = False,
        padding: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize texts into model inputs using processor.apply_chat_template(... tokenize=True ...).

        This matches the official Qwen3-Omni usage style for Instruct models (chat template).
        """
        conversations = self.build_conversations(texts, system_prompt=system_prompt)

        # processor.apply_chat_template can directly return tensors when tokenize=True
        batch = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_dict=True,
            return_tensors=return_tensors,
            padding=padding,
        )

        # Move tensors to the embedding device
        device = self._input_device()
        for k, v in list(batch.items()):
            if torch.is_tensor(v):
                batch[k] = v.to(device)
        return batch

    @torch.inference_mode()
    def encode(
        self,
        texts: Sequence[str],
        system_prompt: Optional[str] = None,
        add_generation_prompt: bool = False,
        return_last_hidden_state: bool = True,
        pooling: PoolingStrategy = "last_token",
    ) -> TextEmbeddingOutput:
        """
        Encode texts and return embeddings.

        Returns:
        - inputs_embeds (token-level) always
        - last_hidden_state + pooled embedding optionally
        """
        batch = self.tokenize(
            texts=texts,
            system_prompt=system_prompt,
            add_generation_prompt=add_generation_prompt,
            padding=True,
        )

        input_ids: torch.LongTensor = batch["input_ids"]
        attention_mask: Optional[torch.LongTensor] = batch.get("attention_mask", None)

        # 1) Token input embeddings (this is what you'll feed via inputs_embeds=...)
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        last_hidden_state: Optional[torch.FloatTensor] = None
        pooled: Optional[torch.FloatTensor] = None

        # 2) (Optional) Contextual embeddings
        if return_last_hidden_state:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            # For CausalLM, hidden_states are provided when output_hidden_states=True
            if outputs.hidden_states is None:
                raise RuntimeError("Model did not return hidden_states. Check Transformers version/config.")
            last_hidden_state = outputs.hidden_states[-1]  # [B,S,H]
            pooled = self._pool_hidden(last_hidden_state, attention_mask, pooling=pooling)

        return TextEmbeddingOutput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            last_hidden_state=last_hidden_state,
            pooled=pooled,
        )

    def _pool_hidden(
        self,
        last_hidden_state: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor],
        pooling: PoolingStrategy,
    ) -> torch.FloatTensor:
        """
        Pool token hidden states into a sentence embedding.
        """
        if pooling == "mean":
            if attention_mask is None:
                return last_hidden_state.mean(dim=1)
            mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # [B,S,1]
            denom = mask.sum(dim=1).clamp(min=1.0)
            return (last_hidden_state * mask).sum(dim=1) / denom

        # default: last_token
        if attention_mask is None:
            return last_hidden_state[:, -1, :]
        lengths = attention_mask.long().sum(dim=1)  # [B]
        # last valid token index = lengths - 1
        idx = (lengths - 1).clamp(min=0)
        bsz = last_hidden_state.size(0)
        return last_hidden_state[torch.arange(bsz, device=last_hidden_state.device), idx, :]

    @torch.inference_mode()
    def sanity_check_inputs_embeds_equivalence(
        self,
        text: str,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Sanity test:
            logits(input_ids=...)  ~=  logits(inputs_embeds=...)
        This validates that the extracted inputs_embeds correspond exactly to the embedding lookup.

        Returns a dict with:
            - max_abs_diff
            - is_allclose
            - used atol/rtol
        """
        batch = self.tokenize([text], padding=True)
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)

        # Choose tolerance based on model dtype (bf16/fp16 needs slightly larger atol)
        model_dtype = self.model.get_input_embeddings().weight.dtype
        if atol is None:
            atol = 1e-4 if model_dtype == torch.float32 else 1e-2
        if rtol is None:
            rtol = 1e-4 if model_dtype == torch.float32 else 1e-2

        # Forward with input_ids
        out_ids = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        logits_ids = out_ids.logits

        # Forward with inputs_embeds
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        out_emb = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        logits_emb = out_emb.logits

        diff = (logits_ids - logits_emb).abs()
        max_abs_diff = diff.max().item()

        is_allclose = torch.allclose(logits_ids, logits_emb, atol=atol, rtol=rtol)

        # Extra: ensure argmax tokens identical (a practical check)
        pred_ids = logits_ids.argmax(dim=-1)
        pred_emb = logits_emb.argmax(dim=-1)
        same_argmax = bool(torch.equal(pred_ids, pred_emb))

        return {
            "max_abs_diff": max_abs_diff,
            "is_allclose": bool(is_allclose),
            "same_argmax": same_argmax,
            "atol": float(atol),
            "rtol": float(rtol),
            "model_dtype": str(model_dtype),
            "logits_shape": tuple(logits_ids.shape),
        }


if __name__ == "__main__":
    # Example runnable demo (adjust to your environment)
    encoder = Qwen3OmniTextEncoder.from_pretrained(
        "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        device_map="auto",
        dtype="auto",
        attn_implementation="flash_attention_2",  # set "flash_attention_2" if you have it
    )
    p = next(encoder.parameters())
    print("param dtype:", p.dtype, "device:", p.device)

    out = encoder.encode(
        texts=["告诉我33乘33等于多少"],
        system_prompt=None,
        return_last_hidden_state=True,
        pooling="last_token",
    )
    print(out)
    print("inputs_embeds:", out.inputs_embeds.shape)
    if out.pooled is not None:
        print("pooled:", out.pooled.shape)

    report = encoder.sanity_check_inputs_embeds_equivalence("你好，Qwen3-Omni！")
    print("sanity_check:", report)
