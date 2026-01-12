# -*- coding: utf-8 -*-
from __future__ import annotations

import difflib
from typing import Dict, Any, Optional, Tuple

import torch

from qwen3_omni_text_encoder import Qwen3OmniTextEncoder


@torch.inference_mode()
def greedy_generate_with_cache(
    model,
    *,
    input_ids: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    max_new_tokens: int = 64,
    eos_token_id: Optional[int] = None,
) -> torch.LongTensor:
    """
    A minimal greedy decoder that supports either:
      - input_ids (prompt tokens), or
      - inputs_embeds (prompt embeddings, full sequence or 1-token prefix)

    Returns the full sequence token ids *when input_ids is provided*.
    When only inputs_embeds is provided (no input_ids), it returns ONLY the generated tokens
    (since the prefix is not a real token id sequence).
    """
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("Provide exactly one of input_ids or inputs_embeds.")

    # First forward: build cache from prompt/prefix
    out = model(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )
    past = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [B,1]

    generated = [next_token]

    # Continue generating
    for _ in range(max_new_tokens - 1):
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break
        out = model(
            input_ids=next_token,  # feed last generated token
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token)

    gen_tokens = torch.cat(generated, dim=1)  # [B, T]

    if input_ids is not None:
        return torch.cat([input_ids, gen_tokens], dim=1)  # full sequence
    else:
        return gen_tokens  # only new tokens (prefix had no ids)


def decode_new_tokens(
    tokenizer,
    full_ids: torch.LongTensor,
    prompt_len: int,
) -> str:
    new_ids = full_ids[0, prompt_len:].tolist()
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def decode_tokens_only(
    tokenizer,
    ids: torch.LongTensor,
) -> str:
    return tokenizer.decode(ids[0].tolist(), skip_special_tokens=True).strip()


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


@torch.inference_mode()
def run_experiment(
    encoder: Qwen3OmniTextEncoder,
    prompt_text: str,
    *,
    add_generation_prompt: bool = True,
    max_new_tokens: int = 64,
) -> Dict[str, Any]:
    """
    Compare:
      A) generate(prompt via input_ids)
      B) generate(prompt via full inputs_embeds)  -> should match A (greedy)
      C) generate(from last_token hidden state only) -> expected to differ a lot
    """
    model = encoder.model
    processor = encoder.processor
    tok = processor.tokenizer

    # Tokenize with chat template so it behaves like instruction/chat generation.
    batch = encoder.tokenize(
        texts=[prompt_text],
        system_prompt=None,
        add_generation_prompt=add_generation_prompt,
        padding=True,
    )
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask", None)

    eos_id = tok.eos_token_id
    prompt_len = input_ids.shape[1]

    # ---- A) baseline: input_ids ----
    full_a = greedy_generate_with_cache(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_id,
    )
    text_a = decode_new_tokens(tok, full_a, prompt_len)

    # ---- B) equivalent: full inputs_embeds ----
    full_embeds = model.get_input_embeddings()(input_ids)
    full_b = greedy_generate_with_cache(
        model,
        inputs_embeds=full_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_id,
    )
    # B returns only generated tokens (because we didn't pass input_ids)
    text_b = decode_tokens_only(tok, full_b)

    # ---- C) only last_token hidden state (last layer) as a 1-token prefix ----
    # Get last hidden state for the same prompt
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    last_hidden = out.hidden_states[-1]  # [1, S, H]

    # last valid token index (avoid padding)
    if attention_mask is not None:
        last_idx = (attention_mask.long().sum(dim=1) - 1).clamp(min=0)  # [1]
        prefix = last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), last_idx, :]  # [1,H]
    else:
        prefix = last_hidden[:, -1, :]  # [1,H]

    prefix = prefix.unsqueeze(1).contiguous()  # [1,1,H]
    prefix_mask = torch.ones((prefix.size(0), 1), device=prefix.device, dtype=torch.long)

    gen_c = greedy_generate_with_cache(
        model,
        inputs_embeds=prefix,
        attention_mask=prefix_mask,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_id,
    )
    text_c = decode_tokens_only(tok, gen_c)

    return {
        "prompt_text": prompt_text,
        "add_generation_prompt": add_generation_prompt,
        "max_new_tokens": max_new_tokens,
        "A_input_ids": text_a,
        "B_full_inputs_embeds": text_b,
        "C_last_token_hidden_only": text_c,
        "sim_A_vs_B": similarity(text_a, text_b),
        "sim_A_vs_C": similarity(text_a, text_c),
        "prompt_len_tokens": int(prompt_len),
        "hidden_size": int(prefix.size(-1)),
    }


if __name__ == "__main__":
    encoder = Qwen3OmniTextEncoder.from_pretrained(
        "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        device_map="auto",
        dtype="auto",
        attn_implementation="flash_attention_2",
    )

    report = run_experiment(
        encoder,
        prompt_text="告诉我33乘33等于多少",
        add_generation_prompt=True,   # 建议 True：让模型进入 assistant 生成态
        max_new_tokens=64,
    )

    print("\n=== Experiment Report ===")
    print("Prompt:", report["prompt_text"])
    print("Prompt tokens:", report["prompt_len_tokens"], "Hidden size:", report["hidden_size"])
    print("\n[A] input_ids generation:\n", report["A_input_ids"])
    print("\n[B] full inputs_embeds generation:\n", report["B_full_inputs_embeds"])
    print("\n[C] last_token hidden only generation:\n", report["C_last_token_hidden_only"])
    print("\nSimilarity A vs B:", report["sim_A_vs_B"])
    print("Similarity A vs C:", report["sim_A_vs_C"])
