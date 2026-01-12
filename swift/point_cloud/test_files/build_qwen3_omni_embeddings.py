#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Precompute token-level *input embeddings* (embedding-table lookup) for all assistant
messages in a SWIFT(ms-swift) registered dataset.

Outputs (under --output_dir):
  - raw.jsonl.gz                 : original samples (one json per line), with _sample_idx added
  - assistant_meta.parquet/jsonl : per-assistant-message metadata
  - embeddings.f16.bin           : flattened token embeddings (float16), shape (total_tokens, hidden_size)
  - token_ids.i32.bin            : flattened token ids (int32), length total_tokens
  - offsets.npy                  : int64 prefix-sum offsets, length (num_messages + 1)
  - info.json                    : metadata for loading
"""

import argparse
import gzip
import json
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    _HAVE_ARROW = True
except Exception:
    _HAVE_ARROW = False


# -----------------------------
# Worker-side global tokenizer
# -----------------------------
_WORKER_TOKENIZER = None
_WORKER_MAX_LENGTH = None
_WORKER_ADD_SPECIAL = None


def _worker_init(tokenizer_dir: str, max_length: int, add_special_tokens: bool) -> None:
    """
    Initializer for each tokenizer worker process: load tokenizer once.
    """
    global _WORKER_TOKENIZER, _WORKER_MAX_LENGTH, _WORKER_ADD_SPECIAL

    # If you use multiple processes, avoid per-process internal tokenizer threading oversubscription.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    tok = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True, use_fast=True)
    if tok.pad_token_id is None:
        # causal LMs often have no pad token; use eos/unk as pad for padding only
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.pad_token = tok.unk_token

    _WORKER_TOKENIZER = tok
    _WORKER_MAX_LENGTH = int(max_length)
    _WORKER_ADD_SPECIAL = bool(add_special_tokens)


def _tokenize_texts(texts: List[str]) -> List[List[int]]:
    """
    Tokenize a list of strings -> list[list[int]] input_ids.
    """
    enc = _WORKER_TOKENIZER(
        texts,
        add_special_tokens=_WORKER_ADD_SPECIAL,
        truncation=True,
        max_length=_WORKER_MAX_LENGTH,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return enc["input_ids"]


# -----------------------------
# Utilities
# -----------------------------
def normalize_content(content: Any) -> str:
    """
    Robustly convert message content into a text string.
    - If content is str: return it
    - If content is list (multimodal-like): extract text fields if possible, else dump json
    - Otherwise: str(content)
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for x in content:
            if isinstance(x, str):
                parts.append(x)
            elif isinstance(x, dict):
                # common format: {"type":"text","text":"..."}
                if x.get("type") == "text" and "text" in x:
                    parts.append(str(x["text"]))
                elif "text" in x:
                    parts.append(str(x["text"]))
        if parts:
            return "\n".join(parts)
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def resolve_model_dir(model_name_or_path: str) -> str:
    """
    Prefer local directory. If not exists, try huggingface_hub snapshot_download.
    (For huge Qwen3-Omni models, you typically want to pass a local path.)
    """
    p = Path(model_name_or_path)
    if p.exists():
        return str(p.resolve())

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise RuntimeError(
            f"Model path '{model_name_or_path}' not found and huggingface_hub is unavailable. "
            f"Please pass a local model directory."
        ) from e

    # Only download necessary files (tokenizer + safetensors + index/json)
    return snapshot_download(
        repo_id=model_name_or_path,
        allow_patterns=[
            "model.safetensors*",
            "*.safetensors",
            "*.json",
            "tokenizer.*",
            "vocab.*",
            "merges.*",
            "*.model",
            "*.txt",
        ],
    )


def find_embedding_weight(model_dir: str) -> Tuple[str, str]:
    """
    Find the token embedding matrix tensor key and which safetensors shard contains it.
    Returns: (tensor_key, safetensors_file_path)

    We try common names first, then heuristics.
    """
    model_path = Path(model_dir)
    index_path = model_path / "model.safetensors.index.json"

    preferred_keys = [
        "model.embed_tokens.weight",
        "embed_tokens.weight",
        "tok_embeddings.weight",
        "transformer.wte.weight",
    ]

    if index_path.exists():
        data = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = data.get("weight_map", {})

        for k in preferred_keys:
            if k in weight_map:
                return k, str((model_path / weight_map[k]).resolve())

        candidates: List[Tuple[str, str]] = []
        for k, fname in weight_map.items():
            if k.endswith("embed_tokens.weight") or "embed_tokens.weight" in k:
                candidates.append((k, fname))
            elif k.endswith("tok_embeddings.weight") or "tok_embeddings.weight" in k:
                candidates.append((k, fname))
            elif k.endswith("wte.weight") or ".wte.weight" in k:
                candidates.append((k, fname))

        if not candidates:
            raise RuntimeError(f"Cannot find embedding weight key from {index_path}")

        k, fname = candidates[0]
        return k, str((model_path / fname).resolve())

    # single-file safetensors
    single = model_path / "model.safetensors"
    if single.exists():
        with safe_open(str(single), framework="pt", device="cpu") as f:
            keys = list(f.keys())

        for k in preferred_keys:
            if k in keys:
                return k, str(single.resolve())

        for k in keys:
            if k.endswith("embed_tokens.weight") or "embed_tokens.weight" in k:
                return k, str(single.resolve())
            if k.endswith("tok_embeddings.weight") or "tok_embeddings.weight" in k:
                return k, str(single.resolve())
            if k.endswith("wte.weight") or ".wte.weight" in k:
                return k, str(single.resolve())

        raise RuntimeError(f"Cannot find embedding weight key in {single}")

    raise RuntimeError(f"No safetensors found in: {model_dir}")


def load_embedding_table(model_name_or_path: str, device: torch.device, out_dtype: torch.dtype):
    """
    Load tokenizer + embedding table only (no full forward model).
    """
    model_dir = resolve_model_dir(model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.unk_token

    emb_key, emb_file = find_embedding_weight(model_dir)

    # Read only that tensor (no need to load whole shard to memory)
    with safe_open(emb_file, framework="pt", device="cpu") as f:
        weight = f.get_tensor(emb_key)  # torch.Tensor on CPU

    # Create Embedding layer
    emb = torch.nn.Embedding.from_pretrained(weight, freeze=True)
    emb = emb.to(device=device, dtype=out_dtype)

    info = {
        "model_name_or_path": model_name_or_path,
        "model_dir": model_dir,
        "embedding_weight_key": emb_key,
        "embedding_weight_file": emb_file,
        "vocab_size": int(weight.shape[0]),
        "hidden_size": int(weight.shape[1]),
        "embedding_runtime_dtype": str(out_dtype).replace("torch.", ""),
        "pad_token_id": int(tokenizer.pad_token_id),
    }
    return tokenizer, emb, info


class RawJsonlGzWriter:
    """
    Save original dataset samples in jsonl.gz for traceability.
    """
    def __init__(self, path: Path, compresslevel: int = 1):
        self.path = path
        self.f = gzip.open(path, "wt", encoding="utf-8", compresslevel=compresslevel)

    def write(self, sample: Dict[str, Any]) -> None:
        self.f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self.f.close()


class MetaWriter:
    """
    Write per-assistant-message metadata either to parquet (preferred) or jsonl.gz (fallback).
    """
    def __init__(self, path: Path, compresslevel: int = 1):
        self.path = path
        self._mode = "parquet" if (_HAVE_ARROW and path.suffix == ".parquet") else "jsonl"

        self._writer = None
        self._schema = None

        if self._mode == "jsonl":
            if path.name.endswith(".jsonl.gz"):
                self.path = path
            else:
                self.path = path.with_suffix(".jsonl.gz")
            self._f = gzip.open(self.path, "wt", encoding="utf-8", compresslevel=compresslevel)

    def write_rows(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return

        if self._mode == "parquet":
            table = pa.Table.from_pylist(rows) if self._schema is None else pa.Table.from_pylist(rows, schema=self._schema)
            if self._writer is None:
                self._schema = table.schema
                # zstd is usually a good balance; if unavailable in your env, switch to "snappy"
                try:
                    self._writer = pq.ParquetWriter(str(self.path), self._schema, compression="zstd")
                except Exception:
                    self._writer = pq.ParquetWriter(str(self.path), self._schema, compression="snappy")
            self._writer.write_table(table)
        else:
            for r in rows:
                self._f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def close(self) -> None:
        if self._mode == "parquet":
            if self._writer is not None:
                self._writer.close()
        else:
            self._f.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="SWIFT dataset id/path, e.g. xxx or ms::xxx")
    parser.add_argument("--model", type=str, required=True,
                        help="Qwen3-Omni model local dir or HF repo id")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--streaming", action="store_true", default=True)

    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--add_special_tokens", action="store_true", default=True)

    # Tokenization task size (texts per process call)
    parser.add_argument("--batch_texts", type=int, default=512)

    # GPU embedding batch size (texts per embedding lookup)
    parser.add_argument("--batch_gpu", type=int, default=128)

    # Parallel tokenization
    parser.add_argument("--num_tokenizer_workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument("--prefetch_batches", type=int, default=4)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])

    parser.add_argument("--raw_limit", type=int, default=None,
                        help="Debug: only process first N samples")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Import SWIFT dataset loader
    try:
        from swift.llm import load_dataset as swift_load_dataset
    except Exception as e:
        raise RuntimeError(
            "Cannot import swift.llm.load_dataset. Please install ms-swift, e.g.:\n"
            "  pip install 'ms-swift[llm]' -U"
        ) from e

    # Load tokenizer + embedding table
    device = torch.device(args.device)
    out_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    tokenizer, emb_layer, emb_info = load_embedding_table(args.model, device=device, out_dtype=out_dtype)

    # Output file paths
    raw_path = out_dir / "raw.jsonl.gz"
    meta_path = out_dir / ("assistant_meta.parquet" if _HAVE_ARROW else "assistant_meta.jsonl.gz")

    # We store embeddings as float16 if dtype is float16/bfloat16; otherwise float32.
    emb_bin_name = "embeddings.f16.bin" if args.dtype in ("float16", "bfloat16") else "embeddings.f32.bin"
    emb_path = out_dir / emb_bin_name

    tok_path = out_dir / "token_ids.i32.bin"
    offsets_path = out_dir / "offsets.npy"
    info_path = out_dir / "info.json"

    raw_writer = RawJsonlGzWriter(raw_path, compresslevel=1)
    meta_writer = MetaWriter(meta_path, compresslevel=1)

    emb_f = open(emb_path, "wb")
    tok_f = open(tok_path, "wb")

    offsets: List[int] = [0]
    assistant_record_idx = 0

    # Load dataset (keep your calling style)
    dataset = swift_load_dataset([args.dataset], seed=args.seed, streaming=args.streaming, remove_unused_columns=False)[0]

    # Tokenization pool (spawn is safer with CUDA)
    ctx = mp.get_context("spawn")
    executor = ProcessPoolExecutor(
        max_workers=args.num_tokenizer_workers,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(emb_info["model_dir"], args.max_length, args.add_special_tokens),
    )

    pending: List[Tuple[Any, List[Dict[str, Any]]]] = []
    text_buf: List[str] = []
    meta_buf: List[Dict[str, Any]] = []

    def submit_tokenize_task() -> None:
        nonlocal text_buf, meta_buf, pending
        if not text_buf:
            return
        texts = text_buf
        metas = meta_buf
        text_buf = []
        meta_buf = []
        fut = executor.submit(_tokenize_texts, texts)
        pending.append((fut, metas))

    def process_tokenized_batch(input_ids_list: List[List[int]], metas: List[Dict[str, Any]]) -> None:
        """
        Convert token ids -> embeddings using embedding table, write binaries + meta + offsets.
        """
        nonlocal assistant_record_idx, offsets

        pad_id = int(tokenizer.pad_token_id)
        hidden = int(emb_layer.weight.shape[1])

        # sub-batch on GPU to control memory
        for start in range(0, len(input_ids_list), args.batch_gpu):
            sub_ids = input_ids_list[start:start + args.batch_gpu]
            sub_metas = metas[start:start + args.batch_gpu]

            lengths = torch.tensor([len(x) for x in sub_ids], device=device, dtype=torch.long)
            max_len = int(lengths.max().item()) if len(sub_ids) > 0 else 0

            if max_len == 0:
                for m in sub_metas:
                    m["assistant_record_idx"] = assistant_record_idx
                    m["n_tokens"] = 0
                    m["token_offset"] = offsets[-1]
                    offsets.append(offsets[-1])
                    assistant_record_idx += 1
                meta_writer.write_rows(sub_metas)
                continue

            input_ids = torch.full((len(sub_ids), max_len), pad_id, dtype=torch.long, device=device)
            for i, ids in enumerate(sub_ids):
                if ids:
                    input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)

            with torch.inference_mode():
                emb = emb_layer(input_ids)  # (B, L, H)
                arange = torch.arange(max_len, device=device)[None, :]
                mask = arange < lengths[:, None]  # (B, L)
                flat_emb = emb[mask]            # (sum_tokens, H)
                flat_ids = input_ids[mask]      # (sum_tokens,)

                # Save embeddings as float16 for size/speed (even if runtime is bf16)
                save_emb_dtype = torch.float16 if args.dtype in ("float16", "bfloat16") else torch.float32
                flat_emb_cpu = flat_emb.to(save_emb_dtype).cpu().numpy()
                flat_ids_cpu = flat_ids.to(torch.int32).cpu().numpy()

            flat_emb_cpu.tofile(emb_f)
            flat_ids_cpu.tofile(tok_f)

            # Update offsets + write meta
            cursor = offsets[-1]
            for i, m in enumerate(sub_metas):
                n = int(lengths[i].item())
                m["assistant_record_idx"] = assistant_record_idx
                m["n_tokens"] = n
                m["token_offset"] = cursor
                cursor += n
                offsets.append(cursor)
                assistant_record_idx += 1

            meta_writer.write_rows(sub_metas)

    # Iterate dataset
    sample_idx = 0
    pbar = tqdm(dataset, desc="Processing samples", unit="sample")
    for ex in pbar:
        if args.raw_limit is not None and sample_idx >= args.raw_limit:
            break

        raw_sample = dict(ex)
        raw_sample["_sample_idx"] = sample_idx
        raw_writer.write(raw_sample)

        messages = ex.get("messages", [])
        if isinstance(messages, list):
            for mi, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    continue
                if msg.get("role") != "assistant":
                    continue
                text = normalize_content(msg.get("content"))
                if not text:
                    continue

                text_buf.append(text)
                meta_buf.append(
                    {
                        "sample_idx": sample_idx,
                        "object_id": ex.get("object_id", None),
                        "message_idx": mi,
                        "content": text,
                    }
                )

                if len(text_buf) >= args.batch_texts:
                    submit_tokenize_task()

                    # Keep inflight bounded; process in submission order to preserve ordering
                    while len(pending) >= args.prefetch_batches:
                        fut0, metas0 = pending.pop(0)
                        input_ids0 = fut0.result()
                        process_tokenized_batch(input_ids0, metas0)

        sample_idx += 1

    # Flush remaining
    submit_tokenize_task()
    while pending:
        fut0, metas0 = pending.pop(0)
        input_ids0 = fut0.result()
        process_tokenized_batch(input_ids0, metas0)

    executor.shutdown(wait=True, cancel_futures=False)

    raw_writer.close()
    meta_writer.close()
    emb_f.close()
    tok_f.close()

    np.save(offsets_path, np.asarray(offsets, dtype=np.int64))

    total_tokens = int(offsets[-1]) if offsets else 0
    emb_info.update(
        {
            "max_length": int(args.max_length),
            "add_special_tokens": bool(args.add_special_tokens),
            "num_samples": int(sample_idx),
            "num_assistant_messages": int(assistant_record_idx),
            "total_tokens": total_tokens,
            "raw_file": raw_path.name,
            "meta_file": meta_writer.path.name,
            "embeddings_file": emb_path.name,
            "token_ids_file": tok_path.name,
            "offsets_file": offsets_path.name,
            "embeddings_storage_dtype": "float16" if args.dtype in ("float16", "bfloat16") else "float32",
        }
    )

    info_path.write_text(json.dumps(emb_info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDone. Saved to: {out_dir}")


if __name__ == "__main__":
    main()
