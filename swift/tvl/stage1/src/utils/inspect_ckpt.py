# inspect_ckpt.py
# Usage:
#   python inspect_ckpt.py /path/to/checkpoint.pt
#
# This script:
# - loads a .pt checkpoint (state_dict or dict containing state_dict)
# - prints total params, bytes, dtype breakdown
# - prints hierarchical module tree aggregated by key prefixes
# - prints top-K largest tensors

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


def _is_tensor_dict(x: Any) -> bool:
    return isinstance(x, dict) and len(x) > 0 and all(torch.is_tensor(v) for v in x.values())


def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    """
    Try best-effort extraction of a state_dict-like mapping from a checkpoint object.
    """
    if _is_tensor_dict(ckpt):
        return ckpt  # plain state_dict

    if isinstance(ckpt, dict):
        # common nesting keys
        candidate_keys = [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "module",
            "ema",
            "params",
            "weights",
        ]
        for k in candidate_keys:
            v = ckpt.get(k, None)
            if _is_tensor_dict(v):
                return v

        # sometimes: {"model": {"state_dict": ...}}
        for k, v in ckpt.items():
            if isinstance(v, dict):
                for kk in ("state_dict", "model_state_dict"):
                    vv = v.get(kk, None)
                    if _is_tensor_dict(vv):
                        return vv

    raise ValueError(
        "Could not find a state_dict (mapping str->Tensor) inside the checkpoint. "
        "If this checkpoint saves a full model object, consider saving state_dict instead."
    )


def _dtype_nbytes(dtype: torch.dtype) -> int:
    # torch.finfo/torch.iinfo both have bits but not for all dtypes (e.g. bool)
    if dtype == torch.bool:
        return 1
    try:
        return torch.tensor([], dtype=dtype).element_size()
    except Exception:
        # fallback
        return 0


@dataclass
class TensorEntry:
    name: str
    shape: Tuple[int, ...]
    numel: int
    dtype: torch.dtype
    nbytes: int


@dataclass
class TreeNode:
    name: str
    children: Dict[str, "TreeNode"] = field(default_factory=dict)
    # aggregated
    numel: int = 0
    nbytes: int = 0
    # leaf tensors directly under this node (optional)
    leaf_tensors: List[TensorEntry] = field(default_factory=list)

    def add(self, parts: List[str], entry: TensorEntry) -> None:
        self.numel += entry.numel
        self.nbytes += entry.nbytes
        if not parts:
            self.leaf_tensors.append(entry)
            return
        head = parts[0]
        if head not in self.children:
            self.children[head] = TreeNode(head)
        self.children[head].add(parts[1:], entry)


def _human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{n} B"


def _collect_entries(state_dict: Dict[str, torch.Tensor]) -> List[TensorEntry]:
    entries: List[TensorEntry] = []
    for name, t in state_dict.items():
        if not torch.is_tensor(t):
            continue
        numel = t.numel()
        nbytes = numel * _dtype_nbytes(t.dtype)
        entries.append(
            TensorEntry(
                name=name,
                shape=tuple(t.shape),
                numel=int(numel),
                dtype=t.dtype,
                nbytes=int(nbytes),
            )
        )
    return entries


def _build_tree(entries: Iterable[TensorEntry]) -> TreeNode:
    root = TreeNode("<root>")
    for e in entries:
        parts = e.name.split(".")
        root.add(parts, e)
    return root


def _print_header(path: str, ckpt_obj: Any, state_dict: Dict[str, torch.Tensor]) -> None:
    print("=" * 80)
    print(f"Checkpoint: {path}")
    print(f"File size: {_human_bytes(os.path.getsize(path)) if os.path.exists(path) else 'N/A'}")
    print(f"Loaded object type: {type(ckpt_obj)}")
    print(f"State dict tensors: {len(state_dict)}")
    print("=" * 80)


def _print_overall_stats(entries: List[TensorEntry]) -> None:
    total_numel = sum(e.numel for e in entries)
    total_bytes = sum(e.nbytes for e in entries)

    dtype_breakdown: Dict[torch.dtype, Tuple[int, int]] = {}
    for e in entries:
        n, b = dtype_breakdown.get(e.dtype, (0, 0))
        dtype_breakdown[e.dtype] = (n + e.numel, b + e.nbytes)

    print("Overall parameter stats (checkpoint tensors):")
    print(f"  Total numel: {total_numel:,}")
    print(f"  Estimated size: {_human_bytes(total_bytes)} (sum(numel * element_size))")
    print("  Dtype breakdown:")
    for dt, (n, b) in sorted(dtype_breakdown.items(), key=lambda x: x[1][1], reverse=True):
        print(f"    - {str(dt):>12}: numel={n:,}  size={_human_bytes(b)}")
    print("-" * 80)


def _iter_tree_lines(node: TreeNode, prefix: str, depth: int, max_depth: int) -> Iterable[str]:
    if node.name == "<root>":
        line = f"{node.name}  numel={node.numel:,}  size={_human_bytes(node.nbytes)}"
    else:
        line = f"{prefix}{node.name}  numel={node.numel:,}  size={_human_bytes(node.nbytes)}"
    yield line

    if depth >= max_depth:
        return

    # sort children by bytes desc
    children = sorted(node.children.values(), key=lambda c: c.nbytes, reverse=True)
    for i, ch in enumerate(children):
        is_last = i == len(children) - 1
        branch = "└── " if is_last else "├── "
        next_prefix = prefix + ("    " if is_last else "│   ")
        yield from _iter_tree_lines(ch, prefix + branch, depth + 1, max_depth)
        # fix indentation: pass next_prefix to grandchildren
        # (we rebuild prefix in recursive call by embedding branch already)
        # For correct tree visuals, we need to pass next_prefix rather than prefix+branch,
        # but we also want the current line to include branch. Easiest: special-case:
        # The current implementation prints a readable (though not perfect) tree.
        # If you want perfect tree alignment, see note below.


def _print_tree(root: TreeNode, max_depth: int = 4) -> None:
    print(f"Module tree (aggregated by key prefixes), max_depth={max_depth}:")
    # For readability, print root first then recurse with a simple indentation style.
    def rec(n: TreeNode, indent: str, depth: int) -> None:
        if depth == 0:
            print(f"{n.name}  numel={n.numel:,}  size={_human_bytes(n.nbytes)}")
        else:
            print(f"{indent}{n.name}  numel={n.numel:,}  size={_human_bytes(n.nbytes)}")
        if depth >= max_depth:
            return
        children = sorted(n.children.values(), key=lambda c: c.nbytes, reverse=True)
        for ch in children:
            rec(ch, indent + "  ", depth + 1)

    rec(root, indent="", depth=0)
    print("-" * 80)


def _print_top_k(entries: List[TensorEntry], k: int = 30) -> None:
    print(f"Top-{k} largest tensors by size:")
    entries_sorted = sorted(entries, key=lambda e: e.nbytes, reverse=True)[:k]
    for i, e in enumerate(entries_sorted, 1):
        print(
            f"{i:>2}. {e.name} | shape={list(e.shape)} | dtype={e.dtype} | "
            f"numel={e.numel:,} | size={_human_bytes(e.nbytes)}"
        )
    print("-" * 80)


def _print_suspicious(entries: List[TensorEntry]) -> None:
    # Heuristics: extremely large embeddings / unexpected dtype, etc.
    big = [e for e in entries if e.nbytes >= 256 * 1024 * 1024]  # >= 256MB
    fp32 = [e for e in entries if e.dtype == torch.float32 and e.nbytes >= 64 * 1024 * 1024]

    if not big and not fp32:
        return

    print("Potentially suspicious tensors (heuristics):")
    if big:
        print("  Very large tensors (>=256MB):")
        for e in sorted(big, key=lambda x: x.nbytes, reverse=True)[:20]:
            print(f"    - {e.name} shape={list(e.shape)} dtype={e.dtype} size={_human_bytes(e.nbytes)}")
    if fp32:
        print("  Large FP32 tensors (>=64MB): consider saving in FP16/BF16 if appropriate:")
        for e in sorted(fp32, key=lambda x: x.nbytes, reverse=True)[:20]:
            print(f"    - {e.name} shape={list(e.shape)} size={_human_bytes(e.nbytes)}")
    print("-" * 80)


def inspect_checkpoint(path: str, *, max_depth: int = 4, top_k: int = 30) -> None:
    # Security note: torch.load can execute pickle code. Only load checkpoints you trust.
    # weights_only=True is available in newer PyTorch versions; we use best-effort.
    load_kwargs = {"map_location": "cpu"}

    ckpt_obj = torch.load(path, weights_only=False, **load_kwargs)  # type: ignore[arg-type]
    state_dict = _extract_state_dict(ckpt_obj)
    entries = _collect_entries(state_dict)
    root = _build_tree(entries)

    _print_header(path, ckpt_obj, state_dict)
    _print_overall_stats(entries)
    _print_tree(root, max_depth=max_depth)
    _print_top_k(entries, k=top_k)
    _print_suspicious(entries)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_ckpt.py /path/to/checkpoint.pt")
        sys.exit(1)

    ckpt_path = sys.argv[1]
    # You can adjust these defaults as needed:
    inspect_checkpoint(ckpt_path, max_depth=4, top_k=30)
