import math
from collections import defaultdict
from typing import Mapping, List, Dict, Any, Optional, Iterable

import torch

def _human_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    units = ["KiB", "MiB", "GiB", "TiB"]
    v = float(n)
    for u in units:
        v /= 1024.0
        if v < 1024.0:
            return f"{v:.2f} {u}"
    return f"{v:.2f} PiB"

def _tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())

def _fmt_shape(shape) -> str:
    try:
        return "(" + ", ".join(str(int(x)) for x in shape) + ")"
    except Exception:
        return str(shape)

def _collect_state_rows(state: Mapping[str, torch.Tensor]) -> List[Dict[str, Any]]:
    rows = []
    for k, v in state.items():
        if not torch.is_tensor(v):
            continue
        rows.append({
            "name": k,
            "shape": tuple(v.shape),
            "dtype": str(v.dtype).replace("torch.", ""),
            "numel": int(v.numel()),
            "bytes": _tensor_nbytes(v),
        })
    rows.sort(key=lambda x: x["name"])
    return rows

def _print_param_table(
    rows: List[Dict[str, Any]],
    *,
    title: str,
    max_rows: Optional[int] = None,
) -> None:
    total_numel = sum(r["numel"] for r in rows)
    total_bytes = sum(r["bytes"] for r in rows)

    print(f"\n=== {title} ===")
    print(f"Tensor entries: {len(rows)}")
    print(f"Total parameters (incl. buffers in state_dict): {total_numel:,}")
    print(f"Total size: {_human_bytes(total_bytes)}")

    # column widths
    name_w = min(120, max([len(r["name"]) for r in rows], default=4))
    shape_w = max([len(_fmt_shape(r["shape"])) for r in rows], default=5)
    dtype_w = max([len(r["dtype"]) for r in rows], default=5)
    numel_w = max([len(f"{r['numel']:,}") for r in rows], default=5)
    bytes_w = max([len(_human_bytes(r["bytes"])) for r in rows], default=5)

    header = (
        f"{'name':<{name_w}}  "
        f"{'shape':<{shape_w}}  "
        f"{'dtype':<{dtype_w}}  "
        f"{'numel':>{numel_w}}  "
        f"{'size':>{bytes_w}}"
    )
    print(header)
    print("-" * len(header))

    show = rows if max_rows is None else rows[:max_rows]
    for r in show:
        print(
            f"{r['name']:<{name_w}}  "
            f"{_fmt_shape(r['shape']):<{shape_w}}  "
            f"{r['dtype']:<{dtype_w}}  "
            f"{r['numel']:,>{numel_w}}  "
            f"{_human_bytes(r['bytes']):>{bytes_w}}"
        )
    if max_rows is not None and len(rows) > max_rows:
        print(f"... ({len(rows) - max_rows} more rows not shown; set max_rows=None to print all)")

def _build_key_tree(keys: Iterable[str]) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    for k in keys:
        parts = k.split(".")
        cur = root
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur.setdefault("__leaf__", []).append(parts[-1])
    return root

def _print_key_tree(
    tree: Dict[str, Any],
    *,
    title: str,
    state: Mapping[str, torch.Tensor],
    indent: str = "  ",
    max_depth: Optional[int] = None,
) -> None:
    print(f"\n=== {title} (key tree) ===")

    def rec(node: Dict[str, Any], prefix: str, depth: int) -> None:
        if max_depth is not None and depth > max_depth:
            print(f"{prefix}{indent}... (depth limit reached)")
            return

        # print leaves for this node
        leaves = node.get("__leaf__", [])
        for leaf in sorted(leaves):
            full = f"{prefix}{leaf}" if prefix else leaf
            t = state.get(full, None)
            if torch.is_tensor(t):
                print(f"{prefix}{indent}{leaf}: {_fmt_shape(t.shape)} [{str(t.dtype).replace('torch.', '')}]")
            else:
                print(f"{prefix}{indent}{leaf}: <non-tensor or missing>")

        # recurse into submodules
        for k in sorted([x for x in node.keys() if x != "__leaf__"]):
            print(f"{prefix}{k}")
            rec(node[k], f"{prefix}{k}.", depth + 1)

    rec(tree, "", depth=0)

def _compare_state_dicts(
    ckpt_state: Mapping[str, torch.Tensor],
    model_state: Mapping[str, torch.Tensor],
    *,
    check_dtype: bool = False,
) -> Dict[str, Any]:
    ckpt_keys = set(k for k, v in ckpt_state.items() if torch.is_tensor(v))
    model_keys = set(k for k, v in model_state.items() if torch.is_tensor(v))

    missing_in_ckpt = sorted(list(model_keys - ckpt_keys))
    unexpected_in_ckpt = sorted(list(ckpt_keys - model_keys))

    shape_mismatch = []
    dtype_mismatch = []

    common = sorted(list(ckpt_keys & model_keys))
    for k in common:
        a = ckpt_state[k]
        b = model_state[k]
        if tuple(a.shape) != tuple(b.shape):
            shape_mismatch.append((k, tuple(a.shape), tuple(b.shape)))
        if check_dtype and str(a.dtype) != str(b.dtype):
            dtype_mismatch.append((k, str(a.dtype), str(b.dtype)))

    return {
        "missing_in_ckpt": missing_in_ckpt,
        "unexpected_in_ckpt": unexpected_in_ckpt,
        "shape_mismatch": shape_mismatch,
        "dtype_mismatch": dtype_mismatch,
        "common_keys": common,
    }

def _print_diff_report(
    diff: Dict[str, Any],
    *,
    title: str,
    max_list: int = 80,
) -> None:
    print(f"\n=== {title} (diff) ===")

    miss = diff["missing_in_ckpt"]
    unexp = diff["unexpected_in_ckpt"]
    sm = diff["shape_mismatch"]
    dm = diff["dtype_mismatch"]

    def _print_list(name: str, items: List[Any]) -> None:
        print(f"{name}: {len(items)}")
        for x in items[:max_list]:
            print(f"  - {x}")
        if len(items) > max_list:
            print(f"  ... ({len(items) - max_list} more)")

    _print_list("Missing in checkpoint (present in model)", miss)
    _print_list("Unexpected in checkpoint (not in model)", unexp)

    print(f"Shape mismatches: {len(sm)}")
    for (k, s_ckpt, s_model) in sm[:max_list]:
        print(f"  - {k}: ckpt={_fmt_shape(s_ckpt)} vs model={_fmt_shape(s_model)}")
    if len(sm) > max_list:
        print(f"  ... ({len(sm) - max_list} more)")

    if dm is not None:
        print(f"Dtype mismatches: {len(dm)}")
        for (k, d_ckpt, d_model) in dm[:max_list]:
            print(f"  - {k}: ckpt={d_ckpt} vs model={d_model}")
        if len(dm) > max_list:
            print(f"  ... ({len(dm) - max_list} more)")
