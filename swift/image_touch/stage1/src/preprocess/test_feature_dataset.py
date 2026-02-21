# ============================
# Main: dataset read test & pretty print
# ============================
# processed_feature_dataset.py
from __future__ import annotations

import os
import yaml
import bisect
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

import time
import random

from swift.point_cloud.stage1.src.data.feature_dataset import ProcessedPointTextFeatureDataset

def _load_yaml_if_exists(path: str) -> Dict[str, Any]:
    if path and os.path.isfile(path):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg or {}
    return {}

def _resolve_path(p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    return os.path.abspath(os.path.expanduser(p))

def _human_bytes(n: int) -> str:
    n = float(n)
    units = ["B", "KB", "MB", "GB", "TB"]
    for u in units:
        if n < 1024.0:
            return f"{n:.2f} {u}"
        n /= 1024.0
    return f"{n:.2f} PB"

def _tensor_stats(x: torch.Tensor) -> Dict[str, Any]:
    # x is CPU tensor (from memmap). Convert to float32 for stable stats.
    x_f = x.detach()
    if x_f.dtype in (torch.float16, torch.bfloat16):
        x_f = x_f.float()
    elif x_f.dtype != torch.float32 and x_f.dtype != torch.float64:
        x_f = x_f.float()

    # guard empty tensor
    if x_f.numel() == 0:
        return dict(min=None, max=None, mean=None, std=None, nan=0, inf=0)

    nan_count = int(torch.isnan(x_f).sum().item())
    inf_count = int(torch.isinf(x_f).sum().item())

    # 若有 NaN/Inf，先替换掉再计算 mean/std，避免输出 nan
    x_safe = torch.nan_to_num(x_f, nan=0.0, posinf=0.0, neginf=0.0)

    return dict(
        min=float(x_safe.min().item()),
        max=float(x_safe.max().item()),
        mean=float(x_safe.mean().item()),
        std=float(x_safe.std(unbiased=False).item()),
        nan=nan_count,
        inf=inf_count,
    )

def _preview_values(x: torch.Tensor, k: int) -> List[float]:
    if k <= 0:
        return []
    flat = x.detach().reshape(-1)
    if flat.numel() == 0:
        return []
    flat = flat[: min(k, flat.numel())]
    if flat.dtype in (torch.float16, torch.bfloat16):
        flat = flat.float()
    return [float(v) for v in flat.tolist()]

def _get_default_indices(n: int) -> List[int]:
    if n <= 0:
        return []
    return [0, n // 2, n - 1] if n >= 3 else list(range(n))

def _normalize_indices(indices: List[int], n: int) -> List[int]:
    out = []
    for idx in indices:
        idx = int(idx)
        if idx < 0:
            idx = n + idx
        if 0 <= idx < n:
            out.append(idx)
    # 去重保持顺序
    seen = set()
    uniq = []
    for i in out:
        if i not in seen:
            uniq.append(i)
            seen.add(i)
    return uniq

def _safe_get(d: Dict[str, Any], path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def main() -> None:
    # ----------------------------
    # 1) locate config (no CLI args)
    # ----------------------------
    # 优先：环境变量 MM_DATASET_CFG 指向测试配置
    # 默认：configs/processed_feature_dataset_test.yaml
    cfg_path = os.environ.get("MM_DATASET_CFG", "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/swift/point_cloud/stage1/src/preprocess/test_feature_dataset.yaml")
    cfg_path = _resolve_path(cfg_path)

    cfg = _load_yaml_if_exists(cfg_path) if cfg_path else {}

    # dataset_info_yaml 获取优先级：
    # 1) cfg.dataset_info_yaml
    # 2) env MM_DATASET_INFO
    # 3) 当前目录 ./dataset_info.yaml

    dataset_info_yaml = _resolve_path(cfg.get("dataset_info_yaml")) \
        or _resolve_path(os.environ.get("MM_DATASET_INFO")) \
        or (_resolve_path("dataset_info.yaml") if os.path.isfile("dataset_info.yaml") else None)

    if not dataset_info_yaml or (not os.path.isfile(dataset_info_yaml)):
        msg = (
            "Cannot find dataset_info.yaml.\n\n"
            "Please provide it via either:\n"
            "  1) configs/processed_feature_dataset_test.yaml with key: dataset_info_yaml\n"
            "  2) env var MM_DATASET_INFO=/abs/path/to/dataset_info.yaml\n"
            "  3) put dataset_info.yaml in current working directory\n\n"
            f"Tried config path: {cfg_path}\n"
        )
        raise FileNotFoundError(msg)

    require_valid = bool(cfg.get("require_valid", True))

    # print options
    compute_stats = bool(_safe_get(cfg, "print.compute_stats", True))
    preview_k = int(_safe_get(cfg, "print.preview_values", 8))
    float_precision = int(_safe_get(cfg, "print.float_precision", 5))
    show_shards_table = bool(_safe_get(cfg, "print.show_shards_table", True))
    check_file_sizes = bool(_safe_get(cfg, "print.check_file_sizes", True))

    # samples options
    indices = _safe_get(cfg, "samples.indices", None)
    random_k = int(_safe_get(cfg, "samples.random_k", 3))
    random_seed = int(_safe_get(cfg, "samples.random_seed", 123))

    # dataloader test
    dl_enabled = bool(_safe_get(cfg, "dataloader_test.enabled", True))
    dl_bs = int(_safe_get(cfg, "dataloader_test.batch_size", 4))
    dl_nw = int(_safe_get(cfg, "dataloader_test.num_workers", 0))
    dl_pin = bool(_safe_get(cfg, "dataloader_test.pin_memory", False))
    dl_nb = int(_safe_get(cfg, "dataloader_test.num_batches", 2))

    # ----------------------------
    # 2) import rich (fallback to print)
    # ----------------------------
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich import box
        from rich.text import Text
    except Exception:
        Console = None  # type: ignore

    if Console is None:
        # fallback mode
        print(f"[INFO] dataset_info_yaml: {dataset_info_yaml}")
        ds = ProcessedPointTextFeatureDataset(dataset_info_yaml=dataset_info_yaml, require_valid=require_valid)
        print(f"[INFO] len(ds)={len(ds)} require_valid={ds.require_valid}")
        x = ds[0]
        print("[SAMPLE 0 KEYS]", x.keys())
        print("text_embeds", x["text_embeds"].shape, x["text_embeds"].dtype)
        print("text_mask", x["text_mask"].shape, x["text_mask"].dtype, "sum", int(x["text_mask"].sum()))
        print("point_tokens", x["point_tokens"].shape, x["point_tokens"].dtype)
        print("object_id", x["object_id"], "global_index", x["global_index"], "valid", x["valid"])
        return

    console = Console()

    # ----------------------------
    # 3) load dataset
    # ----------------------------
    ds = ProcessedPointTextFeatureDataset(dataset_info_yaml=dataset_info_yaml, require_valid=require_valid)

    # sanity check: len consistency
    info_total = int(ds.info.get("num_samples_total", len(ds)))
    if info_total != len(ds):
        console.print(
            Panel(
                f"Warning: dataset_info.num_samples_total={info_total} != len(dataset)={len(ds)}",
                title="Length Mismatch",
                style="yellow",
            )
        )

    # ----------------------------
    # 4) print dataset summary
    # ----------------------------
    summary_lines = []
    summary_lines.append(f"[bold]dataset_info_yaml[/bold]: {dataset_info_yaml}")
    summary_lines.append(f"[bold]require_valid[/bold]: {ds.require_valid}")
    summary_lines.append(f"[bold]len(dataset)[/bold]: {len(ds)}")
    summary_lines.append(f"[bold]num_shards[/bold]: {len(ds.shards)}")

    if isinstance(ds.info, dict):
        for k in ("version", "world_size", "split_mode"):
            if k in ds.info:
                summary_lines.append(f"[bold]{k}[/bold]: {ds.info[k]}")

    if isinstance(ds.features, dict):
        tfeat = ds.features.get("text", {})
        pfeat = ds.features.get("point", {})
        summary_lines.append(
            f"[bold]text features[/bold]: max_len={tfeat.get('max_len')} hidden={tfeat.get('hidden')} dtype={tfeat.get('dtype')}"
        )
        summary_lines.append(
            f"[bold]point features[/bold]: num_tokens={pfeat.get('num_tokens')} trans_dim={pfeat.get('trans_dim')} dtype={pfeat.get('dtype')}"
        )

    console.print(Panel("\n".join(summary_lines), title="ProcessedPointTextFeatureDataset Summary", box=box.ROUNDED))

    # ----------------------------
    # 5) shard table
    # ----------------------------
    if show_shards_table:
        table = Table(title="Shards Overview", box=box.SIMPLE_HEAVY)
        table.add_column("shard_idx", justify="right")
        table.add_column("rank", justify="right")
        table.add_column("num_samples", justify="right")
        table.add_column("range [start,end)", justify="right")
        table.add_column("dtype(text/point)", justify="left")
        table.add_column("files_ok", justify="center")
        table.add_column("size(text/point)", justify="left")

        running = 0
        for i, s in enumerate(ds.shards):
            n = int(s["num_samples"])
            start, end = running, running + n
            running = end

            rank = s.get("rank", i)
            dt_text = str(s.get("text", {}).get("dtype", ""))
            dt_point = str(s.get("point", {}).get("dtype", ""))

            paths = s.get("paths", {})
            must_paths = ["text_embeds", "text_mask", "point_tokens", "object_ids", "global_indices", "valid"]
            ok = True
            for k in must_paths:
                p = paths.get(k, "")
                if not p or (not os.path.isfile(p)):
                    ok = False
                    break

            # file sizes (optional)
            size_str = "-"
            if check_file_sizes and ok:
                te_p = paths["text_embeds"]
                pt_p = paths["point_tokens"]
                try:
                    size_str = f"{_human_bytes(os.path.getsize(te_p))} / {_human_bytes(os.path.getsize(pt_p))}"
                except Exception:
                    size_str = "?"

            table.add_row(
                str(i),
                str(rank),
                str(n),
                f"[{start},{end})",
                f"{dt_text} / {dt_point}",
                "[green]YES[/green]" if ok else "[red]NO[/red]",
                size_str,
            )
        console.print(table)

    # ----------------------------
    # 6) choose sample indices
    # ----------------------------
    n = len(ds)
    if indices is None:
        indices = _get_default_indices(n)
    else:
        indices = list(indices)

    indices = _normalize_indices(indices, n)

    # random samples
    if random_k > 0 and n > 0:
        rng = random.Random(random_seed)
        cand = list(range(n))
        rng.shuffle(cand)
        rand_idx = cand[: min(random_k, n)]
        # merge
        indices = _normalize_indices(indices + rand_idx, n)

    console.print(Panel(f"indices to inspect: {indices}", title="Sampling Plan", box=box.ROUNDED))

    # ----------------------------
    # 7) read & pretty print each sample
    # ----------------------------
    def fmt_float(x: Optional[float]) -> str:
        if x is None:
            return "None"
        return f"{x:.{float_precision}f}"

    for idx in indices:
        title = f"Sample idx={idx}"
        try:
            item = ds[idx]
        except Exception as e:
            console.print(Panel(str(e), title=title + " [ERROR]", style="red", box=box.ROUNDED))
            continue

        # unpack
        obj_id = item.get("object_id", "")
        global_index = item.get("global_index", None)
        valid = item.get("valid", None)

        te: torch.Tensor = item["text_embeds"]
        tm: torch.Tensor = item["text_mask"]
        pt: torch.Tensor = item["point_tokens"]

        # stats
        te_stats = _tensor_stats(te) if compute_stats else {}
        pt_stats = _tensor_stats(pt) if compute_stats else {}

        tm_sum = int(tm.sum().item()) if tm.numel() > 0 else 0

        # preview
        te_prev = _preview_values(te, preview_k)
        pt_prev = _preview_values(pt, preview_k)

        # format table
        t = Table(box=box.MINIMAL_DOUBLE_HEAD)
        t.add_column("Field", style="bold")
        t.add_column("Value")

        t.add_row("object_id", str(obj_id))
        t.add_row("global_index", str(global_index))
        t.add_row("valid", str(valid))

        t.add_row("text_embeds.shape", str(tuple(te.shape)))
        t.add_row("text_embeds.dtype", str(te.dtype))
        t.add_row("text_mask.shape", str(tuple(tm.shape)))
        t.add_row("text_mask.sum_true", str(tm_sum))

        if compute_stats:
            t.add_row(
                "text_embeds.stats",
                f"min={fmt_float(te_stats['min'])} max={fmt_float(te_stats['max'])} "
                f"mean={fmt_float(te_stats['mean'])} std={fmt_float(te_stats['std'])} "
                f"nan={te_stats['nan']} inf={te_stats['inf']}",
            )
        if preview_k > 0:
            t.add_row("text_embeds.preview", str(te_prev))

        t.add_row("point_tokens.shape", str(tuple(pt.shape)))
        t.add_row("point_tokens.dtype", str(pt.dtype))
        if compute_stats:
            t.add_row(
                "point_tokens.stats",
                f"min={fmt_float(pt_stats['min'])} max={fmt_float(pt_stats['max'])} "
                f"mean={fmt_float(pt_stats['mean'])} std={fmt_float(pt_stats['std'])} "
                f"nan={pt_stats['nan']} inf={pt_stats['inf']}",
            )
        if preview_k > 0:
            t.add_row("point_tokens.preview", str(pt_prev))

        # shard mapping info
        shard_idx = bisect.bisect_right(ds.prefix, idx) - 1
        local_idx = idx - ds.prefix[shard_idx]
        t.add_row("shard_idx/local_idx", f"{shard_idx} / {local_idx}")

        console.print(Panel(t, title=title, box=box.ROUNDED))

    # ----------------------------
    # 8) optional: DataLoader test
    # ----------------------------
    if dl_enabled:
        from torch.utils.data import DataLoader

        console.print(Panel(
            f"batch_size={dl_bs}, num_workers={dl_nw}, pin_memory={dl_pin}, num_batches={dl_nb}",
            title="DataLoader Test",
            box=box.ROUNDED
        ))

        # 重要：避免 fork worker 继承已打开 memmap，重新创建一个 dataset 实例
        ds2 = ProcessedPointTextFeatureDataset(dataset_info_yaml=dataset_info_yaml, require_valid=require_valid)

        dl = DataLoader(
            ds2,
            batch_size=dl_bs,
            shuffle=False,
            num_workers=dl_nw,
            pin_memory=dl_pin,
            drop_last=False,
        )

        t0 = time.time()
        nb = 0
        for batch in dl:
            nb += 1
            # batch keys
            # text_embeds: (B,L,H), text_mask: (B,L), point_tokens: (B,G,D)
            te_b = batch["text_embeds"]
            tm_b = batch["text_mask"]
            pt_b = batch["point_tokens"]
            gi_b = batch["global_index"]
            oid_b = batch["object_id"]

            bt = Table(box=box.SIMPLE)
            bt.add_column("Batch Field", style="bold")
            bt.add_column("Shape / Info")
            bt.add_row("text_embeds", f"shape={tuple(te_b.shape)} dtype={te_b.dtype}")
            bt.add_row("text_mask", f"shape={tuple(tm_b.shape)} dtype={tm_b.dtype} sum_true={int(tm_b.sum().item())}")
            bt.add_row("point_tokens", f"shape={tuple(pt_b.shape)} dtype={pt_b.dtype}")
            bt.add_row("global_index", f"shape={tuple(gi_b.shape)} dtype={gi_b.dtype} head={gi_b[:min(5, gi_b.numel())].tolist()}")
            bt.add_row("object_id", f"type={type(oid_b)} head={oid_b[:min(3, len(oid_b))]}")

            console.print(Panel(bt, title=f"Batch {nb}", box=box.ROUNDED))

            if nb >= dl_nb:
                break

        elapsed = time.time() - t0
        console.print(Panel(f"DataLoader test done. batches={nb}, time={elapsed:.3f}s", title="DataLoader Test Result", box=box.ROUNDED))


if __name__ == "__main__":
    main()
