#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统计 point-text memmap 数据集中 caption 的有效 token 数分布，并输出不同截断长度 L 的保留率。

核心定义：
- 有效 token 数 len_i = sum(text_mask[i])   (text_mask: 0/1 or uint8)
- 样本不截断保留率 = P(len_i <= L)
- 有效 token 保留率 = sum(min(len_i, L)) / sum(len_i)

输出：
- stats.json：全局统计 + 分位数 + padding 占用率
- length_hist.csv：length,count
- coverage_selected.csv：用户给定若干 L 的覆盖表
- coverage_full.csv：L=0..max_len 的完整覆盖曲线（方便画图/选 L）
- position_occupancy.csv：pos(1-indexed), occupancy_ratio
"""

import argparse
import csv
import json
import math
import os
from typing import Dict, Any, List, Tuple

import numpy as np
import yaml


def _resolve_path(base_dir: str, p: str) -> str:
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(base_dir, p))


def _percentile_from_hist(hist: np.ndarray, q: float) -> int:
    """
    hist[l] = count of sequences with length l
    return the smallest length L such that CDF(L) >= q
    q in [0,1]
    """
    assert 0.0 <= q <= 1.0
    total = int(hist.sum())
    if total == 0:
        return 0
    target = int(math.ceil(q * total))
    cdf = np.cumsum(hist, dtype=np.int64)
    # first index where cdf >= target
    return int(np.searchsorted(cdf, target, side="left"))


def _write_csv(path: str, header: List[str], rows: List[Tuple[Any, ...]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True, help="dataset_info_yaml 路径")
    ap.add_argument(
        "--require-valid",
        type=int,
        default=1,
        help="是否仅统计 valid==1 的样本（默认 1）。设为 0 则包含无效样本。",
    )
    ap.add_argument(
        "--chunk",
        type=int,
        default=8192,
        help="按 chunk 读取 memmap，控制内存峰值（默认 8192）",
    )
    ap.add_argument(
        "--Ls",
        type=str,
        default="16,24,32,40,48,64,80,96,128,160,192,256",
        help="要输出覆盖率的截断长度列表，逗号分隔",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default="caption_length_stats",
        help="输出目录（默认 caption_length_stats）",
    )
    args = ap.parse_args()

    dataset_info_yaml = args.yaml
    require_valid = bool(args.require_valid)
    chunk = int(args.chunk)
    outdir = args.outdir

    with open(dataset_info_yaml, "r", encoding="utf-8") as f:
        info = yaml.safe_load(f)

    shards = info["shards"]
    base_dir = os.path.dirname(os.path.abspath(dataset_info_yaml))

    # 先确定全局 max_len（不同 shard 可能不同）
    shard_max_lens = [int(s["text"]["max_len"]) for s in shards]
    global_max_len = int(max(shard_max_lens))

    # length_hist[l] = count of captions whose valid length == l
    length_hist = np.zeros(global_max_len + 1, dtype=np.int64)

    # 统计 slots 方便计算 padding rate
    total_samples_included = 0
    total_slots_included = 0  # sum(num_samples_included_in_shard * shard_max_len)
    invalid_samples_skipped = 0  # only meaningful when require_valid=True

    # 逐 shard 读 mask/valid
    for shard_idx, s in enumerate(shards):
        n = int(s["num_samples"])
        Ls = int(s["text"]["max_len"])
        paths = s["paths"]

        mask_path = _resolve_path(base_dir, paths["text_mask"])
        valid_path = _resolve_path(base_dir, paths["valid"])

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"[shard {shard_idx}] text_mask not found: {mask_path}")
        if not os.path.exists(valid_path):
            raise FileNotFoundError(f"[shard {shard_idx}] valid not found: {valid_path}")

        mm_mask = np.memmap(mask_path, mode="r", dtype=np.uint8, shape=(n, Ls))
        mm_valid = np.memmap(valid_path, mode="r", dtype=np.uint8, shape=(n,))

        # chunk loop
        shard_included = 0
        shard_skipped_invalid = 0

        for st in range(0, n, chunk):
            ed = min(n, st + chunk)

            # 读 mask 并求每行有效长度
            m = np.asarray(mm_mask[st:ed], dtype=np.uint8)  # (B, Ls)
            lens = m.sum(axis=1, dtype=np.int32)           # (B,)

            if require_valid:
                v = np.asarray(mm_valid[st:ed], dtype=np.uint8).astype(bool)
                if v.ndim != 1:
                    v = v.reshape(-1)
                shard_skipped_invalid += int((~v).sum())
                lens = lens[v]

            if lens.size == 0:
                continue

            shard_included += int(lens.size)

            # 更新全局 length_hist
            bc = np.bincount(lens, minlength=global_max_len + 1).astype(np.int64)
            length_hist[: bc.shape[0]] += bc

        # 更新 slots 统计
        total_samples_included += shard_included
        total_slots_included += shard_included * Ls
        invalid_samples_skipped += shard_skipped_invalid

        print(
            f"[shard {shard_idx}] num_samples={n}, max_len={Ls}, "
            f"included={shard_included}, skipped_invalid={shard_skipped_invalid}"
        )

    N = int(total_samples_included)
    if N == 0:
        raise RuntimeError("统计到的样本数为 0；请检查 require_valid、valid 标记、以及文件路径。")

    lengths = np.arange(global_max_len + 1, dtype=np.int64)
    total_valid_tokens = int((lengths * length_hist).sum())
    total_len2 = int(((lengths * lengths) * length_hist).sum())

    mean_len = total_valid_tokens / N
    var_len = (total_len2 / N) - (mean_len * mean_len)
    std_len = math.sqrt(max(var_len, 0.0))

    min_len = int(np.searchsorted(np.cumsum(length_hist), 1, side="left"))
    max_len_observed = int(np.max(np.nonzero(length_hist)[0])) if total_valid_tokens > 0 else 0

    percentiles = {
        "p50": _percentile_from_hist(length_hist, 0.50),
        "p75": _percentile_from_hist(length_hist, 0.75),
        "p90": _percentile_from_hist(length_hist, 0.90),
        "p95": _percentile_from_hist(length_hist, 0.95),
        "p99": _percentile_from_hist(length_hist, 0.99),
    }

    # padding / occupancy
    occupancy = (total_valid_tokens / total_slots_included) if total_slots_included > 0 else 0.0
    padding_rate = 1.0 - occupancy

    # coverage curves (L from 0..global_max_len)
    cdf_counts = np.cumsum(length_hist, dtype=np.int64)                 # count(len <= L)
    cdf_tokens = np.cumsum(length_hist * lengths, dtype=np.int64)       # sum(len for len<=L)

    coverage_full_rows = []
    for L in range(global_max_len + 1):
        samples_untruncated = int(cdf_counts[L])
        sample_keep_ratio = samples_untruncated / N

        kept_tokens = int(cdf_tokens[L] + L * (N - samples_untruncated))
        token_keep_ratio = kept_tokens / total_valid_tokens if total_valid_tokens > 0 else 0.0

        avg_trunc_len = kept_tokens / N
        coverage_full_rows.append((L, sample_keep_ratio, token_keep_ratio, avg_trunc_len))

    # position occupancy: 第 pos 个 token（1-index）是有效的比例 = P(len >= pos)
    # len >= pos 等价于 len > pos-1
    # count(len >= pos) = N - cdf_counts[pos-1]
    position_rows = []
    for pos in range(1, global_max_len + 1):
        occupied = int(N - cdf_counts[pos - 1])
        position_rows.append((pos, occupied / N))

    # user-selected Ls
    L_list = []
    for x in args.Ls.split(","):
        x = x.strip()
        if not x:
            continue
        try:
            L_list.append(int(x))
        except ValueError:
            raise ValueError(f"--Ls 包含无法解析的整数: {x}")

    L_list = sorted(set(L_list))
    selected_rows = []
    for L in L_list:
        Lc = max(0, min(L, global_max_len))
        samples_untruncated = int(cdf_counts[Lc])
        sample_keep_ratio = samples_untruncated / N
        kept_tokens = int(cdf_tokens[Lc] + Lc * (N - samples_untruncated))
        token_keep_ratio = kept_tokens / total_valid_tokens if total_valid_tokens > 0 else 0.0
        avg_trunc_len = kept_tokens / N
        selected_rows.append((L, sample_keep_ratio, token_keep_ratio, avg_trunc_len))

    # 输出文件
    os.makedirs(outdir, exist_ok=True)

    # stats.json
    stats = {
        "yaml": os.path.abspath(dataset_info_yaml),
        "require_valid": require_valid,
        "global_max_len": global_max_len,
        "num_samples_included": N,
        "num_invalid_skipped": int(invalid_samples_skipped) if require_valid else 0,
        "total_valid_tokens": total_valid_tokens,
        "mean_len": mean_len,
        "std_len": std_len,
        "min_len": min_len,
        "max_len_observed": max_len_observed,
        "percentiles": percentiles,
        "occupancy_vs_original_slots": occupancy,
        "padding_rate_vs_original_slots": padding_rate,
        "note": {
            "len_definition": "len_i = sum(text_mask[i])",
            "sample_keep_ratio_definition": "P(len_i <= L)  (captions not truncated)",
            "token_keep_ratio_definition": "sum(min(len_i, L)) / sum(len_i)  (valid tokens retained)",
        },
    }
    with open(os.path.join(outdir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # length_hist.csv
    hist_rows = [(int(l), int(c)) for l, c in enumerate(length_hist) if c > 0]
    _write_csv(os.path.join(outdir, "length_hist.csv"), ["length", "count"], hist_rows)

    # coverage csvs
    _write_csv(
        os.path.join(outdir, "coverage_selected.csv"),
        ["L", "sample_keep_ratio", "token_keep_ratio", "avg_truncated_len"],
        selected_rows,
    )
    _write_csv(
        os.path.join(outdir, "coverage_full.csv"),
        ["L", "sample_keep_ratio", "token_keep_ratio", "avg_truncated_len"],
        coverage_full_rows,
    )

    # position occupancy
    _write_csv(
        os.path.join(outdir, "position_occupancy.csv"),
        ["pos_1indexed", "occupancy_ratio"],
        position_rows,
    )

    # 控制台摘要
    print("\n========== Summary ==========")
    print(f"Included samples: {N}")
    if require_valid:
        print(f"Skipped invalid:  {invalid_samples_skipped}")
    print(f"Global max_len (max over shards): {global_max_len}")
    print(f"Valid token length: min={min_len}, max={max_len_observed}, mean={mean_len:.4f}, std={std_len:.4f}")
    print("Percentiles:", ", ".join([f"{k}={v}" for k, v in percentiles.items()]))
    print(f"Occupancy vs original slots: {occupancy:.6f}  (padding_rate={padding_rate:.6f})")
    print(f"Outputs written to: {os.path.abspath(outdir)}")
    print("============================\n")


if __name__ == "__main__":
    main()
