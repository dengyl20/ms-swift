#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
extract_point_projector_from_swift_ckpt.py

用途：
- 从 ms-swift checkpoint（HF sharded safetensors）中抽取 baseline 新增/训练的 point_projector（或 mm_projector）权重
- 保存为一个独立 .pt/.pth 文件，供离线推理脚本直接加载

与 extract_point_ae_state_dict.py 的关系：
- 逻辑对齐：读取 model.safetensors.index.json -> 从 shards 里只取目标 module 的 tensor -> 去掉外层前缀 -> 保存
- 不再依赖 stage1 pretrain ckpt（baseline projector 没有 stage1 “cfg/model” 格式）
- 输出 ckpt 结构与 AE 类似：{"cfg": {...}, "model": state_dict, "source": {...}}

默认会优先抽取：
- point_projector
- mm_projector（兼容你模板里备用名字）

输出 ckpt（torch.save）格式：
{
  "cfg": {
    "model": {
      "arch": "mlp2x_gelu",
      "module_name": "point_projector" or "mm_projector",
      "in_dim": D,
      "out_dim": H
    }
  },
  "model": {
    "0.weight": ...,
    "0.bias": ...,
    "2.weight": ...,
    "2.bias": ...,
    ...
  },
  "source": {
    "swift_ckpt_dir": "...",
    "matched_full_keys": [...],
    "note": "Extracted from ms-swift sharded safetensors"
  }
}
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from safetensors import safe_open


def _torch_load_compat(path: str):
    """兼容不同 torch 版本的 torch.load 参数差异。"""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _build_mlp2x_gelu(in_dim: int, out_dim: int) -> nn.Module:
    """
    与你 pc_model.py 里的 _build_mlp2x_gelu 保持一致：
      Linear(in_dim -> out_dim) + GELU + Linear(out_dim -> out_dim)
    """
    return nn.Sequential(
        nn.Linear(in_dim, out_dim, bias=True),
        nn.GELU(),
        nn.Linear(out_dim, out_dim, bias=True),
    )


def _load_weight_map(swift_ckpt_dir: str) -> Tuple[Dict[str, str], List[str]]:
    """
    返回 (weight_map, shard_files)

    支持两种格式：
    - sharded safetensors: model.safetensors.index.json 存在
    - single safetensors: model.safetensors 存在（无 index），将其视作单 shard
    """
    index_path = os.path.join(swift_ckpt_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map: Dict[str, str] = index.get("weight_map", {})
        if not weight_map:
            raise RuntimeError(f"weight_map is empty in {index_path}")
        shard_files = sorted(set(weight_map.values()))
        return weight_map, shard_files

    # fallback: single-file safetensors
    single_path = os.path.join(swift_ckpt_dir, "model.safetensors")
    if os.path.isfile(single_path):
        with safe_open(single_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
        if not keys:
            raise RuntimeError(f"No tensors found in {single_path}")
        weight_map = {k: "model.safetensors" for k in keys}
        return weight_map, ["model.safetensors"]

    raise FileNotFoundError(
        "Missing checkpoint index or single safetensors file.\n"
        f"Expected one of:\n"
        f"  - {index_path}\n"
        f"  - {single_path}"
    )


def extract_projector_state_dict(
    swift_ckpt_dir: str,
    module_names: List[str],
) -> Tuple[Dict[str, torch.Tensor], str, List[str]]:
    """
    从 HF sharded safetensors checkpoint 中抽取 {module_name} 的权重，
    并把 key 改成 projector 可直接 load 的格式（去掉外层前缀，仅保留 module_name 之后的部分）。

    返回：
      - state_dict: sub_key -> tensor
      - picked_module_name: 实际命中的 module 名
      - matched_full_keys: checkpoint 原始 key 列表（用于诊断）
    """
    weight_map, _ = _load_weight_map(swift_ckpt_dir)

    if not module_names:
        raise ValueError("module_names must be non-empty")

    # 统计每个 module_name 命中的 key 数量，选择命中最多的那个（避免同时存在时选错）
    hits: Dict[str, List[str]] = {mn: [] for mn in module_names}
    for k in weight_map.keys():
        parts = k.split(".")
        for mn in module_names:
            if mn in parts:
                hits[mn].append(k)

    # 选择命中最多的 module_name
    picked = None
    picked_keys: List[str] = []
    for mn in module_names:
        ks = hits.get(mn, [])
        if len(ks) > len(picked_keys):
            picked = mn
            picked_keys = ks

    if picked is None or len(picked_keys) == 0:
        sample_keys = list(weight_map.keys())[:80]
        raise RuntimeError(
            "No keys found for any target module name.\n"
            f"module_names={module_names}\n"
            f"Example checkpoint keys (first 80): {sample_keys}"
        )

    # 按 shard 分组
    shard_to_keys: Dict[str, List[str]] = defaultdict(list)
    for k in picked_keys:
        shard_to_keys[weight_map[k]].append(k)

    proj_state: Dict[str, torch.Tensor] = {}
    matched_full_keys: List[str] = []

    for shard_file, keys in shard_to_keys.items():
        shard_path = os.path.join(swift_ckpt_dir, shard_file)
        if not os.path.isfile(shard_path):
            raise FileNotFoundError(f"Shard missing: {shard_path}")

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for full_key in keys:
                parts = full_key.split(".")
                try:
                    i = parts.index(picked)
                except ValueError:
                    # 理论上不应发生（因为我们就是按包含 picked 来筛）
                    continue
                sub_key = ".".join(parts[i + 1 :])  # projector.load_state_dict 期望的 key
                proj_state[sub_key] = f.get_tensor(full_key)
                matched_full_keys.append(full_key)

    if not proj_state:
        raise RuntimeError(f"Extraction resulted in empty state_dict for module {picked}")

    return proj_state, picked, matched_full_keys


def infer_in_out_dim_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    """
    尽量从 projector state_dict 推断 (in_dim, out_dim)。

    对你当前 baseline（nn.Sequential(Linear, GELU, Linear)）而言：
      - '0.weight' 形状为 (H, D)
      - '2.weight' 形状为 (H, H)

    若找不到 '0.weight'，则退化为从任意二维 weight 猜测。
    """
    if "0.weight" in state_dict:
        w0 = state_dict["0.weight"]
        if w0.dim() != 2:
            raise RuntimeError(f"Expected '0.weight' to be 2D, got shape={tuple(w0.shape)}")
        out_dim = int(w0.shape[0])
        in_dim = int(w0.shape[1])
        return in_dim, out_dim

    # fallback：找一个最像第一层的二维 weight（in_dim != out_dim 的优先；否则取第一个）
    cand_2d: List[Tuple[str, torch.Tensor]] = []
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.dim() == 2 and k.endswith("weight"):
            cand_2d.append((k, v))

    if not cand_2d:
        raise RuntimeError("Cannot infer dims: no 2D weight tensor found in state_dict.")

    # 优先选择非方阵
    non_square = [(k, v) for k, v in cand_2d if int(v.shape[0]) != int(v.shape[1])]
    pick_k, pick_w = (non_square[0] if non_square else cand_2d[0])

    out_dim = int(pick_w.shape[0])
    in_dim = int(pick_w.shape[1])
    return in_dim, out_dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--swift_ckpt_dir",
        default="/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/checkpoints_mlp/v1-20260218-055502/checkpoint-2520",
        help="ms-swift checkpoint-xxxx 目录（包含 model.safetensors(.index.json)）",
    )
    ap.add_argument(
        "--out_ckpt",
        default="/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/checkpoints_mlp/v1-20260218-055502/point_projector_finetuned_checkpoint.pt",
        help="输出的 finetuned projector ckpt 路径（.pt/.pth）",
    )
    ap.add_argument(
        "--module_names",
        default="point_projector,mm_projector",
        help="要抽取的 module 名列表（逗号分隔）。默认: point_projector,mm_projector",
    )
    ap.add_argument(
        "--arch",
        default="mlp2x_gelu",
        help="保存到 cfg.model.arch 的标记字符串（默认 mlp2x_gelu）。",
    )
    ap.add_argument(
        "--strict_sanity_check",
        default=True,
        help="启用严格 sanity check：按 mlp2x_gelu 构建 projector 并 strict load。默认关闭（更兼容非标准结构）。",
    )
    args = ap.parse_args()

    swift_ckpt_dir = args.swift_ckpt_dir
    out_ckpt = args.out_ckpt
    module_names = [x.strip() for x in str(args.module_names).split(",") if x.strip()]

    print(f"[1/4] Extracting projector from: {swift_ckpt_dir}")
    proj_state, picked_module, matched_full_keys = extract_projector_state_dict(
        swift_ckpt_dir=swift_ckpt_dir,
        module_names=module_names,
    )
    print(f"      picked_module={picked_module}")
    print(f"      extracted tensors: {len(proj_state)}")

    print("[2/4] Inferring in_dim/out_dim from extracted tensors")
    in_dim, out_dim = infer_in_out_dim_from_state_dict(proj_state)
    print(f"      inferred in_dim={in_dim}, out_dim={out_dim}")

    out: Dict[str, Any] = {
        "cfg": {
            "model": {
                "arch": str(args.arch),
                "module_name": str(picked_module),
                "in_dim": int(in_dim),
                "out_dim": int(out_dim),
            }
        },
        "model": proj_state,
        "source": {
            "swift_ckpt_dir": swift_ckpt_dir,
            "matched_full_keys": matched_full_keys,
            "note": "Extracted from ms-swift sharded safetensors",
        },
    }

    os.makedirs(os.path.dirname(out_ckpt), exist_ok=True)
    torch.save(out, out_ckpt)
    print(f"[3/4] Saved finetuned projector ckpt: {out_ckpt}")

    # 可选：严格 sanity check（默认不强制，避免你未来改了 projector 结构就直接报错）
    if args.strict_sanity_check:
        print("[4/4] Sanity check: strict load mlp2x_gelu projector")
        proj = _build_mlp2x_gelu(in_dim, out_dim)
        missing, unexpected = proj.load_state_dict(out["model"], strict=False)
        # 为了更明确：如果你启用 strict_sanity_check，就要求完全对齐
        if missing or unexpected:
            raise RuntimeError(
                "Sanity check failed: state_dict mismatch for mlp2x_gelu projector.\n"
                f"missing: {missing}\n"
                f"unexpected: {unexpected}\n"
                "If you changed projector architecture, re-run without --strict_sanity_check "
                "and update the inference script loader accordingly."
            )
        print("      OK: load_state_dict matched (strict-equivalent).")
    else:
        print("[4/4] Sanity check skipped (run with --strict_sanity_check to enable).")


if __name__ == "__main__":
    main()
