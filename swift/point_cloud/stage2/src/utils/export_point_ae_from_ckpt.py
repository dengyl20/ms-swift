#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from safetensors import safe_open


def torch_load_compat(path: str):
    """兼容不同 torch 版本的 torch.load 参数差异。"""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_point_ae_state_dict(swift_ckpt_dir: str) -> Dict[str, torch.Tensor]:
    """
    从 HF sharded safetensors checkpoint 中抽取 point_ae 的权重，
    并把 key 改成 UnifiedPointTextAE 可直接 load 的格式（去掉外层前缀，仅保留 point_ae 之后的部分）。
    """
    index_path = os.path.join(swift_ckpt_dir, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"Missing index file: {index_path}")

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    weight_map: Dict[str, str] = index.get("weight_map", {})
    if not weight_map:
        raise RuntimeError(f"weight_map is empty in {index_path}")

    # 选出所有包含一个模块段名 point_ae 的权重
    ae_keys: List[str] = []
    for k in weight_map.keys():
        parts = k.split(".")
        if "point_ae" in parts:
            ae_keys.append(k)

    if len(ae_keys) == 0:
        # 给一点诊断输出
        sample_keys = list(weight_map.keys())[:50]
        raise RuntimeError(
            "No keys containing module name 'point_ae' found in checkpoint.\n"
            f"Example keys: {sample_keys}"
        )

    # 按 shard 分组：避免重复打开文件
    shard_to_keys: Dict[str, List[str]] = defaultdict(list)
    for k in ae_keys:
        shard_to_keys[weight_map[k]].append(k)

    ae_state: Dict[str, torch.Tensor] = {}
    for shard_file, keys in shard_to_keys.items():
        shard_path = os.path.join(swift_ckpt_dir, shard_file)
        if not os.path.isfile(shard_path):
            raise FileNotFoundError(f"Shard missing: {shard_path}")

        # safetensors: memory-map 读取，只取需要的 tensor
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for full_key in keys:
                # 统一剥掉 point_ae 之前的所有前缀
                parts = full_key.split(".")
                i = parts.index("point_ae")
                sub_key = ".".join(parts[i + 1 :])  # UnifiedPointTextAE 期望的 key
                ae_state[sub_key] = f.get_tensor(full_key)

    return ae_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swift_ckpt_dir", default="/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/checkpoints/v100-20260215-045054/checkpoint-1260", help="ms-swift checkpoint-xxxx 目录（包含 model.safetensors.index.json）")
    ap.add_argument("--ae_pretrain_ckpt", default="/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/checkpoints/cleaned_maxlen_24/best.pt", help="你训练时加载的 stage1 预训练 AE ckpt（含 cfg/model）")
    ap.add_argument("--out_ckpt", default="/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/checkpoints/v100-20260215-045054/point_ae_finetuned_checkpoint-1006.pt", help="输出的 finetuned AE ckpt 路径（.pt/.pth）")
    args = ap.parse_args()

    swift_ckpt_dir = args.swift_ckpt_dir
    ae_pretrain_ckpt = args.ae_pretrain_ckpt
    out_ckpt = args.out_ckpt

    print(f"[1/4] Extracting point_ae from: {swift_ckpt_dir}")
    ae_state = extract_point_ae_state_dict(swift_ckpt_dir)
    print(f"      extracted tensors: {len(ae_state)}")

    print(f"[2/4] Loading stage1 AE pretrain ckpt to reuse cfg: {ae_pretrain_ckpt}")
    pre = torch_load_compat(ae_pretrain_ckpt)
    if "cfg" not in pre or "model" not in pre:
        raise RuntimeError(f"Unexpected stage1 AE ckpt format: keys={list(pre.keys())}")

    # 做一个 key 对齐检查（强烈建议）
    pre_keys = set(pre["model"].keys())
    new_keys = set(ae_state.keys())
    missing = sorted(list(pre_keys - new_keys))
    unexpected = sorted(list(new_keys - pre_keys))

    if missing or unexpected:
        print("[WARN] state_dict keys mismatch between pretrain AE and extracted AE.")
        print(f"       missing({len(missing)}): {missing[:20]}{' ...' if len(missing) > 20 else ''}")
        print(f"       unexpected({len(unexpected)}): {unexpected[:20]}{' ...' if len(unexpected) > 20 else ''}")
        print("       If you changed AE architecture between stage1 and stage2, this is expected.")
        print("       Otherwise, mismatch means extraction prefix logic or training module name differs.")

    # 复用原 ckpt 的 cfg / 其它字段，只替换 model 权重
    out = dict(pre)
    out["model"] = ae_state

    os.makedirs(os.path.dirname(out_ckpt), exist_ok=True)
    torch.save(out, out_ckpt)
    print(f"[3/4] Saved finetuned AE ckpt: {out_ckpt}")

    # 额外做一次“能否 strict load”的 sanity check（不依赖 GPU）
    print("[4/4] Sanity check: strict load UnifiedPointTextAE")
    from swift.point_cloud.stage1.src.models.unified_ae import UnifiedPointTextAE

    ae = UnifiedPointTextAE(out["cfg"]["model"])
    ae.load_state_dict(out["model"], strict=True)
    print("      OK: strict load passed.")


if __name__ == "__main__":
    main()
