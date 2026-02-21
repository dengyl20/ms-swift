# processed_feature_dataset.py
from __future__ import annotations

import os
import yaml
import bisect
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class ProcessedPointTextFeatureDataset(Dataset):
    """
    读取 extract_features.py 生成的 memmap shards。
    支持随机访问，适用于下一阶段训练。
    """

    def __init__(self, dataset_info_yaml: str, require_valid: bool = True):
        super().__init__()
        self.dataset_info_yaml = dataset_info_yaml
        self.require_valid = bool(require_valid)

        with open(dataset_info_yaml, "r") as f:
            info = yaml.safe_load(f)

        self.info = info
        self.shards = info["shards"]
        self.features = info["features"]

        # shard sizes / prefix sums
        self.shard_sizes = [int(s["num_samples"]) for s in self.shards]
        self.prefix = [0]
        for n in self.shard_sizes:
            self.prefix.append(self.prefix[-1] + n)
        self.total = self.prefix[-1]

        # lazy-open caches (per-process/per-worker)
        self._mmaps = {}  # shard_idx -> dict of memmaps

    def __len__(self) -> int:
        return self.total

    def _open_shard(self, shard_idx: int) -> Dict[str, np.memmap]:
        if shard_idx in self._mmaps:
            return self._mmaps[shard_idx]

        s = self.shards[shard_idx]
        paths = s["paths"]

        max_len = int(s["text"]["max_len"])
        hidden = int(s["text"]["hidden"])

        G = int(s["point"]["num_tokens"])
        trans_dim = int(s["point"]["trans_dim"])

        # dtype parse
        # s["text"]["dtype"] looks like "<class 'numpy.float16'>" or "float16" depending on dump
        # 我们以文件真实 dtype 为准：memmap dtype 需与写入一致，这里简单按 float16/float32 推断
        dtype_str = str(s["text"]["dtype"]).lower()
        if "float32" in dtype_str:
            dt = np.float32
        else:
            dt = np.float16

        mm = {
            "text_embeds": np.memmap(paths["text_embeds"], mode="r", dtype=dt, shape=(s["num_samples"], max_len, hidden)),
            "text_mask": np.memmap(paths["text_mask"], mode="r", dtype=np.uint8, shape=(s["num_samples"], max_len)),
            "point_tokens": np.memmap(paths["point_tokens"], mode="r", dtype=dt, shape=(s["num_samples"], G, trans_dim)),
            "object_ids": np.memmap(paths["object_ids"], mode="r", dtype="S32", shape=(s["num_samples"],)),
            "global_indices": np.memmap(paths["global_indices"], mode="r", dtype=np.int64, shape=(s["num_samples"],)),
            "valid": np.memmap(paths["valid"], mode="r", dtype=np.uint8, shape=(s["num_samples"],)),
        }
        self._mmaps[shard_idx] = mm
        return mm

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= self.total:
            raise IndexError(idx)

        shard_idx = bisect.bisect_right(self.prefix, idx) - 1
        local_idx = idx - self.prefix[shard_idx]

        mm = self._open_shard(shard_idx)

        valid = bool(mm["valid"][local_idx].item())
        if self.require_valid and (not valid):
            # 若强制有效样本：简单做法是抛错；训练时建议用自定义 sampler 过滤 valid
            raise RuntimeError(f"Invalid sample at idx={idx} (shard={shard_idx}, local={local_idx})")

        text_embeds = torch.from_numpy(np.array(mm["text_embeds"][local_idx], copy=True))          # (L,H)
        text_mask = torch.from_numpy(np.array(mm["text_mask"][local_idx], copy=True)).bool()       # (L,)
        point_tokens = torch.from_numpy(np.array(mm["point_tokens"][local_idx], copy=True))        # (G,D)

        obj_id = mm["object_ids"][local_idx].tobytes().split(b"\x00", 1)[0].decode("utf-8")
        global_index = int(mm["global_indices"][local_idx].item())

        return {
            "text_embeds": text_embeds,
            "text_mask": text_mask,
            "point_tokens": point_tokens,
            "object_id": obj_id,
            "global_index": global_index,
            "valid": valid,
        }
