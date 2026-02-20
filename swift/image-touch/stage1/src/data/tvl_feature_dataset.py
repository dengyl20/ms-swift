

from __future__ import annotations

import os
import bisect
from typing import Any, Dict, Optional, Sequence

import yaml
import numpy as np
import torch
from torch.utils.data import Dataset


def _parse_float_dtype(dtype_like: Any) -> np.dtype:
    """
    dataset_info.yaml 里 dtype 可能长这样：
      - "<class 'numpy.float16'>"
      - "float16"
      - "torch.float16"
    这里用字符串兜底解析。
    """
    s = str(dtype_like).lower()
    if "float32" in s or "fp32" in s:
        return np.float32
    # 默认 float16
    return np.float16


def _infer_1d_memmap_dtype(path: str, n: int) -> Any:
    """
    自动推断 sample_ids / object_ids 这类 1D memmap 的 dtype。
    通过文件大小 / n 得到每条记录的字节数：
      - 8  -> int64
      - 4  -> int32
      - 2  -> int16
      - 1  -> uint8
      - 其他 -> 认为是定长 bytes string：S{bytes}
    """
    size = os.path.getsize(path)
    if n <= 0:
        raise ValueError(f"num_samples must be > 0, got {n}")
    if size % n != 0:
        raise RuntimeError(f"Cannot infer dtype for {path}: file_size={size} not divisible by n={n}")
    rec_bytes = size // n
    if rec_bytes == 8:
        return np.int64
    if rec_bytes == 4:
        return np.int32
    if rec_bytes == 2:
        return np.int16
    if rec_bytes == 1:
        return np.uint8
    return f"S{rec_bytes}"


class ProcessedTouchTextFeatureDataset(Dataset):
    """
    读取 extract_features 生成的 touch-text memmap shards。
    - 支持随机访问
    - 支持按 rank 只加载对应 shard（推荐配合 DDP / torchrun）
    """

    def __init__(
        self,
        dataset_info_yaml: str,
        require_valid: bool = True,
        only_rank: Optional[int] = None,
        ranks: Optional[Sequence[int]] = None,
        return_ids: bool = True,
    ):
        super().__init__()
        self.dataset_info_yaml = dataset_info_yaml
        self.require_valid = bool(require_valid)
        self.return_ids = bool(return_ids)

        with open(dataset_info_yaml, "r") as f:
            info = yaml.safe_load(f)

        self.info = info
        shards = list(info["shards"])

        # 允许只加载某些 rank 的 shard
        if ranks is not None:
            rank_set = set(int(r) for r in ranks)
            shards = [s for s in shards if int(s.get("rank", -1)) in rank_set]
        elif only_rank is not None:
            only_rank = int(only_rank)
            shards = [s for s in shards if int(s.get("rank", -1)) == only_rank]

        if len(shards) == 0:
            raise RuntimeError(
                f"No shards matched. only_rank={only_rank}, ranks={ranks}. "
                f"Check ranks in {dataset_info_yaml}."
            )

        self.shards = shards
        self.features = info.get("features", {})

        # shard sizes / prefix sums
        self.shard_sizes = [int(s["num_samples"]) for s in self.shards]
        self.prefix = [0]
        for n in self.shard_sizes:
            self.prefix.append(self.prefix[-1] + n)
        self.total = self.prefix[-1]

        # lazy-open caches (per-process/per-worker)
        self._mmaps: Dict[int, Dict[str, np.memmap]] = {}

    def __len__(self) -> int:
        return self.total

    def _open_shard(self, shard_idx: int) -> Dict[str, np.memmap]:
        if shard_idx in self._mmaps:
            return self._mmaps[shard_idx]

        s = self.shards[shard_idx]
        paths = s["paths"]
        n = int(s["num_samples"])

        # text shape
        L = int(s["text"]["max_len"])
        H = int(s["text"]["hidden"])

        # touch shape（兼容 hidden / trans_dim 命名）
        G = int(s["touch"]["num_tokens"])
        D = int(s["touch"].get("hidden", s["touch"].get("trans_dim")))

        # dtype
        dt_text = _parse_float_dtype(s["text"].get("dtype", "float16"))
        dt_touch = _parse_float_dtype(s["touch"].get("dtype", dt_text))

        # 兼容 key 名（有些工程里会叫 point_tokens；这里兜底一下）
        touch_tokens_key = "touch_tokens" if "touch_tokens" in paths else "point_tokens"

        # ids key（你给的是 sample_ids；也兼容 object_ids）
        ids_key = None
        if "sample_ids" in paths:
            ids_key = "sample_ids"
        elif "object_ids" in paths:
            ids_key = "object_ids"

        mm: Dict[str, np.memmap] = {
            "text_embeds": np.memmap(
                paths["text_embeds"], mode="r", dtype=dt_text, shape=(n, L, H)
            ),
            "text_mask": np.memmap(
                paths["text_mask"], mode="r", dtype=np.uint8, shape=(n, L)
            ),
            "touch_tokens": np.memmap(
                paths[touch_tokens_key], mode="r", dtype=dt_touch, shape=(n, G, D)
            ),
            "global_indices": np.memmap(
                paths["global_indices"], mode="r", dtype=np.int64, shape=(n,)
            ),
            "valid": np.memmap(
                paths["valid"], mode="r", dtype=np.uint8, shape=(n,)
            ),
        }

        if self.return_ids and ids_key is not None:
            id_path = paths[ids_key]
            id_dt = _infer_1d_memmap_dtype(id_path, n)
            mm["sample_ids"] = np.memmap(id_path, mode="r", dtype=id_dt, shape=(n,))
            mm["_sample_ids_dtype"] = id_dt  # 仅用于 decode 时判断

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
            raise RuntimeError(f"Invalid sample at idx={idx} (shard={shard_idx}, local={local_idx})")

        # 拷贝到普通 ndarray -> torch，避免 memmap 在多 worker 下奇怪的引用问题
        text = torch.from_numpy(np.array(mm["text_embeds"][local_idx], copy=True))          # (L,H)
        mask = torch.from_numpy(np.array(mm["text_mask"][local_idx], copy=True)).bool()    # (L,)
        touch = torch.from_numpy(np.array(mm["touch_tokens"][local_idx], copy=True))       # (G,D)

        global_index = int(mm["global_indices"][local_idx].item())

        sample_id: Optional[Any] = None
        if self.return_ids and "sample_ids" in mm:
            dt = mm.get("_sample_ids_dtype", None)
            raw = mm["sample_ids"][local_idx]
            # bytes string
            if isinstance(dt, str) and dt.startswith("S"):
                sample_id = raw.tobytes().split(b"\x00", 1)[0].decode("utf-8", errors="ignore")
            else:
                # numeric
                sample_id = int(raw.item()) if hasattr(raw, "item") else raw

        return {
            # 为了训练代码复用：直接提供 touch/text/mask 三个 key
            "touch": touch,
            "text": text,
            "mask": mask,
            "sample_id": sample_id,
            "global_index": global_index,
            "valid": valid,
        }