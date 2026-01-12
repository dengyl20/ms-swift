from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterator, Optional

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from swift.llm import load_dataset


def _stable_hash_to_unit_interval(x: Any) -> float:
    s = str(x).encode("utf-8")
    h = hashlib.md5(s).hexdigest()
    v = int(h[:8], 16)  # 32-bit
    return v / float(16**8)


def extract_assistant_caption(messages: Any, join: str = "all") -> Optional[str]:
    if not isinstance(messages, list):
        return None

    parts = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get("role", None) != "assistant":
            continue
        c = m.get("content", None)
        if isinstance(c, str) and c.strip():
            parts.append(c.strip())

    if not parts:
        return None

    join = (join or "all").lower()
    if join == "last":
        return parts[-1]
    return "\n".join(parts)


class SwiftPointTextStreamingDataset(IterableDataset):
    """
    Streaming dataset wrapper with DDP rank/world_size sharding + dataloader worker sharding.

    输出:
      dict(object_id, points(np.float32)[8192,6], caption(str))
    """

    def __init__(
        self,
        dataset_name_or_path: str,
        *,
        seed: int = 42,
        streaming: bool = True,
        remove_unused_columns: bool = False,
        shuffle_buffer: int = 0,
        assistant_join: str = "all",
        split: str = "train",          # "train" or "val"
        val_ratio: float = 0.01,
        max_samples: Optional[int] = None,
        # ---- new: ddp sharding ----
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        self.dataset_name_or_path = str(dataset_name_or_path)
        self.seed = int(seed)
        self.streaming = bool(streaming)
        self.remove_unused_columns = bool(remove_unused_columns)
        self.shuffle_buffer = int(shuffle_buffer)
        self.assistant_join = str(assistant_join)
        self.split = str(split).lower()
        self.val_ratio = float(val_ratio)
        self.max_samples = max_samples if max_samples is None else int(max_samples)

        self.rank = int(rank)
        self.world_size = int(world_size)

        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _build_swift_iterable(self):
        ds = load_dataset(
            [self.dataset_name_or_path],
            seed=self.seed,
            streaming=self.streaming,
            remove_unused_columns=self.remove_unused_columns,
        )[0]

        # streaming shuffle (if supported)
        if self.shuffle_buffer > 0 and hasattr(ds, "shuffle"):
            try:
                ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=self.seed + self._epoch)
            except TypeError:
                ds = ds.shuffle(buffer_size=self.shuffle_buffer)

        return ds

    def _is_val(self, object_id: Any) -> bool:
        if self.val_ratio <= 0.0:
            return False
        return _stable_hash_to_unit_interval(object_id) < self.val_ratio

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        ds = self._build_swift_iterable()

        worker = get_worker_info()
        num_workers = worker.num_workers if worker is not None else 1
        worker_id = worker.id if worker is not None else 0

        # ---- combined sharding index ----
        # total_shards = world_size * num_workers
        # shard_index  = rank * num_workers + worker_id
        total_shards = max(1, self.world_size) * max(1, num_workers)
        shard_index = max(0, self.rank) * max(1, num_workers) + worker_id

        # Prefer HF native shard() only when the underlying IterableDataset has
        # enough *data sources* (n_shards). For generator-based datasets
        # (IterableDataset.from_generator), n_shards is typically 1.
        # Calling ds.shard(num_shards>n_shards, index>0) can raise IndexError
        # inside HF sharding utils (empty gen_kwargs_list). In that case, we
        # fall back to manual example-level sharding (idx % total_shards).
        used_native_shard = False
        # if total_shards > 1 and hasattr(ds, "shard"):
        #     try:
        #         ds = ds.shard(num_shards=total_shards, index=shard_index)
        #         used_native_shard = True
        #     except TypeError:
        #         used_native_shard = False

        yielded = 0
        for idx, ex in enumerate(ds):
            # Manual fallback sharding when ds.shard is not available
            if (not used_native_shard) and total_shards > 1:
                if (idx % total_shards) != shard_index:
                    continue

            if self.max_samples is not None and yielded >= self.max_samples:
                break

            if not isinstance(ex, dict):
                continue

            object_id = ex.get("object_id", idx)
            is_val = self._is_val(object_id)
            if self.split == "train" and is_val:
                continue
            if self.split == "val" and (not is_val):
                continue

            caption = extract_assistant_caption(ex.get("messages", None), join=self.assistant_join)
            if caption is None or not caption.strip():
                continue

            points = ex.get("points", None)
            if points is None:
                continue

            try:
                pts = np.asarray(points, dtype=np.float32)
            except Exception:
                continue

            if pts.ndim != 2 or pts.shape[1] != 6:
                continue
            if pts.shape[0] != 8192:
                continue

            yielded += 1
            yield {
                "object_id": object_id,
                "points": pts,
                "caption": caption,
            }
