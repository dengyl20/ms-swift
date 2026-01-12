from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterator, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from swift.llm import load_dataset


def _stable_hash_to_unit_interval(x: Any) -> float:
    s = str(x).encode("utf-8")
    h = hashlib.md5(s).hexdigest()
    v = int(h[:8], 16)  # 32-bit
    return v / float(16**8)


def extract_assistant_caption(messages: Any, join: str = "all") -> Optional[str]:
    """
    从 messages 中提取所有 role=assistant 的 content，并按 join 策略合并。
    """
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
    以 streaming 方式从 swift dataset 读取:
      - points: list[list[float]] (8192,6)
      - messages: list[dict]
    产出:
      dict(object_id, points_np, caption)

    注意：这里只做“轻量过滤 + 文本提取 + train/val 切分”；
         不在 Dataset 内做 point encoder / text embedding（避免多进程重复加载大模型）。
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

        self._epoch = 0  # allow set_epoch for different shuffle each epoch

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
            # 让每个 epoch 的顺序不同（如果 ds.shuffle 支持 seed）
            try:
                ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=self.seed + self._epoch)
            except TypeError:
                ds = ds.shuffle(buffer_size=self.shuffle_buffer)

        return ds

    def _is_val(self, object_id: Any) -> bool:
        if self.val_ratio <= 0.0:
            return False
        u = _stable_hash_to_unit_interval(object_id)
        return u < self.val_ratio

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        ds = self._build_swift_iterable()

        # 多 worker 去重：优先用 shard（如果支持）
        worker = get_worker_info()
        if worker is not None:
            if hasattr(ds, "shard"):
                try:
                    ds = ds.shard(num_shards=worker.num_workers, index=worker.id)
                except TypeError:
                    # fallback：下面手动跳过
                    pass

        yielded = 0
        for idx, ex in enumerate(ds):
            # 手动 worker 分片（当 ds 不支持 shard 时）
            if worker is not None and (not hasattr(ds, "shard")):
                if (idx % worker.num_workers) != worker.id:
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

            # points: list[list[float]] => np.float32 (8192,6)
            try:
                pts = np.asarray(points, dtype=np.float32)
            except Exception:
                continue

            if pts.ndim != 2 or pts.shape[1] != 6:
                continue
            # 你给的样例是 (8192,6)；若存在长度不等，建议直接跳过（或自行补齐/采样）
            if pts.shape[0] != 8192:
                continue

            yielded += 1
            yield {
                "object_id": object_id,
                "points": pts,     # numpy float32 (8192,6)
                "caption": caption,
            }
