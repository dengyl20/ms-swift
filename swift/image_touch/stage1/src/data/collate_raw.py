from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch


def collate_points_and_captions(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    batch[i]:
      - points: np.ndarray float32 (8192,6)
      - caption: str
      - object_id: any

    return:
      - points: torch.FloatTensor (B,8192,6)
      - captions: list[str]
      - object_ids: list
    """
    if len(batch) == 0:
        raise RuntimeError("Empty batch encountered.")

    pts = np.stack([b["points"] for b in batch], axis=0).astype(np.float32, copy=False)
    points = torch.from_numpy(pts)  # (B,8192,6)

    captions = [str(b["caption"]) for b in batch]
    object_ids = [b.get("object_id", None) for b in batch]

    return {"points": points, "captions": captions, "object_ids": object_ids}
