from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class NPZPairedDataset(Dataset):
    """
    读取:
      point:   (N, 512, 256)
      text:    (N, max_text_len, 2048)  padded
      lengths: (N,)
    """

    def __init__(self, npz_path: str | Path):
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"NPZ not found: {npz_path}")

        data = np.load(npz_path, allow_pickle=False)
        self.point = data["point"].astype(np.float32)
        self.text = data["text"].astype(np.float32)
        self.lengths = data["lengths"].astype(np.int64)

        if len(self.point) != len(self.text) or len(self.point) != len(self.lengths):
            raise ValueError("Mismatched N among point/text/lengths.")

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, idx: int):
        point = torch.from_numpy(self.point[idx])  # (512,256)
        text = torch.from_numpy(self.text[idx])    # (max_len,2048)
        length = int(self.lengths[idx])
        return point, text, length
