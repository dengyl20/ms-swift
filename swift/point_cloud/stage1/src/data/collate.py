from __future__ import annotations

from typing import List, Tuple

import torch


def make_collate_fn(max_text_len: int):
    """
    兼容两种 text 输入：
      - (L, 2048) 变长
      - (max_text_len, 2048) 已 padding

    输出统一为：
      text: (B, max_text_len, 2048)
      mask: (B, max_text_len) bool
    """
    max_text_len = int(max_text_len)

    def collate(batch: List[Tuple[torch.Tensor, torch.Tensor, int]]):
        points, texts, lengths = zip(*batch)
        point = torch.stack(points, dim=0)

        B = len(texts)
        d_text = int(texts[0].shape[-1])
        text_out = torch.zeros((B, max_text_len, d_text), dtype=texts[0].dtype)
        lengths_out = torch.zeros((B,), dtype=torch.long)

        for i, (t, L) in enumerate(zip(texts, lengths)):
            L = int(L)
            if t.dim() != 2:
                raise ValueError(f"Expected text tensor with 2 dims, got shape {tuple(t.shape)}")
            Li = min(L, max_text_len)
            text_out[i, :Li] = t[:Li]
            lengths_out[i] = Li

        mask = torch.arange(max_text_len).unsqueeze(0) < lengths_out.unsqueeze(1)
        return {"point": point, "text": text_out, "lengths": lengths_out, "mask": mask}

    return collate
