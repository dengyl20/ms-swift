from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    pred/target: (B, L, D)
    mask: (B, L) bool, True=有效
    """
    pred_f = pred.float()
    target_f = target.float()
    per = (pred_f - target_f).pow(2).mean(dim=-1)  # (B,L)

    if mask is None:
        return per.mean()

    w = mask.float()
    return (per * w).sum() / w.sum().clamp_min(1.0)


def masked_cosine_distance(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor], eps: float = 1e-8) -> torch.Tensor:
    pred_f = pred.float()
    target_f = target.float()

    p = F.normalize(pred_f, dim=-1, eps=eps)
    t = F.normalize(target_f, dim=-1, eps=eps)
    dist = 1.0 - (p * t).sum(dim=-1)  # (B,L)

    if mask is None:
        return dist.mean()

    w = mask.float()
    return (dist * w).sum() / w.sum().clamp_min(1.0)


def latent_align_loss(text_latents: torch.Tensor, point_latents: torch.Tensor, align_type: str = "cosine") -> torch.Tensor:
    zt = text_latents.mean(dim=1).float()
    zp = point_latents.mean(dim=1).float()

    if align_type.lower() == "mse":
        return F.mse_loss(zp, zt)

    zt = F.normalize(zt, dim=-1)
    zp = F.normalize(zp, dim=-1)
    return (1.0 - (zt * zp).sum(dim=-1)).mean()
