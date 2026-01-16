from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
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


def masked_cosine_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
    eps: float = 1e-8,
) -> torch.Tensor:
    pred_f = pred.float()
    target_f = target.float()

    p = F.normalize(pred_f, dim=-1, eps=eps)
    t = F.normalize(target_f, dim=-1, eps=eps)
    dist_ = 1.0 - (p * t).sum(dim=-1)  # (B,L)

    if mask is None:
        return dist_.mean()

    w = mask.float()
    return (dist_ * w).sum() / w.sum().clamp_min(1.0)


def _dist_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _gather_with_local_grad(x: torch.Tensor) -> torch.Tensor:
    """
    在 DDP 下收集所有 rank 的 tensor，用于构造更大的 negative pool。

    重要说明：
    - 我们只需要对“本 rank 的样本”保留梯度。
    - 其他 rank 的样本作为 negatives，在本 rank 上视作常量（detach）。
    - 这样不会影响整体训练效果：每个 rank 都会对自己的样本计算 anchor loss。

    该实现不需要自定义 autograd Function，且不会引入额外通信开销。

    前提：各 rank 的 batch size 必须一致（本项目 train/val 已通过 drop_last/DistributedSampler 保证）。
    """
    if not _dist_is_initialized():
        return x

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    x_detached = x.detach()
    gathered = [torch.zeros_like(x_detached) for _ in range(world_size)]
    dist.all_gather(gathered, x_detached)
    gathered[rank] = x
    return torch.cat(gathered, dim=0)


def latent_align_loss(
    text_latents: torch.Tensor,
    point_latents: torch.Tensor,
    align_type: str = "contrastive",
    temperature: float = 0.07,
    gather_distributed: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    对齐 text / point 两个模态的 latents。

    - align_type == "contrastive": 使用 CLIP/BLIP 风格的对比学习（InfoNCE / ITC）loss
      * 对每个 batch 构造全局 (text, point) 相似度矩阵
      * 正样本为配对的 (i,i)，负样本为 batch 内其它样本
      * 双向（text->point 与 point->text）对称交叉熵
      * DDP 下可 all_gather 扩充 negatives

    - align_type == "mse" / "cosine": 保留旧实现，便于做消融

    备注：输入 latents 形状为 (B, N_latent, D)，默认使用 mean pooling 得到全局 embedding。
    """
    at = str(align_type).lower() if align_type is not None else "contrastive"

    zt = text_latents.mean(dim=1).float()   # (B, D)
    zp = point_latents.mean(dim=1).float()  # (B, D)

    if at in {"contrastive", "itc", "clip", "infonce"}:
        # normalize
        zt = F.normalize(zt, dim=-1, eps=eps)
        zp = F.normalize(zp, dim=-1, eps=eps)

        if gather_distributed:
            zt_all = _gather_with_local_grad(zt)
            zp_all = _gather_with_local_grad(zp)
        else:
            zt_all = zt
            zp_all = zp

        b = zt.shape[0]
        if _dist_is_initialized() and gather_distributed:
            # 假设各 rank batch_size 一致
            rank = dist.get_rank()
            targets = torch.arange(b, device=zt.device) + rank * b
        else:
            targets = torch.arange(b, device=zt.device)

        temp = float(temperature)
        if temp <= 0:
            temp = 0.07

        # logits: local anchors vs all candidates
        logits_t2p = (zt @ zp_all.t()) / temp  # (B, B_all)
        logits_p2t = (zp @ zt_all.t()) / temp  # (B, B_all)

        loss_t2p = F.cross_entropy(logits_t2p, targets)
        loss_p2t = F.cross_entropy(logits_p2t, targets)
        return 0.5 * (loss_t2p + loss_p2t)

    if at == "mse":
        return F.mse_loss(zp, zt)

    # default: cosine distance (旧)
    zt = F.normalize(zt, dim=-1, eps=eps)
    zp = F.normalize(zp, dim=-1, eps=eps)
    return (1.0 - (zt * zp).sum(dim=-1)).mean()
