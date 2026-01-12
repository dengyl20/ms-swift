from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from swift.point_cloud.stage1.src.data.collate_raw import collate_points_and_captions
from swift.point_cloud.stage1.src.data.swift_streaming import SwiftPointTextStreamingDataset
from swift.point_cloud.stage1.src.models.frozen_encoders import FrozenPointBERTTokens, FrozenQwenEmbeddingTable
from swift.point_cloud.stage1.src.models.losses import latent_align_loss, masked_cosine_distance, masked_mse
from swift.point_cloud.stage1.src.models.unified_ae import UnifiedPointTextAE
from swift.point_cloud.stage1.src.utils.common import get_device, load_yaml, make_warmup_cosine_lambda, set_global_seed

os.environ['POINT_CLOUD_DATA_PATH'] = '/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/PointLLM/PointLLM/8192_npy'
os.environ['POINT_CLOUD_ANNO_PATH'] = '/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/PointLLM/PointLLM/PointLLM_brief_description_660K.json'


CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "real_swift.yaml"


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, v: float, n: int = 1):
        self.sum += float(v) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)


def cycle(loader: DataLoader) -> Iterator[Dict]:
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def run_validation(
    *,
    model: nn.Module,
    point_enc: FrozenPointBERTTokens,
    text_emb: FrozenQwenEmbeddingTable,
    val_iter: Iterator[Dict],
    cfg: Dict,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    model.eval()

    loss_cfg = cfg["loss"]
    meters = {k: AverageMeter() for k in ["total", "text_recon", "point2text_recon", "align"]}

    val_steps = int(cfg["train"]["val_steps"])
    for _ in range(val_steps):
        batch = next(val_iter)
        points = batch["points"].to(device, non_blocking=True)  # (B,8192,6)
        captions = batch["captions"]

        with torch.no_grad():
            # 外部 encoder：不回传梯度
            point_feat = point_enc(points)  # (B,512,trans_dim)
            text_feat, text_mask = text_emb(captions)  # (B,max_len,2048), (B,max_len)

        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(point_feat=point_feat, text_feat=text_feat, text_mask=text_mask)

            text_recon = (
                loss_cfg["recon_mse"] * masked_mse(out["text_recon"], text_feat, text_mask)
                + loss_cfg["recon_cos"] * masked_cosine_distance(out["text_recon"], text_feat, text_mask)
            )
            p2t = (
                loss_cfg["recon_mse"] * masked_mse(out["text_recon_from_point"], text_feat, text_mask)
                + loss_cfg["recon_cos"] * masked_cosine_distance(out["text_recon_from_point"], text_feat, text_mask)
            )
            align = latent_align_loss(out["text_latents"], out["point_latents"], align_type=loss_cfg["align_type"])

            total = (
                loss_cfg["w_text_recon"] * text_recon
                + loss_cfg["w_point2text_recon"] * p2t
                + loss_cfg["w_align"] * align
            )

        bs = points.size(0)
        meters["total"].update(total.item(), bs)
        meters["text_recon"].update(text_recon.item(), bs)
        meters["point2text_recon"].update(p2t.item(), bs)
        meters["align"].update(align.item(), bs)

    return {k: v.avg for k, v in meters.items()}


def main():
    cfg = load_yaml(CONFIG_PATH)
    set_global_seed(int(cfg.get("seed", 42)))

    device = get_device()
    print(f"[Info] device = {device}")

    # -------------------------
    # 0) build streaming datasets
    # -------------------------
    ds_cfg = cfg["data"]["swift"]
    train_ds = SwiftPointTextStreamingDataset(
        ds_cfg["dataset"],
        seed=int(ds_cfg.get("seed", 42)),
        streaming=bool(ds_cfg.get("streaming", True)),
        remove_unused_columns=bool(ds_cfg.get("remove_unused_columns", False)),
        shuffle_buffer=int(ds_cfg.get("shuffle_buffer", 0)),
        assistant_join=str(ds_cfg.get("assistant_join", "all")),
        split="train",
        val_ratio=float(ds_cfg.get("val_ratio", 0.01)),
        max_samples=ds_cfg.get("max_samples", None),
    )
    val_ds = SwiftPointTextStreamingDataset(
        ds_cfg["dataset"],
        seed=int(ds_cfg.get("seed", 42)),
        streaming=bool(ds_cfg.get("streaming", True)),
        remove_unused_columns=bool(ds_cfg.get("remove_unused_columns", False)),
        shuffle_buffer=0,  # val 通常不 shuffle
        assistant_join=str(ds_cfg.get("assistant_join", "all")),
        split="val",
        val_ratio=float(ds_cfg.get("val_ratio", 0.01)),
        max_samples=ds_cfg.get("max_samples", None),
    )

    tr_cfg = cfg["train"]
    train_loader = DataLoader(
        train_ds,
        batch_size=int(tr_cfg["batch_size"]),
        num_workers=int(tr_cfg.get("num_workers", 0)),
        pin_memory=bool(tr_cfg.get("pin_memory", True)) and (device.type == "cuda"),
        drop_last=bool(tr_cfg.get("drop_last", True)),
        collate_fn=collate_points_and_captions,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(tr_cfg["batch_size"]),
        num_workers=int(tr_cfg.get("num_workers", 0)),
        pin_memory=bool(tr_cfg.get("pin_memory", True)) and (device.type == "cuda"),
        drop_last=False,
        collate_fn=collate_points_and_captions,
    )

    train_iter = cycle(train_loader)
    val_iter = cycle(val_loader)

    # -------------------------
    # 1) build frozen external encoders
    # -------------------------
    ext_cfg = cfg["external_encoders"]

    point_enc = FrozenPointBERTTokens(ext_cfg["point_bert"], device=device)
    text_emb = FrozenQwenEmbeddingTable(ext_cfg["qwen"], device=device)

    # 一致性检查：确保 unified model 的输入维度与外部编码器一致
    mcfg = cfg["model"]
    expected_point_dim = int(mcfg["d_point_in"])
    expected_point_tokens = int(mcfg["point_tokens"])
    if point_enc.trans_dim != expected_point_dim:
        raise ValueError(f"Mismatch: point_enc.trans_dim={point_enc.trans_dim} vs model.d_point_in={expected_point_dim}")
    if point_enc.num_group != expected_point_tokens and ext_cfg["point_bert"]["drop_cls"] is True:
        raise ValueError(f"Mismatch: point_enc.num_group={point_enc.num_group} vs model.point_tokens={expected_point_tokens}")
    if point_enc.num_group + 1 != expected_point_tokens and ext_cfg["point_bert"]["drop_cls"] is False:
        raise ValueError(f"Mismatch: point_enc.num_group={point_enc.num_group} vs model.point_tokens={expected_point_tokens}")

    expected_text_dim = int(mcfg["d_text_in"])
    if text_emb.hidden_size != expected_text_dim:
        raise ValueError(f"Mismatch: qwen embedding hidden_size={text_emb.hidden_size} vs model.d_text_in={expected_text_dim}")

    # -------------------------
    # 2) build trainable mapping model
    # -------------------------
    model = UnifiedPointTextAE(cfg["model"]).to(device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(tr_cfg["lr"]),
        betas=tuple(tr_cfg["betas"]),
        weight_decay=float(tr_cfg["weight_decay"]),
    )

    steps_per_epoch = int(tr_cfg["steps_per_epoch"])
    total_steps = int(tr_cfg["epochs"]) * steps_per_epoch
    warmup_steps = int(tr_cfg["scheduler"]["warmup_steps"])
    lr_lambda = make_warmup_cosine_lambda(warmup_steps, total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)

    use_amp = bool(tr_cfg.get("amp", False)) and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    save_dir = Path(tr_cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    global_step = 0

    loss_cfg = cfg["loss"]
    log_every = int(tr_cfg.get("log_every", 50))
    grad_clip = float(tr_cfg.get("grad_clip", 0.0))

    # -------------------------
    # 3) training loop
    # -------------------------
    for epoch in range(1, int(tr_cfg["epochs"]) + 1):
        model.train()
        train_ds.set_epoch(epoch)
        val_ds.set_epoch(epoch)

        meters = {k: AverageMeter() for k in ["total", "text_recon", "point2text_recon", "align"]}

        for _ in range(steps_per_epoch):
            batch = next(train_iter)
            points = batch["points"].to(device, non_blocking=True)
            captions = batch["captions"]

            with torch.no_grad():
                point_feat = point_enc(points)               # (B,512,trans_dim)
                text_feat, text_mask = text_emb(captions)    # (B,max_len,2048), (B,max_len)

            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(point_feat=point_feat, text_feat=text_feat, text_mask=text_mask)

                text_recon = (
                    loss_cfg["recon_mse"] * masked_mse(out["text_recon"], text_feat, text_mask)
                    + loss_cfg["recon_cos"] * masked_cosine_distance(out["text_recon"], text_feat, text_mask)
                )
                p2t = (
                    loss_cfg["recon_mse"] * masked_mse(out["text_recon_from_point"], text_feat, text_mask)
                    + loss_cfg["recon_cos"] * masked_cosine_distance(out["text_recon_from_point"], text_feat, text_mask)
                )
                align = latent_align_loss(out["text_latents"], out["point_latents"], align_type=loss_cfg["align_type"])

                total = (
                    loss_cfg["w_text_recon"] * text_recon
                    + loss_cfg["w_point2text_recon"] * p2t
                    + loss_cfg["w_align"] * align
                )

            optim.zero_grad(set_to_none=True)
            scaler.scale(total).backward()

            if grad_clip and grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optim)
            scaler.update()
            sched.step()

            bs = points.size(0)
            meters["total"].update(total.item(), bs)
            meters["text_recon"].update(text_recon.item(), bs)
            meters["point2text_recon"].update(p2t.item(), bs)
            meters["align"].update(align.item(), bs)

            global_step += 1
            if global_step % log_every == 0:
                lr = optim.param_groups[0]["lr"]
                print(
                    f"[E{epoch:03d} S{global_step:07d}] lr={lr:.3e} "
                    f"total={meters['total'].avg:.4f} "
                    f"text={meters['text_recon'].avg:.4f} "
                    f"p2t={meters['point2text_recon'].avg:.4f} "
                    f"align={meters['align'].avg:.4f}"
                )

        # validation
        val = run_validation(
            model=model,
            point_enc=point_enc,
            text_emb=text_emb,
            val_iter=val_iter,
            cfg=cfg,
            device=device,
            use_amp=use_amp,
        )
        print(
            f"[E{epoch:03d} VAL] total={val['total']:.4f} "
            f"text={val['text_recon']:.4f} p2t={val['point2text_recon']:.4f} align={val['align']:.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "cfg": cfg,
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "scheduler": sched.state_dict(),
            "val_total": val["total"],
        }
        torch.save(ckpt, save_dir / f"epoch_{epoch:03d}.pt")

        if val["total"] < best_val:
            best_val = val["total"]
            torch.save(ckpt, save_dir / "best.pt")
            print(f"[Info] saved best.pt (best_val={best_val:.4f})")


if __name__ == "__main__":
    main()
