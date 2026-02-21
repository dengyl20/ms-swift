from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.data.collate import make_collate_fn
from src.data.npz_dataset import NPZPairedDataset
from src.data.synthetic import SynthSpec, generate_synth_paired_npz
from src.models.losses import latent_align_loss, masked_cosine_distance, masked_mse
from src.models.unified_ae import UnifiedPointTextAE
from src.utils.common import get_device, load_yaml, make_warmup_cosine_lambda, set_global_seed


CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "default.yaml"


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


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, cfg: Dict, device: torch.device) -> Dict[str, float]:
    model.eval()
    loss_cfg = cfg["loss"]

    meters = {k: AverageMeter() for k in ["total", "text_recon", "point2text_recon", "align"]}

    for batch in loader:
        point = batch["point"].to(device)
        text = batch["text"].to(device)
        mask = batch["mask"].to(device)

        out = model(point_feat=point, text_feat=text, text_mask=mask)

        text_recon = (
            loss_cfg["recon_mse"] * masked_mse(out["text_recon"], text, mask)
            + loss_cfg["recon_cos"] * masked_cosine_distance(out["text_recon"], text, mask)
        )
        p2t = (
            loss_cfg["recon_mse"] * masked_mse(out["text_recon_from_point"], text, mask)
            + loss_cfg["recon_cos"] * masked_cosine_distance(out["text_recon_from_point"], text, mask)
        )
        align = latent_align_loss(out["text_latents"], out["point_latents"], align_type=loss_cfg["align_type"])

        total = loss_cfg["w_text_recon"] * text_recon + loss_cfg["w_point2text_recon"] * p2t + loss_cfg["w_align"] * align

        bs = point.size(0)
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

    # 1) 合成数据：若不存在则生成
    synth_cfg = cfg["data"]["synth"]
    npz_path = Path(synth_cfg["out_path"])
    if not npz_path.exists():
        print(f"[Info] generating synthetic dataset: {npz_path}")
        spec = SynthSpec(
            num_samples=int(synth_cfg["num_samples"]),
            point_tokens=int(cfg["model"]["point_tokens"]),
            d_point=int(cfg["model"]["d_point_in"]),
            max_text_len=int(synth_cfg["max_text_len"]),
            min_text_len=int(synth_cfg["min_text_len"]),
            d_text=int(cfg["model"]["d_text_in"]),
            d_true=int(synth_cfg["d_true"]),
            noise_std_text=float(synth_cfg["noise_std_text"]),
            noise_std_point=float(synth_cfg["noise_std_point"]),
            seed=int(cfg.get("seed", 42)),
        )
        generate_synth_paired_npz(npz_path, spec)

    dataset = NPZPairedDataset(npz_path)

    # 2) train/val split
    val_split = float(synth_cfg["val_split"])
    n = len(dataset)
    n_val = int(n * val_split)
    n_train = n - n_val

    g = torch.Generator().manual_seed(int(cfg.get("seed", 42)))
    perm = torch.randperm(n, generator=g).tolist()
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    # 3) dataloader
    train_cfg = cfg["train"]
    collate_fn = make_collate_fn(cfg["model"]["max_text_len"])

    train_loader = DataLoader(
        train_set,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=collate_fn,
    )

    # 4) model
    model = UnifiedPointTextAE(cfg["model"]).to(device)

    # 5) optimizer & scheduler
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        betas=tuple(train_cfg["betas"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    total_steps = int(train_cfg["epochs"]) * len(train_loader)
    warmup_steps = int(train_cfg["scheduler"]["warmup_steps"])
    lr_lambda = make_warmup_cosine_lambda(warmup_steps, total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)

    use_amp = bool(train_cfg.get("amp", False)) and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    save_dir = Path(train_cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    global_step = 0
    loss_cfg = cfg["loss"]

    # 6) train
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        meters = {k: AverageMeter() for k in ["total", "text_recon", "point2text_recon", "align"]}

        for batch in train_loader:
            point = batch["point"].to(device, non_blocking=True)
            text = batch["text"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(point_feat=point, text_feat=text, text_mask=mask)

                text_recon = (
                    loss_cfg["recon_mse"] * masked_mse(out["text_recon"], text, mask)
                    + loss_cfg["recon_cos"] * masked_cosine_distance(out["text_recon"], text, mask)
                )
                p2t = (
                    loss_cfg["recon_mse"] * masked_mse(out["text_recon_from_point"], text, mask)
                    + loss_cfg["recon_cos"] * masked_cosine_distance(out["text_recon_from_point"], text, mask)
                )
                align = latent_align_loss(out["text_latents"], out["point_latents"], align_type=loss_cfg["align_type"])

                total = loss_cfg["w_text_recon"] * text_recon + loss_cfg["w_point2text_recon"] * p2t + loss_cfg["w_align"] * align

            optim.zero_grad(set_to_none=True)
            scaler.scale(total).backward()

            grad_clip = float(train_cfg.get("grad_clip", 0.0))
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optim)
            scaler.update()
            sched.step()

            bs = point.size(0)
            meters["total"].update(total.item(), bs)
            meters["text_recon"].update(text_recon.item(), bs)
            meters["point2text_recon"].update(p2t.item(), bs)
            meters["align"].update(align.item(), bs)

            global_step += 1
            if global_step % int(train_cfg["log_every"]) == 0:
                lr = optim.param_groups[0]["lr"]
                print(
                    f"[E{epoch:03d} S{global_step:06d}] lr={lr:.3e} "
                    f"total={meters['total'].avg:.4f} "
                    f"text={meters['text_recon'].avg:.4f} "
                    f"p2t={meters['point2text_recon'].avg:.4f} "
                    f"align={meters['align'].avg:.4f}"
                )

        val = evaluate(model, val_loader, cfg, device)
        print(
            f"[E{epoch:03d} VAL] total={val['total']:.4f} "
            f"text={val['text_recon']:.4f} p2t={val['point2text_recon']:.4f} align={val['align']:.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "scheduler": sched.state_dict(),
            "cfg": cfg,
            "global_step": global_step,
            "val_total": val["total"],
        }
        torch.save(ckpt, save_dir / f"epoch_{epoch:03d}.pt")

        if val["total"] < best_val:
            best_val = val["total"]
            torch.save(ckpt, save_dir / "best.pt")
            print(f"[Info] saved best.pt (best_val={best_val:.4f})")


if __name__ == "__main__":
    main()
