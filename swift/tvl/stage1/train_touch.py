from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Tuple, Any, Optional
import os
import hashlib
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)

# ====== TVL touch-text dataset / model（你已经有）======
from swift.tvl.stage1.src.data.touch_fea_dataset import ProcessedTouchTextFeatureDataset
from swift.tvl.stage1.src.models.unified_touch import UnifiedTouchTextAE

# ====== losses / utils：优先用 tvl 的；没有就 fallback 到 point_cloud 版本（少踩坑）======

from swift.tvl.stage1.src.models.losses import latent_align_loss, masked_cosine_distance, masked_mse
from swift.tvl.stage1.src.utils.common import make_warmup_cosine_lambda, set_global_seed, load_yaml



CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "real_tvl.yaml"
console = Console()


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, v: float, n: int = 1):
        self.sum += float(v) * n
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)


def cycle(loader: DataLoader) -> Iterator[Dict]:
    while True:
        for batch in loader:
            yield batch


def ddp_is_enabled() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def ddp_setup() -> Dict[str, int]:
    """
    torchrun 会提供:
      - RANK, WORLD_SIZE, LOCAL_RANK
    """
    if not ddp_is_enabled():
        return {"rank": 0, "world_size": 1, "local_rank": 0}

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=timedelta(minutes=30),
    )

    return {"rank": rank, "world_size": world_size, "local_rank": local_rank}


def ddp_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def sync_meters_across_ranks(meters: Dict[str, AverageMeter], device: torch.device) -> Dict[str, float]:
    """
    把每个 meter 的 (sum, count) 在所有 rank 上 all_reduce，然后返回全局 avg。
    """
    if not (dist.is_available() and dist.is_initialized()):
        return {k: v.avg for k, v in meters.items()}

    out = {}
    for k, m in meters.items():
        t = torch.tensor([m.sum, float(m.count)], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        s = t[0].item()
        c = max(1.0, t[1].item())
        out[k] = s / c
    return out


# =========================
# feature split 辅助逻辑：支持 sample_ids / object_ids
# =========================

def _stable_hash_to_unit_interval_bytes(b: bytes) -> float:
    """
    稳定 hash：md5 前 4 bytes -> [0,1)
    """
    h = hashlib.md5(b).digest()
    v = int.from_bytes(h[:4], byteorder="big", signed=False)
    return v / float(2**32)


def _infer_1d_memmap_dtype(path: str, n: int) -> Any:
    """
    根据文件大小推断 1D memmap dtype（sample_ids/object_ids 常用）。
    - 8 bytes -> int64
    - 否则 -> 定长 bytes string: S{rec_bytes}
    """
    size = os.path.getsize(path)
    if n <= 0:
        raise ValueError(f"num_samples must be > 0, got {n}")
    if size % n != 0:
        raise RuntimeError(f"Cannot infer dtype for {path}: file_size={size} not divisible by n={n}")
    rec_bytes = size // n
    if rec_bytes == 8:
        return np.int64
    return f"S{rec_bytes}"


def _id_value_to_bytes(x: Any) -> bytes:
    """
    把 sample_id/object_id 的单条记录转成 bytes 用于 hash。
    """
    if hasattr(x, "item"):
        x = x.item()

    if isinstance(x, (bytes, bytearray)):
        return bytes(x).split(b"\x00", 1)[0]

    if isinstance(x, str):
        return x.encode("utf-8", errors="ignore")

    if isinstance(x, (int, np.integer)):
        return int(x).to_bytes(8, byteorder="big", signed=True)

    return str(x).encode("utf-8", errors="ignore")


def build_train_val_indices_from_feature_info(
    dataset_info: Dict[str, Any],
    *,
    val_ratio: float,
    filter_invalid: bool,
    max_samples: Optional[int],
) -> Tuple[List[int], List[int]]:
    """
    基于 dataset_info.yaml 中的 (sample_ids 或 object_ids) / valid memmap 构建 train/val indices。
    - val 划分：hash(id_bytes) < val_ratio
    - 可选过滤 invalid（valid==0）
    - max_samples：对 train/val 各自最多取 max_samples
    """
    shards = dataset_info["shards"]
    train_indices: List[int] = []
    val_indices: List[int] = []

    offset = 0
    vr = float(val_ratio)

    for s in shards:
        n = int(s["num_samples"])
        paths = s["paths"]

        # ids: 优先 sample_ids（TVL），其次 object_ids（兼容 point 脚本）
        if "sample_ids" in paths:
            id_path = paths["sample_ids"]
        elif "object_ids" in paths:
            id_path = paths["object_ids"]
        else:
            raise KeyError(f"Neither sample_ids nor object_ids found in shard paths: {list(paths.keys())}")

        id_dt = _infer_1d_memmap_dtype(id_path, n)
        ids_mm = np.memmap(id_path, mode="r", dtype=id_dt, shape=(n,))

        valid_mm = None
        if filter_invalid:
            valid_mm = np.memmap(paths["valid"], mode="r", dtype=np.uint8, shape=(n,))

        for local_idx in range(n):
            if filter_invalid and (not bool(valid_mm[local_idx])):
                continue

            id_bytes = _id_value_to_bytes(ids_mm[local_idx])
            is_val = (vr > 0.0) and (_stable_hash_to_unit_interval_bytes(id_bytes) < vr)
            gidx = offset + local_idx

            if is_val:
                if (max_samples is None) or (len(val_indices) < max_samples):
                    val_indices.append(gidx)
            else:
                if (max_samples is None) or (len(train_indices) < max_samples):
                    train_indices.append(gidx)

        offset += n

    return train_indices, val_indices


def collate_tvl_touch_text_features(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    兼容两种 dataset __getitem__ 输出（你 tvl_feature_dataset.py 可能是其中之一）：

    A) 返回:
      - touch_tokens, text_embeds, text_mask, sample_id, global_index, valid

    B) 返回:
      - touch, text, mask, sample_id, global_index, valid
    """
    batch = [b for b in batch if b is not None and bool(b.get("valid", True))]
    if len(batch) == 0:
        return {}

    def get_text(b):
        return b["text_embeds"] if "text_embeds" in b else b["text"]

    def get_mask(b):
        m = b["text_mask"] if "text_mask" in b else b["mask"]
        return m.bool() if hasattr(m, "bool") else torch.as_tensor(m).bool()

    def get_touch(b):
        if "touch_tokens" in b:
            return b["touch_tokens"]
        if "touch" in b:
            return b["touch"]
        # 极端兜底（如果你 dataset 还叫 point_tokens）
        if "point_tokens" in b:
            return b["point_tokens"]
        raise KeyError(f"Sample missing touch tokens. keys={list(b.keys())}")

    text_embeds = torch.stack([get_text(b) for b in batch], dim=0)
    text_mask = torch.stack([get_mask(b) for b in batch], dim=0)
    touch_tokens = torch.stack([get_touch(b) for b in batch], dim=0)

    sample_ids = []
    for b in batch:
        if "sample_id" in b:
            sample_ids.append(b.get("sample_id"))
        elif "object_id" in b:
            sample_ids.append(b.get("object_id"))
        else:
            sample_ids.append(None)

    global_indices = torch.tensor([int(b.get("global_index", -1)) for b in batch], dtype=torch.long)
    valid = torch.tensor([bool(b.get("valid", True)) for b in batch], dtype=torch.bool)

    return {
        "text_embeds": text_embeds,     # (B, L, H)
        "text_mask": text_mask,         # (B, L) bool
        "touch_tokens": touch_tokens,   # (B, G, D)
        "sample_ids": sample_ids,
        "global_indices": global_indices,
        "valid": valid,
    }


@torch.no_grad()
def run_validation(
    *,
    model: nn.Module,
    val_loader: DataLoader,
    cfg: Dict,
    device: torch.device,
    use_amp: bool,
    rank: int,
    progress: Progress | None = None,
    task_id: int | None = None,
) -> Dict[str, float]:
    model.eval()

    loss_cfg = cfg["loss"]
    meters = {k: AverageMeter() for k in ["total", "text_recon", "touch2text_recon", "touch_recon", "contrastive"]}

    for step, batch in enumerate(val_loader):
        if not batch:
            raise RuntimeError(
                f"Empty batch on rank={rank} step={step}. "
                f"This will desync DDP. Fix dataset/collate/filtering."
            )

        touch_feat = batch["touch_tokens"].to(device, non_blocking=True)
        text_feat = batch["text_embeds"].to(device, non_blocking=True)
        text_mask = batch["text_mask"].to(device, non_blocking=True)

        if device.type != "cuda":
            touch_feat = touch_feat.float()
            text_feat = text_feat.float()
        elif not use_amp:
            touch_feat = touch_feat.float()
            text_feat = text_feat.float()

        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(touch_feat=touch_feat, text_feat=text_feat, text_mask=text_mask)

            text_recon = (
                loss_cfg["recon_mse"] * masked_mse(out["text_recon"], text_feat, text_mask)
                + loss_cfg["recon_cos"] * masked_cosine_distance(out["text_recon"], text_feat, text_mask)
            )
            t2t = (
                loss_cfg["recon_mse"] * masked_mse(out["text_recon_from_touch"], text_feat, text_mask)
                + loss_cfg["recon_cos"] * masked_cosine_distance(out["text_recon_from_touch"], text_feat, text_mask)
            )
            t2touch = (
                loss_cfg["recon_mse"] * masked_mse(out["touch_recon"], touch_feat, mask=None)
                + loss_cfg["recon_cos"] * masked_cosine_distance(out["touch_recon"], touch_feat, mask=None)
            )

            contrastive = latent_align_loss(
                out["text_latents"],
                out["touch_latents"],
                align_type=loss_cfg.get("align_type", "contrastive"),
                temperature=float(loss_cfg.get("contrastive_temperature", 0.07)),
                gather_distributed=bool(loss_cfg.get("contrastive_gather", True)),
            )

            total = (
                loss_cfg["w_text_recon"] * text_recon
                + loss_cfg["w_touch2text_recon"] * t2t
                + float(loss_cfg.get("w_touch_recon", 0.0)) * t2touch
                + loss_cfg["w_align"] * contrastive
            )

        bs = touch_feat.size(0)
        meters["total"].update(total.item(), bs)
        meters["text_recon"].update(text_recon.item(), bs)
        meters["touch2text_recon"].update(t2t.item(), bs)
        meters["touch_recon"].update(t2touch.item(), bs)
        meters["contrastive"].update(contrastive.item(), bs)

        if progress is not None and task_id is not None and rank == 0:
            progress.update(
                task_id,
                advance=1,
                loss=meters["total"].avg,
                text=meters["text_recon"].avg,
                t2t=meters["touch2text_recon"].avg,
                t2touch=meters["touch_recon"].avg,
                itc=meters["contrastive"].avg,
            )

    synced = sync_meters_across_ranks(meters, device=device)
    return synced


def main():
    cfg = load_yaml(CONFIG_PATH)

    ddp_info = ddp_setup()
    rank = ddp_info["rank"]
    world_size = ddp_info["world_size"]
    local_rank = ddp_info["local_rank"]

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    base_seed = int(cfg.get("seed", 42))
    set_global_seed(base_seed + rank)

    if is_main_process(rank):
        console.print(
            f"[bold]DDP[/bold]: enabled={ddp_is_enabled()}  world_size={world_size}  rank={rank}  local_rank={local_rank}"
        )
        console.print(f"[bold]device[/bold]: {device}")

    # -------------------------
    # 0) dataset + split + sampler
    # -------------------------
    ds_cfg = cfg["data"]["features"]
    dataset_info_yaml = str(ds_cfg["dataset_info_yaml"])

    val_ratio = float(ds_cfg.get("val_ratio", 0.01))
    filter_invalid = bool(ds_cfg.get("filter_invalid", True))
    require_valid = bool(ds_cfg.get("require_valid", True))
    max_samples = ds_cfg.get("max_samples", None)
    max_samples = None if max_samples is None else int(max_samples)

    dataset_info = load_yaml(dataset_info_yaml)
    train_indices, val_indices = build_train_val_indices_from_feature_info(
        dataset_info,
        val_ratio=val_ratio,
        filter_invalid=filter_invalid,
        max_samples=max_samples,
    )

    if is_main_process(rank):
        console.print(
            f"[bold]Dataset[/bold]: {dataset_info_yaml}\n"
            f"  total(shards)={sum(int(s['num_samples']) for s in dataset_info['shards'])}\n"
            f"  train_samples={len(train_indices)}  val_samples={len(val_indices)}  val_ratio={val_ratio}\n"
            f"  filter_invalid={filter_invalid}  require_valid={require_valid}  max_samples(per split)={max_samples}"
        )

    full_ds = ProcessedTouchTextFeatureDataset(dataset_info_yaml, require_valid=require_valid)
    train_ds = Subset(full_ds, train_indices)
    val_ds = Subset(full_ds, val_indices)

    tr_cfg = cfg["train"]

    train_sampler = None
    val_sampler = None
    if ddp_is_enabled():
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=base_seed,
            drop_last=bool(tr_cfg.get("drop_last", True)),
        )
        val_sampler = DistributedSampler(
            val_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(tr_cfg["batch_size"]),
        num_workers=int(tr_cfg.get("num_workers", 0)),
        pin_memory=bool(tr_cfg.get("pin_memory", True)) and (device.type == "cuda"),
        drop_last=bool(tr_cfg.get("drop_last", True)),
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=collate_tvl_touch_text_features,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(tr_cfg["batch_size"]),
        num_workers=int(tr_cfg.get("num_workers", 0)),
        pin_memory=bool(tr_cfg.get("pin_memory", True)) and (device.type == "cuda"),
        drop_last=False,
        sampler=val_sampler,
        shuffle=False,
        collate_fn=collate_tvl_touch_text_features,
    )

    steps_per_epoch = len(train_loader)
    val_steps = len(val_loader)

    # -------------------------
    # 1) dataset/model consistency checks
    # -------------------------
    mcfg = cfg["model"]
    s0 = dataset_info["shards"][0]

    meta_max_len = int(s0["text"]["max_len"])
    meta_hidden = int(s0["text"]["hidden"])

    # TVL: touch
    if "touch" not in s0:
        raise KeyError(f"dataset_info.yaml shard has no 'touch' field. keys={list(s0.keys())}")
    meta_G = int(s0["touch"]["num_tokens"])
    meta_D = int(s0["touch"].get("hidden", s0["touch"].get("trans_dim")))

    if meta_max_len != int(mcfg["max_text_len"]):
        raise ValueError(f"Mismatch: dataset text.max_len={meta_max_len} vs model.max_text_len={int(mcfg['max_text_len'])}")
    if meta_hidden != int(mcfg["d_text_in"]):
        raise ValueError(f"Mismatch: dataset text.hidden={meta_hidden} vs model.d_text_in={int(mcfg['d_text_in'])}")
    if meta_G != int(mcfg["touch_tokens"]):
        raise ValueError(f"Mismatch: dataset touch.num_tokens={meta_G} vs model.touch_tokens={int(mcfg['touch_tokens'])}")
    if meta_D != int(mcfg["d_touch_in"]):
        raise ValueError(f"Mismatch: dataset touch.hidden={meta_D} vs model.d_touch_in={int(mcfg['d_touch_in'])}")

    # -------------------------
    # 2) model (+ DDP)
    # -------------------------
    model: nn.Module = UnifiedTouchTextAE(cfg["model"]).to(device)

    if ddp_is_enabled():
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(tr_cfg["lr"]),
        betas=tuple(tr_cfg["betas"]),
        weight_decay=float(tr_cfg["weight_decay"]),
    )

    total_steps = int(tr_cfg["epochs"]) * steps_per_epoch
    warmup_steps = int(tr_cfg["scheduler"]["warmup_steps"])
    lr_lambda = make_warmup_cosine_lambda(warmup_steps, total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)

    use_amp = bool(tr_cfg.get("amp", False)) and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    save_dir = Path(tr_cfg["save_dir"])
    if is_main_process(rank):
        save_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    global_step = 0
    loss_cfg = cfg["loss"]
    grad_clip = float(tr_cfg.get("grad_clip", 0.0))
    epochs = int(tr_cfg["epochs"])

    # Rich progress (only rank0 renders)
    progress = None
    if is_main_process(rank):
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            TextColumn("• lr={task.fields[lr]:.2e}"),
            TextColumn("loss={task.fields[loss]:.4f}"),
            TextColumn("text={task.fields[text]:.4f}"),
            TextColumn("t2t={task.fields[t2t]:.4f}"),
            TextColumn("t2touch={task.fields[t2touch]:.4f}"),
            TextColumn("itc={task.fields[itc]:.4f}"),
            console=console,
            transient=False,
        )

    # -------------------------
    # 3) training loop
    # -------------------------
    try:
        if progress is not None:
            progress.start()

        for epoch in range(1, epochs + 1):
            if ddp_is_enabled():
                dist.barrier(device_ids=[local_rank])
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)

            model.train()
            meters = {k: AverageMeter() for k in ["total", "text_recon", "touch2text_recon", "touch_recon", "contrastive"]}

            train_task_id = None
            if progress is not None:
                train_task_id = progress.add_task(
                    f"Train {epoch}/{epochs} (rank0)",
                    total=steps_per_epoch,
                    lr=0.0,
                    loss=0.0,
                    text=0.0,
                    t2t=0.0,
                    t2touch=0.0,
                    itc=0.0,
                )

            for step, batch in enumerate(train_loader):
                if not batch:
                    raise RuntimeError(
                        f"Empty batch on rank={rank} epoch={epoch} step={step}. "
                        f"This will desync DDP. Fix dataset/collate/filtering."
                    )

                touch_feat = batch["touch_tokens"].to(device, non_blocking=True)
                text_feat = batch["text_embeds"].to(device, non_blocking=True)
                text_mask = batch["text_mask"].to(device, non_blocking=True)

                if device.type != "cuda":
                    touch_feat = touch_feat.float()
                    text_feat = text_feat.float()
                elif not use_amp:
                    touch_feat = touch_feat.float()
                    text_feat = text_feat.float()

                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(touch_feat=touch_feat, text_feat=text_feat, text_mask=text_mask)

                    text_recon = (
                        loss_cfg["recon_mse"] * masked_mse(out["text_recon"], text_feat, text_mask)
                        + loss_cfg["recon_cos"] * masked_cosine_distance(out["text_recon"], text_feat, text_mask)
                    )
                    t2t = (
                        loss_cfg["recon_mse"] * masked_mse(out["text_recon_from_touch"], text_feat, text_mask)
                        + loss_cfg["recon_cos"] * masked_cosine_distance(out["text_recon_from_touch"], text_feat, text_mask)
                    )
                    t2touch = (
                        loss_cfg["recon_mse"] * masked_mse(out["touch_recon"], touch_feat, mask=None)
                        + loss_cfg["recon_cos"] * masked_cosine_distance(out["touch_recon"], touch_feat, mask=None)
                    )

                    contrastive = latent_align_loss(
                        out["text_latents"],
                        out["touch_latents"],
                        align_type=loss_cfg.get("align_type", "contrastive"),
                        temperature=float(loss_cfg.get("contrastive_temperature", 0.07)),
                        gather_distributed=bool(loss_cfg.get("contrastive_gather", True)),
                    )

                    total = (
                        loss_cfg["w_text_recon"] * text_recon
                        + loss_cfg["w_touch2text_recon"] * t2t
                        + float(loss_cfg.get("w_touch_recon", 0.0)) * t2touch
                        + loss_cfg["w_align"] * contrastive
                    )

                optim.zero_grad(set_to_none=True)
                scaler.scale(total).backward()

                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                scaler.step(optim)
                scaler.update()
                sched.step()

                bs = touch_feat.size(0)
                meters["total"].update(total.item(), bs)
                meters["text_recon"].update(text_recon.item(), bs)
                meters["touch2text_recon"].update(t2t.item(), bs)
                meters["touch_recon"].update(t2touch.item(), bs)
                meters["contrastive"].update(contrastive.item(), bs)

                global_step += 1

                if progress is not None and train_task_id is not None:
                    lr = optim.param_groups[0]["lr"]
                    progress.update(
                        train_task_id,
                        advance=1,
                        lr=lr,
                        loss=meters["total"].avg,
                        text=meters["text_recon"].avg,
                        t2t=meters["touch2text_recon"].avg,
                        t2touch=meters["touch_recon"].avg,
                        itc=meters["contrastive"].avg,
                    )

            train_synced = sync_meters_across_ranks(meters, device=device)

            if progress is not None and train_task_id is not None:
                progress.remove_task(train_task_id)

            # validation
            val_task_id = None
            if progress is not None:
                val_task_id = progress.add_task(
                    f"Val   {epoch}/{epochs} (global)",
                    total=val_steps,
                    loss=0.0,
                    text=0.0,
                    t2t=0.0,
                    t2touch=0.0,
                    itc=0.0,
                    lr=optim.param_groups[0]["lr"],
                )

            val = run_validation(
                model=model,
                val_loader=val_loader,
                cfg=cfg,
                device=device,
                use_amp=use_amp,
                rank=rank,
                progress=progress,
                task_id=val_task_id,
            )

            if progress is not None and val_task_id is not None:
                progress.remove_task(val_task_id)

            if is_main_process(rank):
                console.print(
                    f"[bold]Epoch {epoch}/{epochs}[/bold] "
                    f"TRAIN total={train_synced['total']:.4f} text={train_synced['text_recon']:.4f} "
                    f"t2t={train_synced['touch2text_recon']:.4f} t2touch={train_synced['touch_recon']:.4f} "
                    f"itc={train_synced['contrastive']:.4f}"
                )
                console.print(
                    f"[bold]Epoch {epoch}/{epochs}[/bold] "
                    f"VAL   total={val['total']:.4f} text={val['text_recon']:.4f} "
                    f"t2t={val['touch2text_recon']:.4f} t2touch={val['touch_recon']:.4f} "
                    f"itc={val['contrastive']:.4f}"
                )

                raw_model = model.module if isinstance(model, DDP) else model
                ckpt = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "cfg": cfg,
                    "model": raw_model.state_dict(),
                    "optimizer": optim.state_dict(),
                    "scheduler": sched.state_dict(),
                    "val_total": val["total"],
                }
                torch.save(ckpt, save_dir / f"epoch_{epoch:03d}.pt")

                if val["total"] < best_val:
                    best_val = val["total"]
                    torch.save(ckpt, save_dir / "best.pt")
                    console.print(f"[green]Saved best.pt[/green] (best_val={best_val:.4f})")

        if ddp_is_enabled():
            dist.barrier(device_ids=[local_rank])

    finally:
        if progress is not None:
            progress.stop()
        ddp_cleanup()


if __name__ == "__main__":
    main()