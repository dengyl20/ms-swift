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

# === 修改点：替换为新数据集 ===
from swift.point_cloud.stage1.src.data.feature_dataset import ProcessedPointTextFeatureDataset

# === 修改点：loss / model / utils ===
from swift.point_cloud.stage1.src.models.losses import latent_align_loss, masked_cosine_distance, masked_mse
from swift.point_cloud.stage1.src.models.unified_ae import UnifiedPointTextAE
from swift.point_cloud.stage1.src.utils.common import make_warmup_cosine_lambda, set_global_seed, load_yaml


os.environ['POINT_CLOUD_DATA_PATH'] = '/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/PointLLM/PointLLM/8192_npy'
os.environ['POINT_CLOUD_ANNO_PATH'] = '/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/PointLLM/PointLLM/PointLLM_brief_description_660K.json'

CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "real_swift.yaml"
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
    # 旧脚本保留：虽然新流程不再需要 cycle 控制 steps_per_epoch，
    # 但保留该函数不影响逻辑（也不再使用它）。
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
# 新增：feature 数据集辅助逻辑
# =========================


def _stable_hash_to_unit_interval_bytes(b: bytes) -> float:
    """
    与原 streaming dataset 的 _stable_hash_to_unit_interval 等价（取 md5 前 4 bytes -> [0,1)）。
    这里直接用 bytes，避免 decode 成 str 的开销。
    """
    h = hashlib.md5(b).digest()
    v = int.from_bytes(h[:4], byteorder="big", signed=False)  # 32-bit
    return v / float(2**32)


def build_train_val_indices_from_feature_info(
    dataset_info: Dict[str, Any],
    *,
    val_ratio: float,
    filter_invalid: bool,
    max_samples: Optional[int],
) -> Tuple[List[int], List[int]]:
    """
    基于 dataset_info.yaml 中的 object_ids / valid memmap 构建 train/val indices。
    - val 划分：稳定 hash(object_id) < val_ratio
    - 可选过滤 invalid（valid==0 的样本）
    - max_samples：复刻旧逻辑，对 train / val 各自最多取 max_samples
    """
    shards = dataset_info["shards"]

    train_indices: List[int] = []
    val_indices: List[int] = []

    offset = 0
    vr = float(val_ratio)

    for s in shards:
        n = int(s["num_samples"])
        paths = s["paths"]

        # 只打开轻量 memmap（object_ids, valid），避免提前打开大 feature memmap
        obj_mm = np.memmap(paths["object_ids"], mode="r", dtype="S32", shape=(n,))
        valid_mm = None
        if filter_invalid:
            valid_mm = np.memmap(paths["valid"], mode="r", dtype=np.uint8, shape=(n,))

        for local_idx in range(n):
            if filter_invalid:
                if not bool(valid_mm[local_idx]):
                    continue

            # S32 bytes -> 去掉末尾 '\0'
            obj_b = obj_mm[local_idx].tobytes().split(b"\x00", 1)[0]

            is_val = (vr > 0.0) and (_stable_hash_to_unit_interval_bytes(obj_b) < vr)
            gidx = offset + local_idx

            if is_val:
                if (max_samples is None) or (len(val_indices) < max_samples):
                    val_indices.append(gidx)
            else:
                if (max_samples is None) or (len(train_indices) < max_samples):
                    train_indices.append(gidx)

            # 复刻旧逻辑：train/val 各自达到 max_samples 后仍继续扫描另一侧
            # 因此这里不做全局 break。

        offset += n

    return train_indices, val_indices


def collate_point_text_features(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    collate 新数据集返回的 features：
      - text_embeds: (B, L, H)
      - text_mask:   (B, L) bool
      - point_tokens:(B, G, D)
    其余字段保留便于 debug。
    """
    # 双保险：若 batch 中存在 invalid，直接过滤
    batch = [b for b in batch if b is not None and bool(b.get("valid", True))]
    if len(batch) == 0:
        # 理论上不应发生（我们已在 indices 构建时过滤 invalid）
        return {}

    text_embeds = torch.stack([b["text_embeds"] for b in batch], dim=0)
    text_mask = torch.stack([b["text_mask"] for b in batch], dim=0)
    point_tokens = torch.stack([b["point_tokens"] for b in batch], dim=0)

    object_ids = [b.get("object_id", "") for b in batch]
    global_indices = torch.tensor([int(b.get("global_index", -1)) for b in batch], dtype=torch.long)
    valid = torch.tensor([bool(b.get("valid", True)) for b in batch], dtype=torch.bool)

    return {
        "text_embeds": text_embeds,
        "text_mask": text_mask,
        "point_tokens": point_tokens,
        "object_ids": object_ids,
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
    meters = {k: AverageMeter() for k in ["total", "text_recon", "point2text_recon", "point_recon", "contrastive"]}

    for step, batch in enumerate(val_loader):
        if not batch:
            raise RuntimeError(
                f"Empty batch on rank={rank} step={step}. "
                f"This will desync DDP. Fix dataset/collate/filtering."
            )

        point_feat = batch["point_tokens"].to(device, non_blocking=True)
        text_feat = batch["text_embeds"].to(device, non_blocking=True)
        text_mask = batch["text_mask"].to(device, non_blocking=True)

        # 若不用 AMP 或在 CPU 上训练，确保 dtype 合理
        if device.type != "cuda":
            point_feat = point_feat.float()
            text_feat = text_feat.float()
        elif not use_amp:
            point_feat = point_feat.float()
            text_feat = text_feat.float()

        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(point_feat=point_feat, text_feat=text_feat, text_mask=text_mask)

            text_recon = (
                loss_cfg["recon_mse"] * masked_mse(out["text_recon"], text_feat, text_mask)
                + loss_cfg["recon_cos"] * masked_cosine_distance(out["text_recon"], text_feat, text_mask)
            )
            p2t = (
                loss_cfg["recon_mse"] * masked_mse(out["text_recon_from_point"], text_feat, text_mask)
                + loss_cfg["recon_cos"] * masked_cosine_distance(out["text_recon_from_point"], text_feat, text_mask)
            )
            p2p = (
                loss_cfg["recon_mse"] * masked_mse(out["point_recon"], point_feat, mask=None)
                + loss_cfg["recon_cos"] * masked_cosine_distance(out["point_recon"], point_feat, mask=None)
            )

            contrastive = latent_align_loss(
                out["text_latents"],
                out["point_latents"],
                align_type=loss_cfg.get("align_type", "contrastive"),
                temperature=float(loss_cfg.get("contrastive_temperature", 0.07)),
                gather_distributed=bool(loss_cfg.get("contrastive_gather", True)),
            )

            total = (
                loss_cfg["w_text_recon"] * text_recon
                + loss_cfg["w_point2text_recon"] * p2t
                + float(loss_cfg.get("w_point_recon", 0.0)) * p2p
                + loss_cfg["w_align"] * contrastive
            )

        bs = point_feat.size(0)
        meters["total"].update(total.item(), bs)
        meters["text_recon"].update(text_recon.item(), bs)
        meters["point2text_recon"].update(p2t.item(), bs)
        meters["point_recon"].update(p2p.item(), bs)
        meters["contrastive"].update(contrastive.item(), bs)

        # progress bar：只在 rank0 更新
        if progress is not None and task_id is not None and rank == 0:
            progress.update(
                task_id,
                advance=1,
                loss=meters["total"].avg,
                text=meters["text_recon"].avg,
                p2t=meters["point2text_recon"].avg,
                p2p=meters["point_recon"].avg,
                itc=meters["contrastive"].avg,
            )

    # all_reduce 得到全局验证均值
    synced = sync_meters_across_ranks(meters, device=device)
    return synced


def main():
    cfg = load_yaml(CONFIG_PATH)

    ddp_info = ddp_setup()
    rank = ddp_info["rank"]
    world_size = ddp_info["world_size"]
    local_rank = ddp_info["local_rank"]

    # device
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    # seed：不同 rank 用不同 seed，避免 dropout 等随机项完全一致
    base_seed = int(cfg.get("seed", 42))
    set_global_seed(base_seed + rank)

    if is_main_process(rank):
        console.print(
            f"[bold]DDP[/bold]: enabled={ddp_is_enabled()}  world_size={world_size}  rank={rank}  local_rank={local_rank}"
        )
        console.print(f"[bold]device[/bold]: {device}")

    # -------------------------
    # 0) build feature datasets (map-style) + train/val split + DDP sampler sharding
    # -------------------------
    ds_cfg = cfg["data"]["features"]
    dataset_info_yaml = str(ds_cfg["dataset_info_yaml"])

    val_ratio = float(ds_cfg.get("val_ratio", 0.01))
    filter_invalid = bool(ds_cfg.get("filter_invalid", True))
    require_valid = bool(ds_cfg.get("require_valid", True))
    max_samples = ds_cfg.get("max_samples", None)
    max_samples = None if max_samples is None else int(max_samples)

    # 用 dataset_info.yaml 构建 indices（避免提前打开大 memmap）
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

    # 实际 dataset：支持随机访问、len()、可 DataLoader 多 worker
    full_ds = ProcessedPointTextFeatureDataset(dataset_info_yaml, require_valid=require_valid)

    train_ds = Subset(full_ds, train_indices)
    val_ds = Subset(full_ds, val_indices)

    tr_cfg = cfg["train"]

    # DDP：用 DistributedSampler 自动分片/shuffle，不再需要手动 shard 或 steps_per_epoch 控制
    train_sampler = None
    val_sampler = None

    if ddp_is_enabled():
        # train：shuffle=True
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=base_seed,
            # 使用 drop_last 对齐各 rank steps，且避免 padding 造成重复样本
            drop_last=bool(tr_cfg.get("drop_last", True)),
        )
        # val：shuffle=False；drop_last=True 避免 padding 重复（最多丢掉 <world_size 条尾巴）
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
        shuffle=(train_sampler is None),  # 非 DDP 情况下用 DataLoader shuffle
        collate_fn=collate_point_text_features,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(tr_cfg["batch_size"]),
        num_workers=int(tr_cfg.get("num_workers", 0)),
        pin_memory=bool(tr_cfg.get("pin_memory", True)) and (device.type == "cuda"),
        drop_last=False,
        sampler=val_sampler,
        shuffle=False,
        collate_fn=collate_point_text_features,
    )

    # steps_per_epoch / val_steps：现在可直接由 len(dataloader) 得到
    steps_per_epoch = len(train_loader)
    val_steps = len(val_loader)

    # -------------------------
    # 1) dataset/model consistency checks (替代旧 external encoder checks)
    # -------------------------
    mcfg = cfg["model"]

    # 从 shard metadata 校验（只看第一个 shard；通常所有 shard 一致）
    s0 = dataset_info["shards"][0]
    meta_max_len = int(s0["text"]["max_len"])
    meta_hidden = int(s0["text"]["hidden"])
    meta_G = int(s0["point"]["num_tokens"])
    meta_D = int(s0["point"]["trans_dim"])

    if meta_max_len != int(mcfg["max_text_len"]):
        raise ValueError(
            f"Mismatch: dataset text.max_len={meta_max_len} vs model.max_text_len={int(mcfg['max_text_len'])}"
        )
    if meta_hidden != int(mcfg["d_text_in"]):
        raise ValueError(
            f"Mismatch: dataset text.hidden={meta_hidden} vs model.d_text_in={int(mcfg['d_text_in'])}"
        )
    if meta_G != int(mcfg["point_tokens"]):
        raise ValueError(f"Mismatch: dataset point.num_tokens={meta_G} vs model.point_tokens={int(mcfg['point_tokens'])}")
    if meta_D != int(mcfg["d_point_in"]):
        raise ValueError(
            f"Mismatch: dataset point.trans_dim={meta_D} vs model.d_point_in={int(mcfg['d_point_in'])}"
        )

    # -------------------------
    # 2) build trainable mapping model (+ DDP)
    # -------------------------
    model = UnifiedPointTextAE(cfg["model"]).to(device)

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
            TextColumn("p2t={task.fields[p2t]:.4f}"),
            TextColumn("p2p={task.fields[p2p]:.4f}"),
            TextColumn("itc={task.fields[itc]:.4f}"),
            console=console,
            transient=False,
        )

    # -------------------------
    # 3) training loop (按 dataloader 真实长度跑一整个 epoch)
    # -------------------------
    try:
        if progress is not None:
            progress.start()

        for epoch in range(1, epochs + 1):
            if ddp_is_enabled():
                dist.barrier(device_ids=[local_rank])
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)

            # set train mode
            model.train()

            meters = {k: AverageMeter() for k in ["total", "text_recon", "point2text_recon", "point_recon", "contrastive"]}

            train_task_id = None
            if progress is not None:
                train_task_id = progress.add_task(
                    f"Train {epoch}/{epochs} (rank0)",
                    total=steps_per_epoch,
                    lr=0.0,
                    loss=0.0,
                    text=0.0,
                    p2t=0.0,
                    p2p=0.0,
                    itc=0.0,
                )

            for step, batch in enumerate(train_loader):
                if not batch:
                    raise RuntimeError(
                        f"Empty batch on rank={rank} epoch={epoch} step={step}. "
                        f"This will desync DDP. Fix dataset/collate/filtering."
                    )

                point_feat = batch["point_tokens"].to(device, non_blocking=True)
                text_feat = batch["text_embeds"].to(device, non_blocking=True)
                text_mask = batch["text_mask"].to(device, non_blocking=True)

                # 若不用 AMP 或在 CPU 上训练，确保 dtype 合理
                if device.type != "cuda":
                    point_feat = point_feat.float()
                    text_feat = text_feat.float()
                elif not use_amp:
                    point_feat = point_feat.float()
                    text_feat = text_feat.float()

                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(point_feat=point_feat, text_feat=text_feat, text_mask=text_mask)

                    text_recon = (
                        loss_cfg["recon_mse"] * masked_mse(out["text_recon"], text_feat, text_mask)
                        + loss_cfg["recon_cos"] * masked_cosine_distance(out["text_recon"], text_feat, text_mask)
                    )
                    p2t = (
                        loss_cfg["recon_mse"] * masked_mse(out["text_recon_from_point"], text_feat, text_mask)
                        + loss_cfg["recon_cos"] * masked_cosine_distance(out["text_recon_from_point"], text_feat, text_mask)
                    )
                    p2p = (
                        loss_cfg["recon_mse"] * masked_mse(out["point_recon"], point_feat, mask=None)
                        + loss_cfg["recon_cos"] * masked_cosine_distance(out["point_recon"], point_feat, mask=None)
                    )

                    contrastive = latent_align_loss(
                        out["text_latents"],
                        out["point_latents"],
                        align_type=loss_cfg.get("align_type", "contrastive"),
                        temperature=float(loss_cfg.get("contrastive_temperature", 0.07)),
                        gather_distributed=bool(loss_cfg.get("contrastive_gather", True)),
                    )

                    total = (
                        loss_cfg["w_text_recon"] * text_recon
                        + loss_cfg["w_point2text_recon"] * p2t
                        + float(loss_cfg.get("w_point_recon", 0.0)) * p2p
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

                bs = point_feat.size(0)
                meters["total"].update(total.item(), bs)
                meters["text_recon"].update(text_recon.item(), bs)
                meters["point2text_recon"].update(p2t.item(), bs)
                meters["point_recon"].update(p2p.item(), bs)
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
                        p2t=meters["point2text_recon"].avg,
                        p2p=meters["point_recon"].avg,
                        itc=meters["contrastive"].avg,
                    )

            # epoch-level train metrics (global)
            train_synced = sync_meters_across_ranks(meters, device=device)

            if progress is not None and train_task_id is not None:
                progress.remove_task(train_task_id)

            # validation（全量 val_loader）
            val_task_id = None
            if progress is not None:
                val_task_id = progress.add_task(
                    f"Val   {epoch}/{epochs} (global)",
                    total=val_steps,
                    loss=0.0,
                    text=0.0,
                    p2t=0.0,
                    p2p=0.0,
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
                    f"p2t={train_synced['point2text_recon']:.4f} p2p={train_synced['point_recon']:.4f} "
                    f"itc={train_synced['contrastive']:.4f}"
                )
                console.print(
                    f"[bold]Epoch {epoch}/{epochs}[/bold] "
                    f"VAL   total={val['total']:.4f} text={val['text_recon']:.4f} "
                    f"p2t={val['point2text_recon']:.4f} p2p={val['point_recon']:.4f} "
                    f"itc={val['contrastive']:.4f}"
                )

                # save ckpt (rank0 only)
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
