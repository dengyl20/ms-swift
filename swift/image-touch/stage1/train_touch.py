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

# ==========================================================
# 关键：尽量少改动 => 仍然复用你原来的 dataset / model / loss
# 但要求 dataset 已适配 touch_tokens/sample_ids（或至少返回 touch_tokens）
# ==========================================================
from swift.point_cloud.stage1.src.data.feature_dataset import ProcessedPointTextFeatureDataset

from swift.point_cloud.stage1.src.models.losses import latent_align_loss, masked_cosine_distance, masked_mse
from swift.point_cloud.stage1.src.models.unified_ae import UnifiedPointTextAE
from swift.point_cloud.stage1.src.utils.common import make_warmup_cosine_lambda, set_global_seed, load_yaml


# （原脚本保留：这些环境变量在 feature-stage1 训练一般用不到，留着也不影响）
os.environ['POINT_CLOUD_DATA_PATH'] = '/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/PointLLM/PointLLM/8192_npy'
os.environ['POINT_CLOUD_ANNO_PATH'] = '/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/PointLLM/PointLLM/PointLLM_brief_description_660K_cleaned.json'

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
    while True:
        for batch in loader:
            yield batch


def ddp_is_enabled() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def ddp_setup() -> Dict[str, int]:
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
# feature 数据集辅助逻辑（适配 sample_ids / object_ids）
# =========================

def _stable_hash_to_unit_interval_bytes(b: bytes) -> float:
    h = hashlib.md5(b).digest()
    v = int.from_bytes(h[:4], byteorder="big", signed=False)
    return v / float(2**32)


def _infer_1d_memmap_dtype(path: str, n: int) -> Any:
    """
    根据文件大小推断 1D memmap dtype（用于 sample_ids/object_ids）。
    """
    size = os.path.getsize(path)
    if n <= 0:
        raise ValueError(f"num_samples must be > 0, got {n}")
    if size % n != 0:
        raise RuntimeError(f"Cannot infer dtype: file_size={size} not divisible by n={n} ({path})")
    rec_bytes = size // n
    if rec_bytes == 8:
        return np.int64
    if rec_bytes == 4:
        return np.int32
    if rec_bytes == 2:
        return np.int16
    if rec_bytes == 1:
        return np.uint8
    return f"S{rec_bytes}"


def _id_value_to_bytes(x: Any) -> bytes:
    """
    把 sample_id/object_id 的单条记录转换成 bytes，供稳定 hash 分 train/val。
    - 若是 bytes string：去掉 \0
    - 若是数值：用 8 bytes big-endian 表示
    """
    # numpy scalar
    if hasattr(x, "item"):
        x = x.item()

    if isinstance(x, (bytes, bytearray)):
        return bytes(x).split(b"\x00", 1)[0]

    if isinstance(x, str):
        return x.encode("utf-8", errors="ignore")

    if isinstance(x, (int, np.integer)):
        # 用 signed=True 更稳（即使出现负数也不会崩）
        return int(x).to_bytes(8, byteorder="big", signed=True)

    # fallback：转字符串
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
    - val 划分：稳定 hash(id_bytes) < val_ratio
    - 可选过滤 invalid（valid==0 的样本）
    - max_samples：对 train / val 各自最多取 max_samples
    """
    shards = dataset_info["shards"]

    train_indices: List[int] = []
    val_indices: List[int] = []

    offset = 0
    vr = float(val_ratio)

    for s in shards:
        n = int(s["num_samples"])
        paths = s["paths"]

        # ids: 优先 sample_ids，其次 object_ids（兼容你两种 yaml）
        if "sample_ids" in paths:
            id_path = paths["sample_ids"]
        elif "object_ids" in paths:
            id_path = paths["object_ids"]
        else:
            raise KeyError(f"Neither sample_ids nor object_ids found in shard paths: {paths.keys()}")

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


def collate_point_text_features(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    兼容 point-text 与 touch-text 两种 feature 返回：
      - text_embeds: (B, L, H)
      - text_mask:   (B, L) bool
      - point_tokens 或 touch_tokens: (B, G, D)

    为了尽量少改动训练代码：collate 最终统一输出 key="point_tokens"
    （即使语义是 touch_tokens）
    """
    batch = [b for b in batch if b is not None and bool(b.get("valid", True))]
    if len(batch) == 0:
        return {}

    text_embeds = torch.stack([b["text_embeds"] for b in batch], dim=0)
    text_mask = torch.stack([b["text_mask"] for b in batch], dim=0)

    # === 修改点：支持 touch_tokens ===
    def _get_tokens(b: Dict[str, Any]) -> torch.Tensor:
        if "point_tokens" in b:
            return b["point_tokens"]
        if "touch_tokens" in b:
            return b["touch_tokens"]
        if "touch" in b:
            return b["touch"]
        raise KeyError(f"Sample has no point_tokens/touch_tokens keys: {b.keys()}")

    point_tokens = torch.stack([_get_tokens(b) for b in batch], dim=0)

    # ids / indices 兼容
    object_ids = []
    for b in batch:
        if "object_id" in b:
            object_ids.append(b.get("object_id", ""))
        elif "sample_id" in b:
            object_ids.append(str(b.get("sample_id", "")))
        else:
            object_ids.append("")

    global_indices = torch.tensor([int(b.get("global_index", b.get("global_indices", -1))) for b in batch], dtype=torch.long)
    valid = torch.tensor([bool(b.get("valid", True)) for b in batch], dtype=torch.bool)

    return {
        "text_embeds": text_embeds,
        "text_mask": text_mask,
        # 训练代码不改 => 仍输出 point_tokens
        "point_tokens": point_tokens,
        "object_ids": object_ids,
        "global_indices": global_indices,
        "valid": valid,
    }


def _model_forward_compat(model: nn.Module, point_feat: torch.Tensor, text_feat: torch.Tensor, text_mask: torch.Tensor) -> Dict[str, Any]:
    """
    兼容两种 forward 签名：
      - model(point_feat=..., text_feat=..., text_mask=...)
      - model(touch_feat=..., text_feat=..., text_mask=...)
    """
    try:
        return model(point_feat=point_feat, text_feat=text_feat, text_mask=text_mask)
    except TypeError:
        return model(touch_feat=point_feat, text_feat=text_feat, text_mask=text_mask)


def _out_get(out: Dict[str, Any], *keys: str):
    for k in keys:
        if k in out:
            return out[k]
    raise KeyError(f"None of keys exists in out: {keys}. out_keys={list(out.keys())}")


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

        point_feat = batch["point_tokens"].to(device, non_blocking=True)  # 语义可为 touch
        text_feat = batch["text_embeds"].to(device, non_blocking=True)
        text_mask = batch["text_mask"].to(device, non_blocking=True)

        if device.type != "cuda":
            point_feat = point_feat.float()
            text_feat = text_feat.float()
        elif not use_amp:
            point_feat = point_feat.float()
            text_feat = text_feat.float()

        with torch.amp.autocast("cuda", enabled=use_amp):
            out = _model_forward_compat(model, point_feat=point_feat, text_feat=text_feat, text_mask=text_mask)

            text_recon_pred = _out_get(out, "text_recon")
            text_recon_from_point = _out_get(out, "text_recon_from_point", "text_recon_from_touch")
            point_recon_pred = _out_get(out, "point_recon", "touch_recon")

            text_latents = _out_get(out, "text_latents")
            point_latents = _out_get(out, "point_latents", "touch_latents")

            text_recon = (
                loss_cfg["recon_mse"] * masked_mse(text_recon_pred, text_feat, text_mask)
                + loss_cfg["recon_cos"] * masked_cosine_distance(text_recon_pred, text_feat, text_mask)
            )
            p2t = (
                loss_cfg["recon_mse"] * masked_mse(text_recon_from_point, text_feat, text_mask)
                + loss_cfg["recon_cos"] * masked_cosine_distance(text_recon_from_point, text_feat, text_mask)
            )
            p2p = (
                loss_cfg["recon_mse"] * masked_mse(point_recon_pred, point_feat, mask=None)
                + loss_cfg["recon_cos"] * masked_cosine_distance(point_recon_pred, point_feat, mask=None)
            )

            contrastive = latent_align_loss(
                text_latents,
                point_latents,
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
    # 0) build feature datasets + train/val split + DDP sampler
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

    # dataset（要求该 dataset 能读 touch_tokens/sample_ids 或返回 touch_tokens）
    full_ds = ProcessedPointTextFeatureDataset(dataset_info_yaml, require_valid=require_valid)

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

    steps_per_epoch = len(train_loader)
    val_steps = len(val_loader)

    # -------------------------
    # 1) dataset/model consistency checks（适配 touch/point 两种 metadata 命名）
    # -------------------------
    mcfg = cfg["model"]

    s0 = dataset_info["shards"][0]

    # text meta
    meta_max_len = int(s0["text"]["max_len"])
    meta_hidden = int(s0["text"]["hidden"])

    # modality meta: 优先 touch，其次 point
    if "touch" in s0:
        mod0 = s0["touch"]
    elif "point" in s0:
        mod0 = s0["point"]
    else:
        raise KeyError(f"Shard has neither 'touch' nor 'point': keys={s0.keys()}")

    meta_G = int(mod0["num_tokens"])
    meta_D = int(mod0.get("trans_dim", mod0.get("hidden")))

    # model expected: 允许你 config 里仍用 point_tokens/d_point_in，也允许用 touch_tokens/d_touch_in
    exp_max_len = int(mcfg["max_text_len"])
    exp_text_hidden = int(mcfg["d_text_in"])
    exp_G = int(mcfg.get("touch_tokens", mcfg.get("point_tokens")))
    exp_D = int(mcfg.get("d_touch_in", mcfg.get("d_point_in")))

    if meta_max_len != exp_max_len:
        raise ValueError(f"Mismatch: dataset text.max_len={meta_max_len} vs model.max_text_len={exp_max_len}")
    if meta_hidden != exp_text_hidden:
        raise ValueError(f"Mismatch: dataset text.hidden={meta_hidden} vs model.d_text_in={exp_text_hidden}")
    if meta_G != exp_G:
        raise ValueError(f"Mismatch: dataset tokens={meta_G} vs model tokens={exp_G} (touch_tokens/point_tokens)")
    if meta_D != exp_D:
        raise ValueError(f"Mismatch: dataset dim={meta_D} vs model dim={exp_D} (d_touch_in/d_point_in)")

    # -------------------------
    # 2) model (+ DDP)
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

                point_feat = batch["point_tokens"].to(device, non_blocking=True)  # 语义可为 touch
                text_feat = batch["text_embeds"].to(device, non_blocking=True)
                text_mask = batch["text_mask"].to(device, non_blocking=True)

                if device.type != "cuda":
                    point_feat = point_feat.float()
                    text_feat = text_feat.float()
                elif not use_amp:
                    point_feat = point_feat.float()
                    text_feat = text_feat.float()

                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = _model_forward_compat(model, point_feat=point_feat, text_feat=text_feat, text_mask=text_mask)

                    text_recon_pred = _out_get(out, "text_recon")
                    text_recon_from_point = _out_get(out, "text_recon_from_point", "text_recon_from_touch")
                    point_recon_pred = _out_get(out, "point_recon", "touch_recon")

                    text_latents = _out_get(out, "text_latents")
                    point_latents = _out_get(out, "point_latents", "touch_latents")

                    text_recon = (
                        loss_cfg["recon_mse"] * masked_mse(text_recon_pred, text_feat, text_mask)
                        + loss_cfg["recon_cos"] * masked_cosine_distance(text_recon_pred, text_feat, text_mask)
                    )
                    p2t = (
                        loss_cfg["recon_mse"] * masked_mse(text_recon_from_point, text_feat, text_mask)
                        + loss_cfg["recon_cos"] * masked_cosine_distance(text_recon_from_point, text_feat, text_mask)
                    )
                    p2p = (
                        loss_cfg["recon_mse"] * masked_mse(point_recon_pred, point_feat, mask=None)
                        + loss_cfg["recon_cos"] * masked_cosine_distance(point_recon_pred, point_feat, mask=None)
                    )

                    contrastive = latent_align_loss(
                        text_latents,
                        point_latents,
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

            train_synced = sync_meters_across_ranks(meters, device=device)

            if progress is not None and train_task_id is not None:
                progress.remove_task(train_task_id)

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