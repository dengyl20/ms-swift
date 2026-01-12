from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator
import os

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

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

from swift.point_cloud.stage1.src.data.collate_raw import collate_points_and_captions
from swift.point_cloud.stage1.src.data.swift_streaming import SwiftPointTextStreamingDataset
from swift.point_cloud.stage1.src.models.frozen_encoders import FrozenPointBERTTokens, FrozenQwenEmbeddingTable
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

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

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
    rank: int,
    progress: Progress | None = None,
    task_id: int | None = None,
) -> Dict[str, float]:
    model.eval()

    loss_cfg = cfg["loss"]
    meters = {k: AverageMeter() for k in ["total", "text_recon", "point2text_recon", "align"]}

    val_steps = int(cfg["train"]["val_steps"])
    for _ in range(val_steps):
        batch = next(val_iter)
        points = batch["points"].to(device, non_blocking=True)
        captions = batch["captions"]

        # 外部 encoder：不回传梯度
        point_feat = point_enc(points)
        text_feat, text_mask = text_emb(captions)

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

        # progress bar：只在 rank0 更新
        if progress is not None and task_id is not None and rank == 0:
            progress.update(
                task_id,
                advance=1,
                loss=meters["total"].avg,
                text=meters["text_recon"].avg,
                p2t=meters["point2text_recon"].avg,
                align=meters["align"].avg,
            )

    # all_reduce 得到全局验证均值
    synced = sync_meters_across_ranks(meters, device=device)
    return synced


def main():
    import time
    from torch.profiler import (
        profile as torch_profile,
        ProfilerActivity,
        schedule as prof_schedule,
        tensorboard_trace_handler,
    )

    def _now() -> float:
        return time.perf_counter()

    class CudaEventTimer:
        """
        轻量 CUDA event 分段计时器（需要 synchronize 才能读出准确时间）。
        为避免计时本身拖慢训练，只在指定 profiling window 内启用。
        """

        def __init__(self, enabled: bool):
            self.enabled = bool(enabled) and torch.cuda.is_available()
            self._start: dict[str, torch.cuda.Event] = {}
            self._end: dict[str, torch.cuda.Event] = {}

        def start(self, name: str) -> None:
            if not self.enabled:
                return
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._start[name] = ev

        def end(self, name: str) -> None:
            if not self.enabled:
                return
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._end[name] = ev

        def collect(self) -> dict[str, float]:
            if not self.enabled:
                return {}
            torch.cuda.synchronize()
            out: dict[str, float] = {}
            for k, s in self._start.items():
                e = self._end.get(k)
                if e is None:
                    continue
                out[k] = float(s.elapsed_time(e)) / 1000.0  # ms -> s
            return out

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
    # 0) build streaming datasets (rank/world_size sharding)
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
        rank=rank,
        world_size=world_size,
    )
    val_ds = SwiftPointTextStreamingDataset(
        ds_cfg["dataset"],
        seed=int(ds_cfg.get("seed", 42)),
        streaming=bool(ds_cfg.get("streaming", True)),
        remove_unused_columns=bool(ds_cfg.get("remove_unused_columns", False)),
        shuffle_buffer=0,
        assistant_join=str(ds_cfg.get("assistant_join", "all")),
        split="val",
        val_ratio=float(ds_cfg.get("val_ratio", 0.01)),
        max_samples=ds_cfg.get("max_samples", None),
        rank=rank,
        world_size=world_size,
    )

    tr_cfg = cfg["train"]
    train_loader = DataLoader(
        train_ds,
        batch_size=int(tr_cfg["batch_size"]),
        num_workers=int(tr_cfg.get("num_workers", 0)),
        pin_memory=bool(tr_cfg.get("pin_memory", True)) and (device.type == "cuda"),
        drop_last=bool(tr_cfg.get("drop_last", True)),
        collate_fn=collate_points_and_captions,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(tr_cfg["batch_size"]),
        num_workers=int(tr_cfg.get("num_workers", 0)),
        pin_memory=bool(tr_cfg.get("pin_memory", True)) and (device.type == "cuda"),
        drop_last=False,
        collate_fn=collate_points_and_captions,
        prefetch_factor=4,
    )

    train_iter = cycle(train_loader)
    val_iter = cycle(val_loader)

    # -------------------------
    # 1) build frozen external encoders (each rank has its own copy on its GPU)
    # -------------------------
    ext_cfg = cfg["external_encoders"]

    point_enc = FrozenPointBERTTokens(ext_cfg["point_bert"], device=device)
    text_emb = FrozenQwenEmbeddingTable(ext_cfg["qwen"], device=device)

    # consistency checks
    mcfg = cfg["model"]
    expected_point_dim = int(mcfg["d_point_in"])
    expected_point_tokens = int(mcfg["point_tokens"])

    if point_enc.trans_dim != expected_point_dim:
        raise ValueError(
            f"Mismatch: point_enc.trans_dim={point_enc.trans_dim} vs model.d_point_in={expected_point_dim}"
        )

    if ext_cfg["point_bert"]["drop_cls"] is True:
        if point_enc.num_group != expected_point_tokens:
            raise ValueError(
                f"Mismatch: point_enc.num_group={point_enc.num_group} vs model.point_tokens={expected_point_tokens}"
            )
    else:
        if point_enc.num_group + 1 != expected_point_tokens:
            raise ValueError(
                f"Mismatch: point_enc.num_group+1={point_enc.num_group+1} vs model.point_tokens={expected_point_tokens}"
            )

    expected_text_dim = int(mcfg["d_text_in"])
    if text_emb.hidden_size != expected_text_dim:
        raise ValueError(f"Mismatch: qwen hidden_size={text_emb.hidden_size} vs model.d_text_in={expected_text_dim}")

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

    steps_per_epoch = int(tr_cfg["steps_per_epoch"])
    total_steps = int(tr_cfg["epochs"]) * steps_per_epoch
    warmup_steps = int(tr_cfg["scheduler"]["warmup_steps"])
    lr_lambda = make_warmup_cosine_lambda(warmup_steps, total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)

    use_amp = bool(tr_cfg.get("amp", False)) and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    save_dir = Path(tr_cfg["save_dir"])
    if is_main_process(rank):
        save_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    global_step = 0
    loss_cfg = cfg["loss"]
    grad_clip = float(tr_cfg.get("grad_clip", 0.0))

    epochs = int(tr_cfg["epochs"])
    val_steps = int(tr_cfg["val_steps"])

    # -------------------------
    # Profiling / timing config (low intrusion)
    # -------------------------
    # YAML 可选配置示例：
    # profile:
    #   enabled: true
    #   start_step: 10
    #   num_steps: 200
    #   every_n_steps: 1
    #   torch_profiler: true
    #   with_stack: false
    #   record_shapes: false
    #   trace_dir: "tb_profiler"
    prof_cfg = dict(cfg.get("profile", {}))
    prof_enabled = bool(prof_cfg.get("enabled", True))
    prof_start_step = int(prof_cfg.get("start_step", 10))
    prof_num_steps = int(prof_cfg.get("num_steps", 200))
    prof_every = int(prof_cfg.get("every_n_steps", 1))
    prof_use_torch = bool(prof_cfg.get("torch_profiler", True))
    prof_with_stack = bool(prof_cfg.get("with_stack", False))
    prof_record_shapes = bool(prof_cfg.get("record_shapes", False))
    prof_trace_dir = str(prof_cfg.get("trace_dir", "tb_profiler"))

    # rank0-only torch.profiler trace（避免多卡产出多份 trace）
    prof = None
    if prof_enabled and prof_use_torch and is_main_process(rank):
        trace_path = (save_dir / prof_trace_dir)
        trace_path.mkdir(parents=True, exist_ok=True)
        prof = torch_profile(
            activities=[ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else []),
            schedule=prof_schedule(wait=2, warmup=2, active=8, repeat=1),
            on_trace_ready=tensorboard_trace_handler(str(trace_path)),
            record_shapes=prof_record_shapes,
            with_stack=prof_with_stack,
            profile_memory=False,
        )

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
            TextColumn("align={task.fields[align]:.4f}"),
            TextColumn("• data={task.fields[data]:.3f}s"),
            TextColumn("h2d={task.fields[h2d]:.3f}s"),
            TextColumn("penc={task.fields[penc]:.3f}s"),
            TextColumn("temb={task.fields[temb]:.3f}s"),
            TextColumn("fwd={task.fields[fwd]:.3f}s"),
            TextColumn("bwd={task.fields[bwd]:.3f}s"),
            TextColumn("opt={task.fields[opt]:.3f}s"),
            TextColumn("step={task.fields[step]:.3f}s"),
            console=console,
            transient=False,
        )

    # -------------------------
    # 3) training loop
    # -------------------------
    try:
        if progress is not None:
            progress.start()

        if prof is not None:
            prof.__enter__()  # keep profiler alive across epochs

        for epoch in range(1, epochs + 1):
            if ddp_is_enabled():
                dist.barrier()

            # notify dataset epoch (for shuffle seed change)
            train_ds.set_epoch(epoch)
            val_ds.set_epoch(epoch)

            # set train mode
            model.train()

            meters = {k: AverageMeter() for k in ["total", "text_recon", "point2text_recon", "align"]}
            # timing meters (seconds)
            tmeters = {
                k: AverageMeter()
                for k in ["data_wait", "h2d", "point_enc", "text_emb", "fwd_loss", "bwd", "opt_step", "total_step"]
            }

            train_task_id = None
            if progress is not None:
                train_task_id = progress.add_task(
                    f"Train {epoch}/{epochs} (rank0)",
                    total=steps_per_epoch,
                    lr=0.0,
                    loss=0.0,
                    text=0.0,
                    p2t=0.0,
                    align=0.0,
                    data=0.0,
                    h2d=0.0,
                    penc=0.0,
                    temb=0.0,
                    fwd=0.0,
                    bwd=0.0,
                    opt=0.0,
                    step=0.0,
                )

            for _ in range(steps_per_epoch):
                step_wall_t0 = _now()

                # 是否在当前 step 做“精确分段计时”
                # 注意：分段 CUDA event 计时会触发 synchronize（有开销），因此仅在 window 内启用
                want_profile_step = (
                    prof_enabled
                    and (global_step >= prof_start_step)
                    and (global_step < (prof_start_step + prof_num_steps))
                    and ((global_step - prof_start_step) % max(1, prof_every) == 0)
                )

                # 1) Data wait（next(train_iter) 的 wall time，包含 worker 等待 + collate）
                t0 = _now()
                batch = next(train_iter)
                t_data_wait = _now() - t0

                points_cpu = batch["points"]
                captions = batch["captions"]

                # CUDA 分段计时器（按需）
                cet = CudaEventTimer(enabled=(want_profile_step and device.type == "cuda"))

                # 2) H2D copy timing
                if device.type == "cuda":
                    cet.start("h2d")
                    points = points_cpu.to(device, non_blocking=True)
                    cet.end("h2d")
                else:
                    t0 = _now()
                    points = points_cpu.to(device)
                    t_h2d_cpu = _now() - t0

                # 3) 外部 encoder（不回传梯度）
                with torch.no_grad():
                    if device.type == "cuda":
                        cet.start("point_enc")
                        point_feat = point_enc(points)
                        cet.end("point_enc")

                        cet.start("text_emb")
                        text_feat, text_mask = text_emb(captions)
                        cet.end("text_emb")
                    else:
                        t0 = _now()
                        point_feat = point_enc(points)
                        t_point_enc_cpu = _now() - t0

                        t0 = _now()
                        text_feat, text_mask = text_emb(captions)
                        t_text_emb_cpu = _now() - t0

                # 4) forward + loss
                with torch.cuda.amp.autocast(enabled=use_amp):
                    if device.type == "cuda":
                        cet.start("fwd_loss")
                        out = model(point_feat=point_feat, text_feat=text_feat, text_mask=text_mask)
                    else:
                        t0 = _now()
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

                    if device.type == "cuda":
                        cet.end("fwd_loss")
                    else:
                        t_fwd_loss_cpu = _now() - t0

                # 5) backward
                optim.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    cet.start("bwd")
                    scaler.scale(total).backward()
                    cet.end("bwd")
                else:
                    t0 = _now()
                    scaler.scale(total).backward()
                    t_bwd_cpu = _now() - t0

                # 6) opt step
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                if device.type == "cuda":
                    cet.start("opt_step")
                    scaler.step(optim)
                    scaler.update()
                    sched.step()
                    cet.end("opt_step")
                else:
                    t0 = _now()
                    scaler.step(optim)
                    scaler.update()
                    sched.step()
                    t_opt_cpu = _now() - t0

                # metrics
                bs = points.size(0)
                meters["total"].update(total.item(), bs)
                meters["text_recon"].update(text_recon.item(), bs)
                meters["point2text_recon"].update(p2t.item(), bs)
                meters["align"].update(align.item(), bs)

                # timing collect
                t_total_step = _now() - step_wall_t0
                tmeters["data_wait"].update(float(t_data_wait), 1)
                tmeters["total_step"].update(float(t_total_step), 1)

                if device.type == "cuda":
                    # 只有在 want_profile_step 时才做 synchronize 并读出分段 CUDA event 时间
                    if want_profile_step:
                        seg = cet.collect()
                        if seg.get("h2d", 0.0) > 0:
                            tmeters["h2d"].update(seg["h2d"], 1)
                        if seg.get("point_enc", 0.0) > 0:
                            tmeters["point_enc"].update(seg["point_enc"], 1)
                        if seg.get("text_emb", 0.0) > 0:
                            tmeters["text_emb"].update(seg["text_emb"], 1)
                        if seg.get("fwd_loss", 0.0) > 0:
                            tmeters["fwd_loss"].update(seg["fwd_loss"], 1)
                        if seg.get("bwd", 0.0) > 0:
                            tmeters["bwd"].update(seg["bwd"], 1)
                        if seg.get("opt_step", 0.0) > 0:
                            tmeters["opt_step"].update(seg["opt_step"], 1)
                else:
                    tmeters["h2d"].update(float(t_h2d_cpu), 1)
                    tmeters["point_enc"].update(float(t_point_enc_cpu), 1)
                    tmeters["text_emb"].update(float(t_text_emb_cpu), 1)
                    tmeters["fwd_loss"].update(float(t_fwd_loss_cpu), 1)
                    tmeters["bwd"].update(float(t_bwd_cpu), 1)
                    tmeters["opt_step"].update(float(t_opt_cpu), 1)

                global_step += 1

                # torch.profiler step (rank0 only)
                if prof is not None:
                    prof.step()

                # progress update
                if progress is not None and train_task_id is not None:
                    lr = optim.param_groups[0]["lr"]
                    progress.update(
                        train_task_id,
                        advance=1,
                        lr=lr,
                        loss=meters["total"].avg,
                        text=meters["text_recon"].avg,
                        p2t=meters["point2text_recon"].avg,
                        align=meters["align"].avg,
                        data=tmeters["data_wait"].avg,
                        h2d=tmeters["h2d"].avg,
                        penc=tmeters["point_enc"].avg,
                        temb=tmeters["text_emb"].avg,
                        fwd=tmeters["fwd_loss"].avg,
                        bwd=tmeters["bwd"].avg,
                        opt=tmeters["opt_step"].avg,
                        step=tmeters["total_step"].avg,
                    )

            # epoch-level train metrics (global)
            train_synced = sync_meters_across_ranks(meters, device=device)
            train_time_synced = sync_meters_across_ranks(tmeters, device=device)

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
                    p2t=0.0,
                    align=0.0,
                    lr=optim.param_groups[0]["lr"],
                )

            val = run_validation(
                model=model,
                point_enc=point_enc,
                text_emb=text_emb,
                val_iter=val_iter,
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
                    f"p2t={train_synced['point2text_recon']:.4f} align={train_synced['align']:.4f}"
                )
                console.print(
                    f"[bold]Epoch {epoch}/{epochs}[/bold] "
                    f"TIME(s/step) data_wait={train_time_synced['data_wait']:.4f} "
                    f"h2d={train_time_synced.get('h2d', 0.0):.4f} "
                    f"penc={train_time_synced.get('point_enc', 0.0):.4f} "
                    f"temb={train_time_synced.get('text_emb', 0.0):.4f} "
                    f"fwd={train_time_synced.get('fwd_loss', 0.0):.4f} "
                    f"bwd={train_time_synced.get('bwd', 0.0):.4f} "
                    f"opt={train_time_synced.get('opt_step', 0.0):.4f} "
                    f"step={train_time_synced['total_step']:.4f}"
                )
                console.print(
                    f"[bold]Epoch {epoch}/{epochs}[/bold] "
                    f"VAL   total={val['total']:.4f} text={val['text_recon']:.4f} "
                    f"p2t={val['point2text_recon']:.4f} align={val['align']:.4f}"
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
            dist.barrier()

    finally:
        if progress is not None:
            progress.stop()
        if prof is not None:
            try:
                prof.__exit__(None, None, None)
                if is_main_process(rank):
                    console.print(
                        f"[bold]torch.profiler[/bold] trace exported to: {str((save_dir / prof_trace_dir).resolve())}\n"
                        f"View with: tensorboard --logdir {str((save_dir / prof_trace_dir).resolve())}"
                    )
            except Exception:
                pass
        ddp_cleanup()



if __name__ == "__main__":
    main()
