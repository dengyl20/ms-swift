# extract_modelnet40_features.py
from __future__ import annotations

import os
import time
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# 你的 Frozen PointBERT 依赖
# -----------------------------
from swift.llm.model.point_cloud.point_bert import PointBERTConfig, PointBERTEncoder


# ============================================================
# 全局配置（按需直接改这里；不使用 argparse / yaml）
# ============================================================
SEED = 1234

# ModelNet40 test pickle
MODELNET40_PICKLE_PATH = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/PointLLM/PointLLM/modelnet40_test_8192pts_fps.dat"

# 输出特征文件（.pt）
OUTPUT_FEATURE_PATH = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/PointLLM/PointLLM/modelnet40_gray_color.pt"

# 点云基本设置
POINT_NUM = 8192
POINT_DIMS = 6              # 你的数据是 [x y z r g b]
NORMALIZE_PC = False         # 是否对 xyz 做 pc_norm（中心化+unit sphere）
ON_ERROR = "raise"           # "zero" | "raise"
MAX_SAMPLES = -1            # >0 则只处理前 N 个（debug）

# 颜色设置：将随机 RGB 统一替换为固定颜色（中性灰），避免误导冻结的 point encoder
# - 若检测到 RGB 范围像 [0,255]（max>2），则填充 fixed_rgb_01*255（默认 127.5）
# - 若检测到 RGB 范围像 [-1,1]（min<-0.5），则填充 0
# - 否则按 [0,1] 填充 fixed_rgb_01（默认 0.5）
FIXED_RGB_01: Tuple[float, float, float] = (0.5, 0.5, 0.5)

# 推理设置
BATCH_SIZE = 32
NUM_WORKERS = 0             # 明确不做多线程
PIN_MEMORY = True

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# 保存 dtype（与原数据集 memmap 一致：fp16/fp32）
SAVE_DTYPE = "fp16"         # "fp16" | "fp32"

# PointBERT checkpoint & config（保持与你原 extract_features.yaml 一致）
POINT_BERT_CKPT_PATH = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/PointLLM/PointLLM_7B_v1.1_init/point_bert_v1.2.pt"
DROP_CLS = False            # False => 输出 (G+1, D) 也就是 513 tokens（与原数据集对齐）
INPUT_DTYPE = "fp32"        # "fp32" 更稳；可改 "fp16"/"bf16"

POINT_BERT_CONFIG: Dict[str, Any] = {
    "trans_dim": 384,
    "depth": 12,
    "num_heads": 8,
    "drop_path_rate": 0.1,
    "mlp_ratio": 4.0,
    "num_group": 512,
    "group_size": 32,
    "point_dims": 6,
    "encoder_dims": 256,
}


# ============================================================
# ModelNet40 label id -> 英文类别名（标准顺序：0..39）
# 如果你的 dat 文件 label 顺序不同，直接改这个 list 即可。
# ============================================================
MODELNET40_CLASSES: List[str] = [
    "airplane",
    "bathtub",
    "bed",
    "bench",
    "bookshelf",
    "bottle",
    "bowl",
    "car",
    "chair",
    "cone",
    "cup",
    "curtain",
    "desk",
    "door",
    "dresser",
    "flower_pot",
    "glass_box",
    "guitar",
    "keyboard",
    "lamp",
    "laptop",
    "mantel",
    "monitor",
    "night_stand",
    "person",
    "piano",
    "plant",
    "radio",
    "range_hood",
    "sink",
    "sofa",
    "stairs",
    "stool",
    "table",
    "tent",
    "toilet",
    "tv_stand",
    "vase",
    "wardrobe",
    "xbox",
]


# ============================================================
# Utils
# ============================================================
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_int_label(x: Any) -> int:
    # 兼容 numpy scalar / array
    if isinstance(x, np.ndarray):
        return int(x.reshape(-1)[0])
    return int(x)


def pc_norm_np(pc: np.ndarray) -> np.ndarray:
    """
    pc: (N,C), C>=3
    对 xyz 做中心化 + scale 到 unit sphere; 其余 feature 不变
    """
    pc = pc.astype(np.float32, copy=False)
    xyz = pc[:, :3]
    other = pc[:, 3:] if pc.shape[1] > 3 else None

    centroid = xyz.mean(axis=0, dtype=np.float32)
    xyz = xyz - centroid[None, :]

    dist2 = (xyz * xyz).sum(axis=1, dtype=np.float32)
    m = np.sqrt(dist2).max()
    if not np.isfinite(m) or m < 1e-6:
        m = 1.0
    xyz = xyz / m

    if other is not None and other.size > 0:
        out = np.concatenate([xyz, other], axis=1)
    else:
        out = xyz
    return out


def replace_random_rgb_with_fixed(pc: np.ndarray, fixed_rgb_01: Tuple[float, float, float] = FIXED_RGB_01) -> np.ndarray:
    """
    pc: (N,6) with [x y z r g b] (or last 3 dims as RGB-like attributes)
    将后三维统一替换为固定颜色，避免随机颜色误导冻结的 point encoder。

    兼容常见 RGB 数值范围：
      - 若 max(RGB) > 2.0  => 视为 [0,255]，填 fixed_rgb_01*255（默认 127.5）
      - 若 min(RGB) < -0.5 => 视为 [-1,1]，填 0
      - 否则              => 视为 [0,1]，直接填 fixed_rgb_01（默认 0.5）
    """
    if pc.ndim != 2 or pc.shape[1] < 6:
        return pc

    rgb = pc[:, 3:6]
    vmax = float(np.max(rgb))
    vmin = float(np.min(rgb))

    fill = np.asarray(fixed_rgb_01, dtype=np.float32).reshape(3,)
    if vmax > 2.0:
        fill = fill * 255.0
    elif vmin < -0.5:
        fill = np.zeros((3,), dtype=np.float32)

    pc[:, 3:6] = fill[None, :]
    return pc


def parse_save_dtype(s: str) -> torch.dtype:
    s = (s or "fp16").lower()
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32", "float"):
        return torch.float32
    raise ValueError(f"Unsupported SAVE_DTYPE={s}, use fp16/fp32")


# ============================================================
# Dataset (from pickle)
# ============================================================
class ModelNet40PickleDataset(Dataset):
    def __init__(
        self,
        pickle_path: str,
        *,
        normalize_pc: bool = True,
        on_error: str = "zero",
        max_samples: int = -1,
    ):
        super().__init__()
        self.pickle_path = pickle_path
        self.normalize_pc = bool(normalize_pc)
        self.on_error = str(on_error).lower()
        self.max_samples = int(max_samples)

        with open(pickle_path, "rb") as f:
            points_list, labels_list = pickle.load(f)

        if not isinstance(points_list, (list, tuple)) or not isinstance(labels_list, (list, tuple)):
            raise ValueError("pickle must contain (points_list, labels_list)")

        n = min(len(points_list), len(labels_list))
        if self.max_samples > 0:
            n = min(n, self.max_samples)

        self.points_list = points_list[:n]
        self.labels_id = [to_int_label(x) for x in labels_list[:n]]

        # label -> name
        if len(MODELNET40_CLASSES) != 40:
            raise ValueError(f"MODELNET40_CLASSES must have 40 names, got {len(MODELNET40_CLASSES)}")

        self.labels_name = []
        for lab in self.labels_id:
            if 0 <= lab < 40:
                self.labels_name.append(MODELNET40_CLASSES[lab])
            else:
                self.labels_name.append("unknown")

    def __len__(self) -> int:
        return len(self.labels_id)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pts = self.points_list[idx]
        lab_id = int(self.labels_id[idx])
        lab_name = self.labels_name[idx]

        valid = True
        err = ""

        try:
            pc = np.asarray(pts, dtype=np.float32)
            if pc.ndim != 2 or pc.shape[0] != POINT_NUM or pc.shape[1] < 3:
                raise ValueError(f"bad pc shape: {pc.shape}, expect ({POINT_NUM}, >=3)")

            # 保证是 6 维输入（你的 encoder 配置 point_dims=6）
            if pc.shape[1] != POINT_DIMS:
                # 如果你的数据真的不是 6 维，这里会直接报错更安全；
                # 如果你想自动裁剪/补零，可自行扩展。
                raise ValueError(f"bad pc dims: {pc.shape[1]}, expect {POINT_DIMS}")

            if self.normalize_pc:
                pc = pc_norm_np(pc)

            if not np.isfinite(pc).all():
                pc = np.nan_to_num(pc, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

            # ==========================
            # 去掉随机颜色：RGB 置为固定值
            # ==========================
            pc = replace_random_rgb_with_fixed(pc)

        except Exception as e:
            valid = False
            err = f"{type(e).__name__}: {e}"
            if self.on_error == "raise":
                raise
            pc = np.zeros((POINT_NUM, POINT_DIMS), dtype=np.float32)

        return {
            "point_cloud": torch.from_numpy(pc),  # CPU tensor (N,6)
            "label_id": lab_id,
            "label_name": lab_name,
            "valid": valid,
            "error": err,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    pcs = torch.stack([b["point_cloud"] for b in batch], dim=0)  # (B,8192,6)
    label_ids = torch.tensor([b["label_id"] for b in batch], dtype=torch.long)
    label_names = [b["label_name"] for b in batch]
    valid = torch.tensor([1 if b["valid"] else 0 for b in batch], dtype=torch.uint8)
    errors = [b.get("error", "") for b in batch]
    return {
        "point_clouds": pcs,
        "label_ids": label_ids,
        "label_names": label_names,
        "valid": valid,
        "errors": errors,
    }


# ============================================================
# Frozen PointBERT Tokens
# ============================================================
class FrozenPointBERTTokens(nn.Module):
    """
    输入 raw points: (B,8192,6)
    输出 tokens:
      - drop_cls=True:  (B, num_group, trans_dim)
      - drop_cls=False: (B, num_group+1, trans_dim)
    """

    def __init__(self, *, ckpt_path: str, pb_cfg_dict: Dict[str, Any], drop_cls: bool, input_dtype: str, device: str):
        super().__init__()
        self.device = torch.device(device)
        self.drop_cls = bool(drop_cls)

        input_dtype = (input_dtype or "fp32").lower()
        if input_dtype not in ("fp16", "fp32", "bf16"):
            raise ValueError("INPUT_DTYPE must be one of: fp16, bf16, fp32")
        self.input_dtype = input_dtype

        pb_cfg = PointBERTConfig(**pb_cfg_dict)
        self.encoder = PointBERTEncoder(pb_cfg, use_max_pool=False)
        self.encoder.load_checkpoint(ckpt_path, strict=True, map_location="cuda" if "cuda" in device else "cpu", verbose=True)

        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.to(self.device)

        self.trans_dim = int(pb_cfg.trans_dim)
        self.num_group = int(pb_cfg.num_group)

    @torch.inference_mode()
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        points = points.to(self.device, non_blocking=True)

        if self.input_dtype == "fp16":
            points = points.half()
        elif self.input_dtype == "bf16":
            points = points.bfloat16()
        else:
            points = points.float()

        tokens = self.encoder(points, return_tokens=True)  # (B,G+1,D)
        if self.drop_cls:
            tokens = tokens[:, 1:, :]  # (B,G,D)
        return tokens


# ============================================================
# Main
# ============================================================
def main() -> None:
    set_seed(SEED)
    torch.backends.cudnn.benchmark = False

    out_dir = os.path.dirname(OUTPUT_FEATURE_PATH)
    if out_dir:
        ensure_dir(out_dir)

    ds = ModelNet40PickleDataset(
        MODELNET40_PICKLE_PATH,
        normalize_pc=NORMALIZE_PC,
        on_error=ON_ERROR,
        max_samples=MAX_SAMPLES,
    )
    n = len(ds)
    print(f"[modelnet40] loaded: {MODELNET40_PICKLE_PATH}")
    print(f"[modelnet40] num_samples={n}, normalize_pc={NORMALIZE_PC}, max_samples={MAX_SAMPLES}")

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=bool(PIN_MEMORY and ("cuda" in DEVICE)),
        drop_last=False,
        collate_fn=collate_fn,
    )

    encoder = FrozenPointBERTTokens(
        ckpt_path=POINT_BERT_CKPT_PATH,
        pb_cfg_dict=POINT_BERT_CONFIG,
        drop_cls=DROP_CLS,
        input_dtype=INPUT_DTYPE,
        device=DEVICE,
    )

    num_tokens = encoder.num_group if encoder.drop_cls else (encoder.num_group + 1)
    trans_dim = encoder.trans_dim
    save_dt = parse_save_dtype(SAVE_DTYPE)

    # 预分配输出（避免 list 累积导致双份内存）
    point_tokens_out = torch.empty((n, num_tokens, trans_dim), dtype=save_dt, device="cpu")
    object_labels_out: List[str] = [""] * n
    label_ids_out = torch.empty((n,), dtype=torch.long, device="cpu")
    valid_out = torch.empty((n,), dtype=torch.uint8, device="cpu")

    t0 = time.time()
    ptr = 0

    for step, batch in enumerate(loader):
        pcs = batch["point_clouds"]          # CPU (B,8192,6)
        names = batch["label_names"]         # list[str]
        ids = batch["label_ids"]             # CPU tensor
        val = batch["valid"]                 # CPU uint8

        with torch.inference_mode():
            toks = encoder(pcs)              # GPU/CPU (B,T,D)
            toks = toks.to(dtype=save_dt)    # fp16/fp32

        bsz = toks.shape[0]
        s, e = ptr, ptr + bsz

        point_tokens_out[s:e].copy_(toks.detach().cpu())
        object_labels_out[s:e] = list(names)
        label_ids_out[s:e].copy_(ids)
        valid_out[s:e].copy_(val)
        ptr = e

        if (step + 1) % 20 == 0 or ptr == n:
            elapsed = time.time() - t0
            speed = ptr / max(elapsed, 1e-9)
            print(f"[extract] step={step+1:04d} done={ptr}/{n} speed={speed:.2f} samples/s")

    payload = {
        "point_tokens": point_tokens_out,          # (N,T,D) float16/float32
        "object_labels": object_labels_out,        # list[str], len=N
        "label_ids": label_ids_out,                # (N,) int64
        "valid": valid_out,                        # (N,) uint8
        "class_names": MODELNET40_CLASSES,         # list[str], len=40
        "meta": {
            "pickle_path": MODELNET40_PICKLE_PATH,
            "num_samples": int(n),
            "point_num": int(POINT_NUM),
            "point_dims": int(POINT_DIMS),
            "normalize_pc": bool(NORMALIZE_PC),
            "ckpt_path": POINT_BERT_CKPT_PATH,
            "drop_cls": bool(DROP_CLS),
            "num_tokens": int(num_tokens),
            "trans_dim": int(trans_dim),
            "save_dtype": str(save_dt),
        },
    }

    torch.save(payload, OUTPUT_FEATURE_PATH)
    print(f"[done] saved: {OUTPUT_FEATURE_PATH}")
    print(f"[done] point_tokens: shape={tuple(point_tokens_out.shape)}, dtype={point_tokens_out.dtype}")
    print(f"[done] example label_id/name: {int(label_ids_out[0].item())}/{object_labels_out[0]}")


if __name__ == "__main__":
    main()
