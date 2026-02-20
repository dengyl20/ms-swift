#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Render point clouds from two datasets to PNG images (headless-friendly).

New workflow (JSONL-driven):
1) Dataset 1 (PointLLM-style npy):
   - Read ALL lines from DATASET1_JSONL_PATH (jsonl).
   - Each line is expected to contain:
       - object_id (str)
       - ground_truth (any JSON type; will be stringified)
       - pred (any JSON type; will be stringified)
   - Load point cloud: {DATASET1_DATA_PATH}/{object_id}_{POINTNUM}.npy
     Columns: [x, y, z, r, g, b]  (rgb in [0,1] or [0,255])
   - Save a 3-view panel PNG and overlay GT / Pred text for easy comparison.

2) Dataset 2 (ModelNet40 pickle-style):
   - Read ALL lines from DATASET2_JSONL_PATH (jsonl).
   - Each line is expected to contain:
       - ds_idx (int; index into the pickle-loaded points list)
       - gt_label (any JSON type; will be stringified)
       - pred_text (any JSON type; will be stringified)
   - Pickle file contains (points_list, labels_list, ...)
     points are (POINTNUM, 6): [x, y, z, nx, ny, nz]
     We map (nx,ny,nz) from [-1,1] to pseudo-RGB by (n+1)/2.
   - Save a 3-view panel PNG and overlay GT label / Pred text.

Only 3 views are rendered: front / top / side (正视/俯视/侧视).

Edit the GLOBAL CONFIG section. No argparse is used.
"""

from __future__ import annotations

import json
import pickle
import random
import sys
import textwrap
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt


# ======================================================================================
# GLOBAL CONFIG (edit here)
# ======================================================================================

# Common
POINTNUM: int = 8192
OUTPUT_DIR: str = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/swift/point_cloud/stage1/src/eval/results/pointcloud_renders"
RANDOM_SEED: int = 0

# Dataset render switches
# - Set to False if you want to skip rendering that dataset.
RENDER_DATASET1: bool = False
RENDER_DATASET2: bool = True

# Rendering quality / clarity
MAX_POINTS_TO_RENDER: int = 8192   # 0 -> render all points
POINT_SIZE: float = 1.0            # scatter "s" (points^2). Tune: 0.6~2.0
ALPHA: float = 1.0
MARKER: str = "o"                  # "o" usually clearer than "."
DEPTHSHADE: bool = True           # True gives depth perception but alters colors slightly
AXIS_OFF: bool = True
DPI: int = 350                     # increase DPI for clearer details
PANEL_FIGSIZE: Tuple[float, float] = (5.0, 5.0)  # size for EACH view (inches)

# Axis / camera tuning (clarity-focused)
ROBUST_AXES: bool = True           # robust axis scaling to reduce the impact of outliers
ROBUST_PERCENTILE: float = 99.5    # keep central 99.5% range per axis (0.25% trimmed each side)
AXIS_PAD_RATIO: float = 0.03       # add padding around the bbox
SORT_POINTS_BY_VIEW: bool = True   # draw far points first, near points last (better occlusion)
PROJ_TYPE: str = "ortho"           # "ortho" (crisper) or "persp"
CAMERA_DIST: Optional[float] = 7.5 # smaller -> zoom in (matplotlib may deprecate ax.dist); None disables

# 3 viewpoints (name, elev, azim)
# - 正视图: elev=0,  azim=90  (camera on +Y looking toward origin)
# - 俯视图: elev=90, azim=0   (camera on +Z looking down)
# - 侧视图: elev=0,  azim=0   (camera on +X looking toward origin)
#   (If you prefer the opposite side, set azim=180.)
VIEWS: List[Tuple[str, float, float]] = [
    ("+Y", 0, 90),
    ("+Z", 90, 0),
    ("+X", 0, 0),
]

# Text overlay
TEXT_FONT_SIZE: int = 10
TEXT_WRAP_WIDTH: int = 110         # wrap width for overlay text (characters)
MAX_TEXT_CHARS_IN_FIG: int = 800   # prevent huge captions from shrinking the plots too much

# Error handling
STRICT_MISSING_FILES: bool = False  # True -> raise on missing npy / bad ds_idx; False -> skip with warning

# -----------------------
# Dataset 1 (PointLLM npy)
# -----------------------
DATASET1_DATA_PATH: str = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/PointLLM/PointLLM/8192_npy"

# JSONL that drives what to render (NEW)
DATASET1_JSONL_PATH: str = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/swift/point_cloud/stage1/src/eval/outputs/objaverse/infer_point_ae_qwen3_omni_eval.jsonl"  # <<< fill in: path/to/dataset1_results.jsonl
DATASET1_JSONL_OBJECT_ID_KEY: str = "object_id"
DATASET1_JSONL_GT_KEY: str = "gt_answer"
DATASET1_JSONL_PRED_KEY: str = "pred_inject_ae_embedding"

DATASET1_NORMALIZE_XYZ: bool = True
DATASET1_USE_LAST3_AS_COLOR: bool = True

# Keep consistent with your original filtering of corrupted colored-point IDs
DATASET1_SKIP_CORRUPTED_COLOR_IDS: bool = True
DATASET1_CORRUPTED_COLOR_IDS = {
    "6760e543e1d645d5aaacd3803bcae524",
    "b91c0711149d460a8004f9c06d3b7f38",
}

# -----------------------
# Dataset 2 (ModelNet40 pickle)
# -----------------------
DATASET2_DAT_PATH: str = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/PointLLM/PointLLM/modelnet40_test_8192pts_fps.dat"

# JSONL that drives what to render (NEW)
DATASET2_JSONL_PATH: str = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/swift/point_cloud/stage1/src/eval/modelnet40_infer_outputs/predictions_modelnet40_cls.jsonl"  # <<< fill in: path/to/dataset2_results.jsonl
DATASET2_JSONL_INDEX_KEY: str = "ds_idx"
DATASET2_JSONL_GT_LABEL_KEY: str = "gt_label"
DATASET2_JSONL_PRED_TEXT_KEY: str = "pred_text"

DATASET2_NORMALIZE_XYZ: bool = False
DATASET2_SHOW_PICKLE_LABEL: bool = False  # True -> also show pickle label on the figure (debug)

# Dataset 2 coloring:
#   "last3": map last3 (normals) to pseudo RGB by (x+1)/2, also supports true RGB
#   "constant": use a fixed color
#   "z": color by normalized z (grayscale)
DATASET2_COLOR_MODE: str = "constant"
DATASET2_CONSTANT_RGB: Tuple[float, float, float] = (0.5, 0.5, 0.5)


# ======================================================================================
# Utilities
# ======================================================================================

def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def stringify(x) -> str:
    """Safely convert JSON values (str/int/list/dict/None/...) into a printable string."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)


def clip_text(s: str, max_chars: int) -> str:
    s = stringify(s)
    if max_chars is None or max_chars <= 0:
        return s
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def read_jsonl(path: str) -> Iterator[dict]:
    """Stream-read a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception as e:
                warn(f"JSON decode error at {path}:{ln}: {e}")
                continue
            if not isinstance(obj, dict):
                warn(f"Non-dict JSONL record at {path}:{ln}: type={type(obj)}")
                continue
            yield obj


def pc_norm_unit_sphere(xyz: np.ndarray) -> np.ndarray:
    """
    Normalize xyz to zero-mean and unit sphere (same spirit as PointLLM pc_norm).
    xyz: (N,3)
    """
    xyz = xyz.astype(np.float32, copy=False)
    centroid = xyz.mean(axis=0)
    xyz = xyz - centroid
    m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
    if m < 1e-12:
        m = 1e-12
    return xyz / m


def to_rgb(last3: np.ndarray) -> np.ndarray:
    """
    Convert last3 to RGB in [0,1].
    - If already in [0,1], keep.
    - If in [-1,1], map to [0,1] by (x+1)/2 (typical normals).
    - Else assume [0,255] and divide by 255.
    """
    x = np.asarray(last3, dtype=np.float32)
    x_min = float(np.min(x))
    x_max = float(np.max(x))

    if x_min >= 0.0 and x_max <= 1.0:
        rgb = x
    elif x_min >= -1.0 and x_max <= 1.0:
        rgb = (x + 1.0) * 0.5
    else:
        rgb = x / 255.0

    return np.clip(rgb, 0.0, 1.0)


def maybe_downsample(
    xyz: np.ndarray,
    feat3: Optional[np.ndarray],
    max_points: int,
    rng: random.Random,
):
    n = xyz.shape[0]
    if max_points is None or max_points <= 0 or max_points >= n:
        return xyz, feat3
    idx = np.arange(n)
    rng.shuffle(idx)
    idx = idx[:max_points]
    xyz2 = xyz[idx]
    feat2 = feat3[idx] if feat3 is not None else None
    return xyz2, feat2


def _robust_minmax_1d(x: np.ndarray, percentile: float) -> Tuple[float, float]:
    if percentile >= 100.0:
        return float(np.min(x)), float(np.max(x))
    lo = (100.0 - percentile) * 0.5
    hi = 100.0 - lo
    a, b = np.percentile(x, [lo, hi])
    return float(a), float(b)


def set_axes_equal(ax, xyz: np.ndarray) -> None:
    """
    Make 3D axis have equal scale, so objects are not distorted.

    Also supports robust scaling to reduce the impact of outliers (clarity).
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    if ROBUST_AXES:
        x_min, x_max = _robust_minmax_1d(x, ROBUST_PERCENTILE)
        y_min, y_max = _robust_minmax_1d(y, ROBUST_PERCENTILE)
        z_min, z_max = _robust_minmax_1d(z, ROBUST_PERCENTILE)
    else:
        x_min, x_max = float(x.min()), float(x.max())
        y_min, y_max = float(y.min()), float(y.max())
        z_min, z_max = float(z.min()), float(z.max())

    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    max_range = max(x_range, y_range, z_range, 1e-12)

    # padding
    max_range = max_range * (1.0 + float(AXIS_PAD_RATIO))

    x_mid = 0.5 * (x_max + x_min)
    y_mid = 0.5 * (y_max + y_min)
    z_mid = 0.5 * (z_max + z_min)

    ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
    ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
    ax.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)

    # matplotlib>=3.3: better aspect control
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass


def _sort_by_view(
    xyz: np.ndarray,
    feat3: Optional[np.ndarray],
    elev: float,
    azim: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Sort points from far->near for a given (elev, azim) so that near points
    are drawn last. This often improves visual clarity when DEPTHSHADE=False.
    """
    if not SORT_POINTS_BY_VIEW:
        return xyz, feat3

    er = np.deg2rad(float(elev))
    ar = np.deg2rad(float(azim))

    # Approx camera direction vector (origin -> camera)
    v = np.array(
        [np.cos(er) * np.cos(ar), np.cos(er) * np.sin(ar), np.sin(er)],
        dtype=np.float32,
    )
    depth = xyz @ v  # larger -> closer to camera (camera at +v)

    order = np.argsort(depth)  # far -> near
    xyz2 = xyz[order]
    feat2 = feat3[order] if feat3 is not None else None
    return xyz2, feat2


def render_multiview(
    xyz: np.ndarray,
    rgb: Optional[np.ndarray],
    out_path: Path,
    header: str = "",
    lines: Optional[List[str]] = None,
) -> None:
    """
    Render a multi-view panel into a single PNG.
    xyz: (N,3)
    rgb: (N,3) in [0,1] or None
    """
    n_views = len(VIEWS)
    fig_w = PANEL_FIGSIZE[0] * n_views
    fig_h = PANEL_FIGSIZE[1]
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI)

    # Leave some space for header + bottom text.
    top = 0.92 if header else 0.98
    bottom = 0.14 if lines else 0.06
    try:
        fig.subplots_adjust(left=0.01, right=0.99, top=top, bottom=bottom, wspace=0.02)
    except Exception:
        pass

    for i, (view_name, elev, azim) in enumerate(VIEWS, start=1):
        ax = fig.add_subplot(1, n_views, i, projection="3d")

        # projection type (ortho/persp)
        try:
            if PROJ_TYPE:
                ax.set_proj_type(PROJ_TYPE)
        except Exception:
            pass

        # camera distance / zoom (may be deprecated in newer matplotlib)
        if CAMERA_DIST is not None:
            try:
                ax.dist = float(CAMERA_DIST)
            except Exception:
                pass

        xyz_v, rgb_v = _sort_by_view(xyz, rgb, elev=elev, azim=azim)

        if rgb_v is None:
            # fallback: grayscale by z (after sorting)
            z = xyz_v[:, 2]
            z0, z1 = float(z.min()), float(z.max())
            if abs(z1 - z0) < 1e-12:
                c = np.full((xyz_v.shape[0], 3), 0.7, dtype=np.float32)
            else:
                t = (z - z0) / (z1 - z0)
                c = np.stack([t, t, t], axis=1).astype(np.float32)
        else:
            c = rgb_v

        ax.scatter(
            xyz_v[:, 0], xyz_v[:, 1], xyz_v[:, 2],
            s=POINT_SIZE,
            c=c,
            marker=MARKER,
            linewidths=0.0,
            alpha=ALPHA,
            depthshade=DEPTHSHADE,
        )

        ax.view_init(elev=float(elev), azim=float(azim))
        set_axes_equal(ax, xyz_v)

        if AXIS_OFF:
            ax.set_axis_off()

        ax.set_title(view_name, fontsize=11)

    if header:
        fig.suptitle(header, fontsize=12, y=0.995)

    if lines:
        wrapped = []
        for s in lines:
            s = clip_text(s, MAX_TEXT_CHARS_IN_FIG).strip()
            if not s:
                continue
            wrapped.append(textwrap.fill(s, width=TEXT_WRAP_WIDTH))
        if wrapped:
            fig.text(
                0.5, 0.01,
                "\n".join(wrapped),
                ha="center", va="bottom",
                fontsize=TEXT_FONT_SIZE,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)


# ======================================================================================
# Dataset 1 loading
# ======================================================================================

def load_dataset1_pointcloud(object_id: str, data_dir: str, pointnum: int) -> np.ndarray:
    p = Path(data_dir) / f"{object_id}_{pointnum}.npy"
    if not p.is_file():
        raise FileNotFoundError(str(p))
    arr = np.load(str(p))
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Unexpected point cloud shape: {arr.shape}, file={p}")
    return arr.astype(np.float32, copy=False)


def load_dataset1_jobs_from_jsonl(path: str) -> List[Dict[str, str]]:
    """
    Return list of jobs:
      {"object_id": str, "ground_truth": str, "pred": str}
    """
    jobs: List[Dict[str, str]] = []
    for rec in read_jsonl(path):
        oid = stringify(rec.get(DATASET1_JSONL_OBJECT_ID_KEY, "")).strip()
        if not oid:
            continue
        if DATASET1_SKIP_CORRUPTED_COLOR_IDS and oid in DATASET1_CORRUPTED_COLOR_IDS:
            continue

        gt = stringify(rec.get(DATASET1_JSONL_GT_KEY, ""))
        pred = stringify(rec.get(DATASET1_JSONL_PRED_KEY, ""))

        jobs.append({"object_id": oid, "ground_truth": gt, "pred": pred})
    return jobs


# ======================================================================================
# Dataset 2 loading
# ======================================================================================

def load_modelnet40_pickle(dat_path: str):
    with open(dat_path, "rb") as f:
        obj = pickle.load(f)

    if not (isinstance(obj, (list, tuple)) and len(obj) >= 2):
        raise ValueError(f"Unexpected pickle structure: type={type(obj)}, len={getattr(obj,'__len__',None)}")

    points_list, labels = obj[0], obj[1]
    pts_list = [np.asarray(p, dtype=np.float32) for p in points_list]

    def to_int_label(x):
        if isinstance(x, np.ndarray):
            return int(x.reshape(-1)[0])
        return int(x)

    lab_list = [to_int_label(x) for x in labels]
    if len(pts_list) != len(lab_list):
        raise ValueError(f"Length mismatch: points={len(pts_list)} labels={len(lab_list)}")
    return pts_list, lab_list


def load_dataset2_jobs_from_jsonl(path: str, n2: int) -> List[Dict[str, str]]:
    """
    Return list of jobs:
      {"ds_idx": int, "gt_label": str, "pred_text": str}
    """
    jobs: List[Dict[str, str]] = []
    for rec in read_jsonl(path):
        raw_idx = rec.get(DATASET2_JSONL_INDEX_KEY, None)
        try:
            idx = int(raw_idx)
        except Exception:
            continue

        if not (0 <= idx < n2):
            msg = f"ds_idx out of range: {idx} (valid: 0..{n2-1})"
            if STRICT_MISSING_FILES:
                raise IndexError(msg)
            warn(msg)
            continue

        gt_label = stringify(rec.get(DATASET2_JSONL_GT_LABEL_KEY, ""))
        pred_text = stringify(rec.get(DATASET2_JSONL_PRED_TEXT_KEY, ""))

        jobs.append({"ds_idx": idx, "gt_label": gt_label, "pred_text": pred_text})
    return jobs


# ======================================================================================
# Main
# ======================================================================================

def main() -> None:
    rng = _rng(RANDOM_SEED)

    out_root = Path(OUTPUT_DIR)
    out_d1 = out_root / "dataset1"
    out_d2 = out_root / "dataset2"
    ensure_dir(out_root)

    if not RENDER_DATASET1 and not RENDER_DATASET2:
        warn("Both RENDER_DATASET1 and RENDER_DATASET2 are False. Nothing to render.")
        return

    if RENDER_DATASET1:
        ensure_dir(out_d1)
    if RENDER_DATASET2:
        ensure_dir(out_d2)

    # ----------------------------
    # Dataset 1: JSONL-driven render
    # ----------------------------
    manifest1 = []
    if RENDER_DATASET1:
        if not DATASET1_JSONL_PATH:
            raise ValueError("DATASET1_JSONL_PATH is empty. Please set it to your dataset1 jsonl file path.")

        d1_jobs = load_dataset1_jobs_from_jsonl(DATASET1_JSONL_PATH)
        if not d1_jobs:
            warn(f"No valid dataset1 records found in {DATASET1_JSONL_PATH}")
        else:
            print(f"[Dataset1] jobs loaded: {len(d1_jobs)}")

        for i, job in enumerate(d1_jobs):
            oid = job["object_id"]
            gt = job.get("ground_truth", "")
            pred = job.get("pred", "")

            try:
                pc = load_dataset1_pointcloud(oid, DATASET1_DATA_PATH, POINTNUM)
            except FileNotFoundError as e:
                msg = f"Missing dataset1 npy for object_id={oid}: {e}"
                if STRICT_MISSING_FILES:
                    raise
                warn(msg)
                continue

            xyz = pc[:, :3]
            if DATASET1_NORMALIZE_XYZ:
                xyz = pc_norm_unit_sphere(xyz)

            rgb = None
            if DATASET1_USE_LAST3_AS_COLOR and pc.shape[1] >= 6:
                rgb = to_rgb(pc[:, 3:6])

            xyz, rgb = maybe_downsample(xyz, rgb, MAX_POINTS_TO_RENDER, rng)

            header = f"Dataset1 | object_id={oid}"
            lines: List[str] = []
            if gt.strip():
                lines.append(f"GT: {gt}")
            if pred.strip():
                lines.append(f"Pred: {pred}")

            out_path = out_d1 / f"{i:05d}_{oid}.png"
            render_multiview(xyz, rgb, out_path, header=header, lines=lines)

            manifest1.append({
                "file": str(out_path),
                "object_id": oid,
                "ground_truth": gt,
                "pred": pred,
            })

        with open(out_d1 / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest1, f, ensure_ascii=False, indent=2)
    else:
        print("[Dataset1] skipped (RENDER_DATASET1=False)")

    # ----------------------------
    # Dataset 2: JSONL-driven render
    # ----------------------------
    manifest2 = []
    if RENDER_DATASET2:
        pts_list, lab_list = load_modelnet40_pickle(DATASET2_DAT_PATH)
        n2 = len(lab_list)

        if not DATASET2_JSONL_PATH:
            raise ValueError("DATASET2_JSONL_PATH is empty. Please set it to your dataset2 jsonl file path.")

        d2_jobs = load_dataset2_jobs_from_jsonl(DATASET2_JSONL_PATH, n2=n2)
        if not d2_jobs:
            warn(f"No valid dataset2 records found in {DATASET2_JSONL_PATH}")
        else:
            print(f"[Dataset2] jobs loaded: {len(d2_jobs)} (pickle size={n2})")

        for rank, job in enumerate(d2_jobs):
            idx = int(job["ds_idx"])
            gt_label = job.get("gt_label", "")
            pred_text = job.get("pred_text", "")

            pc = pts_list[idx]
            if pc.ndim != 2 or pc.shape[1] < 3:
                msg = f"Unexpected dataset2 point cloud shape: {pc.shape}, idx={idx}"
                if STRICT_MISSING_FILES:
                    raise ValueError(msg)
                warn(msg)
                continue

            xyz = pc[:, :3]
            if DATASET2_NORMALIZE_XYZ:
                xyz = pc_norm_unit_sphere(xyz)

            rgb = None
            if DATASET2_COLOR_MODE == "last3":
                if pc.shape[1] >= 6:
                    rgb = to_rgb(pc[:, 3:6])
            elif DATASET2_COLOR_MODE == "constant":
                rgb = np.tile(np.asarray(DATASET2_CONSTANT_RGB, dtype=np.float32)[None, :], (pc.shape[0], 1))
            elif DATASET2_COLOR_MODE == "z":
                z = xyz[:, 2].astype(np.float32)
                z0, z1 = float(z.min()), float(z.max())
                if abs(z1 - z0) < 1e-12:
                    t = np.zeros_like(z)
                else:
                    t = (z - z0) / (z1 - z0)
                rgb = np.stack([t, t, t], axis=1)
            else:
                raise ValueError(f"Unknown DATASET2_COLOR_MODE={DATASET2_COLOR_MODE!r}")

            xyz, rgb = maybe_downsample(xyz, rgb, MAX_POINTS_TO_RENDER, rng)

            header = f"Dataset2 | ds_idx={idx}"
            lines2: List[str] = []
            if gt_label.strip():
                lines2.append(f"GT label: {gt_label}")
            if pred_text.strip():
                lines2.append(f"Pred: {pred_text}")

            if DATASET2_SHOW_PICKLE_LABEL:
                pickle_lab = int(lab_list[idx])
                lines2.append(f"(pickle label idx -> {pickle_lab})")

            out_path = out_d2 / f"{rank:05d}_idx{idx}.png"
            render_multiview(xyz, rgb, out_path, header=header, lines=lines2)

            manifest2.append({
                "file": str(out_path),
                "ds_idx": idx,
                "gt_label": gt_label,
                "pred_text": pred_text,
                "pickle_label": int(lab_list[idx]),
            })

        with open(out_d2 / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest2, f, ensure_ascii=False, indent=2)
    else:
        print("[Dataset2] skipped (RENDER_DATASET2=False)")

    print("=" * 80)
    print("Done.")
    if RENDER_DATASET1:
        print(f"Dataset1 renders: {out_d1} (n={len(manifest1)})")
    else:
        print("Dataset1 renders: skipped (RENDER_DATASET1=False)")
    if RENDER_DATASET2:
        print(f"Dataset2 renders: {out_d2} (n={len(manifest2)})")
    else:
        print("Dataset2 renders: skipped (RENDER_DATASET2=False)")
    print("Manifests saved as manifest.json in rendered folders.")
    print("=" * 80)


if __name__ == "__main__":
    main()
