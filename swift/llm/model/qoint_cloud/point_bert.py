# -*- coding: utf-8 -*-
"""
Self-contained PointBERT-style point cloud encoder (PointTransformer) extracted from PointLLM.

- Input:  (B, N, 6) where channels are (X, Y, Z, R, G, B)
- Output:
    * if use_max_pool=True (default):  (B, 1, 2*trans_dim)
    * if use_max_pool=False:          (B, G+1, trans_dim)  where G=num_group (includes CLS token)

This file includes all necessary dependencies (Group/FPS/KNN/Encoder/Transformer blocks)
and does NOT rely on any project-local utilities.

Dependencies:
  - torch

Tested with torch >= 1.13 (no external ops required).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Iterable, List, Type, Union

import torch
import torch.nn as nn


# -----------------------------
# Basic utilities
# -----------------------------

def _as_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    """
    Accepts several common checkpoint formats and returns a flat state_dict-like mapping.
    """
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "model_state_dict", "net", "network"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        # fall back: if dict looks like state_dict already
        if all(isinstance(k, str) for k in ckpt.keys()):
            # heuristic: state_dict values are tensors
            if all(torch.is_tensor(v) or isinstance(v, (int, float, list, tuple, dict)) for v in ckpt.values()):
                # keep only tensor items
                return {k: v for k, v in ckpt.items() if torch.is_tensor(v)}
    if isinstance(ckpt, (list, tuple)):
        raise TypeError("Unsupported checkpoint type: list/tuple")
    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")


def _strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not prefix:
        return state_dict
    if not all(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {k[len(prefix):]: v for k, v in state_dict.items()}


def _select_and_strip_prefixes(
    state_dict: Dict[str, torch.Tensor],
    prefixes_to_try: Iterable[str],
) -> Dict[str, torch.Tensor]:
    """
    Try to extract the encoder weights from a larger state_dict by stripping the first matching prefix.
    If no prefix matches all keys, we still attempt to strip for keys that start with the prefix.
    """
    # 1) Strong match: all keys start with prefix -> strip and return.
    for p in prefixes_to_try:
        if p and len(state_dict) > 0 and all(k.startswith(p) for k in state_dict.keys()):
            return {k[len(p):]: v for k, v in state_dict.items()}

    # 2) Weak match: select subset of keys with prefix and strip.
    for p in prefixes_to_try:
        if not p:
            continue
        subset = {k[len(p):]: v for k, v in state_dict.items() if k.startswith(p)}
        if subset:
            return subset

    # 3) No prefix found; return as-is.
    return state_dict


# -----------------------------
# FPS + neighborhood grouping
# -----------------------------

def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    points: (B, N, C)
    idx:    (B, S) or (B, S, K) or any shape starting with B
    return: (B, S, C) or (B, S, K, C)
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def fps(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Pure PyTorch farthest point sampling.

    xyz:   (B, N, 3)
    return:(B, npoint, 3) sampled points (not indices)
    """
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]

    return index_points(xyz, centroids)


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    src: (B, N, C)
    dst: (B, M, C)
    return: (B, N, M) squared distance
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist += torch.sum(src ** 2, dim=-1).view(B, N, 1)
    dist += torch.sum(dst ** 2, dim=-1).view(B, 1, M)
    return dist


def knn_point(nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    xyz:     (B, N, 3)    all points
    new_xyz: (B, S, 3)    query points
    return:  (B, S, nsample) indices of nearest points in xyz for each query
    """
    sqrdists = square_distance(new_xyz, xyz)  # (B, S, N)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class Group(nn.Module):
    """
    Divide a point cloud into local groups (patches) using FPS centers + kNN neighborhoods.

    Input:
        xyz: (B, N, C) where C>=3; if C>3, remaining channels are treated as extra features (e.g., RGB).
    Output:
        neighborhood: (B, G, M, C)  where M=group_size, G=num_group
            - xyz part is normalized to local coordinates by subtracting group center.
            - extra channels are concatenated without normalization.
        center: (B, G, 3)
    """
    def __init__(self, num_group: int, group_size: int):
        super().__init__()
        self.num_group = int(num_group)
        self.group_size = int(group_size)

    def forward(self, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if xyz.ndim != 3:
            raise ValueError(f"Expected xyz shape (B,N,C), got {tuple(xyz.shape)}")
        B, N, C = xyz.shape
        if C < 3:
            raise ValueError(f"Expected at least 3 channels (xyz), got C={C}")
        if self.num_group <= 0 or self.group_size <= 0:
            raise ValueError("num_group and group_size must be positive.")
        if self.group_size > N:
            raise ValueError(f"group_size={self.group_size} cannot exceed N={N}")

        if C > 3:
            xyz_only = xyz[:, :, :3]
            extra = xyz[:, :, 3:]
        else:
            xyz_only = xyz
            extra = None

        # FPS for group centers: (B, G, 3)
        center = fps(xyz_only, self.num_group)

        # kNN neighborhoods: (B, G, M)
        idx = knn_point(self.group_size, xyz_only, center)
        idx_base = torch.arange(0, B, device=xyz.device).view(-1, 1, 1) * N
        idx = (idx + idx_base).view(-1)  # (B*G*M,)

        # gather xyz: (B, G, M, 3)
        neighborhood_xyz = xyz_only.reshape(B * N, 3)[idx, :].view(B, self.num_group, self.group_size, 3).contiguous()

        # local normalization
        neighborhood_xyz = neighborhood_xyz - center.unsqueeze(2)

        if extra is not None:
            extra_dim = extra.shape[-1]
            neighborhood_extra = extra.reshape(B * N, extra_dim)[idx, :].view(B, self.num_group, self.group_size, extra_dim).contiguous()
            neighborhood = torch.cat([neighborhood_xyz, neighborhood_extra], dim=-1)
        else:
            neighborhood = neighborhood_xyz

        return neighborhood, center


# -----------------------------
# Local encoder (PointNet-like)
# -----------------------------

class Encoder(nn.Module):
    """
    Encode each local group (patch) into a single token vector.

    Input:
        point_groups: (B, G, M, point_input_dims)
    Output:
        feature_global: (B, G, encoder_channel)
    """
    def __init__(self, encoder_channel: int, point_input_dims: int = 3):
        super().__init__()
        self.encoder_channel = int(encoder_channel)
        self.point_input_dims = int(point_input_dims)

        self.first_conv = nn.Sequential(
            nn.Conv1d(self.point_input_dims, 128, kernel_size=1, bias=True),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=1, bias=True),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=1, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, kernel_size=1, bias=True),
        )

    def forward(self, point_groups: torch.Tensor) -> torch.Tensor:
        if point_groups.ndim != 4:
            raise ValueError(f"Expected point_groups shape (B,G,M,C), got {tuple(point_groups.shape)}")
        bs, g, n, c = point_groups.shape
        if c != self.point_input_dims:
            raise ValueError(f"point_input_dims mismatch: expected {self.point_input_dims}, got {c}")

        point_groups = point_groups.reshape(bs * g, n, c)  # (BG, M, C)

        # (BG, C, M) -> (BG, 256, M)
        feature = self.first_conv(point_groups.transpose(1, 2).contiguous())

        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # (BG, 256, 1)

        # concat global and local: (BG, 512, M)
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)

        # (BG, encoder_channel, M)
        feature = self.second_conv(feature)

        # max pool over points: (BG, encoder_channel)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]

        return feature_global.view(bs, g, self.encoder_channel)


# -----------------------------
# Transformer backbone (PointBERT-style)
# -----------------------------

def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Stochastic Depth per-sample.

    This is the same functional form as timm's drop_path, but implemented here to avoid extra deps.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor = random_tensor.floor()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        head_dim = dim // self.num_heads
        if dim % self.num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_prob: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder without hierarchical structure.
    """
    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 4,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: Union[float, List[float]] = 0.0,
    ):
        super().__init__()
        if isinstance(drop_path_rate, list):
            if len(drop_path_rate) != depth:
                raise ValueError("drop_path_rate list length must equal depth")
            dpr = drop_path_rate
        else:
            dpr = [float(drop_path_rate)] * depth

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path_prob=dpr[i],
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x + pos)
        return x


@dataclass
class PointBERTConfig:
    """
    Configuration for the PointBERT-style encoder.

    Defaults are aligned with common PointBERT/PointLLM settings, but you should
    match them to the pretrained checkpoint you plan to load.
    """
    # Transformer
    trans_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    drop_path_rate: float = 0.1
    mlp_ratio: float = 4.0

    # Grouping + local encoder
    num_group: int = 512
    group_size: int = 32
    point_dims: int = 6        # XYZRGB
    encoder_dims: int = 1024   # local encoder output dim

    # Optional (kept for parity, not used directly)
    cls_dim: int = 0


class PointBERTEncoder(nn.Module):
    """
    PointBERT-style point cloud encoder (PointTransformer).

    Forward:
        points: (B, N, 6) float tensor
    Returns:
        if use_max_pool=True:  (B, 1, 2*trans_dim)
        else:                  (B, G+1, trans_dim)
    """
    def __init__(self, config: PointBERTConfig, use_max_pool: bool = True):
        super().__init__()
        self.config = config
        self.use_max_pool = bool(use_max_pool)

        self.trans_dim = int(config.trans_dim)
        self.depth = int(config.depth)
        self.drop_path_rate = float(config.drop_path_rate)
        self.num_heads = int(config.num_heads)

        self.group_size = int(config.group_size)
        self.num_group = int(config.num_group)
        self.point_dims = int(config.point_dims)

        # Grouper
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        # Local encoder
        self.encoder_dims = int(config.encoder_dims)
        self.encoder = Encoder(encoder_channel=self.encoder_dims, point_input_dims=self.point_dims)

        # Bridge local encoder -> transformer dim
        self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim, bias=True)

        # CLS token + CLS pos
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        # Position embedding from group centers (xyz only)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128, bias=True),
            nn.GELU(),
            nn.Linear(128, self.trans_dim, bias=True),
        )

        # Stochastic depth schedule
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=float(config.mlp_ratio),
            drop_path_rate=dpr,
        )
        self.norm = nn.LayerNorm(self.trans_dim)

    @torch.no_grad()
    def load_checkpoint(
        self,
        ckpt_path: str,
        *,
        prefixes: Optional[Tuple[str, ...]] = ("module.point_encoder.", "point_encoder.", "module."),
        strict: bool = False,
        map_location: str = "cpu",
        weights_only: Optional[bool] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Load pretrained weights into this encoder.

        This tries to reuse PointLLM/PointBERT checkpoints which commonly store parameters under:
            - "state_dict" key
            - with prefix "module.point_encoder."

        Args:
            ckpt_path: path to .pth/.pt checkpoint.
            prefixes: prefixes to attempt for selecting/stripping keys.
            strict: passed to load_state_dict.
            map_location: torch.load map_location.
            verbose: print missing/unexpected keys.

        Returns:
            dict with keys: missing_keys, unexpected_keys, loaded_keys
        """
        # Prefer safe weight-only loading when supported (PyTorch >= 2.0),
        # but fall back to regular torch.load for older versions or custom checkpoints.
        if weights_only is None:
            try:
                ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=True)
            except TypeError:
                ckpt = torch.load(ckpt_path, map_location=map_location)
            except Exception:
                ckpt = torch.load(ckpt_path, map_location=map_location)
        else:
            try:
                ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=weights_only)
            except TypeError:
                ckpt = torch.load(ckpt_path, map_location=map_location)

        raw_state = _as_state_dict(ckpt)

        state = _select_and_strip_prefixes(raw_state, prefixes or ())

        incompatible = self.load_state_dict(state, strict=strict)

        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        loaded_keys = [k for k in state.keys() if k in self.state_dict().keys()]

        if verbose:
            if missing:
                print(f"[PointBERTEncoder] Missing keys ({len(missing)}):")
                for k in missing[:50]:
                    print("  -", k)
                if len(missing) > 50:
                    print(f"  ... ({len(missing)-50} more)")
            if unexpected:
                print(f"[PointBERTEncoder] Unexpected keys ({len(unexpected)}):")
                for k in unexpected[:50]:
                    print("  -", k)
                if len(unexpected) > 50:
                    print(f"  ... ({len(unexpected)-50} more)")
            if not missing and not unexpected:
                print(f"[PointBERTEncoder] Successfully loaded weights from: {ckpt_path}")

        return {
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "loaded_keys": loaded_keys,
        }

    def forward(self, points: torch.Tensor, *, return_tokens: Optional[bool] = None) -> torch.Tensor:
        """
        Args:
            points: (B, N, 6) tensor, channels = XYZRGB
            return_tokens:
                - True: always return token sequence (B, G+1, trans_dim)
                - False: always return pooled feature (B, 1, 2*trans_dim)
                - None: follow self.use_max_pool

        Returns:
            token sequence or pooled feature, depending on settings.
        """
        if points.ndim != 3:
            raise ValueError(f"Expected points shape (B,N,6), got {tuple(points.shape)}")
        if points.shape[-1] != self.point_dims:
            raise ValueError(f"Expected last dim {self.point_dims}, got {points.shape[-1]}")
        if points.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            points = points.float()
        # CPU kernels for small point-cloud tensors can be slower or unsupported in fp16/bf16.
        # Casting to fp32 improves portability for quick smoke tests on CPU.
        if points.device.type == "cpu" and points.dtype in (torch.float16, torch.bfloat16):
            points = points.float()

        # Divide the point cloud into groups (important for consistent input form)
        neighborhood, center = self.group_divider(points)           # (B,G,M,6), (B,G,3)

        # Encode each group -> tokens
        group_tokens = self.encoder(neighborhood)                   # (B,G,encoder_dims)
        group_tokens = self.reduce_dim(group_tokens)                # (B,G,trans_dim)

        # Prepare CLS token + CLS pos
        B = group_tokens.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)               # (B,1,trans_dim)
        cls_pos = self.cls_pos.expand(B, -1, -1)                    # (B,1,trans_dim)

        # Position embedding for group centers
        pos = self.pos_embed(center)                                # (B,G,trans_dim)

        # Final input to transformer
        x = torch.cat([cls_tokens, group_tokens], dim=1)            # (B,G+1,trans_dim)
        pos = torch.cat([cls_pos, pos], dim=1)                      # (B,G+1,trans_dim)

        # Transformer
        x = self.blocks(x, pos)
        x = self.norm(x)                                            # (B,G+1,trans_dim)

        want_tokens = return_tokens if return_tokens is not None else (not self.use_max_pool)
        if want_tokens:
            return x

        # Pooled representation (same as original): concat cls + max over patch tokens
        pooled = torch.cat([x[:, 0], x[:, 1:].max(dim=1)[0]], dim=-1).unsqueeze(1)  # (B,1,2*trans_dim)
        return pooled
