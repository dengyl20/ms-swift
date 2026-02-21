from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner = dim * mult
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, inner)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(inner, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, N, D)
        key_padding_mask: (B, N) bool, True=忽略（padding）
        """
        xn = self.norm(x)
        out, _ = self.attn(xn, xn, xn, key_padding_mask=key_padding_mask, need_weights=False)
        return self.drop(out)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        kv_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        q:  (B, Nq, D)
        kv: (B, Nk, D)
        kv_key_padding_mask: (B, Nk) bool, True=忽略（padding）
        """
        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)
        out, _ = self.attn(qn, kvn, kvn, key_padding_mask=kv_key_padding_mask, need_weights=False)
        return self.drop(out)


class PerceiverResampler(nn.Module):
    """
    可变长 tokens -> 固定长度 latent tokens
    Resampler blocks:
      - latents cross-attend inputs
      - FFN
      - latents self-attend
      - FFN
    """

    def __init__(
        self,
        dim: int,
        num_latents: int = 32,
        depth: int = 2,
        heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim) * 0.02)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        CrossAttention(dim=dim, heads=heads, dropout=dropout),
                        FeedForward(dim=dim, mult=ff_mult, dropout=dropout),
                        SelfAttention(dim=dim, heads=heads, dropout=dropout),
                        FeedForward(dim=dim, mult=ff_mult, dropout=dropout),
                    ]
                )
            )

    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, N, D)
        x_mask: (B, N) bool, True=有效 token
        """
        B = x.shape[0]
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)  # (B, N_latent, D)

        kv_key_padding_mask = None
        if x_mask is not None:
            kv_key_padding_mask = ~x_mask  # True=ignore

        for cross_attn, cross_ff, self_attn, self_ff in self.layers:
            latents = latents + cross_attn(latents, x, kv_key_padding_mask=kv_key_padding_mask)
            latents = latents + cross_ff(latents)
            latents = latents + self_attn(latents, key_padding_mask=None)
            latents = latents + self_ff(latents)

        return latents


class TransformerDecoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int = 8, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.self_attn = SelfAttention(dim=dim, heads=heads, dropout=dropout)
        self.cross_attn = CrossAttention(dim=dim, heads=heads, dropout=dropout)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout)

    def forward(
        self,
        q: torch.Tensor,
        q_key_padding_mask: Optional[torch.Tensor],
        latents: torch.Tensor,
    ) -> torch.Tensor:
        q = q + self.self_attn(q, key_padding_mask=q_key_padding_mask)
        q = q + self.cross_attn(q, latents, kv_key_padding_mask=None)
        q = q + self.ff(q)
        return q
