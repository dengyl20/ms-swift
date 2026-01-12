from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .layers import PerceiverResampler, TransformerDecoderLayer


class TextEncoder(nn.Module):
    def __init__(
        self,
        d_text_in: int,
        d_model: int,
        max_text_len: int,
        num_latents: int,
        resampler_depth: int,
        heads: int,
        ff_mult: int,
        dropout: float,
        use_text_pos_emb: bool = True,
    ):
        super().__init__()
        self.max_text_len = max_text_len
        self.use_text_pos_emb = use_text_pos_emb

        self.in_norm = nn.LayerNorm(d_text_in)
        self.proj_in = nn.Linear(d_text_in, d_model)
        self.drop = nn.Dropout(dropout)

        if use_text_pos_emb:
            self.pos_emb = nn.Parameter(torch.randn(max_text_len, d_model) * 0.02)
        else:
            self.register_parameter("pos_emb", None)

        self.resampler = PerceiverResampler(
            dim=d_model,
            num_latents=num_latents,
            depth=resampler_depth,
            heads=heads,
            ff_mult=ff_mult,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        x: (B, L_pad, 2048)  已 pad 到 max_text_len
        x_mask: (B, L_pad) bool
        """
        B, L, _ = x.shape
        if L > self.max_text_len:
            x = x[:, : self.max_text_len]
            x_mask = x_mask[:, : self.max_text_len] if x_mask is not None else None
            L = self.max_text_len

        x = self.in_norm(x)
        x = self.proj_in(x)
        if self.use_text_pos_emb:
            x = x + self.pos_emb[:L].unsqueeze(0)
        x = self.drop(x)

        latents = self.resampler(x, x_mask=x_mask)
        return latents


class PointEncoder(nn.Module):
    def __init__(
        self,
        d_point_in: int,
        d_model: int,
        point_tokens: int,
        num_latents: int,
        resampler_depth: int,
        heads: int,
        ff_mult: int,
        dropout: float,
        use_point_pos_emb: bool = True,
    ):
        super().__init__()
        self.point_tokens = point_tokens
        self.use_point_pos_emb = use_point_pos_emb

        self.in_norm = nn.LayerNorm(d_point_in)
        self.proj_in = nn.Linear(d_point_in, d_model)
        self.drop = nn.Dropout(dropout)

        if use_point_pos_emb:
            self.pos_emb = nn.Parameter(torch.randn(point_tokens, d_model) * 0.02)
        else:
            self.register_parameter("pos_emb", None)

        self.resampler = PerceiverResampler(
            dim=d_model,
            num_latents=num_latents,
            depth=resampler_depth,
            heads=heads,
            ff_mult=ff_mult,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 512, 256)  点云 encoder 的输出
        """
        B, N, _ = x.shape
        if N != self.point_tokens:
            raise ValueError(f"Expected point token length {self.point_tokens}, got {N}")

        x = self.in_norm(x)
        x = self.proj_in(x)
        if self.use_point_pos_emb:
            x = x + self.pos_emb.unsqueeze(0)
        x = self.drop(x)

        latents = self.resampler(x, x_mask=None)
        return latents


class SharedTextDecoder(nn.Module):
    """
    非自回归 decoder：latent -> (max_text_len, 2048)
    """
    def __init__(
        self,
        d_text_out: int,
        d_model: int,
        max_text_len: int,
        depth: int,
        heads: int,
        ff_mult: int,
        dropout: float,
    ):
        super().__init__()
        self.max_text_len = max_text_len
        self.query_pos_emb = nn.Parameter(torch.randn(max_text_len, d_model) * 0.02)
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, heads=heads, ff_mult=ff_mult, dropout=dropout) for _ in range(depth)]
        )

        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_text_out)

    def forward(self, latents: torch.Tensor, target_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        latents: (B, N_latent, d_model)
        target_mask: (B, max_text_len) bool
        """
        B = latents.shape[0]
        q = self.query_pos_emb.unsqueeze(0).expand(B, -1, -1)  # (B, max_len, d_model)
        q = self.drop(q)

        q_key_padding_mask = None
        if target_mask is not None:
            q_key_padding_mask = ~target_mask  # True=ignore

        for layer in self.layers:
            q = layer(q, q_key_padding_mask=q_key_padding_mask, latents=latents)

        q = self.out_norm(q)
        out = self.out_proj(q)  # (B, max_len, 2048)
        return out


class UnifiedPointTextAE(nn.Module):
    """
    - Text AE: text -> text_latents -> shared_text_decoder -> text_recon
    - Point->Text: point -> point_latents -> shared_text_decoder -> text_recon_from_point
    """

    def __init__(self, cfg_model: Dict):
        super().__init__()
        self.text_encoder = TextEncoder(
            d_text_in=cfg_model["d_text_in"],
            d_model=cfg_model["d_model"],
            max_text_len=cfg_model["max_text_len"],
            num_latents=cfg_model["num_latents"],
            resampler_depth=cfg_model["resampler_depth"],
            heads=cfg_model["heads"],
            ff_mult=cfg_model["ff_mult"],
            dropout=cfg_model["dropout"],
            use_text_pos_emb=cfg_model.get("use_text_pos_emb", True),
        )
        self.point_encoder = PointEncoder(
            d_point_in=cfg_model["d_point_in"],
            d_model=cfg_model["d_model"],
            point_tokens=cfg_model["point_tokens"],
            num_latents=cfg_model["num_latents"],
            resampler_depth=cfg_model["resampler_depth"],
            heads=cfg_model["heads"],
            ff_mult=cfg_model["ff_mult"],
            dropout=cfg_model["dropout"],
            use_point_pos_emb=cfg_model.get("use_point_pos_emb", True),
        )
        self.shared_text_decoder = SharedTextDecoder(
            d_text_out=cfg_model["d_text_in"],   # 重构回 2048
            d_model=cfg_model["d_model"],
            max_text_len=cfg_model["max_text_len"],
            depth=cfg_model["decoder_depth"],
            heads=cfg_model["heads"],
            ff_mult=cfg_model["ff_mult"],
            dropout=cfg_model["dropout"],
        )

    @staticmethod
    def pool_latents(latents: torch.Tensor) -> torch.Tensor:
        return latents.mean(dim=1)  # (B, d_model)

    def forward(self, point_feat: torch.Tensor, text_feat: torch.Tensor, text_mask: Optional[torch.Tensor]):
        text_latents = self.text_encoder(text_feat, x_mask=text_mask)
        point_latents = self.point_encoder(point_feat)

        text_recon = self.shared_text_decoder(text_latents, target_mask=text_mask)
        text_recon_from_point = self.shared_text_decoder(point_latents, target_mask=text_mask)

        return {
            "text_latents": text_latents,
            "point_latents": point_latents,
            "text_recon": text_recon,
            "text_recon_from_point": text_recon_from_point,
        }
