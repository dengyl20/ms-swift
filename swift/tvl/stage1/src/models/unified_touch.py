from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .layers import PerceiverResampler, TransformerDecoderLayer


# =========================================================
# TextEncoder（完全不变）
# =========================================================
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
        x: (B, L_pad, d_text_in)  已 pad 到 max_text_len
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


# =========================================================
# TouchEncoder（PointEncoder -> TouchEncoder，仅改名/注释）
# =========================================================
class TouchEncoder(nn.Module):
    def __init__(
        self,
        d_touch_in: int,
        d_model: int,
        touch_tokens: int,
        num_latents: int,
        resampler_depth: int,
        heads: int,
        ff_mult: int,
        dropout: float,
        use_touch_pos_emb: bool = True,
    ):
        super().__init__()
        self.touch_tokens = touch_tokens
        self.use_touch_pos_emb = use_touch_pos_emb

        self.in_norm = nn.LayerNorm(d_touch_in)
        self.proj_in = nn.Linear(d_touch_in, d_model)
        self.drop = nn.Dropout(dropout)

        if use_touch_pos_emb:
            self.pos_emb = nn.Parameter(torch.randn(touch_tokens, d_model) * 0.02)
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
        x: (B, G, d_touch_in)
           例如你的 touch_tokens=197, d_touch_in=768 -> (B, 197, 768)
        """
        B, N, _ = x.shape
        if N != self.touch_tokens:
            raise ValueError(f"Expected touch token length {self.touch_tokens}, got {N}")

        x = self.in_norm(x)
        x = self.proj_in(x)
        if self.use_touch_pos_emb:
            x = x + self.pos_emb.unsqueeze(0)
        x = self.drop(x)

        latents = self.resampler(x, x_mask=None)
        return latents


# =========================================================
# SharedTextDecoder（完全不变）
# =========================================================
class SharedTextDecoder(nn.Module):
    """
    非自回归 decoder：latent -> (max_text_len, d_text_out)
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
        out = self.out_proj(q)  # (B, max_len, d_text_out)
        return out


# =========================================================
# TouchDecoder（PointDecoder -> TouchDecoder，仅改名/注释）
# =========================================================
class TouchDecoder(nn.Module):
    """
    非自回归 decoder：latent -> (touch_tokens, d_touch_out)
    与 SharedTextDecoder 同理念：固定 query + cross-attn read latents
    """

    def __init__(
        self,
        d_touch_out: int,
        d_model: int,
        touch_tokens: int,
        depth: int,
        heads: int,
        ff_mult: int,
        dropout: float,
    ):
        super().__init__()
        self.touch_tokens = touch_tokens

        self.query_pos_emb = nn.Parameter(torch.randn(touch_tokens, d_model) * 0.02)
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, heads=heads, ff_mult=ff_mult, dropout=dropout) for _ in range(depth)]
        )

        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_touch_out)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        latents: (B, N_latent, d_model)
        """
        B = latents.shape[0]
        q = self.query_pos_emb.unsqueeze(0).expand(B, -1, -1)  # (B, touch_tokens, d_model)
        q = self.drop(q)

        for layer in self.layers:
            q = layer(q, q_key_padding_mask=None, latents=latents)

        q = self.out_norm(q)
        out = self.out_proj(q)  # (B, touch_tokens, d_touch_out)
        return out


# =========================================================
# UnifiedTouchTextAE（Point->Touch版，Text完全不变）
# - 兼容 point_feat / touch_feat 两种输入
# - 输出同时提供 touch_* & point_* 两套 key，便于完全复用旧 loss/训练代码
# =========================================================
class UnifiedTouchTextAE(nn.Module):
    """
    - Text AE:       text  -> text_latents  -> shared_text_decoder -> text_recon
    - Touch->Text:   touch -> touch_latents -> shared_text_decoder -> text_recon_from_touch
    - Touch AE:      touch -> touch_latents -> touch_decoder       -> touch_recon

    shared_text_decoder 在 Text AE 与 Touch->Text 两条路径中复用（与原版一致）。
    """

    def __init__(self, cfg_model: Dict):
        super().__init__()

        # ---------- Text 部分：完全不变 ----------
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

        self.shared_text_decoder = SharedTextDecoder(
            d_text_out=cfg_model["d_text_in"],  # 重构回 text feature dim，比如 2048
            d_model=cfg_model["d_model"],
            max_text_len=cfg_model["max_text_len"],
            depth=cfg_model["decoder_depth"],
            heads=cfg_model["heads"],
            ff_mult=cfg_model["ff_mult"],
            dropout=cfg_model["dropout"],
        )

        # ---------- Touch 部分：把 point 替换为 touch ----------
        # 为了最省改动：允许你 config 里仍用 d_point_in / point_tokens
        d_touch_in = int(cfg_model.get("d_touch_in", cfg_model.get("d_point_in")))
        touch_tokens = int(cfg_model.get("touch_tokens", cfg_model.get("point_tokens")))
        use_touch_pos_emb = bool(cfg_model.get("use_touch_pos_emb", cfg_model.get("use_point_pos_emb", True)))

        self.touch_encoder = TouchEncoder(
            d_touch_in=d_touch_in,
            d_model=cfg_model["d_model"],
            touch_tokens=touch_tokens,
            num_latents=cfg_model["num_latents"],
            resampler_depth=cfg_model["resampler_depth"],
            heads=cfg_model["heads"],
            ff_mult=cfg_model["ff_mult"],
            dropout=cfg_model["dropout"],
            use_touch_pos_emb=use_touch_pos_emb,
        )

        touch_decoder_depth = int(cfg_model.get("touch_decoder_depth", cfg_model.get("point_decoder_depth", cfg_model["decoder_depth"])))
        self.touch_decoder = TouchDecoder(
            d_touch_out=d_touch_in,  # 重构回 touch token 的 feature 维度（例如 768）
            d_model=cfg_model["d_model"],
            touch_tokens=touch_tokens,
            depth=touch_decoder_depth,
            heads=cfg_model["heads"],
            ff_mult=cfg_model["ff_mult"],
            dropout=cfg_model["dropout"],
        )

    @staticmethod
    def pool_latents(latents: torch.Tensor) -> torch.Tensor:
        return latents.mean(dim=1)  # (B, d_model)

    def forward(
        self,
        # 兼容旧训练代码：你可以继续传 point_feat=touch_tokens
        point_feat: Optional[torch.Tensor] = None,
        # 新训练代码建议用 touch_feat
        touch_feat: Optional[torch.Tensor] = None,
        text_feat: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ):
        """
        touch_feat: (B, touch_tokens, d_touch_in) e.g. (B, 197, 768)
        text_feat:  (B, max_text_len, d_text_in) e.g. (B, 24, 2048)
        text_mask:  (B, max_text_len) bool
        """
        if touch_feat is None:
            if point_feat is None:
                raise ValueError("Either touch_feat or point_feat must be provided.")
            touch_feat = point_feat
        if text_feat is None:
            raise ValueError("text_feat must be provided.")

        text_latents = self.text_encoder(text_feat, x_mask=text_mask)
        touch_latents = self.touch_encoder(touch_feat)

        text_recon = self.shared_text_decoder(text_latents, target_mask=text_mask)
        text_recon_from_touch = self.shared_text_decoder(touch_latents, target_mask=text_mask)

        touch_recon = self.touch_decoder(touch_latents)

        # 同时返回 touch_* 与 point_*（完全复用旧 loss/训练代码会更爽）
        return {
            "text_latents": text_latents,

            # 新命名（推荐）
            "touch_latents": touch_latents,
            "text_recon": text_recon,
            "text_recon_from_touch": text_recon_from_touch,
            "touch_recon": touch_recon,

            # 旧命名（兼容你原来的 point-text loss / train 代码）
            "point_latents": touch_latents,
            "text_recon_from_point": text_recon_from_touch,
            "point_recon": touch_recon,
        }
