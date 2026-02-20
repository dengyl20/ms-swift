# -*- coding: utf-8 -*-
"""
TVL tactile preprocessing utilities (for TVL-LLaMA finetune style pipelines)

This module consolidates the tactile+vision preprocessing logic used in:
- Max-Fu/tvl  -> tvl_llama/data/dataset.py
- Max-Fu/tvl  -> tvl_enc/tvl.py (for modality keys: "vision"/"tactile")

Key behaviors (matching TVL-LLaMA):
- RGB transform: RandomResizedCrop + Normalize(tacvis.RGB_MEAN/STD)
- RGB augment (optional): H/V flip, ColorJitter, Gray, Blur + RRC + Normalize
- tactile process selection:
    subtract_background is None:
        augment_tactile ? tacvis.TAC_AUGMENTS : tacvis.TAC_WBG
    subtract_background == "background":
        augment_tactile ? tacvis.TAC_AUGMENTS_BG : tacvis.TAC_BG
- crop_tacvis (optional): use tacvis.load_vision_data() for SSVTP/HCT
- HCT background subtraction (optional): tacvis.tac_subtract_bg(bg_path, TAC_MEAN_BG, TAC_STD_BG)
- random_drop (optional): randomly drop at most one modality (vision or tactile)

Usage patterns:
- In a Dataset __getitem__, you can call:
    pre = TVLTouchPreprocessor(tvl_repo_path=..., crop_tacvis=..., subtract_background=..., ...)
    img, tac = pre.load_pair(image_path, tactile_path, tactile_background_path=..., dataset_hint=...)
    return {"image": img, "tactile": tac, ...}

Notes:
- This file expects the TVL repo to be importable as python package, e.g.:
    pip install -e /path/to/Max-Fu/tvl
  OR provide tvl_repo_path (or env TVL_REPO_PATH) to inject sys.path.

"""

from __future__ import annotations

import os
import sys
import random
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
from PIL import Image
from swift.tvl.stage1.src.preprocess import tacvis
try:
    import torchvision.transforms as T
except Exception as e:
    raise ImportError(
        "torchvision is required for TVLTouchPreprocessor. "
        "Please install torchvision that matches your torch build."
    ) from e

from rich.pretty import pprint
# -------------------------
# Import helpers (tvl_enc)
# -------------------------




# -------------------------
# Transforms (match TVL-LLaMA dataset.py)
# -------------------------

def _get_bicubic():
    # Match TVL-LLaMA behavior:
    # try torchvision.transforms.InterpolationMode.BICUBIC else PIL.Image.BICUBIC
    try:
        from torchvision.transforms import InterpolationMode
        return InterpolationMode.BICUBIC
    except Exception:
        return Image.BICUBIC


def _random_resized_crop(image_size: int = 224) -> T.RandomResizedCrop:
    """
    Create RandomResizedCrop with args matching TVL-LLaMA dataset.py:
      size=(224,224), scale=(0.9,1.0), ratio=(0.75,1.3333), interpolation=BICUBIC, antialias=None(if supported)
    """
    bicubic = _get_bicubic()
    kwargs = dict(
        size=(image_size, image_size),
        scale=(0.9, 1.0),
        ratio=(0.75, 1.3333),
        interpolation=bicubic,
    )

    # torchvision versions differ in "antialias" support
    try:
        sig = inspect.signature(T.RandomResizedCrop.__init__)
        if "antialias" in sig.parameters:
            kwargs["antialias"] = None
    except Exception:
        # be conservative: do not pass antialias if we can't confirm support
        pass

    return T.RandomResizedCrop(**kwargs)


def build_rgb_transform(
    tvl_repo_path: Optional[str] = None,
    augment_rgb: bool = False,
    image_size: int = 224,
) -> T.Compose:
    """
    RGB transform used by TVL-LLaMA dataset.py (train vs train_aug).
    """


    if not augment_rgb:
        return T.Compose([
            _random_resized_crop(image_size=image_size),
            T.ToTensor(),
            T.Normalize(mean=tacvis.RGB_MEAN, std=tacvis.RGB_STD),
        ])

    # transform_train_aug in TVL-LLaMA dataset.py
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomApply([
            T.ColorJitter(
                brightness=(0.9, 1.1),
                contrast=(0.9, 1.1),
                saturation=0.1,
            )
        ], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([T.GaussianBlur(9, sigma=(0.5, 1.0))], p=0.5),
        _random_resized_crop(image_size=image_size),
        T.ToTensor(),
        T.Normalize(mean=tacvis.RGB_MEAN, std=tacvis.RGB_STD),
    ])


def build_tactile_process(
    tvl_repo_path: Optional[str] = None,
    augment_tactile: bool = False,
    subtract_background: Optional[str] = None,
):
    """
    Pick tactile preprocessing transform handle (for tacvis.load_tactile_data(transform_tac=...)).

    Matches TVL-LLaMA dataset.py:
      if subtract_background is None:
          TAC_AUGMENTS if augment_tactile else TAC_WBG
      elif subtract_background == "background":
          TAC_AUGMENTS_BG if augment_tactile else TAC_BG

    Any other subtract_background is considered unsupported (explicitly),
    to avoid silently deviating from TVL-LLaMA behavior.
    """


    if subtract_background is None:
        return tacvis.TAC_AUGMENTS if augment_tactile else tacvis.TAC_WBG

    if str(subtract_background).lower() == "background":
        return tacvis.TAC_AUGMENTS_BG if augment_tactile else tacvis.TAC_BG

    raise ValueError(
        f"Unsupported subtract_background='{subtract_background}'. "
        "TVL-LLaMA finetune dataset.py uses only: None or 'background'."
    )


# -------------------------
# Preprocessor
# -------------------------

@dataclass
class TVLTouchPreprocessConfig:
    tvl_repo_path: Optional[str] = None

    # dataset behavior flags (match tvl_llama)
    crop_tacvis: bool = False
    subtract_background: Optional[str] = None  # None or "background"
    augment_rgb: bool = False
    augment_tactile: bool = False
    random_drop: bool = False

    image_size: int = 224


class TVLTouchPreprocessor:
    """
    A lightweight helper that mirrors tvl_llama/data/dataset.py preprocessing,
    but can be reused in your own Dataset/Collator.

    Outputs are torch.Tensor in shape:
      image:   (3, H, W) float
      tactile: (3, H, W) float
    """
    tacvis = tacvis
    def __init__(self, cfg: TVLTouchPreprocessConfig):
        self.cfg = cfg


        self.rgb_transform = build_rgb_transform(
            tvl_repo_path=cfg.tvl_repo_path,
            augment_rgb=cfg.augment_rgb,
            image_size=cfg.image_size,
        )
        self.tactile_process = build_tactile_process(
            tvl_repo_path=cfg.tvl_repo_path,
            augment_tactile=cfg.augment_tactile,
            subtract_background=cfg.subtract_background,
        )

    # ---------
    # Utilities
    # ---------

    def zeros_image(self) -> torch.Tensor:
        return torch.zeros(3, self.cfg.image_size, self.cfg.image_size)

    def zeros_tactile(self) -> torch.Tensor:
        return torch.zeros(3, self.cfg.image_size, self.cfg.image_size)

    @staticmethod
    def infer_dataset_hint(*paths: str) -> Optional[str]:
        """
        Infer dataset subset from path strings:
          - contains 'ssvtp' -> 'ssvtp'
          - contains 'hct'   -> 'hct'
        """
        joined = " ".join([p for p in paths if p])
        low = joined.lower()
        if "ssvtp" in low:
            return "ssvtp"
        if "hct" in low:
            return "hct"
        return None

    # -----------------
    # Core load methods
    # -----------------

    def load_image(
        self,
        image_path: str,
        dataset_hint: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Match tvl_llama behavior:
        - if crop_tacvis and dataset in {ssvtp,hct} => tacvis.load_vision_data(...)
        - else => PIL + rgb_transform
        """
        dataset_hint = (dataset_hint or "").lower().strip()

        if self.cfg.crop_tacvis and dataset_hint in ("ssvtp", "hct"):
            if dataset_hint == "ssvtp":
                return self.tacvis.load_vision_data(image_path)
            # hct uses dataset_version="v2" + randomize_crop=True
            return self.tacvis.load_vision_data(image_path, dataset_version="v2", randomize_crop=True)

        # default: plain PIL transform
        img = Image.open(image_path).convert("RGB")
        return self.rgb_transform(img)

    def load_tactile(
        self,
        tactile_path: str,
        dataset_hint: Optional[str] = None,
        tactile_background_path: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Match tvl_llama behavior:
        - ssvtp => tacvis.load_tactile_data(path, transform_tac=self.tactile_process)
        - hct:
            subtract_background is None => same as above
            subtract_background == "background" => use tacvis.tac_subtract_bg(bg_path, TAC_MEAN_BG, TAC_STD_BG)
        """
        dataset_hint = (dataset_hint or "").lower().strip()
        
        if dataset_hint == "ssvtp":
            return self.tacvis.load_tactile_data(tactile_path, transform_tac=self.tactile_process)

        if dataset_hint == "hct":
            if self.cfg.subtract_background is None:
                return self.tacvis.load_tactile_data(tactile_path, transform_tac=self.tactile_process)

            # subtract_background == "background"
            if tactile_background_path is None:
                raise ValueError("HCT tactile background subtraction requires tactile_background_path (got None).")

            transform_tac = self.tacvis.tac_subtract_bg(
                tactile_background_path,
                self.tacvis.TAC_MEAN_BG,
                self.tacvis.TAC_STD_BG,
            )
            return self.tacvis.load_tactile_data(tactile_path, transform_tac=transform_tac)

        # Unknown dataset: fallback to tacvis loader if possible, else raise.
        # (Keeping explicit to avoid silently producing wrong normalization.)
        return self.tacvis.load_tactile_data(tactile_path, transform_tac=self.tactile_process)

    def maybe_random_drop(
        self,
        image: torch.Tensor,
        tactile: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Match tvl_llama:
          drop = random.choice([0,1,2])
            0 => drop vision (image zeros)
            1 => drop tactile (tactile zeros)
            2 => keep both
        """
        if not self.cfg.random_drop:
            return image, tactile

        drop = random.choice([0, 1, 2])
        if drop == 0:
            image = self.zeros_image()
        elif drop == 1:
            tactile = self.zeros_tactile()
        return image, tactile

    def load_pair(
        self,
        image_path: str,
        tactile_path: str,
        tactile_background_path: Optional[str] = None,
        dataset_hint: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience: load image+tactile pair and apply random_drop.
        """
        if dataset_hint is None:
            dataset_hint = self.infer_dataset_hint(image_path, tactile_path, tactile_background_path or "")

        img = self.load_image(image_path, dataset_hint=dataset_hint)
        tac = self.load_tactile(
            tactile_path,
            dataset_hint=dataset_hint,
            tactile_background_path=tactile_background_path,
        )
        img, tac = self.maybe_random_drop(img, tac)
        return img, tac


# -------------------------
# Optional: build TVL ImageBind-like model (used by tvl_llama llama_adapter)
# -------------------------

def build_tvl_imagebind_model(
    tvl_repo_path: Optional[str] = None,
    tactile_model: str = "vit_tiny_patch16_224",
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    """
    Build the TVL model from tvl_enc/tvl.py (used inside tvl_llama llama_adapter.py),
    and optionally load a tactile encoder checkpoint (strict=False like tvl_llama).

    NOTE:
    - tvl_llama's LLaMA_adapter always initializes TVL with active_modalities=[vision, tactile].
    - You can still feed only tactile at forward time (TVL.forward checks keys).

    Returns:
        model: tvl_enc.tvl.TVL
    """

    try:
        from tvl_enc.tvl import TVL, ModalityType  # type: ignore
    except Exception as e:
        raise ImportError(
            "Failed to import `tvl_enc.tvl.TVL`. "
            "Make sure Max-Fu/tvl is installed or TVL_REPO_PATH is set."
        ) from e

    model = TVL(
        active_modalities=[ModalityType.VISION, ModalityType.TACTILE],
        tactile_model=tactile_model,
    )

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        # tvl_llama uses ckpt['model'] and strict=False
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        miss_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"[build_tvl_imagebind_model] Missing keys: {miss_keys}")
        print(f"[build_tvl_imagebind_model] Unexpected keys: {unexpected_keys}")

    if device is not None:
        model = model.to(device)

    return model

