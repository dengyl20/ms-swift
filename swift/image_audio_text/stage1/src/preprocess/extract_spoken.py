from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader
from transformers import (
    Qwen3OmniMoeProcessor,
    Qwen3OmniMoeThinkerForConditionalGeneration,
)

from swift.image_audio_text.stage1.src.data.spokencoco import (
    SpokenCoCoTripletDataset,
    collate_triplets,
)


# =============================
# basic utils
# =============================
def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return {} if cfg is None else cfg


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _parse_np_dtype_save(s: str) -> np.dtype:
    s = (s or "fp16").lower()
    if s in ("fp16", "float16", "half"):
        return np.float16
    if s in ("fp32", "float32", "float"):
        return np.float32
    raise ValueError(f"Unsupported save_dtype: {s} (fp16/fp32)")


def _parse_torch_dtype(s: str) -> torch.dtype:
    s = (s or "fp16").lower()
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp32", "float32", "float"):
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {s} (fp16/bf16/fp32)")


# =============================
# audio loading (expects (C, T))
# =============================
def _linear_resample_np(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Fallback resampler: linear interpolation (ok as last resort). x: (T,) float32."""
    if orig_sr == target_sr:
        return x
    if x.size == 0:
        return x
    new_len = int(round(x.shape[0] * float(target_sr) / float(orig_sr)))
    new_len = max(new_len, 1)
    xp = np.linspace(0.0, 1.0, num=x.shape[0], dtype=np.float32)
    fp = x.astype(np.float32, copy=False)
    xq = np.linspace(0.0, 1.0, num=new_len, dtype=np.float32)
    y = np.interp(xq, xp, fp).astype(np.float32, copy=False)
    return y


def load_audio_as_float32_ct(
    wav_path: str,
    target_sr: int,
) -> np.ndarray:
    """
    返回 np.ndarray, float32, shape=(C,T).
    - 读文件
    - 转 mono
    - 重采样到 target_sr
    """
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(wav_path)

    # 1) read
    try:
        import soundfile as sf  # preferred

        audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    except Exception:
        # fallback to librosa
        import librosa

        audio, sr = librosa.load(wav_path, sr=None, mono=False)
        audio = audio.astype(np.float32, copy=False)

    # audio shape:
    # - mono: (T,)
    # - multi-channel: (T,C) (soundfile) or (C,T) (librosa with mono=False)
    if audio.ndim == 1:
        mono = audio
    elif audio.ndim == 2:
        # detect layout
        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            # likely (C, T)
            mono = audio.mean(axis=0, dtype=np.float32)
        else:
            # likely (T, C)
            mono = audio.mean(axis=1, dtype=np.float32)
    else:
        raise ValueError(f"bad audio ndim: {audio.ndim}, path={wav_path}")

    mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    # 2) resample
    if int(sr) != int(target_sr):
        # try torchaudio first
        done = False
        try:
            import torchaudio

            x = torch.from_numpy(mono).unsqueeze(0)  # (1,T)
            y = torchaudio.functional.resample(x, orig_freq=int(sr), new_freq=int(target_sr))
            mono = y.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
            done = True
        except Exception:
            done = False

        if not done:
            try:
                import librosa

                mono = librosa.resample(mono, orig_sr=int(sr), target_sr=int(target_sr)).astype(np.float32, copy=False)
                done = True
            except Exception:
                done = False

        if not done:
            mono = _linear_resample_np(mono, orig_sr=int(sr), target_sr=int(target_sr))

    # (C,T)
    mono_ct = mono[None, :]  # (1,T)
    return mono_ct


# =============================
# padding / trunc helper
# =============================
def pad_or_trunc_tokens_2d(
    x: torch.Tensor, target_tokens: int
) -> torch.Tensor:
    """
    x: (N, D) -> (target_tokens, D)
    """
    n, d = x.shape
    if n == target_tokens:
        return x
    if n > target_tokens:
        return x[:target_tokens]
    pad = torch.zeros((target_tokens - n, d), dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=0)


# =============================
# Memmap Writer
# =============================
@dataclass
class MemmapWriter:
    out_dir: str
    num_samples: int
    text_tokens: int
    d_text: int
    image_tokens: int
    d_image: int
    audio_tokens: int
    d_audio: int
    dtype: np.dtype

    def __post_init__(self):
        ensure_dir(self.out_dir)

        self.paths = {
            "text_embeds": os.path.join(self.out_dir, "shard_00000.text_embeds.mmap"),
            "text_mask": os.path.join(self.out_dir, "shard_00000.text_mask.mmap"),
            "image_tokens": os.path.join(self.out_dir, "shard_00000.image_tokens.mmap"),
            "audio_tokens": os.path.join(self.out_dir, "shard_00000.audio_tokens.mmap"),
            "image_rel": os.path.join(self.out_dir, "shard_00000.image_rel.mmap"),
            "wav_rel": os.path.join(self.out_dir, "shard_00000.wav_rel.mmap"),
            "global_indices": os.path.join(self.out_dir, "shard_00000.global_indices.mmap"),
            "valid": os.path.join(self.out_dir, "shard_00000.valid.mmap"),
        }

        n = int(self.num_samples)
        self.text_embeds = np.memmap(self.paths["text_embeds"], mode="w+", dtype=self.dtype, shape=(n, self.text_tokens, self.d_text))
        self.text_mask = np.memmap(self.paths["text_mask"], mode="w+", dtype=np.uint8, shape=(n, self.text_tokens))
        self.image_tokens_mm = np.memmap(self.paths["image_tokens"], mode="w+", dtype=self.dtype, shape=(n, self.image_tokens, self.d_image))
        self.audio_tokens_mm = np.memmap(self.paths["audio_tokens"], mode="w+", dtype=self.dtype, shape=(n, self.audio_tokens, self.d_audio))
        self.image_rel = np.memmap(self.paths["image_rel"], mode="w+", dtype="S256", shape=(n,))
        self.wav_rel = np.memmap(self.paths["wav_rel"], mode="w+", dtype="S256", shape=(n,))
        self.global_indices = np.memmap(self.paths["global_indices"], mode="w+", dtype=np.int64, shape=(n,))
        self.valid = np.memmap(self.paths["valid"], mode="w+", dtype=np.uint8, shape=(n,))

        self._ptr = 0

    def write_batch(
        self,
        text_embeds: np.ndarray,      # (B, L, Dt)
        text_mask: np.ndarray,        # (B, L) uint8
        image_tokens: np.ndarray,     # (B, Ti, Di)
        audio_tokens: np.ndarray,     # (B, Ta, Da)
        image_rel: List[str],
        wav_rel: List[str],
        valid: np.ndarray,            # (B,) uint8
    ) -> None:
        b = int(text_embeds.shape[0])
        s = self._ptr
        e = s + b
        if e > self.num_samples:
            raise RuntimeError(f"memmap overflow: ptr={s}, batch={b}, total={self.num_samples}")

        self.text_embeds[s:e] = text_embeds
        self.text_mask[s:e] = text_mask
        self.image_tokens_mm[s:e] = image_tokens
        self.audio_tokens_mm[s:e] = audio_tokens
        self.valid[s:e] = valid
        self.global_indices[s:e] = np.arange(s, e, dtype=np.int64)

        for i in range(b):
            self.image_rel[s + i] = np.bytes_(image_rel[i].encode("utf-8")[:256])
            self.wav_rel[s + i] = np.bytes_(wav_rel[i].encode("utf-8")[:256])

        self._ptr = e

    def flush(self):
        for mm in [
            self.text_embeds,
            self.text_mask,
            self.image_tokens_mm,
            self.audio_tokens_mm,
            self.image_rel,
            self.wav_rel,
            self.global_indices,
            self.valid,
        ]:
            mm.flush()


# =============================
# feature extraction core (batch)
# =============================
@torch.inference_mode()
def extract_batch_features(
    *,
    processor: Qwen3OmniMoeProcessor,
    model: Qwen3OmniMoeThinkerForConditionalGeneration,
    texts: List[str],
    image_paths: List[str],
    audio_paths: List[str],
    max_text_len: int,
    add_special_tokens: bool,
    image_tokens_target: int,
    audio_tokens_target: int,
    device: torch.device,
    save_dtype: np.dtype,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    返回 CPU numpy:
      text_embeds:  (B, L, Dt) save_dtype
      text_mask:    (B, L) uint8
      image_tokens: (B, Ti, Di) save_dtype
      audio_tokens: (B, Ta, Da) save_dtype
      valid:        (B,) uint8
    """
    bsz = len(texts)
    assert len(image_paths) == bsz and len(audio_paths) == bsz

    # ---- detect text overflow (same logic as your point_text filtering) ----
    # detect_len = max_len + 1, to know > max_len
    detect_len = int(max_text_len) + 1
    tok_detect = processor.tokenizer(
        [str(t) for t in texts],
        padding=False,
        truncation=True,
        max_length=detect_len,
        add_special_tokens=bool(add_special_tokens),
        return_attention_mask=False,
        return_token_type_ids=False,
        return_length=True,
    )
    lens = tok_detect.get("length", None)
    if lens is None:
        lens = [len(x) for x in tok_detect["input_ids"]]
    text_ok = np.array([int(l) <= int(max_text_len) for l in lens], dtype=np.uint8)

    # ---- file ok ----
    file_ok = np.array(
        [(os.path.isfile(ip) and os.path.isfile(ap)) for ip, ap in zip(image_paths, audio_paths)],
        dtype=np.uint8,
    )

    valid = (text_ok & file_ok).astype(np.uint8)

    # ---- allocate outputs (zeros) ----
    d_text = int(model.get_input_embeddings().embedding_dim)
    # image/audio hidden dim (most likely == d_text, but don't assume blindly)
    # we'll infer from a tiny forward below for valid subset; if no valid, keep d_text
    d_image = d_text
    d_audio = d_text

    text_embeds_np = np.zeros((bsz, max_text_len, d_text), dtype=save_dtype)
    text_mask_np = np.zeros((bsz, max_text_len), dtype=np.uint8)
    image_tokens_np = np.zeros((bsz, image_tokens_target, d_image), dtype=save_dtype)
    audio_tokens_np = np.zeros((bsz, audio_tokens_target, d_audio), dtype=save_dtype)

    # no valid samples in this batch -> return all zeros quickly
    valid_idx = np.nonzero(valid)[0].tolist()
    if len(valid_idx) == 0:
        return text_embeds_np, text_mask_np, image_tokens_np, audio_tokens_np, valid

    # ---- tokenize text (valid subset), truncation=False (to avoid silent truncation) ----
    tok = processor.tokenizer(
        [str(texts[i]) for i in valid_idx],
        padding="max_length",
        truncation=False,
        max_length=int(max_text_len),
        add_special_tokens=bool(add_special_tokens),
        return_tensors="pt",
    )
    input_ids = tok["input_ids"].to(device, non_blocking=True)
    attn_mask = tok["attention_mask"].to(device, non_blocking=True).bool()

    # ---- text embeddings from embedding table ----
    text_emb = model.get_input_embeddings()(input_ids)  # (Bv, L, Dt)

    # ---- load images + audios (valid subset) ----
    # 注意：单个坏文件不应该搞崩整个 batch，所以这里逐个 try
    images: List[Image.Image] = []
    audios: List[np.ndarray] = []
    kept_map: List[int] = []  # map from local valid-subset index -> original batch index

    target_sr = int(getattr(processor.feature_extractor, "sampling_rate", 16000))

    for j, bi in enumerate(valid_idx):
        try:
            img = Image.open(image_paths[bi]).convert("RGB")
            # audio: (C,T)
            wav_ct = load_audio_as_float32_ct(audio_paths[bi], target_sr=target_sr)
            images.append(img)
            audios.append(wav_ct)
            kept_map.append(bi)
        except Exception:
            # this sample becomes invalid
            valid[bi] = 0

    if len(kept_map) == 0:
        # all valid_idx samples failed loading
        return text_embeds_np, text_mask_np, image_tokens_np, audio_tokens_np, valid

    # ---- shrink text_emb / attn_mask to match kept_map ----
    # Because some "valid" samples might fail loading image/audio above.
    # kept_map corresponds to subset positions [0..Bk-1] inside "images/audios"
    # We need matching rows from text_emb/attn_mask too.
    # But text_emb was built over original valid_idx order.
    # Build mapping: original bi -> position in valid_idx -> row in text_emb
    pos_in_valid_idx = {bi: k for k, bi in enumerate(valid_idx)}
    rows = [pos_in_valid_idx[bi] for bi in kept_map]
    rows_t = torch.tensor(rows, device=device, dtype=torch.long)
    text_emb = text_emb.index_select(0, rows_t)
    attn_mask = attn_mask.index_select(0, rows_t)

    # ---- multimodal processor for image/audio ----
    mm = processor(
        text=None,
        images=images,
        audio=audios,
        return_tensors="pt",
        padding=True,
    )
    # move to device
    mm = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in mm.items()}

    model_dtype = next(model.parameters()).dtype
    # cast float tensors to model dtype for performance
    if "pixel_values" in mm and torch.is_tensor(mm["pixel_values"]):
        mm["pixel_values"] = mm["pixel_values"].to(dtype=model_dtype)
    if "input_features" in mm and torch.is_tensor(mm["input_features"]):
        mm["input_features"] = mm["input_features"].to(dtype=model_dtype)

    # ---- image features ----
    img_out = model.get_image_features(
        pixel_values=mm.get("pixel_values", None),
        image_grid_thw=mm.get("image_grid_thw", None),
        return_dict=True,
    )
    img_h = img_out.last_hidden_state
    # docs say (B, H, W, D) for image features; be robust
    if img_h.ndim == 4:
        b_img, h, w, d_img = img_h.shape
        img_tokens = img_h.reshape(b_img, h * w, d_img)
    elif img_h.ndim == 3:
        b_img, n, d_img = img_h.shape
        img_tokens = img_h
    else:
        raise RuntimeError(f"Unexpected image feature shape: {tuple(img_h.shape)}")

    # ---- audio features ----
    aud_out = model.get_audio_features(
        input_features=mm.get("input_features", None),
        feature_attention_mask=mm.get("feature_attention_mask", None),
        audio_feature_lengths=mm.get("audio_feature_lengths", None),
        return_dict=True,
    )
    aud_tokens = aud_out.last_hidden_state  # (B, N, D)
    if aud_tokens.ndim != 3:
        raise RuntimeError(f"Unexpected audio feature shape: {tuple(aud_tokens.shape)}")
    b_aud, n_aud, d_aud = aud_tokens.shape

    # ---- write back to numpy (scatter to original batch positions) ----
    # update d_image/d_audio if differs from d_text
    if int(d_img) != int(d_text) or int(d_aud) != int(d_text):
        # re-init the numpy buffers with correct dims (rare, but keep safe)
        # NOTE: This should almost never happen; Qwen typically projects to text hidden.
        text_embeds_np = np.zeros((bsz, max_text_len, d_text), dtype=save_dtype)
        text_mask_np = np.zeros((bsz, max_text_len), dtype=np.uint8)
        image_tokens_np = np.zeros((bsz, image_tokens_target, int(d_img)), dtype=save_dtype)
        audio_tokens_np = np.zeros((bsz, audio_tokens_target, int(d_aud)), dtype=save_dtype)

    # convert text to cpu
    text_emb_cpu = text_emb.detach().cpu()
    attn_cpu = attn_mask.detach().cpu().numpy().astype(np.uint8)

    # convert image/audio to cpu (pad/trunc per sample)
    img_tokens_cpu = img_tokens.detach().cpu()
    aud_tokens_cpu = aud_tokens.detach().cpu()

    # dtype conversion helper
    def _to_np(t: torch.Tensor) -> np.ndarray:
        if save_dtype == np.float16:
            return t.to(dtype=torch.float16).numpy()
        return t.to(dtype=torch.float32).numpy()

    for k, bi in enumerate(kept_map):
        # text
        text_embeds_np[bi] = _to_np(text_emb_cpu[k])
        text_mask_np[bi] = attn_cpu[k]

        # image tokens pad/trunc
        it = img_tokens_cpu[k]  # (Ni, Di)
        it = pad_or_trunc_tokens_2d(it, int(image_tokens_target))
        image_tokens_np[bi] = _to_np(it)

        # audio tokens pad/trunc
        at = aud_tokens_cpu[k]  # (Na, Da)
        at = pad_or_trunc_tokens_2d(at, int(audio_tokens_target))
        audio_tokens_np[bi] = _to_np(at)

        valid[bi] = 1

    return text_embeds_np, text_mask_np, image_tokens_np, audio_tokens_np, valid


# =============================
# main extraction
# =============================
def extract_features(cfg: Dict[str, Any]) -> None:
    dcfg = cfg["dataset"]
    rcfg = cfg["runtime"]
    ecfg = cfg["encoders"]

    out_dir = rcfg["output_dir"]
    ensure_dir(out_dir)

    overwrite = bool(rcfg.get("overwrite", False))
    info_path = os.path.join(out_dir, "dataset_info.yaml")
    if (not overwrite) and os.path.exists(info_path):
        raise FileExistsError(f"{info_path} exists. Set runtime.overwrite=true to overwrite.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dtype = _parse_np_dtype_save(rcfg.get("save_dtype", "fp16"))
    torch_dtype = _parse_torch_dtype(ecfg["qwen"].get("torch_dtype", "fp16"))

    model_name = ecfg["qwen"]["model_name_or_path"]
    trust_remote_code = bool(ecfg["qwen"].get("trust_remote_code", True))
    print(f"model_name_or_path: {model_name}, trust_remote_code: {trust_remote_code}, torch_dtype: {torch_dtype}, save_dtype: {save_dtype},dcfg: {dcfg}, ecfg: {ecfg}, rcfg: {rcfg}")
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    # 用 Thinker 就够了：包含 vision/audio backbone + text LM（无需 talker）
    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    # tokenizer pad token safety
    if processor.tokenizer.pad_token is None:
        if processor.tokenizer.eos_token is not None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        else:
            processor.tokenizer.pad_token = processor.tokenizer.convert_ids_to_tokens(0)

    max_text_len = int(ecfg["qwen"].get("max_text_len", 64))
    add_special_tokens = bool(ecfg["qwen"].get("add_special_tokens", False))
    image_tokens_target = int(ecfg["qwen"].get("image_tokens", 256))
    audio_tokens_target = int(ecfg["qwen"].get("audio_tokens", 128))

    # dataset
    train_ds = SpokenCoCoTripletDataset(
        json_path=dcfg["train_json"],
        coco_root=dcfg["coco_root"],
        spokencoco_root=dcfg["spokencoco_root"],
        split="train",
        verify_files=bool(dcfg.get("verify_files", True)),
        skip_missing=bool(dcfg.get("skip_missing", True)),
        max_samples=int(dcfg.get("max_samples", -1)),
    )
    val_ds = SpokenCoCoTripletDataset(
        json_path=dcfg["val_json"],
        coco_root=dcfg["coco_root"],
        spokencoco_root=dcfg["spokencoco_root"],
        split="val",
        verify_files=bool(dcfg.get("verify_files", True)),
        skip_missing=bool(dcfg.get("skip_missing", True)),
        max_samples=int(dcfg.get("max_samples", -1)),
    )
    ds = ConcatDataset([train_ds, val_ds])
    num_samples = len(ds)

    dl = DataLoader(
        ds,
        batch_size=int(rcfg.get("batch_size", 4)),
        shuffle=False,
        num_workers=int(rcfg.get("num_workers", 0)),
        collate_fn=collate_triplets,
        pin_memory=bool(rcfg.get("pin_memory", True)),
        drop_last=False,
    )

    # ---- probe dims by running one tiny dummy forward (only to get hidden sizes) ----
    # text dim
    d_text = int(model.get_input_embeddings().embedding_dim)

    # image/audio dims (safe probe)
    dummy_img = Image.new("RGB", (32, 32), color=(0, 0, 0))
    target_sr = int(getattr(processor.feature_extractor, "sampling_rate", 16000))
    dummy_audio = np.zeros((1, target_sr // 20), dtype=np.float32)  # 0.05s
    mm_probe = processor(text=None, images=[dummy_img], audio=[dummy_audio], return_tensors="pt", padding=True)
    mm_probe = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in mm_probe.items()}
    model_dtype = next(model.parameters()).dtype
    if "pixel_values" in mm_probe and torch.is_tensor(mm_probe["pixel_values"]):
        mm_probe["pixel_values"] = mm_probe["pixel_values"].to(dtype=model_dtype)
    if "input_features" in mm_probe and torch.is_tensor(mm_probe["input_features"]):
        mm_probe["input_features"] = mm_probe["input_features"].to(dtype=model_dtype)

    with torch.inference_mode():
        img_out = model.get_image_features(
            pixel_values=mm_probe.get("pixel_values", None),
            image_grid_thw=mm_probe.get("image_grid_thw", None),
            return_dict=True,
        )
        img_h = img_out.last_hidden_state
        d_image = int(img_h.shape[-1])

        aud_out = model.get_audio_features(
            input_features=mm_probe.get("input_features", None),
            feature_attention_mask=mm_probe.get("feature_attention_mask", None),
            audio_feature_lengths=mm_probe.get("audio_feature_lengths", None),
            return_dict=True,
        )
        d_audio = int(aud_out.last_hidden_state.shape[-1])

    # writer
    writer = MemmapWriter(
        out_dir=out_dir,
        num_samples=num_samples,
        text_tokens=max_text_len,
        d_text=d_text,
        image_tokens=image_tokens_target,
        d_image=d_image,
        audio_tokens=audio_tokens_target,
        d_audio=d_audio,
        dtype=save_dtype,
    )

    log_every = int(rcfg.get("log_every", 20))
    flush_every = int(rcfg.get("flush_every", 50))

    t0 = time.time()
    written = 0

    for step, batch in enumerate(dl):
        texts = batch["text"]
        image_paths = batch["image_path"]
        audio_paths = batch["audio_path"]
        image_rel = batch["image_rel"]
        wav_rel = batch["wav_rel"]

        te, tm, it, at, va = extract_batch_features(
            processor=processor,
            model=model,
            texts=texts,
            image_paths=image_paths,
            audio_paths=audio_paths,
            max_text_len=max_text_len,
            add_special_tokens=add_special_tokens,
            image_tokens_target=image_tokens_target,
            audio_tokens_target=audio_tokens_target,
            device=device,
            save_dtype=save_dtype,
        )

        writer.write_batch(
            text_embeds=te,
            text_mask=tm,
            image_tokens=it,
            audio_tokens=at,
            image_rel=image_rel,
            wav_rel=wav_rel,
            valid=va.astype(np.uint8),
        )

        written += len(texts)

        if flush_every > 0 and (step + 1) % flush_every == 0:
            writer.flush()

        if log_every > 0 and (step + 1) % log_every == 0:
            elapsed = time.time() - t0
            speed = written / max(elapsed, 1e-9)
            valid_ratio = float(writer.valid[:writer._ptr].mean()) if writer._ptr > 0 else 0.0
            print(f"[extract] step={step+1} written={written}/{num_samples} speed={speed:.2f} samples/s valid_ratio={valid_ratio:.4f}")

    writer.flush()

    info = {
        "version": 1,
        "num_samples_total": int(num_samples),
        "model": {
            "model_name_or_path": model_name,
            "torch_dtype": str(torch_dtype),
            "trust_remote_code": bool(trust_remote_code),
        },
        "features": {
            "text": {"max_len": int(max_text_len), "hidden": int(d_text), "dtype": str(save_dtype), "add_special_tokens": bool(add_special_tokens)},
            "image": {"num_tokens": int(image_tokens_target), "hidden": int(d_image), "dtype": str(save_dtype)},
            "audio": {"num_tokens": int(audio_tokens_target), "hidden": int(d_audio), "dtype": str(save_dtype)},
        },
        "shards": [
            {
                "num_samples": int(num_samples),
                "paths": writer.paths,
            }
        ],
        "notes": {
            "valid=0 means (missing file) OR (text token length > max_text_len) OR (decode failure).",
            "text is embedded by model.get_input_embeddings(); image/audio by model.get_image_features/get_audio_features.",
        },
    }
    with open(info_path, "w") as f:
        yaml.safe_dump(info, f, sort_keys=False)

    print(f"[done] wrote features to {out_dir}")
    print(f"[done] dataset_info: {info_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=os.environ.get("MM_CFG", ""))
    args = parser.parse_args()
    if not args.config:
        raise ValueError("Please provide --config or set env MM_CFG")
    cfg = load_yaml(args.config)
    if not cfg:
        raise ValueError(f"Empty config: {args.config}")
    extract_features(cfg)


if __name__ == "__main__":
    main()

