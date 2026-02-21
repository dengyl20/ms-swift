from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SynthSpec:
    num_samples: int
    point_tokens: int
    d_point: int
    max_text_len: int
    min_text_len: int
    d_text: int
    d_true: int
    noise_std_text: float
    noise_std_point: float
    seed: int = 42


def generate_synth_paired_npz(out_path: str | Path, spec: SynthSpec) -> Path:
    """
    生成 paired dataset:
      z ~ N(0, I)
      point[i, t] = z @ W_point + P_point[t] + eps
      text[i, l]  = z @ W_text  + P_text[l]  + eps

    text 存为 padded 到 max_text_len，并保存 lengths。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(spec.seed)

    W_text = (rng.standard_normal((spec.d_true, spec.d_text)).astype(np.float32) / np.sqrt(spec.d_true))
    W_point = (rng.standard_normal((spec.d_true, spec.d_point)).astype(np.float32) / np.sqrt(spec.d_true))

    P_text = (rng.standard_normal((spec.max_text_len, spec.d_text)).astype(np.float32) * 0.2)
    P_point = (rng.standard_normal((spec.point_tokens, spec.d_point)).astype(np.float32) * 0.2)

    point = np.zeros((spec.num_samples, spec.point_tokens, spec.d_point), dtype=np.float32)
    text = np.zeros((spec.num_samples, spec.max_text_len, spec.d_text), dtype=np.float32)
    lengths = np.zeros((spec.num_samples,), dtype=np.int64)

    for i in range(spec.num_samples):
        L = int(rng.integers(spec.min_text_len, spec.max_text_len + 1))
        z = rng.standard_normal((spec.d_true,), dtype=np.float32)

        base_text = z @ W_text      # (d_text,)
        base_point = z @ W_point    # (d_point,)

        noise_text = rng.standard_normal((L, spec.d_text), dtype=np.float32) * spec.noise_std_text
        noise_point = rng.standard_normal((spec.point_tokens, spec.d_point), dtype=np.float32) * spec.noise_std_point

        text[i, :L, :] = base_text[None, :] + P_text[:L, :] + noise_text
        point[i, :, :] = base_point[None, :] + P_point + noise_point
        lengths[i] = L

    np.savez_compressed(
        out_path,
        point=point,
        text=text,
        lengths=lengths,
    )
    return out_path
