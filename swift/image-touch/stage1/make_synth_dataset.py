from __future__ import annotations

from pathlib import Path

from src.data.synthetic import SynthSpec, generate_synth_paired_npz
from src.utils.common import load_yaml, set_global_seed

CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "default.yaml"


def main():
    cfg = load_yaml(CONFIG_PATH)
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)

    synth_cfg = cfg["data"]["synth"]
    model_cfg = cfg["model"]

    out_path = Path(synth_cfg["out_path"])
    spec = SynthSpec(
        num_samples=int(synth_cfg["num_samples"]),
        point_tokens=int(model_cfg["point_tokens"]),
        d_point=int(model_cfg["d_point_in"]),
        max_text_len=int(synth_cfg["max_text_len"]),
        min_text_len=int(synth_cfg["min_text_len"]),
        d_text=int(model_cfg["d_text_in"]),
        d_true=int(synth_cfg["d_true"]),
        noise_std_text=float(synth_cfg["noise_std_text"]),
        noise_std_point=float(synth_cfg["noise_std_point"]),
        seed=seed,
    )

    print(f"[Info] generating synth dataset -> {out_path}")
    generate_synth_paired_npz(out_path, spec)
    print("[Info] done.")


if __name__ == "__main__":
    main()
