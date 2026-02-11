from __future__ import annotations

import bisect
from typing import Any, Dict

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset




import json
import os
from typing import Any, Dict, List

from torch.utils.data import Dataset


def _safe_join(root: str, rel_or_abs: str) -> str:
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    return os.path.normpath(os.path.join(root, rel_or_abs))


class SpokenCoCoTripletDataset(Dataset):
    """SpokenCOCO -> caption 粒度 triplet 样本。"""

    def __init__(
        self,
        json_path: str,
        coco_root: str,
        spokencoco_root: str,
        split: str,
        verify_files: bool = True,
        skip_missing: bool = True,
        max_samples: int = -1,
    ):
        super().__init__()
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        data = raw["data"] if isinstance(raw, dict) and "data" in raw else raw

        self.samples: List[Dict[str, Any]] = []
        self.stats = {
            "total_entries": len(data),
            "total_pairs": 0,
            "missing_image": 0,
            "missing_audio": 0,
            "bad_format": 0,
            "valid": 0,
        }

        for entry in data:
            if max_samples > 0 and len(self.samples) >= max_samples:
                break
            if not isinstance(entry, dict):
                self.stats["bad_format"] += 1
                continue

            image_rel = entry.get("image", "")
            captions = entry.get("captions", [])
            if (not isinstance(image_rel, str)) or (not isinstance(captions, list)):
                self.stats["bad_format"] += 1
                continue

            image_path = _safe_join(coco_root, image_rel)
            image_ok = (not verify_files) or os.path.isfile(image_path)

            for cap in captions:
                self.stats["total_pairs"] += 1
                if not isinstance(cap, dict):
                    self.stats["bad_format"] += 1
                    continue
                wav_rel = cap.get("wav", "")
                if not isinstance(wav_rel, str):
                    self.stats["bad_format"] += 1
                    continue

                audio_path = _safe_join(spokencoco_root, wav_rel)
                audio_ok = (not verify_files) or os.path.isfile(audio_path)

                if not image_ok:
                    self.stats["missing_image"] += 1
                if not audio_ok:
                    self.stats["missing_audio"] += 1

                valid = bool(image_ok and audio_ok)
                if (not valid) and skip_missing:
                    continue

                self.samples.append(
                    {
                        "split": split,
                        "image_rel": image_rel,
                        "image_path": image_path,
                        "wav_rel": wav_rel,
                        "audio_path": audio_path,
                        "text": str(cap.get("text", "")),
                        "speaker": str(cap.get("speaker", "")),
                        "uttid": str(cap.get("uttid", "")),
                        "valid": valid,
                    }
                )

        self.stats["valid"] = len(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def collate_triplets(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    return {
        "split": [b["split"] for b in batch],
        "image_rel": [b["image_rel"] for b in batch],
        "image_path": [b["image_path"] for b in batch],
        "wav_rel": [b["wav_rel"] for b in batch],
        "audio_path": [b["audio_path"] for b in batch],
        "text": [b["text"] for b in batch],
        "speaker": [b["speaker"] for b in batch],
        "uttid": [b["uttid"] for b in batch],
        "valid": [bool(b["valid"]) for b in batch],
    }


class ProcessedImageAudioTextFeatureDataset(Dataset):
    def __init__(self, dataset_info_yaml: str, require_valid: bool = True):
        super().__init__()
        with open(dataset_info_yaml, "r") as f:
            self.info = yaml.safe_load(f)

        self.require_valid = bool(require_valid)
        self.shards = self.info["shards"]

        self.shard_sizes = [int(s["num_samples"]) for s in self.shards]
        self.prefix = [0]
        for n in self.shard_sizes:
            self.prefix.append(self.prefix[-1] + n)
        self.total = self.prefix[-1]
        self._mmaps: Dict[int, Dict[str, np.memmap]] = {}

    def __len__(self) -> int:
        return self.total

    def _open_shard(self, shard_idx: int) -> Dict[str, np.memmap]:
        if shard_idx in self._mmaps:
            return self._mmaps[shard_idx]

        s = self.shards[shard_idx]
        n = int(s["num_samples"])
        dt = np.float32 if "float32" in str(s["dtype"]).lower() else np.float16
        p = s["paths"]

        mm = {
            "text_embeds": np.memmap(p["text_embeds"], mode="r", dtype=dt, shape=(n, s["text_tokens"], s["d_text"])),
            "text_mask": np.memmap(p["text_mask"], mode="r", dtype=np.uint8, shape=(n, s["text_tokens"])),
            "image_tokens": np.memmap(p["image_tokens"], mode="r", dtype=dt, shape=(n, s["image_tokens"], s["d_image"])),
            "audio_tokens": np.memmap(p["audio_tokens"], mode="r", dtype=dt, shape=(n, s["audio_tokens"], s["d_audio"])),
            "image_rel": np.memmap(p["image_rel"], mode="r", dtype="S256", shape=(n,)),
            "wav_rel": np.memmap(p["wav_rel"], mode="r", dtype="S256", shape=(n,)),
            "global_indices": np.memmap(p["global_indices"], mode="r", dtype=np.int64, shape=(n,)),
            "valid": np.memmap(p["valid"], mode="r", dtype=np.uint8, shape=(n,)),
        }
        self._mmaps[shard_idx] = mm
        return mm

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= self.total:
            raise IndexError(idx)

        shard_idx = bisect.bisect_right(self.prefix, idx) - 1
        local_idx = idx - self.prefix[shard_idx]
        mm = self._open_shard(shard_idx)

        valid = bool(mm["valid"][local_idx].item())
        if self.require_valid and (not valid):
            raise RuntimeError(f"Invalid sample idx={idx}")

        image_rel = mm["image_rel"][local_idx].tobytes().split(b"\x00", 1)[0].decode("utf-8")
        wav_rel = mm["wav_rel"][local_idx].tobytes().split(b"\x00", 1)[0].decode("utf-8")

        return {
            "text_embeds": torch.from_numpy(np.array(mm["text_embeds"][local_idx], copy=True)),
            "text_mask": torch.from_numpy(np.array(mm["text_mask"][local_idx], copy=True)).bool(),
            "image_tokens": torch.from_numpy(np.array(mm["image_tokens"][local_idx], copy=True)),
            "audio_tokens": torch.from_numpy(np.array(mm["audio_tokens"][local_idx], copy=True)),
            "image_rel": image_rel,
            "wav_rel": wav_rel,
            "global_index": int(mm["global_indices"][local_idx].item()),
            "valid": valid,
        }

