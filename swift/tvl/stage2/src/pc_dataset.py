# pc_dataset.py
from __future__ import annotations

import os
import json
import pickle
import bisect
import inspect
import hashlib
import shutil
from typing import Any, Dict, Optional, List, Iterator, Callable

import numpy as np
import yaml
from datasets import Dataset as HFDataset
from datasets import load_from_disk

from swift.point_cloud.stage2.src.pc_constants import (
    DEFAULT_MAX_INJECT_TOKENS,
    DEFAULT_REQUIRE_VALID,
    ENV_CONV_JSON_PATH,
    ENV_FEATURE_INFO_YAML,
    ENV_MAX_INJECT_TOKENS,
    ENV_REQUIRE_VALID,
    POINT_TOKEN,
)
from .pc_utils import build_user_prompt_with_points, load_conversation_map, strip_all_point_placeholders

from swift.dataset import DatasetMeta, register_dataset
from swift.dataset.dataset_meta import BaseDatasetLoader
from swift.utils import get_logger, safe_ddp_context

logger = get_logger()


def _noop_preprocess(dataset, **kwargs):
    return dataset


def _get_bool_env(name: str, default: bool) -> bool:
    v = os.environ.get(name, None)
    if v is None:
        return bool(default)
    v = str(v).strip().lower()
    return v not in ("0", "false", "no", "off")


def _file_mtime(path: str) -> float:
    try:
        return float(os.path.getmtime(path))
    except Exception:
        return -1.0


def _build_or_load_conv_cache(conv_json_path: str, cache_path: str) -> Dict[str, Dict[str, str]]:
    """
    Load (or build once) a mapping: object_id -> {"human": ..., "gpt": ...}.

    Cache invalidation:
      - Rebuild when `conversations.json` is newer than the cache.
      - Guard rebuild with `safe_ddp_context` so only one rank writes.
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    cache_ok = os.path.isfile(cache_path) and (_file_mtime(cache_path) >= _file_mtime(conv_json_path))
    if cache_ok:
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    with safe_ddp_context(hash_id=cache_path, use_barrier=True):
        cache_ok = os.path.isfile(cache_path) and (_file_mtime(cache_path) >= _file_mtime(conv_json_path))
        if not cache_ok:
            logger.info(f"[PointCloudDataset] Building conversation cache from: {conv_json_path}")
            conv_map = load_conversation_map(conv_json_path)
            tmp_path = cache_path + ".tmp"
            with open(tmp_path, "wb") as f:
                pickle.dump(conv_map, f)
            os.replace(tmp_path, cache_path)
            logger.info(f"[PointCloudDataset] Saved conversation cache: {cache_path} (size={len(conv_map)})")

    with open(cache_path, "rb") as f:
        return pickle.load(f)


class _ProcessedFeatureReader:
    """
    Lightweight reader for the memmap shards produced by stage1 `extract_features.py`.

    Provides:
      - fast metadata scan (object_id / valid / global_index) without copying huge tensors
      - lazy, per-process memmap opening (safe for multi-worker DataLoader)
      - random access by a single global index (idx in [0, total))
    """

    def __init__(self, dataset_info_yaml: str):
        self.dataset_info_yaml = dataset_info_yaml

        with open(dataset_info_yaml, "r") as f:
            info = yaml.safe_load(f)

        self.info = info
        self.shards: List[Dict[str, Any]] = info["shards"]

        # shard sizes / prefix sums
        self.shard_sizes: List[int] = [int(s["num_samples"]) for s in self.shards]
        self.prefix: List[int] = [0]
        for n in self.shard_sizes:
            self.prefix.append(self.prefix[-1] + n)
        self.total: int = self.prefix[-1]

        # lazy-open caches (per-process/per-worker)
        self._mmaps: Dict[int, Dict[str, np.memmap]] = {}

    def __len__(self) -> int:
        return self.total

    def _dtype_from_shard(self, shard: Dict[str, Any]) -> Any:
        dtype_str = str(shard["text"].get("dtype", "")).lower()
        if "float32" in dtype_str:
            return np.float32
        return np.float16

    def _open_shard(self, shard_idx: int) -> Dict[str, np.memmap]:
        if shard_idx in self._mmaps:
            return self._mmaps[shard_idx]

        s = self.shards[shard_idx]
        paths = s["paths"]

        n = int(s["num_samples"])
        max_len = int(s["text"]["max_len"])
        hidden = int(s["text"]["hidden"])

        G = int(s["point"]["num_tokens"])
        trans_dim = int(s["point"]["trans_dim"])

        dt = self._dtype_from_shard(s)

        mm: Dict[str, np.memmap] = {
            "text_embeds": np.memmap(paths["text_embeds"], mode="r", dtype=dt, shape=(n, max_len, hidden)),
            "text_mask": np.memmap(paths["text_mask"], mode="r", dtype=np.uint8, shape=(n, max_len)),
            "point_tokens": np.memmap(paths["point_tokens"], mode="r", dtype=dt, shape=(n, G, trans_dim)),
            "object_ids": np.memmap(paths["object_ids"], mode="r", dtype="S32", shape=(n,)),
            "global_indices": np.memmap(paths["global_indices"], mode="r", dtype=np.int64, shape=(n,)),
            "valid": np.memmap(paths["valid"], mode="r", dtype=np.uint8, shape=(n,)),
        }
        self._mmaps[shard_idx] = mm
        return mm

    def _locate(self, idx: int) -> (int, int):
        if idx < 0 or idx >= self.total:
            raise IndexError(idx)
        shard_idx = bisect.bisect_right(self.prefix, idx) - 1
        local_idx = idx - self.prefix[shard_idx]
        return shard_idx, local_idx

    @staticmethod
    def _decode_object_id(raw: Any) -> str:
        return raw.tobytes().split(b"\x00", 1)[0].decode("utf-8")

    def get_arrays(self, idx: int) -> Dict[str, Any]:
        shard_idx, local_idx = self._locate(idx)
        mm = self._open_shard(shard_idx)

        # Copy into RAM to avoid holding views into mmap slices.
        text_embeds = np.array(mm["text_embeds"][local_idx], copy=True)  # (L, H)
        text_mask_u8 = np.array(mm["text_mask"][local_idx], copy=True)  # (L,)
        point_tokens = np.array(mm["point_tokens"][local_idx], copy=True)  # (G, D)

        return {
            "text_embeds": text_embeds,
            "text_mask": text_mask_u8.astype(np.bool_),
            "point_tokens": point_tokens,
            "object_id": self._decode_object_id(mm["object_ids"][local_idx]),
            "global_index": int(mm["global_indices"][local_idx].item()),
            "valid": bool(mm["valid"][local_idx].item()),
        }

    def __getstate__(self):
        # Drop opened mmaps when pickling (e.g., for spawn workers).
        state = dict(self.__dict__)
        state["_mmaps"] = {}
        return state


class _PointCloudFeatureTransform:
    """
    HF Dataset transform: take a lightweight index row (raw_idx + cached texts) and
    lazily load memmap tensors to produce the final training sample dict.
    """

    def __init__(
        self,
        feature_info_yaml: str,
        max_inject_tokens: int,
        require_valid: bool,
        point_placeholder: str = POINT_TOKEN,
        legacy_placeholder: str = "<point>",
    ):
        self.feature_info_yaml = feature_info_yaml
        self.max_inject_tokens = int(max_inject_tokens)
        self.require_valid = bool(require_valid)
        self.point_placeholder = str(point_placeholder)
        self.legacy_placeholder = str(legacy_placeholder)

        self._reader: Optional[_ProcessedFeatureReader] = None

    def _get_reader(self) -> _ProcessedFeatureReader:
        if self._reader is None:
            self._reader = _ProcessedFeatureReader(self.feature_info_yaml)
        return self._reader

    def _build_messages(self, human_raw: str, gpt_raw: str, inject_len: int) -> List[Dict[str, str]]:
        q = strip_all_point_placeholders(human_raw, placeholder=self.legacy_placeholder)
        q = strip_all_point_placeholders(q, placeholder=self.point_placeholder)
        user_text = build_user_prompt_with_points(q, inject_len, placeholder=self.point_placeholder)
        return [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": gpt_raw},
        ]

    @staticmethod
    def _is_batched(x: Any) -> bool:
        return isinstance(x, (list, tuple, np.ndarray))

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        `datasets` will pass either:
          - a single example dict (values are scalars), or
          - a batch dict (values are lists)

        We support both for completeness.
        """
        # IMPORTANT:
        # HF Dataset transforms are invoked not only for ds[i] but also for column access ds['lengths'].
        # In those cases, 'raw_idx'/'human'/'gpt' may not be present.
        if "raw_idx" not in batch or "human" not in batch or "gpt" not in batch:
            return batch
        if "messages" in batch and "point_tokens" in batch:
            return batch

        
        if self._is_batched(batch["raw_idx"]):
            out: Dict[str, List[Any]] = {
                "messages": [],
                "point_tokens": [],
                "text_embeds": [],
                "text_mask": [],
                "inject_len": [],
                "object_id": [],
                "global_index": [],
                "valid": [],
            }
            for i in range(len(batch["raw_idx"])):
                one = {k: batch[k][i] for k in batch.keys()}
                one_out = self.__call__(one)
                for k in out.keys():
                    out[k].append(one_out[k])
            return out

        raw_idx = int(batch["raw_idx"])
        human_raw = str(batch["human"])
        gpt_raw = str(batch["gpt"])

        reader = self._get_reader()
        item = reader.get_arrays(raw_idx)

        valid = bool(item["valid"])
        if self.require_valid and (not valid):
            raise RuntimeError(f"Invalid sample at raw_idx={raw_idx}")

        # inject length computed from text_mask, then clamped
        k = int(np.sum(item["text_mask"]))
        k = max(1, min(k, self.max_inject_tokens))

        messages = self._build_messages(human_raw, gpt_raw, k)

        return {
            "messages": messages,
            "point_tokens": item["point_tokens"],
            "text_embeds": item["text_embeds"],
            "text_mask": item["text_mask"],
            "inject_len": int(k),
            "object_id": str(item["object_id"]),
            "global_index": int(item["global_index"]),
            "valid": valid,
        }

    def __getstate__(self):
        # Drop reader when pickling; each worker will lazily reopen mmaps.
        state = dict(self.__dict__)
        state["_reader"] = None
        return state


def _make_cache_fingerprint(
    feature_info_yaml: str,
    conv_json_path: str,
    require_valid: bool,
) -> str:
    """
    Fingerprint for the lightweight Arrow index.

    Note:
      - We do NOT include `max_inject_tokens` here, because inject_len is computed lazily
        in `_PointCloudFeatureTransform` at access time.
    """
    payload = {
        "feature_info_yaml": os.path.abspath(feature_info_yaml),
        "feature_info_yaml_mtime": _file_mtime(feature_info_yaml),
        "conv_json_path": os.path.abspath(conv_json_path),
        "conv_json_path_mtime": _file_mtime(conv_json_path),
        "require_valid": bool(require_valid),
    }
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _hf_from_generator_safe(gen_fn: Callable[[], Iterator[Dict[str, Any]]]) -> HFDataset:
    """
    Compatibility helper: `Dataset.from_generator` is available in modern `datasets`,
    but we keep a safe fallback for older environments.
    """
    try:
        return HFDataset.from_generator(gen_fn)
    except Exception:
        rows = list(gen_fn())
        if not rows:
            return HFDataset.from_dict({"raw_idx": [], "human": [], "gpt": [], "object_id": [], "global_index": []})
        cols: Dict[str, List[Any]] = {k: [] for k in rows[0].keys()}
        for r in rows:
            for k, v in r.items():
                cols[k].append(v)
        return HFDataset.from_dict(cols)


class PointCloudFeatureSFTDatasetLoader(BaseDatasetLoader):
    """
    Non-streaming (map-style) dataset loader for SWIFT.

    Returns a HuggingFace `datasets.Dataset` (NOT IterableDataset), with:
      - __len__ available
      - __getitem__ available
      - supports .shard() / .select() / random access
      - compatible with per_device_train_batch_size > 1

    Each sample dict (after transform) contains:
      - messages: List[{"role","content"}]
      - point_tokens: (G, D) np.ndarray
      - text_embeds:  (L, H) np.ndarray
      - text_mask:    (L,)   np.ndarray(bool)
      - inject_len: int
      - object_id/global_index/valid: debug fields
    """

    def __init__(
        self,
        num_proc: int = 1,
        load_from_cache_file: bool = True,
        streaming: bool = False,
        hub_token: Optional[str] = None,
        strict: bool = False,
        download_mode: str = "reuse_dataset_if_exists",
        columns: Optional[Dict[str, str]] = None,
        remove_unused_columns: bool = False,
    ):
        self.num_proc = num_proc
        self.load_from_cache_file = load_from_cache_file
        self.streaming = bool(streaming)
        self.hub_token = hub_token
        self.strict = strict
        self.download_mode = download_mode
        self.columns = columns
        self.remove_unused_columns = remove_unused_columns

    def load(self, dataset_syntax=None, dataset_meta=None, *, use_hf: Optional[bool] = None):
        feature_info_yaml = os.environ.get(ENV_FEATURE_INFO_YAML, None)
        conv_json_path = os.environ.get(ENV_CONV_JSON_PATH, None)
        if not feature_info_yaml or not conv_json_path:
            raise ValueError(
                f"Missing env vars. Please set:\n"
                f"  export {ENV_FEATURE_INFO_YAML}=/path/to/dataset_info.yaml\n"
                f"  export {ENV_CONV_JSON_PATH}=/path/to/conversations.json"
            )

        max_inject = int(os.environ.get(ENV_MAX_INJECT_TOKENS, str(DEFAULT_MAX_INJECT_TOKENS)))
        require_valid = _get_bool_env(ENV_REQUIRE_VALID, DEFAULT_REQUIRE_VALID)

        # 1) conversation map cache (used only during indexing; final dataset stores human/gpt texts)
        conv_cache_path = os.path.join(os.path.dirname(conv_json_path), ".cache_pointcloud_conv_map.pkl")
        conv_map = _build_or_load_conv_cache(conv_json_path, conv_cache_path)

        # 2) build/load lightweight arrow dataset (raw_idx + cached texts)
        fp = _make_cache_fingerprint(feature_info_yaml, conv_json_path, require_valid)
        cache_root = os.path.join(os.path.dirname(conv_json_path), ".cache_pointcloud_feature_sft")
        ds_disk_path = os.path.join(cache_root, f"hf_ds_{fp}")
        rank = int(os.environ.get("RANK", "0"))
        if _get_bool_env("POINTCLOUD_CACHE_PER_RANK", True):
            ds_disk_path = ds_disk_path + f"_rank{rank}"


        os.makedirs(cache_root, exist_ok=True)

        if self.load_from_cache_file and os.path.isdir(ds_disk_path):
            logger.info(f"[PointCloudDataset] Loading cached HF dataset index from: {ds_disk_path}")
            ds = load_from_disk(ds_disk_path)
        else:
            with safe_ddp_context(hash_id=ds_disk_path, use_barrier=True):
                if not (self.load_from_cache_file and os.path.isdir(ds_disk_path)):
                    logger.info(f"[PointCloudDataset] Building HF dataset index (map-style), cache={ds_disk_path}")
                    reader = _ProcessedFeatureReader(feature_info_yaml)

                    def gen() -> Iterator[Dict[str, Any]]:
                        total = len(reader)
                        kept = 0
                        # iterate shard-by-shard for sequential IO
                        for shard_idx, shard in enumerate(reader.shards):
                            mm = reader._open_shard(shard_idx)
                            n = int(shard["num_samples"])
                            base = reader.prefix[shard_idx]
                            logger.info(f"[PointCloudDataset] Scanning shard {shard_idx}/{len(reader.shards)-1} (n={n})")
                            for local_idx in range(n):
                                if require_valid and (not bool(mm["valid"][local_idx].item())):
                                    continue
                                obj_id = reader._decode_object_id(mm["object_ids"][local_idx])
                                conv = conv_map.get(obj_id)
                                if conv is None:
                                    continue
                                # Be tolerant to key variants.
                                human = conv.get("human", conv.get("query", conv.get("prompt", "")))
                                gpt = conv.get("gpt", conv.get("assistant", conv.get("response", "")))
                                if not human or not gpt:
                                    continue

                                raw_idx = int(base + local_idx)
                                global_index = int(mm["global_indices"][local_idx].item())
                                kept += 1
                                yield {
                                    "raw_idx": raw_idx,
                                    "human": human,
                                    "gpt": gpt,
                                    "object_id": obj_id,
                                    "global_index": global_index,
                                }
                        logger.info(f"[PointCloudDataset] Index build finished. kept={kept}, total_features={total}")

                    ds = _hf_from_generator_safe(gen)

                    # Save for future runs
                    if os.path.isdir(ds_disk_path):
                        shutil.rmtree(ds_disk_path, ignore_errors=True)
                    ds.save_to_disk(ds_disk_path)
                    logger.info(f"[PointCloudDataset] Saved HF dataset index to: {ds_disk_path}")

            ds = load_from_disk(ds_disk_path)

        # 3) attach lazy transform to load memmap arrays & build messages
        transform = _PointCloudFeatureTransform(
            feature_info_yaml=feature_info_yaml,
            max_inject_tokens=max_inject,
            require_valid=require_valid,
            point_placeholder=POINT_TOKEN,
            legacy_placeholder="<point>",
        )
        ds = ds.with_transform(transform)

        logger.info(
            f"[PointCloudDataset] Loaded dataset size={len(ds)} "
            f"(map-style; ignore streaming={self.streaming})"
        )
        return ds


def _call_register_dataset(meta: DatasetMeta, exists_ok: bool) -> None:
    """Compatibility: `register_dataset` signature varies across ms-swift versions."""
    sig = inspect.signature(register_dataset)
    if "exist_ok" in sig.parameters:
        register_dataset(meta, exist_ok=exists_ok)
    elif "exists_ok" in sig.parameters:
        register_dataset(meta, exists_ok=exists_ok)
    else:
        register_dataset(meta)


def register_pointcloud_dataset(exists_ok: bool = True) -> None:
    meta = DatasetMeta(
        dataset_name="pointcloud_feature_sft",
        preprocess_func=_noop_preprocess,
        loader=PointCloudFeatureSFTDatasetLoader,
        huge_dataset=False,  # map-style: supports __len__/__getitem__
        help=(
            "PointCloud feature SFT dataset (map-style, non-streaming). Requires env vars:\n"
            f"  {ENV_FEATURE_INFO_YAML}, {ENV_CONV_JSON_PATH}\n"
            f"Optional:\n  {ENV_MAX_INJECT_TOKENS}, {ENV_REQUIRE_VALID}\n"
            "It builds a lightweight Arrow index and lazily loads memmap tensors at __getitem__ time."
        ),
    )
    _call_register_dataset(meta, exists_ok=exists_ok)
    logger.info("[PointCloudDataset] Registered dataset_name=pointcloud_feature_sft with DatasetMeta.loader.")
