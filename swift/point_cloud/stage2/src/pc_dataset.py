# pc_dataset.py
from __future__ import annotations

import os
import pickle
from typing import Any, Dict, Optional

import numpy as np
from datasets import IterableDataset

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


from swift.llm.dataset import DatasetMeta, register_dataset
from swift.llm.dataset.dataset_meta import BaseDatasetLoader
from swift.utils import get_logger, safe_ddp_context


logger = get_logger()


def _noop_preprocess(dataset, **kwargs):
    # 不做 map，不改列，直接返回
    return dataset


def _get_bool_env(name: str, default: bool) -> bool:
    v = os.environ.get(name, None)
    if v is None:
        return bool(default)
    v = str(v).strip().lower()
    return v not in ("0", "false", "no", "off")


def _build_or_load_conv_cache(conv_json_path: str, cache_path: str) -> Dict[str, Dict[str, str]]:
    """
    为了避免每个进程都从头扫一遍超大 JSON：
    - rank0 构建 cache
    - 其它 rank barrier 后直接 load
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.isfile(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # 用 safe_ddp_context 做 barrier（即使不是 DDP，也能安全用）
    with safe_ddp_context(hash_id=cache_path, use_barrier=True):
        if not os.path.isfile(cache_path):
            logger.info(f"[PointCloudDataset] Building conversation cache from: {conv_json_path}")
            conv_map = load_conversation_map(conv_json_path)
            with open(cache_path, "wb") as f:
                pickle.dump(conv_map, f)
            logger.info(f"[PointCloudDataset] Saved conversation cache: {cache_path}  (size={len(conv_map)})")

    with open(cache_path, "rb") as f:
        return pickle.load(f)


class PointCloudFeatureSFTDatasetLoader(BaseDatasetLoader):
    """
    自定义 dataset loader：只支持 streaming=True。
    产出 HF IterableDataset，每条样本包含：
      - messages: 标准对话
      - point_tokens: (G,D)
      - text_mask: (L,)
      - inject_len: int  (K)
      - object_id/global_index: debug
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
        remove_unused_columns: bool = True,
    ):
        self.streaming = bool(streaming)

    def load(self, dataset_syntax=None, dataset_meta=None, *, use_hf: Optional[bool] = None):
        if not self.streaming:
            raise ValueError(
                "PointCloudFeatureSFTDatasetLoader only supports streaming=True to avoid materializing huge tensors.\n"
                "Please run swift with: --streaming True"
            )

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

        # ===== 复用你自己的 memmap dataset =====
        from swift.point_cloud.stage1.src.data.feature_dataset import ProcessedPointTextFeatureDataset

        # ===== 会话 cache =====
        cache_path = os.path.join(os.path.dirname(conv_json_path), ".cache_first_round.pkl")
        conv_map = _build_or_load_conv_cache(conv_json_path, cache_path)

        def gen():
            ds = ProcessedPointTextFeatureDataset(feature_info_yaml, require_valid=require_valid)
            for idx in range(len(ds)):
                try:
                    item = ds[idx]
                except Exception:
                    continue

                obj_id = str(item["object_id"])
                conv = conv_map.get(obj_id)
                if conv is None:
                    continue

                human_raw = conv["human"]
                gpt_raw = conv["gpt"]

                # 你推理脚本里的做法：从 human 里去掉所有 <point>，得到纯问题
                question_text = strip_all_point_placeholders(human_raw, placeholder=POINT_TOKEN)

                # K：由 text_mask 决定（变长），并 cap
                text_mask = item["text_mask"]  # torch.bool (L,)
                k = int(text_mask.sum().item())
                k = max(1, min(k, max_inject))

                user_text = build_user_prompt_with_points(question_text, k, placeholder=POINT_TOKEN)

                messages = [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": gpt_raw},
                ]

                # 注意：HF IterableDataset 里建议放 numpy / python primitive
                pt = item["point_tokens"].cpu().numpy()        # (G,D)
                tm = item["text_mask"].cpu().numpy().astype(np.bool_)  # (L,)

                yield {
                    "messages": messages,
                    "point_tokens": pt,
                    "text_mask": tm,
                    "inject_len": int(k),
                    "object_id": obj_id,
                    "global_index": int(item.get("global_index", -1)),
                }

        return IterableDataset.from_generator(gen)


def register_pointcloud_dataset(exists_ok: bool = True) -> None:
    register_dataset(
        DatasetMeta(
            ms_dataset_id="pointcloud_feature_sft",
            # 关键：我们自己 loader 生成数据，preprocess 不做任何处理
            preprocess_func=_noop_preprocess,
            loader=PointCloudFeatureSFTDatasetLoader,
            huge_dataset=True,
            help=(
                "PointCloud feature SFT dataset (streaming). Requires env vars:\n"
                f"  {ENV_FEATURE_INFO_YAML}, {ENV_CONV_JSON_PATH}\n"
                f"Optional:\n  {ENV_MAX_INJECT_TOKENS}, {ENV_REQUIRE_VALID}"
            ),
        ),
        exists_ok=exists_ok,
    )
