# pc_model.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from swift.point_cloud.stage2.src.pc_constants import ENV_AE_CKPT_PATH, ENV_FEATURE_INFO_YAML, POINT_TOKEN

from swift.model import Model, ModelGroup, ModelLoader, ModelMeta, register_model
from swift.utils import get_logger


logger = get_logger()

# 用 class 变量保存 tokenizer 的扩展信息（避免 get_model/get_processor 调用顺序差异）
_ADDED_TOKENIZER_LEN: Optional[int] = None
_POINT_TOKEN_ID: Optional[int] = None
_OLD_POINT_IDS: Optional[list] = None


def _torch_load_ckpt(path: str) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _infer_point_trans_dim_from_yaml(feature_info_yaml: str) -> int:
    """
    Read stage1 feature dataset_info.yaml to infer point token feature dim (trans_dim).

    Expected yaml structure (your stage1 extractor output):
      shards:
        - point:
            num_tokens: G
            trans_dim: D
        - ...
    """
    import yaml  # local import to keep pc_model.py lightweight

    with open(feature_info_yaml, "r") as f:
        info = yaml.safe_load(f)

    shards = info.get("shards", None)
    if not isinstance(shards, list) or len(shards) == 0:
        raise RuntimeError(f"Invalid feature_info_yaml: missing 'shards' list. path={feature_info_yaml}")

    trans_dims = []
    for s in shards:
        try:
            td = int(s["point"]["trans_dim"])
            trans_dims.append(td)
        except Exception:
            continue

    if not trans_dims:
        raise RuntimeError(
            f"Invalid feature_info_yaml: cannot find shards[*].point.trans_dim. path={feature_info_yaml}"
        )

    uniq = sorted(set(trans_dims))
    if len(uniq) != 1:
        logger.warning(
            f"[Qwen3OmniPoint] Inconsistent point.trans_dim across shards: {uniq}. "
            f"Will use max={max(uniq)}."
        )
    return int(max(uniq))


def _build_mlp2x_gelu(in_dim: int, out_dim: int) -> nn.Module:
    """
    LLaVA-style projector: `mlp2x_gelu`
      Linear(in_dim -> out_dim) + GELU + Linear(out_dim -> out_dim)

    Pattern widely used in open-source VLMs.
    """
    return nn.Sequential(
        nn.Linear(in_dim, out_dim, bias=True),
        nn.GELU(),
        nn.Linear(out_dim, out_dim, bias=True),
    )


class Qwen3OmniPointModelLoader(ModelLoader):
    """
    你的原 AE baseline：
    - Qwen3Omni Thinker（text-only）
    - 加 <point> special token + resize embedding
    - 加载 stage1 AE，并冻结除 AE 外所有参数
    """

    @staticmethod
    def get_processor(model_dir: str, model_info=None, processor_kwargs=None, **kwargs):
        global _ADDED_TOKENIZER_LEN, _POINT_TOKEN_ID, _OLD_POINT_IDS

        processor_kwargs = processor_kwargs or {}

        from transformers import AddedToken, Qwen3OmniMoeProcessor

        processor = Qwen3OmniMoeProcessor.from_pretrained(model_dir, **processor_kwargs)
        tokenizer = processor.tokenizer

        # 记录“加入 special token 之前”的分词结果，用于初始化新 token embedding
        _OLD_POINT_IDS = tokenizer.encode(POINT_TOKEN, add_special_tokens=False)

        # 加 special token（确保是 1 token）
        num_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": [AddedToken(POINT_TOKEN, lstrip=False, rstrip=False)]}
        )
        _POINT_TOKEN_ID = int(tokenizer.convert_tokens_to_ids(POINT_TOKEN))
        _ADDED_TOKENIZER_LEN = len(tokenizer)

        logger.info(
            f"[Qwen3OmniPoint] Added special token {POINT_TOKEN} (id={_POINT_TOKEN_ID}), "
            f"num_added={num_added}, tokenizer_len={_ADDED_TOKENIZER_LEN}"
        )
        return processor

    @staticmethod
    def get_model(
        model_dir: str,
        model_info: Optional[PretrainedConfig] = None,
        processor: Optional[Any] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        load_model: bool = True,
        **kwargs,
    ):
        global _ADDED_TOKENIZER_LEN, _POINT_TOKEN_ID, _OLD_POINT_IDS

        if not load_model:
            return None

        model_kwargs = model_kwargs or {}
        if not isinstance(model_kwargs, dict):
            raise TypeError(
                f"[Qwen3OmniPoint] model_kwargs must be a dict, but got: {type(model_kwargs)}. "
                "This usually means get_model() signature doesn't match ms-swift calling convention."
            )

        from transformers import Qwen3OmniMoeThinkerForConditionalGeneration

        hf_kwargs: Dict[str, Any] = {}
        for k in [
            "torch_dtype",
            "device_map",
            "attn_implementation",
            "trust_remote_code",
            "low_cpu_mem_usage",
        ]:
            if k in model_kwargs:
                hf_kwargs[k] = model_kwargs[k]

        if "torch_dtype" not in hf_kwargs:
            if "dtype" in model_kwargs:
                hf_kwargs["torch_dtype"] = model_kwargs["dtype"]
            elif "dtype" in kwargs:
                hf_kwargs["torch_dtype"] = kwargs["dtype"]

        for k in [
            "torch_dtype",
            "device_map",
            "attn_implementation",
            "trust_remote_code",
            "low_cpu_mem_usage",
        ]:
            if k in kwargs:
                hf_kwargs[k] = kwargs[k]

        model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(model_dir, **hf_kwargs)

        # resize embedding（必须做，否则新 token id 会越界）
        if _ADDED_TOKENIZER_LEN is not None:
            emb = model.get_input_embeddings()
            if emb.weight.shape[0] != int(_ADDED_TOKENIZER_LEN):
                model.resize_token_embeddings(int(_ADDED_TOKENIZER_LEN))
                logger.info(f"[Qwen3OmniPoint] Resized token embeddings to {_ADDED_TOKENIZER_LEN}")

                try:
                    if _POINT_TOKEN_ID is not None and _OLD_POINT_IDS:
                        with torch.no_grad():
                            emb = model.get_input_embeddings()
                            ids_t = torch.tensor(_OLD_POINT_IDS, device=emb.weight.device, dtype=torch.long)
                            init_vec = emb.weight.data.index_select(0, ids_t).mean(dim=0)
                            emb.weight.data[int(_POINT_TOKEN_ID)].copy_(init_vec)
                        logger.info("[Qwen3OmniPoint] Initialized <point> embedding from old tokenization mean.")
                except Exception as e:
                    logger.warning(f"[Qwen3OmniPoint] Failed to init <point> embedding: {e}")

        # 加载你的 point AE 并挂到模型上
        ae_ckpt = os.environ.get(ENV_AE_CKPT_PATH, None)
        if not ae_ckpt:
            raise ValueError(f"Missing env var: export {ENV_AE_CKPT_PATH}=/path/to/stage1/best.pt")

        ckpt = _torch_load_ckpt(ae_ckpt)
        if "cfg" not in ckpt or "model" not in ckpt:
            raise RuntimeError(f"AE ckpt format unexpected: keys={list(ckpt.keys())}")

        cfg_model = ckpt["cfg"]["model"]

        from swift.point_cloud.stage1.src.models.unified_ae import UnifiedPointTextAE

        point_ae = UnifiedPointTextAE(cfg_model)
        point_ae.load_state_dict(ckpt["model"], strict=True)

        emb_layer = model.get_input_embeddings()
        point_ae.to(device=emb_layer.weight.device, dtype=emb_layer.weight.dtype)

        llm_hidden = int(emb_layer.weight.shape[1])
        ae_hidden = int(cfg_model["d_text_in"])
        if ae_hidden != llm_hidden:
            raise ValueError(f"Hidden mismatch: AE d_text_in={ae_hidden} vs Qwen hidden={llm_hidden}")

        model.point_ae = point_ae
        model.point_token_id = int(_POINT_TOKEN_ID) if _POINT_TOKEN_ID is not None else None

        # 冻结：只训练 AE
        for p in model.parameters():
            p.requires_grad = False
        for p in model.point_ae.parameters():
            p.requires_grad = True

        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.warning(f"[Qwen3OmniPoint] Model ready. trainable={trainable} / total={total}")

        return model


class Qwen3OmniPointMLPModelLoader(ModelLoader):
    """
    新 baseline：用 2-layer MLP projector 替换 stage1 AE。

    - Qwen3Omni Thinker（text-only）
    - 加 <point> special token + resize embedding
    - build `point_projector` = mlp2x_gelu: D -> H -> H
    - freeze all except `point_projector`
    """

    @staticmethod
    def get_processor(model_dir: str, model_info=None, processor_kwargs=None, **kwargs):
        # 复用同一套 tokenizer 扩展逻辑
        return Qwen3OmniPointModelLoader.get_processor(
            model_dir=model_dir, model_info=model_info, processor_kwargs=processor_kwargs, **kwargs
        )

    @staticmethod
    def get_model(
        model_dir: str,
        model_info: Optional[PretrainedConfig] = None,
        processor: Optional[Any] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        load_model: bool = True,
        **kwargs,
    ):
        global _ADDED_TOKENIZER_LEN, _POINT_TOKEN_ID, _OLD_POINT_IDS

        if not load_model:
            return None

        model_kwargs = model_kwargs or {}
        if not isinstance(model_kwargs, dict):
            raise TypeError(
                f"[Qwen3OmniPointMLP] model_kwargs must be a dict, but got: {type(model_kwargs)}. "
                "This usually means get_model() signature doesn't match ms-swift calling convention."
            )

        from transformers import Qwen3OmniMoeThinkerForConditionalGeneration

        hf_kwargs: Dict[str, Any] = {}
        for k in [
            "torch_dtype",
            "device_map",
            "attn_implementation",
            "trust_remote_code",
            "low_cpu_mem_usage",
        ]:
            if k in model_kwargs:
                hf_kwargs[k] = model_kwargs[k]

        if "torch_dtype" not in hf_kwargs:
            if "dtype" in model_kwargs:
                hf_kwargs["torch_dtype"] = model_kwargs["dtype"]
            elif "dtype" in kwargs:
                hf_kwargs["torch_dtype"] = kwargs["dtype"]

        for k in [
            "torch_dtype",
            "device_map",
            "attn_implementation",
            "trust_remote_code",
            "low_cpu_mem_usage",
        ]:
            if k in kwargs:
                hf_kwargs[k] = kwargs[k]

        model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(model_dir, **hf_kwargs)

        # resize embeddings to include <point>
        if _ADDED_TOKENIZER_LEN is not None:
            emb = model.get_input_embeddings()
            if emb.weight.shape[0] != int(_ADDED_TOKENIZER_LEN):
                model.resize_token_embeddings(int(_ADDED_TOKENIZER_LEN))
                logger.info(f"[Qwen3OmniPointMLP] Resized token embeddings to {_ADDED_TOKENIZER_LEN}")

                # 可选：初始化 <point> embedding（注入 inputs_embeds 时不会用到，但保持合理）
                try:
                    if _POINT_TOKEN_ID is not None and _OLD_POINT_IDS:
                        with torch.no_grad():
                            emb = model.get_input_embeddings()
                            ids_t = torch.tensor(_OLD_POINT_IDS, device=emb.weight.device, dtype=torch.long)
                            init_vec = emb.weight.data.index_select(0, ids_t).mean(dim=0)
                            emb.weight.data[int(_POINT_TOKEN_ID)].copy_(init_vec)
                        logger.info("[Qwen3OmniPointMLP] Initialized <point> embedding from old tokenization mean.")
                except Exception as e:
                    logger.warning(f"[Qwen3OmniPointMLP] Failed to init <point> embedding: {e}")

        # build point projector (D from dataset_info.yaml)
        feature_info_yaml = os.environ.get(ENV_FEATURE_INFO_YAML, None)
        if not feature_info_yaml:
            raise ValueError(
                f"Missing env var: export {ENV_FEATURE_INFO_YAML}=/path/to/dataset_info.yaml "
                "(used to infer point feature trans_dim)."
            )

        point_in_dim = int(_infer_point_trans_dim_from_yaml(feature_info_yaml))

        emb_layer = model.get_input_embeddings()
        llm_hidden = int(emb_layer.weight.shape[1])

        point_projector = _build_mlp2x_gelu(point_in_dim, llm_hidden)
        point_projector.to(device=emb_layer.weight.device, dtype=emb_layer.weight.dtype)

        model.point_projector = point_projector
        model.point_token_id = int(_POINT_TOKEN_ID) if _POINT_TOKEN_ID is not None else None

        # freeze all except projector
        for p in model.parameters():
            p.requires_grad = False
        for p in model.point_projector.parameters():
            p.requires_grad = True

        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.warning(f"[Qwen3OmniPointMLP] Model ready. trainable={trainable} / total={total}")

        return model


def register_qwen3_omni_point_model(exists_ok: bool = True) -> None:
    register_model(
        ModelMeta(
            model_type="qwen3_omni_point",
            model_arch="qwen3_omni",
            template="qwen3_omni_point",
            model_groups=ModelGroup(
                [
                    Model("Qwen/Qwen3-Omni-30B-A3B-Instruct", "Qwen/Qwen3-Omni-30B-A3B-Instruct"),
                ]
            ),
            tags=["pointcloud", "qwen3-omni", "projection", "ae"],
            loader=Qwen3OmniPointModelLoader,
        ),
        exist_ok=exists_ok,
    )


def register_qwen3_omni_point_mlp_model(exists_ok: bool = True) -> None:
    """
    Register the new MLP baseline.

    Use with:
      --model_type qwen3_omni_point_mlp
      --template   qwen3_omni_point_cloud_mlp
    """
    register_model(
        ModelMeta(
            model_type="qwen3_omni_point_mlp",
            model_arch="qwen3_omni",
            template="qwen3_omni_point_cloud_mlp",
            model_groups=ModelGroup(
                [
                    Model("Qwen/Qwen3-Omni-30B-A3B-Instruct", "Qwen/Qwen3-Omni-30B-A3B-Instruct"),
                ]
            ),
            tags=["pointcloud", "qwen3-omni", "projection", "mlp2x_gelu"],
            loader=Qwen3OmniPointMLPModelLoader,
        ),
        exist_ok=exists_ok,
    )
