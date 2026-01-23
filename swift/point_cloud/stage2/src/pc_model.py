# pc_model.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch
from transformers import PretrainedConfig

from swift.point_cloud.stage2.src.pc_constants import ENV_AE_CKPT_PATH, POINT_TOKEN


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


class Qwen3OmniPointModelLoader(ModelLoader):
    """
    自定义 model loader：
    - 使用 Transformers 加载 Qwen3Omni Thinker（文本模型即可）
    - 加入 <point> special token + resize embedding
    - 加载 stage1 AE，并把除 AE 外的所有参数冻结
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
        """
        ms-swift register.py 调用形态（你堆栈已证实）：
            get_model(model_dir, config, processor, model_kwargs_dict)

        因此这里必须显式接收 processor 作为第三个位置参数。
        同时做一些版本/错误调用的兼容。
        """
        global _ADDED_TOKENIZER_LEN, _POINT_TOKEN_ID, _OLD_POINT_IDS

        if not load_model:
            return None

        model_kwargs = model_kwargs or {}
        if not isinstance(model_kwargs, dict):
            raise TypeError(
                f"[Qwen3OmniPoint] model_kwargs must be a dict, but got: {type(model_kwargs)}. "
                "This usually means get_model() signature doesn't match ms-swift calling convention."
            )

        # 你推理脚本用的是 ThinkerForConditionalGeneration（text-only），这里沿用以省显存
        from transformers import Qwen3OmniMoeThinkerForConditionalGeneration

        # 透传 HF from_pretrained 关键参数（上游如果传了就带上）
        hf_kwargs: Dict[str, Any] = {}

        # 优先从 model_kwargs 拿（有些 swift 版本会放这里）
        for k in [
            "torch_dtype",
            "device_map",
            "attn_implementation",
            "trust_remote_code",
            "low_cpu_mem_usage",
        ]:
            if k in model_kwargs:
                hf_kwargs[k] = model_kwargs[k]

        # 兼容：有人可能传 dtype（而 HF 用 torch_dtype）
        if "torch_dtype" not in hf_kwargs:
            if "dtype" in model_kwargs:
                hf_kwargs["torch_dtype"] = model_kwargs["dtype"]
            elif "dtype" in kwargs:
                hf_kwargs["torch_dtype"] = kwargs["dtype"]

        # 也允许 kwargs 直接覆盖
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

        # ===== resize embedding（必须做，否则新 token id 会越界）=====
        if _ADDED_TOKENIZER_LEN is not None:
            emb = model.get_input_embeddings()
            if emb.weight.shape[0] != int(_ADDED_TOKENIZER_LEN):
                model.resize_token_embeddings(int(_ADDED_TOKENIZER_LEN))
                logger.info(f"[Qwen3OmniPoint] Resized token embeddings to {_ADDED_TOKENIZER_LEN}")

                # 初始化新 token embedding：用加入前的 old ids embedding 均值
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

        # ===== 加载你的 point AE 并挂到模型上 =====
        ae_ckpt = os.environ.get(ENV_AE_CKPT_PATH, None)
        if not ae_ckpt:
            raise ValueError(f"Missing env var: export {ENV_AE_CKPT_PATH}=/path/to/stage1/best.pt")

        ckpt = _torch_load_ckpt(ae_ckpt)
        if "cfg" not in ckpt or "model" not in ckpt:
            raise RuntimeError(f"AE ckpt format unexpected: keys={list(ckpt.keys())}")

        cfg_model = ckpt["cfg"]["model"]

        # 复用你现有实现（包含 layers 等）
        from swift.point_cloud.stage1.src.models.unified_ae import UnifiedPointTextAE

        point_ae = UnifiedPointTextAE(cfg_model)
        point_ae.load_state_dict(ckpt["model"], strict=True)

        # 放到 embedding device/dtype
        emb_layer = model.get_input_embeddings()
        point_ae.to(device=emb_layer.weight.device, dtype=emb_layer.weight.dtype)

        # 维度一致性检查
        llm_hidden = int(emb_layer.weight.shape[1])
        ae_hidden = int(cfg_model["d_text_in"])
        if ae_hidden != llm_hidden:
            raise ValueError(f"Hidden mismatch: AE d_text_in={ae_hidden} vs Qwen hidden={llm_hidden}")

        model.point_ae = point_ae
        model.point_token_id = int(_POINT_TOKEN_ID) if _POINT_TOKEN_ID is not None else None

        # ===== 冻结：只训练 AE =====
        for p in model.parameters():
            p.requires_grad = False
        for p in model.point_ae.parameters():
            p.requires_grad = True

        # 常见训练设置：避免 use_cache 占显存
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.warning(f"[Qwen3OmniPoint] Model ready. trainable={trainable} / total={total}")

        return model


def register_qwen3_omni_point_model(exists_ok: bool = True) -> None:
    register_model(
        ModelMeta(
            model_type="qwen3_omni_point",
            # 这里尽量复用 ms-swift 已有的 arch 名称；如果你环境里 arch 名不同，你只要改这一行
            model_arch="qwen3_omni",
            template="qwen3_omni_point",
            model_groups=ModelGroup(
                [
                    Model("Qwen/Qwen3-Omni-30B-A3B-Instruct", "Qwen/Qwen3-Omni-30B-A3B-Instruct"),
                ]
            ),
            tags=["pointcloud", "qwen3-omni", "projection"],
            loader=Qwen3OmniPointModelLoader,
        ),
        exist_ok=exists_ok,
    )
