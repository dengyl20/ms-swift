from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from swift.llm.model.point_cloud.point_bert import PointBERTConfig, PointBERTEncoder

from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3OmniMoeThinkerForConditionalGeneration


def _parse_torch_dtype(s: str) -> torch.dtype:
    s = (s or "fp16").lower()
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp32", "float32", "float"):
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {s}")


class FrozenPointBERTTokens(nn.Module):
    """
    输入 raw points: (B,8192,6)
    输出 tokens:
      - 若 drop_cls=True:  (B, num_group, trans_dim)  默认 num_group=512
      - 否则:             (B, num_group+1, trans_dim)
    """

    def __init__(self, cfg: Dict[str, Any], device: torch.device):
        super().__init__()
        self.device = device

        ckpt_path = cfg["ckpt_path"]
        drop_cls = bool(cfg.get("drop_cls", True))
        self.drop_cls = drop_cls

        input_dtype = (cfg.get("input_dtype", "fp32") or "fp32").lower()
        if input_dtype not in ("fp16", "fp32", "bf16"):
            raise ValueError("point_bert.input_dtype must be one of: fp16, bf16, fp32")
        self.input_dtype = input_dtype

        cfg_dict = cfg.get("config", {})
        pb_cfg = PointBERTConfig(**cfg_dict)

        # use_max_pool 随便；我们 forward 明确 return_tokens=True
        self.encoder = PointBERTEncoder(pb_cfg, use_max_pool=False)
        self.encoder.load_checkpoint(ckpt_path, strict=True, map_location="cuda", verbose=True)

        # freeze
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.encoder.to(device)

        self.trans_dim = int(pb_cfg.trans_dim)
        self.num_group = int(pb_cfg.num_group)

    @torch.no_grad()
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        points: (B,8192,6) float
        """
        points = points.to(self.device, non_blocking=True)

        # dtype control（优先稳定性）
        if self.input_dtype == "fp16":
            points = points.half()
        elif self.input_dtype == "bf16":
            points = points.bfloat16()
        else:
            points = points.float()

        tokens = self.encoder(points, return_tokens=True)  # (B,G+1,trans_dim)
        if self.drop_cls:
            tokens = tokens[:, 1:, :]  # (B,G,trans_dim)
        return tokens


class FrozenQwenEmbeddingTable(nn.Module):
    """
    给定 texts(list[str]) -> (embeddings, mask)
      embeddings: (B, max_len, hidden)  hidden=2048（按你设定）
      mask:       (B, max_len) bool
    """

    def __init__(self, cfg: Dict[str, Any], device: torch.device):
        super().__init__()
        self.device = device

        model_name = cfg["model_name_or_path"]
        tok_name = cfg.get("tokenizer_name_or_path", model_name)
        trust_remote_code = bool(cfg.get("trust_remote_code", True))
        torch_dtype = _parse_torch_dtype(cfg.get("torch_dtype", "fp16"))

        self.max_text_len = int(cfg.get("max_text_len", 128))
        self.add_special_tokens = bool(cfg.get("add_special_tokens", False))

        self.tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=trust_remote_code)

        # pad token 处理：很多 LLM tokenizer 默认没有 pad_token
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # 最保守做法：强行设为 0（但你要保证 0 是有效 token）
                self.tokenizer.pad_token = self.tokenizer.convert_ids_to_tokens(0)

        extract_and_discard = bool(cfg.get("extract_embedding_and_discard_model", True))

        # 1) 加载全模型 -> 2) 提取 embedding weight -> 3) 可选丢弃全模型仅保留 nn.Embedding
        model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map="auto",  # 不强依赖 accelerate
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        emb_layer = model.get_input_embeddings()
        weight = emb_layer.weight.detach().clone()  # clone: 允许 del model 后仍保留权重

        if extract_and_discard:
            del model
            # 注意：这里只释放 Python 引用；如在 CUDA 上加载过，可额外 empty_cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.embed = nn.Embedding.from_pretrained(weight, freeze=True)
        else:
            # 保留原模型 embedding（仍然不训练）
            self.embed = emb_layer

        self.embed.to(device)
        self.embed.eval()
        for p in self.embed.parameters():
            p.requires_grad = False

        self.hidden_size = int(weight.shape[1])

    @torch.no_grad()
    def forward(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device, non_blocking=True)
        attn_mask = enc["attention_mask"].to(self.device, non_blocking=True).bool()

        emb = self.embed(input_ids)  # (B, max_len, hidden)
        return emb, attn_mask
