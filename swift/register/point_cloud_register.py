from __future__ import annotations

import inspect
from typing import Any, List

import torch
from torch import nn

from swift.llm import (Model, ModelGroup, ModelMeta, MultiModelKeys, get_model_tokenizer,
                       get_model_tokenizer_with_flash_attn, register_model, register_model_arch, register_template,
                       to_float_dtype)
from swift.llm.model.model.qwen import _compat_qwen3_omni_mixed_data, patch_qwen_vl_utils
from swift.llm.model.patcher import patch_get_input_embeddings
from swift.llm.model.utils import use_submodel_func
from swift.llm.template.template.qwen import Qwen3OmniTemplate, QwenTemplateMeta
from swift.utils import get_env_args


from rich.pretty import pprint
from torchinfo import summary


register_model_arch(
    MultiModelKeys(
        'my_qwen3_omni_point',
        language_model=['thinker.model', 'thinker.lm_head'],
        vision_tower=['thinker.audio_tower', 'thinker.visual', 'thinker.point_encoder'],
        aligner=[
            'thinker.audio_tower.proj1', 'thinker.audio_tower.proj2', 'thinker.visual.merger',
            'thinker.visual.merger_list', 'thinker.point_projector'
        ],
        generator=['talker', 'code2wav'],
    ))


class Qwen3OmniPointTemplate(Qwen3OmniTemplate):
    placeholder_tokens = ['<|image_pad|>', '<|audio_pad|>', '<|video_pad|>', '<|point_pad|>']


class PointCloudProjector(nn.Module):

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features)


def _normalize_points(points: Any, batch_size: int) -> List[List[torch.Tensor]]:
    # points: List[B][T][N][C], leaf: float
    # Keep first two dims as lists; convert each (N, C) to a tensor.
    assert isinstance(points, list) and len(points) == batch_size

    return [
        [torch.tensor(p_bt) for p_bt in points[b]]
        for b in range(batch_size)
    ]


def _encode_point_clouds(
    point_encoder: nn.Module,
    point_projector: nn.Module,
    points: List[List[torch.Tensor]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    flat_points: List[torch.Tensor] = [pc for sample in points for pc in sample]
    if not flat_points:
        return torch.empty((0, 1, point_projector.proj.out_features), device=device, dtype=dtype)
    point_batch = torch.stack(flat_points)
    point_batch = point_batch.to(device=device, dtype=dtype)
    point_features = point_encoder(point_batch)
    point_features = point_projector(point_features)
    return point_features


def _apply_point_embeddings(
    model: nn.Module,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    points: Any,
    point_token_id: int,
) -> torch.Tensor:
    point_mask = input_ids == point_token_id
    if not torch.any(point_mask):
        return inputs_embeds
    batch_size = input_ids.shape[0]
    point_batches = _normalize_points(points, batch_size)
    if not point_batches:
        return inputs_embeds
    point_encoder = model.point_encoder
    point_projector = model.point_projector
    encoder_device = next(point_encoder.parameters()).device
    encoder_dtype = next(point_encoder.parameters()).dtype
    point_features = _encode_point_clouds(
        point_encoder,
        point_projector,
        point_batches,
        device=encoder_device,
        dtype=encoder_dtype,
    )
    point_features = point_features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
    # import ipdb; ipdb.set_trace()

    offset = 0
    for i, sample_points in enumerate(point_batches):
        num_points = len(sample_points)
        if num_points == 0:
            continue
        positions = point_mask[i].nonzero(as_tuple=True)[0]
        if positions.numel() == 0:
            offset += num_points
            continue
        sample_features = point_features[offset:offset + num_points].squeeze(1)
        offset += num_points
        if positions.numel() == num_points:
            for j, pos in enumerate(positions.tolist()):
                inputs_embeds[i, pos] = sample_features[j]
        elif num_points == 1:
            inputs_embeds[i, positions] = sample_features[0].unsqueeze(0).expand(positions.numel(), -1)
        else:
            raise ValueError(
                f'Point tokens ({positions.numel()}) != point clouds ({num_points}). '
                'Either supply one point cloud per placeholder or one per sample.')
    return inputs_embeds


def _patch_point_forward(model: nn.Module, *, point_token_id: int) -> None:
    origin_forward = model.forward
    signature = inspect.signature(origin_forward)

    def forward(*args, points: Any = None, **kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        input_ids = bound.arguments.get('input_ids')
        inputs_embeds = bound.arguments.get('inputs_embeds')
        if points is None:
            points = bound.arguments.get('points')
        if input_ids is not None and points is not None:
            if inputs_embeds is None:
                inputs_embeds = model.get_input_embeddings()(input_ids)
            inputs_embeds = _apply_point_embeddings(
                model,
                input_ids,
                inputs_embeds,
                points,
                point_token_id,
            )
            bound.arguments['inputs_embeds'] = inputs_embeds
        bound.arguments.pop('points', None)
        return origin_forward(*bound.args, **bound.kwargs)

    model.forward = forward


def _attach_point_encoder(model, tokenizer) -> None:
    from swift.llm.model.point_cloud import PointBERTConfig, PointBERTEncoder

    point_config = PointBERTConfig()
    point_encoder = PointBERTEncoder(point_config, use_max_pool=True)
    hidden_size = model.thinker.model.embed_tokens.weight.shape[1]
    point_embed_dim = point_config.trans_dim * 2
    point_projector = PointCloudProjector(point_embed_dim, hidden_size)
    device = next(model.parameters()).device
    point_encoder = to_float_dtype(point_encoder, model.dtype).to(device=device)
    point_projector = to_float_dtype(point_projector, model.dtype).to(device=device)
    model.thinker.point_encoder = point_encoder
    model.thinker.point_projector = point_projector

    point_token_id = tokenizer.encode('<|point_pad|>', add_special_tokens=False)[0]
    _patch_point_forward(model.thinker, point_token_id=point_token_id)
    model.thinker.point_token_id = point_token_id

    ckpt_path = get_env_args('POINT_ENCODER_CKPT', str, None)
    if ckpt_path:
        model.thinker.point_encoder.load_checkpoint(ckpt_path)


def get_model_tokenizer_qwen3_omni_point(model_dir, *args, **kwargs):
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor, Qwen3OmniMoeConfig
    from qwen_omni_utils import vision_process

    kwargs['automodel_class'] = kwargs['automodel_class'] or Qwen3OmniMoeForConditionalGeneration
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_dir, trust_remote_code=True)
    kwargs['tokenizer'] = processor.tokenizer
    kwargs['model_config'] = Qwen3OmniMoeConfig.from_pretrained(model_dir, trust_remote_code=True)
    kwargs['model_config'].thinker_config.audio_token_id = processor.tokenizer.encode('<|audio_pad|>')[0]
    global_vars = patch_qwen_vl_utils(vision_process)
    processor.global_vars = global_vars
    enable_audio_output = get_env_args('ENABLE_AUDIO_OUTPUT', bool, None)
    if enable_audio_output is not None:
        kwargs['model_config'].enable_audio_output = enable_audio_output
    model, _ = get_model_tokenizer_with_flash_attn(model_dir, *args, **kwargs)
    if model:
        _compat_qwen3_omni_mixed_data(model.thinker, processor)
        base_model = model.model if 'AWQ' in model.__class__.__name__ else model
        use_submodel_func(base_model, 'thinker')
        base_model.config.keys_to_ignore_at_inference += ['hidden_states', 'attention_mask']
        base_model.config.talker_config.pad_token_id = None
        patch_get_input_embeddings(base_model.thinker.visual, 'patch_embed')
        patch_get_input_embeddings(base_model.thinker.audio_tower, 'conv_out')
        _attach_point_encoder(base_model, processor.tokenizer)
    return model, processor


register_model(
    ModelMeta(
        'my_qwen3_omni_point',
        [
            ModelGroup([
                Model('Qwen/Qwen3-Omni-30B-A3B-Instruct', 'Qwen/Qwen3-Omni-30B-A3B-Instruct'),
                Model('Qwen/Qwen3-Omni-30B-A3B-Thinking', 'Qwen/Qwen3-Omni-30B-A3B-Thinking'),
                Model('Qwen/Qwen3-Omni-30B-A3B-Captioner', 'Qwen/Qwen3-Omni-30B-A3B-Captioner'),
            ]),
        ],
        'my_qwen3_omni_point',
        get_model_tokenizer_qwen3_omni_point,
        is_multimodal=True,
        model_arch='my_qwen3_omni_point',
        architectures=['Qwen3OmniMoeForConditionalGeneration'],
        requires=['transformers>=4.57.dev0', 'soundfile', 'decord', 'qwen_omni_utils'],
        tags=['vision', 'video', 'audio', 'point-cloud'],
    ))


register_template(
    QwenTemplateMeta(
        'my_qwen3_omni_point',
        template_cls=Qwen3OmniPointTemplate,
        default_system=None,
        thinking_prefix='<think>\n',
    ))


if __name__ == '__main__':
    model, processor = get_model_tokenizer('Qwen/Qwen3-Omni-30B-A3B-Instruct', model_type='my_qwen3_omni_point',use_hf=True)
    print(f'Loaded model: {type(model)}')
