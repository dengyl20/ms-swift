# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Point cloud inference demo for a custom Qwen3-Omni + PointBERT model.

This script:
1) Imports your custom model registration module.
2) Loads one sample from the point cloud dataset.
3) Initializes the point cloud encoder from a pretrained checkpoint.
4) Runs inference with the modified Qwen3-Omni model.
"""
import argparse
import importlib
import os
from typing import Optional

import torch

from swift.llm import InferRequest, PtEngine, RequestConfig, load_dataset

import swift.register.point_cloud_register


DEFAULT_POINT_BERT_CKPT = (
    '/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/'
    'PointLLM/PointLLM_7B_v1.1_init/point_bert_v1.2.pt'
)


def _import_register_module(module_path: Optional[str]) -> None:
    if not module_path:
        return
    try:
        importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f'Failed to import register module: {module_path}. '
            'Make sure the module path is correct and available on PYTHONPATH.'
        ) from exc


def _find_point_encoder(model: torch.nn.Module) -> Optional[torch.nn.Module]:
    candidate_attrs = (
        'point_encoder',
        'point_cloud_encoder',
        'point_bert',
        'pointbert',
    )
    for attr in candidate_attrs:
        if hasattr(model, attr):
            encoder = getattr(model, attr)
            if hasattr(encoder, 'load_checkpoint'):
                return encoder
    for module in model.modules():
        if module is model:
            continue
        if hasattr(module, 'load_checkpoint') and 'point' in module.__class__.__name__.lower():
            return module
    return None


def _load_point_encoder_weights(model: torch.nn.Module, ckpt_path: str) -> None:
    encoder = _find_point_encoder(model)
    if encoder is None:
        raise RuntimeError(
            'Could not locate a point cloud encoder with load_checkpoint(). '
            'Please verify your custom Qwen3-Omni registration exposes the encoder '
            'as an attribute like `point_encoder` or `point_bert`.'
        )
    encoder.load_checkpoint(ckpt_path)


def _build_dataset_syntax(args: argparse.Namespace) -> str:
    tokens = [args.dataset]

    if args.data_path:
        tokens.append(f'data_path={args.data_path}')
    if args.anno_path:
        tokens.append(f'anno_path={args.anno_path}')
    if args.split:
        tokens.append(f'split={args.split}')
    if args.pointnum:
        tokens.append(f'pointnum={args.pointnum}')
    return ':'.join(tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description='Point cloud inference demo (Qwen3-Omni + PointBERT).')
    parser.add_argument('--register-module', default='swift.register.point_cloud_register')
    parser.add_argument('--model-id', default='Qwen/Qwen3-Omni-30B-A3B-Instruct')
    parser.add_argument('--model-type', default='my_qwen3_omni_point', help='Custom model_type registered in ms-swift.')
    parser.add_argument('--point-bert-ckpt', default=DEFAULT_POINT_BERT_CKPT)
    parser.add_argument('--dataset', default='pointllm_point_cloud', help='Full dataset syntax string.')
    parser.add_argument('--data-path', default='/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/PointLLM/PointLLM/8192_npy')
    parser.add_argument('--anno-path', default='/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/PointLLM/PointLLM/PointLLM_brief_description_660K.json')
    parser.add_argument('--split', default='train')
    parser.add_argument('--pointnum', default=8192, type=int)
    parser.add_argument('--max-tokens', default=256, type=int)
    parser.add_argument('--temperature', default=0.0, type=float)
    parser.add_argument('--attn-impl', default='flash_attention_2')
    parser.add_argument('--device-map', default='auto')
    args = parser.parse_args()

    # _import_register_module(args.register_module)

    # dataset_syntax = _build_dataset_syntax(args)
    os.environ['POINT_CLOUD_DATA_PATH'] = args.data_path
    os.environ['POINT_CLOUD_ANNO_PATH'] = args.anno_path

    dataset = load_dataset([args.dataset], seed=42, streaming=True, remove_unused_columns=False)[0]
    for idx, ex in enumerate(dataset):
        sample = ex
        print(sample['messages'][1]['content'])
        print(len(sample['messages'][1]['content']))
        if idx > 30:
            break

    # print('=== Input ===')
    # # import ipdb; ipdb.set_trace()
    # print(sample.keys())
    # print(sample['messages'])

    # engine = PtEngine(
    #     args.model_id,
    #     model_type=args.model_type,
    #     attn_impl=args.attn_impl,
    #     device_map=args.device_map,
    #     use_hf=True,
    #     download_model=False,
    # )

    # _load_point_encoder_weights(engine.model, args.point_bert_ckpt)

    # infer_request = InferRequest(messages=sample['messages'], points=[sample['points']])
    # request_config = RequestConfig(max_tokens=args.max_tokens, temperature=args.temperature)
    # responses = engine.infer([infer_request], request_config)

    # response = responses[0].choices[0].message.content

    # print('=== Output ===')
    # print(response)


if __name__ == '__main__':
    main()
