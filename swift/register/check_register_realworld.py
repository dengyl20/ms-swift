import os
import sys
from typing import Any

import torch
from modelscope import snapshot_download
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

from swift.llm import InferRequest, PtEngine, RequestConfig

import swift.register.point_cloud_register


def _build_point_cloud(num_points: int = 256) -> torch.Tensor:
    # XYZRGB with normalized values.
    return torch.rand(num_points, 6)


def _move_points(points: Any, device: torch.device, dtype: torch.dtype) -> Any:
    if torch.is_tensor(points):
        return points.to(device=device, dtype=dtype)
    if isinstance(points, (list, tuple)):
        return [p.to(device=device, dtype=dtype) if torch.is_tensor(p) else p for p in points]
    return points


def infer_hf():
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        'Qwen/Qwen3-Omni-30B-A3B-Instruct',
        torch_dtype='auto',
        device_map='auto',
        attn_implementation='flash_attention_2',
        trust_remote_code=True,
        local_files_only=True
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained('Qwen/Qwen3-Omni-30B-A3B-Instruct', trust_remote_code=True, local_files_only=True)

    point_cloud = _build_point_cloud()
    conversation = [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'point',
                    'point': point_cloud,
                },
                {
                    'type': 'text',
                    'text': '请描述该点云包含的物体。',
                },
            ],
        },
    ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, points=[point_cloud], return_tensors='pt', padding=True)
    inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    if 'points' in inputs:
        inputs['points'] = _move_points(inputs['points'], model.device, model.dtype)

    generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    print(generated_ids)

    response = processor.batch_decode(
        generated_ids[:, inputs['input_ids'].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    # response = processor.batch_decode(
    #     generated_ids[:, inputs['input_ids'].shape[1]:],
    #     skip_special_tokens=True,
    #     clean_up_tokenization_spaces=False,
    # )[0]
    return inputs['input_ids'][0].tolist(), response


def infer_swift():
    engine = PtEngine(
        'Qwen/Qwen3-Omni-30B-A3B-Instruct',
        model_type='my_qwen3_omni_point',
        attn_impl='flash_attention_2',
        use_hf=True
    )
    point_cloud = _build_point_cloud()
    infer_request = InferRequest(
        messages=[{
            'role': 'user',
            'content': '<point>请描述该点云包含的物体。',
        }],
        points=[point_cloud],
    )
    request_config = RequestConfig(temperature=0, max_tokens=256)
    input_ids = engine.default_template.encode(infer_request)['input_ids']
    print(input_ids)
    resp_list = engine.infer([infer_request], request_config)
    resp = resp_list[0].choices[0].message.content
    return input_ids, resp


if __name__ == '__main__':
    # Enable debug mode to print input_ids and generate_ids from PtEngine.infer
    os.environ['SWIFT_DEBUG'] = '1'
    input_ids_hf, response_hf = infer_hf()
    input_ids_swift, response_swift = infer_swift()
    assert input_ids_hf == input_ids_swift
    assert response_hf == response_swift
