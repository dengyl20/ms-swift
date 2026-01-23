from typing import List
import os
from rich.pretty import pprint

from swift.dataset import load_dataset


def _test_dataset(datasets: List[str], num_proc: int = 1, strict: bool = False, **kwargs):
    dataset = load_dataset(datasets, streaming=True, num_proc=num_proc, strict=strict, **kwargs)

    print(f'dataset[0]: {dataset[0]}')

    ds, _ = dataset

    for i, example in enumerate(ds):
        # example 是一个 dict，包含 'messages' / 'points'
        # pprint(i, example.keys())
        pprint(example['points'])
        pprint(example['messages'])
        
        if i >= 9:
            break



def test_sft():
    # swift/SlimOrca  swift/cosmopedia-100k
    # _test_dataset(['lvjianjin/AdvertiseGen'])
    # _test_dataset(['AI-ModelScope/Duet-v0.5'])
    # _test_dataset(['swift/SlimOrca', 'swift/cosmopedia-100k'])
    # _test_dataset(['OmniData/Zhihu-KOL-More-Than-100-Upvotes'])
    # _test_dataset(['OmniData/Zhihu-KOL'])
    _test_dataset([
        'pointllm_point_cloud'
    ])
    # _test_dataset(['swift/Infinity-Instruct:all'])
    # _test_dataset(['swift/sharegpt:all'])
    # _test_dataset(['AI-ModelScope/sharegpt_gpt4:all'])
    # _test_dataset(['iic/ms_bench'])
    # _test_dataset(['swift/tagengo-gpt4'])


def test_mllm():
    # _test_dataset(['AI-ModelScope/ShareGPT4V:all'])
    # _test_dataset(['AI-ModelScope/LLaVA-Pretrain'])
    # _test_dataset(['swift/TextCaps'])
    # _test_dataset(['swift/RLAIF-V-Dataset:all'])
    # _test_dataset(['swift/OK-VQA_train'])
    # _test_dataset(['swift/OCR-VQA'])
    # _test_dataset(['swift/A-OKVQA'])
    # _test_dataset(['AI-ModelScope/MovieChat-1K-test'])
    _test_dataset([
        'AI-ModelScope/LaTeX_OCR:all', 'modelscope/coco_2014_caption:validation',
        'speech_asr/speech_asr_aishell1_trainsets:validation'
    ],
                  strict=False)
    # _test_dataset(['swift/VideoChatGPT:all'])
    # _test_dataset(['speech_asr/speech_asr_aishell1_trainsets:validation'])
    # _test_dataset(['AI-ModelScope/captcha-images'])
    # _test_dataset(['swift/gpt4v-dataset:all'])
    # _test_dataset(['modelscope/coco_2014_caption:validation'])
    # _test_dataset(['AI-ModelScope/LLaVA-Instruct-150K'], num_proc=16)


def test_agent():
    _test_dataset(['swift/ToolBench'])
    # _test_dataset(['AI-ModelScope/ms_agent_for_agentfabric:all'])


def test_dpo():
    _test_dataset(['AI-ModelScope/orpo-dpo-mix-40k'])
    _test_dataset(['AI-ModelScope/hh-rlhf:all'])
    _test_dataset(['AI-ModelScope/hh_rlhf_cn:all'])
    _test_dataset(['hjh0119/shareAI-Llama3-DPO-zh-en-emoji:all'])


def test_kto():
    _test_dataset(['AI-ModelScope/ultrafeedback-binarized-preferences-cleaned-kto'])


def test_pretrain():
    _test_dataset(['AI-ModelScope/ruozhiba:all'])


def test_dataset_info():
    _test_dataset(['swift/self-cognition#500'], model_name='xiao huang', model_author='swift')
    # _test_dataset(['codefuse-ai/CodeExercise-Python-27k'])


def test_cls():
    _test_dataset(['simpleai/HC3-Chinese:baike'])
    _test_dataset(['simpleai/HC3-Chinese:baike_cls'])


if __name__ == '__main__':
    # 1) 设置 env（你的 load_pointcloud_dataset 支持 env 获取路径）
    DATA_PATH = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/PointLLM/PointLLM/8192_npy"
    ANNO_PATH = "/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/PointLLM/PointLLM/PointLLM_brief_description_660K.json"
    os.environ["POINT_CLOUD_DATA_PATH"] = DATA_PATH
    os.environ["POINT_CLOUD_ANNO_PATH"] = ANNO_PATH
    test_sft()
    # test_agent()
    # test_dpo()
    # test_kto()
    # test_mllm()
    # test_pretrain()
    # test_dataset_info()
    # test_cls()
