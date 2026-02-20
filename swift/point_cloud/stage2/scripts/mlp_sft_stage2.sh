#!/bin/bash
set -euo pipefail

export POINT_FEATURE_DATASET_INFO_YAML=/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/data_features_cleaned_24/dataset_info.yaml
export POINT_CONV_JSON_PATH=/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/PointLLM/PointLLM/PointLLM_brief_description_660K_filtered.json

# MLP baseline 不需要 AE ckpt
# export POINT_AE_CKPT_PATH=/path/to/stage1/best.pt

# 可选
export POINT_MAX_INJECT_TOKENS=24
export POINT_REQUIRE_VALID=1
export POINTCLOUD_CACHE_PER_RANK=1

MODEL_DIR="/vast/users/guangyi.chen/.cache/huggingface/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695"

# 单节点多卡
nproc_per_node=4

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=$nproc_per_node \
swift sft \
  --model "${MODEL_DIR}" \
  --model_type qwen3_omni_point_mlp \
  --template qwen3_omni_point_cloud_mlp \
  --dataset pointcloud_feature_sft#6500 \
  --external_plugins /vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/swift/point_cloud/stage2/src/pc_register.py \
  --streaming False \
  --split_dataset_ratio 0.01 \
  --remove_unused_columns False \
  --output_dir /vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/checkpoints_mlp_single \
  --tuner_type full \
  --num_train_epochs 1 \
  --torch_dtype bfloat16 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --attn_impl flash_attn \
  --packing false \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing false \
  --logging_steps 5 \
  --warmup_ratio 0.05 \
  --learning_rate 5e-5 \
  --freeze_llm True \
  --freeze_parameters_regex '^(?!point_projector\.).*' \
  --trainable_parameters_regex '^point_projector\.' \
  --dataset_num_proc 8 \
  --dataloader_num_workers 8 \
  --strict False
