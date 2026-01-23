export POINT_FEATURE_DATASET_INFO_YAML=/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/data_features_cleaned_24/dataset_info.yaml
export POINT_CONV_JSON_PATH=/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/PointLLM/PointLLM/PointLLM_brief_description_660K_cleaned.json
export POINT_AE_CKPT_PATH=/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/checkpoints/cleaned_maxlen_24/best.pt

# 可选
export POINT_MAX_INJECT_TOKENS=24
export POINT_REQUIRE_VALID=1



swift sft \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --model_type qwen3_omni_point \
  --template qwen3_omni_point \
  --dataset pointcloud_feature_sft \
  --external_plugins /vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/swift/point_cloud/stage2/src/pc_register.py \
  --streaming True \
  --split_dataset_ratio 0 \
  --remove_unused_columns False \
  --output_dir /vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/checkpoints \
  --tuner_type full \
  --num_train_epochs 1 \
  --batch_size 1
