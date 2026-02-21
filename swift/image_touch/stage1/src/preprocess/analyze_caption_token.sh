python -m swift.point_cloud.stage1.src.preprocess.analyze_caption_token \
  --yaml /vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/data_features_cleaned/dataset_info.yaml \
  --require-valid 1 \
  --chunk 8192 \
  --Ls 4,8,16,24,32,40,48,64,80,96,128 \
  --outdir caption_length_stats
