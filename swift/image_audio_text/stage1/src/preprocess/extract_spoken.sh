#!/usr/bin/env bash
set -euo pipefail

BASE="${SLURM_TMPDIR:-/tmp}/${USER}/miopen_${SLURM_JOB_ID:-$$}"
mkdir -p "${BASE}/miopen_user_db" "${BASE}/miopen_cache"

export MIOPEN_USER_DB_PATH="${BASE}/miopen_user_db"
export MIOPEN_CUSTOM_CACHE_DIR="${BASE}/miopen_cache"

export MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}
echo "Using MASTER_PORT=$MASTER_PORT"

export MM_CFG="/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/swift/image_audio_text/stage1/configs/extract_spokencoco_features.yaml"
python -m swift.image_audio_text.stage1.src.preprocess.extract_spoken --config "$MM_CFG"
