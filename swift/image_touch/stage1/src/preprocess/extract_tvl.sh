#!/usr/bin/env bash
set -euo pipefail

# 1) 选一个“本地磁盘/临时盘”的可写目录（示例用 /tmp；如果你是 slurm，优先用 $SLURM_TMPDIR）
BASE="${SLURM_TMPDIR:-/tmp}/${USER}/miopen_${SLURM_JOB_ID:-$$}"

mkdir -p "${BASE}/miopen_user_db"
mkdir -p "${BASE}/miopen_cache"

# 2) 让 MIOpen 的 SQLite User DB 写到这里（核心）
export MIOPEN_USER_DB_PATH="${BASE}/miopen_user_db"
export MIOPEN_CUSTOM_CACHE_DIR="${BASE}/miopen_cache"
# export MIOPEN_DEBUG_DISABLE_SQL_WAL=1
# export MIOPEN_DISABLE_CACHE=1

export MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}
echo "Using MASTER_PORT=$MASTER_PORT"

# 3) 运行你的训练
export MM_CFG="/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/swift/image-touch/stage1/configs/extract_tvl_features.yaml"

torchrun --master_port "$MASTER_PORT" --nproc_per_node=8 -m swift.image-touch.stage1.src.preprocess.extract_tvl_features

