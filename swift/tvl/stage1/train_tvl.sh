#!/usr/bin/env bash
set -euo pipefail

# 1) 建议使用本地临时盘缓存 MIOpen DB
BASE="${SLURM_TMPDIR:-/tmp}/${USER}/miopen_${SLURM_JOB_ID:-$$}"
mkdir -p "${BASE}/db"

# 2) 让 MIOpen 的 SQLite User DB 写到本地临时目录
export MIOPEN_USER_DB_PATH="${BASE}/db"
export MIOPEN_DEBUG_DISABLE_SQL_WAL=1
export MIOPEN_DISABLE_CACHE=1

CONFIG=${1:-swift/point_cloud/stage1/configs/extract_tvl_features.yaml}
NPROC=${NPROC_PER_NODE:-8}
USE_TORCHRUN=${USE_TORCHRUN:-0}

if [[ "${USE_TORCHRUN}" == "1" ]]; then
  MM_CFG="${CONFIG}" torchrun --standalone --nproc_per_node="${NPROC}" -m swift.point_cloud.stage1.src.preprocess.extract_tvl_features
else
  MM_CFG="${CONFIG}" python -m swift.point_cloud.stage1.src.preprocess.extract_tvl_features
fi
