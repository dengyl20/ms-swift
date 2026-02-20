#!/usr/bin/env bash
set -euo pipefail

BASE="${SLURM_TMPDIR:-/tmp}/${USER}/miopen_${SLURM_JOB_ID:-$$}"
mkdir -p "${BASE}/db" "${BASE}/logs"

export MIOPEN_USER_DB_PATH="${BASE}/db"
export MIOPEN_DEBUG_DISABLE_SQL_WAL=1
export MIOPEN_DISABLE_CACHE=1

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/logs/tvl/stage1_yan/train_${TS}.txt"
mkdir -p "$(dirname "${LOG_FILE}")"

echo "BASE=${BASE}"
echo "LOG_FILE=${LOG_FILE}"

script -qfc 'torchrun --standalone --nproc_per_node=8 -m swift.tvl.stage1.train_touch' /dev/null | tee -a "${LOG_FILE}"


