# 1) 选一个“本地磁盘/临时盘”的可写目录（示例用 /tmp；如果你是 slurm，优先用 $SLURM_TMPDIR）
BASE="${SLURM_TMPDIR:-/tmp}/${USER}/miopen_${SLURM_JOB_ID:-$$}"

mkdir -p "${BASE}/db" 

# 2) 让 MIOpen 的 SQLite User DB 写到这里（核心）
export MIOPEN_USER_DB_PATH="${BASE}/db"

export MIOPEN_DEBUG_DISABLE_SQL_WAL=1
export MIOPEN_DISABLE_CACHE=1



# 4) 运行你的训练
python -m swift.point_cloud.stage1.train_swift
