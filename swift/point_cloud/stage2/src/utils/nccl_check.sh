#!/bin/bash
#SBATCH -J point_sft
#SBATCH -p faculty
#SBATCH -A faculty-acc
#SBATCH --qos=bgqos
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:mi210:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=512G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH -o /vast/users/guangyi.chen/causal_group/yunlong.deng/slurm_tools/logs_nccl/%x.%j.out

set -euo pipefail

# --- conda 基路径：按你的实际安装路径 ---
export CONDA_BASE=/vast/users/guangyi.chen/miniconda3

# ===== 你的环境变量 =====
export POINT_FEATURE_DATASET_INFO_YAML=/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/data_features_cleaned_24/dataset_info.yaml
export POINT_CONV_JSON_PATH=/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/PointLLM/PointLLM/PointLLM_brief_description_660K_filtered.json
export POINT_AE_CKPT_PATH=/vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/checkpoints/cleaned_maxlen_24/best.pt

export POINT_MAX_INJECT_TOKENS=24
export POINT_REQUIRE_VALID=1
export POINTCLOUD_CACHE_PER_RANK=1

MODEL_DIR="/vast/users/guangyi.chen/.cache/huggingface/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695"

# ===== 多机 torchrun 关键环境变量 =====
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NNODES="${SLURM_JOB_NUM_NODES}"
export NPROC_PER_NODE=4
export MASTER_PORT=29500
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "NNODES=${NNODES}, NPROC_PER_NODE=${NPROC_PER_NODE}"


# ===== NCCL/RCCL 通信配置：先用 TCP 验证，再切回 IB =====

# --- 方案A：稳定优先（TCP） ---
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
# export NCCL_SOCKET_IFNAME="=enp137s0f0"   # TODO: 改成你们节点间互通网卡名

# --- 方案B：性能优先（IB/RoCE）---
# export NCCL_IB_DISABLE=0
# export NCCL_NET=IB
# export NCCL_IB_HCA="=rocep137s0f0:1"
# export NCCL_IB_GID_INDEX=3              # TODO: 用 show_gids 查到后填写

# 调试信息
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG_FILE=/vast/users/guangyi.chen/causal_group/yunlong.deng/slurm_tools/logs_nccl/nccl.%h.%p.log
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576


srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 --kill-on-bad-exit=1 \
  bash -lc '
    # 关键：加载 conda hook（不要 conda init）
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate qwen3omni
    export NODE_RANK=$SLURM_PROCID
    torchrun --nnodes='"$SLURM_JOB_NUM_NODES"' --node_rank=$NODE_RANK \
      --nproc_per_node=1 --master_addr='"$MASTER_ADDR"' --master_port='"$MASTER_PORT"' \
      /vast/users/guangyi.chen/causal_group/yunlong.deng/Multimodal/ms-swift/swift/point_cloud/stage2/src/utils/nccl_sanity.py
  '
