#!/bin/bash
#SBATCH --output=slurm_logs/%x_%j.out

#SBATCH --job-name=injecagent
#SBATCH --account=bfzb-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1       
#SBATCH --cpus-per-task=32    
#SBATCH --gpus-per-node=1        
#SBATCH --mem=120g           
#SBATCH --time=48:00:00

# Parameter parsing (must match run_injecagent.py SLURM call order)
DATASET=$1
BACKEND_LLM=$2
ATTACK=$3
DEFENSE=$4
ATTACK_PATH=$5
NAME=$6
SEED=$7
LOG_FILE=$8
MONITOR_LLM=$9

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate autodanturbo

export HF_HOME="${HF_HOME:-/work/nvme/bfzb/hf_cache}"
: "${HF_TOKEN:?Set HF_TOKEN before running}"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=lo
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1

python -u main_injecagent.py \
  --model "${BACKEND_LLM}" \
  --defense "${DEFENSE}" \
  --monitor_llm "${MONITOR_LLM}" \
  --name "${NAME}" \
  --seed "${SEED}" \
  > "${LOG_FILE}" 2>&1
