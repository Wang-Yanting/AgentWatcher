#!/bin/bash
#SBATCH --output=slurm_logs/%x_%j.out

#SBATCH --job-name=pytorch
#SBATCH --account=bfzb-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1       
#SBATCH --cpus-per-task=32    
#SBATCH --gpus-per-node=1        
#SBATCH --mem=120g           
#SBATCH --time=48:00:00


# 参数解析
DATASET=$1
BACKEND_LLM=$2
ATTACK=$3
DEFENSE=$4
ATTACK_PATH=$5
NAME=$6
SEED=$7
LOG_FILE=$8
MONITOR_LLM=$9

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

# 加载环境
source ~/.bashrc
conda activate piarena # 替换为你的conda环境
export HF_HOME="${HF_HOME:-/work/nvme/bfzb/hf_cache}"
: "${HF_TOKEN:?Set HF_TOKEN before running}"

# 后台执行并写入日志
python3 -u main.py \
  --dataset "${DATASET}" \
  --backend_llm "${BACKEND_LLM}" \
  --monitor_llm "${MONITOR_LLM}" \
  --attack "${ATTACK}" \
  --defense "${DEFENSE}" \
  --use_vllm \
  --attack_path "${ATTACK_PATH}" \
  --name "${NAME}" \
  --seed "${SEED}" \
  > "${LOG_FILE}" 2>&1
