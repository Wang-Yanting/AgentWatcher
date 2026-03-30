#!/bin/bash
#SBATCH --output=slurm_logs/%x_%j.out

#SBATCH --job-name=agentdyn
#SBATCH --account=bfzb-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=120g
#SBATCH --time=48:00:00
#SBATCH --account=bfzb-dtai-gh
# Parameter parsing for main_agentdyn.py
MODEL=$1
ATTACK=$2
DEFENSE=$3
SUITE=$4
TENSOR_PARALLEL_SIZE=$5
NAME=$6
LOG_FILE=$7
MONITOR_LLM=$8
W_S=$9
W_L=${10}
W_R=${11}
ATTR_K=${12}
ATTRIBUTION_MODEL=${13}

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

export HF_HOME="${HF_HOME:-/work/nvme/bfzb/hf_cache}"
: "${HF_TOKEN:?Set HF_TOKEN before running}"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running}"

source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate agentdojo

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=lo
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1

CMD="python -u main_agentdyn.py --model \"${MODEL}\" --attack \"${ATTACK}\" --defense \"${DEFENSE}\" --tensor_parallel_size \"${TENSOR_PARALLEL_SIZE}\" --name \"${NAME}\""

if [ "${SUITE}" != "all" ] && [ -n "${SUITE}" ]; then
    CMD="${CMD} --suite \"${SUITE}\""
fi

if [ -n "${MONITOR_LLM}" ]; then
    CMD="${CMD} --monitor_llm \"${MONITOR_LLM}\""
fi

if [ -n "${W_S}" ]; then
    CMD="${CMD} --w_s ${W_S}"
fi
if [ -n "${W_L}" ]; then
    CMD="${CMD} --w_l ${W_L}"
fi
if [ -n "${W_R}" ]; then
    CMD="${CMD} --w_r ${W_R}"
fi
if [ -n "${ATTR_K}" ]; then
    CMD="${CMD} --K ${ATTR_K}"
fi
if [ -n "${ATTRIBUTION_MODEL}" ]; then
    CMD="${CMD} --attribution_model \"${ATTRIBUTION_MODEL}\""
fi

eval ${CMD} > "${LOG_FILE}" 2>&1
