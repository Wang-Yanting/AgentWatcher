#!/bin/bash
# Run AgentDyn benchmark with agentwatcher. Auto-sets PIARENA env vars.
# Runs only the 3 AgentDyn suites (shopping, github, dailylife) per paper Fig 1, reports combined average.
# Usage: ./scripts/run_agentdyn.sh [benchmark args...]
# Example: ./scripts/run_agentdyn.sh --model gpt-4o-2024-08-06
# Example: ./scripts/run_agentdyn.sh --benign

# Note: Model names must be lowercase with hyphens (e.g. gpt-4o-2024-08-06, not GPT_4O_2024_08_06)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PIARENA_PATH="${REPO_ROOT}"
export PIARENA_DEFENSE=agentwatcher
export PIARENA_MONITOR_LLM="${AGENTWATCHER_MONITOR_HF_REPO:-SecureLLMSys/AgentWatcher-Qwen3-4B-Instruct-2507}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# Parse --benign from args (strip it so we don't pass it to benchmark)
BENIGN=false
ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--benign" ]; then
        BENIGN=true
    else
        ARGS+=("$arg")
    fi
done

MODEL="gpt-4o-2024-08-06"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${REPO_ROOT}/logs/agentdyn_logs"
mkdir -p "${LOG_DIR}"
if [ "$BENIGN" = true ]; then
    LOG_FILE="${LOG_DIR}/agentwatcher-${MODEL//\//-}-benign-${TIMESTAMP}.txt"
else
    LOG_FILE="${LOG_DIR}/agentwatcher-${MODEL//\//-}-important_instructions-${TIMESTAMP}.txt"
fi

# Initialize conda for non-interactive shells
[ -f "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ] && source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agentdojo
cd "${REPO_ROOT}/agents/agentdyn/src"

if [ "$BENIGN" = true ]; then
    echo "Logging to ${LOG_FILE}"
    echo "Running 3 AgentDyn suites (shopping, github, dailylife) BENIGN (no attack) with agentwatcher"
    exec python -m agentdojo.scripts.benchmark \
        -s shopping -s github -s dailylife \
        --model "${MODEL}" \
        --defense piarena \
        --force-rerun \
        "${ARGS[@]}" \
        2>&1 | tee "${LOG_FILE}"
else
    echo "Logging to ${LOG_FILE}"
    echo "Running 3 AgentDyn suites (shopping, github, dailylife) with agentwatcher, attack=important_instructions, sample_size=200"
    exec python -m agentdojo.scripts.benchmark \
        -s shopping -s github -s dailylife \
        --model "${MODEL}" \
        --defense piarena \
        --attack important_instructions \
        --sample-size 500 \
        --force-rerun \
        "${ARGS[@]}" \
        2>&1 | tee "${LOG_FILE}"
fi
