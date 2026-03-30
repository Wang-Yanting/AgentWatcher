# AgentWatcher

**AgentWatcher** is a detection-based defense against indirect prompt injection in LLM agents. It first runs **causal context attribution** over untrusted context to find the most influential contexts, then applies a **monitor LLM** that classifies those contexts under **explicit, customizable rules**. Compared with fully black-box detectors, this pipeline is **easier to interpret**: attribution shows *where* the model focused, and the monitor LLM’s judgment is based on rule-grounded reasoning. An example for monitor LLM’s output is shown below:

<img src="assets/fig_illustration.jpg" alt="AgentWatcher example output" width="750">

This rule-based detection can leverage the reasoning ability of the monitor LLM to achieve a better trade-off between utility and robustness. **AgentWatcher** achieves **state-of-art** performance on LLM agent benchmarks such as **AgentDyn**:

<img src="assets/figure1_agentdyn_benign_utility_asr.png" alt="AgentWatcher example output" width="750">

This repository is intended for reproducing the **main experiments** from the paper.

## 🔨 Requirements

- Python 3.10+ (tested with conda environments).
- CUDA GPU(s) for the backend LLM (vLLM can be used optionally).
- Hugging Face access for datasets and models.

```bash
pip install -r requirements.txt
huggingface-cli login   # if needed for gated models
```

## 🤗 Monitor LLM on Hugging Face

The trained monitor is a **LoRA adapter** (PEFT). By default, `main.py` automatically loads Hub checkpoint **`SecureLLMSys/AgentWatcher-Qwen3-4B-Instruct-2507`** on [SecureLLMSys](https://huggingface.co/SecureLLMSys) (override with `--monitor_llm`).

## 🔬 Running experiments

All launchers live under `scripts/`. They assume you run from the **AgentWatcher** repo root (or `cd` there first). Please first set the OpenAI key with `export OPENAI_API_KEY=<YOUR_KEY>`. Edit the Python launchers (`all_datasets`, `all_defenses`, `gpus`, `name`, models, etc.) to match your sweep before running.

### Long-context benchmark (`main.py`)

Batch jobs (local GPUs or Slurm):

```bash
python scripts/run_long_context.py
```

This drives `main.py` with vLLM (`--use_vllm`) and writes logs under `logs/main_logs/<name>/...`. With no local GPU it submits Slurm via `scripts/main.sh`.

One-off run without the launcher:

```bash
python main.py \
  --dataset lcc_long \
  --attack combined \
  --defense agentwatcher \
  --backend_llm Qwen/Qwen3-4B-Instruct-2507 \
  --monitor_llm SecureLLMSys/AgentWatcher-Qwen3-4B-Instruct-2507 \
  --use_vllm \
  --name my_run
```

Optional attribution overrides on `main.py`: `--w_s`, `--w_l`, `--w_r`, `--K`, `--attribution_model`.

### AgentDojo
To run AgentDojo/AgentDyn, please first set up the enviornment following PIArena (https://github.com/sleeepeer/PIArena):

```
cd agents/agentdojo && pip install -e . && cd ../..
```

Then run:

```bash
python scripts/run_agentdojo.py
```

This calls `main_agentdojo.py` and writes logs under `logs/agentdojo_logs/...`. With no local GPU it uses `scripts/main_agentdojo.sh` for Slurm. For a single manual run you can call `python main_agentdojo.py --model ... --defense agentwatcher --monitor_llm ...` from the repo root; set `PIARENA_DEFENSE` / `PIARENA_MONITOR_LLM` if your wrapper expects them. Defense `agentwatcher` uses **`PIMonitorLLMDefenseAdapter`**.

### AgentDyn

```bash
./scripts/run_agentdyn.sh
```

Optional: `./scripts/run_agentdyn.sh --model gpt-4o-2024-08-06` for running agentwatcher with gpt-4o as the backbone LLM. The script activates the `agentdojo` conda env, and runs from `agents/agentdyn/src`. We recommend using `tmux` to prevent the run from being interrupted.

### InjecAgent

```bash
python scripts/run_injecagent.py
```

This launches `main_injecagent.py` and logs under `logs/main_logs/...`. Slurm entrypoint: `scripts/main_injecagent.sh`.

Direct invocation without the launcher:

```bash
python main_injecagent.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --defense agentwatcher \
  --monitor_llm SecureLLMSys/AgentWatcher-Qwen3-4B-Instruct-2507 \
  --name my_injecagent_run
```

## Acknowledgement

* This project incorporates code from [PIArena](https://github.com/sleeepeer/PIArena), [AgentDojo](https://github.com/ethz-spylab/agentdojo), [AgentDyn](https://github.com/SaFo-Lab/AgentDyn), [InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent), and [AT2](https://github.com/MadryLab/AT2).

## Citation

If you use this code, please cite the **AgentWatcher** paper (when available).
