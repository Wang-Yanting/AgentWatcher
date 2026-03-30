import os
import glob
import time

# Check GPU availability without initializing CUDA context
# This prevents issues with vLLM multiprocessing
try:
    import subprocess
    result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
    gpu_count_check = len([l for l in result.stdout.strip().split('\n') if l.startswith('GPU')])
    print(f"GPUs available: {gpu_count_check}")
    mode = "local" if gpu_count_check > 0 else "slurm"
except Exception:
    # Fallback to torch if nvidia-smi not available
    import torch
    print("GPUs available:", torch.cuda.device_count())
    mode = "local" if torch.cuda.device_count() > 0 else "slurm"

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

total_jobs = 0
gpu_count = 0
GPU_INTERVAL_MINUTES = 5  # Interval between jobs on the same GPU

def run(dataset, backend_llm, monitor_llm, attack, defense, attack_path=None, name="test", seed=42, data_num=-1):
    """
    Run PIArena evaluation with specified parameters.
    
    Args:
        dataset: Path of dataset or subset in https://huggingface.co/datasets/sleeepeer/PIArena
        backend_llm: Name of the backend LLM to be used
        attack: Type of attack to be used
        defense: Type of defense to be used
        attack_path: Existing attack result to reuse (optional)
        name: Name of the experiment
        seed: Seed for the experiment
    """
    global gpu_count, total_jobs, gpus
    
    # Wait before starting a new GPU cycle (ensures 20 min gap between jobs on same GPU)
    # Only wait in local mode, not when using sbatch/slurm
    if mode == "local" and gpu_count == 0 and total_jobs > 0:
        print(f"\n[run.py] Waiting {GPU_INTERVAL_MINUTES} minutes before next GPU cycle...")
        time.sleep(GPU_INTERVAL_MINUTES * 60)
        print(f"[run.py] Resuming job submission\n")
    
    gpu_id = gpus[gpu_count]
    gpu_count = (gpu_count + 1) % len(gpus)
    gpu_cmd = f"export CUDA_VISIBLE_DEVICES={str(gpu_id)}"
    
    if monitor_llm and not monitor_llm.startswith('/'):
        monitor_llm_abs = os.path.join(_REPO_ROOT, monitor_llm)
        if os.path.isdir(monitor_llm_abs):
            monitor_llm = monitor_llm_abs
            print(f"[run_injecagent.py] Converted monitor_llm to absolute path: {monitor_llm}")
    
    # Create log file path
    backend_llm_name = backend_llm.replace('/', '-')
    monitor_llm_name = monitor_llm.replace('/', '-')
    dataset_name = dataset.split('/')[-1].split('.')[0]

    if attack_path is not None:
        attack_name = attack_path.split('/')[-1].split('.')[0]
    else:
        attack_name = attack

    log_file = f"./logs/main_logs/{name}/{defense}-{monitor_llm_name}/{backend_llm_name}-{defense}.txt"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Build command
    data_num_arg = f"--data_num {data_num}" if data_num > 0 else ""
    if mode == "local":
        cmd = f"{gpu_cmd} && cd {_REPO_ROOT} && python main_injecagent.py \
                --model {backend_llm} \
                --monitor_llm {monitor_llm} \
                --defense {defense} \
                --name {name} \
                {data_num_arg} \
                > {log_file} 2>&1 &"
        os.system(cmd)
    elif mode == "slurm":
        cmd = f"sbatch {_REPO_ROOT}/scripts/main_injecagent.sh \"{dataset}\" \"{backend_llm}\" \"{attack}\" \"{defense}\" \"{attack_path}\" \"{name}\" \"{seed}\" \"{log_file}\" \"{monitor_llm}\""
        print(cmd)
        os.system(cmd)
    total_jobs += 1
    return 1

name = "run_injecagent"
gpus = [0]
all_defenses = [
    "agentwatcher",
]

# Run experiments
for defense in all_defenses:
    dataset = "none"
    backend_llm = "meta-llama/Llama-3.1-8B-Instruct"
    for monitor_llm_path in ["SecureLLMSys/AgentWatcher-Qwen3-4B-Instruct-2507"]:

        run(
            dataset=dataset,
            backend_llm=backend_llm,
            monitor_llm=monitor_llm_path,
            attack=None,
            defense=defense,
            attack_path=None,
            name=name,
            seed=42,
        )

print(f"Started {total_jobs} jobs in total")