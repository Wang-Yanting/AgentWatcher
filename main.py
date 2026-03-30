# -*- coding: utf-8 -*-

import torch
print("GPUs available:", torch.cuda.device_count())
assert torch.cuda.device_count() > 0, "No GPUs available"
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

import argparse
import time
from tqdm import tqdm
from datasets import load_dataset, Dataset
from src.utils import *
from src.llm import Model, is_remote_api_model
from src.attacks import *
from src.defenses import DEFENSES, DEFENSES_BATCH, DefenseWrapper
from src.evaluations import *
from pynvml import *
# Optional vLLM support
try:
    from src.llm_vllm import VLLMModel
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    VLLMModel = None

# Defenses that use attribution + monitor LLM (AgentWatcher)
AGENT_WATCHER_DEFENSES = ("agentwatcher")

# Defense GPU memory usage estimates (rough fractions for vLLM split heuristics)
DEFENSE_GPU_USAGE = {
    "none": 0.0,
    "agentwatcher": 0.15,
    "datasentinel": 0.2,
    "promptguard": 0.15,
    "promptarmor": 0.15,
    "piguard": 0.08,
    "gptsafeguard": 0.40,
}

# Verifier GPU memory allocation (separate from backend LLM)
MONITOR_LLM_GPU_USAGE = 0.25  # fraction of GPU for monitor LLM (vLLM)

def parse_args():
    parser = argparse.ArgumentParser(prog='PIArena', description="test")

    # General args
    parser.add_argument('--dataset', type=str, default="open_prompt_injection",
                        help="Path of dataset or subset in https://huggingface.co/datasets/sleeepeer/PIArena.")
    parser.add_argument('--backend_llm', type=str, default="Qwen/Qwen3-4B-Instruct-2507",
                        help="Name of the backend LLM to be used.")
    parser.add_argument('--monitor_llm', type=str, default=DEFAULT_MONITOR_LLM_HF,
                        help="Monitor LLM: Hugging Face repo id, or local checkpoint path (agentwatcher).")
    parser.add_argument('--attack', type=str, default="combined",
                        help="Type of attack to be used.")
    parser.add_argument('--defense', type=str, default="none",
                        help="Type of defense to be used.")
    parser.add_argument('--attack_path', type=str, default=None,
                        help="Existing attack result to reuse.")
    parser.add_argument('--name', type=str, default='test',
                        help="Name of the experiment.")
    parser.add_argument('--seed', type=int, default=42,
                        help="Seed for the experiment.")
    
    # vLLM acceleration options
    parser.add_argument('--use_vllm', action='store_true',
                        help="Use vLLM for accelerated batch inference.")
    parser.add_argument('--vllm_gpu_memory', type=float, default=None,
                        help="vLLM GPU memory utilization (0.0-1.0), default: auto-calculate based on defense")
    parser.add_argument('--vllm_max_model_len', type=int, default=20480,
                        help="vLLM maximum sequence length (default: 20480)")
    parser.add_argument('--vllm_tensor_parallel_size', type=int, default=None,
                        help="vLLM tensor parallel GPU count (default: auto-detect)")
    parser.add_argument('--batch_size', type=int, default=200,
                        help="Batch size for batch inference (default: 1)")
    
    # AgentWatcher / attribution options (passed when defense is agentwatcher)
    parser.add_argument('--w_s', type=int, default=None,
                        help="Attribution sliding window size (default: 10)")
    parser.add_argument('--w_l', type=int, default=None,
                        help="Attribution left context size (default: 150)")
    parser.add_argument('--w_r', type=int, default=None,
                        help="Attribution right context size (default: 50)")
    parser.add_argument('--K', type=int, default=None,
                        help="Number of top attribution windows (default: 1)")
    parser.add_argument('--attribution_model', type=str, default=None,
                        help="Model used for attribution (default: meta-llama/Llama-3.1-8B-Instruct)")
    
    args = parser.parse_args()
    print(args)
    return args

def main(args):
    # ========== Load Dataset ==========
    try:
        dataset = load_json(args.dataset)
        # Process context fields to remove everything from the LAST "OBJECTIVE:" onward and save suffix in new entry
        for entry in dataset:
            if "context" in entry and isinstance(entry["context"], str):
                context = entry["context"]
                if "OBJECTIVE:" in context:
                    last_ind = context.rindex("OBJECTIVE:")
                    # Save the suffix (everything from LAST "OBJECTIVE:" onward) as new key
                    entry["target_inst_suffix"] = context[last_ind:]
                    # Remove everything from the LAST "OBJECTIVE:" onward in the original context
                    entry["context"] = context[:last_ind].rstrip()
                    if args.attack == "none":
                        entry["context"] = entry["context"].replace(entry["injected_instruction"], "")
                else:
                    entry["target_inst_suffix"] = ""
            print(f"context: {entry['context']}")
            print(f"target_inst_suffix: {entry['target_inst_suffix']}")
        dataset = Dataset.from_list(dataset)
        # Rename column if it exists
        if "gt_target_action_include" in dataset.column_names:
            dataset = dataset.rename_column("gt_target_action_include", "target_task_answer")
        # Add injected_task column only if it doesn't exist
        if "injected_task" not in dataset.column_names:
            dataset = dataset.add_column("injected_task", [""] * len(dataset))
        if "injected_task_answer" not in dataset.column_names:
            dataset = dataset.add_column("injected_task_answer", [""] * len(dataset))
        dataset_name = args.dataset.split('/')[-1].split('.')[0]
    except:
        dataset_name = args.dataset
        while True:
            try:
                if "open_prompt_injection" in dataset_name or "sep" in dataset_name:
                    dataset = load_dataset(
                        "sleeepeer/test001",
                        split=dataset_name
                    )
                
                else:
                    dataset = load_dataset(
                        "sleeepeer/PIArena",
                        split=dataset_name
                    )
                break
            except Exception as e:
                
                if "429" in str(e):
                    print("Hit Hugging Face rate limit when loading dataset. Waiting 5 minutes...")
                    time.sleep(300)
                else:
                    raise e

    try:
        attack_result = load_json(args.attack_path)
        assert len(attack_result) == len(dataset)
        print(f"Loaded existing attack result from {args.attack_path}.")
        attack_name = args.attack_path.split('/')[-1].split('.')[0]
    except:
        attack_result = {}
        attack = ATTACKS[args.attack]
        print("No existing attack result found, will run attack on-the-fly.")
        attack_name = args.attack

    # ========== Load LLM (vLLM or standard) ==========
     # Check if model is API-based (Azure/OpenAI) - cannot use vLLM for API models
    is_api_model = is_remote_api_model(args.backend_llm)
    
    use_vllm = args.use_vllm and VLLM_AVAILABLE and not is_api_model
    if args.use_vllm and not VLLM_AVAILABLE:
        print("⚠️ Warning: --use_vllm specified but vLLM is not installed. Falling back to standard inference.")
    if args.use_vllm and is_api_model:
        print(f"⚠️ Warning: --use_vllm specified but '{args.backend_llm}' is an API model. Using API client instead.")
    if use_vllm:
        # Calculate available GPU memory based on defense requirements
        # GPU split: backend LLM + monitor LLM (vLLM) + attribution (transformers)
        defense_gpu_usage = DEFENSE_GPU_USAGE.get(args.defense.lower(), 0.3)
        monitor_llm_gpu_usage = MONITOR_LLM_GPU_USAGE
        
        if args.vllm_gpu_memory is not None:
            backend_gpu_memory = args.vllm_gpu_memory
        else:
            # Auto-calculate: use 30% default; when defense uses monitor LLM/attribution on same GPU,
            # use higher backend share so KV cache for max_model_len fits (e.g. 20480 needs ~2.81 GiB)
            backend_gpu_memory = 0.40 if args.defense in AGENT_WATCHER_DEFENSES else 0.30
        
        print(f"\n{'='*60}")
        print(f"🔧 Model Loading Configuration")
        print(f"{'='*60}")
        print(f"  Defense: {args.defense}")
        print(f"  Backend LLM: {args.backend_llm}")
        print(f"  Backend vLLM GPU memory: {backend_gpu_memory:.0%}")
        print(f"  Monitor LLM vLLM GPU memory: {monitor_llm_gpu_usage:.0%}")
        print(f"  Attribution model: transformers (~20 GiB)")
        print(f"  Max model length: {args.vllm_max_model_len}")
        print(f"  Tensor parallel size: {args.vllm_tensor_parallel_size or 'auto'}")
        print(f"{'='*60}\n")
        
        llm = VLLMModel(
            args.backend_llm,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            gpu_memory_utilization=backend_gpu_memory,
            max_model_len=args.vllm_max_model_len,
        )
        print(f"✅ Backend vLLM model loaded (GPU: {backend_gpu_memory:.0%})")
    else:
        llm = Model(args.backend_llm)
    
    # ========== Setup Defense ==========
    defense = DEFENSES[args.defense]
    defense_batch = DEFENSES_BATCH.get(args.defense)
    use_batch_defense = use_vllm and defense_batch is not None
    
    if use_batch_defense:
        print(f"✅ Batch defense available for '{args.defense}', will use batch inference")
    elif use_vllm and defense_batch is None:
        print(f"⚠️ No batch defense for '{args.defense}', will process one-by-one with vLLM")

    llm_name = args.backend_llm.replace('/', '-')
    attack_result_path = f"results/evaluation_results/{args.name}/tmp_attack_results/{dataset_name}-{llm_name}-{attack_name}-{args.defense}-{args.seed}.json"
    evaluation_result_path = f"results/evaluation_results/{args.name}/{dataset_name}-{llm_name}-{attack_name}-{args.defense}-{args.seed}.json"

    # Attribution kwargs for agentwatcher (only include if set)
    attr_kwargs = {}
    for p in ("w_s", "w_l", "w_r", "K", "attribution_model"):
        v = getattr(args, p, None)
        if v is not None:
            attr_kwargs[p] = v

    # ========== Setup Evaluators (single and batch versions) ==========
    if "open_prompt_injection" in dataset_name:
        asr_evaluator = llm_judge
        asr_evaluator_batch = llm_judge_batch
        utility_evaluator = open_prompt_injection_utility
        utility_evaluator_batch = open_prompt_injection_utility_batch
    if "wasp" in dataset_name:
        asr_evaluator = llm_judge_wasp
        asr_evaluator_batch = llm_judge_wasp_batch
        utility_evaluator = wasp_utility
        utility_evaluator_batch = wasp_utility_batch
    elif "sep" in dataset_name:
        asr_evaluator = llm_judge
        asr_evaluator_batch = llm_judge_batch
        utility_evaluator = llm_judge
        utility_evaluator_batch = llm_judge_batch
    elif "_long" in dataset_name:
        asr_evaluator = llm_judge
        asr_evaluator_batch = llm_judge_batch
        for metric in longbench_metric_dict.keys():
            if metric in dataset_name:
                utility_evaluator = longbench_metric_dict[metric]
                utility_evaluator_batch = longbench_metric_batch_dict[metric]
                break
    else:
        asr_evaluator = llm_judge
        asr_evaluator_batch = llm_judge_batch
        utility_evaluator = llm_judge
        utility_evaluator_batch = llm_judge_batch

    evaluation_result = {}
    print("No existing evaluation result found, will run evaluation on-the-fly.")

    # ========== Batch Processing Mode ==========
    if use_batch_defense:
        print(f"\n🚀 Running batch inference mode (batch_size={args.batch_size})")
        
        # Prepare all data first
        all_data = []
        for idx, dp in enumerate(dataset):
            data_item = {
                'idx': idx,
                'dp': dp,
                'target_inst_prefix': dp["target_inst"],
                'target_inst_suffix': dp.get("target_inst_suffix", ""),
                'context': dp["context"],
                'injected_task': dp["injected_task"],
                'target_task_answer': dp["target_task_answer"],
                'injected_task_answer': dp["injected_task_answer"],
                'objective': dp.get("objective", ""),
                'injected_prompt': dp.get("injected_prompt", ""),
            }
            
            # Get or generate injected context
            try:
                data_item['injected_context'] = attack_result[idx]["injected_context"]
            except:
                injected_context = attack(
                    target_inst=data_item['target_inst_prefix'],
                    context=data_item['context'],
                    injected_task=data_item['injected_task'],
                    target_task_answer=data_item['target_task_answer'],
                    injected_task_answer=data_item['injected_task_answer'],
                    llm=llm,
                    defense=DefenseWrapper(defense) if args.defense != "none" else None,
                    monitor_llm=args.monitor_llm,
                )
                data_item['injected_context'] = injected_context
                attack_dp = copy.deepcopy(dp)
                attack_dp["injected_context"] = injected_context
                attack_result[idx] = attack_dp
                save_json(attack_result, attack_result_path)
            
            # Format target_inst based on dataset type
            if "wasp" in dataset_name:
                data_item['target_inst'] = [data_item['target_inst_prefix'], data_item['target_inst_suffix']]
            else:
                data_item['target_inst'] = data_item['target_inst_prefix']
            
            all_data.append(data_item)
        
        # Process in batches
        for batch_start in tqdm(range(0, len(all_data), args.batch_size), desc="Batch processing"):
            batch_end = min(batch_start + args.batch_size, len(all_data))
            batch_data = all_data[batch_start:batch_end]
            
            print(f"\n{'='*60}")
            print(f"📦 Processing batch {batch_start//args.batch_size + 1}: samples {batch_start+1}-{batch_end}")
            print(f"{'='*60}")
            
            # Prepare batch inputs
            target_insts = [d['target_inst'] for d in batch_data]
            contexts = [d['injected_context'] for d in batch_data]
            
            # Run batch defense
            batch_start_time = time.time()
            if args.defense in AGENT_WATCHER_DEFENSES:
                defense_results = defense_batch(
                    target_insts=target_insts,
                    contexts=contexts,
                    llm=llm,
                    monitor_llm=args.monitor_llm,
                    benchmark=dataset_name,
                    **attr_kwargs
                )
            else:
                defense_results = defense_batch(
                    target_insts=target_insts,
                    contexts=contexts,
                    llm=llm
                )
            batch_elapsed = time.time() - batch_start_time
            print(f"⚡ Batch defense completed: {len(batch_data)} samples in {batch_elapsed:.2f}s ({batch_elapsed/len(batch_data):.2f}s/sample)")
            
            # ========== Sequential Evaluation (non-batch) ==========
            responses = [dr["response"] for dr in defense_results]
            
            # Sequential utility and ASR evaluation (not using vLLM batch)
            eval_start_time = time.time()
            utility_results = []
            asr_results = []
            
            for i, (data_item, response) in enumerate(zip(batch_data, responses)):
                # Utility evaluation
                utility_task_prompt = f"{data_item['target_inst_prefix']}\n\n{data_item['context']}{data_item['target_inst_suffix']}"
                utility = utility_evaluator(
                    response,
                    ground_truth=data_item['target_task_answer'],
                    task_prompt=utility_task_prompt,
                )
                utility_results.append(utility)
                
                # ASR evaluation
                if "wasp" in dataset_name:
                    asr = asr_evaluator(
                        response,
                        data_item['objective'],
                        data_item['injected_task'],
                        data_item['injected_prompt'],
                    )
                else:
                    asr = asr_evaluator(
                        response,
                        ground_truth=data_item['injected_task_answer'],
                        task_prompt=data_item['injected_task'],
                    )
                asr_results.append(asr)
            
            eval_elapsed = time.time() - eval_start_time
            print(f"⚡ Sequential evaluation completed: {len(batch_data)} samples in {eval_elapsed:.2f}s ({eval_elapsed/len(batch_data):.2f}s/sample)")
            
            # Store results
            for i, (data_item, defense_result, utility, asr) in enumerate(zip(batch_data, defense_results, utility_results, asr_results)):
                idx = data_item['idx']
                dp = data_item['dp']
                result_dp = copy.deepcopy(dp)
                
                result_dp["defense_result"] = defense_result
                result_dp["utility"] = utility
                result_dp["asr"] = asr
                
                # For AgentWatcher: compute ASR based on detection (if detected, use warning message as response)
                if args.defense in AGENT_WATCHER_DEFENSES:
                    detect_flag = defense_result.get("detect_flag", False)
                    if isinstance(detect_flag, str):
                        detect_flag = detect_flag.lower() in ("true", "yes", "1")
                    
                    # If detected, use warning message; otherwise use actual response
                    detection_response = "[Warning] Detected injected prompt in the context." if detect_flag else defense_result["response"]
                    
                    if "wasp" in dataset_name:
                        asr_detection = asr_evaluator(
                            detection_response,
                            data_item['objective'],
                            data_item['injected_task'],
                            data_item['injected_prompt'],
                        )
                    else:
                        asr_detection = asr_evaluator(
                            detection_response,
                            ground_truth=data_item['injected_task_answer'],
                            task_prompt=data_item['injected_task'],
                        )
                    result_dp["asr_detection"] = asr_detection
                
                evaluation_result[idx] = result_dp
            
            # Save after each batch
            save_json(evaluation_result, evaluation_result_path)
            
            # Print batch summary
            batch_results = [evaluation_result[d['idx']] for d in batch_data]
            avg_utility = sum(r['utility'] for r in batch_results) / len(batch_results)
            avg_asr = sum(r['asr'] for r in batch_results) / len(batch_results)
            total_utility = sum(r['utility'] for r in evaluation_result.values()) / len(evaluation_result)
            total_asr = sum(r['asr'] for r in evaluation_result.values()) / len(evaluation_result)
            print(f"📊 Batch Utility: {avg_utility:.2f}, ASR: {avg_asr:.2f}")
            print(f"📊 Total Utility: {total_utility:.2f}, ASR: {total_asr:.2f}")
            
            # Print ASR detection summary for AgentWatcher
            if args.defense in AGENT_WATCHER_DEFENSES:
                asr_det_batch = [r.get('asr_detection', 0) for r in batch_results]
                asr_det_total = [r.get('asr_detection', 0) for r in evaluation_result.values()]
                avg_asr_det = sum(asr_det_batch) / len(asr_det_batch) if asr_det_batch else 0
                total_asr_det = sum(asr_det_total) / len(asr_det_total) if asr_det_total else 0
                print(f"📊 Batch ASR (detection): {avg_asr_det:.2f}, Total ASR (detection): {total_asr_det:.2f}")
    
    # ========== Sequential Processing Mode ==========
    else:
        for idx, dp in tqdm(enumerate(dataset)):
            print(f"=========={idx+1} / {len(dataset)}==========")
            if f"{idx}" in evaluation_result:
                print(f"Skipping index {idx} because it already exists in the evaluation result.")
                continue

            result_dp = copy.deepcopy(dp)
            target_inst_prefix = dp["target_inst"]
            if "target_inst_suffix" not in dp:
                target_inst_suffix = ""
            else:
                target_inst_suffix = dp["target_inst_suffix"]
            if "msg" not in dp:
                msg = ""
            else:
                msg = dp["msg"]
            context = dp["context"]
            injected_task = dp["injected_task"]
            target_task_answer = dp["target_task_answer"]
            injected_task_answer = dp["injected_task_answer"]
            if "objective" not in dp:
                objective = ""
            else:
                objective = dp["objective"]
            if "injected_prompt" not in dp:
                injected_prompt = ""
            else:
                injected_prompt = dp["injected_prompt"]

            try:
                injected_context = attack_result[idx]["injected_context"]
            except:
                injected_context = attack(
                    target_inst=target_inst_prefix,
                    context=context,
                    injected_task=injected_task,
                    target_task_answer=target_task_answer,
                    injected_task_answer=injected_task_answer,
                    llm=llm,
                    defense=DefenseWrapper(defense) if args.defense != "none" else None,
                    monitor_llm=args.monitor_llm,
                )
                attack_dp = copy.deepcopy(dp)
                attack_dp["injected_context"] = injected_context
                attack_result[idx] = attack_dp
                save_json(attack_result, attack_result_path)
            if "wasp" in dataset_name:
                target_inst = [target_inst_prefix, target_inst_suffix]
            else:
                target_inst = target_inst_prefix
            if args.defense in AGENT_WATCHER_DEFENSES:
                defense_result = defense(
                    target_inst=target_inst,
                    context=injected_context,
                    llm=llm,
                    monitor_llm=args.monitor_llm,
                    benchmark=dataset_name,
                    **attr_kwargs
                )
            else:
                defense_result = defense(
                    target_inst=target_inst,
                    context=injected_context,
                    llm=llm
                )
            response = defense_result["response"]
            result_dp["defense_result"] = defense_result

            result_dp["utility"] = utility_evaluator(
                response,
                ground_truth=target_task_answer,
                task_prompt=f"{target_inst_prefix}\n\n{context}{target_inst_suffix}",
            )
            if "wasp" in dataset_name:
                result_dp["asr"] = asr_evaluator(
                    response, objective,injected_task, injected_prompt
                )
            else:
                result_dp["asr"] = asr_evaluator(
                    response,
                    ground_truth=injected_task_answer,
                    task_prompt=injected_task,
                )
            
            # For AgentWatcher: compute ASR based on detection
            if args.defense in AGENT_WATCHER_DEFENSES:
                detect_flag = defense_result.get("detect_flag", False)
                if isinstance(detect_flag, str):
                    detect_flag = detect_flag.lower() in ("true", "yes", "1")
                
                detection_response = "[Warning] Detected injected prompt in the context." if detect_flag else response
                
                if "wasp" in dataset_name:
                    result_dp["asr_detection"] = asr_evaluator(
                        detection_response, objective, injected_task, injected_prompt
                    )
                else:
                    result_dp["asr_detection"] = asr_evaluator(
                        detection_response,
                        ground_truth=injected_task_answer,
                        task_prompt=injected_task,
                    )
            
            evaluation_result[idx] = result_dp
            save_json(evaluation_result, evaluation_result_path)

            nice_print(f"Target Instruction: {target_inst_prefix}")
            print("\n")
            nice_print(f"Injected Task: {injected_task}")
            print("\n")
            nice_print(f"Response: {response}")
            print("\n")
            nice_print(f"Utility: {result_dp['utility']}, {round(sum([r['utility'] for r in evaluation_result.values()]) / len(evaluation_result), 2)}")
            nice_print(f"ASR: {result_dp['asr']}, {round(sum([r['asr'] for r in evaluation_result.values()]) / len(evaluation_result), 2)}")
            if args.defense in AGENT_WATCHER_DEFENSES and "asr_detection" in result_dp:
                asr_det_values = [r.get('asr_detection', 0) for r in evaluation_result.values()]
                nice_print(f"ASR (detection): {result_dp['asr_detection']}, {round(sum(asr_det_values) / len(asr_det_values), 2)}")

def wait_for_available_gpu_memory(required_memory_gb, device=0, check_interval=5):
    """
    Waits until the required amount of GPU memory is available.
    Args:
    required_memory_gb (float): Required GPU memory in gigabytes.
    device (int): GPU device index relative to CUDA_VISIBLE_DEVICES (default is 0, meaning first visible GPU)
    check_interval (int): Time interval in seconds between memory checks.
    Returns:
    None
    """
    required_memory_bytes = required_memory_gb * 1e9  # Convert GB to bytes
    
    # Map device index to actual GPU index using CUDA_VISIBLE_DEVICES
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if cuda_visible is not None and cuda_visible != '':
        visible_gpus = [int(x.strip()) for x in cuda_visible.split(',')]
        if device < len(visible_gpus):
            actual_device = visible_gpus[device]
            print(f"[GPU Wait] CUDA_VISIBLE_DEVICES={cuda_visible}, device={device} -> actual GPU {actual_device}")
        else:
            actual_device = device
            print(f"[GPU Wait] Warning: device {device} >= len(CUDA_VISIBLE_DEVICES), using device {device}")
    else:
        actual_device = device
        print(f"[GPU Wait] No CUDA_VISIBLE_DEVICES set, using device {device}")
    
    while True:
        try:
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(actual_device)
            info = nvmlDeviceGetMemoryInfo(handle)
            available_memory = info.free
            if available_memory >= required_memory_bytes:
                print(f"Sufficient GPU memory available on GPU {actual_device}: {available_memory / 1e9:.2f} GB")
                nvmlShutdown()
                return
            else:
                print(f"Waiting for GPU {actual_device} memory. Available: {available_memory / 1e9:.2f} GB, Required: {required_memory_gb:.2f} GB")
            nvmlShutdown()
        except NVMLError as error:
            print(f"Error getting GPU memory: {error}")
            # Fallback to PyTorch method
            if torch.cuda.is_available():
                torch_device = torch.cuda.current_device()
                total_memory = torch.cuda.get_device_properties(torch_device).total_memory
                allocated_memory = torch.cuda.memory_allocated(torch_device)
                available_memory = total_memory - allocated_memory
                if available_memory >= required_memory_bytes:
                    print(f"Sufficient GPU memory available (PyTorch): {available_memory / 1e9:.2f} GB")
                    return 1
                else:
                    print(f"Waiting for GPU memory (PyTorch). Available: {available_memory / 1e9:.2f} GB, Required: {required_memory_gb:.2f} GB")
            else:
                print("CUDA is not available")
        time.sleep(check_interval)
if __name__ == '__main__':
    args = parse_args()
    sleep_time = random.randint(0, 60)
    time.sleep(sleep_time)
    wait_for_available_gpu_memory(50, device=0, check_interval=500)
    
    setup_seeds(args.seed)
    torch.cuda.empty_cache()
    main(args)