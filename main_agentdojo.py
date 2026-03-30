"""
AgentDojo Benchmark Evaluation with PIArena Defenses

This script evaluates PIArena defenses on the AgentDojo benchmark for
prompt injection attacks in tool-integrated LLM agents.

Usage:
    # Benign utility (no attack) with GPT-4o (OpenAI API)
    python main_agentdojo.py --model gpt-4o-2024-05-13 --attack none

    # OpenAI API models use OPENAI_API_KEY and configs/openai_configs/<model>.yaml

    # With HuggingFace model (starts vLLM server automatically)
    python main_agentdojo.py --model meta-llama/Llama-3.1-8B-Instruct --attack tool_knowledge --defense agentwatcher --monitor_llm <path-or-id>

    # Specify tensor parallel size for large models
    python main_agentdojo.py --model meta-llama/Llama-3.1-70B-Instruct --tensor_parallel_size 4 --attack none

Setup:
    AgentDojo is included as a git submodule with PIArena modifications pre-applied.
    Initialize with: git submodule update --init --recursive
    Then install: cd agentdojo && pip install -e .
"""

import os
import sys
import time
import signal
import subprocess
import argparse
from datetime import datetime

from src.utils import resolve_monitor_llm_path

_REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
AGENTDOJO_PATH = os.path.join(_REPO_ROOT, "agents", "agentdojo")

# Known API model prefixes (don't need vLLM server)
API_MODEL_PREFIXES = ["gpt", "claude", "gemini", "command-r"]

# Bidirectional mapping between lowercase values and enum-style names
_LOWERCASE_TO_ENUM = {
    "gpt-4o": "GPT_4O",
    "gpt-4o-mini": "GPT_4O_MINI",
    "gpt-4.1-mini": "GPT_4_1_MINI",
    "gpt-5-mini": "GPT_5_MINI",
    "gpt-4-0125-preview": "GPT_4_0125_PREVIEW",
    "gpt-4-turbo-2024-04-09": "GPT_4_TURBO_2024_04_09",
    "gpt-3.5-turbo-0125": "GPT_3_5_TURBO_0125",
    "claude-3-opus-20240229": "CLAUDE_3_OPUS_20240229",
    "claude-3-sonnet-20240229": "CLAUDE_3_SONNET_20240229",
    "claude-3-5-sonnet-20240620": "CLAUDE_3_5_SONNET_20240620",
    "claude-3-5-sonnet-20241022": "CLAUDE_3_5_SONNET_20241022",
    "claude-3-7-sonnet-20250219": "CLAUDE_3_7_SONNET_20250219",
    "claude-3-haiku-20240307": "CLAUDE_3_HAIKU_20240307",
    "claude-3-5-haiku": "CLAUDE_3_5_HAIKU",
    "claude-haiku-3": "CLAUDE_HAIKU_3",
    "claude-haiku-4-5": "CLAUDE_HAIKU_4_5",
    "gemini-1.5-pro-002": "GEMINI_1_5_PRO_002",
    "gemini-1.5-pro-001": "GEMINI_1_5_PRO_001",
    "gemini-1.5-flash-002": "GEMINI_1_5_FLASH_002",
    "gemini-1.5-flash-001": "GEMINI_1_5_FLASH_001",
    "gemini-2.0-flash-exp": "GEMINI_2_0_FLASH_EXP",
    "gemini-2.0-flash-001": "GEMINI_2_0_FLASH_001",
    "gemini-2.0-flash": "GEMINI_2_0_FLASH",
    "gemini-2.5-flash-preview-04-17": "GEMINI_2_5_FLASH_PREVIEW_04_17",
    "gemini-2.5-flash": "GEMINI_2_5_FLASH",
    "gemini-2.5-pro-preview-05-06": "GEMINI_2_5_PRO_PREVIEW_05_06",
    "gemini-3-flash-preview": "GEMINI_3_FLASH_PREVIEW",
    "command-r-plus": "COMMAND_R_PLUS",
    "command-r": "COMMAND_R",
}
_ENUM_TO_LOWERCASE = {v: k for k, v in _LOWERCASE_TO_ENUM.items()}
_ALIASES = {"gpt-4-turbo": "gpt-4-turbo-2024-04-09", "gpt-3.5-turbo": "gpt-3.5-turbo-0125"}


def _detect_agentdojo_model_format():
    """Auto-detect whether local AgentDojo CLI expects enum names or lowercase values."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "agentdojo.scripts.benchmark", "--help"],
            capture_output=True, text=True, cwd=f"{AGENTDOJO_PATH}/src",
        )
        if "GPT_4O_MINI" in result.stdout:
            return "enum"
        if "gpt-4o-mini" in result.stdout:
            return "lowercase"
    except Exception:
        pass
    return "lowercase"


_AGENTDOJO_FORMAT = None


def normalize_model_name(model: str) -> str:
    """Normalize model name to the format the local AgentDojo CLI expects."""
    global _AGENTDOJO_FORMAT
    if _AGENTDOJO_FORMAT is None:
        _AGENTDOJO_FORMAT = _detect_agentdojo_model_format()
        print(f"[AgentDojo] Detected model format: {_AGENTDOJO_FORMAT}")

    model = _ALIASES.get(model, model)

    if _AGENTDOJO_FORMAT == "enum":
        return _LOWERCASE_TO_ENUM.get(model, _LOWERCASE_TO_ENUM.get(_ENUM_TO_LOWERCASE.get(model), model))
    else:
        return _ENUM_TO_LOWERCASE.get(model, _ENUM_TO_LOWERCASE.get(_LOWERCASE_TO_ENUM.get(model), model))


def is_api_model(model: str) -> bool:
    """Check if model is an API-based model (OpenAI, Anthropic, Google, Cohere)."""
    model_lower = model.lower()
    return any(model_lower.startswith(prefix) for prefix in API_MODEL_PREFIXES)


def check_agentdojo_installed():
    """Check if AgentDojo is cloned and installed."""
    if not os.path.exists(AGENTDOJO_PATH):
        raise FileNotFoundError(
            f"AgentDojo not found at '{AGENTDOJO_PATH}'. Please run:\n"
            "  git submodule update --init --recursive\n"
            "  cd agentdojo && pip install -e ."
        )

    try:
        import agentdojo  # noqa: F401
    except ImportError:
        raise ImportError(
            "AgentDojo not installed as package. Please run:\n"
            "  cd agentdojo && pip install -e ."
        )


def start_vllm_server(model: str, tensor_parallel_size: int, port: int = 8000) -> tuple[subprocess.Popen, str]:
    """Start vLLM server for HuggingFace model."""
    log_dir = os.path.join(_REPO_ROOT, "results", "agent_evaluations", "agentdojo", "vllm_logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"vllm_server_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    cmd = [
        "vllm", "serve", model,
        "--dtype", "auto",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", "0.8",
        "--max-model-len", "16384",
    ]

    print(f"[vLLM] Starting server: {' '.join(cmd)}")
    print(f"[vLLM] Log file: {log_file}")

    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid if hasattr(os, 'setsid') else None
        )

    # Wait for server to start
    print("[vLLM] Waiting for server to start...")
    max_wait = 300  # 5 minutes max
    waited = 0
    while waited < max_wait:
        time.sleep(10)
        waited += 10

        with open(log_file, 'r') as f:
            log_content = f.read()
            if "Application startup complete" in log_content:
                print(f"[vLLM] Server started successfully after {waited}s")
                return process, log_file
            if "error" in log_content.lower() and "Error" in log_content:
                print(f"[vLLM] Server failed to start. Check log: {log_file}")
                process.terminate()
                raise RuntimeError(f"vLLM server failed to start. Check {log_file}")

        print(f"[vLLM] Still waiting... ({waited}s)")

    print(f"[vLLM] Timeout waiting for server. Check log: {log_file}")
    process.terminate()
    raise RuntimeError(f"vLLM server timeout. Check {log_file}")


def stop_vllm_server(process: subprocess.Popen):
    """Stop vLLM server."""
    if process:
        print("[vLLM] Stopping server...")
        try:
            # Kill the process group
            if hasattr(os, 'killpg'):
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            else:
                process.terminate()
            process.wait(timeout=10)
        except Exception as e:
            print(f"[vLLM] Error stopping server: {e}")
            try:
                process.kill()
            except:
                pass


def run_agentdojo_benchmark(args, model_type: str):
    """Run AgentDojo benchmark with specified configuration.

    Args:
        args: Command line arguments
        model_type: One of 'api', 'huggingface', 'transformers'
    """

    # Set environment variables for PIArena defense
    env = os.environ.copy()
    _pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = _REPO_ROOT + (":" + _pp if _pp else "")
    env["PIARENA_PATH"] = _REPO_ROOT

    if args.defense != "none":
        env["PIARENA_DEFENSE"] = args.defense
    
    # Monitor LLM path for agentwatcher
    if args.monitor_llm:
        monitor_llm_path = resolve_monitor_llm_path(args.monitor_llm, repo_root=_REPO_ROOT)
        env["PIARENA_MONITOR_LLM"] = monitor_llm_path
        print(f"[Monitor LLM] Using: {monitor_llm_path}")
    elif args.defense == "agentwatcher":
        print(f"[WARNING] Defense '{args.defense}' requires --monitor_llm but none was provided")

    # Pass attribution parameters via env vars
    if args.w_s is not None:
        env["PIARENA_W_S"] = str(args.w_s)
    if args.w_l is not None:
        env["PIARENA_W_L"] = str(args.w_l)
    if args.w_r is not None:
        env["PIARENA_W_R"] = str(args.w_r)
    if args.K is not None:
        env["PIARENA_K"] = str(args.K)
    if args.attribution_model is not None:
        env["PIARENA_ATTRIBUTION_MODEL"] = args.attribution_model

    if model_type == "api" and not env.get("OPENAI_API_KEY"):
        print("[WARNING] OPENAI_API_KEY is not set; AgentDojo API models will fail until it is set.")

    # Build command
    cmd = [sys.executable, "-m", "agentdojo.scripts.benchmark"]

    if model_type == "huggingface":
        cmd.extend(["--model", "LOCAL"])
        cmd.extend(["--model-id", args.model])
        cmd.extend(["--tool-delimiter", "input"])
    elif model_type == "transformers":
        cmd.extend(["--model", "TRANSFORMERS"])
        cmd.extend(["--model-id", args.model])
        cmd.extend(["--tool-delimiter", "input"])
    else:
        agentdojo_model = normalize_model_name(args.model)
        cmd.extend(["--model", agentdojo_model])

    # cmd.extend(["--tool-output-format", "json"])

    # Add attack if not benign
    if args.attack != "none":
        cmd.extend(["--attack", args.attack])

    # Add defense
    if args.defense != "none":
        cmd.extend(["--defense", "piarena"])

    # Add suite if specified
    if args.suite:
        cmd.extend(["-s", args.suite])

    # Add user tasks if specified
    if args.user_tasks:
        for ut in args.user_tasks:
            cmd.extend(["-ut", ut])

    # Add sample size if specified (for random sampling under attack)
    if args.sample_size is not None:
        cmd.extend(["--sample-size", str(args.sample_size)])

    # Set custom results directory (use absolute path since cwd changes)
    results_dir = os.path.join(_REPO_ROOT, "results", "agent_evaluations", "agentdojo", args.name)
    os.makedirs(results_dir, exist_ok=True)
    cmd.extend(["--logdir", results_dir])

    print(f"\n[Run] Executing: {' '.join(cmd)}")
    print(f"[Run] Working directory: {AGENTDOJO_PATH}/src")
    if args.defense != "none":
        print(f"[Run] PIARENA_DEFENSE={args.defense}")
    print()

    # Run benchmark
    result = subprocess.run(
        cmd,
        cwd=f"{AGENTDOJO_PATH}/src",
        env=env
    )

    return result.returncode


def main(args):
    # Check AgentDojo is installed
    check_agentdojo_installed()

    if args.model.lower().startswith("azure/"):
        args.model = args.model.split("/", 1)[1]
        print("[Note] azure/ prefix is deprecated; using OpenAI API with OPENAI_API_KEY.")

    if is_api_model(args.model):
        model_type = "api"
    else:
        model_type = "huggingface"

    # Print configuration
    print("\n" + "="*60)
    print("AgentDojo Benchmark")
    print("="*60)
    print(f"  Model:    {args.model}")
    if model_type == "huggingface":
        print(f"  Type:     HuggingFace (vLLM)")
    elif model_type == "transformers":
        print(f"  Type:     HuggingFace (transformers)")
    else:
        print(f"  Type:     API")
    print(f"  Attack:   {args.attack}")
    print(f"  Defense:  {args.defense}")
    print(f"  Suite:    {args.suite or 'all'}")
    if model_type == "huggingface":
        print(f"  TP Size:  {args.tensor_parallel_size}")
    print("="*60)

    vllm_process = None
    returncode = 1

    try:
        # Start vLLM server if needed, otherwise fall back to transformers
        if model_type == "huggingface":
            import shutil
            if shutil.which("vllm") is not None:
                vllm_process, _ = start_vllm_server(
                    args.model,
                    args.tensor_parallel_size,
                    port=8000
                )
            else:
                print(f"[Info] vLLM not found, using transformers directly for: {args.model}")
                model_type = "transformers"

        # Run benchmark
        returncode = run_agentdojo_benchmark(args, model_type)

    except KeyboardInterrupt:
        print("\n[Interrupted] Cleaning up...")
    except Exception as e:
        print(f"\n[Error] {e}")
    finally:
        # Stop vLLM server
        if vllm_process:
            stop_vllm_server(vllm_process)

    # Results are saved to PIArena results directory
    print(f"\n[Done] Results saved in results/agent_evaluations/agentdojo/{args.name}/")

    return returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgentDojo Benchmark Evaluation")
    parser.add_argument("--model", type=str, default="gpt-4o-2024-05-13",
                        help="Model: HuggingFace path (e.g., meta-llama/Llama-3.1-8B-Instruct) "
                             "or API model (e.g., gpt-4o-2024-05-13). Set OPENAI_API_KEY.")
    parser.add_argument("--attack", type=str, default="tool_knowledge",
                        choices=["none", "direct", "important_instructions", "tool_knowledge", "injecagent"],
                        help="Attack type to evaluate (use 'none' for benign utility)")
    parser.add_argument("--defense", type=str, default="none",
                        choices=["none", "promptguard", "datasentinel", "piguard", "gptsafeguard",
                                 "promptarmor", "agentwatcher"],
                        help="Defense to evaluate")
    parser.add_argument("--monitor_llm", type=str, default=None,
                        help="Monitor LLM: Hugging Face repo id or local path (agentwatcher)")
    parser.add_argument("--suite", "-s", type=str, default=None,
                        choices=["workspace", "slack", "travel", "banking"],
                        help="Specific suite to evaluate (default: all)")
    parser.add_argument("--user_tasks", "-ut", type=str, nargs="*", default=None,
                        help="Specific user tasks to evaluate")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Tensor parallel size for vLLM (for HuggingFace models)")
    parser.add_argument("--sample_size", type=int, default=200,
                        help="Randomly sample this many (user_task, injection_task) pairs per suite under attack")
    parser.add_argument("--name", type=str, default="default",
                        help="Experiment name (for reference)")
    parser.add_argument("--w_s", type=int, default=None,
                        help="Attribution sliding window size (default: 10)")
    parser.add_argument("--w_l", type=int, default=None,
                        help="Attribution left context size (default: 150)")
    parser.add_argument("--w_r", type=int, default=None,
                        help="Attribution right context size (default: 50)")
    parser.add_argument("--K", type=int, default=None,
                        help="Number of top attribution windows (default: 1)")
    parser.add_argument("--attribution_model", type=str, default=None,
                        help="Model used for attribution (default: meta-llama/Llama-3.1-8B-Instruct)")
    args = parser.parse_args()

    sys.exit(main(args))
