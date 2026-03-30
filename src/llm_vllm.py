"""
vLLM-based Model class for high-throughput inference.
Supports both single query and batch query for acceleration.
"""

from typing import Union, List, Dict, Optional
import os

# Set multiprocessing method to 'spawn' before any vLLM imports
# This prevents CUDA re-initialization errors in forked subprocesses
os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')

# Lazy import vLLM to avoid import errors when not using vLLM
_vllm_available = None
_vllm_LLM = None
_vllm_SamplingParams = None

def _check_vllm():
    global _vllm_available, _vllm_LLM, _vllm_SamplingParams
    if _vllm_available is None:
        try:
            from vllm import LLM, SamplingParams
            _vllm_LLM = LLM
            _vllm_SamplingParams = SamplingParams
            _vllm_available = True
        except ImportError:
            _vllm_available = False
    return _vllm_available


def _get_available_gpu_count() -> int:
    """
    获取可用的 GPU 数量。
    
    优先级：
    1. CUDA_VISIBLE_DEVICES 环境变量
    2. torch.cuda.device_count()
    """
    import os
    
    # 检查 CUDA_VISIBLE_DEVICES
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        # 如果设置了 CUDA_VISIBLE_DEVICES，计算其中的 GPU 数量
        gpus = [g.strip() for g in cuda_visible.split(",") if g.strip()]
        if gpus:
            return len(gpus)
    
    # 使用 torch 检测
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        pass
    
    # 默认返回 1
    return 1


class VLLMModel:
    """
    vLLM-based model for high-throughput inference.
    
    Supports:
    - Single query (compatible with original Model interface)
    - Batch query for multiple prompts at once
    - Configurable GPU memory utilization
    - Auto-detection of available GPUs for tensor parallelism
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        tensor_parallel_size: int = None,  # None = 自动检测
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 20480,
        dtype: str = "auto",
        trust_remote_code: bool = True,
        **kwargs
    ):
        """
        Initialize vLLM model.
        
        Args:
            model_name_or_path: HuggingFace model name, local path, or LoRA checkpoint path.
                               If a LoRA checkpoint is detected (has adapter_config.json),
                               the base model will be loaded automatically with LoRA enabled.
            tensor_parallel_size: Number of GPUs for tensor parallelism (None = auto-detect all available GPUs)
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
            max_model_len: Maximum sequence length
            dtype: Data type ("auto", "float16", "bfloat16")
            trust_remote_code: Whether to trust remote code
        """
        if not _check_vllm():
            raise ImportError(
                "vLLM is not installed. Install with: pip install vllm"
            )
        
        self.model_name_or_path = model_name_or_path
        self.lora_path = None
        
        # Check if model_name_or_path is a LoRA checkpoint
        # If so, extract base model and set lora_path
        base_model = model_name_or_path
        if os.path.isdir(model_name_or_path):
            adapter_config_path = os.path.join(model_name_or_path, 'adapter_config.json')
            if os.path.exists(adapter_config_path):
                import json
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                base_model = adapter_config.get('base_model_name_or_path', model_name_or_path)
                self.lora_path = model_name_or_path
                print(f"[vLLM] Detected LoRA checkpoint, base model: {base_model}")
        
        # 自动检测 GPU 数量
        if tensor_parallel_size is None:
            tensor_parallel_size = _get_available_gpu_count()
        
        # Get HF cache directory
        hf_cache_dir = os.path.join(os.getenv("HF_HOME"), "hub") if os.getenv("HF_HOME") else None
        
        print(f"⏳ Loading vLLM model: {base_model}")
        if self.lora_path:
            print(f"   LoRA adapter: {self.lora_path}")
        print(f"   Tensor parallel size: {tensor_parallel_size} GPU(s)")
        print(f"   GPU memory utilization: {gpu_memory_utilization:.0%}")
        print(f"   Max model length: {max_model_len}")
        
        # 检查是否禁用 CUDA graph（GH200 等新 GPU 上编译可能非常慢）
        enforce_eager = kwargs.pop('enforce_eager', None)
        if enforce_eager is None:
            # 默认禁用 CUDA graph 以避免长时间编译
            enforce_eager = os.getenv("VLLM_ENFORCE_EAGER", "1").lower() in ("1", "true", "yes")
        
        # 检查是否禁用 prefix caching（长时间运行可能导致 cache 碎片化）
        enable_prefix_caching = kwargs.pop('enable_prefix_caching', None)
        if enable_prefix_caching is None:
            enable_prefix_caching = os.getenv("VLLM_ENABLE_PREFIX_CACHING", "0").lower() in ("1", "true", "yes")
        
        # Enable LoRA if lora_path is set
        enable_lora = self.lora_path is not None
        
        self.model = _vllm_LLM(
            model=base_model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            download_dir=hf_cache_dir,
            enforce_eager=enforce_eager,
            enable_prefix_caching=enable_prefix_caching,
            enable_lora=enable_lora,
            max_lora_rank=64 if enable_lora else None,  # Support LoRA rank up to 64
            **kwargs
        )
        
        self.tensor_parallel_size = tensor_parallel_size
        
        # Get tokenizer from vLLM model
        self.tokenizer = self.model.get_tokenizer()
        
        # Store LoRA request if using LoRA
        self._lora_request = None
        if self.lora_path:
            from vllm.lora.request import LoRARequest
            self._lora_request = LoRARequest("monitor_llm_lora", 1, self.lora_path)
            print(f"✅ LoRA adapter registered: {self.lora_path}")
        
        print(f"✅ vLLM model loaded: {base_model} (on {tensor_parallel_size} GPU(s))")
    
    def _format_messages(self, messages: Union[str, List[Dict[str, str]]]) -> str:
        """Convert messages to a single prompt string using chat template."""
        if isinstance(messages, str):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": messages}
            ]
        
        # Use tokenizer's chat template if available
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            # Fallback: simple concatenation
            prompt = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    prompt += f"System: {content}\n\n"
                elif role == "user":
                    prompt += f"User: {content}\n\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n\n"
            prompt += "Assistant: "
        
        return prompt
    
    def query(
        self,
        messages: Union[str, List[Dict[str, str]]],
        max_new_tokens: int = 2048,
        temperature: float = 0.01,
        do_sample: bool = False,
        top_p: float = 0.95,
        **kwargs
    ) -> str:
        """
        Query the model with a single prompt (compatible with original Model interface).
        
        Args:
            messages: Either a string or list of message dicts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling (if False, uses greedy)
            top_p: Top-p sampling parameter
        
        Returns:
            Generated text string
        """
        results = self.batch_query(
            [messages],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            **kwargs
        )
        return results[0]
    
    def batch_query(
        self,
        messages_list: List[Union[str, List[Dict[str, str]]]],
        max_new_tokens: int = 2048,
        temperature: float = 0.01,
        do_sample: bool = False,
        top_p: float = 0.95,
        **kwargs
    ) -> List[str]:
        """
        Batch query for multiple prompts at once.
        
        This is significantly faster than calling query() multiple times
        because vLLM can process all prompts in parallel.
        
        Args:
            messages_list: List of messages (each can be string or message dicts)
            max_new_tokens: Maximum tokens to generate per prompt
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            top_p: Top-p sampling parameter
        
        Returns:
            List of generated text strings
        """
        if not messages_list:
            return []
        
        # Format all messages to prompts
        prompts = [self._format_messages(m) for m in messages_list]
        
        # Create sampling parameters
        # vLLM uses temperature=0 for greedy decoding
        effective_temp = temperature if do_sample else 0.0
        
        sampling_params = _vllm_SamplingParams(
            max_tokens=max_new_tokens,
            temperature=effective_temp,
            top_p=top_p if do_sample else 1.0,
            **kwargs
        )
        
        # Generate all at once (with LoRA if configured)
        if self._lora_request is not None:
            outputs = self.model.generate(prompts, sampling_params, lora_request=self._lora_request)
        else:
            outputs = self.model.generate(prompts, sampling_params)
        
        # Extract generated text
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            results.append(generated_text)
        
        return results


class HybridModel:
    """
    Hybrid model that uses vLLM for batch inference when available,
    falls back to transformers for single queries.
    
    This is useful when you want fast batch inference but also need
    compatibility with code that expects the original Model interface.
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        use_vllm: bool = True,
        vllm_gpu_memory_utilization: float = 0.85,
        vllm_max_model_len: int = 20480,
        **kwargs
    ):
        """
        Initialize hybrid model.
        
        Args:
            model_name_or_path: HuggingFace model name or local path
            use_vllm: Whether to use vLLM (falls back to transformers if False or unavailable)
            vllm_gpu_memory_utilization: GPU memory fraction for vLLM
            vllm_max_model_len: Max sequence length for vLLM
        """
        self.model_name_or_path = model_name_or_path
        self._vllm_model = None
        self._transformers_model = None
        
        if use_vllm and _check_vllm():
            try:
                self._vllm_model = VLLMModel(
                    model_name_or_path,
                    gpu_memory_utilization=vllm_gpu_memory_utilization,
                    max_model_len=vllm_max_model_len,
                    **kwargs
                )
                self.tokenizer = self._vllm_model.tokenizer
                print(f"✅ Using vLLM backend for {model_name_or_path}")
            except Exception as e:
                print(f"⚠️ Failed to load vLLM model: {e}")
                print(f"   Falling back to transformers...")
                self._load_transformers_model()
        else:
            self._load_transformers_model()
    
    def _load_transformers_model(self):
        """Load model using transformers (original Model class)."""
        from .llm import Model
        self._transformers_model = Model(self.model_name_or_path)
        self.tokenizer = self._transformers_model.tokenizer
        print(f"✅ Using transformers backend for {self.model_name_or_path}")
    
    def query(
        self,
        messages: Union[str, List[Dict[str, str]]],
        max_new_tokens: int = 2048,
        temperature: float = 0.01,
        do_sample: bool = False,
        top_p: float = 0.95,
        **kwargs
    ) -> str:
        """Single query - uses whichever backend is available."""
        if self._vllm_model is not None:
            return self._vllm_model.query(
                messages, max_new_tokens, temperature, do_sample, top_p, **kwargs
            )
        else:
            return self._transformers_model.query(
                messages, max_new_tokens, temperature, do_sample, top_p
            )
    
    def batch_query(
        self,
        messages_list: List[Union[str, List[Dict[str, str]]]],
        max_new_tokens: int = 2048,
        temperature: float = 0.01,
        do_sample: bool = False,
        top_p: float = 0.95,
        **kwargs
    ) -> List[str]:
        """
        Batch query - uses vLLM if available, otherwise falls back to sequential.
        """
        if self._vllm_model is not None:
            return self._vllm_model.batch_query(
                messages_list, max_new_tokens, temperature, do_sample, top_p, **kwargs
            )
        else:
            # Fallback to sequential processing
            return [
                self._transformers_model.query(
                    m, max_new_tokens, temperature, do_sample, top_p
                )
                for m in messages_list
            ]
    
    @property
    def is_vllm(self) -> bool:
        """Check if using vLLM backend."""
        return self._vllm_model is not None


def create_model(
    model_name_or_path: str,
    use_vllm: bool = True,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 20480,
    **kwargs
) -> Union[VLLMModel, 'Model']:
    """
    Factory function to create a model with optional vLLM support.
    
    Args:
        model_name_or_path: Model name or path
        use_vllm: Whether to try using vLLM
        gpu_memory_utilization: GPU memory fraction (for vLLM)
        max_model_len: Max sequence length (for vLLM)
    
    Returns:
        VLLMModel if vLLM is available and use_vllm=True, otherwise original Model
    """
    if use_vllm and _check_vllm():
        try:
            return VLLMModel(
                model_name_or_path,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                **kwargs
            )
        except Exception as e:
            print(f"⚠️ Failed to create vLLM model: {e}")
            print(f"   Falling back to transformers...")
    
    from .llm import Model
    return Model(model_name_or_path)
