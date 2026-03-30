"""Direct HuggingFace Transformers inference without vLLM server."""

import json
import random
import re
from collections.abc import Collection, Sequence

import torch
from pydantic import ValidationError

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.llms.local_llm import _make_system_prompt, _parse_model_output
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage, get_text_content_as_str


class TransformersLLM(BasePipelineElement):
    """Loads a HuggingFace model directly using transformers (no vLLM server needed)."""

    def __init__(self, model_id: str, temperature: float = 0.0, top_p: float = 0.9, max_new_tokens: int = 4096):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[TransformersLLM] Loading model: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.model_id = model_id
        print(f"[TransformersLLM] Model loaded on {self.model.device}")

    def _build_messages(self, messages: Sequence[ChatMessage], runtime: FunctionsRuntime) -> list[dict]:
        built = []
        for m in messages:
            role, content = m["role"], m["content"]
            if role == "system" and content is not None:
                content = _make_system_prompt(get_text_content_as_str(content), runtime.functions.values())
                built.append({"role": "system", "content": content})
            elif role == "tool":
                if "error" in m and m["error"] is not None:
                    content = json.dumps({"error": m["error"]})
                else:
                    func_result = m["content"]
                    if func_result == "None":
                        func_result = "Success"
                    content = json.dumps({"result": func_result})
                built.append({"role": "user", "content": content})
            elif role == "assistant":
                text = get_text_content_as_str(content) if content else ""
                built.append({"role": "assistant", "content": text})
            else:
                text = get_text_content_as_str(content) if content else ""
                built.append({"role": role, "content": text})
        return built

    @torch.no_grad()
    def _generate(self, chat_messages: list[dict]) -> str:
        text = self.tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p

        output_ids = self.model.generate(**inputs, **gen_kwargs)
        new_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        chat_messages = self._build_messages(messages, runtime)
        completion = self._generate(chat_messages)
        print(f"[TransformersLLM] Generated {len(completion)} chars", flush=True)
        output = _parse_model_output(completion)
        return query, runtime, env, [*messages, output], extra_args
