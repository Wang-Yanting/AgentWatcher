from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch
import os
import openai
import yaml
import time

from typing import Union, List, Dict


def is_lora_checkpoint(path: str) -> bool:
    """Check if the path is a LoRA checkpoint directory."""
    if not os.path.isdir(path):
        return False
    return os.path.exists(os.path.join(path, "adapter_config.json"))


def is_remote_api_model(model_name_or_path: str) -> bool:
    """True for OpenAI / Anthropic / Google-style API ids (loads from configs/openai_configs)."""
    if "gpt-oss" in model_name_or_path.lower():
        return False
    n = model_name_or_path.lower()
    if n.startswith("azure/"):
        return True
    stem = model_name_or_path.split("/")[-1]
    piarena_path = os.environ.get("PIARENA_PATH", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cfg_path = os.path.join(piarena_path, "configs", "openai_configs", f"{stem}.yaml")
    if os.path.isfile(cfg_path):
        return True
    base = stem.lower()
    if base.startswith(("gpt-3", "gpt-4", "gpt-5", "chatgpt", "o1", "o3")):
        return True
    if base.startswith(("claude", "gemini", "command-r", "deepseek")):
        return True
    return False


def load_gpt_model(openai_config_path, model_name, api_key_index=0):
    with open(openai_config_path, 'r') as file:
        config = yaml.safe_load(file)['default']
    usable_keys = []
    for item in config:
        item_model = item.get('model_name') or item.get('azure_deployment') or model_name
        if item_model == model_name:
            usable_keys.append(item.copy())

    if not usable_keys:
        raise ValueError(f"No config found for model {model_name} in {openai_config_path}")

    config_item = usable_keys[api_key_index]
    client_class = config_item.pop('client_class')

    actual_model_name = config_item.pop('model_name', None) or config_item.pop('azure_deployment', None) or model_name

    for legacy in ('azure_endpoint', 'api_version'):
        config_item.pop(legacy, None)

    api_key = config_item.pop('api_key', None) or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError(
            f"Missing API key for {model_name}: set OPENAI_API_KEY or add api_key to {openai_config_path}"
        )
    config_item['api_key'] = api_key

    client = eval(client_class)(**config_item)
    client._model_name = actual_model_name
    return client

def get_openai_completion_with_retry(client, sleepsec=10, **kwargs) -> str:
    while 1:
        try: return client.chat.completions.create(**kwargs).choices[0].message.content
        except Exception as e:
            print('OpenAI API error:', e, 'sleeping for', sleepsec, 'seconds', flush=True)
            time.sleep(sleepsec)
            if "400" in str(e):
                return "OpenAI Rejected"

class Model():
    def __init__(self, model_name_or_path):
        """
        Initialize Model.

        Args:
            model_name_or_path: HuggingFace model name, local path, or LoRA checkpoint path.
                               If a LoRA checkpoint is detected, the base model will be loaded
                               automatically from adapter_config.json.
        """
        self.model_name_or_path = model_name_or_path
        self.lora_checkpoint = None

        if is_lora_checkpoint(model_name_or_path):
            self.lora_checkpoint = model_name_or_path
            peft_config = PeftConfig.from_pretrained(model_name_or_path)
            base_model_name = peft_config.base_model_name_or_path
        else:
            base_model_name = model_name_or_path

        if not is_remote_api_model(model_name_or_path):
            while True:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        base_model_name,
                        use_fast=True,
                        trust_remote_code=True,
                        token=os.getenv("HF_TOKEN"),
                        cache_dir=os.getenv("HF_HOME")
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        base_model_name,
                        device_map="auto",
                        dtype="auto",
                        token=os.getenv("HF_TOKEN"),
                        cache_dir=os.getenv("HF_HOME"),
                        trust_remote_code=True,
                    )

                    if self.lora_checkpoint is not None:
                        self.model = PeftModel.from_pretrained(self.model, self.lora_checkpoint)

                    self.model.eval()
                    break
                except Exception as e:
                    if "429" in str(e):
                        print("Hit Hugging Face rate limit when loading model. Waiting 5 minutes...")
                        time.sleep(300)
                    else:
                        raise e

            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            model_name = model_name_or_path.split("/")[-1]
            self.model_name = model_name
            piarena_path = os.environ.get("PIARENA_PATH", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            openai_config_path = os.path.join(piarena_path, f"configs/openai_configs/{model_name}.yaml")
            api_key_index = 0
            self.model = load_gpt_model(openai_config_path, model_name, api_key_index)
            self.tokenizer = None

    def query(
        self,
        messages: Union[str, List[Dict[str, str]]],
        max_new_tokens: int = 1024,
        temperature: float = 0.01,
        do_sample: bool = False,
        top_p: float = 0.95,
        top_k: int = 0,
        skip_special_tokens: bool = True,
        reasoning_intervention = None
    ):
        if self.tokenizer is not None:
            if isinstance(messages, str):
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": messages}
                ]
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
            if hasattr(input_ids, 'input_ids'):
                input_ids = input_ids.input_ids
            input_ids = input_ids.to(self.model.device)
            if reasoning_intervention and isinstance(reasoning_intervention, str):
                intervention_ids = self.tokenizer.encode(reasoning_intervention, add_special_tokens=False, return_tensors="pt").to(self.model.device)
                input_ids = torch.cat([input_ids, intervention_ids], dim=1)
            attention_mask = torch.ones_like(input_ids).to(self.model.device)
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }

            gen_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": do_sample,
                "top_p": top_p,
                "repetition_penalty": 1.2,
            }
            if top_k > 0:
                gen_kwargs["top_k"] = top_k
            outputs = self.model.generate(**gen_kwargs)
            generated_text = self.tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=skip_special_tokens)

        else:
            model_name = getattr(self.model, '_model_name', self.model_name)
            api_temperature = 1 if "gpt-5-mini" in model_name else temperature
            generated_text = get_openai_completion_with_retry(self.model,
                messages=messages,
                model=model_name,
                temperature=api_temperature,
            )
        return generated_text
