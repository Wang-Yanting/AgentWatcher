"""
PIArena Defense Adapter for AgentDojo Pipeline

This module provides an adapter to use PIArena defenses within AgentDojo's
agent pipeline. It filters tool outputs using PIArena's defense execute() method.

Defense Types Supported:
    - Detection (PromptGuard, DataSentinel, PIGuard, AttentionTracker):
      Empty context if injection detected
    - Sanitization (DataFilter, PISanitizer):
      Return cleaned context
    - Hybrid (PromptArmor, PromptLocate):
      Return cleaned context (with detection info)

Usage:
    Set environment variables before running:
        export PIARENA_DEFENSE=promptguard
        export PIARENA_PATH=/path/to/PIArena  # optional, auto-detected
"""

import json
import sys
import os
from collections.abc import Sequence

import torch

# Accumulated defense times (excludes backend LLM) for AgentDojo benchmark summary
_AGENTDOJO_DEFENSE_TIMES: list[float] = []

# Add PIArena to path
PIARENA_PATH = os.environ.get("PIARENA_PATH")
if PIARENA_PATH is None:
    # Try to auto-detect: assume agentdojo is inside PIArena/agents/
    _current = os.path.dirname(os.path.abspath(__file__))
    # Go up from agents/agentdojo/src/agentdojo/agent_pipeline to PIArena root
    _potential_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_current)))))
    if os.path.exists(os.path.join(_potential_path, "piarena")) or os.path.exists(
        os.path.join(_potential_path, "src", "defenses")
    ):
        PIARENA_PATH = _potential_path

if PIARENA_PATH and PIARENA_PATH not in sys.path:
    sys.path.insert(0, PIARENA_PATH)

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.inference_utils import parse
from agentdojo.types import ChatMessage, text_content_block_from_string
from agentdojo.functions_runtime import Env, EmptyEnv, FunctionsRuntime


def recursive_defense(obj, defense, target_inst: str, defense_name: str = None, monitor_llm: str = None, target_model_response: str = None):
    """
    Apply defense recursively to the input object.
    
    The input object can be a dict, list, or string.
    For strings, apply the defense's execute() method.
    For dicts/lists, recurse into nested structures.
    
    For agentwatcher: treat entire object as string and apply defense once.
    
    Args:
        obj: The object to apply defense to (dict, list, or string)
        defense: The PIArena defense instance with execute() method
        target_inst: The user instruction/query for context
        defense_name: Name of the defense (for special handling)
        monitor_llm: Path or id for the monitor LLM (agentwatcher)
        target_model_response: The agent's actual response (for agentwatcher)
        
    Returns:
        The defended/cleaned object with same structure as input
    """
    if defense_name == "agentwatcher":
        if isinstance(obj, (dict, list)):
            context_str = json.dumps(obj, indent=2, ensure_ascii=False)
        else:
            context_str = str(obj)
        
        result = defense.execute(
            target_inst=target_inst, 
            context=context_str, 
            monitor_llm=monitor_llm,
            target_model_response=target_model_response
        )
        t = result.get("time")
        if t is not None:
            _AGENTDOJO_DEFENSE_TIMES.append(float(t))
        if result.get("detect_flag"):
            return ""
        elif "cleaned_context" in result:
            return str(result["cleaned_context"])
        else:
            return context_str
    
    # For other defenses, apply recursively
    if isinstance(obj, dict):
        return {k: recursive_defense(v, defense, target_inst, defense_name, monitor_llm, target_model_response) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_defense(v, defense, target_inst, defense_name, monitor_llm, target_model_response) for v in obj]
    elif isinstance(obj, str):
        # Apply PIArena defense to string content
        result = defense.execute(target_inst=target_inst, context=obj)
        t = result.get("time")
        if t is not None:
            _AGENTDOJO_DEFENSE_TIMES.append(float(t))
        # Get cleaned context based on defense type
        if "cleaned_context" in result and defense_name != "agentwatcher":
            cleaned = result["cleaned_context"]
            if isinstance(cleaned, (dict, list)):
                return json.dumps(cleaned, indent=2, ensure_ascii=False)
            return str(cleaned)
        elif result.get("detect_flag"):
            # Detection-based defense detected injection - return empty
            return ""
        else:
            # No change needed
            return obj
    else:
        return obj


class PIArenaDefenseAdapter(BasePipelineElement):
    """
    AgentDojo pipeline element that filters tool outputs using PIArena defenses.

    This adapter intercepts tool messages in the conversation and applies
    PIArena's defense mechanism to filter potentially malicious content.

    The defense to use is read from the PIARENA_DEFENSE environment variable.
    Default is 'promptguard' if not set.
    """

    def __init__(self, defense_name: str = None, defense_config: dict = None):
        # Get defense name from env or parameter
        self.defense_name = defense_name or os.environ.get("PIARENA_DEFENSE", "promptguard")
        self.defense_config = defense_config

        # Lazy load defense to avoid import issues
        self._defense = None
        print(f"[PIArenaDefenseAdapter] Will use defense: {self.defense_name}")

    def _get_defense(self):
        """Lazy load the defense."""
        if self._defense is None:
            # Initialize CUDA properly before loading defense models
            # This ensures consistent device handling when running in AgentDojo subprocess
            if torch.cuda.is_available():
                torch.cuda.init()
                torch.cuda.set_device(0)
                print(f"[PIArenaDefenseAdapter] CUDA initialized, using device: cuda:0")
            
            from src.defenses import get_defense
            self._defense = get_defense(self.defense_name)
            print(f"[PIArenaDefenseAdapter] Loaded defense: {self.defense_name}")
        return self._defense

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        """
        Process messages and filter tool outputs.

        Only processes if the last message is a tool message.
        Applies defense recursively to all tool messages.
        """

        # Only filter if the last message is a tool message
        if len(messages) == 0 or messages[-1]["role"] != "tool":
            return query, runtime, env, messages, extra_args

        # Only clean the LAST tool message
        msg = messages[-1]
        defense = self._get_defense()
        if msg["role"] == "tool" and "content" in msg:
            try:
                # extract user instruction (same logic as before)
                user_instruction = ""
                for m in messages:
                    if m["role"] == "user":
                        user_instruction = m["content"][0]["content"]
                        break
    
                raw_data = msg["content"][0]["content"]
    
                json_data = parse(raw_data)
                cleaned = recursive_defense(
                    json_data,
                    defense=defense,
                    target_inst=user_instruction,
                )
                cleaned_str = json.dumps(cleaned, indent=2, ensure_ascii=False)
    
                print("\nUser instruction:", user_instruction)
                print("\n\n---------------------------------------------------------")
                print("Tool call result (raw):")
                print(raw_data)
                print("\n\n---------------------------------------------------------")
                print("Tool call result (cleaned):")
                print(cleaned_str)
    
                # Replace ONLY the last tool message's content
                msg["content"] = [text_content_block_from_string(cleaned_str)]
    
            except Exception as e:
                print(f"[PIArenaDefenseAdapter] Skipped cleaning due to error: {e}")

        return query, runtime, env, messages, extra_args
class PIMonitorLLMDefenseAdapter(BasePipelineElement):
    """
    AgentDojo pipeline element that filters tool outputs using PIArena defenses.

    This adapter intercepts tool messages in the conversation and applies
    PIArena's defense mechanism to filter potentially malicious content.

    The defense to use is read from the PIARENA_DEFENSE environment variable.
    
    For agentwatcher, set PIARENA_MONITOR_LLM.
    """

    def __init__(self, defense_name: str = None, defense_config: dict = None, monitor_llm: str = None):
        # Get defense name from env or parameter
        self.defense_name = defense_name or os.environ.get("PIARENA_DEFENSE", "promptguard")
        self.defense_config = defense_config
        self.monitor_llm = monitor_llm or os.environ.get("PIARENA_MONITOR_LLM")

        # Lazy load defense to avoid import issues
        self._defense = None
        self._total_tasks = 0
        self._detected_tasks = 0
        self._current_task_detected = False
        self._last_user_instruction = None
        print(f"[PIArenaDefenseAdapter] Will use defense: {self.defense_name}")
        if self.monitor_llm:
            print(f"[PIArenaDefenseAdapter] Will use monitor LLM: {self.monitor_llm}")
        else:
            print(f"[PIArenaDefenseAdapter] WARNING: No monitor LLM configured (PIARENA_MONITOR_LLM not set)")

    def _get_defense(self):
        """Lazy load the defense."""
        if self._defense is None:
            # Initialize CUDA properly before loading defense models
            # This ensures consistent device handling when running in AgentDojo subprocess
            if torch.cuda.is_available():
                torch.cuda.init()
                torch.cuda.set_device(0)
                print(f"[PIArenaDefenseAdapter] CUDA initialized, using device: cuda:0")
            
            from src.defenses import get_defense
            self._defense = get_defense(self.defense_name)
            print(f"[PIArenaDefenseAdapter] Loaded defense: {self.defense_name}")
        return self._defense


    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        """
        Process messages and filter tool outputs.

        Only processes if the last message is a tool message.
        Applies defense recursively to all tool messages.
        """

        # Defense runs BETWEEN LLM and ToolsExecutor
        # Goal: Detect malicious tool calls BEFORE they execute
        if len(messages) < 2:
            return query, runtime, env, messages, extra_args
        
        # Check if last message is assistant with tool calls (what agent wants to execute)
        if messages[-1]["role"] != "assistant":
            return query, runtime, env, messages, extra_args
        
        # Only run defense when the agent outputs a tool call (not plain text)
        if not messages[-1].get("tool_calls"):
            print("[PIArenaDefenseAdapter] No tool calls in agent response, skipping defense")
            return query, runtime, env, messages, extra_args
        
        print("messages[-1]:",messages[-1])
        # Collect ALL tool messages in history to build full context
        tool_contents = []
        for msg in messages:
            if msg["role"] == "tool" and "content" in msg and msg["content"]:
                content = msg["content"][0]["content"]
                if content and content.strip():
                    tool_contents.append(content)
        
        if not tool_contents:
            print("[PIArenaDefenseAdapter] No tool messages found, skipping defense")
            return query, runtime, env, messages, extra_args
        
        try:
            # Extract user instruction
            user_instruction = ""
            for m in messages:
                if m["role"] == "user":
                    user_instruction = m["content"][0]["content"]
                    break
            
            # Track per-task detection: new task when user instruction changes
            if user_instruction != self._last_user_instruction:
                self._last_user_instruction = user_instruction
                self._total_tasks += 1
                self._current_task_detected = False
            
            # Concatenate all tool outputs as context
            raw_data = "\n\n--------------------------------\n\n".join(tool_contents)
            print(f"[PIArenaDefenseAdapter] Collected {len(tool_contents)} tool messages for context")
            if raw_data == "":
                print("[PIArenaDefenseAdapter] Empty context, skipping defense")
                return query, runtime, env, messages, extra_args
            # Extract agent's actual response (text content or tool calls)
            agent_response = ""
            print("messages[-1][content]:",messages[-1]["content"])
            if messages[-1]["content"]:
                for content_block in messages[-1]["content"]:
                    if content_block.get("type") == "text":
                        agent_response += content_block.get("content", "")
            
            # If no text content, convert tool calls to string as the response
            if not agent_response and messages[-1].get("tool_calls"):
                tool_calls = messages[-1]["tool_calls"]
                tool_call_strs = []
                for tc in tool_calls:
                    func_name = tc.function if hasattr(tc, 'function') else tc.get('function', 'unknown')
                    args = tc.args if hasattr(tc, 'args') else tc.get('arguments', {})
                    tool_call_strs.append(f"Tool call: {func_name}({args})")
                agent_response = "\n".join(tool_call_strs)
                print("[PIArenaDefenseAdapter] Using tool calls as response:", agent_response[:200])
            
            print("\n" + "="*60)
            print("[PIArenaDefenseAdapter] Running defense AFTER agent response")
            print("User instruction:", user_instruction)
            print("Agent response:", agent_response[:200] + "..." if len(agent_response) > 200 else agent_response)
            print("="*60)
            
            defense = self._get_defense()
            defense_kwargs = {}
            for param, env_key in [("w_s", "PIARENA_W_S"), ("w_l", "PIARENA_W_L"),
                                   ("w_r", "PIARENA_W_R"), ("K", "PIARENA_K")]:
                val = os.environ.get(env_key)
                if val is not None:
                    defense_kwargs[param] = int(val)
            val = os.environ.get("PIARENA_ATTRIBUTION_MODEL")
            if val is not None:
                defense_kwargs["attribution_model"] = val
            result = defense.execute(
                target_inst=user_instruction,
                context=raw_data,
                monitor_llm=self.monitor_llm,
                target_model_response=agent_response,
                benchmark="agentdyn",
                **defense_kwargs,
            )
            t = result.get("time")
            if t is not None:
                _AGENTDOJO_DEFENSE_TIMES.append(float(t))
            detected = result.get("detect_flag", False)
            if detected and not self._current_task_detected:
                self._detected_tasks += 1
                self._current_task_detected = True
            detection_rate = self._detected_tasks / self._total_tasks if self._total_tasks > 0 else 0.0
            
            if detected:
                print("[PIArenaDefenseAdapter] INJECTION DETECTED - stopping agent and clearing history")
                # Clear tool calls in current assistant message to prevent execution
                messages[-1]["tool_calls"] = None
                # Replace with warning message
                warning_message = "[WARNING] Prompt injection attack detected"
                messages[-1]["content"] = [text_content_block_from_string(warning_message)]
                
                # Remove all tool call history to prevent agent from continuing
                filtered_messages = []
                for msg in messages:
                    # Keep all non-tool messages
                    if msg["role"] != "tool":
                        # For assistant messages, clear tool_calls
                        if msg["role"] == "assistant":
                            msg_copy = msg.copy()
                            msg_copy["tool_calls"] = None
                            filtered_messages.append(msg_copy)
                        else:
                            filtered_messages.append(msg)
                    # Skip tool messages (delete tool calling history)
                
                print(f"[PIArenaDefenseAdapter] Cleared tool history: {len(messages)} -> {len(filtered_messages)} messages")
                return query, runtime, env, filtered_messages, extra_args
            else:
                print("[PIArenaDefenseAdapter] No injection detected")

        except Exception as e:
            print(f"[PIArenaDefenseAdapter] Skipped cleaning due to error: {e}")
            import traceback
            traceback.print_exc()

        return query, runtime, env, messages, extra_args
