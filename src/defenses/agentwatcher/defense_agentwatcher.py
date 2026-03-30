import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .attention_utils import *
from ...llm import Model
# Aliases: the `monitor_llm` parameter (model id/path) must not shadow these callables.
from ..monitor_llm_module import monitor_llm as _run_monitor_llm, monitor_llm_batch as _run_monitor_llm_batch
import time
import numpy as np

model = None
tokenizer = None
import re

def parse_actions(text):
    """
    Parse out all Action and Action Input pairs from text.
    If no patterns found, returns original text (for plain text responses like AgentDojo).
    """
    if not text:
        return ""
    
    if "```" in text:
        return text.split("```")[1].split("```")[0]
    
    result_string = ""
    
    # Pattern to match Action: and Action Input: pairs
    pattern = r'Action:\s*(.+?)\s*\nAction Input:\s*(.+?)(?=\n|$)'
    
    matches = re.findall(pattern, text, re.DOTALL)
    
    for action, action_input in matches:
        result_string += f"Action: {action.strip()}\nAction Input: {action_input.strip()}\n"

    # Return original text if no patterns found
    if not result_string:
        return text
    
    return result_string
def agentwatcher(
    target_inst,
    context=None,
    system_prompt:str="You are a helpful assistant.",
    llm:Model=None,
    monitor_llm="gpt-4o-mini",
    target_model_response=None,
    **kwargs,
):  
    # Initialize LLM if not provided
    if llm is None:
        llm = Model("gpt-4o-mini")
        print("[AgentWatcher] Initialized default LLM: gpt-4o-mini")
    
    if isinstance(target_inst, list):
        messages = [
            {"role": "system", "content": target_inst[0]},
            {"role": "user", "content": f"{context}{target_inst[1]}"},
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{target_inst}\n\n{context}"},
        ]
   
    if target_model_response is None:
        if llm is not None:
            original_response = llm.query(messages)
        else:
            original_response = "[PIArena] No LLM provided, response is not available."
    else:
        original_response = target_model_response
    original_response = parse_actions(original_response)
    defense_start_time = time.time()
    start_time = time.time()
    attr_kwargs = {p: kwargs[p] for p in ("w_s", "w_l", "w_r", "K", "attribution_model") if p in kwargs}
    print("[AgentWatcher] Using attention-based attribution")
    top_windows_texts, not_in_window_texts, top_windows_intervals, not_in_window_intervals = attribute(
        system_prompt, target_inst, context, original_response, **attr_kwargs
    )

    top_windows_texts[0] = top_windows_texts[0] + "\n \n"
    # Sort top_windows_texts by the start index of their corresponding top_windows_intervals
    sorted_top_windows = sorted(zip(top_windows_intervals, top_windows_texts), key=lambda x: x[0][0] if isinstance(x[0], (list, tuple)) else x[0])
    top_windows_intervals, top_windows_texts = zip(*sorted_top_windows) if sorted_top_windows else ([], [])
    top_windows_intervals = list(top_windows_intervals)
    top_windows_texts = list(top_windows_texts)

    print("attributed_contexts: ", top_windows_texts)
    promptarmor_output = _run_monitor_llm(
        target_inst,
        context='\n\n'.join(top_windows_texts),
        response=original_response,
        system_prompt="You are a helpful assistant.",
        llm=llm,
        monitor_llm=monitor_llm,
        benchmark=kwargs.get("benchmark"),
        use_vllm=kwargs.get("use_vllm", False),
    )
    potential_injection = promptarmor_output["potential_injection"]
    detect_flag = promptarmor_output["detect_flag"]

    if detect_flag and potential_injection is not None:
        # Remove each word in potential_injection from each top_window_text
        cleaned_top_windows_texts = []
        for top_window_text in top_windows_texts:
            if not top_window_text:  # skip empty
                cleaned_top_windows_texts.append(top_window_text)
                continue
            # Tokenize top_window_text and potential_injection by spaces

            window_words = top_window_text.split()
            inject_words = set(potential_injection.split())
            # Remove words appearing in inject_words
            cleaned_words = [w for w in window_words if w not in inject_words]
            cleaned_top_window_text = ' '.join(cleaned_words)
            cleaned_top_windows_texts.append(cleaned_top_window_text)
        top_windows_texts = cleaned_top_windows_texts

        print("===================================")
        if detect_flag and potential_injection is not None:
            # Join segments in the order specified by their intervals
            # Each element in *_intervals is a tuple or value indicating the original order
            # We'll zip the texts with their intervals, then concatenate and sort
            segments_and_intervals = []
            segments_and_intervals.extend(zip(top_windows_intervals, top_windows_texts))
            segments_and_intervals.extend(zip(not_in_window_intervals, not_in_window_texts))
            # Now sort by interval order (assuming interval is (start, end) and we sort by start)
            segments_and_intervals.sort(key=lambda pair: pair[0][0] if isinstance(pair[0], (list, tuple)) else pair[0])
            context = ' '.join([seg for _, seg in segments_and_intervals])
        elif detect_flag and potential_injection is None:
            context = ' '.join(not_in_window_texts) 
    
    end_time = time.time()
    print(f"agentwatcher time: {end_time - start_time} seconds")
    # Defense time excludes the final target LLM query (attribution + monitor LLM + cleaning only)
    defense_elapsed = time.time() - defense_start_time
    
    # If injection detected, return warning instead of querying LLM
    if detect_flag:
        response = "[Warning] Detected injected prompt in the context."
        print(f"Injection detected, returning warning message")
    else:
        if isinstance(target_inst, list):
            messages = [
                {"role": "system", "content": target_inst[0]},
                {"role": "user", "content": f"{context}{target_inst[1]}"},
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{target_inst}\n\n{context}"},
            ]
        start_time = time.time()
        if llm is not None:
            response = llm.query(messages)
        else:
            response = "[PIArena] No LLM provided, response is not available."
        end_time = time.time()
        print(f"LLM generation time: {end_time - start_time} seconds")
    
    return {
        "response": response,
        "cleaned_context": context,
        "monitor_llm_response": promptarmor_output['monitor_llm_response'],
        "full_msg": promptarmor_output['full_msg'],
        "detect_flag": detect_flag,
        "potential_injection": promptarmor_output.get('potential_injection'),
        "time": defense_elapsed,
    }


def agentwatcher_batch(
    target_insts,
    contexts=None,
    system_prompt: str = "You are a helpful assistant.",
    llm: 'Model' = None,
    monitor_llm="gpt-4o-mini",
    **kwargs,
):
    """
    Batch version of AgentWatcher defense, utilizing monitor_llm_batch for verification.
    Args:
        target_insts: List of user instructions (target tasks, possibly formatted).
        contexts: List of untrusted contexts.
        system_prompt: System prompt to set context to the LLM.
        llm: LLM object for querying model.
        monitor_llm: Monitor LLM id or checkpoint path.
    Returns:
        List of dicts, each containing:
          - response: The LLM's final answer based on the cleaned context.
          - cleaned_context: Context with potential injection removed.
          - monitor_llm_response: Raw monitor LLM output.
          - full_msg: Full prompt(s) sent to the monitor LLM.
    """
    import time

    # Convert to empty list if argument was None
    if contexts is None:
        contexts = [None] * len(target_insts)

    results = []

    # Generate the initial responses for all instances from the LLM
    orig_messages = []
    for inst, ctx in zip(target_insts, contexts):
        if isinstance(inst, list):
            orig_messages.append([
                {"role": "system", "content": inst[0]},
                {"role": "user", "content": f"{ctx if ctx is not None else ''}{inst[1]}"},
            ])
        else:
            orig_messages.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{inst}\n\n{ctx if ctx is not None else ''}"},
            ])

    # Query initial response for each sample, batching if supported
    if llm is not None and hasattr(llm, "batch_query"):
        original_responses = llm.batch_query(orig_messages)
    elif llm is not None:
        original_responses = [llm.query(m) for m in orig_messages]
    else:
        original_responses = ["[PIArena] No LLM provided, response is not available."] * len(target_insts)

    # Attribute and clean contexts in batch
    batch_detection_start = time.time()
    batch_top_windows_texts = []
    batch_not_in_window_texts = []
    batch_top_windows_intervals = []
    batch_not_in_window_intervals = []
    attr_kwargs = {p: kwargs[p] for p in ("w_s", "w_l", "w_r", "K", "attribution_model") if p in kwargs}
    print("[AgentWatcher Batch] Using attention-based attribution")
    attribution_start = time.time()
    for inst, ctx, orig_resp in zip(target_insts, contexts, original_responses):
        top_windows_texts, not_in_window_texts, top_windows_intervals, not_in_window_intervals = attribute(
            system_prompt, inst, ctx, orig_resp, **attr_kwargs
        )
        if top_windows_texts and len(top_windows_texts) > 0:
            top_windows_texts[0] = top_windows_texts[0] + "\n \n"
        batch_top_windows_texts.append(top_windows_texts)
        batch_not_in_window_texts.append(not_in_window_texts)
        batch_top_windows_intervals.append(top_windows_intervals)
        batch_not_in_window_intervals.append(not_in_window_intervals)
    attribution_end = time.time()
    print(f"[AgentWatcher Batch] Attribution time: {attribution_end - attribution_start:.2f} seconds ({len(target_insts)} samples)")

    selected_ctxs = ['\n\n'.join(txts) for txts in batch_top_windows_texts]
    promptarmor_outputs = _run_monitor_llm_batch(
        target_insts,
        contexts=selected_ctxs,
        responses=original_responses,
        system_prompt=system_prompt,
        llm=llm,
        monitor_llm=monitor_llm,
        benchmark=kwargs.get("benchmark"),
        use_vllm=kwargs.get("use_vllm", True),
    )

    cleaned_contexts = []
    message_batches = []
    for idx, (
        inst,
        ctx,
        orig_resp,
        top_windows_texts,
        not_in_window_texts,
        top_windows_intervals,
        not_in_window_intervals,
        promptarmor_output,
    ) in enumerate(
        zip(
            target_insts,
            contexts,
            original_responses,
            batch_top_windows_texts,
            batch_not_in_window_texts,
            batch_top_windows_intervals,
            batch_not_in_window_intervals,
            promptarmor_outputs,
        )
    ):
        print("attributed_contexts: ", top_windows_texts)

        potential_injection = promptarmor_output.get("potential_injection")
        detect_flag = promptarmor_output.get("detect_flag")

        # Clean the detected injection from the context 
        if detect_flag and potential_injection is not None:
            cleaned_top_windows_texts = []
            inject_words = set(potential_injection.split())
            for top_window_text in top_windows_texts:
                if not top_window_text:
                    cleaned_top_windows_texts.append(top_window_text)
                    continue
                window_words = top_window_text.split()
                cleaned_words = [w for w in window_words if w not in inject_words]
                cleaned_top_window_text = ' '.join(cleaned_words)
                cleaned_top_windows_texts.append(cleaned_top_window_text)
            top_windows_texts = cleaned_top_windows_texts

            if detect_flag and potential_injection is not None:
                segments_and_intervals = []
                segments_and_intervals.extend(zip(top_windows_intervals, top_windows_texts))
                segments_and_intervals.extend(zip(not_in_window_intervals, not_in_window_texts))
                segments_and_intervals.sort(key=lambda pair: pair[0][0] if isinstance(pair[0], (list, tuple)) else pair[0])
                cleaned_context = ' '.join([seg for _, seg in segments_and_intervals])
            elif detect_flag and potential_injection is None:
                cleaned_context = ' '.join(not_in_window_texts)
        else:
            if detect_flag and potential_injection is None:
                cleaned_context = ' '.join(not_in_window_texts)
            else:
                segments_and_intervals = []
                segments_and_intervals.extend(zip(top_windows_intervals, top_windows_texts))
                segments_and_intervals.extend(zip(not_in_window_intervals, not_in_window_texts))
                segments_and_intervals.sort(key=lambda pair: pair[0][0] if isinstance(pair[0], (list, tuple)) else pair[0])
                cleaned_context = ' '.join([seg for _, seg in segments_and_intervals])


        cleaned_contexts.append(cleaned_context)

        # Track detect_flag for each sample
        # If injection detected, we'll use warning message instead of querying LLM
        if detect_flag:
            message_batches.append(None)  # Placeholder - won't query LLM for this
        else:
            # Prepare messages for the final model query (batched)
            if isinstance(inst, list):
                message_batches.append([
                    {"role": "system", "content": inst[0]},
                    {"role": "user", "content": f"{cleaned_context}{inst[1]}"},
                ])
            else:
                message_batches.append([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{inst}\n\n{cleaned_context}"},
                ])

    batch_detection_end = time.time()
    print(f"[AgentWatcher Batch] Batch detection time: {batch_detection_end - batch_detection_start:.2f} seconds ({len(target_insts)} samples)")

    # Separate samples that need LLM query from those that don't (injection detected)
    non_detected_indices = [i for i, m in enumerate(message_batches) if m is not None]
    non_detected_messages = [message_batches[i] for i in non_detected_indices]
    
    # Query the LLM in batch only for non-detected samples
    batch_responses = ["[Warning] Detected injected prompt in the context."] * len(message_batches)
    
    if non_detected_messages:
        start_time = time.time()
        if llm is not None and hasattr(llm, "batch_query"):
            llm_responses = llm.batch_query(non_detected_messages)
        elif llm is not None:
            llm_responses = [llm.query(m) for m in non_detected_messages]
        else:
            llm_responses = ["[PIArena] No LLM provided, response is not available."] * len(non_detected_messages)
        end_time = time.time()
        print(f"LLM batch generation time: {end_time - start_time} seconds ({len(non_detected_messages)} samples queried)")
        
        # Map responses back to original indices
        for i, resp in zip(non_detected_indices, llm_responses):
            batch_responses[i] = resp
    else:
        print(f"All {len(message_batches)} samples had injection detected, skipping LLM query")

    # Defense time excludes the final target LLM query (attribution + monitor LLM + cleaning only)
    defense_only_batch_time = batch_detection_end - batch_detection_start
    per_sample_time = defense_only_batch_time / len(target_insts) if target_insts else 0.0

    # Assemble results
    for idx, promptarmor_output in enumerate(promptarmor_outputs):
        results.append({
            "response": batch_responses[idx],
            "cleaned_context": cleaned_contexts[idx],
            "monitor_llm_response": promptarmor_output['monitor_llm_response'],
            "full_msg": promptarmor_output['full_msg'],
            "detect_flag": promptarmor_output.get('detect_flag', False),
            "potential_injection": promptarmor_output.get('potential_injection'),
            "time": per_sample_time,
        })
    return results



def contexts_to_segments(context):
    segment_size = 100

    words = context.split(' ')

    # Create a list to hold segments
    segments = []
    
    # Iterate over the words and group them into segments
    for i in range(0, len(words), segment_size):
        # Join a segment of 100 words and add to segments list
        segment = ' '.join(words[i:i + segment_size])+' '
        segments.append(segment)
    
    return segments

def attribute(system_prompt, target_inst, context, answer, **kwargs):
    global model, tokenizer
    w_s = kwargs.get("w_s", 10)
    w_l = kwargs.get("w_l", 150)
    w_r = kwargs.get("w_r", 50)
    K = kwargs.get("K", 3)
    attribution_model = kwargs.get("attribution_model", "meta-llama/Llama-3.1-8B-Instruct")
    if model is None and tokenizer is None:
        model = AutoModelForCausalLM.from_pretrained(attribution_model, torch_dtype="auto", device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(attribution_model, use_fast=True, trust_remote_code=True)
    layers = range(len(model.model.layers))
    start_time = time.time()
    model.eval()  # Set model to evaluation mode
    if isinstance(target_inst, list):
        context = context.replace("\n\nOBSERVATION:", "")#for wasp dataset
        messages = [
            {"role": "system", "content": target_inst[0]},
            {"role": "user", "content": f"\n\nOBSERVATION: [context]{target_inst[1]}"},
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{target_inst}\n\n[context]"},
        ]
    full_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    prompt_part1, prompt_part2 = full_prompt.split("[context]")
    prompt_part1_ids = tokenizer(prompt_part1, return_tensors="pt").input_ids.to(model.device)[0]
    context_ids = tokenizer(context, return_tensors="pt").input_ids.to(model.device)[0][1:]
    prompt_part2_ids = tokenizer(prompt_part2, return_tensors="pt").input_ids.to(model.device)[0]
    target_ids = tokenizer(answer, return_tensors="pt").input_ids.to(model.device)[0]
    

    input_ids = torch.cat([prompt_part1_ids] + [context_ids] + [prompt_part2_ids, target_ids], dim=-1).unsqueeze(0)
    context_length = len(context_ids)
    prompt_length = len(prompt_part1_ids) + context_length + len(prompt_part2_ids)

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True) 
    hidden_states = outputs.hidden_states
    with torch.no_grad():
        batch_size = 1  
        avg_attentions = None  
        for i in layers:
            attentions = get_attention_weights_one_layer(model, hidden_states, i, attribution_start=prompt_length)
            batch_mean = attentions
            if avg_attentions is None:
                avg_attentions = batch_mean[:, :, :, len(prompt_part1_ids):len(prompt_part1_ids) + context_length]
            else:
                avg_attentions += batch_mean[:, :, :, len(prompt_part1_ids):len(prompt_part1_ids) + context_length]
        avg_attentions = (avg_attentions / (len(layers) / batch_size)).mean(dim=0).mean(dim=(0, 1)).to(torch.float16)
        
    importance_values = avg_attentions.to(torch.float32).cpu().numpy()
    context_ids = np.array(context_ids.cpu().numpy())
    # Decode tokens to readable format

    window_size = w_s
    
    offset_begin = w_l
    offset_end = w_r
    top_k = K
    
    if len(context_ids) < top_k * (window_size + offset_begin + offset_end):
        # Context is short, return the full context as a single window
        top_windows_texts = [tokenizer.decode(context_ids)]
        top_windows_intervals = [(0, len(context_ids))]
        mask = np.zeros(len(context_ids), dtype=bool)  # All tokens are in the top window
        not_in_window_texts = []
        not_in_window_intervals = []
        # The rest of the code should gracefully use these and not process further
        return (
            top_windows_texts, 
            not_in_window_texts,
            top_windows_intervals,   # list of (start, end) for window texts
            not_in_window_intervals  # list of (start, end) for non-window texts
        )
    # Calculate mean importance values for every window
    window_scores = []
    for start in range(0, len(context_ids) - window_size + 1):
        end = start + window_size
        values = np.sort(importance_values[start:end])
        group_mean = np.mean(values)
        window_scores.append((group_mean, start, end))
    
    # Sort and select top_k non-overlapping windows
    window_scores_sorted = sorted(window_scores, key=lambda x: x[0], reverse=True)
    
    # Helper function to check if two intervals overlap
    def intervals_overlap(interval1, interval2):
        return interval1[0] < interval2[1] and interval2[0] < interval1[1]
    
    # Collect non-overlapping top-k windows and record their intervals
    top_windows_texts = []
    top_windows_intervals = []
    for _, start, end in window_scores_sorted:
        # Calculate expanded window boundaries
        start_offset = max(0, start - offset_begin)
        end_offset = min(end + offset_end, len(context_ids))
        candidate_interval = (start_offset, end_offset)
        
        # Check if this window overlaps with any already selected window
        is_overlapping = any(intervals_overlap(candidate_interval, existing) 
                            for existing in top_windows_intervals)
        
        if not is_overlapping:
            window_ids = context_ids[start_offset:end_offset]
            top_windows_texts.append(tokenizer.decode(window_ids))
            top_windows_intervals.append((start_offset, end_offset))
            
            # Stop once we have top_k non-overlapping windows
            if len(top_windows_texts) >= top_k:
                break

    # Build a mask for context_ids not in any of the top windows
    mask = np.ones(len(context_ids), dtype=bool)
    for start, end in top_windows_intervals:
        mask[start:end] = False

    # Now, split masked-out tokens into continuous segments and decode those as "not in top windows"
    not_in_window_texts = []
    not_in_window_intervals = []
    current_start = None
    for idx, flag in enumerate(mask):
        if flag and current_start is None:
            current_start = idx
        if not flag and current_start is not None:
            # End of a segment
            not_in_window_texts.append(tokenizer.decode(context_ids[current_start:idx]))
            not_in_window_intervals.append((current_start, idx))
            current_start = None
    if current_start is not None:
        not_in_window_texts.append(tokenizer.decode(context_ids[current_start:]))
        not_in_window_intervals.append((current_start, len(context_ids)))

    end_time = time.time()
    return (
        top_windows_texts, 
        not_in_window_texts,
        top_windows_intervals,   # list of (start, end) for window texts
        not_in_window_intervals  # list of (start, end) for non-window texts
    )


