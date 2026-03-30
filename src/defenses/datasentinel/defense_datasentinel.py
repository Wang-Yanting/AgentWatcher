import time

from ...llm import Model
from .OpenPromptInjection.apps import DataSentinelDetector

DEFAULT_CONFIG = {
    "model_info":{
        "provider":"mistral",
        "name":"mistralai/Mistral-7B-v0.1"
    },
    "api_key_info":{
        "api_keys":[0],
        "api_key_use": 0
    },
    "params":{
        "temperature":0.1,
        "seed":100,
        "gpus":["5","6"],
        "device":"cuda",
        "max_output_tokens":128,
        "ft_path":"sleeepeer/DataSentinel",
        "decoding_method":"greedy"
    }
}

DETECTOR = None
def get_detector(model_config=DEFAULT_CONFIG):
    detector = DataSentinelDetector(model_config)
    return detector

def datasentinel(
    target_inst,
    context=None,
    system_prompt:str="You are a helpful assistant.",
    llm:Model=None,
    config=DEFAULT_CONFIG,
):
    global DETECTOR
    if DETECTOR is None:
        DETECTOR = get_detector(model_config=config)

    start = time.time()
    detect_flag = DETECTOR.detect(context)
    defense_time = time.time() - start
    print("datasentinel detect_flag:",detect_flag)
    if llm is not None:
        if detect_flag:
            response = "[Warning] DataSentinel detected injected prompt in the context."
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
            response = llm.query(messages)
    else:
        response = "[PIArena] No LLM provided, response is not available."

    return {
        "response": response,
        "detect_flag": detect_flag,
        "time": defense_time,
    }


def datasentinel_batch(
    target_insts,
    contexts=None,
    system_prompt: str = "You are a helpful assistant.",
    llm: Model = None,
    config=DEFAULT_CONFIG,
):
    global DETECTOR
    if DETECTOR is None:
        DETECTOR = get_detector(model_config=config)

    if contexts is None:
        contexts = [None] * len(target_insts)

    batch_size = len(target_insts)
    print(f"[DataSentinel Batch] Batch size: {batch_size}")

    detect_flags = []
    for i, ctx in enumerate(contexts):
        ctx_str = ctx if ctx is not None else ""
        flag = DETECTOR.detect(ctx_str)
        detect_flags.append(flag)
        if (i + 1) % 50 == 0:
            print(f"[DataSentinel Batch] Detection progress: {i + 1}/{batch_size}")

    detected_count = sum(detect_flags)
    print(f"[DataSentinel Batch] Detected: {detected_count}/{batch_size}")

    not_detected_indices = [i for i, flag in enumerate(detect_flags) if not flag]

    responses = [""] * batch_size
    for i in range(batch_size):
        if detect_flags[i]:
            responses[i] = "[Warning] DataSentinel detected injected prompt in the context."

    if not_detected_indices and llm is not None:
        messages_batch = []
        for i in not_detected_indices:
            inst = target_insts[i]
            ctx = contexts[i] if contexts[i] is not None else ""
            if isinstance(inst, (list, tuple)) and len(inst) >= 2:
                messages = [
                    {"role": "system", "content": inst[0]},
                    {"role": "user", "content": f"{ctx}{inst[1]}"},
                ]
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{inst}\n\n{ctx}"},
                ]
            messages_batch.append(messages)

        if hasattr(llm, "batch_query"):
            print(f"[DataSentinel Batch] Using vLLM batch inference for {len(not_detected_indices)} samples")
            batch_responses = llm.batch_query(messages_batch)
        else:
            print(f"[DataSentinel Batch] Using sequential inference")
            batch_responses = [llm.query(m) for m in messages_batch]

        for idx, resp in zip(not_detected_indices, batch_responses):
            responses[idx] = resp
    elif llm is None:
        for i in not_detected_indices:
            responses[i] = "[PIArena] No LLM provided, response is not available."

    results = []
    for resp, flag in zip(responses, detect_flags):
        results.append({
            "response": resp,
            "detect_flag": flag,
        })

    print(f"[DataSentinel Batch] Done: {batch_size} samples processed")
    return results