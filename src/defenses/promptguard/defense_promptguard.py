import time

from ...llm import Model
from transformers import pipeline

DEFAULT_CONFIG = {}

DETECTOR = None
def get_detector():
    classifier = pipeline("text-classification", model="meta-llama/Prompt-Guard-86M", device_map="auto")
    return classifier

def promptguard(
    target_inst,
    context=None,
    system_prompt:str="You are a helpful assistant.",
    llm:Model=None,
    config=DEFAULT_CONFIG,
):
    global DETECTOR
    if DETECTOR is None:
        DETECTOR = get_detector()

    start = time.time()
    prediction = DETECTOR(context)[0]["label"]
    detect_flag = ("benign" not in prediction.lower())
    defense_time = time.time() - start

    if llm is not None:
        if detect_flag:
            response = "[Warning] PromptGuard detected injected prompt in the context."
        else:
            if isinstance(target_inst, (list, tuple)) and len(target_inst) >= 2:
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


def promptguard_batch(
    target_insts,
    contexts=None,
    system_prompt: str = "You are a helpful assistant.",
    llm: Model = None,
    config=DEFAULT_CONFIG,
):
    global DETECTOR
    if DETECTOR is None:
        DETECTOR = get_detector()

    if contexts is None:
        contexts = [None] * len(target_insts)

    batch_size = len(target_insts)
    print(f"[PromptGuard Batch] Batch size: {batch_size}")

    raw_contexts = [ctx if ctx is not None else "" for ctx in contexts]

    try:
        batch_preds = DETECTOR(raw_contexts, truncation=True)
    except RuntimeError:
        batch_preds = []
        for ctx in raw_contexts:
            try:
                pred = DETECTOR(ctx, truncation=True)
                batch_preds.append(pred[0] if isinstance(pred, list) else pred)
            except RuntimeError:
                batch_preds.append({"label": "injected"})

    detect_flags = []
    for pred in batch_preds:
        if isinstance(pred, list):
            label = pred[0]["label"]
        else:
            label = pred["label"]
        detect_flags.append("benign" not in label.lower())

    detected_count = sum(detect_flags)
    print(f"[PromptGuard Batch] Detected: {detected_count}/{batch_size}")

    not_detected_indices = [i for i, flag in enumerate(detect_flags) if not flag]

    responses = [""] * batch_size
    for i in range(batch_size):
        if detect_flags[i]:
            responses[i] = "[Warning] PromptGuard detected injected prompt in the context."

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
            print(f"[PromptGuard Batch] Using vLLM batch inference for {len(not_detected_indices)} samples")
            batch_responses = llm.batch_query(messages_batch)
        else:
            print(f"[PromptGuard Batch] Using sequential inference")
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

    print(f"[PromptGuard Batch] Done: {batch_size} samples processed")
    return results
