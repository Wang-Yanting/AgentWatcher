import os
import time

from ...llm import Model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

DEFAULT_CONFIG = {}

DETECTOR = None
def get_detector():
    tokenizer = AutoTokenizer.from_pretrained("leolee99/PIGuard", cache_dir=os.getenv("HF_HOME"))
    model = AutoModelForSequenceClassification.from_pretrained("leolee99/PIGuard", cache_dir=os.getenv("HF_HOME"), trust_remote_code=True, device_map="auto")

    classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    truncation=True,
    max_length=2048,
    )
    return classifier

def piguard(
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
    pred_result = DETECTOR(context)[0]
    prediction = pred_result["label"]
    score = pred_result["score"]
    defense_time = time.time() - start

    if prediction == "benign":
        detect_flag = False
    elif prediction == "injection":
        detect_flag = True
    else:
        raise ValueError(f"Unknown prediction: {prediction}")

    if llm is not None:
        if detect_flag:
            response = "[Warning] PIGuard detected injected prompt in the context."
        else: # benign
            messages = [
                {"role": "system", "content": target_inst[0]},
                {"role": "user", "content": f"{context}{target_inst[1]}"},
            ]
            response = llm.query(messages)
    else:
        response = "[PIArena] No LLM provided, response is not available."

    return {
        "response": response,
        "detect_flag": detect_flag,
        "score": score,
        "time": defense_time,
    }


def piguard_batch(
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
    print(f"[PIGuard Batch] Batch size: {batch_size}")

    raw_contexts = [ctx if ctx is not None else "" for ctx in contexts]

    detect_flags = []
    scores = []
    for i, ctx in enumerate(raw_contexts):
        result = DETECTOR(ctx)
        prediction = result[0]["label"]
        score = result[0]["score"]
        scores.append(score)
        detect_flags.append(prediction == "injection")
        if (i + 1) % 50 == 0:
            print(f"[PIGuard Batch] Detection progress: {i + 1}/{batch_size}")

    detected_count = sum(detect_flags)
    print(f"[PIGuard Batch] Detected: {detected_count}/{batch_size}")

    not_detected_indices = [i for i, flag in enumerate(detect_flags) if not flag]

    responses = [""] * batch_size
    for i in range(batch_size):
        if detect_flags[i]:
            responses[i] = "[Warning] PIGuard detected injected prompt in the context."

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
            print(f"[PIGuard Batch] Using vLLM batch inference for {len(not_detected_indices)} samples")
            batch_responses = llm.batch_query(messages_batch)
        else:
            print(f"[PIGuard Batch] Using sequential inference")
            batch_responses = [llm.query(m) for m in messages_batch]

        for idx, resp in zip(not_detected_indices, batch_responses):
            responses[idx] = resp
    elif llm is None:
        for i in not_detected_indices:
            responses[i] = "[PIArena] No LLM provided, response is not available."

    results = []
    for resp, flag, score in zip(responses, detect_flags, scores):
        results.append({
            "response": resp,
            "detect_flag": flag,
            "score": score,
        })

    print(f"[PIGuard Batch] Done: {batch_size} samples processed")
    return results