from ..llm import Model
import time

DEFAULT_CONFIG = {}

def no_defense(
    target_inst,
    context=None,
    system_prompt:str="You are a helpful assistant.",
    llm:Model=None,
    config=DEFAULT_CONFIG,
):  
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
    print("full message: ", messages)
    if llm is not None:
        response = llm.query(messages)
    else:
        response = "[PIArena] No LLM provided, response is not available."

    return {
        "response": response,
    }


def no_defense_batch(
    target_insts,
    contexts=None,
    system_prompt: str = "You are a helpful assistant.",
    llm: Model = None,
    config=DEFAULT_CONFIG,
):
    """
    Batch version of no_defense, utilizing vLLM batch_query for high throughput.
    
    Args:
        target_insts: List of user instructions (each can be a tuple/list of [system_prompt, user_content]
                      or a string for the user content).
        contexts: List of contexts (one per instruction). If None, uses empty context for all.
        system_prompt: Default system prompt if target_inst is a string.
        llm: LLM object for querying model (should support batch_query for best performance).
        config: Additional config (unused by default).
    
    Returns:
        List of dicts, each containing:
          - response: The LLM's response.
    """
    # Convert to empty list if argument was None
    if contexts is None:
        contexts = [None] * len(target_insts)
    
    batch_size = len(target_insts)
    print(f"[No Defense Batch] Batch size: {batch_size}")
    
    batch_start_time = time.time()
    
    # Build all messages
    all_messages = []
    for inst, ctx in zip(target_insts, contexts):
        ctx_str = ctx if ctx is not None else ""
        
        if isinstance(inst, (list, tuple)) and len(inst) >= 2:
            # inst is [system_prompt, user_content]
            messages = [
                {"role": "system", "content": inst[0]},
                {"role": "user", "content": f"{ctx_str}{inst[1]}"},
            ]
        else:
            # inst is just the user content string
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{inst}\n\n{ctx_str}"},
            ]
        all_messages.append(messages)
    
    # Query LLM - use batch_query if available (vLLM), otherwise fall back to sequential
    query_start = time.time()
    if llm is not None:
        if hasattr(llm, "batch_query"):
            print(f"[No Defense Batch] Using vLLM batch inference")
            responses = llm.batch_query(all_messages)
        else:
            print(f"[No Defense Batch] Using sequential inference (transformers)")
            responses = []
            for i, m in enumerate(all_messages):
                resp = llm.query(m)
                responses.append(resp)
                if (i + 1) % 50 == 0:
                    print(f"[No Defense Batch] Progress: {i + 1}/{batch_size} samples processed")
    else:
        responses = ["[PIArena] No LLM provided, response is not available."] * batch_size
    
    query_end = time.time()
    print(f"[No Defense Batch] LLM query time: {query_end - query_start:.2f} seconds")
    
    batch_end_time = time.time()
    print(f"[No Defense Batch] Total batch time: {batch_end_time - batch_start_time:.2f} seconds")
    
    # Assemble results
    results = []
    for response in responses:
        results.append({
            "response": response,
        })
    
    return results