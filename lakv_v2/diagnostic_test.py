"""
Diagnostic test to isolate where the ! spam comes from.

This tests each stage independently:
1. Solver greedy loop WITHOUT cache injection (baseline)
2. Solver greedy loop + cache reconstruction (does reconstruction corrupt it?)
3. Full pipeline with explicit position_ids
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

_HERE = Path(__file__).parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

QUESTION = "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes muffins for her friends every day with 4933. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"


def test_baseline_solver(model, tokenizer, device):
    """Pure solver greedy decode without any cache manipulation."""
    print("\n" + "="*60)
    print("TEST 1: Baseline Solver (no cache reconstruction)")
    print("="*60)

    messages = [
        {"role": "system", "content": "You are a math assistant."},
        {"role": "user", "content": QUESTION},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=64,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )

    result = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"Output: {result[:200]}")
    has_spam = "!!!!" in result or "!!!!" in result or result.count("!") > 8
    print(f"Has ! spam: {has_spam}")
    return result


def test_cache_reconstruction(model, tokenizer, device):
    """Test if cache reconstruction itself corrupts the cache."""
    print("\n" + "="*60)
    print("TEST 2: Manual greedy loop with proper position_ids (no reconstruction)")
    print("="*60)

    messages = [
        {"role": "system", "content": "You are a math assistant."},
        {"role": "user", "content": QUESTION},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prefix_ids = inputs["input_ids"]

    # Forward on prefix
    with torch.no_grad():
        prefix_out = model(input_ids=prefix_ids, use_cache=True)

    prefix_len = prefix_ids.shape[1]
    running_cache = prefix_out.past_key_values
    next_logits = prefix_out.logits[:, -1, :]

    eos_id = tokenizer.eos_token_id
    generated_ids = []
    cur_pos = prefix_len

    for step in range(64):
        next_token = next_logits.argmax(-1, keepdim=True)
        tok_id = next_token.item()
        if tok_id == eos_id:
            break
        generated_ids.append(tok_id)

        # Explicit position_ids
        position_ids_t = torch.tensor([[cur_pos]], device=device, dtype=torch.long)

        with torch.no_grad():
            out = model(
                input_ids=next_token,
                past_key_values=running_cache,
                position_ids=position_ids_t,
                use_cache=True,
            )

        running_cache = out.past_key_values
        next_logits = out.logits[:, -1, :]
        cur_pos += 1

    result = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"Output: {result[:200]}")
    has_spam = "!!!!" in result or result.count("!") > 8
    print(f"Has ! spam: {has_spam}")
    return result


def test_with_cache_tuple_roundtrip(model, tokenizer, device):
    """Test: does converting cache to tuple and back corrupt it?"""
    print("\n" + "="*60)
    print("TEST 3: Cache tuple roundtrip (convert → reconstruct)")
    print("="*60)

    messages = [
        {"role": "system", "content": "You are a math assistant."},
        {"role": "user", "content": QUESTION},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prefix_ids = inputs["input_ids"]

    # Forward on prefix
    with torch.no_grad():
        prefix_out = model(input_ids=prefix_ids, use_cache=True)

    prefix_len = prefix_ids.shape[1]

    # Convert to tuple (simulate what pipeline does)
    if hasattr(prefix_out.past_key_values, "to_legacy_cache"):
        kv_tuple = prefix_out.past_key_values.to_legacy_cache()
    else:
        kv_tuple = tuple(
            (layer[0], layer[1]) if isinstance(layer, tuple) else (layer.keys, layer.values)
            for layer in prefix_out.past_key_values
        )

    # Convert back to DynamicCache (POTENTIAL BUG HERE)
    if hasattr(DynamicCache, "from_legacy_cache"):
        reconstructed_cache = DynamicCache.from_legacy_cache(kv_tuple)
    else:
        reconstructed_cache = DynamicCache(kv_tuple)

    # Check if _seen_tokens is properly set
    print(f"DynamicCache._seen_tokens after reconstruction: {reconstructed_cache._seen_tokens if hasattr(reconstructed_cache, '_seen_tokens') else 'N/A'}")
    print(f"Expected sequence length: {prefix_len}")

    # Continue from reconstructed cache with explicit position_ids
    next_logits = prefix_out.logits[:, -1, :]
    running_cache = reconstructed_cache

    generated_ids = []
    cur_pos = prefix_len

    for step in range(64):
        next_token = next_logits.argmax(-1, keepdim=True)
        tok_id = next_token.item()
        if tok_id == tokenizer.eos_token_id:
            break
        generated_ids.append(tok_id)

        position_ids_t = torch.tensor([[cur_pos]], device=device, dtype=torch.long)

        with torch.no_grad():
            out = model(
                input_ids=next_token,
                past_key_values=running_cache,
                position_ids=position_ids_t,
                use_cache=True,
            )

        running_cache = out.past_key_values
        next_logits = out.logits[:, -1, :]
        cur_pos += 1

    result = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"Output: {result[:200]}")
    has_spam = "!!!!" in result or result.count("!") > 8
    print(f"Has ! spam: {has_spam}")
    return result


def main():
    parser = argparse.ArgumentParser(description="LAKV diagnostic test")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"[diagnostic] Loading model: {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()
    print("[diagnostic] Model loaded.\n")

    test_baseline_solver(model, tokenizer, args.device)
    test_cache_reconstruction(model, tokenizer, args.device)
    test_with_cache_tuple_roundtrip(model, tokenizer, args.device)

    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)
    print("\nInterpretation:")
    print("- If TEST 1 has spam: model has inherent issues with greedy decode")
    print("- If TEST 2 has spam: position_ids fix didn't work or RoPE is broken")
    print("- If TEST 3 has spam: cache tuple roundtrip corrupts the cache")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
