"""
A/B test: does the digit-copy corruption seen in diagnose_single_agent.py's
bf16 run disappear when the same model runs in float16 instead?

NOTE: this originally tried float32, but a 7B model in fp32 needs ~28-30GB of
VRAM just for weights — if your GPU has less than that (e.g. a 14-16GB T4),
fp32 will OOM no matter how thoroughly you free memory first; it's a hardware
ceiling, not a bug. float16 uses the SAME memory footprint as bf16 (both are
2 bytes/param) but has 10 mantissa bits vs bf16's 7 — meaningfully more
precision at no extra memory cost, which is enough to test the same
hypothesis (does more mantissa precision fix the digit-copy errors) without
needing 2x the VRAM. GSM8K's numbers are all small-magnitude, so we don't
need bf16's wider exponent range here.

Paste this into a fresh Kaggle cell AFTER running diagnose_single_agent.py once
(so you've seen the bf16 baseline), IN THE SAME KERNEL SESSION so `model` is
still in memory to be freed — or just restart the kernel and run this alone
(it will load its own tokenizer if `tokenizer` isn't already in globals). This
script:
  1. Frees the existing bf16 model from GPU memory (with a verified check).
  2. Reloads the SAME model in float16.
  3. Re-runs the exact same 10 GSM8K test questions (same order, same prompt).
  4. Prints, per sample, the bf16 result (hardcoded from your last run) next to
     the fp16 result, and flags whether each known digit-mismatch got fixed.

If the mismatches disappear in fp16 -> bf16 precision is a real, confirmed
contributing cause. If they persist -> it's not a precision issue, and the
model is genuinely misreading/miscopying at these points regardless of dtype.
"""

import gc
import re
import torch

# ── hardcoded bf16 baseline from your last diagnose_single_agent.py run ──
# (idx, correct, pred, mismatched_question_numbers)
BF16_RESULTS = {
    0: {"correct": True,  "pred": "18",     "mismatch": []},
    1: {"correct": True,  "pred": "3",      "mismatch": []},
    2: {"correct": False, "pred": "62900",  "mismatch": ["50000"]},
    3: {"correct": False, "pred": "585",    "mismatch": []},
    4: {"correct": False, "pred": "19",     "mismatch": ["20"]},
    5: {"correct": False, "pred": "66.4",   "mismatch": []},
    6: {"correct": False, "pred": "26",     "mismatch": ["20"]},
    7: {"correct": False, "pred": "181.84", "mismatch": ["200"]},
    8: {"correct": False, "pred": "41.5",   "mismatch": ["30"]},
    9: {"correct": False, "pred": "489",    "mismatch": []},
}

# ── 1. free the existing bf16 model, verifying it actually released VRAM ──
if "model" in globals():
    print("[precision_ab] Freeing existing (bf16) model from GPU memory...")
    before_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    del model
    for name in ("pipe", "pipeline_obj", "corrector", "anchor_table"):
        # in case any other notebook variable holds a reference and keeps the
        # model alive, drop the common ones so gc can actually collect it
        if name in globals():
            del globals()[name]
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    after_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"[precision_ab] GPU memory allocated: {before_mb:.0f} MB -> {after_mb:.0f} MB after cleanup")
    if after_mb > 2000:
        print("[precision_ab] WARNING: >2GB still allocated after cleanup — something else in this "
              "notebook still references the old model (or its optimizer/generation state). "
              "If the load below OOMs, restart the kernel and run ONLY this script fresh.")

free_mb, total_mb = (x / (1024 ** 2) for x in torch.cuda.mem_get_info())
print(f"[precision_ab] GPU free/total: {free_mb:.0f} MB / {total_mb:.0f} MB")
# Qwen2.5-7B in fp16 needs ~14-15GB for weights alone, plus activation memory.
if free_mb < 16000:
    print("[precision_ab] WARNING: less than ~16GB free — this may still OOM. If it does, "
          "restart the kernel and run this script in a clean session (no bf16 model loaded first).")

# ── 2. reload in float16 (same footprint as bf16, more mantissa precision) ─
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

if "tokenizer" not in globals():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

print(f"[precision_ab] Loading {MODEL_NAME} in float16 "
      "(same memory footprint as bf16, more mantissa precision)...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="cuda",
        attn_implementation="eager",
        trust_remote_code=True,
    )
except TypeError:
    # older transformers versions use torch_dtype instead of dtype
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="eager",
        trust_remote_code=True,
    )
model.eval()
print(f"[precision_ab] model dtype: {next(model.parameters()).dtype}")

# ── 3. same 10 samples, same prompt logic as diagnose_single_agent.py ─────
N_SAMPLES = 10
from datasets import load_dataset
ds = load_dataset("gsm8k", "main")
samples = []
for item in ds["test"]:
    gold = item["answer"].split("####")[-1].strip()
    samples.append({"question": item["question"], "answer": gold})
    if len(samples) >= N_SAMPLES:
        break

SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "Solve carefully and return final answer as #### [number]"
)


def extract_numbers(text):
    return [n.replace(",", "") for n in re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)]


def extract_answer(text):
    if not text:
        return None
    for pat in [
        r"####\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)",
        r"\\boxed\{\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)\s*\}",
        r"(?i)answer\s*(?:is|=|:)?\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)",
    ]:
        m = re.findall(pat, text)
        if m:
            return m[-1].replace(",", "")
    nums = extract_numbers(text)
    return nums[-1] if nums else None


def find_digit_mismatches(question, reasoning):
    q_nums = set(extract_numbers(question))
    r_nums = set(extract_numbers(reasoning))
    return sorted(q_nums - r_nums)


# ── 4. re-run and compare against bf16 baseline ───────────────────────────
fixed_count = 0
still_broken_count = 0
newly_broken_count = 0

for i, s in enumerate(samples):
    question, gold = s["question"], s["answer"]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(model.device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0, input_ids.shape[1]:]
    raw_answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # fp16 has a real, separate failure mode independent of the cache-collision
    # bug fixed in cec7714: activations overflowing fp16's ~65504 max range
    # produce NaN/Inf, which typically shows up as degenerate punctuation spam
    # ("!!!!...") rather than a digit-copy error. Flag it explicitly so it's
    # never confused with the precision-mantissa hypothesis this script tests.
    overflow_spam = bool(re.search(r"[!?]{6,}", raw_answer))

    pred = extract_answer(raw_answer)
    correct = pred is not None and pred.strip() == gold
    mismatches = find_digit_mismatches(question, raw_answer)

    base = BF16_RESULTS[i]
    print("\n" + "=" * 100)
    print(f"SAMPLE {i}  |  bf16: correct={base['correct']} pred={base['pred']} mismatch={base['mismatch']}")
    print(f"           |  fp16: correct={correct} pred={pred} mismatch={mismatches}")
    if overflow_spam:
        print("  -> WARNING: '!!!!'-style degenerate punctuation spam detected in fp16 output — "
              "this looks like activation overflow (NaN/Inf), a DIFFERENT fp16 failure mode than "
              "the digit-copy issue this script is testing for. Not the same bug as the historical "
              "cache-collision spam (already fixed in cec7714) — if you see this, it means fp16 "
              "itself is unstable for this model regardless of the cache bug.")

    if base["mismatch"] and not mismatches:
        print("  -> FIXED by fp16: the digit-copy error is gone.")
        fixed_count += 1
    elif base["mismatch"] and mismatches:
        print("  -> STILL BROKEN in fp16: same class of error persists (not a precision issue).")
        still_broken_count += 1
    elif not base["mismatch"] and mismatches:
        print("  -> NEW mismatch introduced in fp16 (unexpected — investigate).")
        newly_broken_count += 1

    if not correct:
        print(f"\n  [fp16 full reasoning]\n{raw_answer}\n")

print("\n" + "=" * 100)
print("PRECISION A/B SUMMARY")
print("=" * 100)
print(f"bf16 digit-mismatch samples that were FIXED in fp16:        {fixed_count}")
print(f"bf16 digit-mismatch samples that are STILL BROKEN in fp16:  {still_broken_count}")
print(f"NEW mismatches introduced only in fp16:                     {newly_broken_count}")
print("\nIf 'FIXED' is high and 'STILL BROKEN' is low: bf16 precision is confirmed as a")
print("real cause of the digit-copy corruption -> worth switching the eval to fp16 (same")
print("memory cost as bf16, just more mantissa precision) or at minimum flagging bf16 as")
print("a known limitation of the sweep.")
print("If 'STILL BROKEN' is high: precision isn't the cause -> look elsewhere (prompting,")
print("model capability, or the eager attention_implementation itself).")
