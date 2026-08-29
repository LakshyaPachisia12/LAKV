"""
Diagnostic cell — inspect single_agent step-by-step to find why accuracy is low.

Paste this whole block into a Kaggle notebook cell. It will reuse `model` /
`tokenizer` if they already exist in your notebook's global namespace (from an
earlier cell), otherwise it loads Qwen2.5-7B-Instruct fresh the same way run.py
does.

What it prints, per sample:
  - the exact question text pulled from the dataset
  - the full chat-templated prompt string sent to the model
  - token count of that prompt
  - the model's real end-of-turn token ids (generation_config.eos_token_id)
  - the FULL raw decoded chain-of-thought the model generated
  - every number in the question vs every number in the model's own reasoning,
    flagging any question-number that got misquoted/altered by the model
  - extracted answer vs gold, correct/incorrect

At the end: aggregate accuracy + how many samples show a "digit mismatch"
(the model writing down a different number than what the question said).
"""

import re
import torch

# ── 1. reuse or load model/tokenizer ──────────────────────────────────────
if "model" not in globals() or "tokenizer" not in globals():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
    print(f"[diagnose] Loading {MODEL_NAME} fresh (not found in globals)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()
else:
    print("[diagnose] Reusing model/tokenizer already in notebook globals.")

print(f"[diagnose] model dtype: {next(model.parameters()).dtype}")
print(f"[diagnose] tokenizer.eos_token_id: {tokenizer.eos_token_id}")
print(f"[diagnose] model.generation_config.eos_token_id: {model.generation_config.eos_token_id}")

# ── 2. load a handful of GSM8K test questions ─────────────────────────────
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
    """Numbers in the question that never appear anywhere in the model's own
    reasoning text — i.e. the model silently substituted a different number."""
    q_nums = set(extract_numbers(question))
    r_nums = set(extract_numbers(reasoning))
    return sorted(q_nums - r_nums)


# ── 3. run single_agent manually, step by step, with full visibility ─────
results = []
for i, s in enumerate(samples):
    print("\n" + "=" * 100)
    print(f"SAMPLE {i}")
    print("=" * 100)

    question = s["question"]
    gold = s["answer"]
    print(f"\n[QUESTION]\n{question}")
    print(f"\n[GOLD ANSWER] {gold}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print(f"\n[FULL PROMPT SENT TO MODEL]\n{prompt_text}")

    input_ids = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(model.device)
    print(f"\n[PROMPT TOKEN COUNT] {input_ids.shape[1]}")

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

    print(f"\n[FULL MODEL REASONING]\n{raw_answer}")

    pred = extract_answer(raw_answer)
    correct = pred is not None and pred.strip() == gold
    mismatches = find_digit_mismatches(question, raw_answer)

    print(f"\n[EXTRACTED ANSWER] {pred}")
    print(f"[CORRECT?] {correct}")
    if mismatches:
        print(f"[DIGIT MISMATCH] Question numbers never referenced in reasoning: {mismatches}")
        print("  -> model may have silently substituted/misread these while reasoning")
    else:
        print("[DIGIT MISMATCH] none — all question numbers appear somewhere in the reasoning")

    results.append({
        "idx": i, "correct": correct, "has_mismatch": bool(mismatches),
        "gold": gold, "pred": pred,
    })

# ── 4. summary ─────────────────────────────────────────────────────────────
n = len(results)
n_correct = sum(r["correct"] for r in results)
n_mismatch = sum(r["has_mismatch"] for r in results)
n_wrong_with_mismatch = sum(r["has_mismatch"] and not r["correct"] for r in results)

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)
print(f"Accuracy: {n_correct}/{n} ({100*n_correct/n:.1f}%)")
print(f"Samples with a digit mismatch (model altered a question number): {n_mismatch}/{n}")
print(f"Of the WRONG answers, how many had a digit mismatch: {n_wrong_with_mismatch}/{n - n_correct if n - n_correct else 1}")
print("\nIf digit-mismatch rate is high among wrong answers, the model is genuinely")
print("misreading/altering numbers mid-reasoning (not an extraction/grading bug).")
print("Next step to isolate precision as a cause: reload the model with")
print("torch_dtype=torch.float32 (no attn_implementation change) and re-run the")
print("SAME wrong samples above — if the mismatches disappear, bf16 precision")
print("is a real contributing factor; if they persist, it's a prompting/model")
print("capability issue instead.")
