"""
Direct inspection of bf16 token-level logits at a known digit-copy failure.

The fp16 A/B test came back inconclusive for the original question (does bf16
precision cause "20" -> "2" style digit-copy errors?) because fp16 collapsed
entirely (NaN/overflow, the classic "!!!!" spam) rather than just being "more
precise" - so it can't serve as a clean comparison. fp32 doesn't fit on this
GPU. Instead of switching dtype, this script inspects bf16's own logits at
the exact step where a known failure happens, so we can see directly whether
the correct digit was a close second-place candidate (supports precision as
a contributing cause) or nowhere close (rules it out, points to something
else - e.g. attention not reliably retrieving that token's context).

Paste into a Kaggle cell (reuses `model`/`tokenizer` if already loaded in
bf16, otherwise loads fresh). Runs the Wendi chickens question (GSM8K test
sample idx 4, "Wendi's flock is 20 chickens") which reliably produces "Wendi
has 2 chickens" in bf16 instead of correctly copying "20". Greedy-decodes
token by token, printing the top-5 candidates + their probabilities at every
step, so you can scroll to the exact point the digit gets picked and see the
margin with your own eyes.
"""

import torch

# ── 1. reuse or load model/tokenizer (bf16, matches the original run) ─────
if "model" not in globals() or "tokenizer" not in globals():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
    print(f"[logit_margins] Loading {MODEL_NAME} in bf16 (not found in globals)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()
else:
    print("[logit_margins] Reusing model/tokenizer already in notebook globals.")
    print(f"[logit_margins] current model dtype: {next(model.parameters()).dtype}")

QUESTION = (
    "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, "
    "containing seeds, mealworms and vegetables to help keep them healthy. She gives "
    "the chickens their feed in three separate meals. In the morning, she gives her "
    "flock of chickens 15 cups of feed. In the afternoon, she gives her chickens "
    "another 25 cups of feed. How many cups of feed does she need to give her chickens "
    "in the final meal of the day if the size of Wendi's flock is 20 chickens?"
)
SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "Solve carefully and return final answer as #### [number]"
)
N_STEPS_TO_INSPECT = 140  # 70 wasn't enough — model was still mid-setup ("1. ... 2. Calculate...")
                          # at step 70; the flock-size copy happens later, inside step 2's detail.

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": QUESTION},
]
prompt_text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
input_ids = tokenizer(
    prompt_text, return_tensors="pt", add_special_tokens=False
)["input_ids"].to(model.device)

print(f"\n[logit_margins] Prompt token count: {input_ids.shape[1]}")
print("[logit_margins] Greedy-decoding step by step, printing top-5 candidates at each step.")
print("[logit_margins] Watch for the step where 'Wendi has' is followed by a number token —")
print("[logit_margins] that's the digit-copy moment we care about.\n")

with torch.no_grad():
    out = model(input_ids=input_ids, use_cache=True)
past_key_values = out.past_key_values
next_logits = out.logits[:, -1, :]

generated_ids = []
for step in range(N_STEPS_TO_INSPECT):
    probs = torch.softmax(next_logits.float(), dim=-1)[0]
    top5_probs, top5_ids = torch.topk(probs, k=5)

    chosen_id = int(top5_ids[0].item())
    chosen_tok = tokenizer.decode([chosen_id])

    candidates = [
        f"{tokenizer.decode([int(tid)])!r}:{float(p):.4f}"
        for tid, p in zip(top5_ids, top5_probs)
    ]
    print(f"step {step:3d} | chosen={chosen_tok!r:12s} | top5: {', '.join(candidates)}")

    if chosen_id in (tokenizer.eos_token_id, *(
        model.generation_config.eos_token_id
        if isinstance(model.generation_config.eos_token_id, (list, tuple))
        else [model.generation_config.eos_token_id]
    )):
        print("  -> EOS reached, stopping early.")
        break

    generated_ids.append(chosen_id)
    next_token = top5_ids[0].view(1, 1)
    with torch.no_grad():
        out = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
        )
    past_key_values = out.past_key_values
    next_logits = out.logits[:, -1, :]

print("\n[logit_margins] Full text generated so far:")
print(tokenizer.decode(generated_ids, skip_special_tokens=True))

print("\n[logit_margins] How to read this:")
print("  - Find the step where the model outputs a digit right after 'Wendi has'.")
print("  - Look at that step's top-5 list: is '20' (or its first token, likely '2' or '20')")
print("    present as a close second/third candidate with a probability near the chosen")
print("    token's? That would support precision/rounding flipping a close call.")
print("  - If the wrong digit has e.g. 90%+ probability and the right one isn't even in the")
print("    top 5, that rules out a close-call precision flip — the model confidently picked")
print("    the wrong token, which points to something else (attention not retrieving the")
print("    right context, or a genuine model weakness on this kind of copy task).")
