"""
Checks the Reasoner's CURRENT prompt (system prompt + few-shot exemplar, both
pulled live from lakv/pipeline.py's PipelineConfig, not hardcoded here) against
the digit-misread question that used to trigger both the chat-turn
hallucination ("Let\\nuser\\nLet\\nassistant...") and the "20 chickens" ->
"2 chickens" misread.

Earlier versions of this script hardcoded the Reasoner's OLD system prompt
text directly, which meant re-running it after later fixes (prompt reword in
3f3336b, few-shot exemplar) silently re-tested stale wording and reproduced
already-fixed bugs. This version imports PipelineConfig directly so it always
reflects whatever pipeline.py currently ships, matching PIPELINE.run()'s
exact message construction (including the few-shot user/assistant turns for
agent_idx == 0 when use_reasoner_few_shot is True).

Paste into a Kaggle cell from the repo root (reuses `model`/`tokenizer` if
already loaded in bf16, otherwise loads fresh). Cheap: ~60 generated tokens x
2 prompts, seconds on GPU, not the full sweep.
"""

import torch
from lakv.pipeline import PipelineConfig

if "model" not in globals() or "tokenizer" not in globals():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
    print(f"[reasoner_prompt] Loading {MODEL_NAME} in bf16 (not found in globals)...")
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
    print("[reasoner_prompt] Reusing model/tokenizer already in notebook globals.")
    print(f"[reasoner_prompt] current model dtype: {next(model.parameters()).dtype}")

QUESTION = (
    "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, "
    "containing seeds, mealworms and vegetables to help keep them healthy. She gives "
    "the chickens their feed in three separate meals. In the morning, she gives her "
    "flock of chickens 15 cups of feed. In the afternoon, she gives her chickens "
    "another 25 cups of feed. How many cups of feed does she need to give her chickens "
    "in the final meal of the day if the size of Wendi's flock is 20 chickens?"
)

# Build the exact message list pipeline.py's run() would build for each
# agent, live from the current PipelineConfig - no hardcoded prompt text here.
_cfg = PipelineConfig(
    use_layer_selection=False, compression_mode="none",
    use_offset_correction=False, reconstruction_strategy="zeros",
)
_reasoner_system = _cfg.system_prompts[0]

PROMPT_MESSAGES = {
    "single_agent (simple)": [
        {"role": "system", "content": (
            "You are a mathematical reasoning assistant. "
            "Solve carefully and return final answer as #### [number]"
        )},
        {"role": "user", "content": QUESTION},
    ],
    "Reasoner (pipeline.py agent 0, current config)": (
        [{"role": "system", "content": _reasoner_system}]
        + (
            [
                {"role": "user", "content": _cfg.reasoner_few_shot_example[0]},
                {"role": "assistant", "content": _cfg.reasoner_few_shot_example[1]},
            ]
            if _cfg.use_reasoner_few_shot else []
        )
        + [{"role": "user", "content": QUESTION}]
    ),
}
print(f"[reasoner_prompt] use_reasoner_few_shot = {_cfg.use_reasoner_few_shot}")
print(f"[reasoner_prompt] Reasoner system prompt: {_reasoner_system!r}")

N_STEPS = 140  # 60 wasn't enough - "Wendi has [N] chickens" moment comes later,
               # same as diagnose_logit_margins.py originally needed 140 not 70

eos_ids = set()
gen_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
if gen_eos is not None:
    eos_ids.update(gen_eos if isinstance(gen_eos, (list, tuple, set)) else [gen_eos])
if tokenizer.eos_token_id is not None:
    eos_ids.add(tokenizer.eos_token_id)

for label, messages in PROMPT_MESSAGES.items():
    print("\n" + "=" * 100)
    print(f"PROMPT: {label}")
    print("=" * 100)

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(model.device)

    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    past_key_values = out.past_key_values
    next_logits = out.logits[:, -1, :]

    generated_ids = []
    for step in range(N_STEPS):
        probs = torch.softmax(next_logits.float(), dim=-1)[0]
        top3_probs, top3_ids = torch.topk(probs, k=3)
        chosen_id = int(top3_ids[0].item())
        chosen_tok = tokenizer.decode([chosen_id])
        candidates = [
            f"{tokenizer.decode([int(tid)])!r}:{float(p):.3f}"
            for tid, p in zip(top3_ids, top3_probs)
        ]
        print(f"  step {step:2d} | chosen={chosen_tok!r:10s} | top3: {', '.join(candidates)}")

        if chosen_id in eos_ids:
            print("    -> EOS reached, stopping early.")
            break
        generated_ids.append(chosen_id)
        with torch.no_grad():
            out = model(input_ids=top3_ids[0].view(1, 1), past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
        next_logits = out.logits[:, -1, :]

    print(f"\n  [{label}] decoded text:")
    print(" ", tokenizer.decode(generated_ids, skip_special_tokens=True))

print("\n" + "=" * 100)
print("How to read this:")
print("  - Watch for the token right after 'Wendi has' — with the few-shot exemplar")
print("    active, check whether the model now says '20' instead of '2'.")
print("  - If 'Reasoner' shows 'user'/'assistant' hallucination near the start ->")
print("    the prompt/exemplar is still triggering it, worth a different exemplar.")
print("  - If it's clean and says '20' correctly -> the few-shot exemplar is helping")
print("    on this known failure case; worth the cheap single_agent+A rerun next.")
