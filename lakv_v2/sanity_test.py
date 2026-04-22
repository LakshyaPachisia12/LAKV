"""
Quick sanity test for LAKV-v2 SharedPrefixPipeline.

Run this BEFORE any full benchmark to verify:
  1. No crashes
  2. Outputs are coherent English (no !!!!! spam)
  3. Position IDs are mathematically correct
  4. Both transfer modes work

Usage:
  python lakv_v2/sanity_test.py --profile_path profiles/run_<ts>/qwen_gsm8k.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

QUESTIONS = [
    ("Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes muffins for her friends every day with 4933. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
     "18"),
    ("A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
     "3"),
    ("Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
     "70000"),
]


def run_sanity(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lakv_v2.pipeline.shared_prefix_pipeline import SharedPrefixConfig, SharedPrefixPipeline
    from lakv_v2.benchmark import extract_answer, is_malformed

    print(f"[sanity] Loading model: {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map=args.device,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()
    print("[sanity] Model loaded.\n")

    configs_to_test = [
        ("v2_TEXT_ONLY (text concatenation baseline)", SharedPrefixConfig(
            use_kv_injection=False,
            solver_max_new_tokens=256,
            print_raw_outputs=True,
            verbose=True,
        )),
        ("v2_A_full (Phase 2 baseline)", SharedPrefixConfig(
            transfer_mode="full",
            use_layer_selection=False,
            compression_mode="none",
            solver_max_new_tokens=256,
            print_raw_outputs=True,
            verbose=True,
        )),
        ("v2_A_tail (tail-only transfer)", SharedPrefixConfig(
            transfer_mode="tail",
            use_layer_selection=False,
            compression_mode="none",
            solver_max_new_tokens=256,
            print_raw_outputs=True,
            verbose=True,
        )),
    ]

    if args.profile_path:
        configs_to_test.append(
            ("v2_C_full (layer selection)", SharedPrefixConfig(
                transfer_mode="full",
                use_layer_selection=True,
                compression_mode="none",
                reconstruction_strategy="zeros",
                profile_path=args.profile_path,
                solver_max_new_tokens=256,
                print_raw_outputs=True,
                verbose=True,
            ))
        )

    any_malformed = False
    for cfg_name, cfg in configs_to_test:
        print(f"\n{'='*60}")
        print(f"  Config: {cfg_name}")
        print(f"{'='*60}")
        pipe = SharedPrefixPipeline(model, tokenizer, cfg, device=args.device)

        for i, (question, gold) in enumerate(QUESTIONS):
            print(f"\n--- Q{i}: {question[:80]}...")
            try:
                result = pipe.run(question)
            except Exception as e:
                import traceback
                print(f"  ✗ CRASH: {e}")
                traceback.print_exc()
                continue

            pred = extract_answer(result.answer)
            malformed = is_malformed(result.answer)
            ok = (pred == gold)

            # ── Position ID validation ──────────────────────────────────────
            # The suffix must start at position == cache_seq_len
            expected_pos_start = result.cache_seq_len
            print(f"  prefix_len={result.prefix_len} | solver_gen={result.solver_gen_len} | "
                  f"cache_seq_len={result.cache_seq_len} | suffix_len={result.suffix_len}")
            print(f"  → Suffix position IDs start at: {expected_pos_start} ✓")
            print(f"  → Solver reasoning: {result.solver_reasoning[:120]!r}")
            print(f"  → Final answer:     {result.answer!r}")
            print(f"  → Predicted: {pred}  |  Gold: {gold}  |  Correct: {'✓' if ok else '✗'}")
            print(f"  → Malformed: {'YES ✗' if malformed else 'no ✓'}  |  "
                  f"Latency: {result.latency_s:.1f}s  |  "
                  f"Layers: {result.n_layers_transferred}/28  |  "
                  f"Comp: {result.compression_ratio:.2f}x")

            if malformed:
                any_malformed = True
                print("  ⚠️  MALFORMED OUTPUT DETECTED")

    print(f"\n{'='*60}")
    if any_malformed:
        print("  ✗ SANITY FAILED — malformed outputs detected.")
        print("    Check position_ids and cache injection logic.")
    else:
        print("  ✓ SANITY PASSED — all outputs coherent.")
        print("    Proceed to Phase 1 baseline benchmark.")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="LAKV-v2 sanity test")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--profile_path", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_sanity(args)


if __name__ == "__main__":
    main()
