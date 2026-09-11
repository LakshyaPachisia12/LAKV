"""
Manual-inspection check for the RLM+KV-relay prototype
(lakv/recursive_pipeline.py) -- a spot check with real scoring, NOT yet a
statistically meaningful evaluation (see n's default below). Runs real
HotpotQA examples through return-channel conditions -- "text" (RLM's
actual mechanism), "kv" (direct fan-in KV merge, no reasoning step),
"kv_synthesis" (fan-in merge + an explicit reasoning pass before
answering), and now "kv_audit_zeroed" / "kv_audit_random" -- the same
zeroed/moment-matched-random causal-audit substitution used throughout
this project (lakv/causal_audit.py), applied here to ONE child's KV
before the fan-in merge (see RecursivePipelineConfig.causal_audit_mode /
causal_audit_child_idx), testing whether the same three-tier ordering
(real > zeroed/random) holds at this topology's merge point, not just the
sequential pipeline's hop-to-hop handoff -- and prints everything:
per-child findings, the synthesis reasoning (when applicable), the audit
log (when auditing), final answers, and real EM/F1 scores via
lakv/qa_scoring.py (the same scorer the rest of this project uses), so
it's not just an eyeball read anymore. Also saves a full JSON transcript
per run to results/recursive_poc_check/.

The CPU-only mechanics (RoPE fan-in merge correctness, and now the
causal-audit-then-merge composition) are verified in
tests/test_recursive_kv_merge.py. This script is the real-model step
after that. Still deliberately small by default -- enough to see a
pattern, not enough to be a statistically powered result; bump --n for a
closer look, but treat everything here as directional, not conclusive,
until it's run through the project's real evaluator/stats pipeline at
n>=50.

Usage:
    python scripts/recursive_poc_check.py --n 15
    python scripts/recursive_poc_check.py --n 15 --n_children 2
    python scripts/recursive_poc_check.py --n 15 --channels kv kv_synthesis text
    python scripts/recursive_poc_check.py --n 15 --channels kv kv_audit_zeroed kv_audit_random
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Running as `python scripts/recursive_poc_check.py` only puts scripts/ on
# sys.path, not the project root -- so `run.py` and the `lakv` package
# aren't importable without this (same fix tests/test_*.py already use).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run import load_model
from lakv.qa_scoring import extract_qa_answer, exact_match_score, f1_score
from lakv.recursive_pipeline import (
    RecursiveKVPipeline, RecursivePipelineConfig, load_hotpotqa_structured,
)


CHANNEL_CHOICES = ["text", "kv", "kv_synthesis", "kv_audit_zeroed", "kv_audit_random"]

# channel name -> (return_channel, causal_audit_mode)
_CHANNEL_SPEC = {
    "text": ("text", "none"),
    "kv": ("kv", "none"),
    "kv_synthesis": ("kv_synthesis", "none"),
    "kv_audit_zeroed": ("kv", "zeroed"),
    "kv_audit_random": ("kv", "random"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n", type=int, default=15)
    parser.add_argument("--n_children", type=int, default=2)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--causal_audit_child_idx", type=int, default=-1,
                         help="Which child's KV gets substituted for *_audit_* channels "
                              "(default: -1, the last child).")
    parser.add_argument(
        "--channels", nargs="+", default=["text", "kv", "kv_synthesis"],
        choices=CHANNEL_CHOICES,
    )
    parser.add_argument("--output_dir", default="results/recursive_poc_check")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_name, device="cuda")
    data = load_hotpotqa_structured(split=args.split, n=args.n)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[recursive_poc_check] transcripts will be saved to: {out_dir}")

    summary = {}
    all_records = {}
    for channel in args.channels:
        return_channel, causal_audit_mode = _CHANNEL_SPEC[channel]
        print(f"\n{'=' * 70}\nCHANNEL: {channel}  (return_channel={return_channel}, "
              f"causal_audit_mode={causal_audit_mode})\n{'=' * 70}")
        config = RecursivePipelineConfig(
            n_children=args.n_children, return_channel=return_channel,
            causal_audit_mode=causal_audit_mode,
            causal_audit_child_idx=args.causal_audit_child_idx,
        )
        pipeline = RecursiveKVPipeline(model, tokenizer, config, device="cuda")

        n_correct = 0
        f1_total = 0.0
        records = []
        for i, item in enumerate(data):
            result = pipeline.run(item["question"], item["passages"])
            pred = extract_qa_answer(result.answer)
            em = exact_match_score(pred, item["answer"])
            f1 = f1_score(pred, item["answer"])
            n_correct += int(em)
            f1_total += f1

            print(f"\n--- Example {i} {'[CORRECT]' if em else '[wrong]'} ---")
            print(f"Question: {item['question']}")
            print(f"Gold answer: {item['answer']}")
            for j, (text, stat) in enumerate(zip(result.child_texts, result.hop_stats)):
                print(f"  Child {j} (seq_len={stat.seq_len}): {text[:300]!r}")
            if result.aggregator_reasoning_text:
                print(f"  Aggregator reasoning: {result.aggregator_reasoning_text[:400]!r}")
            if result.audit_log is not None:
                print(f"  Audit log: {result.audit_log}")
            print(f"Final answer ({channel}): {result.answer[:300]!r} -> extracted: {pred!r} | EM={em} F1={f1:.2f}")

            records.append({
                "idx": i,
                "question": item["question"],
                "gold": item["answer"],
                "predicted": pred,
                "raw_answer": result.answer,
                "em": bool(em),
                "f1": f1,
                "child_texts": result.child_texts,
                "aggregator_reasoning_text": result.aggregator_reasoning_text,
                "audit_log": result.audit_log,
            })

        n = len(data)
        summary[channel] = (n_correct, n, f1_total / max(n, 1))
        all_records[channel] = records
        print(f"\n[{channel}] {n_correct}/{n} exact match ({100 * n_correct / max(n,1):.1f}%), mean F1={f1_total / max(n,1):.3f}")

    print(f"\n{'=' * 70}\nSUMMARY (n={len(data)}, NOT statistically powered at this size)\n{'=' * 70}")
    for channel, (n_correct, n, mean_f1) in summary.items():
        print(f"  {channel:15s} {n_correct}/{n} EM ({100 * n_correct / max(n,1):.1f}%)  mean F1={mean_f1:.3f}")

    out_path = out_dir / "transcripts.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "args": vars(args),
            "summary": {c: {"n_correct": nc, "n": n, "mean_f1": mf1} for c, (nc, n, mf1) in summary.items()},
            "records": all_records,
        }, f, indent=2)
    print(f"\n[recursive_poc_check] full transcripts + audit logs saved to: {out_path}")


if __name__ == "__main__":
    main()
