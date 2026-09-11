"""
Real-model check for lakv/rlm_repl_kv.py -- Stage 2: the actual research
question (relay the sub-call's KV instead of its decoded text) built on
top of Stage 1's now-validated REPL loop.

"Channels" now include both return channels AND causal-audit variants on
the "kv" channel, matching the main pipeline's <base>_audit_<mode> naming
(lakv/evaluator.py::AUDIT_MODE_SUFFIXES): "kv_audit_zeroed" and
"kv_audit_random" substitute the child's KV with zeroed/moment-matched-
random content BEFORE it is spliced, testing whether this fan-in splice
point shows the same three-tier causal ordering (real > zeroed/random)
already established for the sequential pipeline's hop-to-hop relay.
"kv_audit_mismatched" is not yet wired up here -- it needs a pre-built
KVAuditPool of held-out sessions' real child KVs, which this script does
not yet construct; add it as a follow-up once zeroed/random show the
expected pattern (see the causal_audit_mode branch below).

For every example, prints every root turn, every child sub-call's answer
(shown for inspection even in "kv" mode, where it is NOT what the root
actually attends to -- see rlm_repl_kv.py's module docstring), the audit
log for each spliced child KV (if auditing), and real EM/F1 scores. Also
saves a full JSON transcript per run to results/rlm_kv_check/ -- added
specifically because the previous n=5 comparison run's console-only
output could not be inspected after the fact when one example degenerated
into gibberish in both channels; this closes that gap.

The CPU-only mechanics (splicing a child KV onto an already-nonempty root
cache, and now the causal-audit substitution feeding into that same
splice) are verified in tests/test_rlm_repl_kv.py. This remains a small
real-model check (n=5 default) -- the point is confirming each condition
produces coherent-vs-corrupted output in the expected shape, not yet a
scored comparison at a sample size that would mean anything statistically
(see docs/FUTURE_WORK.md Idea #2 / the RLM plan in conversation for the
larger n=15-20 follow-up once this run comes back clean).

Usage:
    python scripts/rlm_repl_kv_check.py --n 5
    python scripts/rlm_repl_kv_check.py --n 5 --channels kv
    python scripts/rlm_repl_kv_check.py --n 5 --channels kv kv_audit_zeroed kv_audit_random
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run import load_model
from lakv.qa_scoring import extract_qa_answer, exact_match_score, f1_score
from lakv.recursive_pipeline import load_hotpotqa_structured
from lakv.rlm_repl_kv import RLMKVSession


CHANNEL_CHOICES = ["text", "kv", "kv_audit_zeroed", "kv_audit_random"]

# channel name -> (return_channel, causal_audit_mode)
_CHANNEL_SPEC = {
    "text": ("text", "none"),
    "kv": ("kv", "none"),
    "kv_audit_zeroed": ("kv", "zeroed"),
    "kv_audit_random": ("kv", "random"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--max_turns", type=int, default=10)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--channels", nargs="+", default=["text", "kv"], choices=CHANNEL_CHOICES)
    parser.add_argument("--output_dir", default="results/rlm_kv_check")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_name, device="cuda")
    data = load_hotpotqa_structured(split=args.split, n=args.n)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[rlm_kv_check] transcripts will be saved to: {out_dir}")

    summary = {}
    all_records = {}
    for channel in args.channels:
        return_channel, causal_audit_mode = _CHANNEL_SPEC[channel]
        print(f"\n{'=' * 70}\nCHANNEL: {channel}  (return_channel={return_channel}, "
              f"causal_audit_mode={causal_audit_mode})\n{'=' * 70}")
        session = RLMKVSession(
            model, tokenizer, device="cuda", return_channel=return_channel,
            max_turns=args.max_turns, causal_audit_mode=causal_audit_mode,
        )

        n_correct = 0
        f1_total = 0.0
        records = []
        for i, item in enumerate(data):
            print(f"\n--- Example {i} ---")
            print(f"Question: {item['question']}")
            print(f"Gold answer: {item['answer']}")

            result = session.run(item["question"], item["passages"])

            for t, text in enumerate(result.turn_texts):
                print(f"  [root turn {t}] {text.strip()[:400]!r}")
            for c, text in enumerate(result.child_texts):
                print(f"  [child {c} answer -- shown for inspection, "
                      f"{'NOT' if return_channel == 'kv' else ''} what the root actually attends to] "
                      f"{text.strip()[:300]!r}")
            for a, log in enumerate(result.audit_logs):
                print(f"  [audit log {a}] {log}")

            if result.hit_max_turns:
                print(f"  [DID NOT FINISH] hit max_turns={args.max_turns}")
                em, f1 = False, 0.0
                pred = None
            else:
                pred = extract_qa_answer(result.answer or "")
                em = exact_match_score(pred, item["answer"])
                f1 = f1_score(pred, item["answer"])
                print(f"  Final answer: {result.answer!r} -> extracted: {pred!r} | EM={em} F1={f1:.2f}")

            print(f"  llm_query() calls made: {result.llm_query_calls}")
            n_correct += int(em)
            f1_total += f1

            records.append({
                "idx": i,
                "question": item["question"],
                "gold": item["answer"],
                "predicted": pred,
                "raw_answer": result.answer,
                "em": bool(em),
                "f1": f1,
                "hit_max_turns": result.hit_max_turns,
                "llm_query_calls": result.llm_query_calls,
                "turn_texts": result.turn_texts,
                "child_texts": result.child_texts,
                "audit_logs": result.audit_logs,
            })

        n = len(data)
        summary[channel] = (n_correct, n, f1_total / max(n, 1))
        all_records[channel] = records
        print(f"\n[{channel}] {n_correct}/{n} exact match ({100 * n_correct / max(n,1):.1f}%), "
              f"mean F1={f1_total / max(n,1):.3f}")

    print(f"\n{'=' * 70}\nSUMMARY (n={len(data)}, NOT statistically powered)\n{'=' * 70}")
    for channel, (n_correct, n, mean_f1) in summary.items():
        print(f"  {channel:16s} {n_correct}/{n} EM ({100 * n_correct / max(n,1):.1f}%)  mean F1={mean_f1:.3f}")

    out_path = out_dir / "transcripts.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "args": vars(args),
            "summary": {c: {"n_correct": nc, "n": n, "mean_f1": mf1} for c, (nc, n, mf1) in summary.items()},
            "records": all_records,
        }, f, indent=2)
    print(f"\n[rlm_kv_check] full transcripts + audit logs saved to: {out_path}")


if __name__ == "__main__":
    main()
