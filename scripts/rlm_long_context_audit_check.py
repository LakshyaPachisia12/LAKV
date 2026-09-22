"""
Causal audit for lakv/rlm_repl_kv.py at a genuine long-context length, on
branch feat/rlm-long-context.

Pivot (2026-09-22) from the dose-response sweep (rlm_long_context_sweep.py):
across three runs of increasing n (3, 8, 20), the "kv's raw-accuracy edge
over text grows with context length" hypothesis trended toward null, not
away from it -- the honest read is that raw accuracy is not where this
long-context regime is going to show something real, at least not without
far more GPU time than is available.

The causal audit is a different, better-motivated bet: it has never once
failed to find a real, significant effect anywhere in this project
(sequential pipeline, fan-in, native-length RLM+KV, even the external C2C
bridge -- see CLAUDE.md), REGARDLESS of whether kv ever beat text on raw
accuracy. It doesn't need kv to be better than text; it only needs real
delegated content to beat corrupted delegated content, which is a
fundamentally easier bar this project has cleared every time it's been
tested. Concentrates GPU time at ONE length (16,000 tokens by default --
closest to RLM's own reported crossover point and the least-bad point in
the dose-response sweep) rather than spreading thin across several.

Reuses RLMKVSession's causal-audit machinery directly (causal_audit_mode,
audit_pool, record_audit_pool) -- no new engineering there. Reuses the
held-out-pre-pass pattern scripts/rlm_repl_kv_check.py already uses for
"kv_audit_mismatched": extra examples drawn from BEYOND the scored range
(via lakv/long_context_hotpotqa.py's loader, same target_tokens), never
overlapping, run once with record_audit_pool set to populate a real-KV
pool before any scored "kv_audit_mismatched" run consumes it.

Also reports the "delegated-and-completed" rate per condition
(scripts/analyze_delegation_completion.py's metric from the native-length
branch, ported here) -- the single cleanest signal found anywhere in this
project's RLM+KV work, and one that doesn't depend on raw accuracy being
high to be informative.

Usage:
    python scripts/rlm_long_context_audit_check.py --n 5 --target_tokens 16000  # smoke test
    python scripts/rlm_long_context_audit_check.py --n 20 --target_tokens 16000
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
# Same determinism fix as scripts/rlm_repl_kv_check.py -- this script
# drives the identical RLMKVSession decode loop, so it's subject to the
# same non-determinism unless forced deterministic before any CUDA
# context exists.
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from run import load_model
from lakv.qa_scoring import extract_qa_answer, exact_match_score, f1_score
from lakv.causal_audit import KVAuditPool
from lakv.long_context_hotpotqa import build_long_context_examples
from lakv.rlm_repl_kv import RLMKVSession
from lakv.stats import mcnemar_test


CHANNEL_CHOICES = ["kv", "kv_audit_zeroed", "kv_audit_random", "kv_audit_mismatched"]
_CHANNEL_SPEC = {
    "kv": "none",
    "kv_audit_zeroed": "zeroed",
    "kv_audit_random": "random",
    "kv_audit_mismatched": "mismatched",
}


def delegated_and_completed(records):
    """Same definition as scripts/analyze_delegation_completion.py: a
    session both attempted real delegation (llm_query_calls > 0) AND
    still finished (not hit_max_turns) -- the only subset where a causal-
    audit substitution could possibly have mattered."""
    return [
        {"idx": r["idx"], "correct": (not r["hit_max_turns"] and r["llm_query_calls"] > 0)}
        for r in records
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--target_tokens", type=int, default=16_000)
    parser.add_argument("--max_turns", type=int, default=15)
    parser.add_argument("--filler_pool_size", type=int, default=500)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--channels", nargs="+", default=CHANNEL_CHOICES, choices=CHANNEL_CHOICES)
    parser.add_argument("--output_dir", default="results/rlm_long_context_audit")
    parser.add_argument("--n_held_out", type=int, default=15,
                         help="Extra long-context examples (same target_tokens, drawn from "
                              "beyond --n, never overlapping) used to build the KVAuditPool "
                              "for 'kv_audit_mismatched'. Ignored if that channel isn't requested.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_name, device="cuda")

    needs_pool = "kv_audit_mismatched" in args.channels
    total_needed = args.n + (args.n_held_out if needs_pool else 0)
    all_examples = build_long_context_examples(
        tokenizer, n=total_needed, target_tokens=args.target_tokens,
        split=args.split, filler_pool_size=args.filler_pool_size, seed=args.seed,
    )
    scored = all_examples[:args.n]
    held_out = all_examples[args.n:total_needed] if needs_pool else []

    actual_tokens = [ex.approx_tokens for ex in scored]
    print(f"[rlm_long_context_audit_check] scored examples: n={len(scored)}, "
          f"token counts min={min(actual_tokens)} max={max(actual_tokens)} "
          f"mean={sum(actual_tokens)/len(actual_tokens):.0f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[rlm_long_context_audit_check] results will be saved to: {out_dir}")

    audit_pool = None
    if needs_pool:
        print(f"\n{'=' * 70}\nBuilding KVAuditPool from {len(held_out)} held-out long-context "
              f"examples (never overlapping the scored {args.n})\n{'=' * 70}")
        audit_pool = KVAuditPool(max_size_per_agent=len(held_out))
        pool_session = RLMKVSession(
            model, tokenizer, device="cuda", return_channel="kv",
            max_turns=args.max_turns, causal_audit_mode="none",
            record_audit_pool=audit_pool,
        )
        for ex in held_out:
            pool_session.run(ex.question, ex.passages)
        print(f"[rlm_long_context_audit_check] pool built, {audit_pool.size(0)} entries recorded")

    all_records = {}
    for channel in args.channels:
        causal_audit_mode = _CHANNEL_SPEC[channel]
        print(f"\n{'=' * 70}\nCHANNEL: {channel} (causal_audit_mode={causal_audit_mode})\n{'=' * 70}")
        session = RLMKVSession(
            model, tokenizer, device="cuda", return_channel="kv",
            max_turns=args.max_turns, causal_audit_mode=causal_audit_mode,
            audit_pool=audit_pool if causal_audit_mode == "mismatched" else None,
        )

        records = []
        for i, ex in enumerate(scored):
            result = session.run(ex.question, ex.passages)
            if result.hit_max_turns:
                pred, em, f1 = None, False, 0.0
            else:
                pred = extract_qa_answer(result.answer or "")
                em = exact_match_score(pred, ex.answer)
                f1 = f1_score(pred, ex.answer)
            print(f"  [{channel}] ex{i}: EM={em} F1={f1:.2f} hit_max_turns={result.hit_max_turns} "
                  f"turns={len(result.turn_texts)} queries={result.llm_query_calls}")
            records.append({
                "idx": i, "correct": bool(em), "f1": f1,
                "predicted": pred, "raw_answer": result.answer,
                "hit_max_turns": result.hit_max_turns,
                "llm_query_calls": result.llm_query_calls,
                "turn_texts": result.turn_texts, "child_texts": result.child_texts,
                "audit_logs": result.audit_logs,
                "approx_tokens": ex.approx_tokens,
                "question": ex.question, "gold": ex.answer,
            })

        n = len(scored)
        n_correct = sum(r["correct"] for r in records)
        mean_f1 = sum(r["f1"] for r in records) / n
        n_timeout = sum(r["hit_max_turns"] for r in records)
        print(f"  [{channel}] summary: {n_correct}/{n} EM ({100*n_correct/n:.1f}%) "
              f"mean F1={mean_f1:.3f} timeouts={n_timeout}/{n}")
        all_records[channel] = records

    print(f"\n{'=' * 70}\nCAUSAL AUDIT SUMMARY (target_tokens={args.target_tokens}, n={args.n})\n{'=' * 70}")
    for channel in args.channels:
        recs = all_records[channel]
        n_correct = sum(r["correct"] for r in recs)
        mean_f1 = sum(r["f1"] for r in recs) / len(recs)
        print(f"  {channel:20s} {n_correct}/{len(recs)} EM ({100*n_correct/len(recs):.1f}%) mean F1={mean_f1:.3f}")

    print(f"\nDelegated-and-completed rate (the metric that resolved the native-length "
          f"audit's hardest legs -- see CLAUDE.md finding 20's UPDATE 2026-09-22):")
    dc_flags = {c: delegated_and_completed(all_records[c]) for c in args.channels}
    for c in args.channels:
        n_yes = sum(f["correct"] for f in dc_flags[c])
        print(f"  {c:20s} {n_yes}/{len(dc_flags[c])} ({100*n_yes/len(dc_flags[c]):.0f}%)")

    if "kv" in args.channels:
        print(f"\nPairwise McNemar (delegated-and-completed) vs. 'kv':")
        base = dc_flags["kv"]
        for c in args.channels:
            if c == "kv":
                continue
            res = mcnemar_test(base, dc_flags[c])
            print(f"  kv vs {c}: p={res.p_value:.4f} (n_discordant={res.n_discordant}, "
                  f"kv-only={res.n_a_only}, {c}-only={res.n_b_only})")

    print(f"\nPairwise McNemar (delegated-and-completed) on EM directly:")
    if "kv" in args.channels:
        base_em = [{"idx": r["idx"], "correct": r["correct"]} for r in all_records["kv"]]
        for c in args.channels:
            if c == "kv":
                continue
            other_em = [{"idx": r["idx"], "correct": r["correct"]} for r in all_records[c]]
            res = mcnemar_test(base_em, other_em)
            print(f"  kv vs {c} (EM): p={res.p_value:.4f} (n_discordant={res.n_discordant})")

    out_path = out_dir / "audit_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "records": all_records}, f, indent=2)
    print(f"\n[rlm_long_context_audit_check] full results saved to: {out_path}")


if __name__ == "__main__":
    main()
