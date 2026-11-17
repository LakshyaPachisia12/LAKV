"""
Manual-inspection check for the RLM+KV-relay prototype
(lakv/recursive_pipeline.py) -- a spot check with real scoring, NOT yet a
statistically meaningful evaluation (see n's default below). Runs real
HotpotQA examples through return-channel conditions -- "text" (RLM's
actual mechanism), "kv" (direct fan-in KV merge, no reasoning step),
"kv_synthesis" (fan-in merge + an explicit reasoning pass before
answering), and now "kv_audit_zeroed" / "kv_audit_random" -- the same
zeroed/moment-matched-random causal-audit substitution used throughout
this project (lakv/causal_audit.py), applied here before the fan-in merge
(see RecursivePipelineConfig.causal_audit_mode / causal_audit_child_idx),
testing whether the same three-tier ordering (real > zeroed/random) holds
at this topology's merge point, not just the sequential pipeline's
hop-to-hop handoff. --causal_audit_child_idx defaults to "all" (every
child substituted, matching the sequential pipeline's full-relay-
corruption strength) rather than a single child -- real-model testing
(2026-09-13) found that corrupting only one of two children understates
the effect: exact-match barely moved (partial redundancy from the other,
untouched child papered over a real qualitative effect visible in 5 of 10
raw outputs still showing the expected garbled-gibberish signature). Pass
an integer instead for the weaker, partial-corruption condition
specifically.

"kv_audit_mismatched" (added 2026-09-16, completing the three-tier
ladder for this topology -- see docs/PROGRESS_REPORT.md) needs a
pre-built KVAuditPool of held-out sessions' real per-child KVs. Built
here via the same held-out-pre-pass pattern Evaluator._build_audit_pool
(lakv/evaluator.py) and scripts/rlm_repl_kv_check.py's own
kv_audit_mismatched support already use: when this channel is requested,
--n_held_out additional examples are drawn from BEYOND the scored range
(never overlapping, so a donor's own question can never be substituted
back into itself), run once with record_audit_pool set
(causal_audit_mode="none" -- real per-child KV recorded, keyed by child
index, not substituted) to populate the pool, then the scored
"kv_audit_mismatched" runs consume it via audit_pool.

Prints everything: per-child findings, the synthesis
reasoning (when applicable), the audit logs (when auditing), final
answers, and real EM/F1 scores via
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
    python scripts/recursive_poc_check.py --n 15 --channels kv kv_audit_zeroed --causal_audit_child_idx -1
    python scripts/recursive_poc_check.py --n 50 --channels text kv kv_audit_zeroed kv_audit_random kv_audit_mismatched
    python scripts/recursive_poc_check.py --n 50 --channels kv kv_audit_mismatched --capture_confidence

--capture_confidence (added 2026-09-18, see lakv/confidence.py) logs the
aggregator's mean top-1 token probability / entropy per example -- a
continuous signal that can resolve real-vs-mismatched differences EM/F1
alone can't at this topology's current n=50 (see CLAUDE.md: that leg is
underpowered by McNemar, p=0.25, though F1's CI already trends real).
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# This script prints raw HotpotQA passage/answer text, which can contain
# characters (accents, en-dashes, etc.) outside Windows' default console
# codepage (cp1252) -- without this, a long run can crash on an
# UnicodeEncodeError from a single print() call, losing all progress since
# results are only written at the very end. errors="replace" degrades
# gracefully (prints "?" for anything unencodable) instead of crashing.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Running as `python scripts/recursive_poc_check.py` only puts scripts/ on
# sys.path, not the project root -- so `run.py` and the `lakv` package
# aren't importable without this (same fix tests/test_*.py already use).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run import load_model
from lakv.qa_scoring import extract_qa_answer, exact_match_score, f1_score
from lakv.causal_audit import DiskKVAuditPool, DonorUnavailable, KVAuditPool
from lakv.donor_traces import LengthMatchedSource
from lakv.recursive_pipeline import (
    RecursiveKVPipeline, RecursivePipelineConfig, load_hotpotqa_structured,
)


CHANNEL_CHOICES = [
    "text", "kv", "kv_synthesis",
    "kv_audit_zeroed", "kv_audit_random", "kv_audit_mismatched", "kv_audit_dropped",
    # Length-matched donor (no padding/truncation) and the RoPE-gap control, see
    # LAKV_runs/fanin_length_matched_design.md. UNTESTED ON GPU.
    "kv_audit_length_matched", "kv_audit_shift_control",
]

# channel name -> (return_channel, causal_audit_mode)
_CHANNEL_SPEC = {
    "text": ("text", "none"),
    "kv": ("kv", "none"),
    "kv_synthesis": ("kv_synthesis", "none"),
    "kv_audit_zeroed": ("kv", "zeroed"),
    "kv_audit_random": ("kv", "random"),
    "kv_audit_mismatched": ("kv", "mismatched"),
    # "all" children dropped (the CLI default for --causal_audit_child_idx)
    # means the aggregator gets no injected KV at all -- genuinely absent,
    # not a substitute. See RecursivePipelineConfig's causal_audit_mode
    # docstring for the AGGREGATOR_SYSTEM_PROMPT caveat this condition has.
    "kv_audit_dropped": ("kv", "dropped"),
    "kv_audit_length_matched": ("kv", "length_matched"),
    "kv_audit_shift_control": ("kv", "shift_control"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n", type=int, default=15)
    parser.add_argument("--n_children", type=int, default=2)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--causal_audit_child_idx", default="all",
                         help="Which child/children's KV get substituted for *_audit_* "
                              "channels: an integer index (partial corruption -- real "
                              "content survives in every other child, understates the "
                              "effect, see lakv/recursive_pipeline.py's config docstring) "
                              "or 'all' (every child substituted, the fair apples-to-apples "
                              "comparison against the sequential pipeline's full-relay "
                              "audit -- default).")
    parser.add_argument(
        "--channels", nargs="+", default=["text", "kv", "kv_synthesis"],
        choices=CHANNEL_CHOICES,
    )
    parser.add_argument("--output_dir", default="results/recursive_poc_check")
    parser.add_argument("--n_held_out", type=int, default=20,
                         help="Held-out examples (drawn from beyond --n, never overlapping) "
                              "used to build the KVAuditPool for 'kv_audit_mismatched'. "
                              "Ignored if that channel isn't requested.")
    parser.add_argument("--reverse_child_merge_order", action="store_true",
                         help="Diagnostic (2026-09-18): reverse which child gets the "
                              "privileged, unshifted first position in the fan-in merge, "
                              "testing whether our merge inherits CanonicalMerge's "
                              "documented 'directionality flaw'. Rerun the same --channels "
                              "at the same --n with and without this flag and compare "
                              "EM/F1 on the identical examples via lakv.stats.")
    parser.add_argument("--capture_confidence", action="store_true",
                         help="Log the aggregator's mean top-1 token probability / entropy "
                              "per example (lakv/confidence.py) -- a continuous signal that "
                              "can resolve real-vs-audited differences EM/F1 can't at this "
                              "topology's current sample sizes.")
    parser.add_argument("--aggregator_sees_passages", action="store_true",
                         help="Passage-access ablation: also put all ten passages in the "
                              "aggregator's own prompt, giving it an independent route to the "
                              "evidence. Run the same --channels with and without this flag.")
    parser.add_argument("--audit_donor_seed", type=int, default=0,
                         help="Seed for the mismatched-condition donor draw (0 = earlier runs).")
    parser.add_argument("--length_matched_pool_size", type=int, default=100,
                         help="Held-out QUESTIONS (2 children each) whose child caches form the disk-backed "
                              "pool for kv_audit_length_matched / kv_audit_shift_control.")
    parser.add_argument("--donor_pool_dir", default=None,
                         help="Directory for the length-matched pool (default <run dir>/audit_pool). "
                              "A complete pool of the requested size is reused.")
    args = parser.parse_args()
    causal_audit_child_idx = (
        args.causal_audit_child_idx if args.causal_audit_child_idx == "all"
        else int(args.causal_audit_child_idx)
    )

    model, tokenizer = load_model(args.model_name, device="cuda")
    needs_pool = "kv_audit_mismatched" in args.channels
    needs_lm = any(c in args.channels for c in ("kv_audit_length_matched", "kv_audit_shift_control"))
    total_n = args.n + max(args.n_held_out if needs_pool else 0,
                           args.length_matched_pool_size if needs_lm else 0)
    all_data = load_hotpotqa_structured(split=args.split, n=total_n)
    data = all_data[:args.n]
    held_out_data = all_data[args.n:args.n + args.n_held_out] if needs_pool else []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[recursive_poc_check] transcripts will be saved to: {out_dir}")

    audit_pool = None
    if needs_pool:
        print(f"\n{'=' * 70}\nBuilding KVAuditPool from {len(held_out_data)} held-out examples "
              f"(never overlapping the scored {args.n})\n{'=' * 70}")
        audit_pool = KVAuditPool(max_size_per_agent=len(held_out_data))
        pool_config = RecursivePipelineConfig(
            n_children=args.n_children, return_channel="kv",
            causal_audit_mode="none", record_audit_pool=audit_pool,
        )
        pool_pipeline = RecursiveKVPipeline(model, tokenizer, pool_config, device="cuda")
        for item in held_out_data:
            pool_pipeline.run(item["question"], item["passages"])
        print(f"[recursive_poc_check] pool built, "
              f"{[audit_pool.size(i) for i in range(args.n_children)]} entries per child")

    lm_source = None
    if needs_lm:
        lm_dir = Path(args.donor_pool_dir) if args.donor_pool_dir else out_dir / "audit_pool"
        lm_dir = lm_dir / "fanin_length_matched"
        lm_pool = DiskKVAuditPool.open(lm_dir)
        size = args.length_matched_pool_size
        if lm_pool.complete and all(lm_pool.size(i) == size for i in range(args.n_children)):
            print(f"[recursive_poc_check] re-using complete length_matched pool ({size} questions) at {lm_dir}")
        else:
            import shutil
            shutil.rmtree(lm_dir, ignore_errors=True)
            lm_pool = DiskKVAuditPool(lm_dir)
            lm_held = all_data[args.n:args.n + size]
            print(f"\n{'=' * 70}\nBuilding disk-backed length_matched pool: {len(lm_held)} held-out questions "
                  f"x {args.n_children} children -> {lm_dir}\n{'=' * 70}")
            lm_cfg = RecursivePipelineConfig(n_children=args.n_children, return_channel="kv",
                                             causal_audit_mode="none", record_audit_pool=lm_pool)
            lm_pipe = RecursiveKVPipeline(model, tokenizer, lm_cfg, device="cuda")
            for j, item in enumerate(lm_held):
                lm_pipe.run(item["question"], item["passages"])
                if (j + 1) % 10 == 0:
                    print(f"  [pool] {j + 1}/{len(lm_held)}")
            lm_pool.mark_complete()
        lm_source = LengthMatchedSource(lm_pool)

    summary = {}
    all_records = {}
    for channel in args.channels:
        return_channel, causal_audit_mode = _CHANNEL_SPEC[channel]
        print(f"\n{'=' * 70}\nCHANNEL: {channel}  (return_channel={return_channel}, "
              f"causal_audit_mode={causal_audit_mode})\n{'=' * 70}")
        config = RecursivePipelineConfig(
            n_children=args.n_children, return_channel=return_channel,
            causal_audit_mode=causal_audit_mode,
            causal_audit_child_idx=causal_audit_child_idx,
            audit_pool=audit_pool if causal_audit_mode == "mismatched" else None,
            reverse_child_merge_order=args.reverse_child_merge_order,
            capture_confidence=args.capture_confidence,
            aggregator_sees_passages=args.aggregator_sees_passages,
            audit_donor_seed=args.audit_donor_seed,
            donor_source=lm_source if causal_audit_mode in ("length_matched", "shift_control") else None,
        )
        pipeline = RecursiveKVPipeline(model, tokenizer, config, device="cuda")

        n_correct = 0
        f1_total = 0.0
        confidence_sum = {"mean_top1_prob": 0.0, "mean_entropy": 0.0}
        n_with_confidence = 0
        records = []
        n_skipped = 0
        for i, item in enumerate(data):
            try:
                result = pipeline.run(item["question"], item["passages"])
            except DonorUnavailable as e:
                # a child had no donor within 5%: question skipped (not scored); pair on the rest
                n_skipped += 1
                print(f"  [skip] {channel} example {i}: {e.reason}")
                records.append({"idx": i, "question": item["question"], "gold": item["answer"], "skipped": True,
                                "skip_reason": e.reason, "audit_logs": [e.log], "em": None, "f1": None})
                continue
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
            if result.audit_logs:
                print(f"  Audit logs: {result.audit_logs}")
            confidence_dict = None
            if result.confidence is not None:
                confidence_dict = {
                    "mean_top1_prob": result.confidence.mean_top1_prob,
                    "mean_entropy": result.confidence.mean_entropy,
                    "n_tokens": result.confidence.n_tokens,
                }
                confidence_sum["mean_top1_prob"] += result.confidence.mean_top1_prob
                confidence_sum["mean_entropy"] += result.confidence.mean_entropy
                n_with_confidence += 1
                print(f"  Confidence: top1_prob={result.confidence.mean_top1_prob:.3f} "
                      f"entropy={result.confidence.mean_entropy:.3f} (n_tokens={result.confidence.n_tokens})")
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
                "audit_logs": result.audit_logs,
                "confidence": confidence_dict,
            })

        n = len(data) - n_skipped   # scored (non-skipped) questions
        mean_confidence = None
        if n_with_confidence > 0:
            mean_confidence = {
                "mean_top1_prob": confidence_sum["mean_top1_prob"] / n_with_confidence,
                "mean_entropy": confidence_sum["mean_entropy"] / n_with_confidence,
                "n_examples": n_with_confidence,
            }
        summary[channel] = (n_correct, n, f1_total / max(n, 1), mean_confidence)
        if n_skipped:
            print(f"[{channel}] skipped {n_skipped}/{len(data)} questions (no donor within 5%)")
        all_records[channel] = records
        conf_str = ""
        if mean_confidence is not None:
            conf_str = (f", mean confidence top1_prob={mean_confidence['mean_top1_prob']:.3f} "
                        f"entropy={mean_confidence['mean_entropy']:.3f}")
        print(f"\n[{channel}] {n_correct}/{n} exact match ({100 * n_correct / max(n,1):.1f}%), "
              f"mean F1={f1_total / max(n,1):.3f}{conf_str}")

    print(f"\n{'=' * 70}\nSUMMARY (n={len(data)}, NOT statistically powered at this size)\n{'=' * 70}")
    for channel, (n_correct, n, mean_f1, mean_confidence) in summary.items():
        conf_str = ""
        if mean_confidence is not None:
            conf_str = (f"  top1_prob={mean_confidence['mean_top1_prob']:.3f} "
                        f"entropy={mean_confidence['mean_entropy']:.3f}")
        print(f"  {channel:15s} {n_correct}/{n} EM ({100 * n_correct / max(n,1):.1f}%)  mean F1={mean_f1:.3f}{conf_str}")

    out_path = out_dir / "transcripts.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "args": vars(args),
            "summary": {
                c: {"n_correct": nc, "n": n, "mean_f1": mf1, "confidence": conf}
                for c, (nc, n, mf1, conf) in summary.items()
            },
            "records": all_records,
        }, f, indent=2)
    print(f"\n[recursive_poc_check] full transcripts + audit logs saved to: {out_path}")


if __name__ == "__main__":
    main()
