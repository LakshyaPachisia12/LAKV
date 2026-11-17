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
"kv_audit_mismatched" (added 2026-09-16, following up now that zeroed/
random showed the expected pattern at n=50 -- see docs/PROGRESS_REPORT.md)
needs a pre-built KVAuditPool of held-out sessions' real child KVs. Built
here via the same held-out-pre-pass pattern Evaluator._build_audit_pool
(lakv/evaluator.py) already uses for the main pipeline: when
"kv_audit_mismatched" is requested, --n_held_out additional examples are
drawn from BEYOND the scored range (never overlapping, so a donor's own
question can never be substituted back into itself), run once with
record_audit_pool set (causal_audit_mode="none" -- real child KV
recorded, not substituted) to populate the pool, then the scored
"kv_audit_mismatched" runs consume it via audit_pool.

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
    python scripts/rlm_repl_kv_check.py --n 50 --channels text kv kv_audit_zeroed kv_audit_random kv_audit_mismatched
    python scripts/rlm_repl_kv_check.py --n 50 --channels kv --no_kv_decision_cue

--no_kv_decision_cue (added 2026-09-22, real-model audit finding): the
"kv" channel's kv_decision_cue nudge (see RLMKVSession's docstring) is
currently asymmetric -- it only ever fires on "kv", never on "text",
because "text" already gets an equivalent signal for free (its own
decoded stdout literally contains the sub-call's answer as readable
text, e.g. "[llm_query result] Brooklyn"). Whether kv_decision_cue
exactly compensates for that structural gap, overcorrects, or
undercorrects was never actually tested -- it was added once, kept, and
the resulting "kv" vs "text" comparison (not significant, p=1.0, see
CLAUDE.md Finding 18) has stood without ever checking what it looks like
without the compensating nudge. Rerun "kv" at the same n/split with this
flag and compare against both the existing "text" and cue-enabled "kv"
results via lakv.stats on the identical questions: if the null result
holds without the cue too, that's a cleaner, more defensible null; if
"kv" gets significantly worse without it, that's a real, citable
quantification of how much of kv's current parity with text depends on
this specific compensating mechanism.

Determinism attempt (2026-09-18): diffing two identical runs
(results/rlm_kv_check/run_20260916_105307 vs run_20260916_203322) found
15/50 (30%) of text-channel raw generations differ run-to-run under
greedy decoding -- unlike the main sequential pipeline, which is
confirmed bit-identical across reruns. This script now forces
torch's deterministic-algorithm mode before any CUDA context exists
(CUBLAS_WORKSPACE_CONFIG must be set before torch is imported, not just
before the model loads) to test whether that non-determinism is
eliminable rather than assumed intrinsic to this prototype's long,
self-referential decode loop. warn_only=True deliberately: this is a
diagnostic first pass, not a silent fix -- if a specific op has no
deterministic implementation, PyTorch will name it in a runtime warning
on stderr rather than the run crashing outright, telling us exactly
where any REMAINING non-determinism comes from instead of only whether
it went away. Verify by rerunning this exact script twice at the same
--n and diffing raw_answer/turn_texts across the two runs' saved JSON,
the same check that originally found the 30% figure.
"""

import os
# Must happen before `import torch` (transitively, via `from run import
# load_model` below) -- cuBLAS reads this at CUDA context creation, which
# happens on first GPU op, not at import time, but setting it this early
# is the documented-safe way to guarantee no CUDA context exists yet
# regardless of import order elsewhere in this process.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import time
import sys
from datetime import datetime
from pathlib import Path

# This script prints raw HotpotQA question/passage/answer text, which can
# contain characters (accents, en-dashes, etc.) outside Windows' default
# console codepage (cp1252) -- without this, a long run can crash on an
# UnicodeEncodeError from a single print() call, losing all progress since
# results are only written at the very end. errors="replace" degrades
# gracefully (prints "?" for anything unencodable) instead of crashing.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
# warn_only=True: report which specific op (if any) has no deterministic
# kernel available, rather than crashing the whole run on the first one
# encountered -- a diagnostic first pass, see module docstring above.
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from run import load_model

NL = chr(10)
from lakv.qa_scoring import extract_qa_answer, exact_match_score, f1_score
from lakv.recursive_pipeline import load_hotpotqa_structured
from lakv.causal_audit import DiskAnswerRegionPool, DonorUnavailable, KVAuditPool
from lakv.donor_traces import AnswerRegionSource
from lakv.rlm_repl_kv import RLMKVSession


CHANNEL_CHOICES = [
    "text", "kv",
    "kv_audit_zeroed", "kv_audit_random", "kv_audit_mismatched", "kv_audit_dropped",
    # Length-matched ANSWER-REGION donor, no padding/truncation (LAKV_runs/fanin_length_matched_design.md;
    # UNTESTED ON GPU).
    "kv_audit_length_matched",
]

# channel name -> (return_channel, causal_audit_mode)
_CHANNEL_SPEC = {
    "text": ("text", "none"),
    "kv": ("kv", "none"),
    "kv_audit_zeroed": ("kv", "zeroed"),
    "kv_audit_random": ("kv", "random"),
    "kv_audit_mismatched": ("kv", "mismatched"),
    # A real child call still happens; its KV is just never spliced onto
    # the root's cache. See RLMKVSession's causal_audit_mode docstring
    # for the kv_decision_cue false-premise caveat this condition has.
    "kv_audit_dropped": ("kv", "dropped"),
    "kv_audit_length_matched": ("kv", "length_matched"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--max_turns", type=int, default=10)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--channels", nargs="+", default=["text", "kv"], choices=CHANNEL_CHOICES)
    parser.add_argument("--output_dir", default="results/rlm_kv_check")
    parser.add_argument("--n_held_out", type=int, default=20,
                         help="Held-out examples (drawn from beyond --n, never overlapping) "
                              "used to build the KVAuditPool for 'kv_audit_mismatched'. "
                              "Ignored if that channel isn't requested.")
    parser.add_argument("--capture_confidence", action="store_true",
                         help="Log the mean top-1 token probability / entropy of the root "
                              "turn that produced each session's final answer "
                              "(lakv/confidence.py) -- a continuous signal that resolved "
                              "the mismatched-vs-zeroed/random leg EM/F1 alone couldn't "
                              "(see CLAUDE.md finding 18's UPDATE 2026-09-17, which used "
                              "turn count/timeout rate for the same purpose; this is the "
                              "topology-agnostic generalization of that idea).")
    parser.add_argument("--no_kv_decision_cue", action="store_true",
                         help="Disable the 'kv' channel's content-free decision nudge "
                              "(RLMKVSession's kv_decision_cue, default on) -- an ablation "
                              "testing whether the currently-null kv-vs-text comparison "
                              "depends on this channel-asymmetric compensating mechanism. "
                              "No effect on 'text' or the audit channels, which never use it.")
    parser.add_argument("--length_matched_pool_size", type=int, default=100,
                         help="Target number of donor ANSWER REGIONS (one per held-out child call) for "
                              "kv_audit_length_matched. Held-out sessions run until this many are recorded "
                              "or --max_pool_sessions is exhausted.")
    parser.add_argument("--max_pool_sessions", type=int, default=150,
                         help="Upper bound on held-out sessions used to fill the length-matched pool.")
    parser.add_argument("--donor_pool_dir", default=None,
                         help="Directory for the length-matched pool (default <run dir>/audit_pool). "
                              "A complete pool with at least the requested number of regions is reused.")
    parser.add_argument("--audit_donor_seed", type=int, default=0,
                         help="Seed for the length-matched donor draw (per-call crc32-seeded).")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_name, device="cuda")
    needs_pool = "kv_audit_mismatched" in args.channels
    needs_lm = "kv_audit_length_matched" in args.channels
    total_n = args.n + max(args.n_held_out if needs_pool else 0, args.max_pool_sessions if needs_lm else 0)
    all_data = load_hotpotqa_structured(split=args.split, n=total_n)
    data = all_data[:args.n]
    held_out_data = all_data[args.n:args.n + args.n_held_out] if needs_pool else []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[rlm_kv_check] transcripts will be saved to: {out_dir}")

    audit_pool = None
    if needs_pool:
        print(f"\n{'=' * 70}\nBuilding KVAuditPool from {len(held_out_data)} held-out examples "
              f"(never overlapping the scored {args.n})\n{'=' * 70}")
        audit_pool = KVAuditPool(max_size_per_agent=len(held_out_data))
        pool_session = RLMKVSession(
            model, tokenizer, device="cuda", return_channel="kv",
            max_turns=args.max_turns, causal_audit_mode="none",
            record_audit_pool=audit_pool,
            kv_decision_cue=not args.no_kv_decision_cue,
        )
        for item in held_out_data:
            pool_session.run(item["question"], item["passages"])
        print(f"[rlm_kv_check] pool built, {audit_pool.size(0)} entries recorded")

    lm_source = None
    pool_timing = None
    if needs_lm:
        import shutil
        lm_dir = (Path(args.donor_pool_dir) if args.donor_pool_dir else out_dir / "audit_pool") / "rlm_answer_regions"
        lm_pool = DiskAnswerRegionPool.open(lm_dir)
        if lm_pool.complete and lm_pool.size(0) >= args.length_matched_pool_size:
            print(f"[rlm_kv_check] re-using complete answer-region pool ({lm_pool.size(0)} regions) at {lm_dir}")
        else:
            shutil.rmtree(lm_dir, ignore_errors=True)
            lm_pool = DiskAnswerRegionPool(lm_dir)
            print(f"{NL}{'=' * 70}{NL}Building answer-region pool: target {args.length_matched_pool_size} regions "
                  f"from up to {args.max_pool_sessions} held-out sessions -> {lm_dir}{NL}{'=' * 70}")
            pool_session = RLMKVSession(
                model, tokenizer, device="cuda", return_channel="kv", max_turns=args.max_turns,
                causal_audit_mode="none", record_audit_pool=lm_pool,
                kv_decision_cue=not args.no_kv_decision_cue)
            sess_times, regions_before = [], 0
            sess_regions = []
            for j, item in enumerate(all_data[args.n:args.n + args.max_pool_sessions]):
                t0 = time.perf_counter()
                pool_session.run(item["question"], item["passages"])
                sess_times.append(time.perf_counter() - t0)
                sess_regions.append(lm_pool.size(0) - regions_before)
                regions_before = lm_pool.size(0)
                if (j + 1) % 10 == 0:
                    print(f"  [pool] {j + 1} sessions, {lm_pool.size(0)} answer regions, "
                          f"mean {sum(sess_times) / len(sess_times):.1f}s/session")
                if lm_pool.size(0) >= args.length_matched_pool_size:
                    break
            lm_pool.mark_complete()
            pool_timing = {"sessions": len(sess_times), "regions": lm_pool.size(0),
                           "mean_s_per_session": sum(sess_times) / max(len(sess_times), 1),
                           "total_s": sum(sess_times), "regions_per_session": sess_regions}
            print(f"[rlm_kv_check] pool built: {pool_timing['regions']} regions from {pool_timing['sessions']} "
                  f"sessions, mean {pool_timing['mean_s_per_session']:.1f}s/session")
        lm_source = AnswerRegionSource(lm_pool)

    summary = {}
    all_records = {}
    for channel in args.channels:
        return_channel, causal_audit_mode = _CHANNEL_SPEC[channel]
        print(f"\n{'=' * 70}\nCHANNEL: {channel}  (return_channel={return_channel}, "
              f"causal_audit_mode={causal_audit_mode})\n{'=' * 70}")
        session = RLMKVSession(
            model, tokenizer, device="cuda", return_channel=return_channel,
            max_turns=args.max_turns, causal_audit_mode=causal_audit_mode,
            audit_pool=audit_pool if causal_audit_mode == "mismatched" else None,
            capture_confidence=args.capture_confidence,
            kv_decision_cue=not args.no_kv_decision_cue,
            donor_source=lm_source if causal_audit_mode == "length_matched" else None,
            audit_donor_seed=args.audit_donor_seed,
        )

        n_skipped = 0
        n_correct = 0
        f1_total = 0.0
        confidence_sum = {"mean_top1_prob": 0.0, "mean_entropy": 0.0}
        n_with_confidence = 0
        records = []
        for i, item in enumerate(data):
            print(f"\n--- Example {i} ---")
            print(f"Question: {item['question']}")
            print(f"Gold answer: {item['answer']}")

            t_start = time.perf_counter()
            try:
                result = session.run(item["question"], item["passages"])
            except DonorUnavailable as e:
                # a child call had no answer-region donor within 5%: the whole session is skipped
                n_skipped += 1
                print(f"  [skip] {channel} example {i}: {e.reason}")
                records.append({"idx": i, "question": item["question"], "gold": item["answer"], "skipped": True,
                                "skip_reason": e.reason, "audit_logs": [e.log], "em": None, "f1": None,
                                "latency_s": time.perf_counter() - t_start})
                continue
            latency_s = time.perf_counter() - t_start

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
                print(f"  Confidence (final-answer turn): top1_prob={result.confidence.mean_top1_prob:.3f} "
                      f"entropy={result.confidence.mean_entropy:.3f} (n_tokens={result.confidence.n_tokens})")

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
                "confidence": confidence_dict,
                "latency_s": latency_s,
            })

        n = len(data) - n_skipped   # scored (non-skipped) sessions
        mean_confidence = None
        if n_with_confidence > 0:
            mean_confidence = {
                "mean_top1_prob": confidence_sum["mean_top1_prob"] / n_with_confidence,
                "mean_entropy": confidence_sum["mean_entropy"] / n_with_confidence,
                "n_examples": n_with_confidence,
            }
        summary[channel] = (n_correct, n, f1_total / max(n, 1), mean_confidence)
        if n_skipped:
            print(f"[{channel}] skipped {n_skipped}/{len(data)} sessions (no donor within 5%)")
        all_records[channel] = records
        conf_str = ""
        if mean_confidence is not None:
            conf_str = (f", mean confidence top1_prob={mean_confidence['mean_top1_prob']:.3f} "
                        f"entropy={mean_confidence['mean_entropy']:.3f} "
                        f"(n={mean_confidence['n_examples']}/{n}, excludes timeouts)")
        print(f"\n[{channel}] {n_correct}/{n} exact match ({100 * n_correct / max(n,1):.1f}%), "
              f"mean F1={f1_total / max(n,1):.3f}{conf_str}")

    print(f"\n{'=' * 70}\nSUMMARY (n={len(data)}, NOT statistically powered)\n{'=' * 70}")
    for channel, (n_correct, n, mean_f1, mean_confidence) in summary.items():
        conf_str = ""
        if mean_confidence is not None:
            conf_str = (f"  top1_prob={mean_confidence['mean_top1_prob']:.3f} "
                        f"entropy={mean_confidence['mean_entropy']:.3f}")
        print(f"  {channel:16s} {n_correct}/{n} EM ({100 * n_correct / max(n,1):.1f}%)  mean F1={mean_f1:.3f}{conf_str}")

    out_path = out_dir / "transcripts.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "args": vars(args),
            "summary": {
                c: {"n_correct": nc, "n": n, "mean_f1": mf1, "confidence": conf}
                for c, (nc, n, mf1, conf) in summary.items()
            },
            "records": all_records,
            "length_matched_pool_timing": pool_timing,
        }, f, indent=2)
    print(f"\n[rlm_kv_check] full transcripts + audit logs saved to: {out_path}")


if __name__ == "__main__":
    main()
