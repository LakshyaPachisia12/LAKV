"""
Dose-response sweep for lakv/rlm_repl_kv.py on branch feat/rlm-long-context.

Tests a sharper hypothesis than "does kv beat text": does kv's advantage
over text GROW with context length, the way RLM's own paper (Zhang,
Kraska, Khattab, arXiv 2512.24601) shows recursive decomposition's
advantage over a flat baseline grows with context length -- reported to
begin around 2^14 = 16,384 tokens? RLM's own paper never compares KV
relay to text relay at all (it only has a text return channel), so this
is a genuinely open question, not a replication of anything cited in
this project.

Runs BOTH return channels ("text" and "kv") at several context lengths
on the SAME underlying HotpotQA questions, via
lakv/long_context_hotpotqa.py's needle-in-haystack loader. Using the
same --seed/--n/--filler_pool_size across lengths gives NESTED haystacks
per question (each longer length's filler is a superset of the shorter
length's, since build_long_context_examples grows a per-example-shuffled
filler list from the front) -- so any difference across lengths is
attributable to context length itself, not to different questions or a
different random haystack composition.

Rather than needing one very large run at a single length to reach
significance, this tests for a TREND in kv's win rate among discordant
pairs across the ordered lengths (lakv.stats.
discordant_pair_trend_across_conditions, a Cochran-Armitage trend test)
-- statistically more efficient for a genuinely trend-shaped hypothesis
than pouring the same total n into one length.

Deliberately caps the default length sweep at 32,000 tokens, staying
under Qwen2.5-7B-Instruct's native 32,768-token window to avoid needing
YaRN/rope-scaling for a first pass -- a bug class this project has
already been burned by once (CLAUDE.md finding 12, Phi-3.5-mini's
LongRoPE collapse).

Usage:
    python scripts/rlm_long_context_sweep.py --n 5 --lengths 2000 8000  # quick smoke test
    python scripts/rlm_long_context_sweep.py --n 25 --lengths 2000 8000 16000 32000

UPDATE 2026-10-01: the original attempt at this sweep (no fixes applied)
trended toward null -- kv's apparent edge over text at 2000 tokens
shrank from +66.7pts (n=3) to +12.5 (n=8) to +0.0 (n=20), and 8000 tokens
sat as a trough for both channels. Root-caused via the causal-audit
pivot (scripts/rlm_long_context_audit_check.py,
docs/RLM_LONG_CONTEXT_LOG.md): 55-60% of sessions never delegate at all
regardless of condition, because the model gets lucky reading a passage
directly before it would ever consider delegating -- that leak-rate
problem, not a real absence of a length effect, is the most likely
reason the original sweep never found anything. Two real fixes for it
(--long_context_prompt, --max_direct_reads_before_nudge) were built on
the audit-check script afterward but never ported back here until now.
This is the first run of the ORIGINAL dose-response question with
those fixes in place -- genuinely unknown whether a real trend emerges
once the leak confound is reduced, or whether it's still null for a
different reason.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run import load_model
from lakv.qa_scoring import extract_qa_answer, exact_match_score, f1_score
from lakv.long_context_hotpotqa import build_long_context_examples
from lakv.rlm_repl_kv import RLMKVSession, make_long_context_system_prompt
from lakv.stats import discordant_pair_trend_across_conditions

# Fixed, --n-independent filler-pool offset -- see
# lakv/long_context_hotpotqa.py's filler_pool_start docstring and
# scripts/rlm_long_context_audit_check.py's identical constant for the
# bug this avoids (a same-seed run with a different n silently gets
# different filler for the same questions otherwise).
FILLER_POOL_START = 300


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n", type=int, default=25,
                         help="Examples per length -- kept modest per length "
                              "deliberately; the trend test pools evidence "
                              "across all lengths rather than needing a large "
                              "n at any single one.")
    parser.add_argument("--lengths", type=int, nargs="+", default=[2000, 8000, 16000, 32000],
                         help="Target token counts for the sweep, ascending.")
    parser.add_argument("--max_turns", type=int, default=15,
                         help="Higher than the plain HotpotQA default (10) -- "
                              "a much larger haystack plausibly needs more "
                              "turns to search before answering.")
    parser.add_argument("--filler_pool_size", type=int, default=500)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--source", default="hotpotqa", choices=["hotpotqa", "musique"],
                         help="Diagnosed 2026-10-02 (docs/RLM_LONG_CONTEXT_LOG.md, grounded in "
                              "the multi-hop QA literature): HotpotQA is a known instance of the "
                              "single-passage-shortcut problem this branch's 'escape hatch' "
                              "finding keeps running into (55-60%% of sessions never delegate at "
                              "all). MuSiQue (Trivedi et al., TACL 2022) is built specifically to "
                              "prevent this -- every reasoning hop is checked to be non-bypassable "
                              "at construction time. Same output shape, same needle-in-haystack "
                              "construction; only the underlying data source differs.")
    parser.add_argument("--output_dir", default="results/rlm_long_context_sweep")
    parser.add_argument("--seed", type=int, default=0,
                         help="Same seed across lengths gives nested haystacks "
                              "per question -- do not vary this between runs "
                              "you intend to compare in the same trend test.")
    parser.add_argument("--long_context_prompt", action="store_true",
                         help="Use the batch-delegation worked-example prompt instead of the "
                              "shared RLM_SYSTEM_PROMPT -- the fix that substantially reduced "
                              "the zero-delegation leak rate on the causal-audit branch. "
                              "Applied to BOTH text and kv channels equally, so it isn't a "
                              "confound on the comparison this script makes.")
    parser.add_argument("--batch_size", type=int, default=20,
                         help="Diagnosed 2026-10-01 (docs/RLM_LONG_CONTEXT_LOG.md): a dose-"
                              "response run under the default batch_size=20 found text "
                              "decisively beating kv at BOTH 2000 and 16000 tokens (10/10 "
                              "discordant pairs favoring text), the opposite of this branch's "
                              "founding hypothesis. The gap shrank between those lengths but "
                              "did not close, suggesting a real, batch-size-driven disadvantage "
                              "specific to kv (its raw splice may dilute relevant signal across "
                              "a large batch in a way text's decode-to-summary step doesn't) on "
                              "top of some genuine small-haystack mismatch at 2000 tokens. Pass "
                              "a small value (e.g. 3-5) to test directly whether kv recovers at "
                              "a smaller batch size, holding context length fixed -- only takes "
                              "effect with --long_context_prompt also set.")
    parser.add_argument("--max_direct_reads_before_nudge", type=int, default=None,
                         help="Same diagnosis as above -- nudges the model to delegate once it "
                              "has read this many passages directly with zero llm_query calls. "
                              "See lakv/rlm_repl_kv.py's docstring for the full rationale.")
    parser.add_argument("--repetition_nudge_max_fires", type=int, default=1,
                         help="1 matches original behavior (fires once per session). Pass "
                              "higher or 0 (unlimited) to let the repeated-query nudge keep "
                              "firing -- see scripts/rlm_long_context_audit_check.py's own "
                              "flag for the diagnosis motivating this.")
    args = parser.parse_args()
    repetition_nudge_max_fires = (
        None if args.repetition_nudge_max_fires == 0 else args.repetition_nudge_max_fires
    )
    system_prompt = (
        make_long_context_system_prompt(args.batch_size) if args.long_context_prompt else None
    )

    model, tokenizer = load_model(args.model_name, device="cuda")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[rlm_long_context_sweep] results will be saved to: {out_dir}")
    if args.long_context_prompt:
        print(f"[rlm_long_context_sweep] using batch-delegation prompt, batch_size={args.batch_size}")

    all_records = {}       # length -> channel -> list of per-example records
    per_length_pairs = {}  # length -> list of (kv_record, text_record)

    for length in args.lengths:
        print(f"\n{'=' * 70}\nCONTEXT LENGTH TARGET: {length} tokens\n{'=' * 70}")
        examples = build_long_context_examples(
            tokenizer, n=args.n, target_tokens=length, split=args.split,
            filler_pool_size=args.filler_pool_size, seed=args.seed,
            filler_pool_start=FILLER_POOL_START, source=args.source,
        )
        actual_tokens = [ex.approx_tokens for ex in examples]
        print(f"  actual token counts: min={min(actual_tokens)} max={max(actual_tokens)} "
              f"mean={sum(actual_tokens) / len(actual_tokens):.0f}")

        channel_records = {"text": [], "kv": []}
        for channel in ["text", "kv"]:
            session = RLMKVSession(
                model, tokenizer, device="cuda", return_channel=channel,
                max_turns=args.max_turns,
                max_direct_reads_before_nudge=args.max_direct_reads_before_nudge,
                system_prompt=system_prompt,
                repetition_nudge_max_fires=repetition_nudge_max_fires,
            )
            for i, ex in enumerate(examples):
                result = session.run(ex.question, ex.passages)
                if result.hit_max_turns:
                    pred, em, f1 = None, False, 0.0
                else:
                    pred = extract_qa_answer(result.answer or "")
                    em = exact_match_score(pred, ex.answer)
                    f1 = f1_score(pred, ex.answer)
                print(f"  [{channel}][{length}] ex{i}: EM={em} F1={f1:.2f} "
                      f"hit_max_turns={result.hit_max_turns} "
                      f"turns={len(result.turn_texts)} queries={result.llm_query_calls}")
                channel_records[channel].append({
                    "idx": i, "correct": bool(em), "f1": f1,
                    "predicted": pred, "raw_answer": result.answer,
                    "hit_max_turns": result.hit_max_turns,
                    "llm_query_calls": result.llm_query_calls,
                    "n_turns": len(result.turn_texts),
                    "approx_tokens": ex.approx_tokens,
                    "question": ex.question, "gold": ex.answer,
                    # Full transcript -- omitted from the first version of
                    # this script, which meant a real n=3 smoke test
                    # (2026-09-22) produced summary numbers with no way to
                    # diagnose WHY a specific example failed (garbage
                    # output? genuine search failure? a plumbing bug?)
                    # without rerunning. Saved from the start now, same as
                    # scripts/rlm_repl_kv_check.py already does.
                    "turn_texts": result.turn_texts,
                    "child_texts": result.child_texts,
                    "n_needle": ex.n_needle, "n_filler": ex.n_filler,
                })
            n_correct = sum(r["correct"] for r in channel_records[channel])
            mean_f1 = sum(r["f1"] for r in channel_records[channel]) / len(channel_records[channel])
            n_timeout = sum(r["hit_max_turns"] for r in channel_records[channel])
            print(f"  [{channel}] summary: {n_correct}/{len(examples)} EM "
                  f"({100 * n_correct / len(examples):.1f}%) mean F1={mean_f1:.3f} "
                  f"timeouts={n_timeout}/{len(examples)}")

        all_records[length] = channel_records
        # Positional zip is safe here (not mcnemar_test's idx-matched
        # _align_by_idx): both lists were built by iterating the SAME
        # `examples` list in the same order, immediately above.
        per_length_pairs[length] = list(zip(channel_records["kv"], channel_records["text"]))

    print(f"\n{'=' * 70}\nDOSE-RESPONSE SUMMARY\n{'=' * 70}")
    for length in args.lengths:
        kv_recs, text_recs = all_records[length]["kv"], all_records[length]["text"]
        kv_acc = sum(r["correct"] for r in kv_recs) / len(kv_recs)
        text_acc = sum(r["correct"] for r in text_recs) / len(text_recs)
        print(f"  {length:>6} tokens: kv={kv_acc * 100:5.1f}%  text={text_acc * 100:5.1f}%  "
              f"gap={100 * (kv_acc - text_acc):+5.1f} pts")

    trend = discordant_pair_trend_across_conditions(per_length_pairs)
    print(f"\nTrend test -- does kv's win rate among discordant pairs grow with length?")
    print(f"  z={trend['z_statistic']:.3f}  p={trend['p_value']:.4f}"
          f"  ({'SIGNIFICANT' if trend['p_value'] < 0.05 else 'not significant'} at alpha=0.05)")
    for length, info in trend["per_condition"].items():
        print(f"    {length:>6} tokens: kv-only={info['a_only']} text-only={info['b_only']} "
              f"n_discordant={info['n_discordant']} (of {info['n_pairs']} pairs)")

    out_path = out_dir / "sweep_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "args": vars(args),
            "records": all_records,
            "trend_test": trend,
        }, f, indent=2)
    print(f"\n[rlm_long_context_sweep] full results saved to: {out_path}")


if __name__ == "__main__":
    main()
