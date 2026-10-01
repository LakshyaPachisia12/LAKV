"""
Analyze the "escape hatch" rate across context lengths from an already-
saved scripts/rlm_long_context_sweep.py run. No GPU required.

Motivation (2026-10-01 research audit): existing agent-engineering
literature treats "tool-skip" / "tool bypass" (an agent answering
directly instead of delegating) purely as a reliability bug to detect
and fix -- nobody appears to treat its RATE as a dose-response quantity
worth measuring as a function of task difficulty, or connects it to
causal-audit validity in agentic systems specifically. This project has
two anecdotal data points so far (24% at native ~10-passage HotpotQA
scale, 55-60% at ~90-130-passage long-context scale,
docs/RLM_LONG_CONTEXT_LOG.md) suggesting the escape-hatch rate GROWS,
not shrinks, as the task scales toward the regime the delegation
architecture is designed for -- the opposite of the naive expectation
that a bigger haystack forces more genuine tool use. This script turns
that into an actual measured trend across the dose-response sweep's own
lengths, using data already collected (no new GPU run needed if a sweep
has already been run with --long_context_prompt and/or
--max_direct_reads_before_nudge).

Definitions, per example:
  escape_hatch   : llm_query_calls == 0 AND NOT hit_max_turns -- the
                   model answered without ever delegating, successfully
                   bypassing the architecture's own mechanism entirely.
  delegated_done : llm_query_calls > 0 AND NOT hit_max_turns -- genuine
                   delegation that also finished (the only subset where
                   a causal-audit substitution on the delegated content
                   could possibly matter).
  gave_up        : llm_query_calls == 0 AND hit_max_turns -- never tried
                   the tool AND never found an answer any other way
                   either; not an "escape," just stalled.

Usage:
    python scripts/analyze_escape_hatch_rate.py results/rlm_long_context_sweep/run_.../sweep_results.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lakv.stats import cochran_armitage_trend_test


def classify(records):
    escape_hatch = sum(1 for r in records if r["llm_query_calls"] == 0 and not r["hit_max_turns"])
    delegated_done = sum(1 for r in records if r["llm_query_calls"] > 0 and not r["hit_max_turns"])
    gave_up = sum(1 for r in records if r["llm_query_calls"] == 0 and r["hit_max_turns"])
    deleg_timeout = sum(1 for r in records if r["llm_query_calls"] > 0 and r["hit_max_turns"])
    return {
        "n": len(records),
        "escape_hatch": escape_hatch,
        "delegated_done": delegated_done,
        "gave_up": gave_up,
        "delegated_timeout": deleg_timeout,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_results_json")
    parser.add_argument("--channel", default="kv",
                         help="Which channel to analyze (default: kv -- the channel a "
                              "causal audit would actually be run on).")
    args = parser.parse_args()

    with open(args.sweep_results_json, encoding="utf-8") as f:
        data = json.load(f)

    lengths = sorted(int(k) for k in data["records"].keys())
    print(f"Escape-hatch rate across context lengths, channel={args.channel!r}:\n")
    print(f"{'length':>8} {'n':>4} {'escape_hatch':>13} {'delegated_done':>15} "
          f"{'gave_up':>8} {'deleg_timeout':>14}")

    successes = []  # escape_hatch count, for the trend test
    totals = []     # n, for the trend test
    for length in lengths:
        records = data["records"][str(length)][args.channel]
        c = classify(records)
        print(f"{length:>8} {c['n']:>4} "
              f"{c['escape_hatch']:>6}/{c['n']:<3} ({100*c['escape_hatch']/c['n']:>5.1f}%) "
              f"{c['delegated_done']:>8}/{c['n']:<3} ({100*c['delegated_done']/c['n']:>5.1f}%) "
              f"{c['gave_up']:>8} {c['delegated_timeout']:>14}")
        successes.append(c["escape_hatch"])
        totals.append(c["n"])

    if len(lengths) >= 2:
        z, p = cochran_armitage_trend_test(successes, totals, scores=[float(l) for l in lengths])
        direction = "GROWS" if z > 0 else "SHRINKS"
        print(f"\nTrend test (does the escape-hatch rate change with context length?): "
              f"z={z:.3f} p={p:.4f}")
        print(f"  Escape-hatch rate {direction} with length "
              f"({'significant' if p < 0.05 else 'not significant'} at alpha=0.05).")
        print("  (naive expectation: a bigger haystack should force more genuine tool use, "
              "i.e. the rate should SHRINK -- this project's prior anecdotal data points the "
              "other way)")


if __name__ == "__main__":
    main()
