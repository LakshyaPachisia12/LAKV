"""
Post-hoc analysis for results/rlm_kv_check/*/transcripts.json: the
"delegated-and-completed" rate discovered 2026-09-22 while investigating
why the RLM+KV confidence signature's raw per-channel average looked
backwards (see CLAUDE.md finding 20's UPDATE 2026-09-22 for the full
story).

Filters each channel to sessions that both attempted at least one real
delegation (llm_query_calls > 0) AND still finished within the turn
budget (not hit_max_turns) -- the only subset where a causal-audit
substitution on the delegated KV could possibly have mattered, since a
session that never delegated never exercised the substitution at all
(confirmed directly: the same ~12/50 zero-delegation questions complete
identically in every channel, contaminating the raw per-channel average
identically regardless of audit condition), and a session that timed
out never produced anything to score in the first place.

No GPU required -- reads an already-saved transcripts.json from
scripts/rlm_repl_kv_check.py.

Usage:
    python scripts/analyze_delegation_completion.py results/rlm_kv_check/run_20260922_093855/transcripts.json
    python scripts/analyze_delegation_completion.py <path> --base_channel kv
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lakv.stats import mcnemar_test


def delegated_and_completed_flags(records: List[dict]) -> List[dict]:
    """One flag per example: True iff the session both called llm_query
    at least once AND produced a scoreable answer (confidence is None
    exactly when hit_max_turns is True, so checking confidence is not
    None doubles as the completion check without needing that field
    directly)."""
    return [
        {
            "idx": r["idx"],
            "correct": (r.get("confidence") is not None and r["llm_query_calls"] > 0),
        }
        for r in records
    ]


def analyze(data: dict, base_channel: str) -> None:
    channels = list(data["records"].keys())
    if base_channel not in channels:
        raise ValueError(f"base_channel {base_channel!r} not among run channels {channels}")
    n = len(data["records"][channels[0]])

    flags: Dict[str, List[dict]] = {c: delegated_and_completed_flags(data["records"][c]) for c in channels}

    # Sanity check surfaced during the 2026-09-22 investigation: the
    # zero-delegation-and-completed subset should be IDENTICAL across
    # every channel (those sessions never exercise the audit at all).
    # Flag it loudly if that's no longer true -- it would mean either
    # the run used different --n_held_out ranges per channel, or
    # something about the data path changed since this was verified.
    zero_deleg_idxs = {
        c: {r["idx"] for r in data["records"][c] if r.get("confidence") is not None and r["llm_query_calls"] == 0}
        for c in channels
    }
    all_same = len(set(frozenset(v) for v in zero_deleg_idxs.values())) <= 1
    if not all_same:
        print("[WARNING] zero-delegation-completion question sets differ across channels -- "
              "the 2026-09-22 finding assumed these are identical; investigate before trusting "
              "the numbers below.")

    print(f"Delegated-and-completed rate (of {n}):")
    for c in channels:
        n_yes = sum(f["correct"] for f in flags[c])
        print(f"  {c:20s} {n_yes}/{n} ({100 * n_yes / n:.0f}%)")

    print(f"\nPairwise McNemar vs. {base_channel!r}:")
    base = flags[base_channel]
    for c in channels:
        if c == base_channel:
            continue
        res = mcnemar_test(base, flags[c])
        print(f"  {base_channel} vs {c}: p={res.p_value:.4f} "
              f"(n_discordant={res.n_discordant}, {base_channel}-only={res.n_a_only}, {c}-only={res.n_b_only})")

    # Also report the remaining pairwise combinations among the non-base
    # channels -- e.g. mismatched vs zeroed/random, the legs this metric
    # was specifically built to resolve.
    others = [c for c in channels if c != base_channel]
    if len(others) > 1:
        print(f"\nPairwise McNemar among the remaining channels:")
        for i in range(len(others)):
            for j in range(i + 1, len(others)):
                a, b = others[i], others[j]
                res = mcnemar_test(flags[a], flags[b])
                print(f"  {a} vs {b}: p={res.p_value:.4f} (n_discordant={res.n_discordant})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("transcripts_json")
    parser.add_argument("--base_channel", default="kv",
                         help="Channel to compare every other channel against (default: kv, "
                              "the real/uncorrupted condition).")
    args = parser.parse_args()

    with open(args.transcripts_json, encoding="utf-8") as f:
        data = json.load(f)

    analyze(data, args.base_channel)


if __name__ == "__main__":
    main()
