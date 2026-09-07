"""
LAKV Module: paired statistical testing between configs.

Phase 0a of the research-extensions plan (see
lakv_research_extensions_prompt.md, delivered alongside this branch). Exists
because every accuracy/F1 delta in the established results table (CLAUDE.md)
has so far been reported as a bare percentage with no significance test — at
n=100, several of the currently-claimed gaps between configs may not be
distinguishable from chance. This module answers that with a paired test
(McNemar's, exact binomial form) for accuracy and a paired bootstrap CI for
F1, both computed on the SAME example set across two configs, matched by
sample index rather than by list position (list-position matching is exactly
the class of bug that broke offset_corrector.py once before in this repo —
see CLAUDE.md's "E is a documented negative result" section).

No GPU required. Run directly against an existing experiment_results.json:

    python -m lakv.stats results/run_20260904_123559/experiment_results.json A D B_int8 A

compares A-vs-D and B_int8-vs-A (config names given in pairs).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats as _scipy_stats


@dataclass
class McNemarResult:
    n_a_only: int       # correct under A, wrong under B
    n_b_only: int       # correct under B, wrong under A
    n_both_correct: int
    n_both_wrong: int
    p_value: float

    @property
    def n_discordant(self) -> int:
        return self.n_a_only + self.n_b_only


def _align_by_idx(per_sample_a: List[dict], per_sample_b: List[dict]) -> List[Tuple[dict, dict]]:
    """Pair per-sample records from two configs' per_sample lists by 'idx'.

    Matching by idx (not list position) matters because a config that hit an
    OOM-skip, or a partial/resumed run, can have per_sample lists that are
    the same length but not aligned sample-for-sample — silently zipping them
    by position would compare the wrong examples against each other without
    any error.
    """
    by_idx_b = {s["idx"]: s for s in per_sample_b}
    pairs = []
    for sa in per_sample_a:
        sb = by_idx_b.get(sa["idx"])
        if sb is not None:
            pairs.append((sa, sb))
    return pairs


def mcnemar_test(per_sample_a: List[dict], per_sample_b: List[dict]) -> McNemarResult:
    """Exact (binomial) McNemar's test on paired correct/incorrect outcomes.

    Uses the exact binomial form rather than the chi-square approximation
    with continuity correction: exact is the safer default whenever the
    discordant-pair count is small, which is likely at n=100-500 on a task
    in the 30-60% accuracy range — the chi-square approximation can be
    unreliable in that regime.
    """
    pairs = _align_by_idx(per_sample_a, per_sample_b)
    if not pairs:
        raise ValueError("No overlapping sample indices between the two config runs")

    n_a_only = sum(1 for sa, sb in pairs if sa["correct"] and not sb["correct"])
    n_b_only = sum(1 for sa, sb in pairs if sb["correct"] and not sa["correct"])
    n_both_correct = sum(1 for sa, sb in pairs if sa["correct"] and sb["correct"])
    n_both_wrong = sum(1 for sa, sb in pairs if not sa["correct"] and not sb["correct"])

    n_disc = n_a_only + n_b_only
    if n_disc == 0:
        p_value = 1.0
    else:
        k = min(n_a_only, n_b_only)
        p_value = _scipy_stats.binomtest(k, n_disc, 0.5, alternative="two-sided").pvalue

    return McNemarResult(n_a_only, n_b_only, n_both_correct, n_both_wrong, p_value)


def bootstrap_f1_diff_ci(
    per_sample_a: List[dict], per_sample_b: List[dict],
    n_boot: int = 10000, alpha: float = 0.05, seed: int = 0,
) -> Tuple[float, float, float]:
    """Paired bootstrap CI for the difference in mean F1 (A minus B).

    Paired — each bootstrap iteration resamples the same example indices for
    both configs — rather than two independent CIs. Independent CIs can both
    look wide while the paired difference is actually tight (or the reverse);
    only the paired version answers the question actually being asked: "is A
    reliably better than B on the SAME examples." Samples with f1=None
    (GSM8K, or an OOM-skipped item) are dropped from both sides before
    pairing.
    """
    pairs = [
        (sa["f1"], sb["f1"])
        for sa, sb in _align_by_idx(per_sample_a, per_sample_b)
        if sa.get("f1") is not None and sb.get("f1") is not None
    ]
    if not pairs:
        raise ValueError("No paired samples with non-null F1 between the two config runs")

    arr = np.array(pairs, dtype=float)  # shape (n, 2)
    n = arr.shape[0]
    observed_diff = float(arr[:, 0].mean() - arr[:, 1].mean())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    resampled = arr[idx]  # (n_boot, n, 2)
    diffs = resampled[:, :, 0].mean(axis=1) - resampled[:, :, 1].mean(axis=1)

    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return observed_diff, float(lo), float(hi)


def estimate_power_at_n_multiplier(
    per_sample_a: List[dict], per_sample_b: List[dict],
    multipliers: Tuple[float, ...] = (1, 1.5, 2, 3, 5),
    alpha: float = 0.05, n_sim: int = 2000, seed: int = 0,
) -> Dict[float, dict]:
    """Monte-Carlo estimate of McNemar power at larger sample sizes.

    Deliberately simulation-based rather than a closed-form McNemar power
    formula: this draws directly from the OBSERVED multinomial distribution
    over (both-wrong, A-only, B-only, both-correct) in the current run, so it
    doesn't depend on correctly recalling/deriving a power formula — it
    answers "if the true effect is what we observed, how often would a
    larger run of size n detect it," which is the practically useful
    question when deciding how big the next run needs to be.
    """
    pairs = _align_by_idx(per_sample_a, per_sample_b)
    n0 = len(pairs)
    if n0 == 0:
        raise ValueError("No overlapping sample indices")

    cat = np.array([
        1 * (sa["correct"] and not sb["correct"])
        + 2 * (sb["correct"] and not sa["correct"])
        + 3 * (sa["correct"] and sb["correct"])
        for sa, sb in pairs
    ])
    # categories: 0=both wrong, 1=A-only, 2=B-only, 3=both correct
    probs = np.array([np.mean(cat == k) for k in range(4)])

    rng = np.random.default_rng(seed)
    results: Dict[float, dict] = {}
    for mult in multipliers:
        n = max(1, round(n0 * mult))
        sig_count = 0
        for _ in range(n_sim):
            draw = rng.choice(4, size=n, p=probs)
            n_a_only = int(np.sum(draw == 1))
            n_b_only = int(np.sum(draw == 2))
            n_disc = n_a_only + n_b_only
            if n_disc == 0:
                p = 1.0
            else:
                k = min(n_a_only, n_b_only)
                p = _scipy_stats.binomtest(k, n_disc, 0.5, alternative="two-sided").pvalue
            if p < alpha:
                sig_count += 1
        results[mult] = {"n": n, "power": sig_count / n_sim}
    return results


def compare(results_json: dict, cfg_a: str, cfg_b: str, n_boot: int = 10000, alpha: float = 0.05) -> dict:
    """High-level convenience: given a loaded experiment_results.json dict and
    two config names, return accuracy delta + McNemar p-value, and (if F1 is
    present, i.e. not gsm8k) F1 delta + paired bootstrap CI."""
    pa = results_json[cfg_a]["per_sample"]
    pb = results_json[cfg_b]["per_sample"]

    mc = mcnemar_test(pa, pb)
    acc_a = results_json[cfg_a]["summary"]["accuracy"]
    acc_b = results_json[cfg_b]["summary"]["accuracy"]

    out = {
        "config_a": cfg_a, "config_b": cfg_b,
        "accuracy_a": acc_a, "accuracy_b": acc_b,
        "accuracy_delta": acc_a - acc_b,
        "mcnemar_p_value": mc.p_value,
        "mcnemar_n_discordant": mc.n_discordant,
        "mcnemar_a_only": mc.n_a_only,
        "mcnemar_b_only": mc.n_b_only,
        "significant_at_0.05": mc.p_value < 0.05,
    }

    has_f1 = any(s.get("f1") is not None for s in pa) and any(s.get("f1") is not None for s in pb)
    if has_f1:
        diff, lo, hi = bootstrap_f1_diff_ci(pa, pb, n_boot=n_boot, alpha=alpha)
        out.update({
            "f1_diff": diff, "f1_diff_ci_low": lo, "f1_diff_ci_high": hi,
            "f1_ci_excludes_zero": (lo > 0) or (hi < 0),
        })
    return out


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Paired significance testing between LAKV configs from an experiment_results.json file."
    )
    parser.add_argument("results_json", help="Path to experiment_results.json")
    parser.add_argument(
        "config_pairs", nargs="+",
        help="Config names given in pairs, e.g. A D B_int8 A compares A-vs-D and B_int8-vs-A",
    )
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument(
        "--power-check", action="store_true",
        help="Also Monte-Carlo estimate McNemar power at 1x/1.5x/2x/3x/5x the current n",
    )
    args = parser.parse_args()

    if len(args.config_pairs) % 2 != 0:
        parser.error("config_pairs must be given in pairs, e.g. A D B_int8 A")

    with open(args.results_json) as f:
        data = json.load(f)

    for i in range(0, len(args.config_pairs), 2):
        cfg_a, cfg_b = args.config_pairs[i], args.config_pairs[i + 1]
        if cfg_a not in data or cfg_b not in data:
            print(f"[skip] {cfg_a} vs {cfg_b}: one or both configs not found in {args.results_json}")
            continue

        result = compare(data, cfg_a, cfg_b, n_boot=args.n_boot)
        print(f"\n=== {cfg_a} vs {cfg_b} ===")
        print(
            f"  accuracy: {result['accuracy_a']*100:.1f}% vs {result['accuracy_b']*100:.1f}% "
            f"(delta {result['accuracy_delta']*100:+.1f} pts)"
        )
        verdict = "SIGNIFICANT" if result["significant_at_0.05"] else "not significant"
        print(f"  McNemar exact p-value: {result['mcnemar_p_value']:.4f}  ({verdict} at alpha=0.05)")
        print(
            f"  discordant pairs: {result['mcnemar_n_discordant']} "
            f"({cfg_a}-only={result['mcnemar_a_only']}, {cfg_b}-only={result['mcnemar_b_only']})"
        )
        if "f1_diff" in result:
            ci_verdict = "excludes zero" if result["f1_ci_excludes_zero"] else "includes zero — not conclusive"
            print(
                f"  F1 delta: {result['f1_diff']*100:+.1f} pts, "
                f"95% bootstrap CI [{result['f1_diff_ci_low']*100:+.1f}, "
                f"{result['f1_diff_ci_high']*100:+.1f}] pts  ({ci_verdict})"
            )

        if args.power_check:
            pa, pb = data[cfg_a]["per_sample"], data[cfg_b]["per_sample"]
            power = estimate_power_at_n_multiplier(pa, pb)
            print("  power to detect this same effect at larger n (Monte Carlo estimate):")
            for mult, info in power.items():
                print(f"    n~{info['n']:>4} ({mult}x current n): power={info['power']*100:.0f}%")


if __name__ == "__main__":
    _cli()
