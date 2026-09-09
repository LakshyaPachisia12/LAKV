"""
Cross-model test of Finding 6 (falsified calibration-confidence predictor):
does a calibration procedure's own tier-separation confidence correlate
with how much accuracy/F1 it costs to act on that ranking (layer-selection
degradation, A vs D)?

Finding 6 as written is n=2 (Qwen 0.491 separation / smaller degradation,
Mistral 0.673 separation / larger degradation) -- real, but not yet a
correlation, just one inverted pair. This module turns >=3 models' worth
of (profile, A-vs-D results) into an actual Pearson/Spearman correlation.

Usage once N model runs exist:
    python -m lakv.generalization \
        --model qwen profiles/<run>/qwen_hotpotqa.json results/<run>/experiment_results.json \
        --model mistral profiles/<run>/mistral-7b-instruct-v0_3_hotpotqa.json results/<run>/experiment_results.json \
        --model llama profiles/<run>/llama-3.1-8b-instruct_hotpotqa.json results/<run>/experiment_results.json \
        ...

Each --model takes: <label> <profile_json_path> <experiment_results_json_path>.
Degradation is computed as A's F1 minus D's F1 (falls back to accuracy if F1
absent, e.g. gsm8k). Needs >=3 models for a correlation to mean anything;
prints a clear warning rather than a spurious coefficient below that.
"""

import argparse
import json
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ModelPoint:
    label: str
    tier_separation: float
    degradation: float  # A's metric minus D's metric; positive = D costs accuracy/F1


def tier_separation(profile_path: str) -> float:
    """Mean importance score gap between kept (tier 1) and dropped (tier 3)
    layers -- the same metric Finding 6 already reports for Qwen (0.491)
    and Mistral (0.673). Uses attention_importance_norm (0-1 scale, matches
    the two already-reported numbers), not the raw or joint score."""
    with open(profile_path, encoding="utf-8") as f:
        p = json.load(f)
    tiers = p["tier_assignment"]
    scores = p["attention_importance_norm"]
    tier1 = [s for s, t in zip(scores, tiers) if t == 1]
    tier3 = [s for s, t in zip(scores, tiers) if t == 3]
    if not tier1 or not tier3:
        raise ValueError(f"{profile_path}: profile has no tier-1 or no tier-3 layers, can't compute separation")
    return (sum(tier1) / len(tier1)) - (sum(tier3) / len(tier3))


def degradation(results_path: str, cfg_a: str = "A", cfg_d: str = "D") -> float:
    """A's mean metric minus D's mean metric, matched by sample idx via the
    same per-sample intersection lakv.stats uses (not just summary deltas,
    though for this module a summary-level delta is precise enough since
    we only need the point estimate, not a CI, per model)."""
    with open(results_path, encoding="utf-8") as f:
        d = json.load(f)
    sa, sd = d[cfg_a]["summary"], d[cfg_d]["summary"]
    if "mean_f1" in sa and "mean_f1" in sd:
        return sa["mean_f1"] - sd["mean_f1"]
    return sa["accuracy"] - sd["accuracy"]


def pearson_r(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return 0.0
    return cov / (vx ** 0.5 * vy ** 0.5)


def spearman_r(xs: List[float], ys: List[float]) -> float:
    def rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        for r, i in enumerate(order):
            ranks[i] = r
        return ranks
    return pearson_r(rank(xs), rank(ys))


def analyze(points: List[ModelPoint]) -> dict:
    xs = [p.tier_separation for p in points]
    ys = [p.degradation for p in points]
    out = {
        "n_models": len(points),
        "points": [(p.label, p.tier_separation, p.degradation) for p in points],
    }
    if len(points) < 3:
        out["warning"] = (
            f"n={len(points)} models: a correlation coefficient here would be "
            "meaningless (2 points always give |r|=1). Need >=3 model families "
            "before this number means anything -- report the raw points, not a "
            "correlation, until then."
        )
        out["pearson_r"] = None
        out["spearman_r"] = None
    else:
        out["pearson_r"] = pearson_r(xs, ys)
        out["spearman_r"] = spearman_r(xs, ys)
        # Finding 6's claim is that HIGHER confidence (separation) associates
        # with HIGHER degradation -- i.e. a POSITIVE correlation would extend
        # the inversion; a negative/zero correlation would mean the n=2
        # pattern doesn't generalize and Finding 6 should be reported as
        # exactly what it was: two data points, not a trend.
    return out


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model", action="append", nargs=3, metavar=("LABEL", "PROFILE_JSON", "RESULTS_JSON"),
        required=True, dest="models",
    )
    parser.add_argument("--cfg-a", default="A")
    parser.add_argument("--cfg-d", default="D")
    args = parser.parse_args()

    points = []
    for label, profile_path, results_path in args.models:
        sep = tier_separation(profile_path)
        deg = degradation(results_path, args.cfg_a, args.cfg_d)
        points.append(ModelPoint(label=label, tier_separation=sep, degradation=deg))
        print(f"{label:<20} tier_separation={sep:.4f}  degradation({args.cfg_a}-{args.cfg_d})={deg:.4f}")

    result = analyze(points)
    print()
    if result["pearson_r"] is not None:
        print(f"Pearson r  = {result['pearson_r']:.4f}")
        print(f"Spearman r = {result['spearman_r']:.4f}")
    else:
        print(result["warning"])


if __name__ == "__main__":
    _cli()
