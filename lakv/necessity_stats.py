"""Statistics for the relay-necessity curve.   *** UNTESTED ON GPU (CPU unit tests only) ***

Per level: real vs dropped accuracy on the SAME questions (paired by idx), difference in pp with
an exact McNemar p-value and a paired-bootstrap 95% CI. Across levels: Holm correction over the 7
distinct level tests (0, 20, 50, 80, 100, gold1, gold2; level 0 is shared by both axes), and a
bootstrap-over-questions slope of the paired difference against the level (levels are nested, so
they are paired by question; the slope uses all levels at once instead of comparing them pairwise).

A "manifest" (JSON) names, for each level, the results file and config name of the real arm and
the dropped arm, so reused runs (level 0, level 100%) and new runs combine freely.
See scripts/make_necessity_curve.py and docs/necessity_curve_manifest.example.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from lakv.stats import bootstrap_f1_diff_ci, mcnemar_test

N_GOLD_NOMINAL = 2          # HotpotQA distractor: 2 gold passages per question
RNG_SEED = 20261113


# ─── paired arrays ──────────────────────────────────────────────────────────

def paired_correct(real_ps: Sequence[dict], dropped_ps: Sequence[dict]) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """(idx list, real_correct, dropped_correct) for questions present and not skipped in both arms."""
    by = {s["idx"]: s for s in dropped_ps}
    idx, r, d = [], [], []
    for a in real_ps:
        b = by.get(a["idx"])
        if b is None or a.get("skipped") or b.get("skipped"):
            continue
        idx.append(a["idx"]); r.append(1.0 if a.get("correct") else 0.0); d.append(1.0 if b.get("correct") else 0.0)
    return idx, np.array(r), np.array(d)


# ─── bootstrap CI for an accuracy difference ────────────────────────────────

def bootstrap_mean_ci(x: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = RNG_SEED
                      ) -> Tuple[float, float, float]:
    """Mean of x with a percentile bootstrap CI (resampling questions)."""
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def accuracy_diff_ci(real_correct: np.ndarray, dropped_correct: np.ndarray, **kw) -> Tuple[float, float, float]:
    """Paired bootstrap CI of mean(real - dropped), as fractions (multiply by 100 for pp)."""
    return bootstrap_mean_ci(real_correct - dropped_correct, **kw)


def diff_of_diffs_ci(a_real, a_drop, b_real, b_drop, **kw) -> Tuple[float, float, float]:
    """Paired bootstrap CI of (diff_b - diff_a) for two hide configurations scored on the same questions."""
    return bootstrap_mean_ci((b_real - b_drop) - (a_real - a_drop), **kw)


# ─── Holm ───────────────────────────────────────────────────────────────────

def holm_adjust(pvalues: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni adjusted p-values (step-down, monotone, capped at 1)."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return {k: adj[k] for k in pvalues}


# ─── dose-response slope ────────────────────────────────────────────────────

def slope_bootstrap(diffs_by_level: Dict[float, np.ndarray], n_boot: int = 10000, alpha: float = 0.05,
                    seed: int = RNG_SEED) -> Dict:
    """Slope of the mean paired difference (real-correct minus dropped-correct) on the level x.

    diffs_by_level maps x -> per-question difference array; all arrays must be over the SAME
    questions in the same order. Because every level has the same number of questions, the OLS
    slope of the stacked data equals the slope of the per-level means. The CI resamples
    questions (columns) jointly across levels, keeping the pairing between levels.
    """
    xs = sorted(diffs_by_level)
    if len(xs) < 2:
        raise ValueError("need at least two levels for a slope")
    D = np.vstack([diffs_by_level[x] for x in xs])           # L x n
    x = np.array(xs, dtype=float)
    xc = x - x.mean()
    denom = float((xc ** 2).sum())
    n = D.shape[1]
    slope = float((xc * D.mean(axis=1)).sum() / denom)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = D[:, idx].mean(axis=2)                            # L x B
    boots = (xc[:, None] * means).sum(axis=0) / denom
    lo, hi = float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))
    p = float(min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean())))
    return {"slope": slope, "ci_low": lo, "ci_high": hi, "bootstrap_p_two_sided": p, "n_questions": n,
            "levels": xs}


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


# ─── manifest -> table ──────────────────────────────────────────────────────

def _load(path: str, cache: Dict[str, dict], base: Optional[Path]) -> dict:
    p = Path(path)
    if base is not None and not p.is_absolute() and not p.exists():
        p = base / p
    key = str(p)
    if key not in cache:
        cache[key] = json.loads(p.read_text(encoding="utf-8"))
    return cache[key]


def _arm(entry: dict, cache, base) -> List[dict]:
    d = _load(entry["file"], cache, base)
    cfg = entry["config"]
    if cfg not in d or "per_sample" not in d[cfg]:
        raise KeyError(f"config {cfg!r} not found in {entry['file']}")
    return d[cfg]["per_sample"]


def level_row(spec: dict, cache: Dict[str, dict], base: Optional[Path] = None, n_boot: int = 10000) -> Dict:
    """One row of the results table for a manifest level entry."""
    real_ps, drop_ps = _arm(spec["real"], cache, base), _arm(spec["dropped"], cache, base)
    idx, r, d = paired_correct(real_ps, drop_ps)
    n = len(idx)
    mc = mcnemar_test(real_ps, drop_ps)
    diff, lo, hi = accuracy_diff_ci(r, d, n_boot=n_boot)
    row = {
        "axis": spec["axis"], "level": spec["level"], "label": spec.get("label", ""),
        "reused": bool(spec.get("reused", False)), "n_pairs": n,
        "acc_real": float(r.mean()) if n else None, "acc_dropped": float(d.mean()) if n else None,
        "diff_pp": 100 * diff, "diff_ci_low_pp": 100 * lo, "diff_ci_high_pp": 100 * hi,
        "mcnemar_p": mc.p_value, "n_real_only": mc.n_a_only, "n_dropped_only": mc.n_b_only,
        "real_file": spec["real"]["file"], "real_config": spec["real"]["config"],
        "dropped_file": spec["dropped"]["file"], "dropped_config": spec["dropped"]["config"],
    }
    wlo_r, whi_r = wilson_ci(int(r.sum()), n)
    wlo_d, whi_d = wilson_ci(int(d.sum()), n)
    row.update(acc_real_ci_low=wlo_r, acc_real_ci_high=whi_r, acc_dropped_ci_low=wlo_d, acc_dropped_ci_high=whi_d)
    if any(s.get("f1") is not None for s in real_ps) and any(s.get("f1") is not None for s in drop_ps):
        fd, flo, fhi = bootstrap_f1_diff_ci(real_ps, drop_ps, n_boot=n_boot)
        row.update(f1_diff_pp=100 * fd, f1_ci_low_pp=100 * flo, f1_ci_high_pp=100 * fhi)
    # realized gold passages hidden: from the per-sample hide log when every real-arm sample has one
    logged = [s["hide"]["n_gold_hidden"] for s in real_ps if s.get("hide") and not s.get("skipped")]
    if logged and len(logged) == len([s for s in real_ps if not s.get("skipped")]):
        row["mean_gold_hidden"] = float(np.mean(logged)); row["gold_hidden_source"] = "logged"
        row["mean_passages_hidden"] = float(np.mean([s["hide"]["n_hidden"] for s in real_ps if s.get("hide")]))
    elif "mean_gold_hidden" in spec:
        row["mean_gold_hidden"] = float(spec["mean_gold_hidden"]); row["gold_hidden_source"] = "manifest"
    else:
        lvl = spec["level"]
        row["mean_gold_hidden"] = float(lvl if spec["axis"] == "gold" else N_GOLD_NOMINAL * lvl / 100.0)
        row["gold_hidden_source"] = "nominal"
    return row, r, d, idx


def build_table(manifest: dict, base: Optional[Path] = None, n_boot: int = 10000) -> Dict:
    """Everything the figure and the report need, from a manifest dict."""
    cache: Dict[str, dict] = {}
    rows, arrays = [], {}
    for spec in manifest["levels"]:
        axes = ["gold", "frac"] if spec["axis"] == "both" else [spec["axis"]]
        row, r, d, idx = level_row(dict(spec, axis=axes[0]), cache, base, n_boot)
        for ax in axes:
            rr = dict(row, axis=ax)
            rows.append(rr)
            arrays[(ax, spec["level"])] = (idx, r, d)
    # Holm over the DISTINCT tests: level 0 (shared) counts once
    tests, seen = {}, set()
    for rw in rows:
        key = "0" if rw["level"] == 0 else f"{rw['axis']}{rw['level']}"
        if key not in seen:
            seen.add(key); tests[key] = rw["mcnemar_p"]
    adj = holm_adjust(tests)
    for rw in rows:
        key = "0" if rw["level"] == 0 else f"{rw['axis']}{rw['level']}"
        rw["holm_p"] = adj[key]
    out = {"rows": rows, "holm_family": sorted(tests), "n_holm_tests": len(tests), "slopes": {}, "sensitivity": []}
    # slopes per axis (levels aligned on the questions common to all levels)
    for ax, to_x in (("gold", lambda lv: float(lv)), ("frac", lambda lv: lv / 100.0)):
        keys = sorted(k for k in arrays if k[0] == ax)
        if len(keys) < 2:
            continue
        common = set(arrays[keys[0]][0])
        for k in keys[1:]:
            common &= set(arrays[k][0])
        order = sorted(common)
        diffs = {}
        for k in keys:
            idx, r, d = arrays[k]
            pos = {i: j for j, i in enumerate(idx)}
            diffs[to_x(k[1])] = np.array([r[pos[i]] - d[pos[i]] for i in order])
        res = slope_bootstrap(diffs, n_boot=n_boot)
        res["unit"] = "change in (real-correct minus dropped-correct) per " + \
            ("gold passage hidden" if ax == "gold" else "100% of passages hidden")
        out["slopes"][ax] = res
    # seed-sensitivity pairs (not part of the Holm family)
    for spec in manifest.get("sensitivity", []):
        row, r, d, idx = level_row(spec, cache, base, n_boot)
        ref = next((k for k in arrays if k == (spec["axis"], spec["level"])), None)
        rec = dict(row)
        if ref is not None:
            ridx, rr, dd = arrays[ref]
            common = sorted(set(ridx) & set(idx))
            pa = {i: j for j, i in enumerate(ridx)}; pb = {i: j for j, i in enumerate(idx)}
            a_r = np.array([rr[pa[i]] for i in common]); a_d = np.array([dd[pa[i]] for i in common])
            b_r = np.array([r[pb[i]] for i in common]); b_d = np.array([d[pb[i]] for i in common])
            dd_, lo_, hi_ = diff_of_diffs_ci(a_r, a_d, b_r, b_d, n_boot=n_boot)
            rec.update(ref_diff_pp=next(x["diff_pp"] for x in rows if x["axis"] == spec["axis"] and x["level"] == spec["level"]),
                       diff_of_diffs_pp=100 * dd_, diff_of_diffs_ci_low_pp=100 * lo_, diff_of_diffs_ci_high_pp=100 * hi_,
                       n_common=len(common))
        out["sensitivity"].append(rec)
    return out
