"""
Standalone test for lakv/stats.py (paired McNemar test + bootstrap F1 CI).

No model / GPU / real results file required — uses small synthetic
per_sample lists shaped exactly like experiment_results.json's
"per_sample" entries.

Run directly for a printout:
    python tests/test_stats.py

Or via pytest (from repo root):
    pytest tests/test_stats.py -v -s
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lakv.stats import (
    mcnemar_test, bootstrap_f1_diff_ci, estimate_power_at_n_multiplier, compare,
    cochran_armitage_trend_test, discordant_pair_trend_across_conditions,
)


def _sample(idx, correct, f1=None):
    return {"idx": idx, "correct": correct, "f1": f1}


def test_mcnemar_identical_configs_gives_p_one():
    # Same outcomes on both sides -> zero discordant pairs -> p=1.0, not a
    # crash and not a spuriously "significant" result.
    a = [_sample(i, bool(i % 2)) for i in range(20)]
    b = [_sample(i, bool(i % 2)) for i in range(20)]
    res = mcnemar_test(a, b)
    assert res.n_discordant == 0
    assert res.p_value == 1.0


def test_mcnemar_detects_strong_difference():
    # A correct on everything, B correct on nothing -> maximally discordant,
    # should come out significant.
    n = 30
    a = [_sample(i, True) for i in range(n)]
    b = [_sample(i, False) for i in range(n)]
    res = mcnemar_test(a, b)
    assert res.n_discordant == n
    assert res.p_value < 0.01


def test_mcnemar_aligns_by_idx_not_position():
    # b is the same data as a but shuffled in list order, plus one extra
    # sample a doesn't have and missing one a does have. If this aligned by
    # list position instead of idx, the result would be wrong/crash.
    a = [_sample(0, True), _sample(1, False), _sample(2, True)]
    b = [_sample(2, True), _sample(0, True), _sample(99, False)]
    res = mcnemar_test(a, b)
    # only idx 0 and 2 overlap, both concordant (True/True) -> zero discordant
    assert res.n_both_correct == 2
    assert res.n_discordant == 0


def test_mcnemar_raises_on_no_overlap():
    a = [_sample(0, True)]
    b = [_sample(1, True)]
    try:
        mcnemar_test(a, b)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_bootstrap_f1_diff_ci_recovers_known_difference():
    # A's F1 is uniformly 0.2 higher than B's on every paired example ->
    # observed diff should be ~0.2 and the CI should exclude zero.
    n = 50
    a = [_sample(i, True, f1=0.7) for i in range(n)]
    b = [_sample(i, True, f1=0.5) for i in range(n)]
    diff, lo, hi = bootstrap_f1_diff_ci(a, b, n_boot=2000, seed=0)
    assert abs(diff - 0.2) < 1e-9
    assert lo <= diff <= hi
    assert lo > 0  # CI should exclude zero given a constant, noiseless gap


def test_bootstrap_f1_diff_ci_no_difference_includes_zero():
    n = 50
    a = [_sample(i, True, f1=0.5) for i in range(n)]
    b = [_sample(i, True, f1=0.5) for i in range(n)]
    diff, lo, hi = bootstrap_f1_diff_ci(a, b, n_boot=2000, seed=0)
    assert diff == 0.0
    assert lo <= 0.0 <= hi


def test_bootstrap_f1_diff_ci_drops_none_f1_samples():
    # gsm8k-style samples with f1=None should be excluded, not crash.
    a = [_sample(0, True, f1=None), _sample(1, True, f1=0.8)]
    b = [_sample(0, True, f1=None), _sample(1, True, f1=0.6)]
    diff, lo, hi = bootstrap_f1_diff_ci(a, b, n_boot=500, seed=0)
    assert abs(diff - 0.2) < 1e-9


def test_power_estimate_increases_with_n_for_a_real_but_underpowered_effect():
    # A small, real effect (55% vs 45% on the discordant pairs) at small n
    # should show power increasing as the simulated n grows.
    import random
    rng = random.Random(0)
    n = 20
    a, b = [], []
    for i in range(n):
        # ~30% discordant rate, slightly favoring A among discordant pairs
        r = rng.random()
        if r < 0.15:
            a.append(_sample(i, True)); b.append(_sample(i, False))   # A-only
        elif r < 0.27:
            a.append(_sample(i, False)); b.append(_sample(i, True))   # B-only
        elif r < 0.6:
            a.append(_sample(i, True)); b.append(_sample(i, True))
        else:
            a.append(_sample(i, False)); b.append(_sample(i, False))

    power = estimate_power_at_n_multiplier(a, b, multipliers=(1, 5), n_sim=500, seed=0)
    assert power[5]["power"] >= power[1]["power"] - 0.05  # allow small MC noise


def test_compare_high_level_matches_manual_calls():
    results_json = {
        "A": {
            "summary": {"accuracy": 0.6},
            "per_sample": [_sample(i, i < 6, f1=0.7 if i < 6 else 0.3) for i in range(10)],
        },
        "B": {
            "summary": {"accuracy": 0.4},
            "per_sample": [_sample(i, i < 4, f1=0.5 if i < 4 else 0.2) for i in range(10)],
        },
    }
    out = compare(results_json, "A", "B", n_boot=500)
    assert abs(out["accuracy_delta"] - 0.2) < 1e-9
    assert "f1_diff" in out
    assert out["f1_diff"] > 0


def test_cochran_armitage_detects_clear_increasing_trend():
    # Proportions rise cleanly 0.1 -> 0.9 across 5 ordered groups of 30 each
    # -- an unmistakable trend, should come out significant.
    successes = [3, 9, 15, 21, 27]
    totals = [30, 30, 30, 30, 30]
    z, p = cochran_armitage_trend_test(successes, totals)
    assert z > 0
    assert p < 0.001


def test_cochran_armitage_flat_proportions_not_significant():
    # Same proportion (0.5) in every group -> no real trend.
    successes = [15, 14, 16, 15, 15]
    totals = [30, 30, 30, 30, 30]
    z, p = cochran_armitage_trend_test(successes, totals)
    assert p > 0.5


def test_cochran_armitage_decreasing_trend_gives_negative_z():
    successes = [27, 21, 15, 9, 3]
    totals = [30, 30, 30, 30, 30]
    z, p = cochran_armitage_trend_test(successes, totals)
    assert z < 0
    assert p < 0.001


def test_cochran_armitage_respects_unequal_spacing_scores():
    # Same successes/totals, but scores are wildly unevenly spaced (e.g.
    # context lengths 2000/8000/16000/32000, not 0/1/2/3) -- the trend
    # should still be detected and the function shouldn't crash or
    # silently ignore the scores argument.
    successes = [3, 12, 20, 27]
    totals = [30, 30, 30, 30]
    z_equal, p_equal = cochran_armitage_trend_test(successes, totals)
    z_real, p_real = cochran_armitage_trend_test(successes, totals, scores=[2000, 8000, 16000, 32000])
    assert p_equal < 0.001
    assert p_real < 0.001
    assert z_equal != z_real  # unequal spacing genuinely changes the statistic


def test_cochran_armitage_raises_on_mismatched_lengths():
    try:
        cochran_armitage_trend_test([1, 2], [10, 10, 10])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_discordant_pair_trend_detects_growing_advantage():
    # 4 conditions (e.g. context lengths). At each, kv (config "a")'s
    # share of discordant pairs grows: 20% -> 40% -> 60% -> 80%.
    per_condition = {}
    for i, length in enumerate([2000, 8000, 16000, 32000]):
        kv_win_rate = 0.2 + 0.2 * i
        n_disc = 20
        n_a_only = round(n_disc * kv_win_rate)
        n_b_only = n_disc - n_a_only
        pairs = (
            [(_sample(j, True), _sample(j, False)) for j in range(n_a_only)]
            + [(_sample(j, False), _sample(j, True)) for j in range(n_a_only, n_a_only + n_b_only)]
        )
        per_condition[length] = pairs
    result = discordant_pair_trend_across_conditions(per_condition)
    assert result["z_statistic"] > 0
    assert result["p_value"] < 0.05
    assert result["per_condition"][32000]["a_only"] > result["per_condition"][2000]["a_only"]


def test_discordant_pair_trend_no_growth_not_significant():
    # kv's discordant win rate is flat (~50%) across all conditions.
    per_condition = {}
    for length in [2000, 8000, 16000, 32000]:
        pairs = (
            [(_sample(j, True), _sample(j, False)) for j in range(10)]
            + [(_sample(j, False), _sample(j, True)) for j in range(10, 20)]
        )
        per_condition[length] = pairs
    result = discordant_pair_trend_across_conditions(per_condition)
    assert result["p_value"] > 0.3


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
