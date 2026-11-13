"""Statistics and figure script for the relay-necessity curve. CPU only, UNTESTED ON GPU.

Synthetic results with a built-in dose-response, plus the two real endpoint runs that are tracked
in the repository (their documented numbers: level 0 diff -3 pp p=0.68; 100% hidden +32 pp p=4.4e-7).
"""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from lakv import necessity_stats as ns

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("make_necessity_curve", ROOT / "scripts" / "make_necessity_curve.py")
mk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mk)

N = 60


def per_sample(correct, n_gold_hidden=None, n_hidden=None):
    out = []
    for i, c in enumerate(correct):
        s = {"idx": i, "question": f"q{i}", "correct": bool(c), "f1": float(c)}
        if n_gold_hidden is not None:
            s["hide"] = {"n_gold_hidden": n_gold_hidden, "n_hidden": n_hidden if n_hidden is not None else n_gold_hidden}
        out.append(s)
    return out


def write_results(path, cfgs):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({c: {"summary": {}, "per_sample": ps} for c, ps in cfgs.items()}))


def dropped_acc(frac_hidden):                       # dropped accuracy falls as more is hidden
    return int(round(N * (0.6 - 0.4 * frac_hidden)))


def correct_array(k, offset=0):
    a = np.zeros(N, dtype=int)
    a[[(offset + j) % N for j in range(k)]] = 1
    return a


def synthetic_manifest(tmp_path, with_logs=True):
    levels = []
    real_k = int(N * 0.55)                          # real arm is flat: the cache compensates
    for axis, lv, hid_frac, gold_hid, n_hid in (("both", 0, 0.0, 0, 0), ("gold", 1, 0.5, 1, 1), ("gold", 2, 1.0, 2, 2),
                                                ("frac", 20, 0.2, None, 2), ("frac", 50, 0.5, None, 5),
                                                ("frac", 80, 0.8, None, 8), ("frac", 100, 1.0, None, 10)):
        f = tmp_path / f"{axis}{lv}" / "experiment_results.json"
        gh = gold_hid if with_logs and gold_hid is not None else None
        write_results(f, {"R": per_sample(correct_array(real_k), gh, n_hid), "D": per_sample(correct_array(dropped_acc(hid_frac)), gh, n_hid)})
        levels.append({"axis": axis, "level": lv, "reused": lv in (0, 100),
                       "real": {"file": str(f), "config": "R"}, "dropped": {"file": str(f), "config": "D"}})
    f = tmp_path / "seed1" / "experiment_results.json"
    write_results(f, {"R": per_sample(correct_array(real_k, 5)), "D": per_sample(correct_array(dropped_acc(0.5), 5))})
    return {"levels": levels, "sensitivity": [{"axis": "frac", "level": 50, "label": "hide50 seed1",
                                               "real": {"file": str(f), "config": "R"}, "dropped": {"file": str(f), "config": "D"}}]}


# ─── primitives ─────────────────────────────────────────────────────────────

def test_holm_known_example_and_properties():
    adj = ns.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})
    p = {f"t{i}": v for i, v in enumerate([1e-9, 0.2, 0.9, 0.04, 0.5, 0.03, 0.7])}
    adj = ns.holm_adjust(p)
    assert all(adj[k] >= p[k] for k in p) and all(0 <= v <= 1 for v in adj.values())
    order = sorted(p, key=p.get)
    assert [adj[k] for k in order] == sorted(adj[k] for k in order)            # monotone in raw p


def test_paired_correct_drops_skipped_and_unmatched():
    a = per_sample([1, 1, 0, 1]); b = per_sample([0, 1, 0, 0])
    b[1]["skipped"] = True
    del b[3]
    idx, r, d = ns.paired_correct(a, b)
    assert idx == [0, 2] and list(r) == [1, 0] and list(d) == [0, 0]


def test_accuracy_diff_ci_basics():
    r = np.ones(40); d = np.zeros(40)
    assert ns.accuracy_diff_ci(r, d, n_boot=500) == (1.0, 1.0, 1.0)
    assert ns.accuracy_diff_ci(r, r, n_boot=500) == (0.0, 0.0, 0.0)
    rng = np.random.default_rng(0)
    r, d = rng.integers(0, 2, 100).astype(float), rng.integers(0, 2, 100).astype(float)
    m, lo, hi = ns.accuracy_diff_ci(r, d, n_boot=2000)
    assert lo <= m <= hi and m == pytest.approx((r - d).mean())
    assert ns.accuracy_diff_ci(r, d, n_boot=2000) == (m, lo, hi)               # seeded: reproducible


def test_wilson_ci_known_value():
    lo, hi = ns.wilson_ci(50, 100)
    assert lo == pytest.approx(0.404, abs=0.002) and hi == pytest.approx(0.596, abs=0.002)
    assert ns.wilson_ci(0, 0)[0] != ns.wilson_ci(0, 0)[0]                      # NaN for n=0


def test_slope_recovers_a_known_dose_response_and_flat_data_includes_zero():
    n = 80
    nested = lambda k: np.concatenate([np.ones(k), np.zeros(n - k)])           # per-question diffs, nested
    res = ns.slope_bootstrap({0: nested(0), 1: nested(40), 2: nested(80)}, n_boot=3000)
    assert res["slope"] == pytest.approx(0.5) and res["ci_low"] > 0 and res["bootstrap_p_two_sided"] < 0.01
    rng = np.random.default_rng(1)
    base = rng.integers(-1, 2, n).astype(float)
    flat = ns.slope_bootstrap({0: base, 1: base.copy(), 2: base.copy()}, n_boot=3000)
    assert flat["slope"] == pytest.approx(0.0) and flat["ci_low"] <= 0 <= flat["ci_high"]
    assert ns.slope_bootstrap({0: nested(0), 1: nested(40), 2: nested(80)}, n_boot=500)["slope"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        ns.slope_bootstrap({0: nested(0)})


# ─── manifest -> table ──────────────────────────────────────────────────────

def test_table_structure_holm_family_and_dose_response(tmp_path):
    t = ns.build_table(synthetic_manifest(tmp_path), n_boot=1500)
    rows = t["rows"]
    assert len(rows) == 8 and sum(1 for r in rows if r["level"] == 0) == 2          # shared level 0 on both axes
    assert t["n_holm_tests"] == 7 and t["holm_family"] == sorted(["0", "gold1", "gold2", "frac20", "frac50", "frac80", "frac100"])
    assert all(r["holm_p"] >= r["mcnemar_p"] - 1e-12 for r in rows)
    zeros = [r for r in rows if r["level"] == 0]
    assert zeros[0]["holm_p"] == zeros[1]["holm_p"] and zeros[0]["diff_pp"] == zeros[1]["diff_pp"]
    diffs = {(r["axis"], r["level"]): r["diff_pp"] for r in rows}
    assert diffs[("frac", 100)] > diffs[("frac", 50)] > diffs[("frac", 20)] > diffs[("frac", 0)]
    assert diffs[("gold", 2)] > diffs[("gold", 1)] > diffs[("gold", 0)]
    for ax in ("gold", "frac"):
        s = t["slopes"][ax]
        assert s["slope"] > 0 and s["ci_low"] > 0 and s["n_questions"] == N
    r100 = next(r for r in rows if r["axis"] == "frac" and r["level"] == 100)
    assert r100["diff_ci_low_pp"] < r100["diff_pp"] < r100["diff_ci_high_pp"] and r100["reused"] is True
    assert r100["acc_real_ci_low"] < r100["acc_real"] < r100["acc_real_ci_high"]


def test_gold_hidden_is_taken_from_logs_when_present_else_nominal(tmp_path):
    t = ns.build_table(synthetic_manifest(tmp_path, with_logs=True), n_boot=300)
    g2 = next(r for r in t["rows"] if r["axis"] == "gold" and r["level"] == 2)
    f50 = next(r for r in t["rows"] if r["axis"] == "frac" and r["level"] == 50)
    assert g2["gold_hidden_source"] == "logged" and g2["mean_gold_hidden"] == 2.0
    assert f50["gold_hidden_source"] == "nominal" and f50["mean_gold_hidden"] == pytest.approx(1.0)   # 2 * 50/100
    t2 = ns.build_table(synthetic_manifest(tmp_path / "x", with_logs=False), n_boot=300)
    assert next(r for r in t2["rows"] if r["axis"] == "gold" and r["level"] == 1)["gold_hidden_source"] == "nominal"


def test_sensitivity_row_is_outside_the_holm_family_and_reports_difference_of_differences(tmp_path):
    t = ns.build_table(synthetic_manifest(tmp_path), n_boot=500)
    assert t["n_holm_tests"] == 7 and len(t["sensitivity"]) == 1
    s = t["sensitivity"][0]
    assert s["n_common"] == N and "diff_of_diffs_pp" in s
    assert s["diff_of_diffs_ci_low_pp"] <= s["diff_of_diffs_pp"] <= s["diff_of_diffs_ci_high_pp"]


def test_missing_config_fails_loudly(tmp_path):
    m = synthetic_manifest(tmp_path)
    m["levels"][1]["real"]["config"] = "NOPE"
    with pytest.raises(KeyError, match="NOPE"):
        ns.build_table(m, n_boot=100)


@pytest.mark.skipif(not (ROOT / "results" / "run_20261011_122408").exists(), reason="tracked results not available")
def test_real_endpoint_runs_reproduce_the_documented_numbers():
    m = {"levels": [
        {"axis": "both", "level": 0, "reused": True,
         "real": {"file": str(ROOT / "results/run_20261009_113254/experiment_results.json"), "config": "A"},
         "dropped": {"file": str(ROOT / "results/run_20261009_113254/experiment_results.json"), "config": "A_audit_dropped"}},
        {"axis": "frac", "level": 100, "reused": True,
         "real": {"file": str(ROOT / "results/run_20261011_122408/experiment_results.json"), "config": "A_nopass"},
         "dropped": {"file": str(ROOT / "results/run_20261011_122408/experiment_results.json"), "config": "A_nopass_audit_dropped"}}]}
    t = ns.build_table(m, n_boot=3000)
    lvl0 = next(r for r in t["rows"] if r["axis"] == "frac" and r["level"] == 0)
    lvl100 = next(r for r in t["rows"] if r["axis"] == "frac" and r["level"] == 100)
    assert (lvl0["n_pairs"], round(lvl0["diff_pp"])) == (100, -3) and lvl0["mcnemar_p"] == pytest.approx(0.68, abs=0.01)
    assert (lvl100["n_pairs"], round(lvl100["diff_pp"])) == (100, 32) and lvl100["mcnemar_p"] < 1e-6
    assert lvl100["diff_ci_low_pp"] > 15 and lvl0["diff_ci_low_pp"] < 0 < lvl0["diff_ci_high_pp"]


# ─── the figure script ──────────────────────────────────────────────────────

def test_script_writes_table_and_both_figures(tmp_path, capsys):
    m = synthetic_manifest(tmp_path / "data")
    mp = tmp_path / "m.json"; mp.write_text(json.dumps(m))
    out = tmp_path / "out"
    assert mk.main(["--manifest", str(mp), "--out_dir", str(out), "--n_boot", "300"]) == 0
    for name in ("necessity_curve.png", "necessity_curve_accuracy.png", "necessity_curve_table.csv", "necessity_curve_table.json"):
        assert (out / name).exists() and (out / name).stat().st_size > 1000, name
    assert (out / "necessity_curve.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    rows = list(csv.DictReader(open(out / "necessity_curve_table.csv", encoding="utf-8")))
    assert len(rows) == 8 + 1 and any("sensitivity" in r["axis"] for r in rows)
    assert {"diff_pp", "diff_ci_low_pp", "mcnemar_p", "holm_p", "mean_gold_hidden"} <= set(rows[0])
    text = capsys.readouterr().out
    assert "slope[gold]" in text and "slope[frac]" in text and "sensitivity" in text


def test_script_table_only_mode(tmp_path):
    mp = tmp_path / "m.json"; mp.write_text(json.dumps(synthetic_manifest(tmp_path / "data")))
    out = tmp_path / "out"
    assert mk.main(["--manifest", str(mp), "--out_dir", str(out), "--n_boot", "200", "--no_figures"]) == 0
    assert (out / "necessity_curve_table.csv").exists() and not (out / "necessity_curve.png").exists()


def test_example_manifest_is_valid_json_with_the_seven_levels_and_reused_endpoints():
    m = json.loads((ROOT / "docs" / "necessity_curve_manifest.example.json").read_text(encoding="utf-8"))
    assert len(m["levels"]) == 7
    assert sorted((l["axis"], l["level"]) for l in m["levels"]) == sorted(
        [("both", 0), ("gold", 1), ("gold", 2), ("frac", 20), ("frac", 50), ("frac", 80), ("frac", 100)])
    assert [l["reused"] for l in m["levels"] if l["level"] in (0, 100)] == [True, True]
    assert m["sensitivity"][0]["level"] == 50 and m["sensitivity"][0]["real"]["config"] == "A_hide50"
