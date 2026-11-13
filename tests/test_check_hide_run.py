"""scripts/check_hide_run.py on results built from the real hiding logic (CPU only; UNTESTED ON GPU)."""
import importlib.util
import json
from pathlib import Path

import pytest

from lakv import necessity_curve as nc

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("check_hide_run", ROOT / "scripts" / "check_hide_run.py")
ck = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ck)


@pytest.fixture(scope="module")
def items():
    d = json.loads((ROOT / "results" / "run_20261009_113254" / "experiment_results.json").read_text(encoding="utf-8"))
    out = []
    for s in d["A"]["per_sample"][:30]:
        passages, _ = nc.split_blob(s["question"])
        order = sorted(range(len(passages)), key=lambda j: nc.stable_hash("pseudo", s["question"], j))
        out.append((s["idx"], s["question"], [nc._title_of(passages[j]) for j in order[:2]]))
    return out


def arm(items, spec, seed=0, answer="x"):
    ps = []
    for idx, blob, gold in items:
        _, info = nc.make_receiver_question(blob, gold, spec, seed)
        ps.append({"idx": idx, "hide": info, "raw_answer": answer, "predicted": answer, "correct": True})
    return {"summary": {}, "per_sample": ps}


def results(items, specs=(nc.HideSpec("gold", 1), nc.HideSpec("frac", 50)), seed=0):
    r = {"_run_metadata": {"hide_seed": seed, "dataset_provenance": {"hf_id": "x"}}}
    for sp in specs:
        r["A" + sp.suffix] = arm(items, sp, seed)
        r["A" + sp.suffix + "_audit_dropped"] = arm(items, sp, seed)
    return r


def test_correct_run_passes(items):
    # item idx 27 (2 passages) is not in the first 30 questions used here
    res = ck.check_hide_results(results(items), expect_seed=0)
    assert res["violations"] == [] and res["warnings"] == [] and res["n_hide_configs"] == 4
    assert res["configs"]["A_hidegold1"]["mean_gold_hidden"] == 1.0
    assert res["configs"]["A_hide50"]["mean_gold_hidden"] == pytest.approx(1.0)
    assert sorted(p[2] for p in res["pairs_identical"]) == [30, 30]


def test_all_levels_pass_on_the_real_blobs(items):
    specs = [nc.HideSpec("frac", l) for l in (20, 50, 80)] + [nc.HideSpec("gold", l) for l in (1, 2)]
    assert ck.check_hide_results(results(items, specs), expect_seed=0)["violations"] == []


def test_different_hidden_sets_between_real_and_dropped_fail(items):
    r = results(items)
    r["A_hide50_audit_dropped"] = arm(items, nc.HideSpec("frac", 50), seed=1)    # a different seed => different sets
    res = ck.check_hide_results(r)
    assert any("hidden passages differ" in v for v in res["violations"])


def test_gold_axis_must_hide_exactly_the_requested_gold(items):
    r = results(items)
    r["A_hidegold1"]["per_sample"][3]["hide"]["n_gold_hidden"] = 0
    assert any("exactly 1 gold" in v for v in ck.check_hide_results(r)["violations"])


def test_fractional_counts_checked_and_seed_mismatch_detected(items):
    r = results(items)
    r["A_hide50"]["per_sample"][0]["hide"]["n_hidden"] = 4
    assert any("expected 5" in v for v in ck.check_hide_results(r)["violations"])
    assert any("!= expected 7" in v for v in ck.check_hide_results(results(items), expect_seed=7)["violations"])


def test_missing_partner_arm_and_missing_metadata(items):
    r = results(items)
    del r["A_hidegold1_audit_dropped"]
    r["_run_metadata"] = {}
    res = ck.check_hide_results(r)
    assert any("no matching A_hidegold1_audit_dropped" in v for v in res["violations"])
    assert len(res["warnings"]) == 2


def test_degenerate_answers_are_reported_not_gated(items):
    r = results(items)
    for s in r["A_hide50"]["per_sample"]:
        s["raw_answer"], s["predicted"] = "", None
    res = ck.check_hide_results(r)
    assert res["configs"]["A_hide50"]["frac_empty_answer"] == 1.0 and res["violations"] == []


def test_cli_exit_status(tmp_path, items, capsys):
    good = tmp_path / "g.json"; good.write_text(json.dumps(results(items)))
    assert ck.main([str(good), "--expect_seed", "0"]) == 0 and "RESULT: PASS" in capsys.readouterr().out
    bad = results(items); bad["A_hide50_audit_dropped"] = arm(items, nc.HideSpec("frac", 50), seed=1)
    b = tmp_path / "b.json"; b.write_text(json.dumps(bad))
    assert ck.main([str(b)]) == 1 and "FAIL" in capsys.readouterr().out
    empty = tmp_path / "e.json"; empty.write_text(json.dumps({"A": {"per_sample": [{"idx": 0}]}}))
    assert ck.main([str(empty)]) == 1
