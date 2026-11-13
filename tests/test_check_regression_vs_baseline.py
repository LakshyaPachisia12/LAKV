"""scripts/check_regression_vs_baseline.py on small synthetic JSON files (CPU only; UNTESTED ON GPU)."""
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "check_regression_vs_baseline",
    Path(__file__).resolve().parent.parent / "scripts" / "check_regression_vs_baseline.py")
cr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cr)


def sample(idx, raw="The answer is: x", pred="x", hops=("r", "v")):
    return {"idx": idx, "question": f"Q{idx}", "raw_answer": raw, "predicted": pred, "hop_texts": list(hops)}


def write(tmp, name, samples, config="A"):
    p = tmp / name
    p.write_text(json.dumps({"_run_metadata": {"x": 1}, config: {"summary": {}, "per_sample": samples}}))
    return str(p)


def test_identical_subset_passes_and_ignores_extra_baseline_samples(tmp_path, capsys):
    base = write(tmp_path, "base.json", [sample(i) for i in range(100)])
    new = write(tmp_path, "new.json", [sample(i) for i in range(20)])
    assert cr.main([new, base]) == 0
    out = capsys.readouterr().out
    assert "n compared: 20" in out and "n identical (raw_answer and predicted): 20" in out and "n differing: 0" in out
    assert "PASS" in out


def test_difference_fails_and_prints_first_five_only(tmp_path, capsys):
    base = write(tmp_path, "base.json", [sample(i) for i in range(20)])
    new = write(tmp_path, "new.json", [sample(i, raw="The answer is: WRONG", pred="WRONG") if i < 8 else sample(i)
                                       for i in range(20)])
    assert cr.main([new, base]) == 1
    out = capsys.readouterr().out
    assert "n differing: 8" in out and "n identical (raw_answer and predicted): 12" in out
    assert out.count("  idx ") == 5 and "... and 3 more" in out


def test_raw_answer_text_difference_alone_counts(tmp_path):
    base = write(tmp_path, "b.json", [sample(0, raw="The answer is: x")])
    new = write(tmp_path, "n.json", [sample(0, raw="The answer is: x.")])  # same parsed answer, different text
    assert cr.main([new, base]) == 1


def test_predicted_difference_alone_counts(tmp_path):
    base = write(tmp_path, "b.json", [sample(0, pred="x")])
    new = write(tmp_path, "n.json", [sample(0, pred="y")])
    assert cr.main([new, base]) == 1


def test_hop_text_difference_is_diagnostic_only(tmp_path, capsys):
    base = write(tmp_path, "b.json", [sample(0, hops=("r", "v"))])
    new = write(tmp_path, "n.json", [sample(0, hops=("r2", "v"))])
    assert cr.main([new, base]) == 0
    assert "DIAGNOSTIC" in capsys.readouterr().out


def test_no_overlap_or_question_mismatch_fails(tmp_path):
    base = write(tmp_path, "b.json", [sample(50)])
    new = write(tmp_path, "n.json", [sample(0)])
    assert cr.main([new, base]) == 1                      # nothing compared
    other = write(tmp_path, "o.json", [dict(sample(50), question="a different question")])
    assert cr.main([other, base]) == 1                    # same idx, different example


def test_missing_config_exits_2(tmp_path):
    base = write(tmp_path, "b.json", [sample(0)])
    new = write(tmp_path, "n.json", [sample(0)], config="D")
    assert cr.main([new, base, "--config", "A"]) == 2


def test_default_baseline_path_is_the_paper_run():
    assert cr.DEFAULT_BASELINE.endswith("run_20261009_113254/experiment_results.json")


def test_compare_runs_pure_function_counts():
    new = {"A": {"per_sample": [sample(0), sample(1, pred="z"), sample(5)]}}
    base = {"A": {"per_sample": [sample(0), sample(1), sample(2)]}}
    r = cr.compare_runs(new, base)
    assert (r["n_compared"], r["n_identical"], r["n_differing"], r["n_only_in_new"]) == (2, 1, 1, 1)


def write_cfg(tmp, name, cfg, samples):
    p = tmp / name
    p.write_text(json.dumps({cfg: {"summary": {}, "per_sample": samples}}))
    return str(p)


def test_baseline_config_name_may_differ(tmp_path, capsys):
    # relay-necessity curve: new A_hide100 must reproduce the baseline's A_nopass arm
    new = write_cfg(tmp_path, "n.json", "A_hide100", [sample(i) for i in range(5)])
    base = write_cfg(tmp_path, "b.json", "A_nopass", [sample(i) for i in range(20)])
    assert cr.main([new, base, "--config", "A_hide100", "--baseline-config", "A_nopass"]) == 0
    out = capsys.readouterr().out
    assert "baseline config A_nopass" in out and "n compared: 5" in out
    # without --baseline-config the baseline lookup uses the same name and fails with exit 2
    assert cr.main([new, base, "--config", "A_hide100"]) == 2


def test_baseline_config_difference_is_detected(tmp_path):
    new = write_cfg(tmp_path, "n.json", "A_hide100", [sample(0, pred="x")])
    base = write_cfg(tmp_path, "b.json", "A_nopass", [sample(0, pred="y")])
    assert cr.main([new, base, "--config", "A_hide100", "--baseline-config", "A_nopass"]) == 1
