"""The GPU-checklist readout script, on synthetic results (CPU only; UNTESTED ON GPU)."""
import importlib.util
import json
import sys
from pathlib import Path

from lakv.donor_traces import donor_dir_name

_spec = importlib.util.spec_from_file_location(
    "check_matched_donors", Path(__file__).resolve().parent.parent / "scripts" / "check_matched_donors.py")
cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cm)


def sample(idx, ok, ans, logs=(), skipped=False):
    return {"idx": idx, "correct": None if skipped else ok, "f1": None if skipped else float(ok),
            "raw_answer": ans, "audit_logs": list(logs), "skipped": skipped}


def rebuild(agent, cos=0.9999, diff=0.01):
    return {"agent_idx": agent, "mode": "self_replay", "pad_positions": 0,
            "rebuild_vs_real": {"comparable": True, "min_cosine": cos, "max_abs_diff": diff, "max_rel_l2": 0.01}}


def test_self_replay_summary():
    A = [sample(i, i % 2 == 0, f"a{i}") for i in range(4)]
    R = [sample(0, True, "a0", [rebuild(1), rebuild(2, 0.99, 0.2)]), sample(1, False, "a1", [rebuild(1), rebuild(2)]),
         sample(2, True, "DIFFERENT", [rebuild(1), rebuild(2)]), sample(3, False, "a3", skipped=True)]
    res = {"A": {"summary": {"accuracy": 0.5}, "per_sample": A},
           "A_audit_self_replay": {"summary": {"accuracy": 0.5}, "per_sample": R}}
    out = cm.summarize_self_replay(res)
    assert out["n_pairs"] == 3 and abs(out["identical_answer_frac"] - 2 / 3) < 1e-9
    assert out["rebuild_vs_real"][2]["min_cosine_worst"] == 0.99 and out["rebuild_vs_real"][2]["max_abs_diff_worst"] == 0.2
    assert out["rebuild_vs_real"][1]["n"] == 3


def test_donor_log_checks_flag_padding_and_out_of_window_ratios():
    good = {"mode": "length_matched", "skipped": False, "pad_positions": 0, "truncated_positions": 0, "len_ratio": 1.04}
    bad_pad = dict(good, pad_positions=3)
    bad_ratio = dict(good, len_ratio=1.2)
    ok = {"A_audit_length_matched": {"summary": {}, "per_sample": [sample(0, True, "x", [good]),
                                                                    sample(1, False, "", skipped=True)]}}
    r = cm.check_donor_logs(ok, "A_audit_length_matched")
    assert r["violations"] == [] and r["n_skipped"] == 1 and r["len_ratio_max"] == 1.04
    for bad in (bad_pad, bad_ratio):
        r = cm.check_donor_logs({"A_audit_length_matched": {"summary": {}, "per_sample": [sample(0, True, "x", [bad])]}},
                                "A_audit_length_matched")
        assert r["violations"]


def test_same_question_donor_flags_reported():
    logs = [{"mode": "matched_wrong_answer", "skipped": False, "pad_positions": 0, "truncated_positions": 0,
             "defends_distractor": [True, False], "hop0_differs_from_greedy": None},
            {"mode": "matched_wrong_answer", "skipped": False, "pad_positions": 0, "truncated_positions": 0,
             "defends_distractor": [True, True]}]
    r = cm.check_donor_logs({"A_audit_matched_wrong_answer": {"summary": {}, "per_sample": [sample(0, True, "x", logs)]}},
                            "A_audit_matched_wrong_answer")
    assert r["defends_distractor_rate"] == 0.75 and r["violations"] == []


def test_donor_dir_name_separates_seeds_for_sampled_traces_only():
    assert donor_dir_name("matched_passages", 0) != donor_dir_name("matched_passages", 1)
    assert donor_dir_name("matched_wrong_answer", 0) == donor_dir_name("matched_wrong_answer", 5) == "matched_wrong_answer"
    assert donor_dir_name("length_matched", 3) == "length_matched"


def test_stats_cli_flags_skipped_samples_and_paired_subset(tmp_path, monkeypatch, capsys):
    from lakv import stats
    a = [{"idx": i, "correct": i != 3, "f1": float(i != 3)} for i in range(4)]
    b = [{"idx": 0, "correct": True, "f1": 1.0}, {"idx": 1, "correct": None, "f1": None, "skipped": True},
         {"idx": 2, "correct": False, "f1": 0.0}, {"idx": 3, "correct": None, "f1": None, "skipped": True}]
    f = tmp_path / "r.json"
    f.write_text(json.dumps({"A": {"summary": {"accuracy": 0.75}, "per_sample": a},
                             "X": {"summary": {"accuracy": 0.5}, "per_sample": b}}))
    monkeypatch.setattr(sys, "argv", ["stats", str(f), "A", "X", "--n-boot", "50"])
    stats._cli()
    out = capsys.readouterr().out
    assert "NOTE skipped samples" in out and "2 paired" in out
