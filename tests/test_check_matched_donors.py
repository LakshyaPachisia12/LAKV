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
            "raw_answer": ans, "predicted": None if skipped else ans, "audit_logs": list(logs), "skipped": skipped}


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
    assert out["n_answers_agree"] == 2 and out["n_rebuild_logs"] == 6 and out["not_comparable"] == 0
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


# ─── self-replay GATE: answer agreement (+ not_comparable), cosine is diagnostic only ───

def replay_results(n=20, n_disagree=0, cos=0.9999, comparable=True, logs_per_sample=2):
    A = [sample(i, True, f"ans{i}") for i in range(n)]
    R = []
    for i in range(n):
        ans = f"CHANGED{i}" if i < n_disagree else f"ans{i}"
        logs = []
        for agent in (1, 2)[:logs_per_sample]:
            r = rebuild(agent, cos=cos)
            if not comparable:
                r["rebuild_vs_real"] = {"comparable": False, "len_a": 10, "len_b": 11}
            logs.append(r)
        R.append(sample(i, True, ans, logs))
    return {"A": {"summary": {"accuracy": 1.0}, "per_sample": A},
            "A_audit_self_replay": {"summary": {"accuracy": 1.0}, "per_sample": R}}


def gate(res, **kw):
    return cm.self_replay_gate(cm.summarize_self_replay(res), **kw)


def test_gate_passes_with_19_of_20_agreeing():
    g = gate(replay_results(n_disagree=1))
    assert g["pass"] and g["reasons"] == []


def test_gate_passes_with_20_of_20():
    assert gate(replay_results(n_disagree=0))["pass"]


def test_gate_fails_with_18_of_20_agreeing():
    g = gate(replay_results(n_disagree=2))
    assert not g["pass"] and any("18 of 20" in r for r in g["reasons"])


def test_low_cosine_does_not_gate_when_answers_agree():
    # cosine 0.5 would have failed the old 0.999 criterion; it must not gate anymore
    g = gate(replay_results(n_disagree=0, cos=0.5))
    assert g["pass"]


def test_not_comparable_always_fails_even_if_every_answer_agrees():
    g = gate(replay_results(n_disagree=0, comparable=False))
    assert not g["pass"] and any("not_comparable" in r and "cache_gen_len" in r for r in g["reasons"])


def test_wrong_number_of_comparable_samples_fails():
    assert not gate(replay_results(n=10))["pass"]                       # default expects 20
    assert gate(replay_results(n=10), expect_pairs=10, min_agree=9)["pass"]


def test_missing_rebuild_logs_fail_the_gate():
    g = gate(replay_results(n_disagree=0, logs_per_sample=1))
    assert not g["pass"] and any("rebuild logs" in r for r in g["reasons"])


def test_answer_agreement_is_normalized_and_handles_none():
    f = cm.answers_agree
    assert f({"predicted": "The Answer."}, {"predicted": "answer"})      # case/article/punctuation normalized
    assert f({"predicted": None}, {"predicted": None})                   # both unparsed
    assert not f({"predicted": None}, {"predicted": "x"})
    assert not f({"predicted": "paris"}, {"predicted": "london"})


def test_cli_exit_status_follows_the_gate_and_prints_cosine_as_diagnostic(tmp_path, monkeypatch, capsys):
    import sys as _sys
    def run(res):
        f = tmp_path / "r.json"
        f.write_text(json.dumps(res))
        monkeypatch.setattr(_sys, "argv", ["check", str(f)])
        rc = cm.main()
        return rc, capsys.readouterr().out
    rc, out = run(replay_results(n_disagree=1, cos=0.5))
    assert rc == 0 and "GATE: PASS" in out and "DIAGNOSTICS" in out and "min_cosine_worst" in out
    rc, out = run(replay_results(n_disagree=3))
    assert rc == 1 and "GATE: FAIL" in out and "FAIL REASON" in out
    rc, out = run(replay_results(n_disagree=0, comparable=False))
    assert rc == 1 and "not_comparable" in out
