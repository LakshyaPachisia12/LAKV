"""CPU-only unit tests for scripts/analyze_delegation_completion.py, the
reusable version of the ad-hoc analysis that found the 2026-09-22
"delegated-and-completed" metric (CLAUDE.md finding 20's UPDATE
2026-09-22). No real transcripts.json needed -- synthetic records shaped
exactly like RLMKVRunResult's saved fields."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_delegation_completion import delegated_and_completed_flags, analyze


def _record(idx, confidence, llm_query_calls):
    return {"idx": idx, "confidence": confidence, "llm_query_calls": llm_query_calls}


def test_flags_true_only_when_completed_and_delegated():
    records = [
        _record(0, {"mean_top1_prob": 0.9}, llm_query_calls=2),   # completed, delegated -> True
        _record(1, {"mean_top1_prob": 0.9}, llm_query_calls=0),   # completed, NOT delegated -> False
        _record(2, None, llm_query_calls=3),                       # timed out (confidence None) -> False
        _record(3, None, llm_query_calls=0),                       # timed out, no delegation -> False
    ]
    flags = delegated_and_completed_flags(records)
    assert [f["correct"] for f in flags] == [True, False, False, False]
    assert [f["idx"] for f in flags] == [0, 1, 2, 3]


def test_analyze_reproduces_known_rates(capsys):
    # 10 questions (idx 0-9), same identity across every channel --
    # matching the real invariant this script's sanity check assumes:
    # idx 0-1 are the "zero-delegation, easy enough to answer directly"
    # questions that complete identically regardless of channel.
    ZERO_DELEG_IDXS = {0, 1}

    def make_channel(delegated_completed_idxs, n_total=10):
        recs = []
        for idx in range(n_total):
            if idx in ZERO_DELEG_IDXS:
                recs.append(_record(idx, {"mean_top1_prob": 0.9}, llm_query_calls=0))
            elif idx in delegated_completed_idxs:
                recs.append(_record(idx, {"mean_top1_prob": 0.9}, llm_query_calls=1))
            else:
                recs.append(_record(idx, None, llm_query_calls=2))  # timed out
        return recs

    data = {
        "records": {
            "kv": make_channel(delegated_completed_idxs={2, 3, 4}),
            "kv_audit_zeroed": make_channel(delegated_completed_idxs=set()),
            "kv_audit_random": make_channel(delegated_completed_idxs=set()),
        }
    }
    analyze(data, base_channel="kv")
    out = capsys.readouterr().out
    assert "kv                   3/10 (30%)" in out
    assert "kv_audit_zeroed      0/10 (0%)" in out
    assert "kv_audit_random      0/10 (0%)" in out
    assert "[WARNING]" not in out  # zero-deleg sets ARE identical here


def test_analyze_warns_when_zero_delegation_sets_differ(capsys):
    # Deliberately give channel B a DIFFERENT zero-delegation-completion
    # question set than channel A -- should trigger the sanity-check
    # warning rather than silently reporting numbers built on a broken
    # assumption.
    data = {
        "records": {
            "kv": [
                _record(0, {"p": 1}, llm_query_calls=0),
                _record(1, None, llm_query_calls=1),
            ],
            "kv_audit_zeroed": [
                _record(0, None, llm_query_calls=1),
                _record(1, {"p": 1}, llm_query_calls=0),  # different idx is the zero-deleg completion
            ],
        }
    }
    analyze(data, base_channel="kv")
    out = capsys.readouterr().out
    assert "[WARNING]" in out


def test_analyze_raises_on_unknown_base_channel():
    data = {"records": {"kv": [_record(0, None, 0)]}}
    try:
        analyze(data, base_channel="not_a_real_channel")
        assert False, "expected ValueError"
    except ValueError:
        pass
