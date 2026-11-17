"""CPU tests for RLM+KV length-matched ANSWER-REGION donors
(DiskAnswerRegionPool, AnswerRegionSource, RLMKVSession._splice_length_matched,
splice_child_kv(pre_sliced_start=...)). Design: LAKV_runs/fanin_length_matched_design.md."""
import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lakv.causal_audit import DiskAnswerRegionPool, NoMatchedDonor
from lakv.donor_traces import AnswerRegionSource
from lakv.rlm_repl_kv import RLMKVSession

THETA = 10000.0


def make_kv(L, layers=2, heads=2, dim=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return tuple((torch.randn(1, heads, L, dim, generator=g), torch.randn(1, heads, L, dim, generator=g))
                 for _ in range(layers))


def length(kv):
    return kv[0][0].shape[2]


def test_pre_sliced_splice_equals_full_cache_splice_exactly():
    root, child, p = make_kv(30, seed=1), make_kv(50, seed=2), 20
    full = RLMKVSession.splice_child_kv(root, child, rope_theta=THETA, answer_only_from=p)
    region = tuple((k[:, :, p:], v[:, :, p:]) for k, v in child)
    pre = RLMKVSession.splice_child_kv(root, region, rope_theta=THETA, pre_sliced_start=p)
    assert length(full) == length(pre) == 30 + 30
    for (k, v), (k2, v2) in zip(full, pre):
        assert torch.equal(k, k2) and torch.equal(v, v2)


def test_default_splice_unchanged_by_new_argument():
    root, child = make_kv(10, seed=3), make_kv(6, seed=4)
    a = RLMKVSession.splice_child_kv(root, child, rope_theta=THETA)
    b = RLMKVSession.splice_child_kv(root, child, rope_theta=THETA, answer_only_from=0, pre_sliced_start=None)
    for (k, v), (k2, v2) in zip(a, b):
        assert torch.equal(k, k2) and torch.equal(v, v2)


@pytest.fixture
def pool(tmp_path):
    p = DiskAnswerRegionPool(tmp_path / "pool")
    # (question, prompt_len, total_len)  -> answer lengths 40, 41, 60, 100
    for i, (q, pl, tot) in enumerate([("qa", 200, 240), ("qb", 90, 131), ("qc", 300, 360), ("qd", 50, 150)]):
        p.add(0, q, make_kv(tot, seed=30 + i), question_text=f"held-out {q}", prompt_len=pl)
    p.mark_complete()
    return p


def test_pool_stores_only_the_answer_region_and_matches_on_answer_length(pool):
    src = AnswerRegionSource(pool)
    real_child = make_kv(500, seed=9)                 # total 500 but answer region only 42 (prompt 458)
    donor, log = src.fetch(agent_idx=0, question_key="scored", real_kv=real_child, rng=random.Random(0), prompt_len=458)
    assert log["real_answer_len"] == 42 and log["real_prompt_len"] == 458
    assert length(donor) in (40, 41)                  # matched on ANSWER length, not total length (500)
    assert abs(length(donor) - 42) <= 0.05 * 42
    assert log["pad_positions"] == 0 and log["truncated_positions"] == 0
    assert log["donor_prompt_len"] in (200, 90) and log["donor_question_key"].startswith(("qa#", "qb#"))
    # stored region is exactly positions [prompt_len:] of the donor's full cache, unmodified
    seeds = {"qa": (30, 200, 240), "qb": (31, 90, 131)}
    qk = log["donor_question_key"].split("#")[0]
    seed, pl, tot = seeds[qk]
    ref = make_kv(tot, seed=seed)
    for (k, v), (kr, vr) in zip(donor, ref):
        assert torch.equal(k, kr[:, :, pl:]) and torch.equal(v, vr[:, :, pl:])


def test_no_padding_all_positions_real_content(pool):
    donor, _ = AnswerRegionSource(pool).fetch(0, "scored", make_kv(300), rng=random.Random(1), prompt_len=240)  # answer 60
    assert length(donor) == 60
    for k, v in donor:
        assert (k.abs().amax(dim=(0, 1, 3)) > 0).all() and (v.abs().amax(dim=(0, 1, 3)) > 0).all()


def test_skip_when_no_donor_within_5_percent(pool):
    with pytest.raises(NoMatchedDonor) as ei:
        AnswerRegionSource(pool).fetch(0, "scored", make_kv(300), rng=random.Random(0), prompt_len=220)  # answer 80
    assert ei.value.log["skipped"] is True and "no donor within 5%" in ei.value.reason


def test_strict_window_has_no_slack_for_short_answers(tmp_path):
    p = DiskAnswerRegionPool(tmp_path / "p")
    p.add(0, "q1", make_kv(30, seed=1), prompt_len=10)       # answer 20
    p.mark_complete()
    src = AnswerRegionSource(p)
    src.fetch(0, "s", make_kv(40), rng=random.Random(0), prompt_len=20)          # answer 20: exact match
    src.fetch(0, "s", make_kv(41), rng=random.Random(0), prompt_len=20)          # answer 21: |20-21| = 1 <= 1.05
    with pytest.raises(NoMatchedDonor):
        src.fetch(0, "s", make_kv(42), rng=random.Random(0), prompt_len=20)      # answer 22: 2 > 1.1


def test_same_question_never_its_own_donor_and_multiple_calls_per_question(tmp_path):
    p = DiskAnswerRegionPool(tmp_path / "p")
    p.add(0, "qa", make_kv(100, seed=1), prompt_len=60)      # call 0 of qa: answer 40
    p.add(0, "qa", make_kv(101, seed=2), prompt_len=60)      # call 1 of qa: answer 41
    p.add(0, "qb", make_kv(100, seed=3), prompt_len=60)      # qb: answer 40
    p.mark_complete()
    assert p.size(0) == 3
    seen = {AnswerRegionSource(p).fetch(0, "qa", make_kv(100), rng=random.Random(s), prompt_len=60)[1]["donor_question_key"]
            for s in range(20)}
    assert all(k.startswith("qb#") for k in seen)             # never qa's own answer regions


def test_entries_without_an_answer_region_are_ignored(tmp_path):
    p = DiskAnswerRegionPool(tmp_path / "p")
    p.add(0, "q", make_kv(30), prompt_len=0)
    p.add(0, "q", make_kv(30), prompt_len=30)
    assert p.size(0) == 0


def test_pool_reopen_roundtrip(pool):
    again = DiskAnswerRegionPool.open(pool.dir)
    assert again.complete and again.size(0) == 4
    _, log = AnswerRegionSource(again).fetch(0, "scored", make_kv(300), rng=random.Random(0), prompt_len=240)
    assert log["donor_prompt_len"] == 300 and log["donor_cache_len"] == 60


class _Cfg:
    rope_theta = THETA


class _Model:
    config = _Cfg()


def _session(pool, seed=0):
    s = RLMKVSession.__new__(RLMKVSession)
    s.model = _Model()
    s.donor_source = AnswerRegionSource(pool)
    s.audit_donor_seed = seed
    return s


def test_splice_length_matched_places_donor_answer_after_root_with_donor_rope_start(pool):
    s = _session(pool)
    root, real_child = make_kv(25, seed=5), make_kv(300, seed=6)
    spliced, log = s._splice_length_matched(root, real_child, prompt_len=240, q_key="scored", call_index=0)  # answer 60
    assert length(spliced) == 25 + 60 and log["mode"] == "length_matched" and log["call_index"] == 0
    assert log["donor_prompt_len"] == 300 and log["pad_positions"] == 0
    # equals splicing the donor's full cache with answer_only_from = the DONOR's prompt_len (not the real one)
    donor_full = make_kv(360, seed=32)
    ref = RLMKVSession.splice_child_kv(root, donor_full, rope_theta=THETA, answer_only_from=300)
    for (k, v), (kr, vr) in zip(spliced, ref):
        assert torch.equal(k, kr) and torch.equal(v, vr)
    # the old (padded) path would have used the REAL prompt_len on the donor's total-length-matched cache
    wrong = RLMKVSession.splice_child_kv(root, make_kv(360, seed=32), rope_theta=THETA, answer_only_from=240)
    assert not torch.equal(spliced[0][0], wrong[0][0])


def test_splice_length_matched_is_deterministic_and_order_independent(tmp_path):
    p = DiskAnswerRegionPool(tmp_path / "p")
    for i in range(6):
        p.add(0, f"q{i}", make_kv(100 + i, seed=i), prompt_len=60)     # answers 40..45
    p.mark_complete()
    s = _session(p, seed=7)
    a = s._splice_length_matched(make_kv(10), make_kv(100, seed=99), 60, "scored", 0)[1]["donor_question_key"]
    s._splice_length_matched(make_kv(10), make_kv(100, seed=99), 60, "other_question", 3)      # unrelated draw in between
    b = s._splice_length_matched(make_kv(10), make_kv(100, seed=99), 60, "scored", 0)[1]["donor_question_key"]
    assert a == b
    c = _session(p, seed=8)._splice_length_matched(make_kv(10), make_kv(100, seed=99), 60, "scored", 0)[1]["donor_question_key"]
    keys = {_session(p, seed=sd)._splice_length_matched(make_kv(10), make_kv(100, seed=99), 60, "scored", 0)[1]["donor_question_key"]
            for sd in range(12)}
    assert len(keys) > 1                                              # the seed really changes the draw


def test_splice_length_matched_propagates_skip(pool):
    with pytest.raises(NoMatchedDonor):
        _session(pool)._splice_length_matched(make_kv(25), make_kv(400), 270, "scored", 0)       # answer 130: none
