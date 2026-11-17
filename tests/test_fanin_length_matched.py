"""CPU tests for fan-in length-matched donors, the RoPE-gap control and merge extra_shifts
(lakv/recursive_pipeline.py). Design: LAKV_runs/fanin_length_matched_design.md."""
import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lakv.anchor_table import AnchorTable
from lakv.causal_audit import DiskKVAuditPool, NoMatchedDonor, apply_causal_audit
from lakv.donor_traces import LengthMatchedSource
from lakv.recursive_pipeline import RecursiveKVPipeline

THETA = 10000.0


def make_kv(L, layers=2, heads=2, dim=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return tuple((torch.randn(1, heads, L, dim, generator=g), torch.randn(1, heads, L, dim, generator=g))
                 for _ in range(layers))


def length(kv):
    return kv[0][0].shape[2]


@pytest.fixture
def pool(tmp_path):
    p = DiskKVAuditPool(tmp_path / "pool")
    # child 0 donors of lengths 100, 104, 120, 200; child 1 donors of 50, 97
    for i, L in enumerate((100, 104, 120, 200)):
        p.add(0, f"d0_{i}", make_kv(L, seed=10 + i), question_text=f"donor question 0/{i}")
    for i, L in enumerate((50, 97)):
        p.add(1, f"d1_{i}", make_kv(L, seed=20 + i), question_text=f"donor question 1/{i}")
    p.mark_complete()
    return p


def test_donor_injected_unmodified_no_pad_or_truncation(pool):
    src = LengthMatchedSource(pool)
    real = make_kv(102, seed=1)                              # window [96.9, 107.1] -> donors 100, 104
    out, log = apply_causal_audit("length_matched", real, agent_idx=0, question_key="q",
                                  donor_source=src, rng=RecursiveKVPipeline.donor_rng(0, "q", 0))
    assert length(out) in (100, 104)
    assert log["pad_positions"] == 0 and log["truncated_positions"] == 0
    assert abs(log["donor_cache_len"] - 102) <= 0.05 * 102 and log["donor_question_key"].startswith("d0_")
    # byte-identical to the stored donor (no zero positions anywhere)
    donor_seed = {"d0_0": 10, "d0_1": 11}[log["donor_question_key"]]
    ref = make_kv(length(out), seed=donor_seed)
    for (k, v), (kr, vr) in zip(out, ref):
        assert torch.equal(k, kr) and torch.equal(v, vr)
    assert all(torch.count_nonzero(k[:, :, -1]) > 0 for k, _ in out)    # last position is real content


def test_pool_is_keyed_by_child_index(pool):
    src = LengthMatchedSource(pool)
    out, log = apply_causal_audit("length_matched", make_kv(97, seed=3), agent_idx=1, question_key="q",
                                  donor_source=src, rng=random.Random(0))
    assert log["donor_question_key"] == "d1_1" and length(out) == 97     # child-1 donors only
    with pytest.raises(NoMatchedDonor):                                   # 200 exists only for child 0
        apply_causal_audit("length_matched", make_kv(200, seed=3), agent_idx=1, question_key="q",
                           donor_source=src, rng=random.Random(0))


def test_skip_when_no_donor_within_5_percent(pool):
    with pytest.raises(NoMatchedDonor) as ei:
        apply_causal_audit("length_matched", make_kv(150), agent_idx=0, question_key="q",
                           donor_source=LengthMatchedSource(pool), rng=random.Random(0))
    assert ei.value.log["skipped"] is True and ei.value.log["nearest_donor_cache_len"] in (120, 200)
    assert "no donor within 5%" in ei.value.reason


def test_same_question_is_never_its_own_donor(pool):
    src = LengthMatchedSource(pool)
    with pytest.raises(NoMatchedDonor):
        apply_causal_audit("length_matched", make_kv(200), agent_idx=0, question_key="d0_3",
                           donor_source=src, rng=random.Random(0))


def test_donor_rng_is_crc32_seeded_and_order_independent():
    a = RecursiveKVPipeline.donor_rng(0, "qkey", 1)
    b = RecursiveKVPipeline.donor_rng(0, "qkey", 1)
    assert [a.random() for _ in range(5)] == [b.random() for _ in range(5)]
    RecursiveKVPipeline.donor_rng(0, "other", 0).random()                  # drawing for another question
    c = RecursiveKVPipeline.donor_rng(0, "qkey", 1)
    assert c.random() == RecursiveKVPipeline.donor_rng(0, "qkey", 1).random()
    assert RecursiveKVPipeline.donor_rng(0, "qkey", 0).random() != RecursiveKVPipeline.donor_rng(0, "qkey", 1).random()
    assert RecursiveKVPipeline.donor_rng(1, "qkey", 1).random() != RecursiveKVPipeline.donor_rng(0, "qkey", 1).random()


def test_donor_choice_deterministic_across_calls(pool):
    src = LengthMatchedSource(pool)
    picks = {apply_causal_audit("length_matched", make_kv(102), agent_idx=0, question_key="q", donor_source=src,
                                rng=RecursiveKVPipeline.donor_rng(0, "q", 0))[1]["donor_question_key"] for _ in range(4)}
    assert len(picks) == 1


# ---------------- merge_child_kv extra_shifts / shift_control ----------------

def test_merge_default_unchanged_and_zero_extra_shifts_equal():
    kvs = [make_kv(10, seed=1), make_kv(7, seed=2), make_kv(5, seed=3)]
    base = RecursiveKVPipeline.merge_child_kv(kvs, rope_theta=THETA)
    same = RecursiveKVPipeline.merge_child_kv(kvs, rope_theta=THETA, extra_shifts=[0, 0, 0])
    for (k, v), (k2, v2) in zip(base, same):
        assert torch.equal(k, k2) and torch.equal(v, v2)
    assert length(base) == 22
    # child 1 sits at +10, child 2 at +17, child 0 untouched
    k_ref = AnchorTable._rope_shift_k(kvs[1][0][0], shift=10, theta=THETA)
    assert torch.allclose(base[0][0][:, :, 10:17], k_ref)
    assert torch.equal(base[0][0][:, :, :10], kvs[0][0][0])


def test_extra_shifts_applied_to_every_child_including_first_and_values_untouched():
    kvs = [make_kv(10, seed=1), make_kv(7, seed=2)]
    out = RecursiveKVPipeline.merge_child_kv(kvs, rope_theta=THETA, extra_shifts=[-6, 4])
    assert torch.allclose(out[0][0][:, :, :10], AnchorTable._rope_shift_k(kvs[0][0][0], shift=-6, theta=THETA))
    assert torch.allclose(out[0][0][:, :, 10:], AnchorTable._rope_shift_k(kvs[1][0][0], shift=10 + 4, theta=THETA), atol=1e-5)
    assert torch.equal(out[0][1][:, :, :10], kvs[0][0][1]) and torch.equal(out[0][1][:, :, 10:], kvs[1][0][1])  # V never shifted
    assert length(out) == 17                                                                                    # shapes unchanged


def test_shift_control_extra_shifts_reproduce_donor_run_relative_geometry():
    """Real children, with a donor-run's query-key distances (formula used in run())."""
    L = [10, 7]
    delta = [3, -2]                                    # donor lengths 13 and 5
    order = [0, 1]
    ds = [delta[i] for i in order]
    total = sum(ds)
    extra = [sum(ds[:c]) - total for c in range(2)]
    assert extra == [-1, 2]
    # key positions of first token of each child, and aggregator position, in both runs
    donor_run_child_pos = [0, L[0] + delta[0]]
    donor_run_agg = sum(L[i] + delta[i] for i in range(2))
    ctrl_child_pos = [0 + extra[0], L[0] + extra[1]]
    ctrl_agg = sum(L)
    for c in range(2):
        assert donor_run_agg - donor_run_child_pos[c] == ctrl_agg - ctrl_child_pos[c]
    # and the donor-run relative offset between the two children
    assert donor_run_child_pos[1] - donor_run_child_pos[0] == ctrl_child_pos[1] - ctrl_child_pos[0]


def test_rope_shift_negative_roundtrip():
    k = make_kv(9, seed=7)[0][0]
    back = AnchorTable._rope_shift_k(AnchorTable._rope_shift_k(k, shift=13, theta=THETA), shift=-13, theta=THETA)
    assert torch.allclose(back, k, atol=1e-5)
