"""
Standalone test for lakv/causal_audit.py (Phase 1 causal audit substitutions).

No model / GPU required — uses small synthetic KV tuples shaped like real
past_key_values: a tuple of (K, V) pairs, each (1, n_kv_heads, seq_len, head_dim).

Run directly for a printout:
    python tests/test_causal_audit.py

Or via pytest (from repo root):
    pytest tests/test_causal_audit.py -v -s
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lakv.causal_audit import (
    zero_kv, moment_matched_random_kv, KVAuditPool, apply_causal_audit,
)


def _make_kv(seq_len=10, n_layers=3, n_kv_heads=4, head_dim=8, seed=0, mean=0.0, std=1.0):
    g = torch.Generator().manual_seed(seed)
    kv = []
    for _ in range(n_layers):
        k = torch.randn(1, n_kv_heads, seq_len, head_dim, generator=g) * std + mean
        v = torch.randn(1, n_kv_heads, seq_len, head_dim, generator=g) * std + mean
        kv.append((k, v))
    return tuple(kv)


def test_zero_kv_preserves_shape_zeroes_content():
    real = _make_kv(seed=1)
    zeroed = zero_kv(real)
    assert len(zeroed) == len(real)
    for (k, v), (rk, rv) in zip(zeroed, real):
        assert k.shape == rk.shape and v.shape == rv.shape
        assert torch.all(k == 0) and torch.all(v == 0)


def test_moment_matched_random_differs_but_matches_statistics():
    real = _make_kv(seed=2, seq_len=200, mean=3.0, std=2.0)  # bigger n for stable stats
    gen = torch.Generator().manual_seed(0)
    rand_kv = moment_matched_random_kv(real, generator=gen)
    for (k, v), (rk, rv) in zip(rand_kv, real):
        assert k.shape == rk.shape
        assert not torch.allclose(k, rk)  # actually different content
        # statistics should be close (same distribution family, different draw)
        assert abs(k.float().mean().item() - rk.float().mean().item()) < 0.5
        assert abs(k.float().std().item() - rk.float().std().item()) < 0.5


def test_pool_add_and_sample_excludes_current_question():
    pool = KVAuditPool(max_size_per_agent=5)
    kv_a = _make_kv(seed=10)
    kv_b = _make_kv(seed=11)
    pool.add(agent_idx=1, question_key="qA", kv=kv_a)
    pool.add(agent_idx=1, question_key="qB", kv=kv_b)

    target_shape = kv_a[0][0].shape
    substitute, resized = pool.sample(agent_idx=1, exclude_question_key="qA",
                                       target_shape=target_shape, device="cpu")
    assert substitute is not None
    assert not resized
    # must be qB's content (the only eligible entry), not qA's own
    assert torch.allclose(substitute[0][0], kv_b[0][0])


def test_pool_deduplicates_same_question_key():
    pool = KVAuditPool()
    kv1 = _make_kv(seed=20)
    kv2 = _make_kv(seed=21)  # different content, same question_key
    pool.add(agent_idx=0, question_key="dup", kv=kv1)
    pool.add(agent_idx=0, question_key="dup", kv=kv2)
    assert pool.size(0) == 1  # second add with the same key is a no-op


def test_pool_evicts_oldest_when_over_capacity():
    pool = KVAuditPool(max_size_per_agent=2)
    for i in range(4):
        pool.add(agent_idx=0, question_key=f"q{i}", kv=_make_kv(seed=30 + i))
    assert pool.size(0) == 2
    # oldest (q0, q1) should have been evicted; q2/q3 remain
    remaining_keys = {e.question_key for e in pool._pools[0]}
    assert remaining_keys == {"q2", "q3"}


def test_pool_sample_resizes_mismatched_seq_len():
    pool = KVAuditPool()
    donor = _make_kv(seq_len=15, seed=40)
    pool.add(agent_idx=1, question_key="donor", kv=donor)

    target = _make_kv(seq_len=25, seed=41)  # different (longer) seq_len
    target_shape = target[0][0].shape
    substitute, resized = pool.sample(agent_idx=1, exclude_question_key="not_donor",
                                       target_shape=target_shape, device="cpu")
    assert substitute is not None
    assert resized
    assert substitute[0][0].shape == target_shape  # padded up to match
    # first 15 positions should be the real donor content, rest zero-padded
    assert torch.allclose(substitute[0][0][:, :, :15, :], donor[0][0])
    assert torch.all(substitute[0][0][:, :, 15:, :] == 0)


def test_pool_sample_truncates_when_donor_longer():
    pool = KVAuditPool()
    donor = _make_kv(seq_len=30, seed=50)
    pool.add(agent_idx=1, question_key="donor", kv=donor)

    target = _make_kv(seq_len=12, seed=51)
    target_shape = target[0][0].shape
    substitute, resized = pool.sample(agent_idx=1, exclude_question_key="not_donor",
                                       target_shape=target_shape, device="cpu")
    assert resized
    assert substitute[0][0].shape == target_shape
    assert torch.allclose(substitute[0][0], donor[0][0][:, :, :12, :])


def test_pool_sample_returns_none_when_no_eligible_entries():
    pool = KVAuditPool()
    substitute, resized = pool.sample(agent_idx=5, exclude_question_key="whatever",
                                       target_shape=torch.Size([1, 4, 10, 8]), device="cpu")
    assert substitute is None


def test_apply_causal_audit_none_mode_passthrough():
    real = _make_kv(seed=60)
    out, log = apply_causal_audit("none", real, agent_idx=1, question_key="q")
    assert out is real
    assert log == {"mode": "none"}


def test_apply_causal_audit_mismatched_raises_on_empty_pool():
    pool = KVAuditPool()
    real = _make_kv(seed=61)
    try:
        apply_causal_audit("mismatched", real, agent_idx=1, question_key="q", pool=pool)
        assert False, "expected RuntimeError for an unpopulated pool"
    except RuntimeError:
        pass


def test_apply_causal_audit_mismatched_requires_pool():
    real = _make_kv(seed=62)
    try:
        apply_causal_audit("mismatched", real, agent_idx=1, question_key="q", pool=None)
        assert False, "expected ValueError when no pool is given"
    except ValueError:
        pass


def test_apply_causal_audit_unknown_mode_raises():
    real = _make_kv(seed=63)
    try:
        apply_causal_audit("not_a_real_mode", real, agent_idx=1, question_key="q")
        assert False, "expected ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
