"""
Standalone test for the multi-source KV fan-in merge in
lakv/recursive_pipeline.py (RecursiveKVPipeline.merge_child_kv).

This is the mechanics validation step for the RLM+KV-relay prototype: before
spending a GPU run on it, check that the core algebraic assumption behind
merging independently-computed child KV caches actually holds.

The assumption: each child's KV cache was computed by its own independent
forward pass, so its K tensors are RoPE-rotated for LOCAL positions
0..len-1. To sit correctly in the aggregator's combined sequence, a child
placed after `cumulative_len` prior tokens needs its K tensors to represent
GLOBAL positions cumulative_len..cumulative_len+len-1 instead. merge_child_kv
does this by calling AnchorTable._rope_shift_k(k, shift=cumulative_len) on
each non-first child's K -- the same primitive already in production for
config E's offset correction, repurposed here.

The algebraic property this depends on: RoPE's rotate-half formula is a
per-frequency-pair 2D rotation, and 2D rotations compose additively in
angle (R(delta) . R(p) == R(p + delta)). So shifting an already-rotated key
(rotated for position p) by a further `delta` should be bit-for-bit
equivalent to rotating the ORIGINAL unrotated key directly by `p + delta`.
test_rope_shift_additivity checks this directly, independent of the merge
code -- it's what would silently break (producing keys that encode the
wrong position, the RoPE-position-drift failure mode this project has
already characterized in config E and the int4/RoPE finding) if
_rope_shift_k's formula didn't actually match HF's real rotate-half
convention.

No model / GPU required -- synthetic tensors only.

Run directly for a printout:
    python tests/test_recursive_kv_merge.py

Or via pytest (from repo root):
    pytest tests/test_recursive_kv_merge.py -v -s
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lakv.anchor_table import AnchorTable
from lakv.recursive_pipeline import RecursiveKVPipeline


THETA = 1_000_000.0


def _make_kv_tuple(seed: int, n_layers: int, batch: int, heads: int, seq: int, head_dim: int) -> tuple:
    g = torch.Generator().manual_seed(seed)
    layers = []
    for _ in range(n_layers):
        k = torch.randn(batch, heads, seq, head_dim, generator=g)
        v = torch.randn(batch, heads, seq, head_dim, generator=g)
        layers.append((k, v))
    return tuple(layers)


def test_rope_shift_additivity():
    """Shifting an already-position-p-rotated key by `delta` must equal
    rotating the ORIGINAL unrotated key directly by `p + delta`. This is
    the exact operation merge_child_kv relies on to re-base a child's
    cache into the aggregator's combined position range."""
    torch.manual_seed(0)
    batch, heads, seq, head_dim = 1, 4, 3, 16  # head_dim must be even
    k0 = torch.randn(batch, heads, seq, head_dim)  # "original", unrotated keys
    p, delta = 7, 5

    k_at_p = AnchorTable._rope_shift_k(k0, shift=p, theta=THETA)
    k_at_p_via_two_shifts = AnchorTable._rope_shift_k(k_at_p, shift=delta, theta=THETA)
    k_at_p_plus_delta_direct = AnchorTable._rope_shift_k(k0, shift=p + delta, theta=THETA)

    max_diff = (k_at_p_via_two_shifts - k_at_p_plus_delta_direct).abs().max().item()
    assert torch.allclose(k_at_p_via_two_shifts, k_at_p_plus_delta_direct, atol=1e-4), (
        f"RoPE shift is not additive as expected -- max diff {max_diff}. "
        "merge_child_kv's re-basing of non-first children would encode the "
        "wrong position if this doesn't hold."
    )
    print(f"[OK] rope_shift_additivity: max diff = {max_diff:.2e}")


def test_rope_shift_zero_is_noop():
    torch.manual_seed(1)
    k0 = torch.randn(1, 4, 3, 16)
    out = AnchorTable._rope_shift_k(k0, shift=0, theta=THETA)
    assert torch.equal(out, k0)
    print("[OK] rope_shift_zero_is_noop")


def test_merge_child_kv_shapes():
    n_layers, batch, heads, head_dim = 3, 1, 4, 16
    len_a, len_b, len_c = 5, 7, 2
    kv_a = _make_kv_tuple(10, n_layers, batch, heads, len_a, head_dim)
    kv_b = _make_kv_tuple(20, n_layers, batch, heads, len_b, head_dim)
    kv_c = _make_kv_tuple(30, n_layers, batch, heads, len_c, head_dim)

    merged = RecursiveKVPipeline.merge_child_kv([kv_a, kv_b, kv_c], rope_theta=THETA)

    assert len(merged) == n_layers
    expected_len = len_a + len_b + len_c
    for layer_idx, (k, v) in enumerate(merged):
        assert k.shape == (batch, heads, expected_len, head_dim), (layer_idx, k.shape)
        assert v.shape == (batch, heads, expected_len, head_dim), (layer_idx, v.shape)
    print(f"[OK] merge_child_kv_shapes: {n_layers} layers, merged seq_len={expected_len}")


def test_merge_child_kv_first_child_unshifted():
    """Child 0 sits at the start of the merged sequence, so it needs no
    RoPE shift -- its K/V should appear in the merged cache byte-for-byte
    unchanged."""
    n_layers, batch, heads, head_dim = 2, 1, 4, 16
    len_a, len_b = 5, 3
    kv_a = _make_kv_tuple(10, n_layers, batch, heads, len_a, head_dim)
    kv_b = _make_kv_tuple(20, n_layers, batch, heads, len_b, head_dim)

    merged = RecursiveKVPipeline.merge_child_kv([kv_a, kv_b], rope_theta=THETA)

    for layer_idx in range(n_layers):
        k_merged, v_merged = merged[layer_idx]
        k_a, v_a = kv_a[layer_idx]
        assert torch.equal(k_merged[:, :, :len_a, :], k_a)
        assert torch.equal(v_merged[:, :, :len_a, :], v_a)
    print("[OK] merge_child_kv_first_child_unshifted")


def test_merge_child_kv_v_never_shifted():
    """V carries no positional encoding, so every child's V must appear in
    the merged cache unchanged regardless of position -- only K should
    differ from the original for non-first children."""
    n_layers, batch, heads, head_dim = 2, 1, 4, 16
    len_a, len_b = 5, 3
    kv_a = _make_kv_tuple(10, n_layers, batch, heads, len_a, head_dim)
    kv_b = _make_kv_tuple(20, n_layers, batch, heads, len_b, head_dim)

    merged = RecursiveKVPipeline.merge_child_kv([kv_a, kv_b], rope_theta=THETA)

    for layer_idx in range(n_layers):
        k_merged, v_merged = merged[layer_idx]
        k_b, v_b = kv_b[layer_idx]
        assert torch.equal(v_merged[:, :, len_a:, :], v_b), "V must be unchanged"
        assert not torch.equal(k_merged[:, :, len_a:, :], k_b), (
            "K for a non-first child must differ from its original (unshifted) "
            "value -- if this fails, the shift silently became a no-op."
        )
    print("[OK] merge_child_kv_v_never_shifted")


def test_merge_child_kv_matches_direct_rotation_from_zero():
    """End-to-end check: if child B's KV had instead been an UNROTATED key
    (position 0), shifting it by cumulative_len should exactly match
    rotating that same original key directly to position cumulative_len --
    i.e. merge_child_kv's shift is equivalent to "as if child B's forward
    pass had used position_ids starting at cumulative_len from the start,"
    which is the actual correctness goal of the merge."""
    n_layers, batch, heads, head_dim = 1, 1, 2, 16
    len_a, len_b = 5, 4
    kv_a = _make_kv_tuple(10, n_layers, batch, heads, len_a, head_dim)

    torch.manual_seed(99)
    k0_b = torch.randn(batch, heads, len_b, head_dim)  # pretend: position-0 (unrotated) key
    v_b = torch.randn(batch, heads, len_b, head_dim)
    # Simulate child B's real cache as if it were computed with position_ids 0..len_b-1
    # (i.e. this K0 rotated at position 0 is itself -- shift=0 is a no-op, confirmed above)
    kv_b = ((k0_b, v_b),)

    merged = RecursiveKVPipeline.merge_child_kv([kv_a, kv_b], rope_theta=THETA)
    k_merged_b = merged[0][0][:, :, len_a:, :]

    k_expected = AnchorTable._rope_shift_k(k0_b, shift=len_a, theta=THETA)
    assert torch.allclose(k_merged_b, k_expected, atol=1e-5)
    print("[OK] merge_child_kv_matches_direct_rotation_from_zero")


def test_causal_audit_zeroed_child_then_merge_matches_manual_zero_merge():
    """The wiring this session adds to RecursiveKVPipeline.run(): apply
    apply_causal_audit to ONE child's KV before merge_child_kv, not
    instead of it -- mirrors the same pattern just added to
    lakv/rlm_repl_kv.py's splice point, applied here to the fan-in merge
    instead. Confirms substituting child index 1's KV via
    apply_causal_audit(mode="zeroed", ...) and then merging equals
    manually zeroing that child and merging directly, with the other
    child's real content passing through unchanged."""
    from lakv.causal_audit import apply_causal_audit

    n_layers, batch, heads, head_dim = 2, 1, 4, 16
    len_a, len_b = 9, 6
    kv_a = _make_kv_tuple(1, n_layers, batch, heads, len_a, head_dim)
    kv_b = _make_kv_tuple(2, n_layers, batch, heads, len_b, head_dim)

    audited_b, audit_log = apply_causal_audit(
        "zeroed", kv_b, agent_idx=1, question_key="fake_q",
    )
    assert audit_log["mode"] == "zeroed"
    merged_via_audit = RecursiveKVPipeline.merge_child_kv([kv_a, audited_b], rope_theta=THETA)

    manually_zeroed_b = tuple((torch.zeros_like(k), torch.zeros_like(v)) for k, v in kv_b)
    merged_manual = RecursiveKVPipeline.merge_child_kv([kv_a, manually_zeroed_b], rope_theta=THETA)

    for layer_idx in range(n_layers):
        assert torch.equal(merged_via_audit[layer_idx][0], merged_manual[layer_idx][0])
        assert torch.equal(merged_via_audit[layer_idx][1], merged_manual[layer_idx][1])

    # child A (untouched) must pass through completely unchanged
    for layer_idx in range(n_layers):
        k_a, v_a = kv_a[layer_idx]
        assert torch.equal(merged_via_audit[layer_idx][0][:, :, :len_a, :], k_a)
        assert torch.equal(merged_via_audit[layer_idx][1][:, :, :len_a, :], v_a)
    print("[OK] causal_audit_zeroed_child_then_merge_matches_manual_zero_merge")


def test_causal_audit_mode_none_is_a_pure_passthrough():
    """mode="none" must reproduce the exact pre-audit merge -- confirms
    adding the audit hook did not change anything for the real,
    non-audited case."""
    from lakv.causal_audit import apply_causal_audit

    n_layers, batch, heads, head_dim = 2, 1, 4, 16
    kv_b = _make_kv_tuple(2, n_layers, batch, heads, 6, head_dim)
    audited_b, audit_log = apply_causal_audit("none", kv_b, agent_idx=1, question_key="fake_q")
    assert audit_log["mode"] == "none"
    for (ak, av), (bk, bv) in zip(audited_b, kv_b):
        assert torch.equal(ak, bk) and torch.equal(av, bv)
    print("[OK] causal_audit_mode_none_is_a_pure_passthrough (recursive_pipeline)")


if __name__ == "__main__":
    test_rope_shift_additivity()
    test_rope_shift_zero_is_noop()
    test_merge_child_kv_shapes()
    test_merge_child_kv_first_child_unshifted()
    test_merge_child_kv_v_never_shifted()
    test_merge_child_kv_matches_direct_rotation_from_zero()
    test_causal_audit_zeroed_child_then_merge_matches_manual_zero_merge()
    test_causal_audit_mode_none_is_a_pure_passthrough()
    print("\nAll recursive KV merge tests passed.")
