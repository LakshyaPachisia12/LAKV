"""
Standalone test for real nibble-packing in lakv/kv_compressor.py
(_pack_nibbles / _unpack_nibbles, and their wiring into compress()/
decompress() for bits==4).

Context: every 4-bit quantized tensor in this codebase used to be stored
as one full torch.uint8 per element -- no packing existed anywhere in this
file, so a 4-bit value occupied the same one byte an 8-bit value did. This
was first caught as a byte-ACCOUNTING bug (compress() claimed a packing
benefit that didn't physically exist -- see git history and
docs/naacl2027_paper_draft.md's Limitations for the "No bit-packing"
paragraph, and RESULTS_SUMMARY.md/CLAUDE.md for the corrected numbers that
resulted). This test file is for the follow-up: actually implementing
real nibble-packing, so bits==4 configs achieve genuine additional
compression instead of just reporting an honest ~2.00x (matching bits==8).

No model / GPU required -- synthetic tensors throughout.

Run directly for a printout:
    python tests/test_kv_compressor_nibble_packing.py

Or via pytest (from repo root):
    pytest tests/test_kv_compressor_nibble_packing.py -v -s
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lakv.kv_compressor import (
    KVCompressor, _pack_nibbles, _unpack_nibbles, _quantize,
)


def _make_kv(seed: int, seq_len: int = 64, n_kv_heads: int = 4, head_dim: int = 128) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, n_kv_heads, seq_len, head_dim, generator=g).to(torch.bfloat16)


def _make_pkv(seed: int, n_layers: int = 28, **kw) -> tuple:
    return tuple(
        (_make_kv(seed * 1000 + 2 * i, **kw), _make_kv(seed * 1000 + 2 * i + 1, **kw))
        for i in range(n_layers)
    )


# ─── pack/unpack round-trip, isolated from compress()/decompress() ───────────

def test_pack_halves_byte_count_for_even_length():
    q = torch.randint(0, 16, (256,), dtype=torch.uint8)
    packed = _pack_nibbles(q)
    assert packed.numel() == 128
    assert packed.dtype == torch.uint8


def test_pack_rounds_up_for_odd_length():
    q = torch.randint(0, 16, (255,), dtype=torch.uint8)
    packed = _pack_nibbles(q)
    assert packed.numel() == 128  # ceil(255/2)


def test_unpack_recovers_exact_values_even_length():
    q = torch.randint(0, 16, (4, 4, 64, 128), dtype=torch.uint8)
    packed = _pack_nibbles(q)
    recovered = _unpack_nibbles(packed, tuple(q.shape))
    assert torch.equal(recovered, q)


def test_unpack_recovers_exact_values_odd_length():
    # Odd total element count forces the padding path in _pack_nibbles.
    q = torch.randint(0, 16, (127,), dtype=torch.uint8)
    packed = _pack_nibbles(q)
    recovered = _unpack_nibbles(packed, (127,))
    assert torch.equal(recovered, q)


def test_pack_unpack_covers_full_nibble_range():
    # 0 and 15 (the qmax for 4-bit) are the values most likely to break a
    # sign/overflow bug in a hand-rolled bit-shift implementation.
    q = torch.tensor([0, 15, 0, 15, 15, 0], dtype=torch.uint8)
    packed = _pack_nibbles(q)
    recovered = _unpack_nibbles(packed, (6,))
    assert torch.equal(recovered, q)


def test_pack_does_not_mutate_input():
    q = torch.randint(0, 16, (100,), dtype=torch.uint8)
    original = q.clone()
    _pack_nibbles(q)
    assert torch.equal(q, original)


# ─── full compress()/decompress() round trip: packing must be invisible ──────

def test_uniform_int4_kivi_compress_decompress_matches_prepacking_values():
    """The whole point of packing is that it's a storage-format change, not
    a numeric one -- decompressed values must be bit-identical to what a
    no-packing implementation would produce; only compressed_bytes should
    differ. Proven directly: dequantize the SAME quantized tensor two ways
    -- (a) straight through _dequantize_per_channel, no packing involved,
    and (b) pack then immediately unpack, then dequantize -- and require
    identical output."""
    from lakv.kv_compressor import _quantize_per_channel, _dequantize_per_channel

    pkv = _make_pkv(seed=7, n_layers=1)
    k0, _ = pkv[0]

    k_q, k_scale, k_zp = _quantize_per_channel(k0, bits=4)
    direct = _dequantize_per_channel(k_q, k_scale, k_zp)

    packed = _pack_nibbles(k_q)
    unpacked = _unpack_nibbles(packed, tuple(k_q.shape))
    via_packing = _dequantize_per_channel(unpacked, k_scale, k_zp)

    assert torch.equal(direct, via_packing)

    # And the full compress()/decompress() path shouldn't crash or reshape
    # anything -- shape is the one thing decompress()'s own assertion
    # (line ~556 in kv_compressor.py) already checks, confirm it passes.
    comp = KVCompressor(mode="uniform_int4_kivi_k_channel")
    msg = comp.compress(pkv)
    recovered = comp.decompress(msg, device="cpu")
    k0_rec, v0_rec = recovered[0]
    assert k0_rec.shape == k0.shape


def test_bits4_packing_actually_halves_reported_bytes_vs_bits8():
    """The real point of this whole change: bits==4 should now cost
    meaningfully less than bits==8 for the same shape, not the same amount
    (which was the bug this test file exists to fix)."""
    pkv = _make_pkv(seed=11, n_layers=4)

    comp8 = KVCompressor(mode="uniform_int8")
    msg8 = comp8.compress(pkv)

    comp4 = KVCompressor(mode="uniform_int4")
    msg4 = comp4.compress(pkv)

    # bits==4 must now be smaller than bits==8, not equal to it (that was
    # exactly the bug: pre-packing, both landed at ~1 byte/element).
    assert msg4.compressed_bytes < msg8.compressed_bytes
    # And it should be close to half (allowing for scale/zero-point
    # overhead and any odd-element padding, both small relative to the
    # main tensor payload).
    ratio = msg8.compressed_bytes / msg4.compressed_bytes
    assert 1.8 < ratio < 2.1, f"expected ~2x, got {ratio:.3f}x"


def test_bits4_kivi_full_per_token_v_also_packs_correctly():
    """The one mode where K AND V both go through non-per-head quantize
    paths (per-channel K, per-token V) -- exercises _unpack_nibbles being
    called on both k_q and v_q with their own (matching) shape."""
    pkv = _make_pkv(seed=13, n_layers=2, seq_len=37)  # odd seq_len -> odd numel paths get exercised

    comp = KVCompressor(mode="uniform_int4_kivi_full")
    msg = comp.compress(pkv)
    recovered = comp.decompress(msg, device="cpu")

    k0, v0 = pkv[0]
    k0_rec, v0_rec = recovered[0]
    assert k0_rec.shape == k0.shape
    assert v0_rec.shape == v0.shape


def test_bits8_layers_are_not_packed():
    """bits==8 already fills a byte -- packing it would be lossy (only 4
    bits survive a nibble) or pointless. Confirm compress() leaves it
    alone: k_q keeps its original (batch, heads, seq, dim) shape and one
    full byte per element, not a flattened/halved packed tensor."""
    pkv = _make_pkv(seed=17, n_layers=1)
    comp = KVCompressor(mode="uniform_int8")
    msg = comp.compress(pkv)
    cl = msg.layers[0]
    original_numel = 1
    for d in cl.shape:
        original_numel *= d
    assert tuple(cl.k_q.shape) == cl.shape
    assert cl.k_q.numel() == original_numel
    assert cl.k_q.nbytes == original_numel  # 1 byte per element, unpacked


def test_bits16_layers_untouched_by_packing_logic():
    """mode='none' (bits==16) never goes through _pack_nibbles at all --
    confirm k_q is still the raw bf16 tensor, not reinterpreted as uint8."""
    pkv = _make_pkv(seed=19, n_layers=1)
    comp = KVCompressor(mode="none")
    msg = comp.compress(pkv)
    cl = msg.layers[0]
    assert cl.k_q.dtype == torch.bfloat16


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
