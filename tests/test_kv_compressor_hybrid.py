"""
Standalone test for uniform_int4_hybrid: K quantized per-channel (KIVI-
inspired, RoPE-safe) + V rotated before quantizing (Hadamard, proven to
reduce outlier error, safe since V has no RoPE applied).

This mode is a recombination of two independently-validated pieces
(tests/test_kv_compressor_kivi.py and tests/test_kv_compressor_rotation.py),
not new untested math — these tests exist to confirm the recombination
itself is wired correctly: K's path must be bit-identical to
uniform_int4_kivi_k_channel's, V's path must be bit-identical to
uniform_int4_rotated_v_only's.

No model / GPU required.

Run directly for a printout:
    python tests/test_kv_compressor_hybrid.py

Or via pytest (from repo root):
    pytest tests/test_kv_compressor_hybrid.py -v -s
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lakv.kv_compressor import KVCompressor


def _make_synthetic_kv(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 4, 512, 128, generator=g).to(torch.bfloat16)


def test_hybrid_k_path_matches_kivi_exactly():
    t_k = _make_synthetic_kv(seed=1)
    t_v = _make_synthetic_kv(seed=2)
    kv = ((t_k, t_v),)

    msg_kivi = KVCompressor(mode="uniform_int4_kivi_k_channel").compress(kv)
    msg_hybrid = KVCompressor(mode="uniform_int4_hybrid").compress(kv)

    k_kivi, k_hybrid = msg_kivi.layers[0], msg_hybrid.layers[0]
    assert torch.equal(k_kivi.k_q, k_hybrid.k_q)
    assert torch.equal(k_kivi.k_scale, k_hybrid.k_scale)
    assert torch.equal(k_kivi.k_zp, k_hybrid.k_zp)


def test_hybrid_v_path_matches_rotated_v_only_exactly():
    t_k = _make_synthetic_kv(seed=3)
    t_v = _make_synthetic_kv(seed=4)
    kv = ((t_k, t_v),)

    msg_vonly = KVCompressor(mode="uniform_int4_rotated_v_only").compress(kv)
    msg_hybrid = KVCompressor(mode="uniform_int4_hybrid").compress(kv)

    v_vonly, v_hybrid = msg_vonly.layers[0], msg_hybrid.layers[0]
    assert torch.equal(v_vonly.v_q, v_hybrid.v_q)
    assert torch.equal(v_vonly.v_scale, v_hybrid.v_scale)
    assert torch.equal(v_vonly.v_zp, v_hybrid.v_zp)


def test_hybrid_full_round_trip():
    t_k = _make_synthetic_kv(seed=5)
    t_v = _make_synthetic_kv(seed=6)
    comp = KVCompressor(mode="uniform_int4_hybrid")
    msg = comp.compress(((t_k, t_v),))
    recon = comp.decompress(msg, device="cpu")

    assert recon[0][0].shape == t_k.shape
    assert recon[0][1].shape == t_v.shape
    assert msg.compression_ratio > 3.5


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
