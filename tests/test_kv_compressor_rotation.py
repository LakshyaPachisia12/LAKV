"""
Standalone test for rotation-based (TurboQuant/PolarQuant-inspired) INT4
quantization in lakv/kv_compressor.py.

No model / GPU required — uses the same synthetic outlier-injected tensor
shape as tests/test_kv_compressor_clipping.py, so the two are directly
comparable: does rotation help with the exact failure mode outlier clipping
was already built to address?

Run directly for a before/after printout:
    python tests/test_kv_compressor_rotation.py

Or via pytest (from repo root):
    pytest tests/test_kv_compressor_rotation.py -v -s
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lakv.kv_compressor import (
    KVCompressor, _quantize, _dequantize, _rotate, _unrotate, _hadamard_matrix,
)


def _make_synthetic_kv(seed: int = 0, n_outliers: int = 6) -> torch.Tensor:
    """Same construction as test_kv_compressor_clipping.py's helper: (1, 4,
    512, 128) tensor, N(0,1) bulk + a few injected outliers ~30-60x std."""
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(1, 4, 512, 128, generator=g)

    idx_h = torch.randint(0, 4, (n_outliers,), generator=g)
    idx_s = torch.randint(0, 512, (n_outliers,), generator=g)
    idx_d = torch.randint(0, 128, (n_outliers,), generator=g)
    signs = torch.where(torch.rand(n_outliers, generator=g) > 0.5, 1.0, -1.0)
    magnitudes = 30.0 + torch.rand(n_outliers, generator=g) * 30.0

    for i in range(n_outliers):
        t[0, idx_h[i], idx_s[i], idx_d[i]] = signs[i] * magnitudes[i]
    return t.to(torch.bfloat16)


def test_hadamard_matrix_is_orthonormal():
    h = _hadamard_matrix(128)
    identity = h @ h.T
    assert torch.allclose(identity, torch.eye(128), atol=1e-5)


def test_hadamard_matrix_rejects_non_power_of_2():
    try:
        _hadamard_matrix(100)
        assert False, "expected ValueError for non-power-of-2 dimension"
    except ValueError:
        pass


def test_rotate_unrotate_is_near_lossless_on_its_own():
    # No quantization at all -- rotating then un-rotating should recover the
    # original almost exactly (only bf16 round-trip + float32 matmul error,
    # NOT quantization error).
    t = torch.randn(1, 4, 512, 128, dtype=torch.bfloat16)
    recon = _unrotate(_rotate(t))
    mse = torch.mean((t.float() - recon.float()) ** 2).item()
    assert mse < 1e-4, f"rotation round-trip should be near-lossless, got mse={mse}"


def test_rotation_preserves_shape_and_dtype():
    t = torch.randn(1, 4, 512, 128, dtype=torch.bfloat16)
    rotated = _rotate(t)
    assert rotated.shape == t.shape
    assert rotated.dtype == t.dtype


def test_rotated_int4_reduces_error_vs_plain_int4_on_outlier_tensor():
    """The actual claim being tested: does rotating before quantizing reduce
    error on a tensor with a few large outliers, same failure mode outlier
    clipping already targets?"""
    t = _make_synthetic_kv(seed=1)

    # plain int4, no clipping, no rotation
    q_plain, s_plain, zp_plain = _quantize(t, bits=4)
    recon_plain = _dequantize(q_plain, s_plain, zp_plain)
    mse_plain = torch.mean((t.float() - recon_plain.float()) ** 2).item()

    # rotated int4, no clipping
    t_rot = _rotate(t)
    q_rot, s_rot, zp_rot = _quantize(t_rot, bits=4)
    recon_rot = _unrotate(_dequantize(q_rot, s_rot, zp_rot))
    mse_rot = torch.mean((t.float() - recon_rot.float()) ** 2).item()

    print(f"  plain int4 mse={mse_plain:.6f}  |  rotated int4 mse={mse_rot:.6f}  "
          f"|  reduction={100*(1 - mse_rot/mse_plain):.1f}%")
    assert mse_rot < mse_plain, (
        f"expected rotation to reduce quantization error on an outlier-heavy tensor, "
        f"got plain={mse_plain} rotated={mse_rot}"
    )


def test_full_compressor_round_trip_uniform_int4_rotated():
    """End-to-end through KVCompressor.compress()/decompress(), matching how
    the real pipeline calls it (not the lower-level functions directly)."""
    t_k = _make_synthetic_kv(seed=2)
    t_v = _make_synthetic_kv(seed=3)
    kv = ((t_k, t_v),)

    comp = KVCompressor(mode="uniform_int4_rotated")
    msg = comp.compress(kv)
    recon = comp.decompress(msg, device="cpu")

    assert recon[0][0].shape == t_k.shape
    assert msg.compression_ratio > 3.5  # ~4x expected for int4, same as plain uniform_int4

    mse = torch.mean((t_k.float() - recon[0][0].float()) ** 2).item()
    assert mse < 5.0  # sanity ceiling, not a tight bound -- just catches a broken pipeline


def test_v_only_mode_leaves_k_bit_identical_to_plain_int4():
    """uniform_int4_rotated_v_only must leave K on the EXACT same code path
    as plain uniform_int4 (no rotation at all) -- only V gets rotated. This
    is the isolating diagnostic added after a real-model run of the full
    (K+V rotated) mode produced out-of-distribution tokens plausibly caused
    by the rotation interacting with K's already-applied RoPE encoding."""
    t_k = _make_synthetic_kv(seed=6)
    t_v = _make_synthetic_kv(seed=7)
    kv = ((t_k, t_v),)

    comp_plain = KVCompressor(mode="uniform_int4")
    comp_vonly = KVCompressor(mode="uniform_int4_rotated_v_only")

    msg_plain = comp_plain.compress(kv)
    msg_vonly = comp_vonly.compress(kv)

    # K's quantized bytes, scale, and zero-point must be BIT IDENTICAL
    # between the two modes -- if they're not, K is being touched by the
    # rotation somehow, which would defeat the whole point of this isolating
    # test before it's even run on GPU.
    k_plain, k_vonly = msg_plain.layers[0], msg_vonly.layers[0]
    assert torch.equal(k_plain.k_q, k_vonly.k_q)
    assert torch.equal(k_plain.k_scale, k_vonly.k_scale)
    assert torch.equal(k_plain.k_zp, k_vonly.k_zp)

    # V, by contrast, must differ (it went through rotation before quant).
    v_plain, v_vonly = msg_plain.layers[0], msg_vonly.layers[0]
    assert not torch.equal(v_plain.v_q, v_vonly.v_q)

    # Round trip still reconstructs the right shape and a sane error level.
    recon = comp_vonly.decompress(msg_vonly, device="cpu")
    assert recon[0][0].shape == t_k.shape
    assert recon[0][1].shape == t_v.shape
    # K reconstruction must match plain int4's reconstruction exactly too.
    recon_plain = comp_plain.decompress(msg_plain, device="cpu")
    assert torch.equal(recon[0][0], recon_plain[0][0])


def test_compressed_layer_shape_field_unaffected_by_rotation():
    # CompressedLayer.shape is used by decompress()'s own shape assertion --
    # rotation only touches the LAST axis in place, must not change it.
    t_k = _make_synthetic_kv(seed=4)
    t_v = _make_synthetic_kv(seed=5)
    comp = KVCompressor(mode="uniform_int4_rotated")
    msg = comp.compress(((t_k, t_v),))
    assert msg.layers[0].shape == tuple(t_k.shape)
    # decompress()'s internal assert would already catch a mismatch, but
    # calling it here makes the check explicit and fast to read.
    comp.decompress(msg, device="cpu")


def test_unknown_mode_still_rejected():
    try:
        KVCompressor(mode="not_a_real_mode")
        assert False, "expected ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
