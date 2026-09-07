"""
Standalone test for KIVI-inspired per-channel K quantization in
lakv/kv_compressor.py (uniform_int4_kivi_k_channel).

Added after a real-model test of uniform_int4_rotated (Hadamard rotation of
both K and V) produced out-of-distribution tokens, diagnosed as likely
caused by the rotation interacting with K's already-applied RoPE encoding.
KIVI (Liu et al., ICML'24) sidesteps rotation entirely for K: quantize
per-channel (reduce over sequence only) instead of per-head (reduce over
sequence AND channel jointly), motivated by RoPE keeping magnitude more
consistent within a channel across positions than across channels.

No model / GPU required — synthetic tensors, including one shaped to make
the per-channel-vs-per-head distinction concrete (an outlier confined to a
single channel across many positions, which per-head quantization smears
across the whole head's range and per-channel quantization should isolate).

Run directly for a printout:
    python tests/test_kv_compressor_kivi.py

Or via pytest (from repo root):
    pytest tests/test_kv_compressor_kivi.py -v -s
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lakv.kv_compressor import (
    KVCompressor, _quantize, _quantize_per_channel, _dequantize_per_channel,
)


def _make_channel_outlier_kv(seed: int = 0) -> torch.Tensor:
    """(1, 4, 512, 128) tensor: N(0, 1) bulk, but ONE channel (dim index 7)
    has a much larger scale (std=20) across EVERY position, in EVERY head --
    the kind of "consistent large-magnitude channel" RoPE is described as
    disrupting when K is cached post-rotation. Per-head quantization has to
    stretch its single range to cover that channel's magnitude everywhere,
    crushing the other 127 channels. Per-channel quantization gives that one
    channel its own range and leaves the other 127 untouched.
    """
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(1, 4, 512, 128, generator=g)
    t[:, :, :, 7] *= 20.0
    return t.to(torch.bfloat16)


def test_per_channel_shape_is_batch_heads_dim():
    t = _make_channel_outlier_kv(seed=1)
    q, scale, zp = _quantize_per_channel(t, bits=4)
    assert scale.shape == (1, 4, 128)
    assert zp.shape == (1, 4, 128)
    assert q.shape == t.shape


def test_per_channel_round_trip_shape_and_dtype():
    t = _make_channel_outlier_kv(seed=2)
    q, scale, zp = _quantize_per_channel(t, bits=4)
    recon = _dequantize_per_channel(q, scale, zp)
    assert recon.shape == t.shape
    assert recon.dtype == torch.bfloat16


def test_per_channel_isolates_single_channel_outlier_better_than_per_head():
    """The actual claim: does giving each channel its own quant range
    protect the OTHER 127 channels from one channel's large magnitude,
    compared to per-head quantization which shares one range across all of
    them?"""
    t = _make_channel_outlier_kv(seed=3)

    # per-head (existing _quantize): one range for the whole head, channel 7
    # included -- should badly crush the other channels' precision.
    q_head, s_head, zp_head = _quantize(t, bits=4)
    from lakv.kv_compressor import _dequantize
    recon_head = _dequantize(q_head, s_head, zp_head)

    # per-channel: channel 7 gets its own range, others are untouched by it.
    q_chan, s_chan, zp_chan = _quantize_per_channel(t, bits=4)
    recon_chan = _dequantize_per_channel(q_chan, s_chan, zp_chan)

    # Compare error on the NON-outlier channels only (everything except dim 7).
    mask = torch.ones(128, dtype=torch.bool)
    mask[7] = False
    err_head = torch.mean((t[..., mask].float() - recon_head[..., mask].float()) ** 2).item()
    err_chan = torch.mean((t[..., mask].float() - recon_chan[..., mask].float()) ** 2).item()

    print(f"  non-outlier-channel mse: per-head={err_head:.6f}  per-channel={err_chan:.6f}  "
          f"reduction={100*(1 - err_chan/err_head):.1f}%")
    assert err_chan < err_head, (
        f"expected per-channel quantization to protect non-outlier channels better, "
        f"got per-head={err_head} per-channel={err_chan}"
    )


def test_full_compressor_round_trip_kivi_mode():
    """End-to-end through KVCompressor.compress()/decompress()."""
    t_k = _make_channel_outlier_kv(seed=4)
    t_v = _make_channel_outlier_kv(seed=5)
    kv = ((t_k, t_v),)

    comp = KVCompressor(mode="uniform_int4_kivi_k_channel")
    msg = comp.compress(kv)
    recon = comp.decompress(msg, device="cpu")

    assert recon[0][0].shape == t_k.shape
    assert recon[0][1].shape == t_v.shape
    assert msg.compression_ratio > 3.5  # ~4x, matching plain uniform_int4


def test_kivi_mode_leaves_v_bit_identical_to_plain_int4():
    """V is untouched by this mode -- must match plain uniform_int4 exactly,
    same discipline as the rotation V-only isolation test."""
    t_k = _make_channel_outlier_kv(seed=6)
    t_v = _make_channel_outlier_kv(seed=7)
    kv = ((t_k, t_v),)

    comp_plain = KVCompressor(mode="uniform_int4")
    comp_kivi = KVCompressor(mode="uniform_int4_kivi_k_channel")

    msg_plain = comp_plain.compress(kv)
    msg_kivi = comp_kivi.compress(kv)

    v_plain, v_kivi = msg_plain.layers[0], msg_kivi.layers[0]
    assert torch.equal(v_plain.v_q, v_kivi.v_q)
    assert torch.equal(v_plain.v_scale, v_kivi.v_scale)
    assert torch.equal(v_plain.v_zp, v_kivi.v_zp)

    # K, by contrast, must differ (different quantization axis entirely).
    k_plain, k_kivi = msg_plain.layers[0], msg_kivi.layers[0]
    assert k_plain.k_scale.shape != k_kivi.k_scale.shape


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
