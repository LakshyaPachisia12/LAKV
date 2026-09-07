"""
Standalone test for KIVI's other asymmetric half: per-token V quantization
(uniform_int4_kivi_full = K per-channel + V per-token).

Mirror of tests/test_kv_compressor_kivi.py, which validated K's per-channel
axis. This validates the V-side axis: does giving each TOKEN POSITION its
own quantization range protect other positions from one position's
consistently large magnitude, the way per-channel protected other channels
from one channel's magnitude?

No model / GPU required.

Run directly for a printout:
    python tests/test_kv_compressor_kivi_full.py

Or via pytest (from repo root):
    pytest tests/test_kv_compressor_kivi_full.py -v -s
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lakv.kv_compressor import (
    KVCompressor, _quantize, _dequantize, _quantize_per_token, _dequantize_per_token,
)


def _make_token_outlier_kv(seed: int = 0) -> torch.Tensor:
    """(1, 4, 512, 128) tensor: N(0, 1) bulk, but ONE token position (seq
    index 100) has a much larger scale (std=20) across every channel, in
    every head -- the mirror of the channel-outlier test, now the "loud
    token" case per-token quantization is meant to isolate.
    """
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(1, 4, 512, 128, generator=g)
    t[:, :, 100, :] *= 20.0
    return t.to(torch.bfloat16)


def test_per_token_shape_is_batch_heads_seq():
    t = _make_token_outlier_kv(seed=1)
    q, scale, zp = _quantize_per_token(t, bits=4)
    assert scale.shape == (1, 4, 512)
    assert zp.shape == (1, 4, 512)
    assert q.shape == t.shape


def test_per_token_round_trip_shape_and_dtype():
    t = _make_token_outlier_kv(seed=2)
    q, scale, zp = _quantize_per_token(t, bits=4)
    recon = _dequantize_per_token(q, scale, zp)
    assert recon.shape == t.shape
    assert recon.dtype == torch.bfloat16


def test_per_token_isolates_single_token_outlier_better_than_per_head():
    """The actual claim: does giving each TOKEN its own quant range protect
    the OTHER 511 positions from one position's large magnitude, compared to
    per-head quantization which shares one range across all of them?"""
    t = _make_token_outlier_kv(seed=3)

    q_head, s_head, zp_head = _quantize(t, bits=4)
    recon_head = _dequantize(q_head, s_head, zp_head)

    q_tok, s_tok, zp_tok = _quantize_per_token(t, bits=4)
    recon_tok = _dequantize_per_token(q_tok, s_tok, zp_tok)

    mask = torch.ones(512, dtype=torch.bool)
    mask[100] = False
    err_head = torch.mean((t[:, :, mask, :].float() - recon_head[:, :, mask, :].float()) ** 2).item()
    err_tok = torch.mean((t[:, :, mask, :].float() - recon_tok[:, :, mask, :].float()) ** 2).item()

    print(f"  non-outlier-token mse: per-head={err_head:.6f}  per-token={err_tok:.6f}  "
          f"reduction={100*(1 - err_tok/err_head):.1f}%")
    assert err_tok < err_head, (
        f"expected per-token quantization to protect non-outlier positions better, "
        f"got per-head={err_head} per-token={err_tok}"
    )


def test_full_compressor_round_trip_kivi_full_mode():
    t_k = _make_token_outlier_kv(seed=4)
    t_v = _make_token_outlier_kv(seed=5)
    comp = KVCompressor(mode="uniform_int4_kivi_full")
    msg = comp.compress(((t_k, t_v),))
    recon = comp.decompress(msg, device="cpu")

    assert recon[0][0].shape == t_k.shape
    assert recon[0][1].shape == t_v.shape
    assert msg.compression_ratio > 3.5


def test_kivi_full_k_path_matches_kivi_k_channel_exactly():
    """K's treatment in uniform_int4_kivi_full must be bit-identical to
    uniform_int4_kivi_k_channel's -- only V's axis is new here."""
    t_k = _make_token_outlier_kv(seed=6)
    t_v = _make_token_outlier_kv(seed=7)
    kv = ((t_k, t_v),)

    msg_k_only = KVCompressor(mode="uniform_int4_kivi_k_channel").compress(kv)
    msg_full = KVCompressor(mode="uniform_int4_kivi_full").compress(kv)

    k_only, k_full = msg_k_only.layers[0], msg_full.layers[0]
    assert torch.equal(k_only.k_q, k_full.k_q)
    assert torch.equal(k_only.k_scale, k_full.k_scale)
    assert torch.equal(k_only.k_zp, k_full.k_zp)

    # V must differ -- different quantization axis entirely.
    v_only, v_full = msg_k_only.layers[0], msg_full.layers[0]
    assert v_only.v_scale.shape != v_full.v_scale.shape


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
