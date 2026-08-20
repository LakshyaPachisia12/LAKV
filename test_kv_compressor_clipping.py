"""
Standalone test for outlier-aware INT4 clipping in kv_compressor.py.

No model / GPU required — uses synthetic tensors (normal distribution + a
handful of injected outlier values, the kind of thing real KV activations
produce that blows out pure min-max quantization).

Run directly for a before/after printout:
    python test_kv_compressor_clipping.py

Or via pytest:
    pytest test_kv_compressor_clipping.py -v -s
"""

import torch

from kv_compressor import KVCompressor, _quantize, _dequantize


def _make_synthetic_kv(seed: int = 0, n_outliers: int = 6) -> torch.Tensor:
    """(1, 4, 512, 128) tensor: N(0, 1) bulk + a few injected outliers.

    Outliers are placed at random (head, seq, dim) positions with magnitude
    ~30-60x the bulk std — the same kind of rare, large activation spikes
    that real attention KV states show and that pure min-max quantization
    is defenseless against (one outlier stretches the whole head's [min,max]
    range, so every non-outlier value gets crushed into a few quant bins).
    """
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(1, 4, 512, 128, generator=g)

    idx_h = torch.randint(0, 4, (n_outliers,), generator=g)
    idx_s = torch.randint(0, 512, (n_outliers,), generator=g)
    idx_d = torch.randint(0, 128, (n_outliers,), generator=g)
    signs = torch.where(torch.rand(n_outliers, generator=g) > 0.5, 1.0, -1.0)
    magnitudes = 30.0 + torch.rand(n_outliers, generator=g) * 30.0  # 30-60 std

    for i in range(n_outliers):
        t[0, idx_h[i], idx_s[i], idx_d[i]] = signs[i] * magnitudes[i]

    return t.to(torch.bfloat16)


def _reconstruction_error(t: torch.Tensor, bits: int, clip_percentile):
    q, scale, zp = _quantize(t, bits, clip_percentile=clip_percentile)
    recon = _dequantize(q, scale, zp)

    orig = t.float()
    recon_f = recon.float()

    mse = torch.mean((orig - recon_f) ** 2).item()
    max_abs_err = torch.max((orig - recon_f).abs()).item()
    cos_sim = torch.nn.functional.cosine_similarity(
        orig.flatten().unsqueeze(0), recon_f.flatten().unsqueeze(0)
    ).item()
    return mse, max_abs_err, cos_sim


# ── pytest tests ─────────────────────────────────────────────────────────────

def test_int8_unaffected_by_outlier_clipping_flag():
    """INT8 must produce byte-identical output whether outlier_clipping is on
    or off — the compressor only ever applies clipping to 4-bit layers."""
    k = _make_synthetic_kv(seed=1)
    v = _make_synthetic_kv(seed=2)
    kv = ((k, v),)

    comp_off = KVCompressor(mode="uniform_int8", outlier_clipping=False)
    comp_on = KVCompressor(mode="uniform_int8", outlier_clipping=True, clip_percentile=99.5)

    msg_off = comp_off.compress(kv)
    msg_on = comp_on.compress(kv)

    layer_off, layer_on = msg_off.layers[0], msg_on.layers[0]
    assert torch.equal(layer_off.k_q, layer_on.k_q)
    assert torch.equal(layer_off.k_scale, layer_on.k_scale)
    assert torch.equal(layer_off.k_zp, layer_on.k_zp)


def test_int4_clipping_reduces_bulk_reconstruction_error():
    """With injected outliers, INT4 min-max quantization should have much
    worse reconstruction error on the bulk of values than percentile-clipped
    INT4 — clipping should cut MSE substantially."""
    t = _make_synthetic_kv(seed=3, n_outliers=6)

    mse_unclipped, _, cos_unclipped = _reconstruction_error(t, bits=4, clip_percentile=None)
    mse_clipped, _, cos_clipped = _reconstruction_error(t, bits=4, clip_percentile=99.5)

    assert mse_clipped < mse_unclipped
    assert cos_clipped > cos_unclipped


def test_int4_clipping_saturates_outliers_not_crashes():
    """Clipped quantization must still run cleanly on tensors containing
    outliers — outliers should saturate to boundary bins, not raise/NaN."""
    t = _make_synthetic_kv(seed=4, n_outliers=6)
    q, scale, zp = _quantize(t, bits=4, clip_percentile=99.5)
    recon = _dequantize(q, scale, zp)
    assert not torch.isnan(recon).any()
    assert not torch.isinf(recon).any()


# ── before/after report (also runnable as a plain script) ────────────────────

def _print_report():
    print("Outlier-aware INT4 clipping — before/after reconstruction error")
    print("=" * 72)
    print("Synthetic tensor: (1, 4, 512, 128) N(0,1) + 6 injected outliers (30-60 std)\n")

    for bits, label in ((8, "INT8"), (4, "INT4")):
        t = _make_synthetic_kv(seed=42, n_outliers=6)
        mse_off, maxerr_off, cos_off = _reconstruction_error(t, bits=bits, clip_percentile=None)
        mse_on, maxerr_on, cos_on = _reconstruction_error(t, bits=bits, clip_percentile=99.5)

        print(f"[{label}]")
        print(f"  no clipping     : mse={mse_off:.6f}  max_abs_err={maxerr_off:.4f}  cosine_sim={cos_off:.6f}")
        print(f"  99.5% clipping  : mse={mse_on:.6f}  max_abs_err={maxerr_on:.4f}  cosine_sim={cos_on:.6f}")
        if mse_off > 0:
            print(f"  MSE reduction   : {(1 - mse_on / mse_off) * 100:.1f}%")
        print()

    # Full compress/decompress path via KVCompressor, matching real usage
    print("Full KVCompressor.compress()/decompress() round trip, mode=uniform_int4")
    print("-" * 72)
    k = _make_synthetic_kv(seed=10, n_outliers=6)
    v = _make_synthetic_kv(seed=11, n_outliers=6)
    kv = ((k, v),)

    for outlier_clipping, label in ((False, "no clipping"), (True, "99.5% clipping")):
        comp = KVCompressor(mode="uniform_int4", outlier_clipping=outlier_clipping, clip_percentile=99.5)
        msg = comp.compress(kv)
        recon = comp.decompress(msg, device="cpu")
        k_orig = k.float().flatten()
        k_recon = recon[0][0].float().flatten()
        mse = torch.mean((k_orig - k_recon) ** 2).item()
        cos = torch.nn.functional.cosine_similarity(k_orig.unsqueeze(0), k_recon.unsqueeze(0)).item()
        print(f"  {label:16s}: mse={mse:.6f}  cosine_sim={cos:.6f}  ratio={msg.compression_ratio:.2f}x")


if __name__ == "__main__":
    _print_report()
