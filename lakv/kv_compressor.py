"""
LAKV Module 3: KVCompressor

Compresses and decompresses KV cache tensors using per-head min-max
quantisation. Supports four modes: none, uniform_int8, uniform_int4, adaptive
(tier-based).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from lakv.calibration_profiler import LayerProfile


# ─── dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class CompressedLayer:
    k_q: torch.Tensor          # uint8 (or bfloat16 if mode='none')
    v_q: torch.Tensor
    k_scale: torch.Tensor      # shape (batch, heads)
    k_zp: torch.Tensor         # shape (batch, heads)
    v_scale: torch.Tensor      # shape (batch, heads)
    v_zp: torch.Tensor         # shape (batch, heads)
    shape: tuple               # original shape e.g. (1, 4, seq_len, 128)
    bits: int                  # 8, 4, or 16
    layer_idx: int             # position in original 28-layer tuple


@dataclass
class KVMessage:
    layers: List[CompressedLayer]
    mode: str
    original_bytes: int
    compressed_bytes: int

    @property
    def compression_ratio(self) -> float:
        return self.original_bytes / max(self.compressed_bytes, 1)

    def size_mb(self) -> float:
        return self.compressed_bytes / (1024 ** 2)

    def original_mb(self) -> float:
        return self.original_bytes / (1024 ** 2)


# ─── quantisation math ────────────────────────────────────────────────────────

def _quantize(tensor: torch.Tensor, bits: int, clip_percentile: Optional[float] = None):
    """Per-head min-max quantisation.
    tensor shape: (batch, heads, seq, dim)
    Returns: (quantized_uint8, scale_per_head, zero_point_per_head)

    clip_percentile: if set (e.g. 99.5), the per-head min/max used to derive
    scale/zero_point are computed from the [100-p, p] percentile range instead
    of the true min/max. This keeps a handful of outlier values from blowing
    out the quant step size for the whole head. The original (unclipped)
    tensor is still what gets quantized — values outside the clipped range
    just saturate to the nearest boundary bin via the existing clamp(0, qmax),
    same as standard outlier-clipped quantization.
    """
    t = tensor.detach().float()
    qmax = (1 << bits) - 1

    if clip_percentile is not None:
        b, h, s, d = t.shape
        flat = t.reshape(b, h, s * d)
        lo_q = (100.0 - clip_percentile) / 100.0
        hi_q = clip_percentile / 100.0
        t_min = torch.quantile(flat, lo_q, dim=-1, keepdim=True).unsqueeze(-1)
        t_max = torch.quantile(flat, hi_q, dim=-1, keepdim=True).unsqueeze(-1)
    else:
        # Reduce over seq and dim dimensions -> per-head min/max
        t_min = t.amin(dim=(-2, -1), keepdim=True)
        t_max = t.amax(dim=(-2, -1), keepdim=True)

    same = (t_max == t_min)
    scale = torch.where(same, torch.ones_like(t_max), (t_max - t_min) / qmax)
    zero_point = t_min

    q = ((t - zero_point) / scale).round().clamp(0, qmax).to(torch.uint8)
    return q, scale.squeeze(-1).squeeze(-1), zero_point.squeeze(-1).squeeze(-1)


def _dequantize(q: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    """Reconstruct bfloat16 from per-head quantisation."""
    s = scale.unsqueeze(-1).unsqueeze(-1).float()
    zp = zero_point.unsqueeze(-1).unsqueeze(-1).float()
    t = q.to(torch.float32) * s + zp
    return t.to(torch.bfloat16)


# ─── compressor ───────────────────────────────────────────────────────────────

class KVCompressor:
    def __init__(self, mode: str, profile: LayerProfile = None,
                 outlier_clipping: bool = False, clip_percentile: float = 99.5):
        """
        Args:
            mode: 'none' | 'uniform_int8' | 'uniform_int4' | 'adaptive'
            profile: required only when mode == 'adaptive'
            outlier_clipping: if True, INT4 quantization derives its per-head
                scale/zero_point from the [100-clip_percentile, clip_percentile]
                percentile range instead of the true min/max, so a few outlier
                values don't blow out the quant step for the whole head.
                INT8 (and 'none') are unaffected — this only ever applies to
                4-bit layers, on the theory that INT8's 256 levels already
                have enough headroom to absorb outliers without clipping.
            clip_percentile: the percentile to clip to when outlier_clipping
                is enabled. Default 99.5 (i.e. clip to the 0.5th/99.5th range).
        """
        if mode not in ("none", "uniform_int8", "uniform_int4", "adaptive"):
            raise ValueError(f"Unknown compression mode: {mode}")
        if mode == "adaptive" and profile is None:
            raise ValueError("'adaptive' mode requires a LayerProfile")
        self.mode = mode
        self.profile = profile
        self.outlier_clipping = outlier_clipping
        self.clip_percentile = clip_percentile

    def _get_bits_for_layer(self, layer_idx: int, tier_info: Optional[Dict[int, int]] = None) -> int:
        if self.mode == 'none':
            return 16
        if self.mode == 'uniform_int8':
            return 8
        if self.mode == 'uniform_int4':
            return 4
        
        # adaptive: Tier 1 → INT8 (high-importance layers, best fidelity)
        #           Tier 2 → INT4 (medium-importance, save bandwidth)
        #           Tier 3 → INT8 (fallback; these layers should be dropped
        #                          by LayerSelector before reaching compressor,
        #                          but guard here just in case)
        if tier_info and layer_idx in tier_info:
            tier = tier_info[layer_idx]
        elif self.profile:
            tier = self.profile.tier_assignment[layer_idx]
        else:
            tier = 2  # fallback

        if tier == 1:
            return 8   # INT8 — preserve high-importance layers
        elif tier == 2:
            return 4   # INT4 — compress medium-importance layers
        else:
            return 8   # INT8 fallback for any stray Tier-3 layers

    def compress(
        self,
        past_key_values: Tuple,
        tier_info: Optional[Dict[int, int]] = None,
        layer_indices: Optional[List[int]] = None,
    ) -> KVMessage:
        """Quantise a full KV cache tuple."""
        compressed_layers = []
        original_bytes = 0
        compressed_bytes = 0

        if layer_indices is not None and len(layer_indices) != len(past_key_values):
            raise ValueError(
                "layer_indices length must match past_key_values length: "
                f"{len(layer_indices)} != {len(past_key_values)}"
            )

        for local_idx, (k, v) in enumerate(past_key_values):
            layer_idx = layer_indices[local_idx] if layer_indices is not None else local_idx
            original_bytes += k.nbytes + v.nbytes
            bits = self._get_bits_for_layer(layer_idx, tier_info)

            if bits == 16:
                b, h = k.shape[0], k.shape[1]
                cl = CompressedLayer(
                    k_q=k.cpu(),
                    v_q=v.cpu(),
                    k_scale=torch.ones(b, h),
                    k_zp=torch.zeros(b, h),
                    v_scale=torch.ones(b, h),
                    v_zp=torch.zeros(b, h),
                    shape=tuple(k.shape),
                    bits=16,
                    layer_idx=layer_idx
                )
                compressed_bytes += k.nbytes + v.nbytes
            else:
                clip_pct = self.clip_percentile if (self.outlier_clipping and bits == 4) else None
                k_q, k_scale, k_zp = _quantize(k, bits, clip_percentile=clip_pct)
                v_q, v_scale, v_zp = _quantize(v, bits, clip_percentile=clip_pct)

                cl = CompressedLayer(
                    k_q=k_q.cpu(),
                    v_q=v_q.cpu(),
                    k_scale=k_scale.cpu(),
                    k_zp=k_zp.cpu(),
                    v_scale=v_scale.cpu(),
                    v_zp=v_zp.cpu(),
                    shape=tuple(k.shape),
                    bits=bits,
                    layer_idx=layer_idx
                )
                bytes_per_el = 1 if bits == 8 else 0.5
                n_el = k.numel() + v.numel()
                compressed_bytes += int(n_el * bytes_per_el)
                # Overhead for scale/zp per head
                compressed_bytes += (k_scale.numel() + v_scale.numel()) * 4 * 2

            compressed_layers.append(cl)

        return KVMessage(
            layers=compressed_layers,
            mode=self.mode,
            original_bytes=original_bytes,
            compressed_bytes=compressed_bytes
        )

    def decompress(self, message: KVMessage, device: str = 'cuda') -> Tuple:
        """Reconstruct bfloat16 past_key_values from a KVMessage."""
        result = []

        for cl in message.layers:
            if cl.bits == 16:
                k = cl.k_q.to(device)
                v = cl.v_q.to(device)
            else:
                k = _dequantize(
                    cl.k_q.to(device),
                    cl.k_scale.to(device),
                    cl.k_zp.to(device)
                )
                v = _dequantize(
                    cl.v_q.to(device),
                    cl.v_scale.to(device),
                    cl.v_zp.to(device)
                )
            
            assert k.shape == torch.Size(cl.shape), f"Shape mismatch: {k.shape} vs {cl.shape}"
            result.append((k, v))

        return tuple(result)

    @staticmethod
    def run_sanity_check(device='cuda'):
        """Verifies compress/decompress math works."""
        print("Running KVCompressor sanity check...")
        dummy_k = torch.randn(1, 4, 512, 128, dtype=torch.bfloat16).to(device)
        dummy_v = torch.randn(1, 4, 512, 128, dtype=torch.bfloat16).to(device)
        dummy_kv = ((dummy_k, dummy_v),)

        for mode in ['none', 'uniform_int8', 'uniform_int4']:
            comp = KVCompressor(mode=mode)
            msg = comp.compress(dummy_kv)
            recon = comp.decompress(msg, device=device)

            k_orig = dummy_k.float().flatten()
            k_recon = recon[0][0].float().flatten()

            cos_sim = F.cosine_similarity(k_orig.unsqueeze(0), k_recon.unsqueeze(0)).item()
            mse = torch.mean((dummy_k.float() - recon[0][0].float())**2).item()

            print(f"  mode={mode:12s} | original={msg.original_mb():.2f}MB | "
                  f"compressed={msg.size_mb():.2f}MB | ratio={msg.compression_ratio:.2f}x | "
                  f"cosine_sim={cos_sim:.6f} | mse={mse:.8f}")

            if mode == 'uniform_int8' and cos_sim < 0.999:
                print("  WARNING: cosine_sim below 0.999 for INT8 — bug!")
        
        print("Sanity check complete.")
