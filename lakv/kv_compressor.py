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


def _pack_nibbles(q: torch.Tensor) -> torch.Tensor:
    """Pack a uint8 tensor whose values fit in 4 bits (0-15) two-per-byte.

    This is the real nibble-packing that was previously missing entirely
    from this file: every 4-bit quantized tensor was stored as one full
    uint8 per element (see the compress() byte-accounting fix elsewhere in
    this file's history), which is why 4-bit and 8-bit configs used to
    report the same physical storage. Packing here actually halves it.

    Returns a 1D uint8 tensor of length ceil(numel/2). Odd-length inputs
    get one wasted nibble in the final byte (high nibble used, low nibble
    zero-padded) rather than crashing.
    """
    flat = q.reshape(-1).to(torch.uint8)
    n = flat.numel()
    if n % 2 == 1:
        pad = torch.zeros(1, dtype=torch.uint8, device=flat.device)
        flat = torch.cat([flat, pad])
    pairs = flat.view(-1, 2)
    return ((pairs[:, 0] << 4) | pairs[:, 1]).to(torch.uint8)


def _unpack_nibbles(packed: torch.Tensor, shape: tuple) -> torch.Tensor:
    """Inverse of _pack_nibbles. shape is the original (pre-flatten,
    pre-packing) tensor shape; its element count determines how much of
    the final (possibly padded) byte to keep."""
    numel = 1
    for d in shape:
        numel *= d
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F
    interleaved = torch.stack([high, low], dim=1).reshape(-1)[:numel]
    return interleaved.to(torch.uint8).reshape(shape)


def _quantize_per_channel(tensor: torch.Tensor, bits: int, clip_percentile: Optional[float] = None):
    """Per-(head, channel) min-max quantization — one scale/zero-point per
    (batch, head, head_dim) triple, reducing over the SEQUENCE axis only
    (not head_dim), unlike _quantize() above which lumps seq+dim together
    into one range per head.

    KIVI-inspired (Liu et al., ICML'24, arXiv 2402.02750) — motivated by K
    specifically: RoPE (already applied to K before it's cached) mixes pairs
    of channels by a position-dependent amount, which makes per-head
    min-max quantization (one range for every channel AND every position)
    a much worse fit for K than for V. Per-channel quantization keeps each
    channel's own scale, only reducing over position — this is intended for
    K specifically, NOT a general replacement for V's existing per-head
    quantization (V has no RoPE applied, so it isn't implicated in the
    failure mode this addresses).

    Simplified relative to the original KIVI paper: no group-wise windowing
    over sequence chunks, no fp16 residual buffer for the most recent
    tokens. Verify against the primary source before treating this as a
    faithful reproduction.

    tensor shape: (batch, heads, seq, dim)
    Returns: (quantized_uint8, scale[b,h,d], zero_point[b,h,d])
    """
    t = tensor.detach().float()
    qmax = (1 << bits) - 1

    if clip_percentile is not None:
        lo_q = (100.0 - clip_percentile) / 100.0
        hi_q = clip_percentile / 100.0
        t_min = torch.quantile(t, lo_q, dim=2, keepdim=True)  # reduce over seq only
        t_max = torch.quantile(t, hi_q, dim=2, keepdim=True)
    else:
        t_min = t.amin(dim=2, keepdim=True)  # (b, h, 1, d)
        t_max = t.amax(dim=2, keepdim=True)

    same = (t_max == t_min)
    scale = torch.where(same, torch.ones_like(t_max), (t_max - t_min) / qmax)
    zero_point = t_min

    q = ((t - zero_point) / scale).round().clamp(0, qmax).to(torch.uint8)
    return q, scale.squeeze(2), zero_point.squeeze(2)  # (b, h, d)


def _dequantize_per_channel(q: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    """Inverse of _quantize_per_channel. scale/zero_point shape (b, h, d)."""
    s = scale.unsqueeze(2).float()        # (b, h, 1, d)
    zp = zero_point.unsqueeze(2).float()  # (b, h, 1, d)
    t = q.to(torch.float32) * s + zp
    return t.to(torch.bfloat16)


def _quantize_per_token(tensor: torch.Tensor, bits: int, clip_percentile: Optional[float] = None):
    """Per-(head, token/position) min-max quantization — one scale/zero-point
    per (batch, head, seq) triple, reducing over head_dim only (the mirror
    image of _quantize_per_channel, which reduces over seq and keeps
    head_dim separate).

    KIVI-inspired (Liu et al., ICML'24) — this is the OTHER half of KIVI's
    asymmetric design, intended for V specifically. V has no RoPE applied
    (unlike K, so it isn't implicated in the RoPE-interaction failure
    _quantize_per_channel addresses), but the paper's motivation for
    per-token V grouping is a different, real structural property: some
    token positions contribute consistently larger-magnitude value vectors
    than others (some tokens' content simply matters more), which per-head
    quantization forces every position to share one range for. Per-token
    quantization gives each position its own range instead — the same fix
    idea as per-channel, applied to the axis where V's real variation lives
    rather than the axis where K's does.

    Simplified relative to the original KIVI paper: no group-wise windowing
    over sequence chunks, no fp16 residual buffer for the most recent
    tokens. Verify against the primary source before treating this as a
    faithful reproduction.

    tensor shape: (batch, heads, seq, dim)
    Returns: (quantized_uint8, scale[b,h,s], zero_point[b,h,s])
    """
    t = tensor.detach().float()
    qmax = (1 << bits) - 1

    if clip_percentile is not None:
        lo_q = (100.0 - clip_percentile) / 100.0
        hi_q = clip_percentile / 100.0
        t_min = torch.quantile(t, lo_q, dim=-1, keepdim=True)  # reduce over dim only
        t_max = torch.quantile(t, hi_q, dim=-1, keepdim=True)
    else:
        t_min = t.amin(dim=-1, keepdim=True)  # (b, h, s, 1)
        t_max = t.amax(dim=-1, keepdim=True)

    same = (t_max == t_min)
    scale = torch.where(same, torch.ones_like(t_max), (t_max - t_min) / qmax)
    zero_point = t_min

    q = ((t - zero_point) / scale).round().clamp(0, qmax).to(torch.uint8)
    return q, scale.squeeze(-1), zero_point.squeeze(-1)  # (b, h, s)


def _dequantize_per_token(q: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    """Inverse of _quantize_per_token. scale/zero_point shape (b, h, s)."""
    s = scale.unsqueeze(-1).float()        # (b, h, s, 1)
    zp = zero_point.unsqueeze(-1).float()  # (b, h, s, 1)
    t = q.to(torch.float32) * s + zp
    return t.to(torch.bfloat16)


# ─── rotation (TurboQuant/PolarQuant-inspired int4 quantization) ──────────────
#
# Phase 3 of the research-extensions plan. Rotating a tensor by a fixed
# orthonormal Walsh-Hadamard matrix redistributes each head's per-dimension
# magnitude evenly across all dimensions BEFORE quantizing — a few large
# outlier values that would otherwise blow out one dimension's quant range
# get spread thin across every dimension instead, so per-head min-max
# quantization has much less to lose. The rotation itself is lossless (it's
# orthogonal — H @ H.T == I, so un-rotating after dequantization exactly
# inverts it up to floating-point error); only the quantization step in
# between loses information, same as the unrotated path.
#
# NOTE ON FIDELITY: this is our own implementation of the core "rotate before
# quantizing" idea behind TurboQuant/PolarQuant (Zandieh et al., ICLR'26) —
# specifically the PolarQuant half (Walsh-Hadamard rotation + scalar
# quantization). It does NOT implement TurboQuant's QJL 1-bit bias-correction
# step; that was deliberately left out because its exact procedure wasn't
# independently verified against the paper before this was written (see
# lakv_research_extensions_prompt.md, Phase 3), and per this project's own
# convention (CLAUDE.md), an unverified guess at someone else's algorithm is
# worse than a clearly-labeled simplification. Verify against the primary
# source before citing this as a faithful TurboQuant reproduction.
#
# Requires head_dim to be a power of 2 (128 for both Qwen2.5-7B and
# Mistral-7B-v0.3 — true for every model this project currently uses, but NOT
# checked/padded for automatically if a future model's head_dim isn't).

_HADAMARD_CACHE: Dict[Tuple[int, str], torch.Tensor] = {}


def _hadamard_matrix(n: int) -> torch.Tensor:
    """n x n orthonormal Walsh-Hadamard matrix via Sylvester's recursive
    doubling construction. Returns H / sqrt(n) (orthonormal, not just
    orthogonal) so applying it is a pure rotation with no scale change."""
    if n & (n - 1) != 0:
        raise ValueError(f"Hadamard rotation requires a power-of-2 dimension, got {n}")
    h = torch.tensor([[1.0]], dtype=torch.float32)
    while h.shape[0] < n:
        h = torch.cat([
            torch.cat([h, h], dim=1),
            torch.cat([h, -h], dim=1),
        ], dim=0)
    return h / (n ** 0.5)


def _get_hadamard(n: int, device) -> torch.Tensor:
    key = (n, str(device))
    if key not in _HADAMARD_CACHE:
        _HADAMARD_CACHE[key] = _hadamard_matrix(n).to(device)
    return _HADAMARD_CACHE[key]


def _rotate(tensor: torch.Tensor) -> torch.Tensor:
    """Rotate the last (head_dim) axis by a fixed orthonormal Hadamard matrix."""
    h = _get_hadamard(tensor.shape[-1], tensor.device)
    return (tensor.float() @ h).to(tensor.dtype)


def _unrotate(tensor: torch.Tensor) -> torch.Tensor:
    """Invert _rotate. H is orthonormal, so H.T == H^-1 — exact (up to
    floating-point error) inversion of the rotation step, independent of
    whatever quantization error was introduced in between."""
    h = _get_hadamard(tensor.shape[-1], tensor.device)
    return (tensor.float() @ h.T).to(tensor.dtype)


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
        if mode not in ("none", "uniform_int8", "uniform_int4", "uniform_int4_rotated",
                        "uniform_int4_rotated_v_only", "uniform_int4_kivi_k_channel",
                        "uniform_int4_hybrid", "uniform_int4_kivi_full", "adaptive"):
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
        if self.mode in ('uniform_int4', 'uniform_int4_rotated', 'uniform_int4_rotated_v_only',
                         'uniform_int4_kivi_k_channel', 'uniform_int4_hybrid', 'uniform_int4_kivi_full'):
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
                    # Kept on their current device (GPU) — this whole pipeline
                    # runs in a single process on one GPU, nothing ever
                    # actually serializes/transmits a KVMessage over a wire,
                    # so the old .cpu() here was a pure round-trip cost (moved
                    # off GPU, then decompress() immediately moved it right
                    # back). original_bytes/compressed_bytes accounting below
                    # is unaffected — it's tensor.nbytes math, device-independent.
                    k_q=k,
                    v_q=v,
                    k_scale=torch.ones(b, h, device=k.device),
                    k_zp=torch.zeros(b, h, device=k.device),
                    v_scale=torch.ones(b, h, device=k.device),
                    v_zp=torch.zeros(b, h, device=k.device),
                    shape=tuple(k.shape),
                    bits=16,
                    layer_idx=layer_idx
                )
                compressed_bytes += k.nbytes + v.nbytes
            else:
                clip_pct = self.clip_percentile if (self.outlier_clipping and bits == 4) else None
                # Rotate BEFORE quantizing (uniform_int4_rotated /
                # uniform_int4_rotated_v_only) — spreads outlier magnitude
                # evenly across head_dim so per-head min-max quantization has
                # less to lose. See the rotation section above for why this
                # is lossless on its own — FOR A GENERIC TENSOR. K is NOT a
                # generic tensor: RoPE (rotary position embeddings) is
                # already applied to K before it's cached, pairing up
                # dimensions with a position-and-frequency-dependent phase.
                # A real-model test of uniform_int4_rotated (both K and V
                # rotated) produced wildly out-of-distribution decoded tokens
                # (literal Java class names, random CJK characters) — a
                # failure SHAPE inconsistent with ordinary quantization noise
                # and consistent with this second, RoPE-agnostic rotation
                # scrambling RoPE's paired-dimension phase structure in K.
                # uniform_int4_rotated_v_only exists to isolate this: rotate
                # V (no RoPE applied to V, so no equivalent risk) but leave K
                # unrotated, as a diagnostic before attempting any RoPE-aware
                # fix. See lakv_research_extensions_prompt.md Phase 3 update,
                # 2026-09-07.
                # uniform_int4_hybrid: the two pieces above, recombined —
                # K uses KIVI's per-channel grouping (RoPE-safe, never mixes
                # across dimensions), V uses the Hadamard rotation (proven to
                # reduce outlier error, and V has no RoPE to disrupt). Built
                # only after both pieces were independently validated on
                # their own — this is a recombination of already-tested
                # parts, not a new untested idea, but it should still only be
                # trusted over uniform_int4_kivi_k_channel alone if the real
                # numbers show it earns the extra complexity.
                # uniform_int4_kivi_full: KIVI's OTHER asymmetric half — K
                # per-channel (same as uniform_int4_kivi_k_channel) AND V
                # per-token (new — see _quantize_per_token's docstring for
                # why V's real variation is expected to live across
                # positions, the mirror of K's across-channel variation).
                # No rotation involved anywhere in this mode.
                rotate_k = self.mode == "uniform_int4_rotated"
                rotate_v = self.mode in ("uniform_int4_rotated", "uniform_int4_rotated_v_only",
                                          "uniform_int4_hybrid")
                k_in = _rotate(k) if rotate_k else k
                v_in = _rotate(v) if rotate_v else v

                use_k_per_channel = self.mode in ("uniform_int4_kivi_k_channel", "uniform_int4_hybrid",
                                                   "uniform_int4_kivi_full")
                use_v_per_token = self.mode == "uniform_int4_kivi_full"
                if use_k_per_channel:
                    # KIVI-inspired: K quantized per-channel (reduces over
                    # sequence only, keeps head_dim separate) instead of
                    # per-head — see _quantize_per_channel's docstring for
                    # why this targets K's RoPE-induced quantization
                    # difficulty specifically. What happens to V depends on
                    # mode: unchanged in uniform_int4_kivi_k_channel, rotated
                    # in uniform_int4_hybrid, per-token in
                    # uniform_int4_kivi_full — K's axis and V's treatment are
                    # independent knobs.
                    k_q, k_scale, k_zp = _quantize_per_channel(k_in, bits, clip_percentile=clip_pct)
                else:
                    k_q, k_scale, k_zp = _quantize(k_in, bits, clip_percentile=clip_pct)
                if use_v_per_token:
                    v_q, v_scale, v_zp = _quantize_per_token(v_in, bits, clip_percentile=clip_pct)
                else:
                    v_q, v_scale, v_zp = _quantize(v_in, bits, clip_percentile=clip_pct)

                # Real nibble-packing: previously k_q/v_q were always stored
                # as one full uint8 per element regardless of bits, so a
                # 4-bit value took the same byte an 8-bit value did (see git
                # history for the byte-accounting-bug fix that first caught
                # this). Packing two 4-bit values per byte here makes bits==4
                # genuinely half the storage of bits==8, not just nominally.
                # Only applies to bits==4 -- an 8-bit value already fills a
                # byte, nothing to pack. decompress() must unpack using the
                # same shape recorded below before dequantizing.
                if bits == 4:
                    k_q_stored = _pack_nibbles(k_q)
                    v_q_stored = _pack_nibbles(v_q)
                else:
                    k_q_stored = k_q
                    v_q_stored = v_q

                cl = CompressedLayer(
                    # Same reasoning as the bits==16 branch above — no .cpu().
                    k_q=k_q_stored,
                    v_q=v_q_stored,
                    k_scale=k_scale,
                    k_zp=k_zp,
                    v_scale=v_scale,
                    v_zp=v_zp,
                    shape=tuple(k.shape),
                    bits=bits,
                    layer_idx=layer_idx
                )
                compressed_bytes += k_q_stored.nbytes + v_q_stored.nbytes
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
                # bits==4 layers were nibble-packed in compress() -- unpack
                # back to cl.shape before dequantizing. bits==8 was never
                # packed (a byte already fits an 8-bit value), so k_q/v_q
                # are already the right shape.
                k_q = _unpack_nibbles(cl.k_q, cl.shape) if cl.bits == 4 else cl.k_q
                v_q = _unpack_nibbles(cl.v_q, cl.shape) if cl.bits == 4 else cl.v_q

                if message.mode in ("uniform_int4_kivi_k_channel", "uniform_int4_hybrid",
                                     "uniform_int4_kivi_full"):
                    k = _dequantize_per_channel(
                        k_q.to(device),
                        cl.k_scale.to(device),
                        cl.k_zp.to(device)
                    )
                else:
                    k = _dequantize(
                        k_q.to(device),
                        cl.k_scale.to(device),
                        cl.k_zp.to(device)
                    )
                if message.mode == "uniform_int4_kivi_full":
                    v = _dequantize_per_token(
                        v_q.to(device),
                        cl.v_scale.to(device),
                        cl.v_zp.to(device)
                    )
                else:
                    v = _dequantize(
                        v_q.to(device),
                        cl.v_scale.to(device),
                        cl.v_zp.to(device)
                    )
                if message.mode == "uniform_int4_rotated":
                    k = _unrotate(k)
                    v = _unrotate(v)
                elif message.mode in ("uniform_int4_rotated_v_only", "uniform_int4_hybrid"):
                    v = _unrotate(v)

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

        for mode in ['none', 'uniform_int8', 'uniform_int4', 'uniform_int4_rotated']:
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
