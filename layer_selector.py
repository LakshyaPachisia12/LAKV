"""
LAKV Module 2: LayerSelector

Takes a full 28-layer past_key_values tuple and returns only the Tier 1 + Tier 2
layers. Completely independent of compression. Provides reconstruction strategies
(zeros / nearest / interpolate) for restoring a full 28-layer KV tuple.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import torch

from calibration_profiler import LayerProfile


# ─── dataclass ────────────────────────────────────────────────────────────────

@dataclass
class SelectionMask:
    kept_layer_indices: List[int]       # e.g. [0, 1, 2, 4, 5, 7, …]
    dropped_layer_indices: List[int]    # e.g. [3, 6, 9, …]
    tier_per_kept_layer: Dict[int, int] # {layer_idx: tier}
    total_layers: int                   # 28
    seq_len: int
    reconstruction_strategy: str = ""   # set at reconstruct time


# ─── selector ─────────────────────────────────────────────────────────────────

class LayerSelector:
    """Filter and reconstruct KV cache layers according to tier assignment."""

    def __init__(self, profile: LayerProfile):
        self.profile = profile
        self.tier_assignment = profile.tier_assignment  # list of int, length 28
        self.n_layers = profile.n_layers

    @classmethod
    def _from_custom_indices(cls, keep_indices: List[int], n_layers: int = 28) -> "LayerSelector":
        """Build a selector that keeps exactly the given layer indices (ablation use)."""
        keep_set = set(keep_indices)
        tier_assignment = [1 if i in keep_set else 3 for i in range(n_layers)]
        # Minimal stub profile — only tier_assignment and n_layers are used by select()
        from dataclasses import fields
        dummy_floats = [0.0] * n_layers
        from calibration_profiler import LayerProfile
        dummy = LayerProfile(
            model_name="custom", dataset_name="custom",
            n_calibration_examples=0, n_layers=n_layers,
            attention_importance=dummy_floats, offset_variance=dummy_floats,
            effective_rank=dummy_floats, attention_importance_norm=dummy_floats,
            offset_variance_norm=dummy_floats, effective_rank_norm=dummy_floats,
            joint_score=dummy_floats, tier_assignment=tier_assignment,
            estimated_bytes_per_layer=dummy_floats,
        )
        return cls(dummy)

    # ── select ────────────────────────────────────────────────────────────

    def select(
        self, past_key_values: Tuple
    ) -> Tuple[Tuple, SelectionMask]:
        """Return only Tier 1 and Tier 2 layers.

        Args:
            past_key_values: tuple of length n_layers, each element is a
                (K, V) tuple with shape (batch, n_kv_heads, seq_len, head_dim).

        Returns:
            selected_kv: tuple of (K, V) for kept layers only
            mask: SelectionMask recording what was kept / dropped
        """
        kept_layers = []
        kept_indices = []
        dropped_indices = []
        tier_per_kept = {}

        for idx in range(self.n_layers):
            tier = self.tier_assignment[idx]
            if tier in (1, 2):
                kept_layers.append(past_key_values[idx])
                kept_indices.append(idx)
                tier_per_kept[idx] = tier
            else:
                dropped_indices.append(idx)

        # determine seq_len from the first kept layer
        seq_len = kept_layers[0][0].shape[2] if kept_layers else 0

        mask = SelectionMask(
            kept_layer_indices=kept_indices,
            dropped_layer_indices=dropped_indices,
            tier_per_kept_layer=tier_per_kept,
            total_layers=self.n_layers,
            seq_len=seq_len,
        )
        return tuple(kept_layers), mask

    # ── reconstruct ───────────────────────────────────────────────────────

    def reconstruct(
        self,
        selected_kv: Tuple,
        mask: SelectionMask,
        strategy: str = "zeros",
    ) -> Tuple:
        """Rebuild a full 28-layer past_key_values from kept layers.

        Strategies for dropped layers:
            'zeros'       – zero tensors with matching shape
            'nearest'     – copy from nearest kept layer by index
            'interpolate' – linearly interpolate between two nearest kept layers
        """
        mask.reconstruction_strategy = strategy

        # build a mapping: kept_layer_index → tensor pair
        kv_map: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for i, layer_idx in enumerate(mask.kept_layer_indices):
            kv_map[layer_idx] = selected_kv[i]

        # reference shape from first kept layer
        ref_k, ref_v = next(iter(kv_map.values()))

        full_kv: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for idx in range(mask.total_layers):
            if idx in kv_map:
                full_kv.append(kv_map[idx])
            else:
                if strategy == "zeros":
                    full_kv.append(self._zeros_like(ref_k, ref_v))
                elif strategy == "nearest":
                    full_kv.append(self._nearest(idx, kv_map))
                elif strategy == "interpolate":
                    full_kv.append(self._interpolate(idx, kv_map, mask))
                else:
                    raise ValueError(f"Unknown reconstruction strategy: {strategy}")

        return tuple(full_kv)

    # ── reconstruction helpers ────────────────────────────────────────────

    @staticmethod
    def _zeros_like(
        ref_k: torch.Tensor, ref_v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.zeros_like(ref_k), torch.zeros_like(ref_v))

    @staticmethod
    def _nearest(
        idx: int, kv_map: Dict[int, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        best = min(kv_map.keys(), key=lambda k: abs(k - idx))
        return kv_map[best]

    @staticmethod
    def _interpolate(
        idx: int,
        kv_map: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        mask: SelectionMask,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        kept = sorted(kv_map.keys())
        lower = [k for k in kept if k < idx]
        upper = [k for k in kept if k > idx]

        if not lower and not upper:
            # shouldn't happen, but fallback
            ref = next(iter(kv_map.values()))
            return (torch.zeros_like(ref[0]), torch.zeros_like(ref[1]))
        if not lower:
            return kv_map[upper[0]]
        if not upper:
            return kv_map[lower[-1]]

        lo, hi = lower[-1], upper[0]
        alpha = (idx - lo) / (hi - lo)
        k_interp = kv_map[lo][0] * (1 - alpha) + kv_map[hi][0] * alpha
        v_interp = kv_map[lo][1] * (1 - alpha) + kv_map[hi][1] * alpha
        return (k_interp, v_interp)
