"""
LAKV-v2 Module: KV Selector
Provides manual and tier-based selection of KV cache layers.
"""

from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class SelectionMask:
    kept_layer_indices: List[int]
    dropped_layer_indices: List[int]
    total_layers: int

class LayerSelectorV2:
    def __init__(self, keep_indices: List[int] = None, total_layers: int = 28, reconstructor=None):
        if keep_indices is None:
            keep_indices = list(range(total_layers))
        self.keep_indices = sorted(keep_indices)
        self.total_layers = total_layers
        self.reconstructor = reconstructor

    def select(self, past_key_values: Tuple) -> Tuple[Tuple, SelectionMask]:
        kept_layers = []
        dropped_indices = []
        
        for i in range(self.total_layers):
            if i in self.keep_indices:
                kept_layers.append(past_key_values[i])
            else:
                dropped_indices.append(i)
                
        mask = SelectionMask(
            kept_layer_indices=self.keep_indices.copy(),
            dropped_layer_indices=dropped_indices,
            total_layers=self.total_layers
        )
        return tuple(kept_layers), mask

    def reconstruct(self, selected_kv: Tuple, mask: SelectionMask) -> Tuple:
        if self.reconstructor is None:
            raise ValueError("Reconstructor is required for layer drops.")
        return self.reconstructor.reconstruct(selected_kv, mask)
