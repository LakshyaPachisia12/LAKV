"""
LAKV-v2 Module: Reconstructor V2
Restores omitted layers using nearest copy or global mean statistics, 
preventing zero-collapse observed in the original LAKV.
"""

import torch
from typing import Tuple, Dict
from lakv_v2.cache.selector import SelectionMask

class ReconstructorV2:
    def __init__(self, strategy: str = "nearest"):
        if strategy not in ["nearest", "mean_fill"]:
            raise ValueError("LAKV-v2 currently restricts to 'nearest', 'mean_fill' until baseline parity established.")
        self.strategy = strategy

    def reconstruct(self, selected_kv: Tuple, mask: SelectionMask) -> Tuple:
        kv_map = {}
        for i, opt in enumerate(mask.kept_layer_indices):
            kv_map[opt] = selected_kv[i]
            
        full_kv = []
        for i in range(mask.total_layers):
            if i in kv_map:
                full_kv.append(kv_map[i])
            else:
                if self.strategy == "nearest":
                    full_kv.append(self._nearest(i, kv_map))
                elif self.strategy == "mean_fill":
                    full_kv.append(self._mean_fill(kv_map))
                    
        return tuple(full_kv)

    def _nearest(self, idx: int, kv_map: Dict[int, Tuple[torch.Tensor, torch.Tensor]]):
        best = min(kv_map.keys(), key=lambda k: abs(k - idx))
        return kv_map[best]

    def _mean_fill(self, kv_map: Dict[int, Tuple[torch.Tensor, torch.Tensor]]):
        all_k = torch.stack([kv[0] for kv in kv_map.values()])
        all_v = torch.stack([kv[1] for kv in kv_map.values()])
        # Compute mean across layers, keep float16
        mean_k = all_k.mean(dim=0).to(torch.float16)
        mean_v = all_v.mean(dim=0).to(torch.float16)
        return (mean_k, mean_v)
