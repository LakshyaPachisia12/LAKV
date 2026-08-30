"""
LAKV-v2 Module: Compressor V2
Explicit precision policies leveraging existing quantization routines.

ARCHIVED during the folder reorg: implementation_plan.txt already flagged
this as a duplicate of the root lakv/kv_compressor.py, unreferenced by the
live v2 path (lakv_v2/pipeline/shared_prefix_pipeline.py imports the root
module directly instead). Kept for reference only.
"""

import torch
from typing import Tuple, Dict, Optional
from lakv.kv_compressor import _quantize, _dequantize, CompressedLayer, KVMessage

class CompressorV2:
    def __init__(self, mode: str = "none", explicit_policy: Optional[Dict[int, int]] = None):
        """
        mode: 'none', 'uniform_int8', 'uniform_int4', or 'mixed'
        explicit_policy: if mode is 'mixed', dict mapping relative layer ID to bits (16, 8, 4).
        """
        self.mode = mode
        self.explicit_policy = explicit_policy or {}

    def _get_bits(self, rel_layer_idx: int) -> int:
        if self.mode == "none":
            return 16
        if self.mode == "uniform_int8":
            return 8
        if self.mode == "uniform_int4":
            return 4
        if self.mode == "mixed":
            return self.explicit_policy.get(rel_layer_idx, 8) # Fallback to 8
        return 16

    def compress(self, past_key_values: Tuple) -> KVMessage:
        compressed_layers = []
        original_bytes = 0
        compressed_bytes = 0

        for idx, (k, v) in enumerate(past_key_values):
            original_bytes += k.nbytes + v.nbytes
            bits = self._get_bits(idx)

            if bits == 16:
                b, h = k.shape[0], k.shape[1]
                cl = CompressedLayer(
                    k_q=k.cpu(), v_q=v.cpu(),
                    k_scale=torch.ones(b, h), k_zp=torch.zeros(b, h),
                    v_scale=torch.ones(b, h), v_zp=torch.zeros(b, h),
                    shape=tuple(k.shape), bits=16, layer_idx=idx
                )
                compressed_bytes += k.nbytes + v.nbytes
            else:
                k_q, k_s, k_z = _quantize(k, bits)
                v_q, v_s, v_z = _quantize(v, bits)
                cl = CompressedLayer(
                    k_q=k_q.cpu(), v_q=v_q.cpu(),
                    k_scale=k_s.cpu(), k_zp=k_z.cpu(),
                    v_scale=v_s.cpu(), v_zp=v_z.cpu(),
                    shape=tuple(k.shape), bits=bits, layer_idx=idx
                )
                bytes_per_el = {8: 1, 4: 0.5}[bits]
                compressed_bytes += int((k.numel() + v.numel()) * bytes_per_el)
                compressed_bytes += (k_s.numel() + v_s.numel()) * 4 * 2

            compressed_layers.append(cl)

        return KVMessage(compressed_layers, self.mode, original_bytes, compressed_bytes)

    def decompress(self, message: KVMessage, device: str = "cuda") -> Tuple:
        res = []
        for cl in message.layers:
            if cl.bits == 16:
                k, v = cl.k_q.to(device), cl.v_q.to(device)
            else:
                k = _dequantize(cl.k_q.to(device), cl.k_scale.to(device), cl.k_zp.to(device))
                v = _dequantize(cl.v_q.to(device), cl.v_scale.to(device), cl.v_zp.to(device))
            # Shape restore manually if needed
            res.append((k, v))
        return tuple(res)
