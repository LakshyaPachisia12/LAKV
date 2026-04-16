"""
LAKV-v2 Module: Cache Aligner
Fixes position and attention mask alignment for transferred KV Cache.
Specific to Qwen2.5 HuggingFace implementation.
"""

import torch
from transformers import DynamicCache
from typing import Tuple

def _to_tuple(pkv) -> tuple:
    if isinstance(pkv, DynamicCache):
        return pkv.to_legacy_cache()
    return pkv

def _to_dynamic(pkv: tuple) -> DynamicCache:
    return DynamicCache.from_legacy_cache(pkv)

class CacheAligner:
    """
    Handles exactly the RoPE alignment and attention mask extension for
    cache injected from Agent 1 to Agent 2.
    """
    def __init__(self, device: str = "cuda"):
        self.device = device

    def prepare_handoff_cache(
        self, received_kv_tuple: Tuple
    ) -> DynamicCache:
        """
        Takes raw selected/decompressed/reconstructed tuple KV.
        Returns a DynamicCache ready to inject into the model.
        """
        return _to_dynamic(received_kv_tuple)

    def compute_receiver_position_ids(
        self, 
        received_kv_tuple: Tuple,
        receiver_input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Given the injected cache sequence length L, and the new input of length S,
        returns the correct `position_ids` starting exactly at L.
        """
        cache_seq_len = received_kv_tuple[0][0].shape[2]
        new_seq_len = receiver_input_ids.shape[1]
        
        position_ids = torch.arange(
            cache_seq_len, 
            cache_seq_len + new_seq_len, 
            device=self.device
        ).unsqueeze(0)
        
        return position_ids

    def build_attention_mask(
        self, 
        received_kv_tuple: Tuple,
        receiver_input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        The receiver model requires an attention mask that covers BOTH 
        the inherited cache length and the new input tokens.
        If cache length is C and new tokens length is N, 
        attention_mask shape should be [batch, C + N].
        """
        cache_seq_len = received_kv_tuple[0][0].shape[2]
        new_seq_len = receiver_input_ids.shape[1]
        batch_size = receiver_input_ids.shape[0]

        # Everything is ones (attend to all inherited tokens + new tokens)
        attention_mask = torch.ones(
            (batch_size, cache_seq_len + new_seq_len), 
            dtype=torch.long, 
            device=self.device
        )
        return attention_mask
