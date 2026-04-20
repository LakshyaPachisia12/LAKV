"""
LAKV Module 4: OffsetCorrector

Applies anchor-table-based KV offset correction before injecting cache into
the next agent. Corrects for prefix-induced KV deviations using the shared
AnchorTable, replacing the original no-op stub.
"""

from typing import Optional, Tuple

import torch

from kv_compressor import KVMessage, CompressedLayer
from anchor_table import AnchorTable, question_key as make_key


class OffsetCorrector:
    """
    Corrects KV caches using the shared AnchorTable.

    Flow per hop:
      1. query_correction(question, agent_id) → corrected_kv | None
      2. If hit: rebuild KVMessage from corrected tensors (no compression loss added)
      3. If miss: return original message unchanged (caller falls back to full relay)
    """

    def __init__(self, anchor_table: Optional[AnchorTable] = None,
                 anchor_table_path: Optional[str] = None):
        # anchor_table_path kept for backwards compat with old stub signature
        self.anchor_table = anchor_table

    def correct(
        self,
        kv_message: KVMessage,
        sender_suffix_len: int,
        receiver_suffix_len: int,
        question: Optional[str] = None,
        agent_id: Optional[str] = None,
        query_hidden: Optional[torch.Tensor] = None,
        device: str = "cuda",
    ) -> Tuple[KVMessage, bool]:
        """
        Apply offset correction if possible.

        Args:
            kv_message: compressed KV cache from the sending agent.
            sender_suffix_len: token count of sender's suffix (unused in v1).
            receiver_suffix_len: token count of receiver's suffix (unused in v1).
            question: the shared placeholder text (math question).
            agent_id: which agent will receive this cache (e.g. "agent_1").
            query_hidden: (1, seq, hidden) hidden states from receiver's base KV.
            device: target device for corrected tensors.

        Returns:
            (corrected_message, was_corrected): corrected KVMessage + bool flag.
        """
        if (self.anchor_table is None or question is None
                or agent_id is None or query_hidden is None):
            return kv_message, False

        key = make_key(question)
        result = self.anchor_table.query_correction(key, agent_id, query_hidden, device)
        if result is None:
            return kv_message, False

        corrected_kv, confidence = result

        # Rebuild KVMessage from corrected float16 tensors (mode='none', no re-quantisation)
        # We preserve original bytes accounting so stats stay comparable.
        new_layers = []
        for layer_idx, (k, v) in enumerate(corrected_kv):
            if layer_idx >= len(kv_message.layers):
                break
            orig = kv_message.layers[layer_idx]
            new_layers.append(CompressedLayer(
                k_q=k.cpu(),
                v_q=v.cpu(),
                k_scale=torch.ones(k.shape[0], k.shape[1]),
                k_zp=torch.zeros(k.shape[0], k.shape[1]),
                v_scale=torch.ones(v.shape[0], v.shape[1]),
                v_zp=torch.zeros(v.shape[0], v.shape[1]),
                shape=tuple(k.shape),
                bits=16,
                layer_idx=orig.layer_idx,
            ))

        from kv_compressor import KVMessage as KVM
        corrected_msg = KVM(
            layers=new_layers,
            mode="anchor_corrected",
            original_bytes=kv_message.original_bytes,
            compressed_bytes=kv_message.compressed_bytes,
        )
        return corrected_msg, True
