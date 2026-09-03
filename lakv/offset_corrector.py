"""
LAKV Module 4: OffsetCorrector

Applies anchor-table-based KV offset correction before injecting cache into
the next agent. Corrects for prefix-induced KV deviations using the shared
AnchorTable, replacing the original no-op stub.
"""

from typing import Optional, Tuple

import torch

from lakv.kv_compressor import KVMessage, CompressedLayer
from lakv.anchor_table import AnchorTable, question_key as make_key


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
        self.last_offset_log = {}

    def correct(
        self,
        kv_message: KVMessage,
        sender_suffix_len: int = 0,
        receiver_suffix_len: int = 0,
        question: Optional[str] = None,
        agent_id: Optional[str] = None,
        channel_key: Optional[str] = None,
        sender_seq_len: Optional[int] = None,
        receiver_prompt_len: Optional[int] = None,
        query_hidden: Optional[torch.Tensor] = None,
        query_base_kv: Optional[Tuple] = None,
        device: str = "cuda",
        rope_theta: float = 1_000_000.0,
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
            query_base_kv: THIS question's own base_kv (from compute_base_kv) —
                forwarded to AnchorTable.query_correction as the reconstruction's
                base content, so a correction transfers the matched anchor's
                position/prefix DELTA onto this question's own facts rather
                than substituting the anchor's own (different) content.
            device: target device for corrected tensors.

        Returns:
            (corrected_message, was_corrected): corrected KVMessage + bool flag.
        """
        sender_len = sender_seq_len if sender_seq_len is not None else sender_suffix_len
        receiver_len = receiver_prompt_len if receiver_prompt_len is not None else receiver_suffix_len
        raw_offset = int(sender_len - receiver_len)
        # NOTE: "applied_offset" below is diagnostic-only (logged for
        # inspection, not consumed by evaluator.py) and, despite the name,
        # is NOT what pipeline.py uses to position the receiver's new tokens
        # — that value is computed separately in pipeline.py from
        # target_prompt_len - corrected_seq_len (matching anchor_table.py's
        # own RoPE target_shift) and only when a correction actually hit.
        # This field predates that fix and is a different, sender-length-
        # based quantity; kept for backwards-compatible logging only.
        applied_offset = max(0, raw_offset)
        self.last_offset_log = {
            "sender_len": int(sender_len),
            "receiver_len": int(receiver_len),
            "raw_offset": raw_offset,
            "applied_offset": applied_offset,
        }

        effective_channel = channel_key or agent_id

        if (self.anchor_table is None or question is None
                or effective_channel is None or query_hidden is None):
            return kv_message, False

        key = make_key(question)
        result = self.anchor_table.query_correction(
            key,
            effective_channel,
            query_hidden,
            query_base_kv=query_base_kv,
            device=device,
            target_prompt_len=receiver_len,
            rope_theta=rope_theta,
        )
        # Surface the best candidate's raw L2 distance regardless of hit/miss
        # — this is what anchor_max_distance would threshold on, and needs to
        # be visible in real run data before picking a value (see
        # AnchorTable.max_distance docstring).
        self.last_offset_log["anchor_min_distance"] = self.anchor_table.last_query_min_distance
        if result is None:
            return kv_message, False

        corrected_kv, confidence = result

        # Rebuild KVMessage from corrected bfloat16 tensors (mode='none', no re-quantisation)
        # We preserve original bytes accounting so stats stay comparable.
        #
        # corrected_kv is a DENSE, full 28-layer tuple indexed by REAL layer
        # index (anchor_table.update() is called with raw_kv_tuple, i.e.
        # before layer selection ever runs — see pipeline.py::run()). But
        # kv_message.layers only holds the layers layer selection actually
        # KEPT (e.g. 20/28), in ascending real-index order — NOT a dense
        # 0..19 range. The old code did `enumerate(corrected_kv)` and
        # matched position i against kv_message.layers[i], silently
        # assuming those two indexings lined up. They don't, as soon as any
        # layer below position len(kv_message.layers)-1 was dropped (always
        # true once any selection happens) — real layer 5's corrected KV
        # would get relabeled and injected as whatever layer kv_message.
        # layers[5] actually was (e.g. real layer 7), scrambling which
        # attention layer's weights see which cache. Confirmed as the cause
        # of E/E_int8 producing pure noise output (checked raw generated
        # text — token-soup garbage from the very first token, consistent
        # with cache data landing in the wrong layer's attention entirely).
        #
        # Fix: index corrected_kv by each kept layer's REAL layer_idx, not
        # by its position in the (sparse) kept-layer list.
        new_layers = []
        for orig in kv_message.layers:
            layer_idx = orig.layer_idx
            if layer_idx >= len(corrected_kv):
                break
            k, v = corrected_kv[layer_idx]
            new_layers.append(CompressedLayer(
                # No .cpu() — same reasoning as kv_compressor.py's compress():
                # single-process pipeline, nothing ever actually transmits a
                # KVMessage over a wire, so the round-trip was pure overhead.
                k_q=k,
                v_q=v,
                k_scale=torch.ones(k.shape[0], k.shape[1], device=k.device),
                k_zp=torch.zeros(k.shape[0], k.shape[1], device=k.device),
                v_scale=torch.ones(v.shape[0], v.shape[1], device=k.device),
                v_zp=torch.zeros(v.shape[0], v.shape[1], device=k.device),
                shape=tuple(k.shape),
                bits=16,
                layer_idx=layer_idx,
            ))

        from lakv.kv_compressor import KVMessage as KVM
        corrected_msg = KVM(
            layers=new_layers,
            mode="anchor_corrected",
            original_bytes=kv_message.original_bytes,
            compressed_bytes=kv_message.compressed_bytes,
        )
        return corrected_msg, True
