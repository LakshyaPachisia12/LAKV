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
        real_kv: Optional[Tuple] = None,
        real_kv_layer_indices: Optional[list] = None,
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
            real_kv: the REAL, actually-relayed KV for this hop's kept layers
                (post layer-selection, pre-compression — the exact content
                config D would transmit unmodified), in the same order as
                kv_message.layers. Forwarded to AnchorTable.query_correction
                as what the anchor delta gets added ON TOP OF, so a
                correction refines the real transmitted content rather than
                substituting a reconstruction for it (matches the actual
                KVCOMM reference design — see query_correction's docstring).
            real_kv_layer_indices: real (0-27) layer index for each entry in
                real_kv, i.e. selection_mask.kept_layer_indices, or None when
                layer selection is off (real_kv already covers all layers).
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
            real_kv=real_kv,
            real_kv_layer_indices=real_kv_layer_indices,
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
        # corrected_kv is now already POSITION-aligned with kv_message.layers
        # 1:1 — query_correction builds it by walking real_kv_layer_indices,
        # which pipeline.py passes in as this exact same order (selection_
        # mask.kept_layer_indices). No real-vs-position lookup needed here
        # any more (a previous version of this code needed one, when
        # corrected_kv was a dense 28-layer tuple built independently of
        # which layers were actually selected — that mismatch is what
        # produced the pure-noise output from a couple of iterations ago).
        new_layers = []
        for orig, (k, v) in zip(kv_message.layers, corrected_kv):
            layer_idx = orig.layer_idx
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

        # Recompute compressed_bytes from the actual new_layers tensors —
        # they're real bf16 (bits=16) now, not whatever kv_message.compressed_
        # bytes reflected pre-correction (e.g. INT8/adaptive-int4 for E/E_int8).
        # Carrying that stale figure forward understated the real transmitted
        # size on every hop where a correction hit. Same accounting convention
        # kv_compressor.py itself uses for its own bits==16 layers: real
        # tensor .nbytes, no bytes_per_el formula (that only applies to the
        # quantized branches, which this rebuilt message never uses).
        corrected_compressed_bytes = sum(k.nbytes + v.nbytes for k, v in corrected_kv)

        from lakv.kv_compressor import KVMessage as KVM
        corrected_msg = KVM(
            layers=new_layers,
            mode="anchor_corrected",
            original_bytes=kv_message.original_bytes,
            compressed_bytes=corrected_compressed_bytes,
        )
        return corrected_msg, True
