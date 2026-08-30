"""
LAKV: Shared Anchor Table

Stores per-agent KV deviation observations (delta = actual_kv - base_kv) for
shared placeholder content (the math question). Enables cross-agent KV correction
without full retransmission.

Architecture (novel vs KVCOMM):
  KVCOMM: per-agent anchor pools, no cross-agent sharing.
  Ours  : one shared pool, Agent 0's observations immediately benefit Agent 1+.

Usage:
  table = AnchorTable(max_size=20, entropy_threshold=0.3)
  # After agent i generates KV for question q:
  table.update(question_key, agent_id, base_kv, actual_kv, hidden_states)
  # Before injecting KV into agent j:
  result = table.query_correction(question_key, agent_id, query_hidden)
  if result:
      corrected_kv, confidence = result
"""

import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


# ─── dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class AgentOffsets:
    """Stored KV deviation for one agent: delta = actual_kv - base_kv."""
    delta_k: List[torch.Tensor]  # [n_layers], each (1, n_kv_heads, seq, head_dim)
    delta_v: List[torch.Tensor]


@dataclass
class AnchorEntry:
    placeholder_key: str               # hash of question tokens
    base_k: List[torch.Tensor]         # KV of question with NO prefix (28 layers)
    base_v: List[torch.Tensor]
    embedding: torch.Tensor            # mean-pool hidden states [hidden_dim], CPU
    agent_offsets: Dict[str, AgentOffsets] = field(default_factory=dict)
    access_count: int = 0
    created_at: float = field(default_factory=time.time)


# ─── anchor table ─────────────────────────────────────────────────────────────

class AnchorTable:
    """
    Shared cross-agent anchor pool. Thread-safety is NOT guaranteed (single-process
    eval loop only). Evicts least-frequently-used entries when full.
    """

    def __init__(self, max_size: int = 20, entropy_threshold: float = 0.3,
                 min_confidence: float = 0.5, verbose: bool = False,
                 graceful_degradation: bool = True):
        self.max_size = max_size
        self.entropy_threshold = entropy_threshold
        self.min_confidence = min_confidence
        self.verbose = verbose
        # When True (KVCOMM-style): an ambiguous/low-confidence match still
        # gets a blended correction applied rather than being rejected
        # outright. Rejecting means the caller falls back to completely
        # uncorrected KV (see OffsetCorrector.correct on a None result) —
        # the same "wrong position, no RoPE correction" failure mode that
        # collapsed Config E before the target_prompt_len fix. A low-
        # confidence blended correction is expected to usually be closer to
        # right than applying no correction at all. Set False to restore
        # the old strict all-or-nothing gate for ablation/comparison.
        self.graceful_degradation = graceful_degradation
        self._pool: Dict[str, AnchorEntry] = {}  # key → AnchorEntry
        self._hit_log: List[dict] = []            # for experiment logging
        self._admission_log: List[dict] = []      # _should_add_new_anchor call history

    # ── public API ────────────────────────────────────────────────────────

    def update(
        self,
        question_key: str,
        agent_id: str,
        base_kv: Tuple,             # tuple of (K, V) per layer, no-prefix KV
        actual_kv: Tuple,           # tuple of (K, V) per layer, in-context KV
        hidden_states: torch.Tensor,  # (1, seq, hidden) last-layer hidden states
        prompt_seq_len: Optional[int] = None,
        rope_theta: float = 1_000_000.0,
    ) -> None:
        """Compute ΔKV and store under (question_key, agent_id).

        New entries are added only when paper Eq. 5 is satisfied:
          P_anchor(σ) = (L_σ > max_α L_α) AND (H|A > τ log|A|)
        i.e. the new sequence is longer than all existing anchors AND no
        sufficiently close match already exists (high entropy of weights).
        Existing entries are always updated (agent offset refresh).
        """
        n_layers = len(base_kv)

        delta_k = []
        delta_v = []
        for layer_idx in range(n_layers):
            bk, bv = base_kv[layer_idx]
            ak, av = actual_kv[layer_idx]
            # Suffix-aligned + de-rotated delta computation
            base_seq = bk.shape[2]
            actual_prompt_seq = min(int(prompt_seq_len) if prompt_seq_len is not None else int(ak.shape[2]), int(ak.shape[2]))
            start = max(actual_prompt_seq - base_seq, 0)
            end = min(start + base_seq, int(ak.shape[2]))
            seg_len = max(end - start, 1)

            bk_seg = bk[:, :, :seg_len, :]
            bv_seg = bv[:, :, :seg_len, :]
            ak_seg = ak[:, :, start:end, :]
            av_seg = av[:, :, start:end, :]

            pos_shift = max(actual_prompt_seq - base_seq, 0)
            ak_aligned = self._rope_shift_k(ak_seg, shift=-pos_shift, theta=rope_theta)

            delta_k.append((ak_aligned - bk_seg).cpu())
            delta_v.append((av_seg - bv_seg).cpu())

        emb = self._embed(hidden_states)
        base_seq_len = base_kv[0][0].shape[2]

        if question_key in self._pool:
            # Always refresh the offset for this agent on an existing entry
            entry = self._pool[question_key]
            entry.agent_offsets[agent_id] = AgentOffsets(delta_k=delta_k, delta_v=delta_v)
        elif self._should_add_new_anchor(base_seq_len, emb):
            # Paper Eq. 5: only add if longer than all existing AND no good match
            self.evict_if_full()
            self._pool[question_key] = AnchorEntry(
                placeholder_key=question_key,
                base_k=[bk.cpu() for bk, _ in base_kv],
                base_v=[bv.cpu() for _, bv in base_kv],
                embedding=emb,
                agent_offsets={agent_id: AgentOffsets(delta_k=delta_k, delta_v=delta_v)},
            )
        # If Eq. 5 not satisfied the entry is skipped — existing anchors are
        # close enough that adding a near-duplicate would not improve retrieval.

    def query_correction(
        self,
        question_key: str,
        agent_id: str,
        query_hidden: torch.Tensor,   # (1, seq, hidden) last-layer hidden states
        device: str = "cuda",
        target_prompt_len: Optional[int] = None,
        rope_theta: float = 1_000_000.0,
    ) -> Optional[Tuple[Tuple, float]]:
        """
        Returns (corrected_kv_tuple, confidence) if a usable anchor is found,
        None if the pool is empty or entropy check fails (caller should use
        transmitted KV as-is).

        corrected_kv_tuple: tuple of (K, V) per layer on `device`
        confidence: scalar in [0, 1]; 1 = perfect single-anchor match
        """
        # Filter to entries that have offsets for this agent_id
        candidates = [e for e in self._pool.values() if agent_id in e.agent_offsets]
        if not candidates:
            self._log_miss(question_key, agent_id)
            if self.verbose:
                print(f"  [AnchorTable] MISS key={question_key} agent={agent_id} reason=no_candidates")
            return None

        query_emb = self._embed(query_hidden)
        weights = self._similarity_weights(query_emb, candidates)  # [n_cands]

        # Entropy gate: if weights are too spread (ambiguous), the match is
        # low-quality. With graceful_degradation=False (old behavior) this
        # rejects outright; with graceful_degradation=True (default) we keep
        # going and still apply a blended correction — see __init__ docstring
        # on why "no correction" is worse than "uncertain correction" here.
        entropy = -(weights * (weights + 1e-9).log()).sum().item()
        max_entropy = self.entropy_threshold * torch.tensor(len(candidates)).float().log().item()
        entropy_ambiguous = len(candidates) > 1 and entropy > max(max_entropy, 0.01)
        if entropy_ambiguous and not self.graceful_degradation:
            self._log_miss(question_key, agent_id, entropy=entropy, reason="entropy")
            if self.verbose:
                print(f"  [AnchorTable] MISS key={question_key} agent={agent_id} "
                      f"reason=entropy entropy={entropy:.4f} max_entropy={max_entropy:.4f}")
            return None

        # Confidence = 1 - normalized_entropy (1.0 for a single perfect match)
        norm_entropy = entropy / max(torch.tensor(len(candidates)).float().log().item(), 1e-9)
        confidence = float(1.0 - norm_entropy)

        # Confidence floor: a borderline match that squeaks past the entropy gate
        # can still be a poor-quality anchor. With graceful_degradation=False
        # (old behavior) this skips correction entirely, falling back to the
        # transmitted KV as-is (uncorrected, wrong position). With
        # graceful_degradation=True we still apply the blended correction —
        # a low-confidence correction is expected to usually beat no
        # correction at all, since "no correction" means the receiver gets
        # KV at the sender's raw position with no RoPE adjustment.
        if confidence < self.min_confidence and not self.graceful_degradation:
            self._log_miss(question_key, agent_id, entropy=entropy, reason="low_confidence")
            if self.verbose:
                print(f"  [AnchorTable] MISS key={question_key} agent={agent_id} "
                      f"reason=low_confidence confidence={confidence:.4f} "
                      f"min_confidence={self.min_confidence:.4f}")
            return None

        # Interpolate delta across candidates. Different candidates (different
        # questions) can have different stored delta sequence lengths, so
        # truncate to the shortest common length before summing — a plain
        # sum() of mismatched shapes raises regardless of how small a
        # candidate's weight is, and this path is reached more often now
        # that graceful_degradation no longer rejects ambiguous (i.e.
        # genuinely multi-candidate) matches before we get here.
        n_layers = len(candidates[0].agent_offsets[agent_id].delta_k)
        min_delta_seq = min(
            candidates[ci].agent_offsets[agent_id].delta_k[0].shape[2]
            for ci in range(len(candidates))
        )
        delta_k_interp = []
        delta_v_interp = []
        for layer_idx in range(n_layers):
            dk = sum(weights[ci].item() * candidates[ci].agent_offsets[agent_id].delta_k[layer_idx][:, :, :min_delta_seq, :]
                     for ci in range(len(candidates)))
            dv = sum(weights[ci].item() * candidates[ci].agent_offsets[agent_id].delta_v[layer_idx][:, :, :min_delta_seq, :]
                     for ci in range(len(candidates)))
            delta_k_interp.append(dk)
            delta_v_interp.append(dv)

        # Blend base KV with the SAME weights used for the delta, so the
        # base and delta are drawn from a consistent mixture of anchors
        # instead of "delta blended across all candidates, base taken from
        # only the single best-matching one" (the previous behavior — an
        # inconsistency that matters most exactly when confidence is low,
        # i.e. when weight mass is spread across multiple candidates whose
        # base KVs may differ substantially, since they come from different
        # questions). All candidate base_k/base_v are truncated/left as-is;
        # sequence-length alignment happens per-layer below using the
        # blended base's own length.
        base_seq_lens = [c.base_k[0].shape[2] for c in candidates]
        min_base_seq = min(base_seq_lens)
        base_k_interp = []
        base_v_interp = []
        for layer_idx in range(n_layers):
            bk = sum(weights[ci].item() * candidates[ci].base_k[layer_idx][:, :, :min_base_seq, :].to(torch.float32)
                     for ci in range(len(candidates)))
            bv = sum(weights[ci].item() * candidates[ci].base_v[layer_idx][:, :, :min_base_seq, :].to(torch.float32)
                     for ci in range(len(candidates)))
            base_k_interp.append(bk.to(candidates[0].base_k[layer_idx].dtype))
            base_v_interp.append(bv.to(candidates[0].base_v[layer_idx].dtype))

        best_idx = int(weights.argmax().item())
        candidates[best_idx].access_count += 1

        corrected = []
        for layer_idx in range(n_layers):
            bk = base_k_interp[layer_idx].to(device)
            bv = base_v_interp[layer_idx].to(device)
            dk = delta_k_interp[layer_idx].to(device)
            dv = delta_v_interp[layer_idx].to(device)
            # Sequence length: base and delta blends can have different
            # lengths (min_base_seq vs min_delta_seq computed separately
            # above) — use the shorter of the two so the add below can
            # never mismatch shapes.
            seq = min(bk.shape[2], dk.shape[2])
            k_corr = bk[:, :, :seq, :] + dk[:, :, :seq, :]
            v_corr = bv[:, :, :seq, :] + dv[:, :, :seq, :]
            if target_prompt_len is not None:
                target_shift = max(int(target_prompt_len) - seq, 0)
                k_corr = self._rope_shift_k(k_corr, shift=target_shift, theta=rope_theta)
            corrected.append((k_corr, v_corr))

        delta_norm = float(sum(
            dk.float().norm().item() for dk in delta_k_interp
        ) / max(len(delta_k_interp), 1))
        self._log_hit(question_key, agent_id, confidence, len(candidates), delta_norm=delta_norm)
        if self.verbose:
            print(f"  [AnchorTable] HIT  key={question_key} agent={agent_id} "
                  f"conf={confidence:.3f} n_cands={len(candidates)} "
                  f"mean_delta_k_norm={delta_norm:.4f}")
        return tuple(corrected), confidence

    def evict_if_full(self) -> None:
        if len(self._pool) < self.max_size:
            return
        # Evict LFU among the oldest half of entries
        sorted_entries = sorted(self._pool.items(),
                                key=lambda kv: (kv[1].access_count, kv[1].created_at))
        evict_key = sorted_entries[0][0]
        del self._pool[evict_key]

    def hit_rate(self) -> float:
        if not self._hit_log:
            return 0.0
        hits = sum(1 for e in self._hit_log if e["hit"])
        return hits / len(self._hit_log)

    def pool_size(self) -> int:
        return len(self._pool)

    def hit_log(self) -> List[dict]:
        return list(self._hit_log)

    def admission_log(self) -> List[dict]:
        return list(self._admission_log)

    def call_counts(self) -> Dict[str, int]:
        """Summary counters for empirical verification of Claim 2."""
        return {
            "admission_calls": len(self._admission_log),
            "admissions_added": sum(1 for e in self._admission_log if e["decision"]),
            "admissions_rejected": len(self._admission_log) - sum(
                1 for e in self._admission_log if e["decision"]),
            "correction_calls": len(self._hit_log),
            "corrections_applied": sum(1 for e in self._hit_log if e["hit"]),
            "corrections_skipped": sum(1 for e in self._hit_log if not e["hit"]),
        }

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _embed(hidden_states: torch.Tensor) -> torch.Tensor:
        """Mean-pool last-layer hidden states → [hidden_dim] on CPU."""
        # hidden_states: (batch, seq, hidden) — take batch 0, mean over seq
        return hidden_states[0].float().mean(dim=0).cpu()

    @staticmethod
    def _similarity_weights(query_emb: torch.Tensor,
                             candidates: List[AnchorEntry]) -> torch.Tensor:
        """softmax(-||h_q - h_c||₂) for each candidate. Returns [n_cands]."""
        dists = torch.stack([
            torch.dist(query_emb, c.embedding, p=2) for c in candidates
        ])
        return torch.softmax(-dists, dim=0)

    @staticmethod
    def _rope_shift_k(k: torch.Tensor, shift: int, theta: float = 1_000_000.0) -> torch.Tensor:
        """Apply a constant RoPE shift to key tensor along the rotary dimension."""
        if shift == 0:
            return k
        d = k.shape[-1]
        if d % 2 != 0:
            return k

        kf = k.float()
        half = d // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=k.device, dtype=torch.float32) / half))
        angle = float(shift) * inv_freq
        cos = torch.cos(angle).view(1, 1, 1, half)
        sin = torch.sin(angle).view(1, 1, 1, half)

        x1 = kf[..., :half]
        x2 = kf[..., half:]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        return torch.cat([y1, y2], dim=-1).to(k.dtype)

    def _should_add_new_anchor(self, seq_len: int, emb: torch.Tensor) -> bool:
        """Paper Eq. 5: add new anchor only when BOTH conditions hold.

        1. Length criterion: the sequence is longer than all existing anchors
           (adds genuinely new information to the pool).
        2. Entropy criterion: no existing anchor is a close match (high entropy
           of softmax similarity weights means no single dominant match).
        """
        if not self._pool:
            self._log_admission(seq_len, True, reason="empty_pool")
            return True
        existing = list(self._pool.values())
        # Condition 1 — length
        max_existing_len = max(e.base_k[0].shape[2] for e in existing)
        if seq_len <= max_existing_len:
            self._log_admission(seq_len, False, reason="length",
                                 max_existing_len=max_existing_len)
            return False
        # Condition 2 — entropy
        weights = self._similarity_weights(emb, existing)
        entropy = -(weights * (weights + 1e-9).log()).sum().item()
        n = len(existing)
        max_entropy = self.entropy_threshold * (torch.tensor(float(n)).log().item() if n > 1 else 1.0)
        decision = entropy > max(max_entropy, 0.01)
        self._log_admission(seq_len, decision, reason="entropy",
                             max_existing_len=max_existing_len,
                             entropy=entropy, max_entropy=max_entropy)
        return decision

    def _log_admission(self, seq_len, decision, reason, **extra):
        entry = {
            "seq_len": seq_len, "decision": decision, "reason": reason,
            "pool_size": len(self._pool), **extra,
        }
        self._admission_log.append(entry)
        if self.verbose:
            print(f"  [AnchorTable] _should_add_new_anchor seq_len={seq_len} "
                  f"decision={decision} reason={reason} pool_size={len(self._pool)} {extra}")

    def _log_hit(self, key, agent_id, confidence, n_candidates, delta_norm=None):
        self._hit_log.append({
            "hit": True, "key": key, "agent_id": agent_id,
            "confidence": confidence, "n_candidates": n_candidates,
            "delta_norm": delta_norm,
            "pool_size": len(self._pool),
        })

    def _log_miss(self, key, agent_id, entropy=None, reason="no_candidates"):
        self._hit_log.append({
            "hit": False, "key": key, "agent_id": agent_id,
            "reason": reason, "entropy": entropy,
            "pool_size": len(self._pool),
        })


# ─── utilities ────────────────────────────────────────────────────────────────

def question_key(question: str) -> str:
    """Stable hash key for a question string."""
    return hashlib.md5(question.encode()).hexdigest()[:12]


def compute_base_kv(model, tokenizer, question: str, device: str = "cuda") -> Tuple:
    """
    Compute the KV cache for `question` with NO system prefix.
    This is the 'base' that anchors are measured relative to.
    """
    prompt = (
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, output_hidden_states=True)

    # past_key_values may be DynamicCache — convert to tuple. The transformers
    # internal cache API has changed across versions, so try each in order:
    # 1. to_legacy_cache() (older stable API)
    # 2. key_cache/value_cache attributes (some intermediate versions)
    # 3. .layers attribute exposing per-layer (keys, values) objects (newer API)
    from transformers import DynamicCache
    pkv = outputs.past_key_values
    if isinstance(pkv, DynamicCache):
        if hasattr(pkv, "to_legacy_cache"):
            pkv = pkv.to_legacy_cache()
        elif hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
            pkv = tuple(zip(pkv.key_cache, pkv.value_cache))
        elif hasattr(pkv, "layers"):
            pkv = tuple((layer.keys, layer.values) for layer in pkv.layers)
        else:
            raise AttributeError(
                "DynamicCache exposes none of to_legacy_cache(), "
                "key_cache/value_cache, or .layers — unsupported transformers "
                "version's cache API."
            )

    # last hidden state for embedding
    last_hidden = outputs.hidden_states[-1]  # (1, seq, hidden)
    return pkv, last_hidden
