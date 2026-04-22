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

    def __init__(self, max_size: int = 20, entropy_threshold: float = 0.3):
        self.max_size = max_size
        self.entropy_threshold = entropy_threshold
        self._pool: Dict[str, AnchorEntry] = {}  # key → AnchorEntry
        self._hit_log: List[dict] = []            # for experiment logging

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
        """Compute ΔKV and store under (question_key, agent_id)."""
        n_layers = len(base_kv)

        delta_k = []
        delta_v = []
        for layer_idx in range(n_layers):
            bk, bv = base_kv[layer_idx]
            ak, av = actual_kv[layer_idx]
            base_seq = bk.shape[2]
            actual_prompt_seq = int(prompt_seq_len) if prompt_seq_len is not None else int(ak.shape[2])
            actual_prompt_seq = min(actual_prompt_seq, int(ak.shape[2]))

            # Use prompt-only KV segment and align to the base segment by suffix.
            # base prompt: user+question+assistant
            # actual prompt: system+user+question+assistant (plus optional different prefix)
            start = max(actual_prompt_seq - base_seq, 0)
            end = min(start + base_seq, int(ak.shape[2]))
            seg_len = max(end - start, 1)

            bk_seg = bk[:, :, :seg_len, :]
            bv_seg = bv[:, :, :seg_len, :]
            ak_seg = ak[:, :, start:end, :]
            av_seg = av[:, :, start:end, :]

            # KVCOMM-style positional alignment for K: de-rotate by prefix position shift.
            pos_shift = max(actual_prompt_seq - base_seq, 0)
            ak_aligned = self._rope_shift_k(ak_seg, shift=-pos_shift, theta=rope_theta)

            delta_k.append((ak_aligned - bk_seg).cpu())
            delta_v.append((av_seg - bv_seg).cpu())

        emb = self._embed(hidden_states)

        if question_key in self._pool:
            entry = self._pool[question_key]
            entry.agent_offsets[agent_id] = AgentOffsets(delta_k=delta_k, delta_v=delta_v)
        else:
            self.evict_if_full()
            self._pool[question_key] = AnchorEntry(
                placeholder_key=question_key,
                base_k=[bk.cpu() for bk, _ in base_kv],
                base_v=[bv.cpu() for _, bv in base_kv],
                embedding=emb,
                agent_offsets={agent_id: AgentOffsets(delta_k=delta_k, delta_v=delta_v)},
            )

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
            return None

        query_emb = self._embed(query_hidden)
        weights = self._similarity_weights(query_emb, candidates)  # [n_cands]

        # Entropy gate: if weights are too spread (ambiguous), fall back
        entropy = -(weights * (weights + 1e-9).log()).sum().item()
        max_entropy = self.entropy_threshold * torch.tensor(len(candidates)).float().log().item()
        if len(candidates) > 1 and entropy > max(max_entropy, 0.01):
            self._log_miss(question_key, agent_id, entropy=entropy, reason="entropy")
            return None

        # Confidence = 1 - normalized_entropy (1.0 for a single perfect match)
        norm_entropy = entropy / max(torch.tensor(len(candidates)).float().log().item(), 1e-9)
        confidence = float(1.0 - norm_entropy)

        # Interpolate delta across candidates
        n_layers = len(candidates[0].agent_offsets[agent_id].delta_k)
        delta_k_interp = []
        delta_v_interp = []
        for layer_idx in range(n_layers):
            dk = sum(weights[ci].item() * candidates[ci].agent_offsets[agent_id].delta_k[layer_idx]
                     for ci in range(len(candidates)))
            dv = sum(weights[ci].item() * candidates[ci].agent_offsets[agent_id].delta_v[layer_idx]
                     for ci in range(len(candidates)))
            delta_k_interp.append(dk)
            delta_v_interp.append(dv)

        # Apply correction: corrected = base + Δ
        # Use the best-matching candidate's base KV
        best_idx = int(weights.argmax().item())
        best_entry = candidates[best_idx]
        best_entry.access_count += 1

        corrected = []
        for layer_idx in range(n_layers):
            bk = best_entry.base_k[layer_idx].to(device)
            bv = best_entry.base_v[layer_idx].to(device)
            dk = delta_k_interp[layer_idx].to(device)
            dv = delta_v_interp[layer_idx].to(device)
            # Sequence length: use base length (shortest safe overlap)
            seq = bk.shape[2]
            k_corr = bk + dk[:, :, :seq, :]
            v_corr = bv + dv[:, :, :seq, :]

            # Re-apply RoPE position shift so corrected K matches receiver prompt context.
            if target_prompt_len is not None:
                target_shift = max(int(target_prompt_len) - seq, 0)
                k_corr = self._rope_shift_k(k_corr, shift=target_shift, theta=rope_theta)

            corrected.append((k_corr, v_corr))

        self._log_hit(question_key, agent_id, confidence, len(candidates))
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

    def _log_hit(self, key, agent_id, confidence, n_candidates):
        self._hit_log.append({
            "hit": True, "key": key, "agent_id": agent_id,
            "confidence": confidence, "n_candidates": n_candidates,
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

    # past_key_values may be DynamicCache — convert to tuple
    from transformers import DynamicCache
    pkv = outputs.past_key_values
    if isinstance(pkv, DynamicCache):
        pkv = pkv.to_legacy_cache()

    # last hidden state for embedding
    last_hidden = outputs.hidden_states[-1]  # (1, seq, hidden)
    return pkv, last_hidden
