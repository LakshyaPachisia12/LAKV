"""
LAKV Module: causal audit of relayed KV cache.

Phase 1 of the research-extensions plan (lakv_research_extensions_prompt.md).
Paper: "When Does Latent Communication Pay? A Causal Audit of Relayed KV
Caches in Multi-Agent LLMs" (arXiv 2608.04893) — summarized via web search,
not read from the primary source. Treat the mechanism below as this
project's own design inspired by that idea, not a verified reproduction of
its exact method.

Answers: is a config's accuracy coming from the REAL relayed KV content, or
just from the receiving agent having SOME non-empty cache to attend over?
Substitutes the KV handed from one agent to the next with:
  - zeroed        : every K/V tensor zeroed out
  - random        : per-tensor moment-matched noise (same mean/std, no
                    real structure)
  - mismatched    : another, unrelated HELD-OUT question's real KV at the
                    same hop (same shapes as closely as possible, wrong
                    content) — see KVAuditPool below for how "held-out" is
                    enforced.

If the real config clearly beats all three substitutes, that's evidence the
relay is transmitting real reasoning content. If a substitute comes close,
that's a real, reportable finding that qualifies the "KV relay works" story
— see the research-extensions brief's Phase 1 section for how to interpret
either outcome.
"""

from __future__ import annotations

import random as _random_module
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


KVTuple = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]  # per-layer (K, V)


# ─── stateless substitutions ──────────────────────────────────────────────────

def zero_kv(real_kv: KVTuple) -> KVTuple:
    """Zero out every K/V tensor, preserving shape/dtype/device."""
    return tuple((torch.zeros_like(k), torch.zeros_like(v)) for k, v in real_kv)


def moment_matched_random_kv(real_kv: KVTuple, generator: Optional[torch.Generator] = None) -> KVTuple:
    """Replace each K/V tensor with noise matching ITS OWN per-tensor
    mean/std — not a single global mean/std shared across all layers/tensors,
    since KV magnitude varies a lot layer-to-layer and we want "same
    distribution shape, wrong content" per tensor, not a coarser single
    distribution that would itself be an unrealistic substitute.
    """
    out = []
    for k, v in real_kv:
        k_f, v_f = k.float(), v.float()
        k_mean, k_std = k_f.mean().item(), k_f.std().clamp_min(1e-6).item()
        v_mean, v_std = v_f.mean().item(), v_f.std().clamp_min(1e-6).item()
        rk = torch.normal(k_mean, k_std, size=k.shape, generator=generator,
                           device=k.device, dtype=torch.float32).to(k.dtype)
        rv = torch.normal(v_mean, v_std, size=v.shape, generator=generator,
                           device=v.device, dtype=torch.float32).to(v.dtype)
        out.append((rk, rv))
    return tuple(out)


def _match_seq_len(donor: torch.Tensor, target_shape: torch.Size, seq_dim: int = 2) -> Tuple[torch.Tensor, bool]:
    """Truncate or zero-pad donor along seq_dim to match target_shape there.

    Every other dim (batch, n_kv_heads, head_dim) comes from the same model
    and is assumed already equal — only seq_len (how many tokens the donor
    question's hop happened to produce) can legitimately differ between two
    different questions.
    """
    if donor.shape == target_shape:
        return donor, False
    donor_len = donor.shape[seq_dim]
    target_len = target_shape[seq_dim]
    if donor_len > target_len:
        idx = [slice(None)] * donor.dim()
        idx[seq_dim] = slice(0, target_len)
        return donor[tuple(idx)].contiguous(), True
    if donor_len < target_len:
        pad_shape = list(donor.shape)
        pad_shape[seq_dim] = target_len - donor_len
        pad = torch.zeros(pad_shape, dtype=donor.dtype, device=donor.device)
        return torch.cat([donor, pad], dim=seq_dim), True
    return donor, False


# ─── mismatched-example pool ──────────────────────────────────────────────────

@dataclass
class _PoolEntry:
    question_key: str
    kv: KVTuple  # stored on CPU
    question_text: str = ""  # full question text, for post-hoc "did the wrong
    # answer bleed content from the DONOR question" analysis — question_key
    # alone is a hash, not reversible back to readable text.


class KVAuditPool:
    """Holds real, per-hop relayed KV captured from a small set of HELD-OUT
    questions — questions that are never among the ones actually being
    scored in an audit run, so "mismatched" mode can never leak a question's
    own content back to itself.

    Build this via a separate pre-pass (see Evaluator._build_audit_pool in
    evaluator.py) BEFORE running any scored `mismatched` config, then treat
    it as read-only during the scored run. Keyed by agent_idx (the
    RECEIVING agent's index in the chain) since hop 1 (Verifier receiving
    from Reasoner) and hop 2 (Finalizer receiving from Verifier) draw from
    different content distributions.
    """

    def __init__(self, max_size_per_agent: int = 20):
        self.max_size_per_agent = max_size_per_agent
        self._pools: Dict[int, List[_PoolEntry]] = {}

    def add(self, agent_idx: int, question_key: str, kv: KVTuple, question_text: str = "") -> None:
        pool = self._pools.setdefault(agent_idx, [])
        if any(e.question_key == question_key for e in pool):
            return
        # Stored on CPU so a 20-example pool across both hops doesn't sit on
        # the GPU the whole time competing with the model's own memory —
        # moved back to the target device only at sample() time.
        cpu_kv = tuple((k.detach().to("cpu"), v.detach().to("cpu")) for k, v in kv)
        pool.append(_PoolEntry(question_key, cpu_kv, question_text))
        if len(pool) > self.max_size_per_agent:
            pool.pop(0)

    def size(self, agent_idx: int) -> int:
        return len(self._pools.get(agent_idx, []))

    def sample(
        self, agent_idx: int, exclude_question_key: str, target_shape: torch.Size,
        device, rng: Optional[_random_module.Random] = None,
    ) -> Tuple[Optional[KVTuple], bool, str]:
        """Return (substitute_kv, was_shape_resized, donor_question_text).
        substitute_kv is None only if this agent_idx's pool has no eligible
        entries at all — this should never happen if the pool was pre-built
        correctly (it means the pool wasn't populated for this agent_idx
        before the audit run started); callers should treat None as a setup
        error, not a per-sample condition to paper over silently.

        donor_question_text (empty string if substitute_kv is None) records
        WHICH held-out question's content was actually substituted in — not
        just that a substitution happened. Exists so a "mismatched" run's
        wrong answers can be checked for whether they reflect the donor
        question's topic bleeding through, rather than only measuring the
        accuracy drop — a corrupted hop silently steering the final answer
        toward unrelated content would be a materially different, more
        concerning finding than an accuracy drop alone.
        """
        pool = self._pools.get(agent_idx, [])
        eligible = [e for e in pool if e.question_key != exclude_question_key]
        if not eligible:
            return None, False, ""

        rng = rng or _random_module
        entry = rng.choice(eligible)
        resized = False
        out = []
        for k, v in entry.kv:
            k_dev, v_dev = k.to(device), v.to(device)
            k_matched, k_resized = _match_seq_len(k_dev, target_shape)
            v_matched, v_resized = _match_seq_len(v_dev, target_shape)
            resized = resized or k_resized or v_resized
            out.append((k_matched, v_matched))
        return tuple(out), resized, entry.question_text


# ─── dispatch ─────────────────────────────────────────────────────────────────

def apply_causal_audit(
    mode: str,
    real_kv: KVTuple,
    agent_idx: int,
    question_key: str,
    pool: Optional[KVAuditPool] = None,
    generator: Optional[torch.Generator] = None,
    rng: Optional[_random_module.Random] = None,
) -> Tuple[KVTuple, dict]:
    """Dispatch to the right substitution for PipelineConfig.causal_audit_mode.

    Returns (kv_to_inject, audit_log). audit_log always has a "mode" key;
    "mismatched" additionally reports whether the donor had to be
    truncated/padded to match this question's own KV shape.
    """
    if mode == "none":
        return real_kv, {"mode": "none"}

    if mode == "zeroed":
        return zero_kv(real_kv), {"mode": "zeroed"}

    if mode == "random":
        return moment_matched_random_kv(real_kv, generator=generator), {"mode": "random"}

    if mode == "mismatched":
        if pool is None:
            raise ValueError("causal_audit_mode='mismatched' requires a KVAuditPool")
        target_shape = real_kv[0][0].shape
        device = real_kv[0][0].device
        substitute, was_resized, donor_question = pool.sample(
            agent_idx, question_key, target_shape, device, rng=rng)
        if substitute is None:
            # The pool for this agent_idx was never populated before this
            # audit run started — a setup bug (wrong pool passed in, or the
            # pre-pass never ran for this agent_idx), not a legitimate
            # per-sample outcome. Fail loudly rather than silently falling
            # back to zeros, which would quietly mix audit conditions inside
            # one supposedly-uniform "mismatched" run.
            raise RuntimeError(
                f"KVAuditPool has no eligible entries for agent_idx={agent_idx} "
                f"(question_key={question_key!r}) — was the pool pre-built for this "
                f"agent before running the audit config?"
            )
        return substitute, {
            "mode": "mismatched", "shape_resized": was_resized,
            "donor_question": donor_question,
        }

    raise ValueError(f"Unknown causal_audit_mode: {mode!r}")
