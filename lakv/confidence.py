"""
Process-level confidence signature for causal-audit runs -- a topology-
agnostic complement to the RLM+KV-specific behavioral signature (timeout
rate / turn count / delegation calls; see docs/PROGRESS_REPORT.md's
2026-09-17 entry and CLAUDE.md finding 18's UPDATE 2026-09-17).

Motivation (2026-09-18 research audit, comparing this project against
arXiv 2608.04893 -- "When Does Latent Communication Pay?", which runs the
same zeroed/random/mismatched causal audit across three PUBLISHED systems
and model checkpoints): several of this project's own real-vs-mismatched
legs are underpowered on exact-match/F1 alone at n=50 (see CLAUDE.md
"Not yet statistically established"). EM/F1 is binary/discrete per
example; a confidence signal is continuous, so it can resolve differences
those metrics can't -- the same principle that let the RLM+KV behavioral
signature resolve a leg accuracy couldn't (mismatched vs. random,
p<0.0001 on turn-count/timeout-rate vs. p=0.50 on EM). This generalizes
that idea into a signal every topology in this project can produce from
its own final-answer generation logits, not one specific to agentic
multi-turn sessions -- unlike turn count / delegation calls, which only
exist for a dynamic REPL loop, mean top-1 probability and entropy are
computable from a sequential pipeline's Finalizer, a fan-in aggregator,
and an RLM+KV session's answer-producing turn alike.

Deliberately computed from whatever logits each pipeline already has at
its final-answer generation step (raw per-token logits, or HF's
`output_scores` -- both are post-logits-processor scores, e.g. after
repetition_penalty, consistent with what `generate()` itself used to
sample). This is a relative, within-run comparison between audit
conditions (real vs. zeroed vs. random vs. mismatched) computed on the
SAME model/prompt structure/decoding settings -- any shared bias from
logits processors applies equally across conditions and does not
threaten the causal comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch


@dataclass
class ConfidenceStats:
    mean_top1_prob: float
    mean_entropy: float
    n_tokens: int


def stats_from_logits(logits_list: Sequence[torch.Tensor]) -> Optional[ConfidenceStats]:
    """logits_list: one (1, vocab) or (vocab,) tensor per generated token,
    in generation order. Returns None for an empty sequence (e.g. the
    very first sampled token was EOS, so no real answer content exists to
    measure confidence over)."""
    if not logits_list:
        return None
    top1_probs: List[float] = []
    entropies: List[float] = []
    for logits in logits_list:
        probs = torch.softmax(logits.reshape(-1).float(), dim=-1)
        top1_probs.append(probs.max().item())
        # clamp_min avoids log(0) for the numerically-impossible-but-not-
        # worth-crashing-over case of exactly-zero probability mass.
        entropies.append(-(probs * probs.clamp_min(1e-12).log()).sum().item())
    n = len(top1_probs)
    return ConfidenceStats(
        mean_top1_prob=sum(top1_probs) / n,
        mean_entropy=sum(entropies) / n,
        n_tokens=n,
    )
