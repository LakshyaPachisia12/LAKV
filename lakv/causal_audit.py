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

import json
import random as _random_module
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from lakv.cache_corruption import (apply_corruption_mode, is_corruption_mode,
                                   parse_corruption_mode)


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


def describe_seq_len_handling(donor_len: int, target_len: int) -> dict:
    """How _match_seq_len treats a donor of donor_len for a target of target_len.

    Pure bookkeeping; changes no behavior. handling is "exact" (same length),
    "truncated" (donor longer: its tail is cut) or "zero_padded" (donor shorter:
    zeros are appended after the donor's real positions). pad_fraction is the
    share of the injected cache's target_len positions that are all-zero
    padding (0.0 unless zero_padded); truncated_fraction is the share of the
    donor's own positions that were discarded.
    """
    if donor_len == target_len:
        handling, pad, cut = "exact", 0, 0
    elif donor_len > target_len:
        handling, pad, cut = "truncated", 0, donor_len - target_len
    else:
        handling, pad, cut = "zero_padded", target_len - donor_len, 0
    return {
        "handling": handling,
        "donor_cache_len": int(donor_len),
        "target_cache_len": int(target_len),
        "pad_positions": int(pad),
        "pad_fraction": (pad / target_len) if target_len else 0.0,
        "truncated_positions": int(cut),
        "truncated_fraction": (cut / donor_len) if donor_len else 0.0,
    }


# ─── mismatched-example pool ──────────────────────────────────────────────────

@dataclass
class _PoolEntry:
    question_key: str
    kv: KVTuple  # stored on CPU
    question_text: str = ""  # full question text, for post-hoc "did the wrong
    # answer bleed content from the DONOR question" analysis — question_key
    # alone is a hash, not reversible back to readable text.
    order: int = -1  # insertion order within this agent's pool (0 = first held-out
    # question recorded). Lets a per-sample log say WHICH donor was used.


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
        self._next_order: Dict[int, int] = {}
        # Optional: question_key -> offset into the held-out slice
        # (dataset index = n_scored + offset). Filled by the evaluator after
        # the pre-pass; used only for logging.
        self.donor_heldout_offsets: Dict[str, int] = {}

    def add(self, agent_idx: int, question_key: str, kv: KVTuple, question_text: str = "") -> None:
        pool = self._pools.setdefault(agent_idx, [])
        if any(e.question_key == question_key for e in pool):
            return
        # Stored on CPU so a 20-example pool across both hops doesn't sit on
        # the GPU the whole time competing with the model's own memory —
        # moved back to the target device only at sample() time.
        cpu_kv = tuple((k.detach().to("cpu"), v.detach().to("cpu")) for k, v in kv)
        order = self._next_order.get(agent_idx, 0)
        self._next_order[agent_idx] = order + 1
        pool.append(_PoolEntry(question_key, cpu_kv, question_text, order))
        if len(pool) > self.max_size_per_agent:
            pool.pop(0)

    def size(self, agent_idx: int) -> int:
        return len(self._pools.get(agent_idx, []))

    def sample_with_info(
        self, agent_idx: int, exclude_question_key: str, target_shape: torch.Size,
        device, rng: Optional[_random_module.Random] = None,
    ) -> Tuple[Optional[KVTuple], bool, str, dict]:
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
            return None, False, "", {}

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
        # Bookkeeping only (no effect on the substitute): every layer's K/V
        # shares one seq length, so layer 0's K describes the whole cache.
        info = describe_seq_len_handling(entry.kv[0][0].shape[2], target_shape[2])
        info.update({
            "donor_question_key": entry.question_key,
            "donor_pool_order": entry.order,
            "donor_heldout_offset": self.donor_heldout_offsets.get(entry.question_key),
            "pool_size": len(pool),
            "n_eligible_donors": len(eligible),
        })
        return tuple(out), resized, entry.question_text, info

    def sample(
        self, agent_idx: int, exclude_question_key: str, target_shape: torch.Size,
        device, rng: Optional[_random_module.Random] = None,
    ) -> Tuple[Optional[KVTuple], bool, str]:
        """Original 3-tuple interface; see sample_with_info for the full record."""
        substitute, resized, donor_text, _info = self.sample_with_info(
            agent_idx, exclude_question_key, target_shape, device, rng=rng)
        return substitute, resized, donor_text




# ─── matched-donor audit modes (UNTESTED ON GPU; CPU unit tests only) ───────
#
# See docs/matched_donor_design.md. Four modes beyond zeroed/random/mismatched/
# dropped, all handled through a "donor source" object instead of KVAuditPool:
#   length_matched        other question's real cache, length within 5% of the
#                         target, injected UNMODIFIED (never padded/truncated)
#   matched_passages      same question, sampled alternate reasoning trace
#   matched_wrong_answer  same question, trace defending a distractor answer
#   self_replay           the question's OWN trace rebuilt from token ids
#                         (validation of the rebuild machinery)
DONOR_SOURCE_MODES = ("length_matched", "matched_passages", "matched_wrong_answer", "self_replay")
LENGTH_MATCH_TOLERANCE = 0.05


class DonorUnavailable(Exception):
    """A donor required by an audit mode does not exist for this sample.

    The evaluator records the sample as skipped (not scored, not counted in
    the arm's accuracy denominator). `log` is merged into the sample's
    audit_logs so the reason is visible.
    """

    def __init__(self, reason: str, log: Optional[dict] = None):
        super().__init__(reason)
        self.reason = reason
        self.log = dict(log or {}, skipped=True, skip_reason=reason)


class NoMatchedDonor(DonorUnavailable):
    """length_matched: no pooled donor within the length tolerance."""


class DiskKVAuditPool:
    """Disk-backed pool of held-out questions' real relayed caches.

    Used by length_matched. Each donor cache is torch.saved to its own file;
    only a small index (key, text, length, path) is held in RAM, and exactly
    ONE donor file is loaded per substitution. Same add()/size() interface as
    KVAuditPool, so it can be passed as the pipeline's record_audit_pool.

    Disk (not RAM) cost per held-out question is about 260 MB for
    Qwen2.5-7B (see docs/matched_donor_design.md); RAM is one donor at a time.
    """

    INDEX_NAME = "index.json"

    def __init__(self, directory):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._index: Dict[int, List[dict]] = {}
        self._next_order: Dict[int, int] = {}
        self.complete = False

    # ---- persistence -------------------------------------------------
    @classmethod
    def open(cls, directory) -> "DiskKVAuditPool":
        """Re-open an existing pool (e.g. on --resume) from its index file."""
        pool = cls(directory)
        idx_path = pool.dir / cls.INDEX_NAME
        if idx_path.exists():
            data = json.loads(idx_path.read_text(encoding="utf-8"))
            pool._index = {int(k): v for k, v in data["index"].items()}
            pool._next_order = {a: max((e["order"] for e in v), default=-1) + 1
                                for a, v in pool._index.items()}
            pool.complete = bool(data.get("complete", False))
        return pool

    def _write_index(self) -> None:
        (self.dir / self.INDEX_NAME).write_text(
            json.dumps({"complete": self.complete, "index": self._index}), encoding="utf-8")

    def mark_complete(self) -> None:
        self.complete = True
        self._write_index()

    # ---- building ----------------------------------------------------
    def add(self, agent_idx: int, question_key: str, kv: KVTuple, question_text: str = "") -> None:
        entries = self._index.setdefault(agent_idx, [])
        if any(e["key"] == question_key for e in entries):
            return
        order = self._next_order.get(agent_idx, 0)
        self._next_order[agent_idx] = order + 1
        rel = f"hop{agent_idx}/{order:04d}_{question_key}.pt"
        path = self.dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        cpu_kv = tuple((k.detach().to("cpu").contiguous(), v.detach().to("cpu").contiguous())
                       for k, v in kv)
        torch.save({"kv": cpu_kv}, path)
        entries.append({"key": question_key, "text": question_text, "len": int(kv[0][0].shape[2]),
                        "order": order, "path": rel})
        self._write_index()

    def size(self, agent_idx: int) -> int:
        return len(self._index.get(agent_idx, []))

    # ---- selection ---------------------------------------------------
    def eligible(self, agent_idx: int, exclude_key: str, target_len: int,
                 tolerance: float = LENGTH_MATCH_TOLERANCE) -> List[dict]:
        return [e for e in self._index.get(agent_idx, [])
                if e["key"] != exclude_key and abs(e["len"] - target_len) <= tolerance * target_len]

    def select_length_matched(self, agent_idx: int, exclude_key: str, target_len: int, device,
                              rng: Optional[_random_module.Random] = None,
                              tolerance: float = LENGTH_MATCH_TOLERANCE) -> Tuple[KVTuple, dict]:
        """Choose uniformly (seeded rng) among donors within tolerance of
        target_len, load ONLY that donor, and return it unmodified."""
        pool = self._index.get(agent_idx, [])
        elig = self.eligible(agent_idx, exclude_key, target_len, tolerance)
        if not elig:
            others = [e["len"] for e in pool if e["key"] != exclude_key]
            nearest = min(others, key=lambda L: abs(L - target_len)) if others else None
            raise NoMatchedDonor("no donor within %.0f%% of target length" % (tolerance * 100), {
                "mode": "length_matched", "agent_idx": agent_idx, "target_cache_len": int(target_len),
                "pool_size": len(pool), "nearest_donor_cache_len": nearest, "tolerance": tolerance})
        rng = rng or _random_module
        entry = rng.choice(elig)
        blob = torch.load(self.dir / entry["path"], map_location=device, weights_only=True)
        kv = blob["kv"]
        donor_len = int(kv[0][0].shape[2])
        extra = {"donor_prompt_len": entry["prompt_len"]} if "prompt_len" in entry else {}
        return kv, {
            **extra,
            "agent_idx": agent_idx, "skipped": False,
            "donor_question_key": entry["key"], "donor_question": entry["text"],
            "donor_pool_order": entry["order"], "donor_cache_len": donor_len,
            "target_cache_len": int(target_len), "len_ratio": donor_len / target_len,
            "handling": "exact" if donor_len == target_len else "unmodified_within_tolerance",
            "pad_positions": 0, "truncated_positions": 0, "tolerance": tolerance,
            "n_eligible": len(elig), "pool_size": len(pool),
        }


class DiskAnswerRegionPool(DiskKVAuditPool):
    """Disk pool of delegation CHILD ANSWER REGIONS (RLM+KV length-matched donors).

    The root only ever receives the child's answer region (the positions after its
    prompt: see RLMKVSession.splice_child_kv), so that is what is stored and what
    length matching is done on: an entry's "len" is the ANSWER length, and the donor
    is later spliced by its own prompt length (RoPE start), never padded/truncated.
    One held-out session can contribute several entries (one per child call); entries
    are keyed "<question_key>#<n>" and remember their question ("qkey") so a donor
    never comes from the scored question itself.
    Design: LAKV_runs/fanin_length_matched_design.md (UNTESTED ON GPU).
    """

    records_answer_regions = True

    def add(self, agent_idx: int, question_key: str, kv: KVTuple, question_text: str = "",
            prompt_len: int = 0) -> None:
        total = int(kv[0][0].shape[2])
        if prompt_len <= 0 or prompt_len >= total:
            return  # no answer region to store
        entries = self._index.setdefault(agent_idx, [])
        order = self._next_order.get(agent_idx, 0)
        self._next_order[agent_idx] = order + 1
        rel = f"hop{agent_idx}/{order:04d}_{question_key}.pt"
        path = self.dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        region = tuple((k[:, :, prompt_len:, :].detach().to("cpu").contiguous(),
                        v[:, :, prompt_len:, :].detach().to("cpu").contiguous()) for k, v in kv)
        torch.save({"kv": region}, path)
        entries.append({"key": f"{question_key}#{order}", "qkey": question_key, "text": question_text,
                        "len": total - prompt_len, "prompt_len": int(prompt_len), "order": order, "path": rel})
        self._write_index()

    def eligible(self, agent_idx: int, exclude_key: str, target_len: int,
                 tolerance: float = LENGTH_MATCH_TOLERANCE) -> List[dict]:
        return [e for e in self._index.get(agent_idx, [])
                if e.get("qkey", e["key"]) != exclude_key and abs(e["len"] - target_len) <= tolerance * target_len]


# ─── dispatch ─────────────────────────────────────────────────────────────────

def apply_causal_audit(
    mode: str,
    real_kv: KVTuple,
    agent_idx: int,
    question_key: str,
    pool: Optional[KVAuditPool] = None,
    generator: Optional[torch.Generator] = None,
    rng: Optional[_random_module.Random] = None,
    donor_source=None,
    context: Optional[dict] = None,
    audit_seed: int = 0,
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

    if is_corruption_mode(mode):
        # Dose-response corruption (lakv/cache_corruption.py): operator applied to
        # the REAL cache ("real_*") or to a clean length-matched wrong cache ("lm_*").
        target, _op, _pct = parse_corruption_mode(mode)
        base_kv, donor_log = real_kv, None
        if target == "lm":
            if donor_source is None:
                raise ValueError(f"causal_audit_mode={mode!r} requires a length_matched donor_source")
            base_kv, donor_log = donor_source.fetch(agent_idx=agent_idx, question_key=question_key,
                                                    real_kv=real_kv, rng=rng, **(context or {}))
        out, info = apply_corruption_mode(mode, base_kv, question_key, agent_idx, audit_seed=audit_seed)
        log = {"mode": mode, **info}
        if donor_log is not None:
            log["donor"] = donor_log
        return out, log

    if mode in DONOR_SOURCE_MODES:
        if donor_source is None:
            raise ValueError(f"causal_audit_mode={mode!r} requires a donor_source "
                             f"(built by the evaluator's pre-pass; see docs/matched_donor_design.md)")
        substitute, log = donor_source.fetch(agent_idx=agent_idx, question_key=question_key,
                                             real_kv=real_kv, rng=rng, **(context or {}))
        return substitute, {"mode": mode, **log}
    if mode == "mismatched":
        if pool is None:
            raise ValueError("causal_audit_mode='mismatched' requires a KVAuditPool")
        target_shape = real_kv[0][0].shape
        device = real_kv[0][0].device
        substitute, was_resized, donor_question, handling_info = pool.sample_with_info(
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
            **handling_info,
        }

    raise ValueError(f"Unknown causal_audit_mode: {mode!r}")
