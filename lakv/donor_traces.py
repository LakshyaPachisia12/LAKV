"""Same-question donors for the matched-donor audit modes.

*** UNTESTED ON GPU. CPU unit tests only (tests/test_matched_donors.py). ***
Design: docs/matched_donor_design.md.

What is stored for a same-question donor: ONLY generated token ids (plus a
few scalars: seeds, cache_gen_len, a validated distractor string, flags).
Never a KV cache and never any prompt or instruction text. A donor's cache is
REBUILT at use time by a deterministic prefill of

    STANDARD prompt ids (the exact prompt the scored chain gives that agent)
  + the stored generated ids

so even for matched_wrong_answer the "defend X" instruction, which exists
only while the donor is being GENERATED, is absent from every cache and from
every scored-chain prompt.

Modes (all inject the donor UNMODIFIED: no padding, no truncation):
  matched_passages      same question; trace re-generated with sampling, seeded
  matched_wrong_answer  same question; trace generated while prompted to defend
                        a validated distractor answer (greedy)
  self_replay           validation: the scored chain's OWN trace, rebuilt the
                        same way; should reproduce the real arm
"""
from __future__ import annotations

import json
import random as _random_module
import zlib
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from lakv.anchor_table import question_key
from lakv.causal_audit import (DiskKVAuditPool, DonorUnavailable, LENGTH_MATCH_TOLERANCE)
from lakv.qa_scoring import exact_match_score, f1_score, normalize_answer

# Generation-time only. These strings are never part of a stored trace, a
# rebuilt cache, or a scored-chain prompt.
DEFEND_SUFFIX_TEMPLATE = (
    "\n\nAnother reader concluded that the answer is {x}. Build the strongest case that {x} is "
    "correct using the passages, and state it as your answer."
)
DISTRACTOR_SYSTEM = "You write plausible but incorrect answers to questions, for evaluation purposes."
DISTRACTOR_ASK = ("\n\nGive a plausible but INCORRECT short answer to the question (a few words). "
                  "Reply with only that answer.")
DISTRACTOR_MAX_F1 = 0.2
TRACE_VERSION = 1


# ─── seeding ────────────────────────────────────────────────────────────────

def stable_seed(*parts) -> int:
    """Process-independent seed from arbitrary parts (zlib.crc32; Python's
    built-in hash() is randomized per process and must not be used)."""
    return zlib.crc32("|".join(str(p) for p in parts).encode("utf-8")) & 0x7FFFFFFF


def donor_dir_name(mode: str, base_seed: int) -> str:
    """Directory for a mode's stored donors. matched_passages traces depend on
    the sampling seed, so the seed is part of the path (a different
    --audit_donor_seed must never silently reuse another seed's traces);
    matched_wrong_answer is greedy and seed-independent."""
    return f"{mode}_seed{base_seed}" if mode == "matched_passages" else mode


# ─── trace store (token ids only) ───────────────────────────────────────────

class TraceStore:
    """One small JSON file per question key: generated token ids + scalars."""

    def __init__(self, directory):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def save(self, key: str, record: dict) -> None:
        record = dict(record, version=TRACE_VERSION, question_key=key)
        self._path(key).write_text(json.dumps(record), encoding="utf-8")

    def load(self, key: str) -> Optional[dict]:
        p = self._path(key)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ─── cache rebuild ──────────────────────────────────────────────────────────

def rebuild_cache(model, segments: Sequence[Sequence[int]], device) -> tuple:
    """KV cache for a chain of token-id segments, by deterministic prefill.

    Segment i is forwarded with the cache of segments 0..i-1 as past and
    position_ids continuing from the cache length, exactly how the chain
    builds a hop's cache (see LAKVPipeline._generate_intermediate_with_hidden).
    Uses the bare decoder (model.model) when present, so the full-vocab
    logits for thousands of positions are never materialized.
    """
    from lakv.pipeline import LAKVPipeline
    base = getattr(model, "model", model)
    cache, pos = None, 0
    with torch.no_grad():
        for seg in segments:
            ids = torch.tensor([list(seg)], dtype=torch.long, device=device)
            kwargs = {"input_ids": ids, "use_cache": True}
            if cache is not None:
                kwargs["past_key_values"] = cache
                kwargs["position_ids"] = torch.arange(pos, pos + ids.shape[1], device=device).unsqueeze(0)
                kwargs["attention_mask"] = torch.ones((1, pos + ids.shape[1]), dtype=torch.long, device=device)
            out = base(**kwargs)
            cache = out.past_key_values
            pos += ids.shape[1]
    return LAKVPipeline._to_tuple(cache)


def trace_segments(prompt_ids_per_hop: Sequence[Sequence[int]], gen_ids_per_hop: Sequence[Sequence[int]],
                   cache_gen_len_per_hop: Sequence[int], upto_hop: int) -> List[List[int]]:
    """prompt + the generated ids that the real chain actually cached, per hop.

    The real chain's cache holds the generated ids MINUS the final token on the
    generate() path (HF generate never forwards its last token), which is why
    cache_gen_len is stored per hop instead of assuming len(gen_ids).
    """
    segs = []
    for h in range(upto_hop + 1):
        c = int(cache_gen_len_per_hop[h])
        segs.append(list(prompt_ids_per_hop[h]) + list(gen_ids_per_hop[h][:c]))
    return segs


def kv_diff_stats(a: tuple, b: tuple) -> dict:
    """Numerical comparison of two caches (self-replay validation)."""
    la, lb = int(a[0][0].shape[2]), int(b[0][0].shape[2])
    if la != lb or len(a) != len(b):
        return {"comparable": False, "len_a": la, "len_b": lb}
    max_abs, min_cos, max_rel = 0.0, 1.0, 0.0
    for (ka, va), (kb, vb) in zip(a, b):
        for x, y in ((ka, kb), (va, vb)):
            x, y = x.float().flatten(), y.float().flatten().to(x.device)
            d = (x - y)
            max_abs = max(max_abs, float(d.abs().max()))
            nx, ny = x.norm(), y.norm()
            cos = float((x @ y) / (nx * ny)) if float(nx * ny) > 0 else 1.0
            min_cos = min(min_cos, cos)
            max_rel = max(max_rel, float(d.norm() / nx) if float(nx) > 0 else 0.0)
    return {"comparable": True, "len": la, "max_abs_diff": max_abs, "min_cosine": min_cos, "max_rel_l2": max_rel}


# ─── donor generation (pre-pass) ────────────────────────────────────────────

@contextmanager
def decoding(pipeline, do_sample: bool):
    """Temporarily set do_sample on the pipeline's generation_kwargs."""
    old = pipeline.config.generation_kwargs
    pipeline.config.generation_kwargs = dict(old, do_sample=do_sample)
    try:
        yield
    finally:
        pipeline.config.generation_kwargs = old


def _seed_all(seed: int) -> None:
    _random_module.seed(seed)
    torch.manual_seed(seed)  # seeds CPU and all CUDA devices


def generate_chain_ids(pipeline, question: str, *, system_suffix: str = "", do_sample: bool,
                       seeds: Optional[Sequence[int]] = None, hops: int = 2) -> Tuple[List[List[int]], List[int]]:
    """Run the Reasoner and Verifier hops once and return (gen_ids per hop,
    cache_gen_len per hop). Only the two relayed hops are produced (the
    Finalizer's cache is never relayed)."""
    gen_ids, cache_len = [], []
    kv = None
    with decoding(pipeline, do_sample):
        for hop in range(hops):
            ids = pipeline._agent_input_ids(question, hop, system_suffix=system_suffix)
            if seeds is not None:
                _seed_all(seeds[hop])
            raw_kv, _hidden, _text = pipeline._generate_intermediate_with_hidden(
                ids, kv, max_new=pipeline.config.intermediate_max_new_tokens,
                position_offset=0, receiver_prompt_len=ids.shape[1])
            injected = kv[0][0].shape[2] if kv is not None else 0
            gen = list(pipeline._last_generated_ids)
            c = int(raw_kv[0][0].shape[2] - injected - ids.shape[1])
            assert 0 <= c <= len(gen), f"cached generated length {c} not within generated ids ({len(gen)})"
            gen_ids.append(gen)
            cache_len.append(c)
            kv = raw_kv
    del kv
    return gen_ids, cache_len


def propose_distractor(pipeline, question: str) -> str:
    """Greedy, model-generated plausible-but-wrong short answer."""
    tok = pipeline.tokenizer
    messages = [{"role": "system", "content": DISTRACTOR_SYSTEM},
                {"role": "user", "content": question + DISTRACTOR_ASK}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(pipeline.device)
    with torch.no_grad():
        out = pipeline.model.generate(input_ids=ids, max_new_tokens=24, do_sample=False)
    reply = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
    first = reply.splitlines()[0] if reply else ""
    return first.strip().strip("\"'.")


def distractor_ok(x: str, gold: str) -> bool:
    """A usable distractor is non-empty, not a match for gold, and nearly
    disjoint from it (F1 < 0.2)."""
    if not x or not x.strip():
        return False
    return (not exact_match_score(x, gold)) and f1_score(x, gold) < DISTRACTOR_MAX_F1


def _contains(text: str, x: str) -> bool:
    return bool(x) and normalize_answer(x) in normalize_answer(text)


def generate_donor_traces(pipeline, samples: List[dict], mode: str, store: TraceStore,
                          base_seed: int, log=print) -> dict:
    """Pre-pass over the SCORED questions: write one token-id trace per
    question. Resumable (existing traces are kept). Returns counts."""
    assert mode in ("matched_passages", "matched_wrong_answer"), mode
    tok = pipeline.tokenizer
    counts = {"written": 0, "reused": 0, "unavailable": 0}
    for i, s in enumerate(samples):
        q = s["question"]
        key = question_key(q)
        if store.exists(key):
            counts["reused"] += 1
            continue
        if mode == "matched_passages":
            seeds = [stable_seed(base_seed, mode, key, h) for h in (0, 1)]
            gen_ids, cache_len = generate_chain_ids(pipeline, q, do_sample=True, seeds=seeds)
            # Vacuity check: does the sampled hop-0 trace differ from the greedy one?
            greedy_ids, _ = generate_chain_ids(pipeline, q, do_sample=False, hops=1)
            g0, s0 = greedy_ids[0], gen_ids[0]
            common = 0
            for a, b in zip(g0, s0):
                if a != b:
                    break
                common += 1
            rec = {"mode": mode, "seeds": seeds, "gen_ids": gen_ids, "cache_gen_len": cache_len,
                   "distractor": None,
                   "hop0_differs_from_greedy": g0 != s0,
                   "hop0_common_prefix_frac": common / max(len(g0), len(s0), 1)}
        else:
            gold = str(s["answer"]).strip()
            x = propose_distractor(pipeline, q)
            if not distractor_ok(x, gold):
                store.save(key, {"mode": mode, "unavailable": "distractor_failed", "distractor": x})
                counts["unavailable"] += 1
                log(f"  [donor] q{i}: distractor rejected ({x!r} vs gold {gold!r})")
                continue
            suffix = DEFEND_SUFFIX_TEMPLATE.format(x=x)
            gen_ids, cache_len = generate_chain_ids(pipeline, q, system_suffix=suffix, do_sample=False)
            texts = [tok.decode(g, skip_special_tokens=True) for g in gen_ids]
            rec = {"mode": mode, "seeds": None, "gen_ids": gen_ids, "cache_gen_len": cache_len,
                   "distractor": x, "defends_distractor": [_contains(t, x) for t in texts]}
        store.save(key, rec)
        counts["written"] += 1
        if (i + 1) % 10 == 0:
            log(f"  [donor] {mode}: {i + 1}/{len(samples)} traces")
    return counts


# ─── donor sources (called through apply_causal_audit) ──────────────────────

def _len(kv: tuple) -> int:
    return int(kv[0][0].shape[2])


class LengthMatchedSource:
    """length_matched: disk pool, one donor loaded per substitution, unmodified."""

    def __init__(self, pool: DiskKVAuditPool, tolerance: float = LENGTH_MATCH_TOLERANCE):
        self.pool, self.tolerance = pool, tolerance

    def fetch(self, agent_idx, question_key, real_kv, rng=None, **_ctx):
        return self.pool.select_length_matched(
            agent_idx, question_key, _len(real_kv), real_kv[0][0].device, rng=rng, tolerance=self.tolerance)


class TraceReplaySource:
    """matched_passages / matched_wrong_answer: rebuild the stored same-question
    trace with the STANDARD prompts and inject it unmodified."""

    def __init__(self, store: TraceStore, mode: str):
        assert mode in ("matched_passages", "matched_wrong_answer"), mode
        self.store, self.mode = store, mode

    def fetch(self, agent_idx, question_key, real_kv, rng=None, pipeline=None, question=None, **_ctx):
        rec = self.store.load(question_key)
        if rec is None:
            raise DonorUnavailable("no_trace_stored", {"mode": self.mode, "agent_idx": agent_idx})
        if rec.get("unavailable"):
            raise DonorUnavailable(rec["unavailable"], {"mode": self.mode, "agent_idx": agent_idx,
                                                        "distractor": rec.get("distractor")})
        hops = agent_idx  # agent k receives the cache of hops 0..k-1
        prompt_ids = [pipeline._agent_input_ids(question, h)[0].tolist() for h in range(hops)]  # STANDARD prompts
        segs = trace_segments(prompt_ids, rec["gen_ids"], rec["cache_gen_len"], hops - 1)
        kv = rebuild_cache(pipeline.model, segs, pipeline.device)
        donor_len, target_len = _len(kv), _len(real_kv)
        return kv, {
            "agent_idx": agent_idx, "skipped": False, "donor_type": self.mode,
            "donor_cache_len": donor_len, "target_cache_len": target_len, "len_ratio": donor_len / target_len,
            "handling": "rebuilt_from_token_ids_unmodified", "pad_positions": 0, "truncated_positions": 0,
            "rebuilt_by_prefill": True, "seeds": rec.get("seeds"), "distractor": rec.get("distractor"),
            "hop0_differs_from_greedy": rec.get("hop0_differs_from_greedy"),
            "hop0_common_prefix_frac": rec.get("hop0_common_prefix_frac"),
            "defends_distractor": rec.get("defends_distractor"),
        }


class SelfReplaySource:
    """self_replay (validation): rebuild THIS run's own trace from token ids and
    compare numerically with the cache the chain actually produced."""

    def fetch(self, agent_idx, question_key, real_kv, rng=None, pipeline=None, question=None, **_ctx):
        tr = pipeline._run_trace
        hops = agent_idx
        if any(h not in tr for h in range(hops)):
            raise DonorUnavailable("run_trace_missing", {"mode": "self_replay", "agent_idx": agent_idx})
        segs = trace_segments([tr[h]["prompt_ids"] for h in range(hops)], [tr[h]["gen_ids"] for h in range(hops)],
                              [tr[h]["cache_gen_len"] for h in range(hops)], hops - 1)
        kv = rebuild_cache(pipeline.model, segs, pipeline.device)
        stats = kv_diff_stats(real_kv, kv)
        return kv, {"agent_idx": agent_idx, "skipped": False, "donor_type": "self_replay",
                    "donor_cache_len": _len(kv), "target_cache_len": _len(real_kv),
                    "handling": "rebuilt_from_token_ids_unmodified", "pad_positions": 0,
                    "rebuilt_by_prefill": True, "rebuild_vs_real": stats}
