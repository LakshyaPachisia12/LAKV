"""
Synthetic long-context stress test for lakv/rlm_repl_kv.py -- a needle-
in-haystack variant of HotpotQA, built entirely from data already in
this project. No new dataset, no retriever, no external generator
script (deliberately not integrating RULER's own generator/infra --
see module rationale below).

Motivation (2026-09-22 branch feat/rlm-long-context): RLM's own paper
(Zhang, Kraska, Khattab, arXiv 2512.24601) reports recursive
decomposition begins beating a flat baseline "notably beyond 2^14
tokens" (16,384) on needle-in-haystack-style tasks (their S-NIAH
benchmark, itself built on NVIDIA's RULER). This project's existing
RLM+KV topology has only ever been tested on HotpotQA's own ~10-passage,
~1.5-2.5K-token distractor setting -- a regime that does not need
decomposition at all, and where the "kv" vs "text" return-channel
comparison has stayed a null result (see CLAUDE.md Finding 18, and the
2026-09-22 research-audit discussion that led to this branch). This
module tests the SAME topology and causal-audit methodology in the
regime RLM's own paper identifies as where decomposition should start to
matter: each scored question's real HotpotQA passages (the "needle")
are combined with a large pool of FILLER passages drawn from OTHER,
non-overlapping HotpotQA questions (the "haystack"), shuffled so the
needle's position is not fixed or predictable, and grown until the
combined passage list reaches approximately --target_tokens as measured
by the real tokenizer this project already uses -- not a passage-count
heuristic.

Deliberately targets 16K-32K tokens by default, not RLM's own much
larger benchmarks (up to 2^20 tokens for S-NIAH, 6-11M for
BrowseComp-Plus). Staying under Qwen2.5-7B-Instruct's native
32,768-token window avoids needing YaRN/rope-scaling for a first pass at
this question -- a bug class this project has already been burned by
once (see CLAUDE.md finding 12, Phi-3.5-mini's LongRoPE collapse) and
has no reason to reopen before establishing whether the basic setup even
works.

Reuses recursive_pipeline.py's format_passage() for identical passage
formatting, and the same held-out-pool discipline used throughout this
project's causal-audit machinery (lakv/causal_audit.py's KVAuditPool,
Evaluator._build_audit_pool): the filler pool is drawn from examples
BEYOND the scored range, never overlapping, so a filler passage can
never coincidentally be the real answer to a scored question, and no
scored question's own passages can leak into another scored question's
haystack.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional

from lakv.recursive_pipeline import format_passage


@dataclass
class LongContextExample:
    question: str
    answer: str
    passages: List[str]            # needle passages + filler, shuffled together
    needle_idxs: List[int] = field(default_factory=list)  # diagnostic only,
    # not consumed by RLMKVSession -- see build_long_context_examples's
    # docstring for the one known imprecision (string-identity matching).
    n_needle: int = 0
    n_filler: int = 0
    approx_tokens: int = 0


def _load_raw_hotpotqa(split: str, n: int) -> List[dict]:
    """Minimal, deliberately separate from
    recursive_pipeline.load_hotpotqa_structured: that loader stops at
    `n` items total; this one needs a much larger pool (scored questions
    plus a large, disjoint filler source) from the same underlying
    dataset object, loaded once via a single pass."""
    from datasets import load_dataset
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor")
    data = []
    for item in ds[split]:
        titles = item["context"]["title"]
        sentences = item["context"]["sentences"]
        passages = [format_passage(t, s) for t, s in zip(titles, sentences)]
        data.append({
            "question": item["question"],
            "answer": item["answer"],
            "passages": passages,
        })
        if n is not None and len(data) >= n:
            break
    return data


def build_long_context_examples(
    tokenizer,
    n: int,
    target_tokens: int = 24_000,
    split: str = "validation",
    filler_pool_size: int = 500,
    seed: int = 0,
    check_every: int = 5,
    max_filler_per_example: int = 2000,
) -> List[LongContextExample]:
    """Builds `n` needle-in-haystack examples.

    Each keeps its own real HotpotQA passages (the needle) and pads with
    filler passages drawn from a SEPARATE pool of `filler_pool_size`
    questions immediately following the scored range (never overlapping
    -- same discipline as KVAuditPool elsewhere in this project) until
    the combined passage list's token count (measured with the real
    tokenizer, not a heuristic) reaches approximately `target_tokens`.

    Each example gets its own independently-shuffled slice of the filler
    pool (not a shared order truncated to different lengths, and no
    passage repeated within one example's own haystack) -- two easy
    mistakes that would make this a less realistic, more easily-gamed
    stress test than a real needle-in-haystack task.

    known imprecision: `needle_idxs` is computed via string-identity
    membership in the shuffled combined list, so if a filler passage
    happens to be byte-identical to one of the needle passages (possible
    if two different HotpotQA questions reference the exact same short
    passage text), it would be misattributed. This field is diagnostic
    only -- never consumed by RLMKVSession or the causal-audit machinery
    -- so this is a cosmetic limitation, not a correctness bug in the
    actual experiment.

    max_filler_per_example is a hard stop (not just target_tokens) so a
    filler pool exhausted mid-example (filler_pool_size too small for
    the requested target_tokens) fails loudly via the returned example's
    approx_tokens being visibly short of target_tokens, rather than
    looping until the pool runs out silently.
    """
    rng = random.Random(seed)
    total_needed = n + filler_pool_size
    all_data = _load_raw_hotpotqa(split, total_needed)
    if len(all_data) < total_needed:
        raise ValueError(
            f"Requested n={n} + filler_pool_size={filler_pool_size} = "
            f"{total_needed} examples, but split {split!r} only has "
            f"{len(all_data)} available. Reduce filler_pool_size or n."
        )
    scored = all_data[:n]
    filler_source = all_data[n:total_needed]
    filler_passages: List[str] = [
        p for item in filler_source for p in item["passages"]
    ]
    if not filler_passages:
        raise ValueError("filler_pool_size produced zero filler passages -- increase it.")

    examples: List[LongContextExample] = []
    for item in scored:
        needle = list(item["passages"])

        shuffled_filler = list(filler_passages)
        rng.shuffle(shuffled_filler)

        combined = list(needle)
        n_filler = 0

        def _token_count(passages: List[str]) -> int:
            text = "\n\n".join(passages)
            return len(tokenizer(text, add_special_tokens=False)["input_ids"])

        n_tokens = _token_count(combined)
        while n_tokens < target_tokens and n_filler < min(
            len(shuffled_filler), max_filler_per_example
        ):
            take = shuffled_filler[n_filler:n_filler + check_every]
            combined.extend(take)
            n_filler += len(take)
            n_tokens = _token_count(combined)

        rng.shuffle(combined)
        needle_idxs = [i for i, p in enumerate(combined) if p in needle]

        examples.append(LongContextExample(
            question=item["question"],
            answer=item["answer"],
            passages=combined,
            needle_idxs=needle_idxs,
            n_needle=len(needle),
            n_filler=n_filler,
            approx_tokens=n_tokens,
        ))
    return examples


def main():
    """CPU-only inspection tool -- prints token-count/passage-count
    stats for a small batch without running any model, so the loader's
    real behavior against the real tokenizer can be sanity-checked
    before spending any GPU time on it.

    Usage: python -m lakv.long_context_hotpotqa --n 5 --target_tokens 24000
    """
    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--target_tokens", type=int, default=24_000)
    parser.add_argument("--filler_pool_size", type=int, default=500)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    examples = build_long_context_examples(
        tokenizer, n=args.n, target_tokens=args.target_tokens,
        filler_pool_size=args.filler_pool_size,
    )
    for i, ex in enumerate(examples):
        print(f"[{i}] approx_tokens={ex.approx_tokens} "
              f"n_needle={ex.n_needle} n_filler={ex.n_filler} "
              f"total_passages={len(ex.passages)} "
              f"needle_idxs={ex.needle_idxs}")
        print(f"    Q: {ex.question}")
        print(f"    A: {ex.answer}")


if __name__ == "__main__":
    main()
