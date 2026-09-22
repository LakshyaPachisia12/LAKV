"""CPU-only unit tests for lakv/long_context_hotpotqa.py -- the needle-
in-haystack loader added on branch feat/rlm-long-context (2026-09-22).
Uses a fake tokenizer and monkeypatches _load_raw_hotpotqa so these run
with no network access and no real HotpotQA download, matching this
project's convention of testing pure logic before any real-model/real-
data step (see tests/test_recursive_kv_merge.py, tests/test_rlm_repl_kv.py
for the same pattern)."""

import pytest

from lakv import long_context_hotpotqa as lch


class WordCountTokenizer:
    """Deterministic fake: one token per whitespace-separated word. Good
    enough to test the growth/stopping logic without needing the real
    (slow, network-dependent) Qwen tokenizer."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def _fake_dataset(n_questions: int, words_per_passage: int = 20, n_passages: int = 10):
    """n_questions items, each with n_passages passages of
    words_per_passage words -- enough structure to test needle/filler
    separation and token growth without touching the real dataset."""
    data = []
    for qi in range(n_questions):
        passages = [
            f"[Title{qi}_{pi}] " + " ".join(f"w{qi}_{pi}_{w}" for w in range(words_per_passage))
            for pi in range(n_passages)
        ]
        data.append({
            "question": f"question {qi}?",
            "answer": f"answer{qi}",
            "passages": passages,
        })
    return data


@pytest.fixture
def patch_loader(monkeypatch):
    def _patch(n_questions, **kwargs):
        fake_data = _fake_dataset(n_questions, **kwargs)

        def _fake_load_raw(split, n):
            if n is not None and len(fake_data) < n:
                return list(fake_data)
            return list(fake_data[:n]) if n is not None else list(fake_data)

        monkeypatch.setattr(lch, "_load_raw_hotpotqa", _fake_load_raw)
        return fake_data

    return _patch


def test_reaches_approximately_target_tokens(patch_loader):
    patch_loader(n_questions=20, words_per_passage=20, n_passages=10)
    tokenizer = WordCountTokenizer()
    examples = lch.build_long_context_examples(
        tokenizer, n=2, target_tokens=500, filler_pool_size=18, check_every=2,
    )
    assert len(examples) == 2
    for ex in examples:
        # Growth loop stops as soon as n_tokens >= target -- overshoot by
        # at most one batch (check_every passages' worth of words) is
        # expected and fine, never a large or unbounded overshoot.
        assert ex.approx_tokens >= 500
        assert ex.approx_tokens < 500 + 2 * 20 * 2  # generous slack


def test_needle_passages_all_present_and_filler_is_disjoint_source(patch_loader):
    fake_data = patch_loader(n_questions=20, words_per_passage=15, n_passages=10)
    tokenizer = WordCountTokenizer()
    examples = lch.build_long_context_examples(
        tokenizer, n=3, target_tokens=800, filler_pool_size=15, check_every=3,
    )
    for i, ex in enumerate(examples):
        needle = fake_data[i]["passages"]
        # Every real (needle) passage survives into the final shuffled list.
        for p in needle:
            assert p in ex.passages
        assert ex.n_needle == len(needle)
        # Filler passages come only from examples 3..17 (beyond the
        # scored range of 3), never from another SCORED question or from
        # the question's own needle re-added as "filler".
        filler_source_texts = {p for item in fake_data[3:18] for p in item["passages"]}
        non_needle = [p for p in ex.passages if p not in needle]
        assert non_needle, "expected some filler to have been added"
        assert all(p in filler_source_texts for p in non_needle)


def test_no_repeated_filler_passage_within_one_example(patch_loader):
    patch_loader(n_questions=20, words_per_passage=10, n_passages=10)
    tokenizer = WordCountTokenizer()
    examples = lch.build_long_context_examples(
        tokenizer, n=1, target_tokens=2000, filler_pool_size=18, check_every=4,
    )
    ex = examples[0]
    assert len(ex.passages) == len(set(ex.passages)), "a filler passage repeated within one example"


def test_different_examples_get_different_filler_ordering(patch_loader):
    fake_data = patch_loader(n_questions=20, words_per_passage=10, n_passages=10)
    tokenizer = WordCountTokenizer()
    examples = lch.build_long_context_examples(
        tokenizer, n=2, target_tokens=600, filler_pool_size=18, check_every=3, seed=1,
    )
    # Different needles guarantee the full lists differ trivially -- the
    # real thing being tested is that filler selection isn't a single
    # shared sequence truncated per example (which would make the SET of
    # filler passages used identical, just prefix-length-limited).
    needle_0 = set(fake_data[0]["passages"])
    needle_1 = set(fake_data[1]["passages"])
    filler_0 = {p for p in examples[0].passages if p not in needle_0}
    filler_1 = {p for p in examples[1].passages if p not in needle_1}
    assert examples[0].passages != examples[1].passages
    assert filler_0 != filler_1


def test_raises_when_filler_pool_too_small(patch_loader):
    patch_loader(n_questions=5)
    tokenizer = WordCountTokenizer()
    with pytest.raises(ValueError, match="only has"):
        lch.build_long_context_examples(tokenizer, n=3, filler_pool_size=10)


def test_max_filler_per_example_caps_growth_even_if_target_unreached(patch_loader):
    patch_loader(n_questions=20, words_per_passage=5, n_passages=10)
    tokenizer = WordCountTokenizer()
    examples = lch.build_long_context_examples(
        tokenizer, n=1, target_tokens=10**9,  # unreachable target
        filler_pool_size=18, check_every=5, max_filler_per_example=20,
    )
    ex = examples[0]
    assert ex.n_filler <= 20
    assert ex.approx_tokens < 10**9  # loop stopped early via the cap, not the target
