"""CPU-only unit tests for lakv/confidence.py -- the process-level
confidence signature added 2026-09-18 (see that module's docstring for
the full motivation). No GPU/model needed: these test the pure math on
dummy logits tensors, the same pattern tests/test_recursive_kv_merge.py
and tests/test_rlm_repl_kv.py use for their own CPU-verifiable mechanics,
before any real-model wiring is trusted."""

import math

import torch

from lakv.confidence import stats_from_logits


def test_empty_list_returns_none():
    assert stats_from_logits([]) is None


def test_single_confident_token_has_high_top1_prob_low_entropy():
    vocab = 100
    logits = torch.full((1, vocab), -10.0)
    logits[0, 5] = 20.0  # one token completely dominates
    stats = stats_from_logits([logits])
    assert stats is not None
    assert stats.n_tokens == 1
    assert stats.mean_top1_prob > 0.999
    assert stats.mean_entropy < 0.01


def test_uniform_logits_have_low_top1_prob_high_entropy():
    vocab = 100
    logits = torch.zeros((1, vocab))  # every token equally likely
    stats = stats_from_logits([logits])
    assert stats is not None
    expected_prob = 1.0 / vocab
    expected_entropy = math.log(vocab)
    assert abs(stats.mean_top1_prob - expected_prob) < 1e-4
    assert abs(stats.mean_entropy - expected_entropy) < 1e-3


def test_averages_across_multiple_tokens():
    vocab = 10
    confident = torch.full((1, vocab), -10.0)
    confident[0, 0] = 20.0
    uniform = torch.zeros((1, vocab))
    stats = stats_from_logits([confident, uniform])
    assert stats is not None
    assert stats.n_tokens == 2
    # Mean should sit strictly between the two individual extremes.
    solo_confident = stats_from_logits([confident])
    solo_uniform = stats_from_logits([uniform])
    assert solo_uniform.mean_top1_prob < stats.mean_top1_prob < solo_confident.mean_top1_prob
    assert solo_confident.mean_entropy < stats.mean_entropy < solo_uniform.mean_entropy


def test_accepts_1d_and_2d_logits_shapes():
    vocab = 20
    logits_2d = torch.randn(1, vocab)
    logits_1d = logits_2d.reshape(-1)
    stats_2d = stats_from_logits([logits_2d])
    stats_1d = stats_from_logits([logits_1d])
    assert stats_2d.mean_top1_prob == stats_1d.mean_top1_prob
    assert stats_2d.mean_entropy == stats_1d.mean_entropy


def test_real_content_more_confident_than_corrupted_is_the_expected_use_pattern():
    """Not a real-model test (no GPU here) -- just documents/verifies the
    comparison shape this module exists to support: a peaked distribution
    (simulating a model that 'knows' the answer) should score higher
    top1_prob / lower entropy than a near-uniform one (simulating garbage
    input), the same direction the causal-audit ladder expects for
    real-vs-zeroed/random conditions."""
    vocab = 50
    real_like = [torch.full((1, vocab), -5.0) for _ in range(5)]
    for t in real_like:
        t[0, 7] = 15.0
    garbage_like = [torch.randn(1, vocab) * 0.01 for _ in range(5)]  # near-uniform

    real_stats = stats_from_logits(real_like)
    garbage_stats = stats_from_logits(garbage_like)
    assert real_stats.mean_top1_prob > garbage_stats.mean_top1_prob
    assert real_stats.mean_entropy < garbage_stats.mean_entropy
