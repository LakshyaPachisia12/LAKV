"""Layer counts must come from the model config, never a literal 28.

CPU-only; uses a fake config object (no model is loaded).
"""
import random
from types import SimpleNamespace

import pytest

from lakv.evaluator import ABLATION_CONFIGS, resolve_ablation_indices
from lakv.layer_selector import LayerSelector, resolve_n_layers

MODELS = {"qwen2.5-7b": 28, "mistral-7b": 32, "qwen3-8b": 36}


@pytest.mark.parametrize("name,n", MODELS.items())
def test_resolve_n_layers_reads_config(name, n):
    assert resolve_n_layers(None, SimpleNamespace(num_hidden_layers=n)) == n


def test_explicit_n_layers_wins_over_config():
    assert resolve_n_layers(7, SimpleNamespace(num_hidden_layers=28)) == 7


def test_no_silent_default():
    with pytest.raises(ValueError):
        resolve_n_layers(None, None)
    with pytest.raises(ValueError):
        resolve_n_layers(None, SimpleNamespace())  # config without num_hidden_layers


@pytest.mark.parametrize("name,n", MODELS.items())
def test_from_custom_indices_sizes_selector_from_config(name, n):
    cfg = SimpleNamespace(num_hidden_layers=n)
    sel = LayerSelector._from_custom_indices([0, 1, 2], model_config=cfg)
    assert sel.n_layers == n
    assert len(sel.tier_assignment) == n
    assert [i for i, t in enumerate(sel.tier_assignment) if t == 1] == [0, 1, 2]
    assert all(t == 3 for t in sel.tier_assignment[3:])


def test_from_custom_indices_rejects_out_of_range():
    with pytest.raises(ValueError):
        LayerSelector._from_custom_indices([30], model_config=SimpleNamespace(num_hidden_layers=28))


def test_from_custom_indices_positional_n_layers_still_works():
    sel = LayerSelector._from_custom_indices([0], 32)
    assert sel.n_layers == 32


def test_ablation_indices_identical_to_legacy_on_28_layers():
    # The pre-change code hardcoded these exact values for Qwen2.5-7B.
    for seed in (0, 1, 2):
        legacy = sorted(random.Random(seed).sample(range(28), 20))
        assert resolve_ablation_indices(ABLATION_CONFIGS[f"C_random_20_s{seed}"], 28) == legacy
    assert resolve_ablation_indices(ABLATION_CONFIGS["C_top20"], 28) == list(range(20))
    assert resolve_ablation_indices(ABLATION_CONFIGS["C_bottom20"], 28) == list(range(8, 28))


@pytest.mark.parametrize("name,n", MODELS.items())
def test_ablation_indices_scale_with_layer_count(name, n):
    for key, spec in ABLATION_CONFIGS.items():
        idx = resolve_ablation_indices(spec, n)
        assert len(idx) == 20 and len(set(idx)) == 20
        assert all(0 <= i < n for i in idx)
    assert resolve_ablation_indices(ABLATION_CONFIGS["C_bottom20"], n)[-1] == n - 1
    assert resolve_ablation_indices(ABLATION_CONFIGS["C_top20"], n)[0] == 0


def test_n_keep_larger_than_model_raises():
    with pytest.raises(ValueError):
        resolve_ablation_indices(ABLATION_CONFIGS["C_top20"], 16)
