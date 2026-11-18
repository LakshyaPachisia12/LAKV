"""CPU tests for the `A_nopass_layers_{first3,mid3,last3}` layer-band ablation configs.

Default behaviour must be unchanged: the legacy ablation presets resolve to the same indices and still
see the passages, and the new configs differ from the plain `A_nopass` preset only in the layer-selection
fields."""
import dataclasses

import pytest
import torch

from lakv.evaluator import ABLATION_CONFIGS, LAYER_BAND_CONFIGS, Evaluator, PRESETS, resolve_ablation_indices

class _Tok:
    eos_token_id = 1


THIRDS = ["A_nopass_layers_first3", "A_nopass_layers_mid3", "A_nopass_layers_last3"]


@pytest.mark.parametrize("n,expected", [
    (28, ([0, 9], [9, 19], [19, 28])),
    (32, ([0, 11], [11, 21], [21, 32])),
    (36, ([0, 12], [12, 24], [24, 36])),
])
def test_thirds_partition_the_layers(n, expected):
    got = [resolve_ablation_indices(LAYER_BAND_CONFIGS[c], n) for c in THIRDS]
    assert got == [list(range(a, b)) for a, b in expected]
    flat = sum(got, [])
    assert flat == list(range(n))                      # disjoint, ordered, covers every layer once


def test_qwen28_third_sizes():
    assert [len(resolve_ablation_indices(LAYER_BAND_CONFIGS[c], 28)) for c in THIRDS] == [9, 10, 9]


def test_unknown_third_and_tiny_model_rejected():
    with pytest.raises(ValueError):
        resolve_ablation_indices({"type": "third", "which": 3}, 28)
    with pytest.raises(ValueError):
        resolve_ablation_indices({"type": "third", "which": 0}, 2)


def test_legacy_ablation_presets_unchanged():
    legacy = {"C_top20": list(range(20)), "C_bottom20": list(range(8, 28))}
    for name, idx in legacy.items():
        assert resolve_ablation_indices(ABLATION_CONFIGS[name], 28) == idx
        assert "receiver_sees_passages" not in ABLATION_CONFIGS[name]   # default True path untouched
    assert set(ABLATION_CONFIGS) >= {"C_random_20_s0", "C_random_20_s1", "C_random_20_s2", "C_top20", "C_bottom20"}


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=6,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    return Qwen2ForCausalLM(cfg).eval()


def _build(model, name):
    ev = Evaluator(model, tokenizer=_Tok(), device="cpu")
    pipe, kind = ev._build_pipeline(name, None, dataset="hotpotqa", greedy=True)
    return pipe, kind


def test_third_pipeline_is_hidden_and_differs_from_a_nopass_only_in_layer_fields(tiny_model):
    base, _ = _build(tiny_model, "A_nopass")
    assert base.config.receiver_sees_passages is False
    for name, which in zip(THIRDS, range(3)):
        pipe, kind = _build(tiny_model, name)
        assert kind == "multi"
        assert pipe.config.receiver_sees_passages is False          # hidden setting
        assert pipe.config.causal_audit_mode == "none"
        want = list(range(which * 2, which * 2 + 2))                # 6 layers -> 2/2/2
        assert pipe.selector is not None
        kept = [i for i, t in enumerate(pipe.selector.tier_assignment) if t in (1, 2)]
        assert kept == want
        a, b = dataclasses.asdict(base.config), dataclasses.asdict(pipe.config)
        layer_fields = {"use_layer_selection", "profile_path", "_custom_layer_indices"}
        diff = {k for k in a if a[k] != b[k]}
        assert diff <= layer_fields, diff - layer_fields


def test_run_metadata_reports_hidden_setting_for_thirds():
    from lakv.run_metadata import decoding_for_config
    for name in THIRDS:
        d = decoding_for_config(name)
        assert d["receiver_sees_passages"] is False and d["ablation_spec"]["type"] == "third"
        assert d["use_layer_selection"] is True
    assert decoding_for_config("C_top20")["receiver_sees_passages"] is True


def test_ablation_config_builds_when_a_profile_path_is_given():
    """Regression: the ablation branch referenced an un-imported LayerProfile and crashed with --profile_path."""
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=28,
                                     num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=256)).eval()
    ev = Evaluator(m, tokenizer=_Tok(), device="cpu")
    pipe, kind = ev._build_pipeline("A_nopass_layers_mid3", "profiles/qwen_gsm8k.json", dataset="hotpotqa", greedy=True)
    kept = [i for i, t in enumerate(pipe.selector.tier_assignment) if t in (1, 2)]
    assert kept == list(range(9, 19)) and kind == "multi"
