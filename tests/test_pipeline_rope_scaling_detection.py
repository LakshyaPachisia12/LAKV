"""
Standalone test for LAKVPipeline._uses_nondefault_rope_scaling(), which
gates _generate()'s position_offset==0 fast path (prime the cache with one
manual forward() call, then hand the rest of decode off to model.generate()).

Found via a real Phi-3.5-mini-instruct run: that fast path relies on
model.generate() correctly inferring the next token's true position from
past_key_values alone. That's fine for a flat RoPE theta (Qwen2.5-7B,
Mistral-7B-v0.3, and Qwen3-8B all have rope_type='default', and the fast
path produces correct output for all three, confirmed on real runs) -- but
Phi-3.5-mini uses LongRoPE (rope_type='longrope'), whose scaling depends on
whether the true absolute sequence position has crossed
original_max_position_embeddings, information generate() has no way to
reconstruct across a cache it did not build from scratch itself. On a real
run this produced pure word-salad output from the Finalizer specifically
(the only step that goes through this fast path -- the Reasoner/Verifier,
which use _generate_intermediate()'s fully manual per-token loop instead,
stayed completely coherent on the same run).

No model / GPU required -- this only exercises the config-inspection logic,
via a bare SimpleNamespace standing in for the real model.config object.

Run directly for a printout:
    python tests/test_pipeline_rope_scaling_detection.py

Or via pytest (from repo root):
    pytest tests/test_pipeline_rope_scaling_detection.py -v -s
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lakv.pipeline import LAKVPipeline


def _fake_pipeline(rope_parameters=None, rope_scaling=None):
    """A LAKVPipeline._uses_nondefault_rope_scaling() only ever reads
    self.model.config -- stand in with the minimum needed, no real
    __init__ (which needs an actual model/tokenizer) required."""
    return SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(rope_parameters=rope_parameters, rope_scaling=rope_scaling)
        )
    )


def test_qwen2p5_shape_is_default_not_flagged():
    fake_self = _fake_pipeline(rope_parameters={"rope_theta": 1000000.0, "rope_type": "default"})
    assert LAKVPipeline._uses_nondefault_rope_scaling(fake_self) is False


def test_mistral_shape_is_default_not_flagged():
    fake_self = _fake_pipeline(rope_parameters={"rope_theta": 1000000.0, "rope_type": "default"})
    assert LAKVPipeline._uses_nondefault_rope_scaling(fake_self) is False


def test_qwen3_8b_shape_is_default_not_flagged():
    fake_self = _fake_pipeline(rope_parameters={"rope_theta": 1000000, "rope_type": "default"})
    assert LAKVPipeline._uses_nondefault_rope_scaling(fake_self) is False


def test_phi3_5_mini_longrope_shape_is_flagged():
    fake_self = _fake_pipeline(rope_parameters={
        "rope_theta": 10000.0, "rope_type": "longrope",
        "long_factor": [1.0, 2.0], "short_factor": [1.0, 1.0],
        "original_max_position_embeddings": 4096,
    })
    assert LAKVPipeline._uses_nondefault_rope_scaling(fake_self) is True


def test_legacy_rope_scaling_dict_present_is_flagged():
    # Older-style configs that predate the rope_parameters nesting (see
    # _get_rope_theta's own fallback for the same distinction) expose
    # rope_scaling directly instead.
    fake_self = _fake_pipeline(rope_parameters=None, rope_scaling={"type": "longrope"})
    assert LAKVPipeline._uses_nondefault_rope_scaling(fake_self) is True


def test_legacy_no_scaling_at_all_is_not_flagged():
    fake_self = _fake_pipeline(rope_parameters=None, rope_scaling=None)
    assert LAKVPipeline._uses_nondefault_rope_scaling(fake_self) is False


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
