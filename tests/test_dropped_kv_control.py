"""
Standalone test for LAKVPipeline._should_inject_kv(), which gates whether
a hop decompresses and injects the upstream KV cache at all.

This exists for the "dropped" causal-audit control: a genuine no-relay
condition, distinct from zeroed/random/mismatched (lakv/causal_audit.py),
which all substitute an already-decompressed cache with something else.
"dropped" must never reach decompression at all -- injected_kv_tuple stays
at its default None, and the receiver falls through to the same native
no-past_key_values generation path the Reasoner (agent_idx 0) already
always uses when it has nothing to inject.

No model / GPU required -- _should_inject_kv is a pure function of
(kv_message, causal_audit_mode), exercised directly as a staticmethod.

Run directly for a printout:
    python tests/test_dropped_kv_control.py

Or via pytest (from repo root):
    pytest tests/test_dropped_kv_control.py -v -s
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lakv.pipeline import LAKVPipeline
from lakv.evaluator import AUDIT_MODE_SUFFIXES

_FAKE_KV_MESSAGE = object()  # _should_inject_kv only checks "is not None"


def test_dropped_mode_skips_injection_when_a_cache_exists():
    assert LAKVPipeline._should_inject_kv(_FAKE_KV_MESSAGE, "dropped") is False


def test_dropped_mode_with_no_upstream_cache_also_skips_injection():
    # agent_idx 0 (the Reasoner): no predecessor, kv_message is already None
    # regardless of causal_audit_mode -- confirms this isn't double-counted
    # as a "dropped intervention" when there was nothing to drop.
    assert LAKVPipeline._should_inject_kv(None, "dropped") is False


def test_none_mode_injects_normally():
    assert LAKVPipeline._should_inject_kv(_FAKE_KV_MESSAGE, "none") is True


def test_zeroed_mode_still_injects_before_substitution():
    # zeroed/random/mismatched operate on the cache AFTER it's decompressed
    # (lakv/causal_audit.py), so the injection path must still be entered
    # for them -- only "dropped" bypasses it entirely.
    assert LAKVPipeline._should_inject_kv(_FAKE_KV_MESSAGE, "zeroed") is True


def test_random_mode_still_injects_before_substitution():
    assert LAKVPipeline._should_inject_kv(_FAKE_KV_MESSAGE, "random") is True


def test_mismatched_mode_still_injects_before_substitution():
    assert LAKVPipeline._should_inject_kv(_FAKE_KV_MESSAGE, "mismatched") is True


def test_no_kv_message_never_injects_regardless_of_mode():
    for mode in ("none", "zeroed", "random", "mismatched", "dropped"):
        assert LAKVPipeline._should_inject_kv(None, mode) is False


def test_audit_mode_suffix_dropped_is_registered():
    # Guards against a typo silently making "A_audit_dropped" an unknown
    # config name instead of dispatching to causal_audit_mode="dropped".
    assert AUDIT_MODE_SUFFIXES["_audit_dropped"] == "dropped"


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
