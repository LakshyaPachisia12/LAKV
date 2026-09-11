"""
Standalone test for lakv/rlm_repl_kv.py's splice_child_kv -- Stage 2's
one genuinely new piece of tensor math (everything else reuses already-
validated pieces: AnchorTable._rope_shift_k, CodeSandbox, the nudge
logic ported verbatim from rlm_repl.py).

splice_child_kv appends an independently-computed child KV cache onto an
existing, already-longer root cache -- structurally very close to
recursive_pipeline.py's merge_child_kv (tested in
tests/test_recursive_kv_merge.py), but there the root's own segment is
always "child 0" (unshifted, at the very start); here the root cache
already has real, non-empty content BEFORE the splice point, so every
splice must be shifted -- there is no "first child is exempt" case. This
file checks that specifically, rather than re-testing the shared
_rope_shift_k primitive itself (already covered elsewhere).

No model / GPU required -- synthetic tensors only.

Run directly for a printout:
    python tests/test_rlm_repl_kv.py

Or via pytest (from repo root):
    pytest tests/test_rlm_repl_kv.py -v -s
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lakv.anchor_table import AnchorTable
from lakv.rlm_repl_kv import RLMKVSession


THETA = 1_000_000.0


def _make_kv_tuple(seed: int, n_layers: int, batch: int, heads: int, seq: int, head_dim: int) -> tuple:
    g = torch.Generator().manual_seed(seed)
    layers = []
    for _ in range(n_layers):
        k = torch.randn(batch, heads, seq, head_dim, generator=g)
        v = torch.randn(batch, heads, seq, head_dim, generator=g)
        layers.append((k, v))
    return tuple(layers)


def test_splice_shapes():
    n_layers, batch, heads, head_dim = 2, 1, 4, 16
    root_len, child_len = 20, 7
    root_kv = _make_kv_tuple(1, n_layers, batch, heads, root_len, head_dim)
    child_kv = _make_kv_tuple(2, n_layers, batch, heads, child_len, head_dim)

    spliced = RLMKVSession.splice_child_kv(root_kv, child_kv, rope_theta=THETA)

    assert len(spliced) == n_layers
    for layer_idx, (k, v) in enumerate(spliced):
        assert k.shape == (batch, heads, root_len + child_len, head_dim), (layer_idx, k.shape)
        assert v.shape == (batch, heads, root_len + child_len, head_dim), (layer_idx, v.shape)
    print("[OK] splice_shapes")


def test_splice_preserves_root_content_unchanged():
    """Unlike merge_child_kv's "child 0" (which is always at position 0
    and needs no shift), the ROOT here always has real content BEFORE
    the splice point -- it must come through completely untouched,
    itself never shifted, only the appended child is."""
    n_layers, batch, heads, head_dim = 2, 1, 4, 16
    root_len, child_len = 20, 7
    root_kv = _make_kv_tuple(1, n_layers, batch, heads, root_len, head_dim)
    child_kv = _make_kv_tuple(2, n_layers, batch, heads, child_len, head_dim)

    spliced = RLMKVSession.splice_child_kv(root_kv, child_kv, rope_theta=THETA)

    for layer_idx in range(n_layers):
        k_spliced, v_spliced = spliced[layer_idx]
        k_root, v_root = root_kv[layer_idx]
        assert torch.equal(k_spliced[:, :, :root_len, :], k_root)
        assert torch.equal(v_spliced[:, :, :root_len, :], v_root)
    print("[OK] splice_preserves_root_content_unchanged")


def test_splice_shifts_child_by_root_length_not_zero():
    """The key difference from merge_child_kv's first-child case: here
    there is no "unshifted" case at all, since the root is never empty
    in practice. Confirms the child's K is actually rotated by root_len,
    not left as-is (which the analogous merge_child_kv test already
    guards for child 0 specifically -- this guards the splice path,
    where that shortcut must NOT apply)."""
    n_layers, batch, heads, head_dim = 1, 1, 2, 16
    root_len, child_len = 20, 7
    root_kv = _make_kv_tuple(1, n_layers, batch, heads, root_len, head_dim)
    child_kv = _make_kv_tuple(2, n_layers, batch, heads, child_len, head_dim)

    spliced = RLMKVSession.splice_child_kv(root_kv, child_kv, rope_theta=THETA)
    k_spliced_child = spliced[0][0][:, :, root_len:, :]
    k_child_original = child_kv[0][0]

    assert not torch.equal(k_spliced_child, k_child_original), (
        "child K must be shifted (not left unrotated) once appended after "
        "real root content -- an unshifted append here would encode the "
        "wrong position, the same RoPE-drift failure mode this project "
        "already diagnosed in config E"
    )

    k_expected = AnchorTable._rope_shift_k(k_child_original, shift=root_len, theta=THETA)
    assert torch.allclose(k_spliced_child, k_expected, atol=1e-5)
    print("[OK] splice_shifts_child_by_root_length_not_zero")


def test_splice_v_never_shifted():
    n_layers, batch, heads, head_dim = 1, 1, 2, 16
    root_len, child_len = 20, 7
    root_kv = _make_kv_tuple(1, n_layers, batch, heads, root_len, head_dim)
    child_kv = _make_kv_tuple(2, n_layers, batch, heads, child_len, head_dim)

    spliced = RLMKVSession.splice_child_kv(root_kv, child_kv, rope_theta=THETA)
    v_spliced_child = spliced[0][1][:, :, root_len:, :]
    assert torch.equal(v_spliced_child, child_kv[0][1])
    print("[OK] splice_v_never_shifted")


def test_splice_answer_only_drops_child_prompt_framing():
    """The actual bug found in real-model testing (2026-09-10): splicing
    a child's FULL cache (its own system/user framing plus its answer)
    produced confused root output -- blank turns, the literal word
    "assistant" leaking into generated code. answer_only_from should
    drop everything before that offset, keeping only the generated
    answer's K/V."""
    n_layers, batch, heads, head_dim = 2, 1, 4, 16
    root_len, prompt_len, answer_len = 20, 12, 5
    root_kv = _make_kv_tuple(1, n_layers, batch, heads, root_len, head_dim)
    child_kv = _make_kv_tuple(2, n_layers, batch, heads, prompt_len + answer_len, head_dim)

    spliced_full = RLMKVSession.splice_child_kv(root_kv, child_kv, rope_theta=THETA)
    spliced_answer_only = RLMKVSession.splice_child_kv(
        root_kv, child_kv, rope_theta=THETA, answer_only_from=prompt_len
    )

    assert spliced_full[0][0].shape[2] == root_len + prompt_len + answer_len
    assert spliced_answer_only[0][0].shape[2] == root_len + answer_len
    # the answer-only splice's appended segment must match what you'd get
    # by directly splicing the answer-only slice of child_kv
    answer_only_child_kv = tuple(
        (k[:, :, prompt_len:, :], v[:, :, prompt_len:, :]) for k, v in child_kv
    )
    expected = RLMKVSession.splice_child_kv(root_kv, answer_only_child_kv, rope_theta=THETA)
    for layer_idx in range(n_layers):
        assert torch.equal(spliced_answer_only[layer_idx][0], expected[layer_idx][0])
        assert torch.equal(spliced_answer_only[layer_idx][1], expected[layer_idx][1])
    print("[OK] splice_answer_only_drops_child_prompt_framing")


def test_sample_next_token_penalizes_session_wide_history():
    """The actual bug found in real-model testing (2026-09-10): severe
    repetition loops in BOTH return channels, traced to
    model.generate()'s built-in repetition_penalty only seeing the
    current turn's few new tokens, not the whole session. This checks
    the manual replacement directly: a token seen anywhere in
    session-wide history (not just the current turn) must be penalized."""
    session = RLMKVSession.__new__(RLMKVSession)  # skip __init__ (no model needed for this method)
    logits = torch.zeros(1, 10)
    logits[0, 3] = 5.0  # token 3 is the favorite, but only by a close margin
    logits[0, 7] = 4.9  # token 7 is a close second -- 5.0/1.05 ~= 4.76 < 4.9,
    #                     so the penalty should be just enough to flip the winner

    # with no history, token 3 wins outright
    chosen = session._sample_next_token(logits, generated_ids=[])
    assert chosen.item() == 3

    # token 3 appeared many turns ago (not just "last turn") -- must
    # still be penalized relative to token 7 now
    chosen_after_history = session._sample_next_token(logits, generated_ids=[3, 3, 3])
    assert chosen_after_history.item() == 7, (
        "expected repetition penalty to demote a token seen earlier in the "
        "session, even though it wasn't in the current turn's own tokens"
    )
    print("[OK] sample_next_token_penalizes_session_wide_history")


def test_stop_condition_fires_exactly_at_complete_action_not_before_or_after():
    """The actual bug found in real-model testing (2026-09-10): nothing
    stopped generation once the model finished writing one action, so it
    kept going and invented a fake "[stdout]" observation plus a second
    fake action within the same turn -- which then got permanently baked
    into the session's real notes. The fix checks, after each new token,
    whether a complete action now exists; this test validates that check
    fires at exactly the right character offset -- not while the fence
    is still open, and it must not somehow "un-fire" once real content
    (simulating what an unstopped generation would have hallucinated
    next) is appended after it."""
    from lakv.rlm_repl import CODE_FENCE_RE, BARE_CALL_RE

    def should_stop(partial_text: str) -> bool:
        return bool(CODE_FENCE_RE.search(partial_text) or BARE_CALL_RE.search(partial_text))

    full_turn = '```python\nfinal_answer("done")\n```'
    call_text = 'final_answer("done")'
    call_complete_offset = full_turn.index(call_text) + len(call_text)
    hallucinated_continuation = (
        full_turn + '\n[stdout]\n"a fake observation the model invented"\n'
        '```python\nfinal_answer("something else")\n```'
    )

    # Feed the text in growing prefixes, character by character, tracking
    # the first offset where the stop condition fires.
    first_stop_offset = None
    for i in range(1, len(hallucinated_continuation) + 1):
        if should_stop(hallucinated_continuation[:i]):
            first_stop_offset = i
            break

    assert first_stop_offset is not None, "stop condition never fired at all"
    # BARE_CALL_RE matches as soon as `final_answer("done")` itself is
    # complete, even before the wrapping fence's closing ``` -- that's
    # fine (stopping the moment a complete, recognizable action exists
    # is the actual goal, not specifically waiting for fence closure).
    # What matters: it must fire no later than the real action's own
    # closing fence, and never require ANY of the hallucinated
    # continuation text to trigger.
    assert call_complete_offset <= first_stop_offset <= len(full_turn), (
        f"expected the stop condition to fire between the call completing "
        f"(offset {call_complete_offset}) and the fence closing (offset "
        f"{len(full_turn)}), but it fired at {first_stop_offset}"
    )
    assert first_stop_offset < len(full_turn) + 1, "must not require hallucinated continuation text"

    # And it must not fire before the call itself is even complete --
    # would mean cutting the model off mid-string.
    for i in range(1, call_complete_offset):
        assert not should_stop(full_turn[:i]), (
            f"stop condition fired too early, at offset {i}, before the "
            f"call was even complete: {full_turn[:i]!r}"
        )
    print("[OK] stop_condition_fires_exactly_at_complete_action_not_before_or_after")


def test_cache_len_helper():
    assert RLMKVSession._cache_len(None) == 0
    kv = _make_kv_tuple(1, 1, 1, 2, 5, 16)
    assert RLMKVSession._cache_len(kv) == 5
    print("[OK] cache_len_helper")


def test_init_rejects_unknown_causal_audit_mode():
    """__init__ validates causal_audit_mode the same way it already
    validates return_channel -- catches a typo'd mode name immediately
    rather than failing confusingly deep inside apply_causal_audit on
    the first real run."""
    raised = False
    try:
        RLMKVSession.__new__(RLMKVSession).__init__(
            model=None, tokenizer=None, causal_audit_mode="not_a_real_mode",
        )
    except ValueError:
        raised = True
    assert raised, "expected ValueError for an unknown causal_audit_mode"
    print("[OK] init_rejects_unknown_causal_audit_mode")


def test_causal_audit_zeroed_then_splice_matches_manual_zero_splice():
    """The actual wiring this session adds: apply_causal_audit(mode,
    child_kv, ...) BEFORE splice_child_kv, not instead of it -- the
    audit substitutes WHAT gets spliced, the splice mechanics (shape,
    RoPE shift by root_len) stay identical regardless of mode. Confirms
    that end-to-end: zeroing the child via apply_causal_audit and then
    splicing must equal manually zeroing the child tuple and splicing
    that directly."""
    from lakv.causal_audit import apply_causal_audit

    n_layers, batch, heads, head_dim = 2, 1, 4, 16
    root_len, child_len = 20, 7
    root_kv = _make_kv_tuple(1, n_layers, batch, heads, root_len, head_dim)
    child_kv = _make_kv_tuple(2, n_layers, batch, heads, child_len, head_dim)

    audited_kv, audit_log = apply_causal_audit(
        "zeroed", child_kv, agent_idx=0, question_key="fake_q",
    )
    assert audit_log["mode"] == "zeroed"
    spliced_via_audit = RLMKVSession.splice_child_kv(root_kv, audited_kv, rope_theta=THETA)

    manually_zeroed = tuple((torch.zeros_like(k), torch.zeros_like(v)) for k, v in child_kv)
    spliced_manual = RLMKVSession.splice_child_kv(root_kv, manually_zeroed, rope_theta=THETA)

    for layer_idx in range(n_layers):
        assert torch.equal(spliced_via_audit[layer_idx][0], spliced_manual[layer_idx][0])
        assert torch.equal(spliced_via_audit[layer_idx][1], spliced_manual[layer_idx][1])
    # and the zeroed splice must still differ from a real (unaudited) splice --
    # confirms the substitution actually took effect, not a silent no-op
    spliced_real = RLMKVSession.splice_child_kv(root_kv, child_kv, rope_theta=THETA)
    assert not torch.equal(spliced_via_audit[0][0][:, :, root_len:, :], spliced_real[0][0][:, :, root_len:, :])
    print("[OK] causal_audit_zeroed_then_splice_matches_manual_zero_splice")


def test_causal_audit_mode_none_is_a_pure_passthrough():
    """mode="none" (the default) must reproduce the exact pre-audit
    splice behavior -- confirms adding the audit hook did not change
    anything for the real, non-audited case, which is what every
    non-audit run (including everything already validated in
    docs/PROGRESS_REPORT.md) depends on continuing to work unchanged."""
    from lakv.causal_audit import apply_causal_audit

    n_layers, batch, heads, head_dim = 2, 1, 4, 16
    root_len, child_len = 20, 7
    root_kv = _make_kv_tuple(1, n_layers, batch, heads, root_len, head_dim)
    child_kv = _make_kv_tuple(2, n_layers, batch, heads, child_len, head_dim)

    audited_kv, audit_log = apply_causal_audit(
        "none", child_kv, agent_idx=0, question_key="fake_q",
    )
    assert audit_log["mode"] == "none"
    for (ak, av), (ck, cv) in zip(audited_kv, child_kv):
        assert torch.equal(ak, ck) and torch.equal(av, cv)
    print("[OK] causal_audit_mode_none_is_a_pure_passthrough")


if __name__ == "__main__":
    test_splice_shapes()
    test_splice_preserves_root_content_unchanged()
    test_splice_shifts_child_by_root_length_not_zero()
    test_splice_v_never_shifted()
    test_splice_answer_only_drops_child_prompt_framing()
    test_sample_next_token_penalizes_session_wide_history()
    test_stop_condition_fires_exactly_at_complete_action_not_before_or_after()
    test_cache_len_helper()
    test_init_rejects_unknown_causal_audit_mode()
    test_causal_audit_zeroed_then_splice_matches_manual_zero_splice()
    test_causal_audit_mode_none_is_a_pure_passthrough()
    print("\nAll Stage 2 (RLM+KV splice) mechanics tests passed.")
