"""
Standalone test for lakv/rlm_repl.py -- the real (code-executing) RLM
decomposition mechanism, Stage 1.

Tests the two things that can go wrong before ever touching a real model:
(1) CodeSandbox actually executes code, captures/truncates stdout, catches
errors instead of crashing, blocks unsafe builtins, and correctly detects
final_answer() calls; (2) RLMSession's turn loop correctly drives that
sandbox from a scripted sequence of model responses -- using a stub
generate_fn, not a real model, so this validates the LOOP MECHANICS
(message history, code extraction, turn budget, the "no code found" nudge)
independently of whether a real 7B model can actually produce good code,
which is a separate, real risk this project is aware of (see rlm_repl.py's
module docstring) and can only be checked on a GPU.

No model / GPU required -- a scripted stub stands in for the LLM.

Run directly for a printout:
    python tests/test_rlm_repl.py

Or via pytest (from repo root):
    pytest tests/test_rlm_repl.py -v -s
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lakv.rlm_repl import CodeSandbox, RLMSession, MAX_STDOUT_CHARS


def make_stub(responses):
    """Returns a stub generate_fn that yields `responses` in order, then
    repeats the LAST response for any further calls. Needed because
    RLMSession now asks for one confirmation before accepting a
    final_answer reached after touching very few passages (see
    run()'s low_exploration_confirmed logic) -- repeating the same
    final_answer(...) response models a scripted "model" re-confirming
    its own answer when asked, without every test needing to hand-write
    an extra confirmation turn."""
    call_log = {"n": 0}

    def stub_generate(messages):
        i = min(call_log["n"], len(responses) - 1)
        call_log["n"] += 1
        return responses[i]

    return stub_generate


# ── CodeSandbox tests ───────────────────────────────────────────────────

def test_sandbox_executes_code_and_captures_stdout():
    sandbox = CodeSandbox("Q?", ["p0", "p1"], llm_query=lambda t: "unused")
    turn = sandbox.execute("print(len(passages))")
    assert turn.error is None
    assert turn.stdout.strip() == "2"
    print("[OK] sandbox_executes_code_and_captures_stdout")


def test_sandbox_truncates_long_stdout():
    sandbox = CodeSandbox("Q?", [], llm_query=lambda t: "unused")
    turn = sandbox.execute("print('x' * 5000)")
    assert turn.stdout_truncated is True
    assert len(turn.stdout) < 5000
    assert "truncated" in turn.stdout
    print(f"[OK] sandbox_truncates_long_stdout (kept {len(turn.stdout)} of 5000+ chars, "
          f"cap={MAX_STDOUT_CHARS})")


def test_sandbox_catches_errors_instead_of_crashing():
    sandbox = CodeSandbox("Q?", [], llm_query=lambda t: "unused")
    turn = sandbox.execute("1 / 0")
    assert turn.error is not None
    assert "ZeroDivisionError" in turn.error
    print("[OK] sandbox_catches_errors_instead_of_crashing")


def test_sandbox_blocks_unsafe_builtins():
    sandbox = CodeSandbox("Q?", [], llm_query=lambda t: "unused")
    for dangerous in ["open('x', 'w')", "__import__('os')", "eval('1')", "exec('1')"]:
        turn = sandbox.execute(dangerous)
        assert turn.error is not None, f"expected {dangerous!r} to be blocked, but it ran clean"
    print("[OK] sandbox_blocks_unsafe_builtins")


def test_sandbox_llm_query_routes_to_injected_function_and_counts_calls():
    calls = []

    def fake_llm_query(text):
        calls.append(text)
        return f"response to: {text}"

    sandbox = CodeSandbox("Q?", ["p0"], llm_query=fake_llm_query)
    turn = sandbox.execute("r = llm_query('hello')\nprint(r)")
    assert turn.error is None
    # llm_query now echoes its own result as a side effect (see
    # test_sandbox_llm_query_echoes_result_even_inside_a_loop), so the
    # model's own explicit print(r) shows the same text a second time --
    # redundant here but harmless, and necessary for the case where the
    # call isn't the model's own printed/last-line expression at all.
    assert turn.stdout.count("response to: hello") == 2
    assert calls == ["hello"]
    assert sandbox.llm_query_calls == 1
    print("[OK] sandbox_llm_query_routes_to_injected_function_and_counts_calls")


def test_sandbox_auto_prints_last_bare_expression():
    """The actual bug found in real-model testing (2026-09-10): the model
    called llm_query(...) without wrapping it in print(), and got back
    empty stdout -- it never saw the answer it had just asked for. This
    checks the fix directly: a bare expression as the last line of the
    code should have its value shown automatically, Jupyter-style, even
    with no explicit print()."""
    sandbox = CodeSandbox("Q?", [], llm_query=lambda t: "the answer is 42")
    turn = sandbox.execute("llm_query('what is the answer?')")
    assert turn.error is None
    assert "the answer is 42" in turn.stdout
    print("[OK] sandbox_auto_prints_last_bare_expression")


def test_sandbox_auto_print_does_not_duplicate_explicit_print():
    sandbox = CodeSandbox("Q?", [], llm_query=lambda t: "unused")
    turn = sandbox.execute("print('only once')")
    assert turn.stdout.strip() == "only once"
    print("[OK] sandbox_auto_print_does_not_duplicate_explicit_print")


def test_sandbox_auto_print_skips_none_results():
    """A bare expression that evaluates to None (e.g. final_answer(...),
    which returns nothing) must not print a spurious 'None' line."""
    sandbox = CodeSandbox("Q?", [], llm_query=lambda t: "unused")
    turn = sandbox.execute("final_answer('done')")
    assert turn.stdout.strip() == ""
    print("[OK] sandbox_auto_print_skips_none_results")


def test_sandbox_auto_print_only_affects_last_line():
    """Earlier bare expressions in the same code block should NOT
    auto-print -- only the very last line gets the Jupyter-style
    treatment, matching real REPL/notebook semantics."""
    sandbox = CodeSandbox("Q?", [], llm_query=lambda t: "unused")
    turn = sandbox.execute("1 + 1\nprint('explicit')\n2 + 2")
    # last line (2 + 2) auto-prints; the bare "1 + 1" on its own earlier
    # line must NOT have auto-printed too
    assert turn.stdout.strip().splitlines() == ["explicit", "4"]
    print("[OK] sandbox_auto_print_only_affects_last_line")


def test_code_fence_regex_accepts_label_variations():
    from lakv.rlm_repl import CODE_FENCE_RE
    for fenced in [
        "```python\nprint(1)\n```",
        "```Python\nprint(1)\n```",
        "```py\nprint(1)\n```",
        "```\nprint(1)\n```",
    ]:
        assert CODE_FENCE_RE.search(fenced) is not None, f"failed to match: {fenced!r}"
    print("[OK] code_fence_regex_accepts_label_variations")


def test_sandbox_llm_query_echoes_result_even_inside_a_loop():
    """The actual bug found in real-model testing (2026-09-10): the
    earlier auto-print fix only covers a bare call sitting alone on the
    LAST line -- a call buried inside a for-loop or if-block (exactly
    what realistic search code looks like) still vanished silently,
    since Python itself doesn't display values from nested statements.
    llm_query must now announce its own result unconditionally,
    regardless of where in the code it's called from."""
    sandbox = CodeSandbox("Q?", ["p0", "p1"], llm_query=lambda t: "found it")
    turn = sandbox.execute(
        "for p in passages:\n"
        "    if p == 'p1':\n"
        "        llm_query(p)\n"
        "        break\n"
    )
    assert turn.error is None
    assert "found it" in turn.stdout
    print("[OK] sandbox_llm_query_echoes_result_even_inside_a_loop")


def test_references_passages_variable_ignores_string_mentions():
    """The actual bug found in real-model testing (2026-09-10): the model
    wrote llm_query("...I should check the passages for clues.") -- the
    word "passages" only appears inside a natural-language string, with
    no real access to the passages variable -- and a naive substring
    check wrongly counted that as "touched passages," permanently
    disabling the grounding nudge for the rest of that example."""
    from lakv.rlm_repl import _references_passages_variable

    assert _references_passages_variable(
        'llm_query("I should check the passages for clues.")'
    ) is False
    assert _references_passages_variable("passages[0]") is True
    assert _references_passages_variable("for p in passages:\n    print(p)") is True
    assert _references_passages_variable("llm_query(passages[2])") is True
    # invalid syntax must not crash the check -- just count as "not touched"
    assert _references_passages_variable("final_answer,No") is False
    print("[OK] references_passages_variable_ignores_string_mentions")


def test_sandbox_final_answer_coerces_non_string_to_string():
    """Real-model testing (2026-09-10) found the model sometimes calls
    final_answer(False) (a bool, following a yes/no question's literal
    wording) instead of a string -- this must not silently produce an
    unscoreable answer downstream."""
    sandbox = CodeSandbox("Q?", [], llm_query=lambda t: "unused")
    sandbox.execute("final_answer(False)")
    assert sandbox.answer == "False"
    assert isinstance(sandbox.answer, str)
    print("[OK] sandbox_final_answer_coerces_non_string_to_string")


def test_sandbox_final_answer_sets_done_and_answer():
    sandbox = CodeSandbox("Q?", [], llm_query=lambda t: "unused")
    assert sandbox.done is False
    turn = sandbox.execute("final_answer('42')")
    assert sandbox.done is True
    assert sandbox.answer == "42"
    assert turn.final_answer == "42"
    print("[OK] sandbox_final_answer_sets_done_and_answer")


# ── RLMSession loop tests (scripted stub model, no GPU) ────────────────

def test_session_drives_multi_turn_loop_to_final_answer():
    """Scripted 'model': turn 1 calls llm_query, turn 2 reads the
    observation and calls final_answer. Checks the loop correctly extracts
    code, executes it, feeds the observation back, and stops on
    final_answer -- the core mechanics a real model will also rely on.
    Zero passage touches here, so the low-exploration confirmation fires
    once (see make_stub) before the run actually ends."""
    responses = [
        "Let me check a passage.\n```python\nr = llm_query('what is 2+2?')\nprint(r)\n```",
        "```python\nfinal_answer('4')\n```",
    ]
    session = RLMSession(generate_fn=make_stub(responses), llm_query=lambda t: "4", max_turns=6)
    result = session.run("dummy question", ["p0"])
    assert result.answer == "4"
    assert result.hit_max_turns is False
    assert result.llm_query_calls == 1
    assert len(result.turns) == 3  # llm_query turn + final_answer + confirm re-final_answer
    print("[OK] session_drives_multi_turn_loop_to_final_answer")


def test_session_nudges_when_no_code_block_found():
    responses = [
        "I am thinking about this without writing any code yet.",
        "```python\nfinal_answer('ok')\n```",
    ]
    session = RLMSession(generate_fn=make_stub(responses), llm_query=lambda t: "unused", max_turns=6)
    result = session.run("dummy question", [])
    assert result.answer == "ok"
    # one turn nudged (no code found -> no sandbox turn recorded), one
    # turn executes final_answer, one more confirms it (zero passage touches)
    assert len(result.turns) == 2
    print("[OK] session_nudges_when_no_code_block_found")


def test_session_accepts_bare_final_answer_with_no_fence():
    """The actual bug found in real-model testing (2026-09-10): the model
    reliably stops wrapping final_answer(...) in a code fence once it has
    decided on an answer, even after using a fence correctly earlier in
    the same run -- and kept repeating the same unfenced line every turn
    until timeout, since the old code required a fence to recognize any
    action at all. This checks the fallback directly."""
    responses = [
        "```python\nr = llm_query('q')\nprint(r)\n```",
        'final_answer("no fence here")',  # bare, no ```python wrapper
    ]
    session = RLMSession(generate_fn=make_stub(responses), llm_query=lambda t: "answer", max_turns=6)
    result = session.run("dummy question", [])
    assert result.answer == "no fence here"
    assert result.hit_max_turns is False
    print("[OK] session_accepts_bare_final_answer_with_no_fence")


def test_session_accepts_bare_llm_query_with_no_fence():
    responses = [
        "llm_query('what is the capital?')",  # bare from turn 1
        "```python\nfinal_answer('Paris')\n```",
    ]
    calls = []
    session = RLMSession(
        generate_fn=make_stub(responses),
        llm_query=lambda t: calls.append(t) or "Paris",
        max_turns=6,
    )
    result = session.run("dummy question", [])
    assert result.answer == "Paris"
    assert calls == ["what is the capital?"]
    print("[OK] session_accepts_bare_llm_query_with_no_fence")


def test_session_nudges_once_after_ungrounded_llm_query_calls():
    """Real-model testing (2026-09-10) found the model sometimes never
    touches `passages` at all -- repeated llm_query calls with bare,
    ungrounded questions, getting confident-sounding fabricated answers
    back. This checks the added nudge: fires once, only after >=2
    llm_query calls with no `passages` reference in the executed code."""
    responses = [
        "```python\nllm_query('q1')\n```",
        "```python\nllm_query('q2')\n```",
        "```python\nfinal_answer('done')\n```",
    ]
    session = RLMSession(generate_fn=make_stub(responses), llm_query=lambda t: "ok", max_turns=6)
    result = session.run("dummy question", ["p0"])

    nudge_count = sum(
        1 for m in result.transcript
        if m["role"] == "user" and "without ever reading" in m["content"]
    )
    assert nudge_count == 1, f"expected exactly one nudge, got {nudge_count}"
    assert result.answer == "done"
    print("[OK] session_nudges_once_after_ungrounded_llm_query_calls")


def test_session_nudges_even_when_word_passages_appears_in_a_string():
    """End-to-end regression test for the exact real-model failure: the
    model's llm_query string argument happens to contain the word
    "passages" in natural language, with no real variable access. The
    nudge must still fire after 2 such calls -- it must not be fooled
    into thinking passages were actually read."""
    responses = [
        '```python\nllm_query("I should check the passages for clues.")\n```',
        '```python\nllm_query("still guessing, no passages read")\n```',
        "```python\nfinal_answer('done')\n```",
    ]
    session = RLMSession(generate_fn=make_stub(responses), llm_query=lambda t: "ok", max_turns=6)
    result = session.run("dummy question", ["p0"])

    nudge_count = sum(
        1 for m in result.transcript
        if m["role"] == "user" and "without ever reading" in m["content"]
    )
    assert nudge_count == 1, (
        f"expected exactly one nudge despite 'passages' appearing in strings, got {nudge_count}"
    )
    print("[OK] session_nudges_even_when_word_passages_appears_in_a_string")


def test_session_nudges_once_on_exact_repeated_code():
    """Real-model testing (2026-09-10) found the model sometimes reruns
    the EXACT same (unhelpful) code repeatedly with no adaptation. The
    nudge should fire once, on the first repeat, not wait for multiple."""
    responses = [
        "```python\nx = 1\nprint(x)\n```",
        "```python\ny = 2\nprint(y)\n```",  # different code -- no nudge yet
        "```python\ny = 2\nprint(y)\n```",  # exact repeat of previous -- nudge
        "```python\nfinal_answer('done')\n```",
    ]
    session = RLMSession(generate_fn=make_stub(responses), llm_query=lambda t: "unused", max_turns=6)
    result = session.run("dummy question", [])

    nudge_count = sum(
        1 for m in result.transcript
        if m["role"] == "user" and "exact same code" in m["content"]
    )
    assert nudge_count == 1
    assert result.answer == "done"
    print("[OK] session_nudges_once_on_exact_repeated_code")


def test_session_no_nudge_when_passages_touched():
    responses = [
        "```python\nprint(passages[0])\n```",
        "```python\nllm_query('q1')\n```",
        "```python\nllm_query('q2')\n```",
        "```python\nfinal_answer('done')\n```",
    ]
    session = RLMSession(generate_fn=make_stub(responses), llm_query=lambda t: "ok", max_turns=6)
    result = session.run("dummy question", ["p0 text"])

    nudge_count = sum(
        1 for m in result.transcript
        if m["role"] == "user" and "without ever reading" in m["content"]
    )
    assert nudge_count == 0
    print("[OK] session_no_nudge_when_passages_touched")


def test_session_reports_hit_max_turns_when_never_answering():
    def stub_generate(messages):
        return "```python\nprint('still thinking')\n```"

    session = RLMSession(generate_fn=stub_generate, llm_query=lambda t: "unused", max_turns=3)
    result = session.run("dummy question", [])
    assert result.answer is None
    assert result.hit_max_turns is True
    assert len(result.turns) == 3
    print("[OK] session_reports_hit_max_turns_when_never_answering")


def test_session_feeds_error_back_as_observation_not_crash():
    responses = [
        "```python\n1 / 0\n```",
        "```python\nfinal_answer('recovered')\n```",
    ]
    session = RLMSession(generate_fn=make_stub(responses), llm_query=lambda t: "unused", max_turns=6)
    result = session.run("dummy question", [])
    assert result.answer == "recovered"
    assert result.turns[0].error is not None
    print("[OK] session_feeds_error_back_as_observation_not_crash")


def test_session_confirms_low_exploration_before_accepting_answer():
    """Real-model testing (2026-09-10) found the dominant failure pattern
    across wrong answers was giving up after checking only 1-3 of the
    (often 10) passages. This checks the confirmation step directly: a
    final_answer reached with fewer than 3 passage-touching turns must
    NOT end the run immediately -- it should prompt once for
    confirmation, and only end once confirmed."""
    responses = [
        "```python\nfinal_answer('too fast')\n```",
        "```python\nfinal_answer('too fast')\n```",  # confirms the same answer
    ]
    session = RLMSession(generate_fn=make_stub(responses), llm_query=lambda t: "unused", max_turns=6)
    result = session.run("dummy question", ["p0"])

    confirm_count = sum(
        1 for m in result.transcript
        if m["role"] == "user" and "Before finalizing" in m["content"]
    )
    assert confirm_count == 1
    assert result.answer == "too fast"
    print("[OK] session_confirms_low_exploration_before_accepting_answer")


def test_session_no_confirmation_needed_after_real_exploration():
    """Three distinct passage-touching turns should be enough to accept
    final_answer immediately, with no confirmation nudge."""
    responses = [
        "```python\npassages[0]\n```",
        "```python\npassages[1]\n```",
        "```python\npassages[2]\n```",
        "```python\nfinal_answer('found it')\n```",
    ]
    session = RLMSession(generate_fn=make_stub(responses), llm_query=lambda t: "unused", max_turns=6)
    result = session.run("dummy question", ["p0", "p1", "p2"])

    confirm_count = sum(
        1 for m in result.transcript
        if m["role"] == "user" and "Before finalizing" in m["content"]
    )
    assert confirm_count == 0
    assert result.answer == "found it"
    print("[OK] session_no_confirmation_needed_after_real_exploration")


if __name__ == "__main__":
    test_sandbox_executes_code_and_captures_stdout()
    test_sandbox_truncates_long_stdout()
    test_sandbox_catches_errors_instead_of_crashing()
    test_sandbox_blocks_unsafe_builtins()
    test_sandbox_auto_prints_last_bare_expression()
    test_sandbox_auto_print_does_not_duplicate_explicit_print()
    test_sandbox_auto_print_skips_none_results()
    test_sandbox_auto_print_only_affects_last_line()
    test_code_fence_regex_accepts_label_variations()
    test_sandbox_llm_query_routes_to_injected_function_and_counts_calls()
    test_references_passages_variable_ignores_string_mentions()
    test_sandbox_llm_query_echoes_result_even_inside_a_loop()
    test_sandbox_final_answer_coerces_non_string_to_string()
    test_sandbox_final_answer_sets_done_and_answer()
    test_session_drives_multi_turn_loop_to_final_answer()
    test_session_nudges_when_no_code_block_found()
    test_session_accepts_bare_final_answer_with_no_fence()
    test_session_accepts_bare_llm_query_with_no_fence()
    test_session_nudges_once_after_ungrounded_llm_query_calls()
    test_session_nudges_even_when_word_passages_appears_in_a_string()
    test_session_nudges_once_on_exact_repeated_code()
    test_session_no_nudge_when_passages_touched()
    test_session_reports_hit_max_turns_when_never_answering()
    test_session_feeds_error_back_as_observation_not_crash()
    test_session_confirms_low_exploration_before_accepting_answer()
    test_session_no_confirmation_needed_after_real_exploration()
    print("\nAll RLM REPL mechanics tests passed.")
