from mock_backend import MockBackend

from lakv.rlm.config import RLMBudget, RLMConfig
from lakv.rlm.engine import RLMEngine
from lakv.rlm.metrics import compute_metrics

CONTEXT_TEXT = "alpha beta gamma delta epsilon zeta eta theta iota kappa"


def make_engine(root_actions, **budget_overrides):
    backend = MockBackend(root_actions)
    budget = RLMBudget(**budget_overrides)
    config = RLMConfig(budget=budget)
    return RLMEngine(backend, config), backend


def test_immediate_final():
    engine, _ = make_engine(['{"action": "final", "answer": "the answer"}'])
    result = engine.run(CONTEXT_TEXT, "what is it?")
    assert result.answer == "the answer"
    assert result.termination_reason == "final"
    assert result.n_calls == 1


def test_chunk_then_llm_query_then_final():
    engine, backend = make_engine([
        '{"action": "chunk_context", "chunk_size": 3}',
        '{"action": "llm_query", "chunk_ids": ["chunk_000000"], "query": "q"}',
        '{"action": "final", "answer": "done"}',
    ])
    result = engine.run(CONTEXT_TEXT, "what is it?")
    assert result.termination_reason == "final"
    assert result.answer == "done"
    # 3 root calls + 1 sub call
    assert result.n_calls == 4
    access_events = result.trace.context_access_events()
    assert any(e["access_type"] == "llm_query" and e["chunk_id"] == "chunk_000000" for e in access_events)


def test_max_iterations_exhaustion_without_final():
    varying_actions = [
        '{"action": "inspect_context"}',
        '{"action": "chunk_context", "chunk_size": 3}',
        '{"action": "list_chunks"}',
        '{"action": "inspect_context"}',
        '{"action": "list_chunks"}',
    ]
    engine, _ = make_engine(varying_actions, max_root_iterations=3)
    result = engine.run(CONTEXT_TEXT, "what is it?")
    assert result.termination_reason == "max_iterations"
    assert result.n_calls == 3


def test_cycle_detection_on_repeated_identical_action():
    engine, _ = make_engine(['{"action": "inspect_context"}'] * 10)
    result = engine.run(CONTEXT_TEXT, "what is it?")
    assert result.termination_reason == "cycle_detected"
    # default max_repeated_identical_actions is 3
    assert result.n_calls == 3


def test_budget_exceeded_on_max_total_llm_calls():
    engine, _ = make_engine([
        '{"action": "chunk_context", "chunk_size": 3}',
        '{"action": "llm_query", "chunk_ids": ["chunk_000000"], "query": "q"}',
        '{"action": "final", "answer": "unreachable"}',
    ], max_total_llm_calls=2)
    result = engine.run(CONTEXT_TEXT, "what is it?")
    assert result.termination_reason == "budget_exceeded:max_total_llm_calls"


def test_rlm_query_rejected_when_depth_budget_is_zero():
    engine, _ = make_engine([
        '{"action": "chunk_context", "chunk_size": 3}',
        '{"action": "rlm_query", "chunk_ids": ["chunk_000000"], "query": "q"}',
        '{"action": "final", "answer": "done"}',
    ], max_depth=0)
    result = engine.run(CONTEXT_TEXT, "what is it?")
    assert result.termination_reason == "final"
    assert result.answer == "done"
    # rlm_query was rejected, not executed -> no depth-1 invocation was ever recorded
    assert all(e["depth"] == 0 for e in result.trace.invocation_events())


def test_rlm_query_recurses_when_depth_budget_allows():
    engine, backend = make_engine([
        '{"action": "chunk_context", "chunk_size": 3}',
        '{"action": "rlm_query", "chunk_ids": ["chunk_000000"], "query": "sub question"}',
        '{"action": "final", "answer": "root final"}',
    ], max_depth=1)
    # the recursive child node also asks the (same) mock backend for a root
    # action — queue one more so its own loop terminates with FINAL too.
    backend._root_actions.insert(2, '{"action": "final", "answer": "child final"}')
    result = engine.run(CONTEXT_TEXT, "what is it?")
    assert result.termination_reason == "final"
    depths = {e["depth"] for e in result.trace.invocation_events()}
    assert 1 in depths  # the recursive call actually happened at depth 1
    recursive_access = [e for e in result.trace.context_access_events() if e["led_to_recursion"]]
    assert len(recursive_access) == 1


def test_aggregate_note_used_as_best_effort_when_budget_exhausted():
    engine, _ = make_engine([
        '{"action": "aggregate", "note": "partial finding X"}',
        '{"action": "inspect_context"}',
    ], max_root_iterations=2)
    result = engine.run(CONTEXT_TEXT, "what is it?")
    assert result.termination_reason == "max_iterations"
    assert result.answer == "partial finding X"


def test_metrics_computable_after_run_with_no_crash():
    engine, _ = make_engine([
        '{"action": "chunk_context", "chunk_size": 3}',
        '{"action": "get_chunk", "chunk_id": "chunk_000000"}',
        '{"action": "get_chunk", "chunk_id": "chunk_000000"}',  # revisit
        '{"action": "final", "answer": "done"}',
    ])
    result = engine.run(CONTEXT_TEXT, "what is it?")
    m = result.metrics
    assert m["unique_chunks_accessed"] == 1
    assert m["total_chunk_accesses"] == 2
    assert m["repeated_accesses"] == 1
    assert m["revisit_rate"] == 0.5
