import pytest

from lakv.rlm.actions import parse_action, validate_action
from lakv.rlm.errors import InvalidActionError


def test_parse_valid_final_action():
    a = parse_action('{"action": "final", "answer": "42"}')
    assert a.action_type == "final"
    assert a.args["answer"] == "42"


def test_parse_action_wrapped_in_prose_and_markdown_fence():
    raw = 'Sure, here is my action:\n```json\n{"action": "final", "answer": "42"}\n```\nDone.'
    a = parse_action(raw)
    assert a.action_type == "final"
    assert a.args["answer"] == "42"


def test_unknown_action_type_rejected():
    with pytest.raises(InvalidActionError):
        validate_action({"action": "delete_everything"})


def test_missing_required_field_rejected():
    with pytest.raises(InvalidActionError):
        validate_action({"action": "get_chunk"})  # missing chunk_id


def test_wrong_type_for_field_rejected():
    with pytest.raises(InvalidActionError):
        validate_action({"action": "slice_context", "token_start": "zero", "token_end": 10})


def test_optional_field_defaults_applied():
    a = validate_action({"action": "chunk_context", "chunk_size": 512})
    assert a.args["overlap"] == 0


def test_not_a_json_object_rejected():
    with pytest.raises(InvalidActionError):
        parse_action("this is not json at all")


def test_llm_query_requires_list_and_string():
    a = validate_action({"action": "llm_query", "chunk_ids": ["chunk_000001"], "query": "what happened?"})
    assert a.args["chunk_ids"] == ["chunk_000001"]

    with pytest.raises(InvalidActionError):
        validate_action({"action": "llm_query", "chunk_ids": "chunk_000001", "query": "x"})


def test_action_identity_is_stable_for_cycle_detection():
    a1 = validate_action({"action": "get_chunk", "chunk_id": "chunk_000001"})
    a2 = validate_action({"action": "get_chunk", "chunk_id": "chunk_000001"})
    a3 = validate_action({"action": "get_chunk", "chunk_id": "chunk_000002"})
    assert a1.identity() == a2.identity()
    assert a1.identity() != a3.identity()
