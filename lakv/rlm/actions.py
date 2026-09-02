"""Structured action interface (spec section 12) — NOT a code-execution REPL.

The root model emits one JSON object per step describing what it wants to
do next. We parse and validate that JSON against a fixed schema before
executing anything. This deliberately satisfies spec section 13's fallback
("implement a controlled action interface" instead of unrestricted
model-generated Python) rather than section 13's full REPL option — a
sandboxed executor is real security/correctness surface that isn't needed
to get the access-pattern evidence this phase is after. If a later phase
finds the model needs genuinely programmatic (loop/branch) exploration that
this fixed action set can't express, that's the trigger to build the
REPL sandbox — not before.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from lakv.rlm.errors import InvalidActionError

# action_type -> {"required": {field: type}, "optional": {field: (type, default)}}
ACTION_SCHEMAS: Dict[str, dict] = {
    "inspect_context": {"required": {}, "optional": {"chunk_id": (str, None)}},
    "slice_context": {"required": {"token_start": int, "token_end": int}, "optional": {}},
    "search_context": {"required": {"pattern": str}, "optional": {}},
    "chunk_context": {"required": {"chunk_size": int}, "optional": {"overlap": (int, 0)}},
    "get_chunk": {"required": {"chunk_id": str}, "optional": {}},
    "list_chunks": {"required": {}, "optional": {}},
    "llm_query": {"required": {"chunk_ids": list, "query": str}, "optional": {}},
    "rlm_query": {"required": {"chunk_ids": list, "query": str}, "optional": {}},
    "aggregate": {"required": {"note": str}, "optional": {"chunk_ids": (list, [])}},
    "final": {"required": {"answer": str}, "optional": {}},
}

ACTION_TYPES = frozenset(ACTION_SCHEMAS)


@dataclass
class Action:
    action_type: str
    args: Dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""

    def identity(self) -> tuple:
        """Hashable fingerprint used for repeated-identical-action / cycle
        detection (spec section 11)."""
        try:
            return (self.action_type, json.dumps(self.args, sort_keys=True))
        except TypeError:
            return (self.action_type, str(self.args))


def validate_action(data: dict, raw_text: str = "") -> Action:
    if not isinstance(data, dict):
        raise InvalidActionError(f"action must be a JSON object, got {type(data).__name__}")

    action_type = data.get("action")
    if action_type not in ACTION_SCHEMAS:
        raise InvalidActionError(
            f"unknown action type {action_type!r}; must be one of {sorted(ACTION_TYPES)}"
        )

    schema = ACTION_SCHEMAS[action_type]
    args: Dict[str, Any] = {}

    for name, expected_type in schema["required"].items():
        if name not in data:
            raise InvalidActionError(f"action {action_type!r} missing required field {name!r}")
        val = data[name]
        if not isinstance(val, expected_type):
            raise InvalidActionError(
                f"action {action_type!r} field {name!r} must be {expected_type.__name__}, "
                f"got {type(val).__name__}"
            )
        args[name] = val

    for name, (expected_type, default) in schema["optional"].items():
        if name in data and data[name] is not None:
            val = data[name]
            if not isinstance(val, expected_type):
                raise InvalidActionError(
                    f"action {action_type!r} field {name!r} must be {expected_type.__name__}, "
                    f"got {type(val).__name__}"
                )
            args[name] = val
        else:
            args[name] = default

    return Action(action_type=action_type, args=args, raw_text=raw_text)


def parse_action(raw_text: str) -> Action:
    """Parse the model's raw output into a validated Action. Tolerates the
    model wrapping the JSON in prose/markdown fences by extracting the first
    balanced {...} block before giving up."""
    text = raw_text.strip()
    try:
        data = json.loads(text)
        return validate_action(data, raw_text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        data = json.loads(candidate)
                        return validate_action(data, raw_text)
                    except json.JSONDecodeError:
                        break

    raise InvalidActionError(f"could not parse a valid JSON action from model output: {raw_text!r}")
