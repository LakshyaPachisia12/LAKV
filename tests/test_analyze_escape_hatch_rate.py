"""CPU-only unit test for scripts/analyze_escape_hatch_rate.py's
classify() -- the four-way split (escape_hatch / delegated_done /
gave_up / delegated_timeout) the escape-hatch dose-response analysis is
built on."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_escape_hatch_rate import classify


def _rec(llm_query_calls, hit_max_turns):
    return {"llm_query_calls": llm_query_calls, "hit_max_turns": hit_max_turns}


def test_classify_four_way_split():
    records = [
        _rec(0, False),  # escape_hatch
        _rec(0, False),  # escape_hatch
        _rec(3, False),  # delegated_done
        _rec(0, True),   # gave_up
        _rec(2, True),   # delegated_timeout
    ]
    result = classify(records)
    assert result == {
        "n": 5, "escape_hatch": 2, "delegated_done": 1,
        "gave_up": 1, "delegated_timeout": 1,
    }


def test_classify_all_escape_hatch():
    records = [_rec(0, False) for _ in range(4)]
    result = classify(records)
    assert result["escape_hatch"] == 4
    assert result["delegated_done"] == result["gave_up"] == result["delegated_timeout"] == 0


def test_classify_empty_list():
    assert classify([]) == {
        "n": 0, "escape_hatch": 0, "delegated_done": 0,
        "gave_up": 0, "delegated_timeout": 0,
    }
