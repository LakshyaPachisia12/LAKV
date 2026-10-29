"""
Tests for run.py's dataset loaders, focused on the 2WikiMultihopQA adapter
added to extend the causal-audit methodology (lakv/causal_audit.py) beyond
HotpotQA. No GPU/model needed -- datasets.load_dataset is monkeypatched with
a small fake object shaped exactly like the real framolfese/2WikiMultihopQA
rows (verified against the real dataset via the HF datasets-server API
before writing this).

Covers section 19.A/B/C of the 2Wiki extension brief (dataset loading,
schema/adapter mapping, donor-distinctness prerequisites) at the adapter
level -- NOT a second causal-audit test suite. The substitution/donor-
selection mechanism itself (KVAuditPool, apply_causal_audit) is already
covered by tests/test_causal_audit.py and is deliberately not duplicated
here; this file only checks that the new loader feeds that existing
machinery correctly-shaped, genuinely-distinct examples.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run
from lakv.pipeline import PROMPT_SETS
from lakv.evaluator import TWO_AGENT_BENCH_PROMPTS
from lakv.anchor_table import question_key


class _FakeDatasetDict(dict):
    """Minimal stand-in for a datasets.DatasetDict: ds[split] -> list of items."""


def _make_fake_item(qid: str, question: str, answer: str, titles, sentences):
    return {
        "id": qid,
        "question": question,
        "answer": answer,
        "type": "compositional",
        "evidences": [],
        "supporting_facts": {"title": [titles[0]], "sent_id": [0]},
        "context": {"title": titles, "sentences": sentences},
    }


def _fake_2wiki_rows():
    item_a = _make_fake_item(
        "id-a",
        "Who is the mother of the director of film Polish-Russian War?",
        "Malgorzata Braunek",
        ["Polish-Russian War (film)", "Xawery Zulawski"],
        [["Polish-Russian War is a 2009 film."], ["Xawery Zulawski is its director."]],
    )
    item_b = _make_fake_item(
        "id-b",
        "What is the capital of the country where the Eiffel Tower is located?",
        "Paris",
        ["Eiffel Tower", "France"],
        [["The Eiffel Tower is in Paris."], ["France is a country in Europe."]],
    )
    return [item_a, item_b]


def test_load_2wikimultihopqa_maps_schema_like_hotpotqa(monkeypatch):
    rows = _fake_2wiki_rows()
    fake_ds = _FakeDatasetDict(validation=rows)

    def fake_load_dataset(repo_id, *args, **kwargs):
        assert repo_id == "framolfese/2WikiMultihopQA"
        return fake_ds

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    out = run.load_2wikimultihopqa(split="validation", n=None)
    assert len(out) == 2
    for item, raw in zip(out, rows):
        assert set(item.keys()) == {"question", "answer"}
        assert item["answer"] == raw["answer"]
        # question field must still carry the real question text plus the
        # formatted context passages, same convention as load_hotpotqa.
        assert raw["question"] in item["question"]
        assert "Context:\n" in item["question"]
        for title in raw["context"]["title"]:
            assert f"[{title}]" in item["question"]


def test_load_2wikimultihopqa_respects_n(monkeypatch):
    rows = _fake_2wiki_rows()
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: _FakeDatasetDict(validation=rows))
    out = run.load_2wikimultihopqa(split="validation", n=1)
    assert len(out) == 1


def test_load_dataset_samples_dispatches_2wikimultihopqa(monkeypatch):
    rows = _fake_2wiki_rows()
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: _FakeDatasetDict(validation=rows))
    out = run.load_dataset_samples("2wikimultihopqa", "validation", n=None)
    assert len(out) == 2
    assert out[0]["answer"] == rows[0]["answer"]


def test_2wikimultihopqa_examples_hash_to_distinct_question_keys(monkeypatch):
    """Prerequisite for mismatched-donor selection (KVAuditPool excludes by
    question_key, see lakv/causal_audit.py): two real, distinct 2Wiki
    examples must never collide on question_key, since the full formatted
    "Context:...Question:..." string is what gets hashed."""
    rows = _fake_2wiki_rows()
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: _FakeDatasetDict(validation=rows))
    out = run.load_dataset_samples("2wikimultihopqa", "validation", n=None)
    keys = [question_key(item["question"]) for item in out]
    assert len(set(keys)) == len(keys)


def test_prompt_sets_reuses_hotpotqa_prompts_for_2wikimultihopqa():
    # Must be the SAME prompts as HotpotQA (same object), not a redesigned
    # or merely textually-similar copy -- the whole point of the extension
    # is "same methodology, different dataset."
    assert PROMPT_SETS["2wikimultihopqa"] is PROMPT_SETS["hotpotqa"]


def test_two_agent_bench_prompts_reuses_hotpotqa_prompts_for_2wikimultihopqa():
    assert TWO_AGENT_BENCH_PROMPTS["2wikimultihopqa"] is TWO_AGENT_BENCH_PROMPTS["hotpotqa"]


def test_dataset_choice_registered_in_cli():
    # main() builds its ArgumentParser internally rather than exposing one
    # at module scope -- confirm the --dataset flag's choices were actually
    # extended, not just the loader dict above.
    import inspect
    src = inspect.getsource(run.main)
    assert '"2wikimultihopqa"' in src
