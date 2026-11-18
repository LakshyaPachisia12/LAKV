"""CPU tests for the opt-in gold-title support of the HotpotQA / 2WikiMultihopQA loaders in run.py.

The default output of both loaders (and of load_dataset_samples) must stay exactly the old
{question, answer} dicts; gold titles are added only with with_gold_titles=True (the hide configs).
Tests marked 'cached' read the real 2WikiMultihopQA validation parquet from the local HF cache and
are skipped when it is absent. Nothing here touches a GPU or the network.
"""
import glob
import os
from types import SimpleNamespace

import pytest

import run as run_module
from lakv import necessity_curve as nc


def _item(q, a, titles, sents, gold):
    return {"question": q, "answer": a, "context": {"title": titles, "sentences": sents},
            "supporting_facts": {"title": gold, "sent_id": list(range(len(gold)))}}


class _Split(list):
    _fingerprint = "fp-2wiki"
    info = SimpleNamespace(version="1.0.0", builder_name="2wiki")
    cache_files = []


@pytest.fixture
def fake_2wiki(monkeypatch):
    rows = [_item("Who?", "Ann", ["T1", "T2", "T3", "T4"], [["s1."], ["s2."], ["s3."], ["s4."]], ["T2", "T2", "T4"]),
            _item("Yes?", "yes", ["U1", "U2"], [["u1."], ["u2."]], ["U1", "U2"])]
    seen = {}

    def fake_load(name, *a, **k):
        seen["name"] = name
        return {"validation": _Split(rows)}
    monkeypatch.setattr("datasets.load_dataset", fake_load)
    return seen


def test_default_2wiki_dicts_are_exactly_question_and_answer(fake_2wiki):
    out = run_module.load_2wikimultihopqa("validation", None)
    assert [set(x) for x in out] == [{"question", "answer"}] * 2
    assert out[0]["question"] == "Context:\n[T1] s1.\n\n[T2] s2.\n\n[T3] s3.\n\n[T4] s4.\n\nQuestion: Who?"


def test_gold_titles_opt_in_does_not_change_question_or_answer(fake_2wiki):
    d = run_module.load_2wikimultihopqa("validation", None)
    g = run_module.load_2wikimultihopqa("validation", None, with_gold_titles=True)
    assert [(x["question"], x["answer"]) for x in d] == [(x["question"], x["answer"]) for x in g]
    assert g[0]["gold_titles"] == ["T2", "T4"] and g[1]["gold_titles"] == ["U1", "U2"]   # deduplicated, order kept


def test_load_dataset_samples_dispatch_passes_the_flag_only_to_qa_loaders(fake_2wiki):
    assert set(run_module.load_dataset_samples("2wikimultihopqa", "validation", 1)[0]) == {"question", "answer"}
    assert "gold_titles" in run_module.load_dataset_samples("2wikimultihopqa", "validation", 1, with_gold_titles=True)[0]
    assert fake_2wiki["name"] == "framolfese/2WikiMultihopQA"


def test_2wiki_provenance_is_recorded_and_names_the_dataset(fake_2wiki):
    run_module.load_2wikimultihopqa("validation", 1)
    p = run_module.DATASET_PROVENANCE
    assert p["hf_id"] == "framolfese/2WikiMultihopQA" and p["split"] == "validation" and p["n_loaded"] == 1


def test_gold_hide_works_on_the_2wiki_shape_with_four_gold_passages(fake_2wiki):
    g = run_module.load_2wikimultihopqa("validation", None, with_gold_titles=True)
    for spec, want in ((nc.HideSpec("gold", 1), 1), (nc.HideSpec("gold", 2), 2)):
        view, info = nc.make_receiver_question(g[0]["question"], g[0]["gold_titles"], spec, 0)
        assert info["n_gold_hidden"] == min(want, 2) and info["n_hidden"] == info["n_gold_hidden"]  # no distractor hidden


# ── real cached data ──────────────────────────────────────────────────────────

def _cached_val():
    pat = os.path.expanduser("~/.cache/huggingface/hub/datasets--framolfese--2WikiMultihopQA/snapshots/*/data/validation-*.parquet")
    hits = glob.glob(pat)
    return hits[0] if hits else None


@pytest.mark.skipif(_cached_val() is None, reason="2WikiMultihopQA validation not in the local HF cache")
def test_cached_2wiki_every_gold_title_matches_exactly_one_passage_and_blob_splits():
    import pandas as pd
    df = pd.read_parquet(_cached_val())
    bad = []
    for i, r in df.iterrows():
        titles, sents = list(r["context"]["title"]), list(r["context"]["sentences"])
        blob = "Context:\n" + "\n\n".join(f"[{t}] " + " ".join(s) for t, s in zip(titles, sents)) + f"\n\nQuestion: {r['question']}"
        gold = list(dict.fromkeys(r["supporting_facts"]["title"]))
        try:
            passages, _ = nc.split_blob(blob)
            assert len(passages) == len(titles) == 10
            gi = nc.gold_indices(passages, gold)
            assert len(gi) == len(gold) and len(gold) in (2, 4)
            for spec in (nc.HideSpec("gold", 1), nc.HideSpec("gold", 2)):
                _, info = nc.make_receiver_question(blob, gold, spec, 0)
                assert info["n_gold_hidden"] == min(spec.level, len(gold)) and info["n_hidden"] == info["n_gold_hidden"]
        except Exception as e:  # collect, report the first few
            bad.append((i, type(e).__name__, str(e)[:100]))
    assert not bad, f"{len(bad)} of {len(df)} questions failed, first: {bad[:5]}"
