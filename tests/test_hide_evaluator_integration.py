"""Hide configs through the evaluator, the loader and the provenance record.

CPU only, UNTESTED ON GPU. The pipeline is replaced by a recording fake, so these tests check the
wiring (who gets which receiver view, what is logged, what is rejected), not model behavior.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import run as run_module
from lakv import necessity_curve as nc
from lakv.evaluator import AUDIT_MODE_SUFFIXES, Evaluator, validate_hide_config
from lakv.run_metadata import dataset_provenance

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def dataset():
    d = json.loads((ROOT / "results" / "run_20261009_113254" / "experiment_results.json").read_text(encoding="utf-8"))
    out = []
    for s in d["A"]["per_sample"][:6]:
        passages, _ = nc.split_blob(s["question"])
        order = sorted(range(len(passages)), key=lambda j: nc.stable_hash("pseudo", s["question"], j))
        out.append({"question": s["question"], "answer": s["gold"],
                    "gold_titles": [nc._title_of(passages[j]) for j in order[:2]]})
    return out


class FakePipe:
    """Records how run() was called."""
    def __init__(self, calls):
        self.calls = calls
        self.config = SimpleNamespace(print_raw_outputs=False)
        self.last_run_offset_logs = []
        self.last_run_audit_logs = []

    def run(self, question, **kw):
        self.calls.append((question, kw))
        return SimpleNamespace(answer="The answer is: x", hop_stats=[], hop_texts=["r"], total_compressed_mb=1.0,
                               total_original_mb=2.0, overall_compression_ratio=2.0, finalizer_latency_seconds=0.1)


def make_evaluator(monkeypatch, calls_by_cfg):
    ev = Evaluator.__new__(Evaluator)
    ev.model = SimpleNamespace(config=SimpleNamespace(_name_or_path="fake", num_hidden_layers=2), dtype="fp32")
    ev.tokenizer, ev.device = None, "cpu"

    def fake_build(self, cfg_name, *a, **k):
        return FakePipe(calls_by_cfg.setdefault(cfg_name, [])), "multi"
    monkeypatch.setattr(Evaluator, "_build_pipeline", fake_build)
    return ev


def run_cfgs(monkeypatch, tmp_path, dataset, cfgs, hide_seed=0, prov=None):
    calls = {}
    ev = make_evaluator(monkeypatch, calls)
    res = ev.run_experiment(dataset, None, configs_to_run=cfgs, n_samples=len(dataset), output_dir=str(tmp_path),
                            dataset_name="hotpotqa", hide_seed=hide_seed, dataset_provenance=prov)
    return res, calls


def test_real_and_dropped_arms_receive_identical_hidden_sets(monkeypatch, tmp_path, dataset):
    cfgs = ["A_hide50", "A_hide50_audit_dropped", "A_hidegold1", "A_hidegold1_audit_dropped"]
    res, calls = run_cfgs(monkeypatch, tmp_path, dataset, cfgs)
    for real, dropped in (("A_hide50", "A_hide50_audit_dropped"), ("A_hidegold1", "A_hidegold1_audit_dropped")):
        assert [kw["receiver_question"] for _, kw in calls[real]] == [kw["receiver_question"] for _, kw in calls[dropped]]
        assert [x["hide"]["hidden_idx"] for x in res[real]["per_sample"]] == \
               [x["hide"]["hidden_idx"] for x in res[dropped]["per_sample"]]
    # the sender still gets the full question; only the receiver view is reduced
    for (q, kw), s in zip(calls["A_hide50"], dataset):
        assert q == s["question"] and kw["receiver_question"] != s["question"]
        assert len(kw["receiver_question"]) < len(s["question"])


def test_per_sample_hide_log_and_gold_axis_semantics(monkeypatch, tmp_path, dataset):
    res, _ = run_cfgs(monkeypatch, tmp_path, dataset, ["A_hidegold2", "A_hide50"])
    for x in res["A_hidegold2"]["per_sample"]:
        h = x["hide"]
        assert h["axis"] == "gold" and h["level"] == 2 and h["n_gold_hidden"] == 2 and h["n_hidden"] == 2
        assert h["hide_seed"] == 0 and len(h["hidden_titles"]) == 2
    for x in res["A_hide50"]["per_sample"]:
        assert x["hide"]["n_hidden"] == 5 and x["hide"]["n_gold_hidden"] == 1


def test_plain_configs_are_called_exactly_as_before_and_log_nothing_new(monkeypatch, tmp_path, dataset):
    res, calls = run_cfgs(monkeypatch, tmp_path, dataset, ["A", "A_audit_dropped", "A_nopass"])
    for cfg in ("A", "A_audit_dropped", "A_nopass"):
        assert all(kw == {} for _, kw in calls[cfg])             # no receiver_question keyword at all
        assert all("hide" not in x for x in res[cfg]["per_sample"])


def test_hide100_runs_as_the_nopass_config(monkeypatch, tmp_path, dataset):
    res, calls = run_cfgs(monkeypatch, tmp_path, dataset, ["A_hide100", "A_hide100_audit_dropped"])
    assert all(kw == {} for cfg in calls for _, kw in calls[cfg])          # alias: no receiver view needed
    assert set(calls) == {"A_hide100", "A_hide100_audit_dropped"}          # (the real _build_pipeline maps them to _nopass)
    assert nc.split_hide_suffix("A_hide100", AUDIT_MODE_SUFFIXES)[0] == "A_nopass"


def test_hide_seed_changes_the_hidden_sets_and_is_recorded(monkeypatch, tmp_path, dataset):
    r0, _ = run_cfgs(monkeypatch, tmp_path / "s0", dataset, ["A_hide50"], hide_seed=0)
    r1, _ = run_cfgs(monkeypatch, tmp_path / "s1", dataset, ["A_hide50"], hide_seed=1)
    h0 = [x["hide"]["hidden_idx"] for x in r0["A_hide50"]["per_sample"]]
    h1 = [x["hide"]["hidden_idx"] for x in r1["A_hide50"]["per_sample"]]
    assert h0 != h1
    saved = json.loads((tmp_path / "s1" / "experiment_results.json").read_text(encoding="utf-8"))
    assert saved["_run_metadata"]["hide_seed"] == 1
    assert saved["_run_metadata"]["decoding_per_config"]["A_hide50"]["hide"]["level"] == 50


def test_dataset_provenance_lands_in_the_results_metadata(monkeypatch, tmp_path, dataset):
    prov = {"hf_id": "hotpotqa/hotpot_qa", "fingerprint": "abc123", "cache_revision_dirs": ["deadbeef"]}
    run_cfgs(monkeypatch, tmp_path, dataset, ["A"], prov=prov)
    saved = json.loads((tmp_path / "experiment_results.json").read_text(encoding="utf-8"))
    assert saved["_run_metadata"]["dataset_provenance"] == prov


def test_missing_gold_titles_fail_before_any_generation(monkeypatch, tmp_path, dataset):
    bad = [dict(dataset[0], gold_titles=None)] + dataset[1:]
    calls = {}
    ev = make_evaluator(monkeypatch, calls)
    with pytest.raises(ValueError, match="gold_titles"):
        ev.run_experiment(bad, None, configs_to_run=["A_hide50"], n_samples=len(bad), output_dir=str(tmp_path),
                          dataset_name="hotpotqa")
    assert not calls.get("A_hide50")


@pytest.mark.parametrize("bad", ["D_hide50", "A_nopass_hide50", "A_hide50_audit_mismatched",
                                 "A_hide50_audit_zeroed", "text_agent_hide50", "A_hidegold2_audit_random"])
def test_unsupported_hide_combinations_are_rejected_before_any_model_use(bad):
    ev = Evaluator.__new__(Evaluator)            # no model: rejection must happen before it is touched
    with pytest.raises(ValueError):
        ev._build_pipeline(bad, None)


def test_validate_hide_config_accepts_real_and_dropped_on_A():
    spec = nc.HideSpec("frac", 50)
    validate_hide_config("A", spec, "A_hide50")
    validate_hide_config("A_audit_dropped", spec, "A_hide50_audit_dropped")
    with pytest.raises(ValueError):
        validate_hide_config("A_audit_mismatched", spec, "x")


# ─── loader: gold_titles, unchanged question text, provenance ───────────────

class _FakeSplit(list):
    _fingerprint = "fp-123"
    info = SimpleNamespace(version="1.0.0", builder_name="hotpot_qa")
    cache_files = [{"filename": "/home/someone/.cache/huggingface/datasets/hotpotqa___hotpot_qa/distractor/0.0.0/REV42/x.arrow"}]


def _item(q, a, titles, sents, gold):
    return {"question": q, "answer": a, "context": {"title": titles, "sentences": sents},
            "supporting_facts": {"title": gold, "sent_id": list(range(len(gold)))}}


@pytest.fixture
def fake_hotpot(monkeypatch):
    rows = [_item("Q1?", "A1", ["Alpha", "Beta", "Gamma"], [["a1.", "a2."], ["b1."], ["c1."]], ["Alpha", "Alpha", "Beta"]),
            _item("Q2?", "A2", ["Delta", "Eps"], [["d1."], ["e1."]], ["Delta", "Eps"])]
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: {"validation": _FakeSplit(rows)})
    return rows


def test_load_hotpotqa_adds_gold_titles_and_keeps_the_question_text(fake_hotpot):
    out = run_module.load_hotpotqa("validation", None)
    assert [set(x) for x in out] == [{"question", "answer", "gold_titles"}] * 2
    assert out[0]["gold_titles"] == ["Alpha", "Beta"]                      # deduplicated, order kept
    assert out[0]["question"] == "Context:\n[Alpha] a1. a2.\n\n[Beta] b1.\n\n[Gamma] c1.\n\nQuestion: Q1?"
    # and the hiding logic can use it end to end
    view, info = nc.make_receiver_question(out[0]["question"], out[0]["gold_titles"], nc.HideSpec("gold", 1), 0)
    assert info["n_gold_hidden"] == 1 and info["n_passages"] == 3


def test_load_hotpotqa_fills_dataset_provenance(fake_hotpot):
    out = run_module.load_hotpotqa("validation", 1)
    p = run_module.DATASET_PROVENANCE
    assert p["hf_id"] == "hotpotqa/hotpot_qa" and p["split"] == "validation" and p["n_loaded"] == 1
    assert p["fingerprint"] == "fp-123" and p["cache_revision_dirs"] == ["REV42"] and p["num_rows_split"] == 2
    assert p["cache_files"] == ["distractor/0.0.0/REV42"] or p["cache_files"][0].endswith("REV42/x.arrow")
    assert "someone" not in json.dumps(p)                                  # no user directory names


def test_2wikimultihopqa_loader_is_unchanged(monkeypatch):
    src = Path(run_module.__file__).read_text(encoding="utf-8")
    a, b = src.index("def load_2wikimultihopqa("), src.index("def load_dataset_samples")
    assert "gold_titles" not in src[a:b]


def test_dataset_provenance_digests_change_with_the_data():
    split = _FakeSplit([{}])
    a = [{"question": "q", "answer": "a", "gold_titles": ["x"]}]
    p1 = dataset_provenance(split, hf_id="h", config="c", split="validation", samples=a)
    p2 = dataset_provenance(split, hf_id="h", config="c", split="validation", samples=a)
    p3 = dataset_provenance(split, hf_id="h", config="c", split="validation",
                            samples=[{"question": "q", "answer": "CHANGED", "gold_titles": ["x"]}])
    p4 = dataset_provenance(split, hf_id="h", config="c", split="validation",
                            samples=[{"question": "q", "answer": "a", "gold_titles": ["y"]}])
    assert p1 == p2
    assert p1["questions_answers_sha256"] != p3["questions_answers_sha256"]
    assert p1["gold_titles_sha256"] != p4["gold_titles_sha256"] and p1["questions_answers_sha256"] == p4["questions_answers_sha256"]
    json.dumps(p1)


def test_dataset_provenance_never_raises_on_odd_objects():
    p = dataset_provenance(object(), hf_id="h", config="c", split="validation", samples=[])
    assert p["hf_id"] == "h" and p["n_loaded"] == 0
