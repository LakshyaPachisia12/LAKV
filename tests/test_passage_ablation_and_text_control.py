"""CPU-only tests for the passage-access ablation, the wrong-text control and
the donor-seed plumbing (paper experiments 1-3). No model is loaded."""
import random

import pytest

from lakv.evaluator import parse_audit_config_name
from lakv.pipeline import LAKVPipeline, PipelineConfig
from lakv.recursive_pipeline import RecursivePipelineConfig
from lakv_v2.pipeline.text_agent import (
    TextAgentPipeline,
    TextAgentPipelineConfig,
    question_only,
    receiver_user_content,
)

BLOB = "Context:\n[A] Alpha text.\n\n[B] Beta text.\n\nQuestion: Who wrote Alpha?"


# ---------------------------------------------------------------- name parsing
@pytest.mark.parametrize(
    "name,expected",
    [
        ("A", ("A", "A", "none", False)),
        ("A_audit_dropped", ("A", "A", "dropped", False)),
        ("A_nopass", ("A", "A_nopass", "none", True)),
        ("A_nopass_audit_dropped", ("A", "A_nopass", "dropped", True)),
        ("A_nopass_audit_mismatched", ("A", "A_nopass", "mismatched", True)),
        ("B_int8_audit_zeroed", ("B_int8", "B_int8", "zeroed", False)),
        ("text_agent", ("text_agent", "text_agent", "none", False)),
        ("text_agent_audit_mismatched", ("text_agent", "text_agent", "mismatched", False)),
        ("text_agent_nopass_audit_dropped", ("text_agent", "text_agent_nopass", "dropped", True)),
    ],
)
def test_parse_audit_config_name(name, expected):
    assert parse_audit_config_name(name) == expected


# ------------------------------------------------------------- question_only
def test_question_only_strips_passages():
    assert question_only(BLOB) == "Question: Who wrote Alpha?"
    assert LAKVPipeline._question_only(BLOB) == "Question: Who wrote Alpha?"


def test_question_only_leaves_gsm8k_style_blob_unchanged():
    q = "Janet has 3 ducks. How many eggs?"
    assert question_only(q) == q
    assert LAKVPipeline._question_only(q) == q


def test_question_only_uses_last_marker():
    tricky = "Context:\n[A] He asked.\n\nQuestion: is this a passage?\n\nQuestion: real one?"
    assert question_only(tricky) == "Question: real one?"


# -------------------------------------------------------- receiver_user_content
def test_receiver_content_real_has_prev_text_and_passages():
    c = receiver_user_content(BLOB, "Reasoner", "REAL", "none", None, True)
    assert "REAL" in c and "Alpha text" in c


def test_receiver_content_mismatched_swaps_text_only():
    c = receiver_user_content(BLOB, "Reasoner", "REAL", "mismatched", "DONOR", True)
    assert "DONOR" in c and "REAL" not in c and "Alpha text" in c


def test_receiver_content_dropped_has_no_prev_section():
    c = receiver_user_content(BLOB, "Reasoner", "REAL", "dropped", None, True)
    assert "REAL" not in c and "response" not in c and "Alpha text" in c


def test_receiver_content_nopass_removes_passages_in_every_mode():
    for mode, donor in (("none", None), ("mismatched", "DONOR"), ("dropped", None)):
        c = receiver_user_content(BLOB, "Reasoner", "REAL", mode, donor, False)
        assert "Alpha text" not in c and "Who wrote Alpha?" in c


# --------------------------------------------------------- text pipeline logic
class _Probe(TextAgentPipeline):
    """TextAgentPipeline with generation stubbed out: records each user
    message and returns a deterministic text per agent."""

    def __init__(self, *a, **k):
        super().__init__(None, None, *a, **k)
        self.seen = []

    def _generate(self, system_prompt, user_content, max_new_tokens, include_few_shot=False):
        self.seen.append(user_content)
        return f"gen{len(self.seen) - 1}"


def test_text_pipeline_real_relays_previous_text():
    p = _Probe(TextAgentPipelineConfig(dataset="hotpotqa"))
    p.run(BLOB)
    assert "gen0" in p.seen[1] and "gen1" in p.seen[2]


def test_text_pipeline_dropped_never_shows_previous_text():
    p = _Probe(TextAgentPipelineConfig(dataset="hotpotqa", text_audit_mode="dropped"))
    p.run(BLOB)
    assert "gen0" not in p.seen[1] and "gen1" not in p.seen[2]
    assert [l["mode"] for l in p.last_run_audit_logs] == ["dropped", "dropped"]


def test_text_pipeline_mismatched_uses_donor_for_same_hop():
    pool = {0: ["D0a", "D0b"], 1: ["D1a", "D1b"]}
    p = _Probe(TextAgentPipelineConfig(dataset="hotpotqa", text_audit_mode="mismatched"),
               text_pool=pool)
    p.run(BLOB)
    assert any(d in p.seen[1] for d in pool[0]) and "gen0" not in p.seen[1]
    assert any(d in p.seen[2] for d in pool[1]) and "gen1" not in p.seen[2]


def test_text_pipeline_mismatched_requires_pool():
    with pytest.raises(ValueError):
        TextAgentPipeline(None, None, TextAgentPipelineConfig(dataset="hotpotqa",
                                                              text_audit_mode="mismatched"))


def test_text_pipeline_records_pool_in_prepass():
    pool = {}
    p = _Probe(TextAgentPipelineConfig(dataset="hotpotqa"), record_text_pool=pool)
    p.run(BLOB)
    assert pool == {0: ["gen0"], 1: ["gen1"]}


def test_text_pipeline_donor_seed_changes_draw():
    pool = {0: [f"a{i}" for i in range(20)], 1: [f"b{i}" for i in range(20)]}
    picks = set()
    for seed in range(6):
        p = _Probe(TextAgentPipelineConfig(dataset="hotpotqa", text_audit_mode="mismatched",
                                           audit_donor_seed=seed), text_pool=pool)
        p.run(BLOB)
        picks.add(p.seen[1])
    assert len(picks) > 1
    # same seed -> same draw (determinism)
    a = _Probe(TextAgentPipelineConfig(dataset="hotpotqa", text_audit_mode="mismatched",
                                       audit_donor_seed=3), text_pool=pool)
    b = _Probe(TextAgentPipelineConfig(dataset="hotpotqa", text_audit_mode="mismatched",
                                       audit_donor_seed=3), text_pool=pool)
    a.run(BLOB); b.run(BLOB)
    assert a.seen == b.seen


def test_text_audit_mode_validated():
    with pytest.raises(ValueError):
        TextAgentPipelineConfig(dataset="hotpotqa", text_audit_mode="zeroed")


# ------------------------------------------------------------ config defaults
def test_defaults_preserve_every_earlier_run():
    cfg = PipelineConfig(use_layer_selection=False, compression_mode="none",
                         use_offset_correction=False, reconstruction_strategy="zeros",
                         dataset="hotpotqa")
    assert cfg.receiver_sees_passages is True and cfg.audit_donor_seed == 0
    rcfg = RecursivePipelineConfig()
    assert rcfg.aggregator_sees_passages is False and rcfg.audit_donor_seed == 0
    # seed 0 reproduces the previously hard-coded RNG stream
    assert random.Random(0).random() == random.Random(cfg.audit_donor_seed).random()


# ------------------------------------------- evaluator wiring (regression test)
def _bare_evaluator():
    """An Evaluator with no model: enough to exercise _build_pipeline's text branch."""
    from lakv.evaluator import Evaluator
    ev = Evaluator.__new__(Evaluator)
    ev.model, ev.tokenizer, ev.device = None, None, "cpu"
    return ev


def test_plain_text_agent_prepass_receives_the_recorder():
    # Regression: plain "text_agent" was special-cased and returned a pipeline
    # WITHOUT the pool recorder, so the mismatched pool stayed empty.
    ev = _bare_evaluator()
    pool = {}
    pipe, kind = ev._build_pipeline("text_agent", None, dataset="hotpotqa", record_audit_pool=pool)
    assert kind == "text" and pipe.record_text_pool is pool


def test_text_prepass_pool_fills_and_feeds_mismatched_variant():
    ev = _bare_evaluator()
    pool = {}
    pipe, _ = ev._build_pipeline("text_agent", None, dataset="hotpotqa", record_audit_pool=pool)
    # stub generation so no model is needed
    pipe._generate = lambda s, u, m, include_few_shot=False: "txt"
    for _ in range(3):
        pipe.run(BLOB)
    assert pool == {0: ["txt"] * 3, 1: ["txt"] * 3}
    mm, kind = ev._build_pipeline("text_agent_audit_mismatched", None, dataset="hotpotqa",
                                  audit_pools={"text_agent": pool})
    assert kind == "text" and mm.config.text_audit_mode == "mismatched" and mm.text_pool is pool


def test_text_nopass_variants_get_their_own_pool_key():
    ev = _bare_evaluator()
    pool = {0: ["a"], 1: ["b"]}
    mm, _ = ev._build_pipeline("text_agent_nopass_audit_mismatched", None, dataset="hotpotqa",
                               audit_pools={"text_agent_nopass": pool})
    assert mm.config.receiver_sees_passages is False and mm.text_pool is pool
    with pytest.raises(ValueError):
        ev._build_pipeline("text_agent_nopass_audit_mismatched", None, dataset="hotpotqa",
                           audit_pools={"text_agent": pool})
