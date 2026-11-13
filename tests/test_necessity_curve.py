"""Passage hiding for the relay-necessity curve. CPU only, UNTESTED ON GPU.

Real question blobs: the 100 HotpotQA items in the tracked baseline run
results/run_20261009_113254. Their gold titles are not stored in the results, so structural
properties (counts, nesting, determinism, exact-view identities) use two PSEUDO-gold passages
per item chosen by crc32. The real gold identity (supporting_facts) is exercised by the loader
test in tests/test_hotpotqa_gold_titles.py.
"""
import json
import os
import statistics as st
import subprocess
import sys
from pathlib import Path

import pytest

from lakv import necessity_curve as nc
from lakv.necessity_curve import HideSpec
from lakv.pipeline import LAKVPipeline

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ("_audit_zeroed", "_audit_random", "_audit_mismatched", "_audit_dropped")


@pytest.fixture(scope="module")
def blobs():
    d = json.loads((ROOT / "results" / "run_20261009_113254" / "experiment_results.json").read_text(encoding="utf-8"))
    return [s["question"] for s in d["A"]["per_sample"]]


def pseudo_gold_titles(blob):
    passages, _ = nc.split_blob(blob)
    order = sorted(range(len(passages)), key=lambda j: nc.stable_hash("pseudo", blob, j))
    return [nc._title_of(passages[j]) for j in order[:2]]


def frac(level):
    return HideSpec("frac", level)


def gold(level):
    return HideSpec("gold", level)


def hidden_for(blob, spec, seed=0):
    passages, _ = nc.split_blob(blob)
    g = nc.gold_indices(passages, pseudo_gold_titles(blob))
    from lakv.anchor_table import question_key
    return nc.select_hidden(passages, g, spec, question_key(blob), seed), g, passages


# ─── suffix parsing ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,core,spec", [
    ("A_hide20", "A", HideSpec("frac", 20)),
    ("A_hide50", "A", HideSpec("frac", 50)),
    ("A_hide80", "A", HideSpec("frac", 80)),
    ("A_hidegold1", "A", HideSpec("gold", 1)),
    ("A_hidegold2", "A", HideSpec("gold", 2)),
    ("A_hide50_audit_dropped", "A_audit_dropped", HideSpec("frac", 50)),
    ("A_hidegold2_audit_dropped", "A_audit_dropped", HideSpec("gold", 2)),
    ("A", "A", None),
    ("A_audit_dropped", "A_audit_dropped", None),
    ("A_nopass", "A_nopass", None),
    ("A_nopass_audit_dropped", "A_nopass_audit_dropped", None),
])
def test_split_hide_suffix(name, core, spec):
    assert nc.split_hide_suffix(name, AUDIT) == (core, spec)


def test_hide100_is_an_alias_of_nopass():
    assert nc.split_hide_suffix("A_hide100", AUDIT) == ("A_nopass", None)
    assert nc.split_hide_suffix("A_hide100_audit_dropped", AUDIT) == ("A_nopass_audit_dropped", None)


@pytest.mark.parametrize("bad", ["A_hide0", "A_hide30", "A_hide101", "A_hidegold0", "A_hidegold3", "A_nopass_hide50"])
def test_unsupported_levels_and_combinations_raise(bad):
    with pytest.raises(ValueError):
        nc.split_hide_suffix(bad, AUDIT)


def test_suffix_property_roundtrips():
    for spec in (frac(20), frac(50), frac(80), gold(1), gold(2)):
        assert nc.split_hide_suffix("A" + spec.suffix, AUDIT)[1] == spec


# ─── hashing ────────────────────────────────────────────────────────────────

def test_stable_hash_is_crc32_and_ignores_pythonhashseed():
    import zlib
    assert nc.stable_hash(0, "k", "u") == zlib.crc32(b"0|k|u") & 0x7FFFFFFF
    code = ("from lakv.necessity_curve import stable_hash; "
            "print(stable_hash(0,'key','g','[Alpha] some passage text'))")
    outs = {subprocess.check_output([sys.executable, "-c", code], env=dict(os.environ, PYTHONHASHSEED=hs), cwd=str(ROOT)
                                    ).decode().strip() for hs in ("0", "1", "4242")}
    assert len(outs) == 1


def test_no_builtin_hash_is_called_in_the_module():
    import ast
    tree = ast.parse(Path(nc.__file__).read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "hash"]
    assert calls == []


# ─── blob splitting and exact view identities (all 100 real items) ──────────

def test_every_real_blob_splits_and_roundtrips(blobs):
    assert len(blobs) == 100
    sizes = []
    for b in blobs:
        passages, qpart = nc.split_blob(b)
        assert qpart.startswith("Question: ")
        assert nc.CONTEXT_PREFIX + "\n\n".join(passages) + nc.QUESTION_MARKER + qpart[len("Question: "):] == b
        sizes.append(len(passages))
    assert sizes.count(10) == 99 and sizes[27] == 2     # item idx 27 has only 2 passages


def test_level0_view_is_the_original_blob_and_all_hidden_view_is_question_only(blobs):
    for b in blobs:
        passages, qpart = nc.split_blob(b)
        assert nc.render_view(b, passages, qpart, []) == b
        assert nc.render_view(b, passages, qpart, list(range(len(passages)))) == LAKVPipeline._question_only(b)


def test_hide100_select_hides_everything_and_matches_nopass_view(blobs):
    for b in blobs:
        hidden, _, passages = hidden_for(b, frac(100))
        assert hidden == list(range(len(passages)))
        _, qpart = nc.split_blob(b)
        assert nc.render_view(b, passages, qpart, hidden) == LAKVPipeline._question_only(b)


# ─── hidden-set properties ──────────────────────────────────────────────────

def test_fractional_counts_are_exact_for_ten_passages(blobs):
    for b in blobs:
        passages, _ = nc.split_blob(b)
        if len(passages) != 10:
            continue
        for level, k in ((20, 2), (50, 5), (80, 8), (100, 10)):
            assert len(hidden_for(b, frac(level))[0]) == k


def test_share_of_n_rule_for_items_with_fewer_than_ten_passages(blobs):
    b = blobs[27]
    passages, _ = nc.split_blob(b)
    assert len(passages) == 2
    for level in (20, 50, 80):
        hidden, g, _ = hidden_for(b, frac(level))
        assert 0 <= len(hidden) <= 2 and set(hidden) <= {0, 1}
        assert len(hidden) == (level * 2 + nc.stable_hash(0, __import__("lakv.anchor_table", fromlist=["x"]).question_key(b), "u") % 100) // 100
    assert hidden_for(b, gold(2))[0] == [0, 1]                            # both gold hidden -> question only
    assert hidden_for(b, gold(1))[0] and len(hidden_for(b, gold(1))[0]) == 1


def test_gold_axis_hides_exactly_the_requested_gold_and_never_a_distractor(blobs):
    for b in blobs:
        for level in (1, 2):
            hidden, gidx, passages = hidden_for(b, gold(level))
            assert len(hidden) == level and set(hidden) <= set(gidx)
        assert set(hidden_for(b, gold(1))[0]) <= set(hidden_for(b, gold(2))[0])


def test_hidden_sets_are_nested_across_levels(blobs):
    for b in blobs:
        prev = set()
        for level in (20, 50, 80, 100):
            cur = set(hidden_for(b, frac(level))[0])
            assert prev <= cur
            prev = cur


def test_fifty_percent_hides_exactly_one_gold_per_ten_passage_item(blobs):
    # 0.2 * 5 = 1.0 exactly, so hide50 and hidegold1 differ only by the hidden distractors
    for b in blobs:
        passages, _ = nc.split_blob(b)
        if len(passages) == 10:
            hidden, gidx, _ = hidden_for(b, frac(50))
            assert sum(1 for j in hidden if j in gidx) == 1 and len(hidden) == 5


def test_gold_is_hidden_at_the_expected_rate_on_the_fractional_axis(blobs):
    for level, expected in ((20, 0.4), (50, 1.0), (80, 1.6), (100, 2.0)):
        mean_gold = st.mean(sum(1 for j in hidden_for(b, frac(level))[0] if j in hidden_for(b, frac(level))[1])
                            for b in blobs if len(nc.split_blob(b)[0]) == 10)
        assert abs(mean_gold - expected) < 0.15, (level, mean_gold)


def test_deterministic_and_seed_sensitive(blobs):
    a = [hidden_for(b, frac(50), 0)[0] for b in blobs]
    assert a == [hidden_for(b, frac(50), 0)[0] for b in blobs]
    assert a != [hidden_for(b, frac(50), 1)[0] for b in blobs]       # a different seed hides different passages


def test_hidden_set_does_not_depend_on_the_arm(blobs):
    # make_receiver_question has no arm/audit parameter at all; same inputs -> same output
    import inspect
    assert set(inspect.signature(nc.make_receiver_question).parameters) == {"blob", "gold_titles", "spec", "hide_seed"}
    b = blobs[3]
    g = pseudo_gold_titles(b)
    assert nc.make_receiver_question(b, g, frac(50), 0) == nc.make_receiver_question(b, g, frac(50), 0)


# ─── views ──────────────────────────────────────────────────────────────────

def test_partial_view_keeps_visible_passages_in_order_and_drops_hidden_ones(blobs):
    b = blobs[0]
    g = pseudo_gold_titles(b)
    view, info = nc.make_receiver_question(b, g, frac(50), 0)
    passages, qpart = nc.split_blob(b)
    vis, vq = nc.split_blob(view)
    assert vq == qpart
    assert vis == [p for j, p in enumerate(passages) if j not in info["hidden_idx"]]
    assert info["n_hidden"] == 5 and info["n_passages"] == 10 and info["n_gold"] == 2
    for j in info["hidden_idx"]:
        assert passages[j] not in view                      # hidden passage text is really absent
    for j in range(len(passages)):
        if j not in info["hidden_idx"]:
            assert passages[j] in view                      # every visible passage is present


def test_gold2_view_keeps_every_distractor(blobs):
    b = blobs[0]
    g = pseudo_gold_titles(b)
    view, info = nc.make_receiver_question(b, g, gold(2), 0)
    vis, _ = nc.split_blob(view)
    passages, _ = nc.split_blob(b)
    gidx = nc.gold_indices(passages, g)
    assert vis == [p for j, p in enumerate(passages) if j not in gidx] and info["n_gold_hidden"] == 2


def test_info_is_json_serialisable_and_logs_hidden_titles(blobs):
    b = blobs[1]
    _, info = nc.make_receiver_question(b, pseudo_gold_titles(b), gold(1), 0)
    json.dumps(info)
    assert len(info["hidden_titles"]) == info["n_hidden"] == 1


def test_missing_or_wrong_gold_titles_fail_loudly(blobs):
    with pytest.raises(ValueError, match="gold_titles"):
        nc.make_receiver_question(blobs[0], None, gold(1), 0)
    with pytest.raises(ValueError, match="matches 0 passages"):
        nc.make_receiver_question(blobs[0], ["No Such Title", "Another"], gold(1), 0)
    with pytest.raises(ValueError):
        nc.split_blob("a GSM8K style question with no context")


def test_titles_containing_brackets_are_matched_by_prefix_not_parsed():
    blob = "Context:\n[Foo [bar] baz] Text one.\n\n[Plain] Text two.\n\n[Third] Text three.\n\nQuestion: q?"
    passages, _ = nc.split_blob(blob)
    assert nc.gold_indices(passages, ["Foo [bar] baz", "Plain"]) == [0, 1]
