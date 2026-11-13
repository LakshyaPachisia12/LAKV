"""Matched-donor audit modes: CPU unit tests only.

UNTESTED ON GPU. Nothing here loads a real model. The rebuild tests use a
tiny random Qwen2 in fp32 on CPU (transformers as installed, 4.x); the real
pipeline runs transformers 5.x on GPU, which these tests do not exercise.
"""
import json
import os
import random
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from lakv import causal_audit as ca
from lakv import donor_traces as dt
from lakv.causal_audit import (DiskKVAuditPool, DonorUnavailable, NoMatchedDonor, apply_causal_audit)
from lakv.evaluator import (AUDIT_NEW_MODE_BASES, Evaluator, parse_audit_config_name,
                            valid_audit_config_names)


def kv(seq, layers=2, fill=1.0, heads=2, dim=4):
    return tuple((torch.full((1, heads, seq, dim), float(fill) + li), torch.full((1, heads, seq, dim), -float(fill) - li))
                 for li in range(layers))


# ─── stable seeding ─────────────────────────────────────────────────────────

def test_stable_seed_known_value_and_range():
    assert dt.stable_seed(0, "matched_passages", "abc", 1) == dt.stable_seed(0, "matched_passages", "abc", 1)
    assert dt.stable_seed(0, "m", "k", 0) != dt.stable_seed(0, "m", "k", 1)
    assert 0 <= dt.stable_seed("x") <= 0x7FFFFFFF
    import zlib
    assert dt.stable_seed(0, "m", "k", 1) == zlib.crc32(b"0|m|k|1") & 0x7FFFFFFF


def test_stable_seed_does_not_depend_on_pythonhashseed():
    code = "from lakv.donor_traces import stable_seed; print(stable_seed(0,'matched_passages','abc123',1))"
    outs = set()
    for hs in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        outs.add(subprocess.check_output([sys.executable, "-c", code], env=env,
                                         cwd=os.path.dirname(os.path.dirname(__file__))).decode().strip())
    assert len(outs) == 1


# ─── suffixes ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,expected", [
    ("A_audit_length_matched", ("A", "A", "length_matched", False)),
    ("A_audit_matched_passages", ("A", "A", "matched_passages", False)),
    ("A_audit_matched_wrong_answer", ("A", "A", "matched_wrong_answer", False)),
    ("A_audit_self_replay", ("A", "A", "self_replay", False)),
    ("A_nopass_audit_length_matched", ("A", "A_nopass", "length_matched", True)),
    ("A_nopass_audit_matched_wrong_answer", ("A", "A_nopass", "matched_wrong_answer", True)),
    ("A_audit_mismatched", ("A", "A", "mismatched", False)),   # unchanged
    ("A_audit_dropped", ("A", "A", "dropped", False)),         # unchanged
])
def test_suffix_parsing(name, expected):
    assert parse_audit_config_name(name) == expected


def test_valid_names_all_parse():
    names = valid_audit_config_names()
    assert len(names) == 8
    for n in names:
        assert parse_audit_config_name(n)[2] in ca.DONOR_SOURCE_MODES


def test_non_A_base_rejected_before_any_model_use():
    ev = Evaluator.__new__(Evaluator)
    assert AUDIT_NEW_MODE_BASES == ("A",)
    with pytest.raises(ValueError, match="A"):
        ev._build_donor_sources(["D_audit_length_matched"], [], [], 0, None, "legacy", "hotpotqa",
                                False, 0, "/nonexistent", None, 100)


# ─── disk-backed length_matched pool ────────────────────────────────────────

def make_pool(tmp_path, lens, agent_idx=1):
    pool = DiskKVAuditPool(tmp_path / "pool")
    for i, L in enumerate(lens):
        pool.add(agent_idx, f"q{i}", kv(L, fill=i + 1), question_text=f"question {i}")
    pool.mark_complete()
    return pool


def test_window_is_inclusive_five_percent(tmp_path):
    pool = make_pool(tmp_path, [95, 105, 106, 94, 100, 200])
    elig = {e["len"] for e in pool.eligible(1, "nobody", 100)}
    assert elig == {95, 105, 100}  # 106 and 94 are outside +-5%


def test_own_question_excluded(tmp_path):
    pool = make_pool(tmp_path, [100, 100])
    assert [e["key"] for e in pool.eligible(1, "q0", 100)] == ["q1"]


def test_no_donor_raises_with_diagnostics(tmp_path):
    pool = make_pool(tmp_path, [50, 200])
    with pytest.raises(NoMatchedDonor) as ei:
        pool.select_length_matched(1, "me", 100, "cpu")
    log = ei.value.log
    assert log["skipped"] is True and "5%" in log["skip_reason"]
    assert log["target_cache_len"] == 100 and log["nearest_donor_cache_len"] == 50 and log["pool_size"] == 2
    assert isinstance(ei.value, DonorUnavailable)


def test_injected_donor_is_bit_identical_and_never_padded_or_truncated(tmp_path):
    pool = make_pool(tmp_path, [103])  # 3% longer than target 100
    donor = kv(103, fill=1)
    out, log = pool.select_length_matched(1, "me", 100, "cpu", rng=random.Random(0))
    assert out[0][0].shape[2] == 103  # NOT truncated to 100
    for (k, v), (dk, dv) in zip(out, donor):
        assert torch.equal(k, dk) and torch.equal(v, dv)
    assert log["pad_positions"] == 0 and log["truncated_positions"] == 0
    assert log["handling"] == "unmodified_within_tolerance" and log["skipped"] is False
    assert abs(log["len_ratio"] - 1.03) < 1e-9

    pool2 = make_pool(tmp_path / "b", [97])  # 3% shorter: must not be padded
    out2, log2 = pool2.select_length_matched(1, "me", 100, "cpu")
    assert out2[0][0].shape[2] == 97 and log2["pad_positions"] == 0
    assert not torch.any(out2[0][0] == 0)  # no zero tail


def test_exact_length_labelled_exact(tmp_path):
    pool = make_pool(tmp_path, [100])
    assert pool.select_length_matched(1, "me", 100, "cpu")[1]["handling"] == "exact"


def test_exactly_one_donor_file_loaded_per_fetch(tmp_path, monkeypatch):
    pool = make_pool(tmp_path, [100, 101, 102, 103, 99])
    calls = []
    real_load = torch.load
    monkeypatch.setattr(ca.torch, "load", lambda *a, **k: (calls.append(a[0]), real_load(*a, **k))[1])
    pool.select_length_matched(1, "me", 100, "cpu", rng=random.Random(3))
    assert len(calls) == 1


def test_index_in_ram_holds_no_tensors(tmp_path):
    pool = make_pool(tmp_path, [100, 110])
    json.dumps(pool._index)  # serialisable => metadata only
    assert all("kv" not in e for es in pool._index.values() for e in es)


def test_selection_deterministic_under_fixed_seed_and_uniform_over_eligible(tmp_path):
    pool = make_pool(tmp_path, [100, 101, 102, 99, 98])
    a = pool.select_length_matched(1, "me", 100, "cpu", rng=random.Random(7))[1]["donor_question_key"]
    b = pool.select_length_matched(1, "me", 100, "cpu", rng=random.Random(7))[1]["donor_question_key"]
    assert a == b
    seen = {pool.select_length_matched(1, "me", 100, "cpu", rng=random.Random(s))[1]["donor_question_key"]
            for s in range(60)}
    assert len(seen) >= 4  # draws are spread over the eligible donors, not just the closest


def test_pool_reopens_from_index_for_resume(tmp_path):
    pool = make_pool(tmp_path, [100, 105])
    again = DiskKVAuditPool.open(tmp_path / "pool")
    assert again.complete and again.size(1) == 2
    out, log = again.select_length_matched(1, "me", 100, "cpu", rng=random.Random(0))
    assert log["donor_question_key"] in ("q0", "q1")
    # an unfinished pool re-opens as incomplete
    p2 = DiskKVAuditPool(tmp_path / "p2"); p2.add(1, "z", kv(10))
    assert not DiskKVAuditPool.open(tmp_path / "p2").complete


def test_duplicate_question_not_added_twice(tmp_path):
    pool = DiskKVAuditPool(tmp_path / "p")
    pool.add(1, "same", kv(10)); pool.add(1, "same", kv(10))
    assert pool.size(1) == 1


def test_skip_rate_falls_as_pool_grows_on_synthetic_lengths(tmp_path):
    rng = random.Random(0)
    lens = [int(rng.lognormvariate(7.4, 0.25)) for _ in range(400)]
    targets = lens[:60]
    def skip_rate(m):
        pool = [L for L in lens[60:60 + m]]
        return sum(not any(abs(L - t) <= 0.05 * t for L in pool) for t in targets) / len(targets)
    assert skip_rate(20) > skip_rate(100) > skip_rate(300) - 1e-9
    assert skip_rate(20) > 0.05


# ─── dispatch ───────────────────────────────────────────────────────────────

class FakeSource:
    def __init__(self, out=None, exc=None):
        self.out, self.exc, self.calls = out, exc, []
    def fetch(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        return self.out, {"agent_idx": kw["agent_idx"], "skipped": False, "marker": 1}


@pytest.mark.parametrize("mode", list(ca.DONOR_SOURCE_MODES))
def test_dispatch_uses_donor_source_and_merges_log(mode):
    donor = kv(7, fill=9)
    src = FakeSource(out=donor)
    out, log = apply_causal_audit(mode, kv(10), 2, "qk", donor_source=src, context={"pipeline": "P", "question": "Q"})
    assert out is donor and log["mode"] == mode and log["marker"] == 1
    assert src.calls[0]["pipeline"] == "P" and src.calls[0]["question"] == "Q" and src.calls[0]["agent_idx"] == 2


def test_dispatch_requires_source_and_propagates_skip():
    with pytest.raises(ValueError):
        apply_causal_audit("length_matched", kv(10), 1, "qk")
    with pytest.raises(NoMatchedDonor):
        apply_causal_audit("length_matched", kv(10), 1, "qk", donor_source=FakeSource(exc=NoMatchedDonor("x")))


def test_old_modes_unchanged_by_new_signature():
    out, log = apply_causal_audit("zeroed", kv(4), 1, "qk")
    assert log == {"mode": "zeroed"} and torch.all(out[0][0] == 0)


# ─── trace store / donor generation (fake pipeline) ─────────────────────────

class FakeTok:
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"t{i}" for i in ids)


class FakePipe:
    """Stands in for LAKVPipeline for generation-side tests."""

    def __init__(self):
        self.tokenizer = FakeTok()
        self.device = "cpu"
        self.config = SimpleNamespace(generation_kwargs={"do_sample": False, "temperature": 0.7},
                                      intermediate_max_new_tokens=512)
        self._last_generated_ids = []
        self.suffix_calls = []
        self.sampling_seen = []

    def _agent_input_ids(self, question, hop, system_suffix=""):
        self.suffix_calls.append((hop, system_suffix))
        return torch.arange(5 + hop).unsqueeze(0)

    def _generate_intermediate_with_hidden(self, ids, kv_in, max_new, position_offset, receiver_prompt_len):
        self.sampling_seen.append(self.config.generation_kwargs["do_sample"])
        n = 6
        if self.config.generation_kwargs["do_sample"]:
            gen = torch.randint(0, 50, (n,)).tolist()   # consumes the seeded RNG
        else:
            gen = list(range(n))
        self._last_generated_ids = gen
        injected = kv_in[0][0].shape[2] if kv_in is not None else 0
        total = injected + ids.shape[1] + (n - 1)       # generate() never forwards the last token
        return kv(total, layers=1), None, "text"


def samples(n=3):
    return [{"question": f"Context: c{i}\n\nQuestion: q{i}", "answer": f"gold{i}"} for i in range(n)]


def test_distractor_filter():
    assert dt.distractor_ok("Paris", "London")
    assert not dt.distractor_ok("london", "London")           # exact match
    assert not dt.distractor_ok("", "London")
    assert not dt.distractor_ok("   ", "London")
    assert not dt.distractor_ok("the london area", "London")  # F1 >= 0.2 with gold


def test_matched_passages_traces_are_ids_only_seeded_and_resumable(tmp_path):
    pipe, store = FakePipe(), dt.TraceStore(tmp_path / "t")
    c1 = dt.generate_donor_traces(pipe, samples(), "matched_passages", store, base_seed=0, log=lambda *_: None)
    assert c1 == {"written": 3, "reused": 0, "unavailable": 0}
    rec = store.load(ca_key := __import__("lakv.anchor_table", fromlist=["x"]).question_key(samples()[0]["question"]))
    assert set(rec) >= {"gen_ids", "cache_gen_len", "seeds", "version", "question_key"}
    assert rec["seeds"] == [dt.stable_seed(0, "matched_passages", rec["question_key"], h) for h in (0, 1)]
    assert rec["cache_gen_len"] == [5, 5]  # gen length 6, final token not cached
    assert all(isinstance(t, int) for g in rec["gen_ids"] for t in g)
    # nothing but ids and scalars: no tensors, no prompt text, no instruction
    blob = json.dumps(rec)
    for forbidden in ("Another reader", "strongest case", "defend", "kv", "Context"):
        assert forbidden not in blob
    # same seed -> same sampled ids on a rerun into a fresh store
    store2 = dt.TraceStore(tmp_path / "t2")
    dt.generate_donor_traces(FakePipe(), samples(), "matched_passages", store2, base_seed=0, log=lambda *_: None)
    assert store2.load(rec["question_key"])["gen_ids"] == rec["gen_ids"]
    # a different base seed -> different ids
    store3 = dt.TraceStore(tmp_path / "t3")
    dt.generate_donor_traces(FakePipe(), samples(), "matched_passages", store3, base_seed=1, log=lambda *_: None)
    assert store3.load(rec["question_key"])["gen_ids"] != rec["gen_ids"]
    # resume: existing traces are kept
    c2 = dt.generate_donor_traces(pipe, samples(), "matched_passages", store, base_seed=0, log=lambda *_: None)
    assert c2 == {"written": 0, "reused": 3, "unavailable": 0}


def test_matched_passages_sampling_on_during_generation_and_restored(tmp_path):
    pipe = FakePipe()
    dt.generate_donor_traces(pipe, samples(1), "matched_passages", dt.TraceStore(tmp_path), 0, log=lambda *_: None)
    assert True in pipe.sampling_seen and False in pipe.sampling_seen  # sampled chain + greedy vacuity check
    assert pipe.config.generation_kwargs["do_sample"] is False          # restored


def test_wrong_answer_instruction_only_at_generation_never_stored(tmp_path, monkeypatch):
    monkeypatch.setattr(dt, "propose_distractor", lambda pipeline, q: "Zanzibar")
    pipe, store = FakePipe(), dt.TraceStore(tmp_path)
    dt.generate_donor_traces(pipe, samples(1), "matched_wrong_answer", store, 0, log=lambda *_: None)
    # the instruction was applied to BOTH hops while generating...
    suffixes = [s for _, s in pipe.suffix_calls]
    assert len(suffixes) == 2 and all("Zanzibar" in s and "strongest case" in s for s in suffixes)
    # ...and is absent from the stored trace (only the validated answer string is kept)
    key = __import__("lakv.anchor_table", fromlist=["x"]).question_key(samples(1)[0]["question"])
    rec = store.load(key)
    assert rec["distractor"] == "Zanzibar" and rec["seeds"] is None
    assert "strongest case" not in json.dumps(rec) and "Another reader" not in json.dumps(rec)
    assert pipe.config.generation_kwargs["do_sample"] is False


def test_rejected_distractor_marks_trace_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(dt, "propose_distractor", lambda pipeline, q: "gold0")  # equals gold -> rejected
    pipe, store = FakePipe(), dt.TraceStore(tmp_path)
    c = dt.generate_donor_traces(pipe, samples(1), "matched_wrong_answer", store, 0, log=lambda *_: None)
    assert c["unavailable"] == 1 and pipe.suffix_calls == []  # no generation attempted
    src = dt.TraceReplaySource(store, "matched_wrong_answer")
    key = __import__("lakv.anchor_table", fromlist=["x"]).question_key(samples(1)[0]["question"])
    with pytest.raises(DonorUnavailable) as ei:
        src.fetch(agent_idx=1, question_key=key, real_kv=kv(10), pipeline=pipe, question=samples(1)[0]["question"])
    assert ei.value.reason == "distractor_failed" and ei.value.log["skipped"] is True


# ─── rebuild by prefill (tiny real model, CPU fp32) ─────────────────────────

@pytest.fixture(scope="module")
def tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    return Qwen2ForCausalLM(cfg).eval()


def close(a, b, tol=1e-5):
    return all(torch.allclose(x, y, atol=tol) for (ka, va), (kb, vb) in zip(a, b) for x, y in ((ka, kb), (va, vb)))


def test_two_segment_rebuild_equals_one_shot_prefill(tiny_model):
    seg0, seg1 = [3, 5, 7, 9, 11, 13], [2, 4, 6, 8, 10]
    two = dt.rebuild_cache(tiny_model, [seg0, seg1], "cpu")
    one = dt.rebuild_cache(tiny_model, [seg0 + seg1], "cpu")
    assert two[0][0].shape[2] == 11 and close(two, one)


def test_rebuild_equals_incremental_decode_cache(tiny_model):
    from lakv.pipeline import LAKVPipeline
    ids = [3, 5, 7, 9, 11, 13, 15, 17]
    base = tiny_model.model
    with torch.no_grad():
        out = base(input_ids=torch.tensor([ids[:4]]), use_cache=True)
        cache, pos = out.past_key_values, 4
        for t in ids[4:]:
            out = base(input_ids=torch.tensor([[t]]), past_key_values=cache,
                       position_ids=torch.tensor([[pos]]), use_cache=True)
            cache, pos = out.past_key_values, pos + 1
    incremental = LAKVPipeline._to_tuple(cache)
    rebuilt = dt.rebuild_cache(tiny_model, [ids], "cpu")
    stats = dt.kv_diff_stats(incremental, rebuilt)
    assert stats["comparable"] and stats["max_abs_diff"] < 1e-4 and stats["min_cosine"] > 0.99999


def test_trace_segments_drop_uncached_final_token():
    segs = dt.trace_segments([[1, 2, 3], [4, 5]], [[10, 11, 12, 13], [20, 21, 22]], [3, 2], upto_hop=1)
    assert segs == [[1, 2, 3, 10, 11, 12], [4, 5, 20, 21]]
    assert dt.trace_segments([[1, 2, 3], [4, 5]], [[10, 11, 12, 13], [20]], [3, 1], upto_hop=0) == [[1, 2, 3, 10, 11, 12]]


def test_kv_diff_stats_identical_and_perturbed():
    a = kv(6, fill=2)
    s0 = dt.kv_diff_stats(a, kv(6, fill=2))
    assert s0["max_abs_diff"] == 0 and abs(s0["min_cosine"] - 1) < 1e-6
    b = tuple((k + 0.5, v) for k, v in a)
    s1 = dt.kv_diff_stats(a, b)
    assert s1["max_abs_diff"] == pytest.approx(0.5) and s1["min_cosine"] < 1
    assert dt.kv_diff_stats(a, kv(7))["comparable"] is False


class ReplayPipe:
    """Tiny-model stand-in exposing exactly what the replay sources use."""

    def __init__(self, model):
        self.model, self.device = model, "cpu"
        self.suffix_calls, self._run_trace = [], {}

    def _agent_input_ids(self, question, hop, system_suffix=""):
        self.suffix_calls.append((hop, system_suffix))
        base = [[1, 2, 3, 4], [5, 6, 7, 8, 9]][hop]
        return torch.tensor([base])


def test_trace_replay_uses_standard_prompts_and_injects_unmodified(tiny_model, tmp_path):
    store = dt.TraceStore(tmp_path)
    q = "Q"
    key = __import__("lakv.anchor_table", fromlist=["x"]).question_key(q)
    store.save(key, {"mode": "matched_passages", "seeds": [1, 2], "gen_ids": [[10, 11, 12], [20, 21, 22, 23]],
                     "cache_gen_len": [2, 3], "distractor": None})
    pipe = ReplayPipe(tiny_model)
    src = dt.TraceReplaySource(store, "matched_passages")
    kv1, log1 = src.fetch(agent_idx=1, question_key=key, real_kv=kv(50), pipeline=pipe, question=q)
    kv2, log2 = src.fetch(agent_idx=2, question_key=key, real_kv=kv(90), pipeline=pipe, question=q)
    assert kv1[0][0].shape[2] == 4 + 2 and kv2[0][0].shape[2] == 6 + 5 + 3     # prompt + cached gen, per hop
    assert all(sfx == "" for _, sfx in pipe.suffix_calls)                       # STANDARD prompts only
    assert log1["pad_positions"] == 0 and log1["handling"] == "rebuilt_from_token_ids_unmodified"
    assert log2["donor_cache_len"] == 14 and log2["target_cache_len"] == 90 and log2["rebuilt_by_prefill"] is True
    expected = dt.rebuild_cache(tiny_model, [[1, 2, 3, 4, 10, 11], [5, 6, 7, 8, 9, 20, 21, 22]], "cpu")
    assert close(kv2, expected)
    with pytest.raises(DonorUnavailable):
        src.fetch(agent_idx=1, question_key="missing", real_kv=kv(5), pipeline=pipe, question=q)


def test_self_replay_matches_real_and_reports_diff(tiny_model):
    pipe = ReplayPipe(tiny_model)
    pipe._run_trace = {0: {"prompt_ids": [1, 2, 3, 4], "gen_ids": [10, 11, 12], "cache_gen_len": 2},
                       1: {"prompt_ids": [5, 6, 7, 8, 9], "gen_ids": [20, 21, 22, 23], "cache_gen_len": 3}}
    real = dt.rebuild_cache(tiny_model, [[1, 2, 3, 4, 10, 11], [5, 6, 7, 8, 9, 20, 21, 22]], "cpu")
    out, log = dt.SelfReplaySource().fetch(agent_idx=2, question_key="k", real_kv=real, pipeline=pipe, question="Q")
    assert log["rebuild_vs_real"]["comparable"] and log["rebuild_vs_real"]["max_abs_diff"] < 1e-6
    assert log["donor_type"] == "self_replay" and log["pad_positions"] == 0
    with pytest.raises(DonorUnavailable):
        dt.SelfReplaySource().fetch(agent_idx=2, question_key="k", real_kv=real,
                                    pipeline=ReplayPipe(tiny_model), question="Q")


# ─── stats and evaluator: skipped samples ───────────────────────────────────

def test_stats_pairs_skip_skipped_samples():
    from lakv.stats import _align_by_idx, compare
    a = [{"idx": i, "correct": True, "f1": 1.0} for i in range(4)]
    b = [{"idx": 0, "correct": False, "f1": 0.0}, {"idx": 1, "correct": None, "f1": None, "skipped": True},
         {"idx": 2, "correct": True, "f1": 1.0}, {"idx": 3, "correct": None, "f1": None, "skipped": True}]
    assert [x["idx"] for x, _ in _align_by_idx(a, b)] == [0, 2]
    res = {"A": {"summary": {"accuracy": 1.0}, "per_sample": a}, "X": {"summary": {"accuracy": 0.5}, "per_sample": b}}
    out = compare(res, "A", "X")
    assert out["n_pairs"] == 2 and out["n_skipped_b"] == 2 and out["accuracy_a_on_pairs"] == 1.0
    assert out["accuracy_b_on_pairs"] == 0.5


def test_evaluator_records_skips_and_excludes_them_from_accuracy(tmp_path, monkeypatch):
    class Pipe:
        config = SimpleNamespace(print_raw_outputs=False)
        last_run_offset_logs = []

        def __init__(self):
            self.last_run_audit_logs = []

        def run(self, question):
            if "q1" in question:
                self.last_run_audit_logs = [{"agent_idx": 1, "mode": "length_matched", "skipped": False}]
                raise NoMatchedDonor("no donor within 5% of target length", {"mode": "length_matched", "agent_idx": 2})
            self.last_run_audit_logs = []
            return SimpleNamespace(answer="The answer is: gold0" if "q0" in question else "The answer is: nope",
                                   hop_stats=[], hop_texts=["r"], total_compressed_mb=1.0, total_original_mb=2.0,
                                   overall_compression_ratio=2.0, finalizer_latency_seconds=0.1)

    ev = Evaluator.__new__(Evaluator)
    ev.model = SimpleNamespace(config=SimpleNamespace(_name_or_path="fake", num_hidden_layers=2), dtype="fp32")
    ev.tokenizer, ev.device = None, "cpu"
    monkeypatch.setattr(Evaluator, "_build_pipeline", lambda self, *a, **k: (Pipe(), "multi"))
    monkeypatch.setattr(Evaluator, "_build_donor_sources", lambda self, *a, **k: {})
    data = [{"question": f"Context: c\n\nQuestion: q{i}", "answer": f"gold{i}"} for i in range(3)]
    res = ev.run_experiment(data, None, configs_to_run=["A_audit_length_matched"], n_samples=3,
                            output_dir=str(tmp_path), dataset_name="hotpotqa")
    r = res["A_audit_length_matched"]
    s, ps = r["summary"], r["per_sample"]
    assert s["n_samples"] == 3 and s["n_skipped"] == 1 and s["n_scored"] == 2
    assert s["accuracy"] == pytest.approx(s["n_correct"] / 2)            # denominator excludes the skip
    assert s["parse_failures"] <= 1 and s["status"] == "completed"
    skipped = [x for x in ps if x.get("skipped")]
    assert len(skipped) == 1 and skipped[0]["correct"] is None and skipped[0]["skip_reason"].startswith("no donor")
    assert any(l.get("skipped") for l in skipped[0]["audit_logs"])
    saved = json.loads((tmp_path / "experiment_results.json").read_text(encoding="utf-8"))
    assert "_run_metadata" in saved and saved["A_audit_length_matched"]["summary"]["n_skipped"] == 1
