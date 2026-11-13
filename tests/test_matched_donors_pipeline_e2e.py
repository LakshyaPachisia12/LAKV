"""End-to-end CPU test of the matched-donor plumbing through the REAL LAKVPipeline.

UNTESTED ON GPU. Uses a tiny random Qwen2 (fp32, CPU) and a stub tokenizer, so it
checks wiring and the cache-length bookkeeping, not accuracy or text quality, and
runs under whatever transformers is installed here (4.x), not the 5.x used on GPU.

The key assertion: a hop's cache REBUILT from (standard prompt ids + stored generated
ids truncated to cache_gen_len) has exactly the length, and numerically the content,
of the cache the real decode path produced. self_replay logs that comparison.
"""
import pytest
import torch

from lakv import donor_traces as dt
from lakv.anchor_table import question_key
from lakv.causal_audit import DonorUnavailable
from lakv.pipeline import LAKVPipeline, PipelineConfig

@pytest.fixture(autouse=True)
def _manual_decode_loops_for_hops_ge_1(monkeypatch):
    """The installed transformers (4.44) cannot run the pipeline's "prime, then hand off
    to model.generate()" fast path (hops >= 1): it raises an IndexError inside rotary
    embedding on the UNMODIFIED branch tip too. Forcing the pipeline's own manual
    per-token loop for those hops lets this CPU test run. Hop 0 (no injection) still uses
    the real model.generate() path. The generate()-handoff cache bookkeeping for hops >= 1
    (final token not forwarded) is therefore NOT covered here; the GPU self-replay step
    (rebuilt length must equal the real cache length, comparable=True) is the check for it.
    """
    monkeypatch.setattr(LAKVPipeline, "_uses_nondefault_rope_scaling", lambda self: True)


QUESTION = "Context:\n[Alpha] Alpha is a thing. Beta is another.\n\n[Gamma] Gamma is third.\n\nQuestion: What is Alpha?"
GOLD = "thing"


class StubTok:
    """Char-level tokenizer. ids 4..63; id 1 is EOS (never produced by the prompt)."""

    def __init__(self, eos_token_id=1):
        self.eos_token_id = eos_token_id

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return "||".join(f"{m['role']}:{m['content']}" for m in messages)

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        return {"input_ids": torch.tensor([[4 + (ord(c) % 60) for c in text]])}

    def decode(self, ids, skip_special_tokens=True):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return " ".join(f"t{int(i)}" for i in ids if not (skip_special_tokens and int(i) == self.eos_token_id))


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                      rope_theta=10000.0)
    m = Qwen2ForCausalLM(cfg).eval()
    m.generation_config.pad_token_id = 0
    m.generation_config.eos_token_id = 1
    return m


def make_pipe(model, mode="none", donor_source=None, eos=1, max_new=7, nopass=False):
    cfg = PipelineConfig(use_layer_selection=False, compression_mode="none", use_offset_correction=False,
                         reconstruction_strategy="zeros", dataset="hotpotqa",
                         intermediate_max_new_tokens=max_new, final_max_new_tokens=max_new,
                         causal_audit_mode=mode, receiver_sees_passages=not nopass)
    tok = StubTok(eos_token_id=eos)
    model.generation_config.eos_token_id = eos
    return LAKVPipeline(model, tok, cfg, device="cpu", donor_source=donor_source)


def test_standard_prompt_helper_matches_run_and_suffix_changes_only_generation(tiny_model):
    pipe = make_pipe(tiny_model)
    base = pipe._agent_input_ids(QUESTION, 0)
    assert torch.equal(base, pipe._agent_input_ids(QUESTION, 0, system_suffix=""))
    assert pipe._agent_input_ids(QUESTION, 0, system_suffix=" DEFEND X").shape[1] > base.shape[1]
    # nopass: later agents see only the question, so the prompt is shorter than agent 0's would be with passages
    nop = make_pipe(tiny_model, nopass=True)
    assert nop._agent_input_ids(QUESTION, 1).shape[1] < pipe._agent_input_ids(QUESTION, 1).shape[1]


def test_run_records_trace_and_cache_gen_len_is_consistent(tiny_model):
    pipe = make_pipe(tiny_model)
    pipe.run(QUESTION)
    tr = pipe._run_trace
    assert set(tr) == {0, 1}
    for h in (0, 1):
        assert 0 <= tr[h]["cache_gen_len"] <= len(tr[h]["gen_ids"])
        assert tr[h]["prompt_ids"] == pipe._agent_input_ids(QUESTION, h)[0].tolist()


@pytest.mark.parametrize("eos,label", [(1, "stops at max tokens"), (None, "eos = first generated token")])
def test_self_replay_rebuild_matches_real_chain_cache(tiny_model, eos, label):
    if eos is None:
        # make the model's own first greedy token the EOS id so hops end immediately/early
        probe = make_pipe(tiny_model)
        probe.run(QUESTION)
        eos = probe._run_trace[0]["gen_ids"][0]
    pipe = make_pipe(tiny_model, mode="self_replay", donor_source=dt.SelfReplaySource(), eos=eos)
    pipe.run(QUESTION)
    logs = {l["agent_idx"]: l for l in pipe.last_run_audit_logs if l.get("mode") == "self_replay"}
    assert set(logs) == {1, 2}, label
    for a, l in logs.items():
        r = l["rebuild_vs_real"]
        assert r["comparable"], f"{label}: rebuilt length differs from the real cache at agent {a}: {r}"
        assert r["max_abs_diff"] < 1e-3 and r["min_cosine"] > 0.9999, (label, a, r)
        assert l["pad_positions"] == 0


def test_self_replay_run_reproduces_the_plain_run_on_fp32_cpu(tiny_model):
    plain = make_pipe(tiny_model).run(QUESTION)
    replay = make_pipe(tiny_model, mode="self_replay", donor_source=dt.SelfReplaySource()).run(QUESTION)
    assert replay.answer == plain.answer and replay.hop_texts == plain.hop_texts


def test_matched_passages_traces_roundtrip_through_real_pipeline(tiny_model, tmp_path):
    gen_pipe = make_pipe(tiny_model)
    store = dt.TraceStore(tmp_path / "mp")
    sample = {"question": QUESTION, "answer": GOLD}
    dt.generate_donor_traces(gen_pipe, [sample], "matched_passages", store, base_seed=0, log=lambda *_: None)
    rec = store.load(question_key(QUESTION))
    assert rec["seeds"][0] == dt.stable_seed(0, "matched_passages", question_key(QUESTION), 0)
    assert all(0 <= c <= len(g) for g, c in zip(rec["gen_ids"], rec["cache_gen_len"]))
    # determinism: same seed -> identical sampled ids
    store2 = dt.TraceStore(tmp_path / "mp2")
    dt.generate_donor_traces(make_pipe(tiny_model), [sample], "matched_passages", store2, base_seed=0, log=lambda *_: None)
    assert store2.load(question_key(QUESTION))["gen_ids"] == rec["gen_ids"]
    assert gen_pipe.config.generation_kwargs["do_sample"] is False  # restored after sampled generation

    # scored chain consuming the stored donor: unmodified injection, standard prompts, no padding
    scored = make_pipe(tiny_model, mode="matched_passages", donor_source=dt.TraceReplaySource(store, "matched_passages"))
    scored.run(QUESTION)
    logs = {l["agent_idx"]: l for l in scored.last_run_audit_logs if l.get("mode") == "matched_passages"}
    assert set(logs) == {1, 2}
    p0 = scored._agent_input_ids(QUESTION, 0).shape[1]
    assert logs[1]["donor_cache_len"] == p0 + rec["cache_gen_len"][0]
    assert all(l["pad_positions"] == 0 and l["handling"] == "rebuilt_from_token_ids_unmodified" for l in logs.values())


def test_matched_wrong_answer_cache_is_built_from_standard_prompts(tiny_model, tmp_path, monkeypatch):
    monkeypatch.setattr(dt, "propose_distractor", lambda pipeline, q: "Zanzibar island")
    store = dt.TraceStore(tmp_path / "wa")
    sample = {"question": QUESTION, "answer": GOLD}
    dt.generate_donor_traces(make_pipe(tiny_model), [sample], "matched_wrong_answer", store, 0, log=lambda *_: None)
    rec = store.load(question_key(QUESTION))
    assert rec["distractor"] == "Zanzibar island" and len(rec["defends_distractor"]) == 2

    # rebuilt cache == prefill of STANDARD prompt + stored ids; it must NOT equal a prefill that used the defend prompt
    pipe = make_pipe(tiny_model)
    src = dt.TraceReplaySource(store, "matched_wrong_answer")
    real = make_pipe(tiny_model).run  # noqa: F841 (ensures pipeline builds)
    kv, log = src.fetch(agent_idx=1, question_key=question_key(QUESTION), real_kv=((torch.zeros(1, 2, 3, 8),) * 2,) * 2,
                        pipeline=pipe, question=QUESTION)
    std_prompt = pipe._agent_input_ids(QUESTION, 0)[0].tolist()
    expected = dt.rebuild_cache(tiny_model, [std_prompt + rec["gen_ids"][0][:rec["cache_gen_len"][0]]], "cpu")
    assert dt.kv_diff_stats(kv, expected)["max_abs_diff"] == 0.0
    defend_prompt = pipe._agent_input_ids(QUESTION, 0, system_suffix=dt.DEFEND_SUFFIX_TEMPLATE.format(x="Zanzibar island"))[0].tolist()
    contaminated = dt.rebuild_cache(tiny_model, [defend_prompt + rec["gen_ids"][0][:rec["cache_gen_len"][0]]], "cpu")
    assert not dt.kv_diff_stats(kv, contaminated).get("comparable", False)  # different length: instruction absent


def test_missing_trace_raises_donor_unavailable_through_pipeline(tiny_model, tmp_path):
    src = dt.TraceReplaySource(dt.TraceStore(tmp_path / "empty"), "matched_passages")
    pipe = make_pipe(tiny_model, mode="matched_passages", donor_source=src)
    with pytest.raises(DonorUnavailable) as ei:
        pipe.run(QUESTION)
    assert ei.value.reason == "no_trace_stored"
