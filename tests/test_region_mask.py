"""CPU tests for the opt-in region-ablation mask (config `mask_region`, names A_nopass_region_*_only).

Default (None) must leave every existing path untouched; with it on, the hidden ranges must partition the
relayed cache exactly (passages = tokens inside the Context block, reasoning = every generated segment)."""
import pytest
import torch

from lakv.cache_corruption import hidden_mask_from_ranges
from lakv.evaluator import REGION_CONFIGS, ABLATION_CONFIGS
from lakv.pipeline import LAKVPipeline, PipelineConfig
from lakv.regions import cache_segments, passage_token_span, region_ranges

QUESTION = "Context:\n[Alpha] Alpha is a thing. Beta is another.\n\n[Gamma] Gamma is third.\n\nQuestion: What is Alpha?"


class CharTok:
    """Round-trippable char-level tokenizer (ASCII), with offsets and the chat-template call the pipeline uses."""
    eos_token_id = 1

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"

    def __call__(self, text, return_tensors=None, add_special_tokens=False, return_offsets_mapping=False):
        ids = [ord(c) for c in text]
        if return_offsets_mapping:
            return {"input_ids": ids, "offset_mapping": [(i, i + 1) for i in range(len(ids))]}
        return {"input_ids": torch.tensor([ids])}

    def decode(self, ids, skip_special_tokens=True):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return "".join(chr(int(i)) for i in ids if not (skip_special_tokens and int(i) == 1))


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=160, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096)
    m = Qwen2ForCausalLM(cfg).eval()
    m.generation_config.pad_token_id = 0
    m.generation_config.eos_token_id = 1
    return m


def make_pipe(model, **kw):
    kw.setdefault("receiver_sees_passages", False)
    cfg = PipelineConfig(use_layer_selection=False, compression_mode="none", use_offset_correction=False,
                         reconstruction_strategy="zeros", dataset="hotpotqa",
                         intermediate_max_new_tokens=6, final_max_new_tokens=6, **kw)
    cfg.generation_kwargs = dict(cfg.generation_kwargs, do_sample=False)
    return LAKVPipeline(model, CharTok(), cfg, device="cpu")


def test_hidden_mask_from_ranges_exact():
    h = hidden_mask_from_ranges(10, [(2, 4), (7, 10)])
    assert torch.nonzero(h).flatten().tolist() == [2, 3, 7, 8, 9]
    assert not hidden_mask_from_ranges(5, []).any()
    with pytest.raises(ValueError):
        hidden_mask_from_ranges(5, [(3, 6)])


def test_passage_span_is_exactly_the_context_text():
    tok = CharTok()
    text = tok.apply_chat_template([{"role": "system", "content": "S"}, {"role": "user", "content": QUESTION}])
    ids = [ord(c) for c in text]
    a, b = passage_token_span(tok, ids)
    assert text[a:b] == QUESTION[len("Context:\n"):QUESTION.rfind("\n\nQuestion: ")]


def test_passage_span_rejects_missing_block_and_roundtrip_failure():
    tok = CharTok()
    with pytest.raises(ValueError):
        passage_token_span(tok, [ord(c) for c in "<user>no context here"])

    class Bad(CharTok):
        def __call__(self, text, **kw):
            out = super().__call__(text, **kw)
            out["input_ids"] = out["input_ids"][:-1] + [99]
            return out
    ids = [ord(c) for c in "<user>" + QUESTION]
    with pytest.raises(ValueError):
        passage_token_span(Bad(), ids)


def fake_trace():
    return {0: {"prompt_ids": list(range(30)), "cache_gen_len": 8},
            1: {"prompt_ids": list(range(12)), "cache_gen_len": 5}}


def test_segments_partition_cache_at_each_hop():
    tr = fake_trace()
    s1 = cache_segments(tr, 1)
    assert [(s["start"], s["end"]) for s in s1] == [(0, 30), (30, 38)]
    s2 = cache_segments(tr, 2)
    assert [(s["start"], s["end"]) for s in s2] == [(0, 30), (30, 38), (38, 50), (50, 55)]
    assert all(x["end"] == y["start"] for x, y in zip(s2, s2[1:]))


def test_reasoning_ranges_are_all_generated_segments_and_layout_mismatch_raises():
    tr = fake_trace()
    assert region_ranges("reasoning", tr, 1, CharTok(), 38) == [(30, 38)]
    assert region_ranges("reasoning", tr, 2, CharTok(), 55) == [(30, 38), (50, 55)]
    with pytest.raises(ValueError):
        region_ranges("reasoning", tr, 2, CharTok(), 54)
    with pytest.raises(ValueError):
        region_ranges("nonsense", tr, 2, CharTok(), 55)


def test_config_names_registered_and_legacy_untouched():
    assert set(REGION_CONFIGS) == {"A_nopass_region_passages_only", "A_nopass_region_reasoning_only"}
    assert not (set(REGION_CONFIGS) & set(ABLATION_CONFIGS))
    assert REGION_CONFIGS["A_nopass_region_passages_only"]["mask"] == "reasoning"
    assert REGION_CONFIGS["A_nopass_region_reasoning_only"]["mask"] == "passages"
    assert PipelineConfig(use_layer_selection=False, compression_mode="none", use_offset_correction=False, reconstruction_strategy="zeros").mask_region is None


def test_default_off_is_bit_identical_and_records_nothing(tiny_model):
    a = make_pipe(tiny_model)
    b = make_pipe(tiny_model)
    ra, rb = a.run(QUESTION), b.run(QUESTION)
    assert ra.answer == rb.answer and ra.hop_texts == rb.hop_texts
    assert not any(l.get("mode") == "region" for l in a.last_run_audit_logs)


@pytest.mark.parametrize("mask", ["reasoning", "passages"])
def test_pipeline_masks_exactly_the_region_at_every_receiving_hop(tiny_model, mask):
    pipe = make_pipe(tiny_model, mask_region=mask)
    seen = []
    orig = pipe._apply_hidden_mask

    def spy(m, hidden, cache_len, position_offset, full_len=None):
        seen.append(hidden.clone())
        return orig(m, hidden, cache_len, position_offset, full_len)
    pipe._apply_hidden_mask = spy
    pipe.run(QUESTION)
    logs = [l for l in pipe.last_run_audit_logs if l.get("mode") == "region"]
    assert [l["agent_idx"] for l in logs] == [1, 2]
    assert len(seen) >= 2
    tr = pipe._run_trace
    # the last mask handed to a model call at each hop is checked (generate() hands off more than once)
    for l in logs:
        segs = cache_segments(tr, l["agent_idx"])
        assert l["cache_len"] == segs[-1]["end"]
        gen = {i for s in segs if s["kind"] == "gen" for i in range(s["start"], s["end"])}
        match = [h for h in seen if h.shape[0] == l["cache_len"]]
        assert match
        for h in match:
            hidden_idx = set(torch.nonzero(h).flatten().tolist())
            if mask == "reasoning":
                assert hidden_idx == gen and len(gen) > 0
            else:
                a, b = passage_token_span(CharTok(), tr[0]["prompt_ids"])
                assert hidden_idx == set(range(a, b)) and not (hidden_idx & gen)


def test_region_mask_rejected_with_audit_or_visible_passages(tiny_model):
    p = make_pipe(tiny_model, mask_region="reasoning", causal_audit_mode="real_pad10")
    with pytest.raises(ValueError):
        p.run(QUESTION)
    p2 = make_pipe(tiny_model, mask_region="reasoning", receiver_sees_passages=True)
    with pytest.raises(ValueError):
        p2.run(QUESTION)


def test_real_qwen_tokenizer_span_matches_passage_text():
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", local_files_only=True)
    except Exception:
        pytest.skip("Qwen tokenizer not in the local cache")
    text = tok.apply_chat_template([{"role": "system", "content": "You are a helper."},
                                    {"role": "user", "content": QUESTION}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    a, b = passage_token_span(tok, ids)
    inner = tok.decode(ids[a:b])
    want = QUESTION[len("Context:\n"):QUESTION.rfind("\n\nQuestion: ")]
    assert inner.strip() in want and len(inner.strip()) >= len(want.strip()) - 6
