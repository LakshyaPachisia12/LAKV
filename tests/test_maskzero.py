"""CPU tests for the opt-in `_maskzero` attention masking of corruption-modified cache positions.

Default OFF must be bit-identical to the existing behaviour; with the mask on it must cover exactly
the modified range and leave every position id unchanged. Uses a tiny random Qwen2 (fp32, CPU)."""
import pytest
import torch

from lakv.cache_corruption import apply_corruption, hidden_mask_from_log, split_maskzero
from lakv.evaluator import parse_audit_config_name
from lakv.pipeline import LAKVPipeline, PipelineConfig


class StubTok:
    eos_token_id = 1

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return "||".join(f"{m['role']}:{m['content']}" for m in messages)

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        return {"input_ids": torch.tensor([[4 + (ord(c) % 60) for c in text]])}

    def decode(self, ids, skip_special_tokens=True):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return " ".join(f"t{int(i)}" for i in ids if not (skip_special_tokens and int(i) == 1))


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


def make_pipe(model, **kw):
    cfg = PipelineConfig(use_layer_selection=False, compression_mode="none", use_offset_correction=False,
                         reconstruction_strategy="zeros", dataset="hotpotqa",
                         intermediate_max_new_tokens=6, final_max_new_tokens=6, **kw)
    cfg.generation_kwargs = dict(cfg.generation_kwargs, do_sample=False)
    return LAKVPipeline(model, StubTok(), cfg, device="cpu")


def rand_kv(L, layers=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return tuple((torch.randn(1, 2, L, 8, generator=g), torch.randn(1, 2, L, 8, generator=g)) for _ in range(layers))


def log_for(op, pct, L=40, carried=None, target="real"):
    kv = rand_kv(L)
    out, info = apply_corruption(kv, op, pct)
    return out, {"mode": f"{target}_{op}{pct}", "target": target, **info}


# ── name handling ──
def test_split_maskzero_and_parse_unchanged():
    assert split_maskzero("A_nopass_audit_real_pad10_maskzero") == ("A_nopass_audit_real_pad10", True)
    assert split_maskzero("A_nopass_audit_real_pad10") == ("A_nopass_audit_real_pad10", False)
    for n in ("A_nopass_audit_real_pad10", "A_nopass_audit_mismatched", "A_nopass_audit_real_fill25"):
        assert parse_audit_config_name(n + "_maskzero") == parse_audit_config_name(n)


# ── the mask covers exactly the modified range ──
def test_mask_pad_covers_only_appended():
    out, lg = log_for("pad", 10)
    h = hidden_mask_from_log(lg)
    L0, L1 = lg["length_before"], lg["length_after"]
    assert h.shape[0] == L1 == out[0][0].shape[2]
    assert not h[:L0].any() and h[L0:].all() and int(h.sum()) == L1 - L0
    assert bool((out[0][0][:, :, L0:, :] == 0).all())


@pytest.mark.parametrize("op", ["fill", "fillhead"])
def test_mask_fill_covers_exactly_the_zeroed_range(op):
    out, lg = log_for(op, 25)
    h = hidden_mask_from_log(lg)
    idx = torch.nonzero(h).flatten().tolist()
    assert idx == list(range(lg["modified_start"], lg["modified_end"]))
    zeroed = (out[0][0].abs().sum(dim=(0, 1, 3)) == 0).nonzero().flatten().tolist()
    assert idx == zeroed


def test_mask_trunc_hides_nothing_and_carry_is_kept():
    _, lg = log_for("trunc", 25)
    assert not hidden_mask_from_log(lg).any()
    carried = torch.zeros(lg["length_before"], dtype=torch.bool)
    carried[3:6] = True
    h = hidden_mask_from_log(lg, carried=carried)
    assert h.shape[0] == lg["length_after"] and torch.nonzero(h).flatten().tolist() == [3, 4, 5]


def test_mask_real_carries_previous_hop_and_lm_does_not():
    _, lg = log_for("pad", 10)
    carried = torch.zeros(lg["length_before"], dtype=torch.bool)
    carried[0:2] = True
    h = hidden_mask_from_log(lg, carried=carried)
    assert h[:2].all() and int(h.sum()) == 2 + lg["length_after"] - lg["length_before"]
    _, lg2 = log_for("pad", 10, target="lm")
    assert int(hidden_mask_from_log(lg2, carried=carried).sum()) == lg2["length_after"] - lg2["length_before"]


def test_mask_mismatched_hides_only_zero_padded_tail():
    h = hidden_mask_from_log({"mode": "mismatched", "target_cache_len": 50, "pad_positions": 12})
    assert torch.nonzero(h).flatten().tolist() == list(range(38, 50))
    assert not hidden_mask_from_log({"mode": "mismatched", "target_cache_len": 50, "pad_positions": 0}).any()


def test_mask_unsupported_mode_raises():
    with pytest.raises(ValueError):
        hidden_mask_from_log({"mode": "zeroed"})


# ── pipeline plumbing ──
def test_apply_hidden_mask_exact_and_position_ids(tiny_model):
    pipe = make_pipe(tiny_model)
    hidden = torch.zeros(10, dtype=torch.bool)
    hidden[7:] = True
    ones = torch.ones((1, 14), dtype=torch.long)
    m, extra = pipe._apply_hidden_mask(ones, hidden, 10, 0)
    assert m[0, :7].eq(1).all() and m[0, 7:10].eq(0).all() and m[0, 10:].eq(1).all() and extra == {}
    assert ones.eq(1).all()                                         # input not mutated
    m2, extra2 = pipe._apply_hidden_mask(torch.ones((1, 15), dtype=torch.long), hidden, 10, 0, full_len=15)
    assert extra2["position_ids"].tolist() == [list(range(15))]
    with pytest.raises(ValueError):
        pipe._apply_hidden_mask(ones, hidden, 10, 3)               # offset-corrected configs unsupported


def test_default_off_bit_identical_and_mask_keeps_positions(tiny_model, monkeypatch):
    pipe = make_pipe(tiny_model)
    ids = StubTok()("question text", return_tensors="pt")["input_ids"]
    kv, _ = apply_corruption(rand_kv(30), "pad", 20)
    L = kv[0][0].shape[2]
    hidden = torch.zeros(L, dtype=torch.bool)
    hidden[30:] = True

    seen = []
    orig = tiny_model.forward

    def spy(*a, **k):
        seen.append({"pos": None if k.get("position_ids") is None else k["position_ids"].clone(),
                     "mask": None if k.get("attention_mask") is None else k["attention_mask"].clone(),
                     "n_in": k["input_ids"].shape[1]})
        return orig(*a, **k)

    def run(h):
        seen.clear()
        monkeypatch.setattr(tiny_model, "forward", spy)
        try:
            return pipe._generate(ids, tuple((k.clone(), v.clone()) for k, v in kv), max_new_tokens=6,
                                  hidden_positions=h)[0], list(seen)
        finally:
            monkeypatch.setattr(tiny_model, "forward", orig)

    base_text, base_seen = run(None)
    nohide_text, _ = run(torch.zeros(L, dtype=torch.bool))
    assert base_text == nohide_text                                 # all-False mask == no mask
    assert base_seen[0]["mask"].eq(1).all()                         # default path: all-ones mask, as before
    masked_text, masked_seen = run(hidden)
    P = ids.shape[1]
    # prime step: identical position ids, mask zero exactly on the hidden range
    assert torch.equal(masked_seen[0]["pos"], base_seen[0]["pos"])
    assert torch.equal(masked_seen[0]["pos"], torch.arange(L, L + P).unsqueeze(0))
    assert masked_seen[0]["mask"][0, 30:L].eq(0).all() and masked_seen[0]["mask"][0, :30].eq(1).all()
    assert masked_seen[0]["mask"][0, L:].eq(1).all()
    # decode steps (generate() handoff): the unmasked path leaves position_ids implicit (derived from the
    # cache length); the masked path passes them explicitly and they must equal exactly that: L+P, L+P+1, ...
    assert len(masked_seen) > 1 and len(masked_seen) == len(base_seen)
    for i, (b, m) in enumerate(zip(base_seen[1:], masked_seen[1:])):
        assert m["pos"].tolist() == [[L + P + i]]
        if b["pos"] is not None:
            assert torch.equal(m["pos"], b["pos"])
        assert m["mask"][0, 30:L].eq(0).all() and m["mask"][0, :30].eq(1).all()


def test_masking_equals_deleting_the_keys_with_same_positions(tiny_model):
    """Masking positions == removing them from the cache while the new tokens keep their original positions."""
    ids = torch.tensor([[5, 9, 12, 7, 20]])
    kv, _ = apply_corruption(rand_kv(24), "pad", 25)
    L = kv[0][0].shape[2]
    from lakv.pipeline import LAKVPipeline as P
    n_real = 24
    cache_full = P._to_dynamic_cache(tuple((k.clone(), v.clone()) for k, v in kv))
    mask = torch.ones((1, L + 5), dtype=torch.long)
    mask[0, n_real:L] = 0
    pos = torch.arange(L, L + 5).unsqueeze(0)
    with torch.no_grad():
        a = tiny_model(input_ids=ids, past_key_values=cache_full, position_ids=pos, attention_mask=mask,
                       use_cache=True).logits[:, -1]
        cut = P._to_dynamic_cache(tuple((k[:, :, :n_real].clone(), v[:, :, :n_real].clone()) for k, v in kv))
        b = tiny_model(input_ids=ids, past_key_values=cut, position_ids=pos,
                       attention_mask=torch.ones((1, n_real + 5), dtype=torch.long), use_cache=True).logits[:, -1]
        unmasked = tiny_model(input_ids=ids, past_key_values=P._to_dynamic_cache(tuple((k.clone(), v.clone()) for k, v in kv)),
                              position_ids=pos, attention_mask=torch.ones((1, L + 5), dtype=torch.long),
                              use_cache=True).logits[:, -1]
    assert torch.allclose(a, b, atol=1e-5)
    assert not torch.allclose(a, unmasked, atol=1e-4)               # the mask actually changes the result


QUESTION = "Context:\n[Alpha] Alpha is a thing. Beta is another.\n\n[Gamma] Gamma is third.\n\nQuestion: What is Alpha?"


def test_full_run_flag_off_unchanged_and_on_masks_both_hops(tiny_model):
    off = make_pipe(tiny_model, causal_audit_mode="real_pad10", receiver_sees_passages=False)
    on = make_pipe(tiny_model, causal_audit_mode="real_pad10", receiver_sees_passages=False,
                   mask_modified_positions=True)
    r_off = off.run(QUESTION)
    r_on = on.run(QUESTION)
    assert all("mask_n_hidden" not in l for l in off.last_run_audit_logs)       # default: nothing recorded
    logs = on.last_run_audit_logs
    assert [l["agent_idx"] for l in logs] == [1, 2]
    n1 = logs[0]["length_after"] - logs[0]["length_before"]
    n2 = logs[1]["length_after"] - logs[1]["length_before"]
    assert logs[0]["mask_n_hidden"] == n1                       # hop 1: just its own padded tail
    assert logs[1]["mask_n_hidden"] == n1 + n2                  # hop 2: carried hop-1 zeros + new tail
    assert r_off.answer is not None and r_on.answer is not None


def test_flag_with_offset_correction_rejected(tiny_model):
    p = make_pipe(tiny_model, causal_audit_mode="real_pad10", mask_modified_positions=True)
    p.config.use_offset_correction = True
    with pytest.raises(ValueError):
        p.run(QUESTION)
