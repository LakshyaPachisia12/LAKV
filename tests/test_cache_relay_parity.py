"""
Correctness gate: native cache continuation vs. LAKV extract -> reconstruct ->
continue, on a token-identical prefix + continuation.

This is the Third-Phase gate required before any exact-reuse / cross-context /
compression research work proceeds: if the pipeline's own KV extraction and
DynamicCache reconstruction round-trip (LAKVPipeline._to_tuple /
_to_dynamic_cache — used on every hop of every config, including the
uncompressed, no-layer-drop, no-anchor-correction Config A baseline) is not
numerically sound, nothing built on top of it can be trusted either.

Isolates exactly one variable: the SAME prefix forward pass's past_key_values
is used two ways —
  (a) native: passed directly into the continuation forward (no extraction).
  (b) LAKV:   extracted to a plain tuple via _to_tuple(), then rebuilt into a
      fresh DynamicCache via _to_dynamic_cache(), then passed into an
      otherwise-identical continuation forward.
Everything else (prefix content, continuation tokens, position_ids,
attention_mask) is held fixed and mirrors exactly what LAKVPipeline._forward /
_generate use for position_offset == 0 (see pipeline.py). No sampling is
involved anywhere, so there is no seeding concern — this compares the exact
distributions both paths compute, not generated text.

Needs a real GPU + the actual Qwen2.5-7B-Instruct weights, so it's opted out
of by default under plain `pytest` (would hang/OOM/fail confusingly on a
CPU-only box). Run explicitly:

    pytest tests/test_cache_relay_parity.py -v -s

Requires network access (or a local HF cache) to fetch the model on first run.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lakv.pipeline import LAKVPipeline

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# Token-identical prefix/continuation pair. Prefix mirrors a real agent's
# system+question turn (chat-template formatted); continuation mirrors a
# plausible reasoning span a receiving agent would continue from — content
# doesn't matter for this test (no sampling, no scoring), only that both
# paths see the exact same token ids.
PREFIX_TEXT = (
    "<|im_start|>system\n"
    "You are a careful math reasoning assistant. Show your work step by step.\n"
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes "
    "muffins with 4 more. She sells the rest at $2 each. How much does she "
    "make daily?\n"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)
CONTINUATION_TEXT = (
    "Janet has 16 eggs per day. She uses 3 + 4 = 7 eggs, leaving "
    "16 - 7 = 9 eggs to sell. At $2 each, she makes 9 * 2 = $18 per day."
)

# Pure bf16 round-trip, no compression/selection/anchor-correction — the only
# possible source of numerical difference is the extract/rebuild itself, so
# thresholds are tight. Set well below "would change generation," not at it.
MAX_ABS_LOGIT_DIFF_THRESHOLD = 1e-2
MEAN_ABS_LOGIT_DIFF_THRESHOLD = 1e-3
REL_L2_THRESHOLD = 1e-3
COSINE_SIM_THRESHOLD = 0.9999
KL_DIV_THRESHOLD = 1e-3


def _load_model_and_tokenizer():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def _metrics(native: torch.Tensor, lakv: torch.Tensor) -> dict:
    """All reported on float32-upcast, flattened tensors."""
    a = native.float().reshape(-1)
    b = lakv.float().reshape(-1)
    diff = (a - b).abs()
    return {
        "max_abs_error": diff.max().item(),
        "mean_abs_error": diff.mean().item(),
        "rel_l2_error": (a - b).norm().item() / max(a.norm().item(), 1e-12),
        "cosine_similarity": torch.nn.functional.cosine_similarity(
            a.unsqueeze(0), b.unsqueeze(0)
        ).item(),
    }


def _kl_divergence(native_logits: torch.Tensor, lakv_logits: torch.Tensor) -> float:
    """KL(softmax(native) || softmax(lakv)) on the final-token next-token
    distribution — the quantity that actually determines whether generation
    diverges between the two paths."""
    p = torch.softmax(native_logits.float(), dim=-1)
    log_p = torch.log_softmax(native_logits.float(), dim=-1)
    log_q = torch.log_softmax(lakv_logits.float(), dim=-1)
    return (p * (log_p - log_q)).sum().item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a real GPU + Qwen2.5-7B-Instruct")
def test_native_vs_lakv_cache_relay_parity():
    model, tokenizer = _load_model_and_tokenizer()
    device = next(model.parameters()).device

    prefix_ids = tokenizer(PREFIX_TEXT, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    continuation_ids = tokenizer(CONTINUATION_TEXT, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    prefix_len = prefix_ids.shape[1]
    cont_len = continuation_ids.shape[1]

    with torch.no_grad():
        prefix_out = model(input_ids=prefix_ids, use_cache=True, output_hidden_states=True)
    native_prefix_cache = prefix_out.past_key_values

    # LAKV path: extract -> rebuild, exactly what every hop of every config does
    # (LAKVPipeline._to_tuple / _to_dynamic_cache), before any compression,
    # layer selection, or anchor correction touches it.
    extracted = LAKVPipeline._to_tuple(native_prefix_cache)
    rebuilt_cache = LAKVPipeline._to_dynamic_cache(extracted)

    # Continuation forward setup mirrors LAKVPipeline._forward/_generate for
    # position_offset == 0 exactly: position_ids continue right after the
    # cache length, attention_mask covers cache + new tokens, all ones.
    position_ids = torch.arange(
        prefix_len, prefix_len + cont_len, device=device
    ).unsqueeze(0)
    attention_mask = torch.ones((1, prefix_len + cont_len), dtype=torch.long, device=device)

    with torch.no_grad():
        native_out = model(
            input_ids=continuation_ids,
            past_key_values=native_prefix_cache,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
        )
        lakv_out = model(
            input_ids=continuation_ids,
            past_key_values=rebuilt_cache,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
        )

    native_logits = native_out.logits[0]      # (cont_len, vocab)
    lakv_logits = lakv_out.logits[0]
    native_hidden = native_out.hidden_states[-1][0]  # (cont_len, hidden)
    lakv_hidden = lakv_out.hidden_states[-1][0]

    logit_metrics = _metrics(native_logits, lakv_logits)
    hidden_metrics = _metrics(native_hidden, lakv_hidden)
    final_kl = _kl_divergence(native_logits[-1], lakv_logits[-1])
    final_cosine = torch.nn.functional.cosine_similarity(
        native_logits[-1].float().unsqueeze(0), lakv_logits[-1].float().unsqueeze(0)
    ).item()
    native_next_tok = int(native_logits[-1].argmax().item())
    lakv_next_tok = int(lakv_logits[-1].argmax().item())

    print("\n=== Native vs LAKV cache-relay parity ===")
    print(f"prefix_len={prefix_len} continuation_len={cont_len}")
    print(f"[logits, all {cont_len} continuation positions] {logit_metrics}")
    print(f"[hidden states, last layer, all positions]      {hidden_metrics}")
    print(f"[final-token next-token distribution] KL={final_kl:.6g} cosine={final_cosine:.8f}")
    print(f"[final-token argmax] native={native_next_tok} lakv={lakv_next_tok} "
          f"match={native_next_tok == lakv_next_tok}")

    assert not math.isnan(logit_metrics["max_abs_error"]), "NaN in logit comparison"
    assert logit_metrics["max_abs_error"] < MAX_ABS_LOGIT_DIFF_THRESHOLD, (
        f"max abs logit error {logit_metrics['max_abs_error']} exceeds "
        f"{MAX_ABS_LOGIT_DIFF_THRESHOLD} — the extract/rebuild round-trip is "
        f"not numerically transparent, investigate before trusting any config "
        f"built on top of it"
    )
    assert logit_metrics["mean_abs_error"] < MEAN_ABS_LOGIT_DIFF_THRESHOLD
    assert logit_metrics["rel_l2_error"] < REL_L2_THRESHOLD
    assert logit_metrics["cosine_similarity"] > COSINE_SIM_THRESHOLD
    assert final_kl < KL_DIV_THRESHOLD
    assert native_next_tok == lakv_next_tok, (
        "native and LAKV-relayed paths disagree on the greedy next token — "
        "a real, generation-visible divergence from the cache round-trip alone"
    )


if __name__ == "__main__":
    test_native_vs_lakv_cache_relay_parity()
