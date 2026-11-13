"""Tiny-model harness that records exactly what LAKVPipeline.run feeds each agent.

CPU only, UNTESTED ON GPU. Not a test module (no tests collected from it). Used by
tests/test_receiver_question.py and by the one-off generator that produced
tests/fixtures/run_prompts_golden.json from the UNMODIFIED pipeline (exp/consolidate-v9).

Stub tokenizer: char-level ids 4..63, EOS id 1. Tiny random Qwen2 (fp32). The installed
transformers (4.44) cannot run the pipeline's generate()-handoff fast path for hops >= 1
(IndexError, fails on the unmodified tip too), so `_uses_nondefault_rope_scaling` is forced
True to use the pipeline's own manual decode loop. Only prompts are compared against the
golden file (platform independent); generated text is only compared within one process.
"""
import copy

import torch

BLOB = ("Context:\n[Alpha Town] Alpha Town is a town in Beta County. It was founded in 1850.\n\n"
        "[Beta County] Beta County is a county. Its seat is Alpha Town.\n\n"
        "[Gamma Lake] Gamma Lake is a lake. It is unrelated.\n\n"
        "Question: In which county is Alpha Town?")
GSM = "Janet has 3 apples and buys 4 more. How many apples does she have?"


class RecordingTok:
    def __init__(self, eos_token_id=1):
        self.eos_token_id = eos_token_id
        self.chat_calls = []     # deep copies of the `messages` passed to apply_chat_template
        self.id_calls = []       # token-id lists produced for each prompt

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        self.chat_calls.append(copy.deepcopy(messages))
        return "||".join(f"{m['role']}:{m['content']}" for m in messages)

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        ids = [4 + (ord(c) % 60) for c in text]
        self.id_calls.append(ids)
        return {"input_ids": torch.tensor([ids])}

    def decode(self, ids, skip_special_tokens=True):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return " ".join(f"t{int(i)}" for i in ids if not (skip_special_tokens and int(i) == self.eos_token_id))


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


def build_pipe(model, dataset="hotpotqa", receiver_sees_passages=True, max_new=6):
    from lakv.pipeline import LAKVPipeline, PipelineConfig
    cfg = PipelineConfig(use_layer_selection=False, compression_mode="none", use_offset_correction=False,
                         reconstruction_strategy="zeros", dataset=dataset,
                         intermediate_max_new_tokens=max_new, final_max_new_tokens=max_new,
                         receiver_sees_passages=receiver_sees_passages)
    tok = RecordingTok()
    pipe = LAKVPipeline(model, tok, cfg, device="cpu")
    pipe._uses_nondefault_rope_scaling = lambda: True   # instance-level only; see module docstring
    return pipe, tok


def record(model, question, dataset="hotpotqa", receiver_sees_passages=True, run_kwargs=None):
    pipe, tok = build_pipe(model, dataset, receiver_sees_passages)
    res = pipe.run(question, **(run_kwargs or {}))
    return {"chat_calls": tok.chat_calls, "id_calls": tok.id_calls,
            "answer": res.answer, "hop_texts": list(res.hop_texts)}
