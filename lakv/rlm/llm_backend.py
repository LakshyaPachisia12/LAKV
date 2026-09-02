"""The seam between the RLM engine and model inference (spec section 17).

`LLMBackend` is the interface the engine calls through — it does not know or
care whether a call is served by a plain HF `generate()` (this phase) or by
LAKV's KV-relay pipeline (a later phase). Swapping backends should require
zero changes to engine.py.

Phase 1 scope: only `HFBackend` (a single model, plain `model.generate()`,
mirrors lakv/rlm_scaffold.py's _generate and run.py's load_model conventions)
is implemented. `InferenceRequest`/`InferenceResult` here are the plain
version of what spec section 17 asks for; a `KVCacheHandle` and an
`LAKVBackend` that reuses lakv/pipeline.py's KV injection are the documented
next step in docs/research/RLM_LAKV_INTERFACE.md, not built yet — LAKV's KV
relay is presently hand-rolled around a fixed hop chain (see pipeline.py),
and adapting it to serve arbitrary RLM sub-calls is a real design task of
its own that deserves to happen once there's trace evidence showing it's
worth it (spec section 36's ordering).
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import torch


@dataclass
class InferenceRequest:
    system_prompt: str
    user_content: str
    max_new_tokens: int
    do_sample: bool = False
    temperature: float = 0.6
    top_p: float = 0.95


@dataclass
class InferenceResult:
    text: str
    input_token_count: int
    output_token_count: int
    latency_s: float


class LLMBackend(Protocol):
    def generate(self, request: InferenceRequest) -> InferenceResult:
        ...

    @property
    def model_name(self) -> str:
        ...

    @property
    def dtype(self) -> str:
        ...

    @property
    def device(self) -> str:
        ...


class HFBackend:
    """Wraps a loaded HF model/tokenizer. Construct via run.py's load_model()
    (or any AutoModelForCausalLM/AutoTokenizer pair) and pass in here."""

    def __init__(self, model, tokenizer, model_name: str = "", device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self._model_name = model_name
        self._device = device

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dtype(self) -> str:
        return str(next(self.model.parameters()).dtype)

    @property
    def device(self) -> str:
        return self._device

    def generate(self, request: InferenceRequest) -> InferenceResult:
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_content},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(self.device)
        attention_mask = torch.ones_like(input_ids)

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=request.max_new_tokens,
            do_sample=request.do_sample,
            num_beams=1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if request.do_sample:
            gen_kwargs["temperature"] = request.temperature
            gen_kwargs["top_p"] = request.top_p

        t0 = perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(**gen_kwargs)
        latency_s = perf_counter() - t0

        new_tokens = output_ids[0, input_ids.shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return InferenceResult(
            text=text,
            input_token_count=int(input_ids.shape[1]),
            output_token_count=int(new_tokens.shape[0]),
            latency_s=latency_s,
        )
