"""
LAKV RLM scaffold — Phase 5 / EXP-7 of the RLM addendum in docs/RESEARCH_PLAN.md.

A minimal, text-only reproduction of Recursive Language Models (Zhang, Kraska,
Khattab — MIT CSAIL, arXiv:2512.24601): the root call treats the context as a
set of chunks it can dispatch sub-calls over, rather than reading everything
in one prefill.

Deliberate scope cut vs. the full RLM paper: this is fixed depth-1 recursion
over a DETERMINISTIC chunking (one sub-call per HotpotQA distractor paragraph)
instead of a REPL loop where the root model writes/executes code to decide
how to slice its context. A real REPL loop needs a sandboxed code-execution
environment and open-ended nesting; that's a much bigger build than EXP-7
needs to answer its actual question (does recursive decomposition + later,
KV relay at the call boundary help at all). Fixed fan-out over HotpotQA's
already-chunked 10 paragraphs gets the same recursive-call-boundary structure
LAKV cares about without that infrastructure. See docs/RESEARCH_PLAN.md's
"Gap 7" for the follow-on generalization if this turns out to matter.

do_sample defaults to False (greedy): this session found that GSM8K digit
accuracy is highly sensitive to decoding config, so keep this scaffold
deterministic/reproducible until that investigation (elsewhere, in progress)
lands — don't compound an unresolved decoding-quality question with a new
untested pipeline.
"""

from dataclasses import dataclass, field
from time import perf_counter
from typing import List, Optional

import torch

from lakv.qa_scoring import extract_qa_answer


_SUB_SYSTEM_PROMPT = (
    "You are a careful reading-comprehension assistant. You will be given ONE "
    "passage and a question. Decide whether the passage helps answer the "
    "question. Respond in exactly this format:\n"
    "Relevant: yes/no\n"
    "Fact: <the specific fact or answer span from the passage, or 'none' if not relevant>"
)

_ROOT_SYSTEM_PROMPT = (
    "You are a careful reading-comprehension assistant. You will be given a "
    "question and a numbered list of facts gathered from different passages, "
    "some of which may be irrelevant. Combine the relevant facts and answer "
    "the question. Output exactly one line in this format: "
    "The answer is: <answer>, where <answer> is a short word or phrase, not a "
    "full sentence."
)


@dataclass
class RLMPipelineConfig:
    sub_max_new_tokens: int = 80
    root_max_new_tokens: int = 48
    do_sample: bool = False
    temperature: float = 0.6
    top_p: float = 0.95
    max_chunks: Optional[int] = None  # cap fan-out; None = use every paragraph given


@dataclass
class RLMCallRecord:
    role: str  # "sub" or "root"
    chunk_idx: Optional[int]
    latency_s: float
    raw_output: str


@dataclass
class RLMResult:
    question: str
    answer: str
    sub_answers: List[str]
    calls: List[RLMCallRecord]
    n_calls: int
    total_latency_s: float

    @property
    def sub_call_latency_s(self) -> float:
        return sum(c.latency_s for c in self.calls if c.role == "sub")

    @property
    def root_call_latency_s(self) -> float:
        return sum(c.latency_s for c in self.calls if c.role == "root")


class RLMPipeline:
    """Depth-1 recursive decomposition: one sub-call per context chunk, one
    root call to aggregate. See module docstring for why this isn't a full
    REPL-driven RLM yet.
    """

    def __init__(self, model, tokenizer, config: RLMPipelineConfig = None, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or RLMPipelineConfig()
        self.device = device

    def _generate(self, system_prompt: str, user_content: str, max_new_tokens: int) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
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
            max_new_tokens=max_new_tokens,
            do_sample=self.config.do_sample,
            num_beams=1,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if self.config.do_sample:
            gen_kwargs["temperature"] = self.config.temperature
            gen_kwargs["top_p"] = self.config.top_p

        with torch.no_grad():
            output_ids = self.model.generate(**gen_kwargs)

        new_tokens = output_ids[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _sub_call(self, chunk_idx: int, chunk_text: str, question: str) -> RLMCallRecord:
        user_content = f"Passage: {chunk_text}\n\nQuestion: {question}"
        t0 = perf_counter()
        raw = self._generate(_SUB_SYSTEM_PROMPT, user_content, self.config.sub_max_new_tokens)
        latency = perf_counter() - t0
        return RLMCallRecord(role="sub", chunk_idx=chunk_idx, latency_s=latency, raw_output=raw)

    def _root_call(self, question: str, sub_answers: List[str]) -> RLMCallRecord:
        facts = "\n".join(f"{i+1}. {a}" for i, a in enumerate(sub_answers))
        user_content = f"Question: {question}\n\nFacts gathered:\n{facts}"
        t0 = perf_counter()
        raw = self._generate(_ROOT_SYSTEM_PROMPT, user_content, self.config.root_max_new_tokens)
        latency = perf_counter() - t0
        return RLMCallRecord(role="root", chunk_idx=None, latency_s=latency, raw_output=raw)

    def run(self, paragraphs: List[str], question: str) -> RLMResult:
        chunks = paragraphs
        if self.config.max_chunks is not None:
            chunks = chunks[: self.config.max_chunks]

        t_start = perf_counter()
        calls: List[RLMCallRecord] = []
        sub_answers: List[str] = []

        for i, chunk in enumerate(chunks):
            record = self._sub_call(i, chunk, question)
            calls.append(record)
            sub_answers.append(record.raw_output.strip())

        root_record = self._root_call(question, sub_answers)
        calls.append(root_record)
        total_latency = perf_counter() - t_start

        answer = extract_qa_answer(root_record.raw_output) or root_record.raw_output.strip()

        return RLMResult(
            question=question,
            answer=answer,
            sub_answers=sub_answers,
            calls=calls,
            n_calls=len(calls),
            total_latency_s=total_latency,
        )
