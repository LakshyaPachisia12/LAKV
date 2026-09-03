"""
LAKV-v2: Text-Relay Baseline

Same 3-agent role structure as Config A (Reasoner -> Verifier -> Finalizer,
reusing PipelineConfig.system_prompts so the roles are byte-for-byte
identical to A), but agents communicate by literally embedding the previous
agent's decoded text into the next agent's user message instead of injecting
KV cache. No manual decode loop, no AnchorTable - every agent calls
model.generate() on its own fresh, complete prompt, same as SingleAgentPipeline.

This exists to answer the question every KV-relay accuracy number this
project has produced so far could not answer: does relaying KV actually beat
the much simpler, much cheaper (in bytes) alternative of just relaying text?
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

from lakv.pipeline import PipelineConfig, _REASONER_FEWSHOT_EXAMPLES


def _default_system_prompts(dataset: str = "gsm8k") -> List[str]:
    # Matches Config A's PipelineConfig exactly (see evaluator.py PRESETS["A"])
    # so the only variable between text_agent and A is the comms channel.
    return PipelineConfig(
        use_layer_selection=False, compression_mode="none",
        use_offset_correction=False, reconstruction_strategy="zeros",
        dataset=dataset,
    ).system_prompts


@dataclass
class TextAgentPipelineConfig:
    dataset: str = "gsm8k"
    system_prompts: Optional[List[str]] = None
    n_agents: int = 3
    # See PipelineConfig.intermediate_max_new_tokens (lakv/pipeline.py) — same
    # fix, same reasoning: 200 was truncating Reasoner/Verifier hops mid-
    # derivation on ~85-90% of questions, measured this session.
    intermediate_max_new_tokens: int = 512
    final_max_new_tokens: int = 1536
    # Greedy by default — see SingleAgentPipelineConfig / PipelineConfig for
    # rationale. temperature/top_p kept at Qwen's recommended values for when
    # sampling is turned back on (e.g. self-consistency).
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.8
    print_raw_outputs: bool = False
    # Off by default — same rationale as SingleAgentPipelineConfig.use_few_shot
    # and PipelineConfig.use_reasoner_few_shot. Applied only to the first
    # (Reasoner) hop, GSM8K only, when explicitly turned on.
    use_few_shot: bool = False
    few_shot_examples: List[Tuple[str, str]] = field(
        default_factory=lambda: list(_REASONER_FEWSHOT_EXAMPLES)
    )

    def __post_init__(self):
        if self.system_prompts is None:
            self.system_prompts = _default_system_prompts(self.dataset)


@dataclass
class TextAgentResult:
    answer: str
    hop_texts: List[str]          # decoded text per non-final agent (Reasoner, Verifier)
    hop_bytes: List[int]          # UTF-8 byte length of each inter-agent handoff
    total_bytes: int
    hop_latencies: List[float] = field(default_factory=list)  # wall-clock seconds per agent, index 0..n_agents-1 (Reasoner, Verifier, Finalizer)


class TextAgentPipeline:
    def __init__(self, model, tokenizer, config: TextAgentPipelineConfig = None, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or TextAgentPipelineConfig()
        self.device = device

    def _generate(self, system_prompt: str, user_content: str, max_new_tokens: int,
                   include_few_shot: bool = False) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        if include_few_shot:
            for exemplar_q, exemplar_a in self.config.few_shot_examples:
                messages.append({"role": "user", "content": exemplar_q})
                messages.append({"role": "assistant", "content": exemplar_a})
        messages.append({"role": "user", "content": user_content})
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

    def run(self, question: str) -> TextAgentResult:
        prompts = self.config.system_prompts
        n_agents = self.config.n_agents
        hop_texts: List[str] = []
        hop_bytes: List[int] = []
        hop_latencies: List[float] = []
        prev_text = None
        answer = ""

        role_names = ["Reasoner", "Verifier", "Finalizer"]

        for agent_idx in range(n_agents):
            system_prompt = prompts[agent_idx] if agent_idx < len(prompts) else prompts[-1]
            is_last = (agent_idx == n_agents - 1)

            if prev_text is None:
                user_content = question
            else:
                prev_role = role_names[agent_idx - 1] if agent_idx - 1 < len(role_names) else f"Agent {agent_idx - 1}"
                user_content = (
                    f"Question: {question}\n\n"
                    f"{prev_role}'s response:\n{prev_text}\n"
                )

            max_new = self.config.final_max_new_tokens if is_last else self.config.intermediate_max_new_tokens
            include_few_shot = (agent_idx == 0 and self.config.use_few_shot and self.config.dataset == "gsm8k")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _t0 = time.perf_counter()
            text = self._generate(system_prompt, user_content, max_new, include_few_shot=include_few_shot)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            hop_latencies.append(time.perf_counter() - _t0)

            if is_last:
                answer = text
            else:
                hop_texts.append(text)
                # Bytes charged to this hop = the handoff payload actually
                # transmitted to the NEXT agent, i.e. this agent's own
                # decoded text (what gets embedded in the next prompt).
                hop_bytes.append(len(text.encode("utf-8")))
                prev_text = text

            if getattr(self.config, "print_raw_outputs", False):
                print(f"\n[RAW OUTPUT Text Agent {agent_idx}] {text}\n")

        total_bytes = sum(hop_bytes)
        return TextAgentResult(
            answer=answer,
            hop_texts=hop_texts,
            hop_bytes=hop_bytes,
            total_bytes=total_bytes,
            hop_latencies=hop_latencies,
        )
