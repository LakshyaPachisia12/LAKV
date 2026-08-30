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

from dataclasses import dataclass
from typing import List, Optional

import torch

from lakv.pipeline import PipelineConfig


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
    intermediate_max_new_tokens: int = 200
    final_max_new_tokens: int = 1536
    do_sample: bool = True
    temperature: float = 0.6
    top_p: float = 0.95
    print_raw_outputs: bool = False

    def __post_init__(self):
        if self.system_prompts is None:
            self.system_prompts = _default_system_prompts(self.dataset)


@dataclass
class TextAgentResult:
    answer: str
    hop_texts: List[str]          # decoded text per non-final agent (Reasoner, Verifier)
    hop_bytes: List[int]          # UTF-8 byte length of each inter-agent handoff
    total_bytes: int


class TextAgentPipeline:
    def __init__(self, model, tokenizer, config: TextAgentPipelineConfig = None, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or TextAgentPipelineConfig()
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

    def run(self, question: str) -> TextAgentResult:
        prompts = self.config.system_prompts
        n_agents = self.config.n_agents
        hop_texts: List[str] = []
        hop_bytes: List[int] = []
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
            text = self._generate(system_prompt, user_content, max_new)

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
        )
