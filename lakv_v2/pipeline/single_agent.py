"""
LAKV-v2: Single Agent Baseline

A plain pipeline that doesn't use KV caching relay, to be used as an absolute baseline.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch

from lakv.pipeline import _REASONER_FEWSHOT_EXAMPLES

_SYSTEM_PROMPTS = {
    "gsm8k": (
        "You are a mathematical reasoning assistant. Before using any number "
        "from the question in a calculation, first restate it exactly as "
        "given in the question, so you don't accidentally substitute a "
        "different value. Solve carefully and return final answer as "
        "#### [number]"
    ),
    "hotpotqa": (
        "You are a careful reading-comprehension assistant. Read the given "
        "context passages and answer the question. Before answering, first "
        "quote the exact supporting sentence(s) from the context word-for-word, "
        "so you don't rely on a misremembered or paraphrased detail. Then, on "
        "its own final line, output exactly this format: The answer is: "
        "<answer>, where <answer> is a short word or phrase copied from the "
        "context, not a full sentence."
    ),
}


@dataclass
class SingleAgentPipelineConfig:
    dataset: str = "gsm8k"
    system_prompt: Optional[str] = None
    print_raw_outputs: bool = False
    max_new_tokens: int = 1536
    # Greedy by default — real eval harnesses only sample when paired with
    # self-consistency (multi-sample majority voting), never bare for a single
    # generation. temperature/top_p kept at Qwen's own recommended values (its
    # generation_config.json) for whenever sampling is turned back on.
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.8
    # Off by default — see PipelineConfig.use_reasoner_few_shot (lakv/pipeline.py)
    # for why: real but statistically-unproven accuracy effect at the sample
    # sizes tested, real prompt-length cost. GSM8K-specific — HotpotQA is
    # reading comprehension, not arithmetic, so this is never applied there
    # regardless. Mechanism kept available — pass use_few_shot=True to opt in.
    use_few_shot: bool = False
    few_shot_examples: List[Tuple[str, str]] = field(
        default_factory=lambda: list(_REASONER_FEWSHOT_EXAMPLES)
    )

    def __post_init__(self):
        if self.system_prompt is None:
            self.system_prompt = _SYSTEM_PROMPTS[self.dataset]

class SingleAgentPipeline:
    def __init__(self, model, tokenizer, config: SingleAgentPipelineConfig = None, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or SingleAgentPipelineConfig()
        self.device = device

    def run(self, question: str) -> str:
        messages = [{"role": "system", "content": self.config.system_prompt}]
        if self.config.use_few_shot and self.config.dataset == "gsm8k":
            for exemplar_q, exemplar_a in self.config.few_shot_examples:
                messages.append({"role": "user", "content": exemplar_q})
                messages.append({"role": "assistant", "content": exemplar_a})
        messages.append({"role": "user", "content": question})
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
            max_new_tokens=self.config.max_new_tokens,
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
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        if getattr(self.config, "print_raw_outputs", False):
            print(f"\n[RAW OUTPUT Single Agent] {answer}\n")
            
        return answer
