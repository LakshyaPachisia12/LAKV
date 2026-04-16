"""
LAKV-v2: Single Agent Baseline

A plain pipeline that doesn't use KV caching relay, to be used as an absolute baseline.
"""

from dataclasses import dataclass
import torch

@dataclass
class SingleAgentPipelineConfig:
    system_prompt: str = (
        "You are a mathematical reasoning assistant. "
        "Solve carefully and return final answer as #### [number]"
    )

class SingleAgentPipeline:
    def __init__(self, model, tokenizer, config: SingleAgentPipelineConfig = None, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or SingleAgentPipelineConfig()
        self.device = device

    def run(self, question: str) -> str:
        prompt_text = (
            f"<|im_start|>system\n{self.config.system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        input_ids = self.tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.1,
            )
            
        new_tokens = output_ids[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
