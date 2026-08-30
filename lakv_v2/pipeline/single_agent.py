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
    print_raw_outputs: bool = False
    max_new_tokens: int = 1536
    do_sample: bool = True
    temperature: float = 0.6
    top_p: float = 0.95

class SingleAgentPipeline:
    def __init__(self, model, tokenizer, config: SingleAgentPipelineConfig = None, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or SingleAgentPipelineConfig()
        self.device = device

    def run(self, question: str) -> str:
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": question},
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
