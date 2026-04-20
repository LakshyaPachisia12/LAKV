"""
LAKV-v2 Module: Two-Agent Pipeline
Connects a Solver Agent (generates latent KV reasoning)
to a Finalizer Agent (consumes KV to produce final numeric output).
"""

import torch
from dataclasses import dataclass, field
from typing import Optional, Tuple
from lakv_v2.cache.aligner import CacheAligner
import time

@dataclass
class TwoAgentPipelineConfig:
    solver_prompt: str = "You are a mathematical reasoning agent. Solve carefully step by step."
    finalizer_prompt: str = "You have access to prior reasoning memory. Use it to solve the problem. Return only: #### [number]"
    # Cache modification components injected during execution
    selector = None
    compressor = None
    anchor_table = None   # Optional[AnchorTable] — shared cross-agent pool

class TwoAgentPipeline:
    def __init__(self, model, tokenizer, config: TwoAgentPipelineConfig, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.aligner = CacheAligner(device)

    def run(self, question: str) -> dict:
        t0 = time.time()
        anchor_hit = False
        anchor_confidence = 0.0

        # ---------------------------------------------
        # 0. BASE KV for anchor table (no-prefix)
        # ---------------------------------------------
        base_kv = None
        base_hidden = None
        if self.config.anchor_table is not None:
            from anchor_table import compute_base_kv, question_key as make_key
            base_kv, base_hidden = compute_base_kv(
                self.model, self.tokenizer, question, self.device)

        # ---------------------------------------------
        # 1. SOLVER AGENT (Generates KV Memory)
        # ---------------------------------------------
        prompt_1 = (
            f"<|im_start|>system\n{self.config.solver_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        input_ids_1 = self.tokenizer(prompt_1, return_tensors="pt")["input_ids"].to(self.device)

        with torch.no_grad():
            out_1 = self.model.generate(
                input_ids=input_ids_1,
                max_new_tokens=256,
                return_dict_in_generate=True,
                output_hidden_states=True,
                use_cache=True,
            )

        raw_kv_tuple = self.aligner._to_tuple(out_1.past_key_values)
        # hidden_states: tuple of (n_layers+1) each (1, seq, hidden) — take last token's last layer
        solver_hidden = out_1.hidden_states[-1][-1]  # (1, 1, hidden) → use base_hidden instead

        # Populate anchor table with solver's observation
        if self.config.anchor_table is not None and base_kv is not None:
            from anchor_table import question_key as make_key
            q_key = make_key(question)
            self.config.anchor_table.update(
                q_key, "agent_0", base_kv, raw_kv_tuple, base_hidden)

        # ---------------------------------------------
        # 2. KV MANIPULATION (Selection & Compression)
        # ---------------------------------------------
        processed_kv = raw_kv_tuple
        compression_stats = {}

        if self.config.selector:
            processed_kv, mask = self.config.selector.select(processed_kv)
            compression_stats["layers_sent"] = len(mask.kept_layer_indices)

        if self.config.compressor:
            message = self.config.compressor.compress(processed_kv)
            compression_stats["original_mb"] = message.original_mb()
            compression_stats["compressed_mb"] = message.size_mb()
            compression_stats["ratio"] = message.compression_ratio
            processed_kv = self.config.compressor.decompress(message, device=self.device)

        if self.config.selector:
            processed_kv = self.config.selector.reconstruct(processed_kv, mask)

        # Anchor table correction: try to improve processed_kv for the finalizer
        if self.config.anchor_table is not None and base_hidden is not None:
            from anchor_table import question_key as make_key
            q_key = make_key(question)
            result = self.config.anchor_table.query_correction(
                q_key, "agent_1", base_hidden, device=self.device)
            if result is not None:
                corrected_kv, conf = result
                processed_kv = corrected_kv
                anchor_hit = True
                anchor_confidence = conf
                compression_stats["anchor_hit"] = True
                compression_stats["anchor_confidence"] = conf

        # ---------------------------------------------
        # 3. FINALIZER AGENT (Consumes KV Memory)
        # ---------------------------------------------
        prompt_2 = (
            f"<|im_start|>system\n{self.config.finalizer_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        input_ids_2 = self.tokenizer(prompt_2, return_tensors="pt")["input_ids"].to(self.device)
        
        dynamic_cache = self.aligner.prepare_handoff_cache(processed_kv)
        # Extending attention mask and generating explicitly mapped position IDs
        attn_mask = self.aligner.build_attention_mask(processed_kv, input_ids_2)
        
        # Try native huggingface model.generate
        # Qwen2.5 automatically handles position_ids securely if attention_mask accurately maps past vs current lengths.
        with torch.no_grad():
            out_2 = self.model.generate(
                input_ids=input_ids_2,
                max_new_tokens=64,
                past_key_values=dynamic_cache,
                attention_mask=attn_mask,
                do_sample=False
            )
            
        new_tokens = out_2[0, input_ids_2.shape[1]:]
        final_answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        latency = time.time() - t0

        return {
            "answer": final_answer,
            "latency": latency,
            "compression_stats": compression_stats,
            "anchor_hit": anchor_hit,
            "anchor_confidence": anchor_confidence,
        }
