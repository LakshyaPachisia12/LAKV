"""
LAKV Module 5: LAKVPipeline

Multi-agent KV-cache relay pipeline.

Each agent in the chain:
  1. Receives the compressed KV from the previous agent (if any).
  2. Encodes a SHORT continuation turn on top of that KV.
  3. Generates up to `intermediate_max_tokens` tokens (all agents generate).
  4. The KV now includes the generated reasoning/verification tokens.
  5. Compresses + selects layers, then transmits to the next agent.

Agent 1 sends the full [system][user: question][assistant: <reasoning>] KV.
Agent 2+ encode only a short new user turn ("Verify the above…") appended to
the injected KV, then generate their contribution.

This way each agent genuinely READS the previous agent's output via the KV,
and the model never sees a semantically incoherent "stacked prompt" context.

Transformers 4.36+ stores past_key_values as DynamicCache, not raw tuples.
Conversion helpers:
  _to_tuple         : DynamicCache -> plain tuple of (K, V) pairs  (after generate)
  _to_dynamic_cache : plain tuple  -> DynamicCache                 (before injection)
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from transformers import DynamicCache

from calibration_profiler import LayerProfile
from layer_selector import LayerSelector, SelectionMask
from kv_compressor import KVCompressor, KVMessage
from offset_corrector import OffsetCorrector


# --- config / result dataclasses -------------------------------------------

@dataclass
class PipelineConfig:
    use_layer_selection: bool       # True for Config C/D
    compression_mode: str           # 'none'|'uniform_int8'|'uniform_int4'|'adaptive'
    use_offset_correction: bool
    reconstruction_strategy: str    # 'zeros'|'nearest'|'interpolate'
    n_agents: int = 3
    profile_path: Optional[str] = None
    # Tokens each intermediate agent may generate before passing KV forward.
    intermediate_max_tokens: int = 200
    # Tokens the final agent may generate.
    final_max_tokens: int = 512
    # System prompt for agent 0 (the first agent receives the full question).
    first_system_prompt: str = (
        "You are Agent 1 in a math reasoning relay. "
        "Solve the problem step by step using concise arithmetic reasoning. "
        "Do not output the final answer line. "
        "End your response with: Reasoning complete."
    )
    # Short user turn injected to continue the conversation for agents 1+.
    continuation_prompts: List[str] = field(default_factory=lambda: [
        "You are Agent 2. Review the reasoning above for mistakes. "
        "If you find an error, correct it and continue from the corrected step. "
        "Keep the response concise and do not output the final answer line. "
        "End your response with: Verification complete.",
        "You are Agent 3 (final). Using all prior context, output exactly one line "
        "in this format: The answer is [number]. "
        "Do not add any explanation, units, or extra text.",
    ])


@dataclass
class HopStat:
    agent_idx: int
    original_mb: float
    compressed_mb: float
    compression_ratio: float
    n_layers_transmitted: int
    n_layers_total: int             # 28


@dataclass
class RunResult:
    answer: str
    hop_stats: List[HopStat]
    total_bytes_transmitted: int
    total_compressed_mb: float
    total_original_mb: float
    overall_compression_ratio: float


# --- pipeline ---------------------------------------------------------------

class LAKVPipeline:
    """Multi-agent KV-cache relay pipeline for Qwen2.5-7B."""

    N_LAYERS = 28

    def __init__(self, model, tokenizer, config: PipelineConfig, device: str = "cuda"):
        self.model     = model
        self.tokenizer = tokenizer
        self.config    = config
        self.device    = device

        # load profile if provided
        self.profile: Optional[LayerProfile] = None
        if config.profile_path:
            self.profile = LayerProfile.load(config.profile_path)

        # sub-modules
        self.selector: Optional[LayerSelector] = None
        if config.use_layer_selection and self.profile:
            self.selector = LayerSelector(self.profile)

        self.verbose = getattr(config, 'verbose', False)

        self.compressor = KVCompressor(
            mode=config.compression_mode,
            profile=self.profile if config.compression_mode == "adaptive" else None,
        )

        self.corrector: Optional[OffsetCorrector] = None
        if config.use_offset_correction:
            self.corrector = OffsetCorrector()

    # -- public API ----------------------------------------------------------

    def run(self, question: str) -> RunResult:
        """Execute the n-agent pipeline and return the final answer."""
        hop_stats: List[HopStat] = []
        kv_message: Optional[KVMessage] = None
        selection_mask: Optional[SelectionMask] = None
        total_bytes = 0
        answer = ""

        n_agents = self.config.n_agents

        # Track full conversation history
        messages = [
            {"role": "system", "content": self.config.first_system_prompt},
            {"role": "user",   "content": question},
        ]

        for agent_idx in range(n_agents):
            is_last = (agent_idx == n_agents - 1)
            max_new = (
                self.config.final_max_tokens if is_last
                else self.config.intermediate_max_tokens
            )

            # -- decompress + inject KV from previous agent -----------------
            injected_kv_tuple: Optional[tuple] = None
            if kv_message is not None:
                decompressed = self.compressor.decompress(kv_message, device=self.device)

                if self.selector and selection_mask is not None:
                    decompressed = self.selector.reconstruct(
                        decompressed,
                        selection_mask,
                        strategy=self.config.reconstruction_strategy,
                    )

                injected_kv_tuple = tuple(
                    (k.to(self.device), v.to(self.device)) for k, v in decompressed
                )

            # -- build input_ids for this agent -----------------------------
            # Agent 0 encodes the full conversation. Agent 1+ with injected KV
            # must encode only the newly appended continuation user turn.
            # Always apply template to full history
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            full_enc = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

            if injected_kv_tuple is None:
                input_ids = full_enc["input_ids"]
                attn_mask = full_enc["attention_mask"]
            else:
                # Encode the previous turn's full context to find the split point
                prev_messages = messages[:-1]  # everything except the last user turn
                prev_text = self.tokenizer.apply_chat_template(
                    prev_messages, tokenize=False, add_generation_prompt=False
                )
                prev_len = len(self.tokenizer(prev_text, return_tensors="pt")["input_ids"][0])

                # Feed only the new tokens — the KV already covers prev_len tokens
                input_ids = full_enc["input_ids"][:, prev_len:]
                attn_mask = full_enc["attention_mask"][:, prev_len:]

                if input_ids.shape[1] == 0:
                    raise RuntimeError("Delta encoding produced zero tokens.")

            # -- generate (all agents generate) -----------------------------
            raw_output_ids, full_kv_tuple = self._generate_with_kv(
                input_ids, attn_mask, injected_kv_tuple, max_new
            )

            new_tokens = raw_output_ids[0, input_ids.shape[1]:]
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            if is_last:
                answer = generated_text
            else:
                # Add agent's reasoning to history, plus next user turn
                messages.append({"role": "assistant", "content": generated_text})
                
                cont_idx = agent_idx
                cont_prompts = self.config.continuation_prompts
                user_msg = (
                    cont_prompts[cont_idx] if cont_idx < len(cont_prompts)
                    else cont_prompts[-1]
                )
                messages.append({"role": "user", "content": user_msg})
                # -- layer selection ----------------------------------------
                tier_info: Optional[Dict[int, int]] = None
                if self.selector:
                    filtered_kv, selection_mask = self.selector.select(full_kv_tuple)
                    tier_info = selection_mask.tier_per_kept_layer
                    n_transmitted = len(selection_mask.kept_layer_indices)
                else:
                    filtered_kv = full_kv_tuple
                    selection_mask = None
                    n_transmitted = self.N_LAYERS

                # -- compression --------------------------------------------
                kv_message = self.compressor.compress(filtered_kv, tier_info=tier_info)

                # -- offset correction (stub) --------------------------------
                if self.corrector:
                    kv_message = self.corrector.correct(
                        kv_message,
                        sender_suffix_len=input_ids.shape[1],
                        receiver_suffix_len=input_ids.shape[1],
                    )

                total_bytes += kv_message.compressed_bytes
                if self.verbose:
                    print(
                        f"    [KV] Agent {agent_idx} -> "
                        f"{len(kv_message.layers)} layers | "
                        f"Original: {kv_message.original_bytes / 1e6:.2f} MB | "
                        f"Compressed: {kv_message.compressed_bytes / 1e6:.2f} MB | "
                        f"Ratio: {kv_message.compression_ratio:.2f}x"
                    )
                hop_stats.append(HopStat(
                    agent_idx=agent_idx,
                    original_mb=kv_message.original_bytes / 1e6,
                    compressed_mb=kv_message.compressed_bytes / 1e6,
                    compression_ratio=kv_message.compression_ratio,
                    n_layers_transmitted=n_transmitted,
                    n_layers_total=self.N_LAYERS,
                ))

        total_orig = sum(h.original_mb for h in hop_stats)
        total_comp = sum(h.compressed_mb for h in hop_stats)

        return RunResult(
            answer=answer,
            hop_stats=hop_stats,
            total_bytes_transmitted=total_bytes,
            total_compressed_mb=total_comp,
            total_original_mb=total_orig,
            overall_compression_ratio=total_orig / max(total_comp, 1e-9),
        )

    # -- cache conversion helpers -------------------------------------------

    @staticmethod
    def _to_tuple(pkv) -> tuple:
        """Convert DynamicCache -> plain tuple of (K, V) pairs."""
        if isinstance(pkv, DynamicCache):
            return pkv.to_legacy_cache()
        return pkv

    @staticmethod
    def _to_dynamic_cache(kv_tuple: tuple) -> DynamicCache:
        """Wrap plain (K, V) tuple -> DynamicCache for Transformers injection."""
        return DynamicCache.from_legacy_cache(kv_tuple)

    # -- core generation method ---------------------------------------------

    def _generate_with_kv(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        injected_kv_tuple: Optional[tuple],
        max_new_tokens: int,
    ) -> Tuple[torch.Tensor, tuple]:
        """
        Generate tokens, optionally continuing from an injected KV cache.

        Returns:
          output_ids   : full token tensor (input + generated) shape [1, T+G]
          full_kv_tuple: the complete KV cache after generation (all layers)
        """
        generation_attention_mask = attention_mask

        kwargs: dict = {
            "input_ids":              input_ids,
            "max_new_tokens":         max_new_tokens,
            "do_sample":              False,
            "use_cache":              True,
            "repetition_penalty":     1.1,
            "return_dict_in_generate": True,
        }

        if injected_kv_tuple is not None:
            kv_seq_len = self._validate_injected_kv(injected_kv_tuple)
            kwargs["past_key_values"] = self._to_dynamic_cache(injected_kv_tuple)

            position_ids = torch.arange(
                kv_seq_len,
                kv_seq_len + input_ids.shape[1],
                device=input_ids.device,
                dtype=torch.long,
            )
            kwargs["position_ids"] = position_ids.unsqueeze(0)
            kwargs["cache_position"] = position_ids

            if generation_attention_mask is None:
                generation_attention_mask = torch.ones_like(input_ids)

            if generation_attention_mask.shape[1] == input_ids.shape[1]:
                prefix_mask = torch.ones(
                    (generation_attention_mask.shape[0], kv_seq_len),
                    dtype=generation_attention_mask.dtype,
                    device=generation_attention_mask.device,
                )
                generation_attention_mask = torch.cat(
                    [prefix_mask, generation_attention_mask], dim=1
                )
            else:
                expected_len = kv_seq_len + input_ids.shape[1]
                if generation_attention_mask.shape[1] != expected_len:
                    raise RuntimeError(
                        "Injected-KV attention_mask length mismatch: "
                        f"expected {expected_len}, got {generation_attention_mask.shape[1]}"
                    )

            if self.verbose:
                print(
                    f"    [KV] Injected cache seq={kv_seq_len}, "
                    f"new input seq={input_ids.shape[1]}, "
                    f"attention seq={generation_attention_mask.shape[1]}"
                )

        kwargs["attention_mask"] = generation_attention_mask

        with torch.no_grad():
            outputs = self.model.generate(**kwargs)

        output_ids   = outputs.sequences          # [1, T+G]
        full_kv_tuple = self._to_tuple(outputs.past_key_values)

        return output_ids, full_kv_tuple

    def _validate_injected_kv(self, injected_kv_tuple: tuple) -> int:
        if len(injected_kv_tuple) != self.N_LAYERS:
            raise RuntimeError(
                f"Injected KV must contain {self.N_LAYERS} layers, got {len(injected_kv_tuple)}."
            )

        seq_lens = set()
        for layer_idx, (k, v) in enumerate(injected_kv_tuple):
            if k.shape[2] != v.shape[2]:
                raise RuntimeError(
                    f"Injected KV layer {layer_idx} has mismatched K/V seq lens: "
                    f"{k.shape[2]} vs {v.shape[2]}."
                )
            seq_lens.add(int(k.shape[2]))

        if len(seq_lens) != 1:
            raise RuntimeError(
                f"Injected KV layers must share one seq_len; got {sorted(seq_lens)}."
            )

        return next(iter(seq_lens))
