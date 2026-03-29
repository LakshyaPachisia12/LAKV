"""
LAKV Module 5: LAKVPipeline

Thin orchestrator — connects profile → selector → compressor → corrector
into a 3-agent forward-pass pipeline. Zero compression / selection logic here.

Transformers 4.36+ stores past_key_values as DynamicCache, not raw tuples.
Conversion helpers:
  _to_tuple  : DynamicCache → plain tuple of (K, V) pairs  (after forward)
  _to_dynamic_cache : plain tuple → DynamicCache           (before injection)
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


# ─── config / result dataclasses ──────────────────────────────────────────────

@dataclass
class PipelineConfig:
    use_layer_selection: bool       # True for Config C/D/E
    compression_mode: str           # 'none'|'uniform_int8'|'uniform_int4'|'adaptive'
    use_offset_correction: bool     # True for Config E only
    reconstruction_strategy: str    # 'zeros'|'nearest'|'interpolate'
    n_agents: int = 3
    profile_path: Optional[str] = None
    system_prompts: List[str] = field(default_factory=lambda: [
        "You are a mathematical reasoning agent. Analyze the problem carefully.",
        "You are a verification agent. Review the reasoning and identify any errors.",
        "You are a solution agent. Provide only the final numerical answer.",
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


# ─── pipeline ─────────────────────────────────────────────────────────────────

class LAKVPipeline:
    """Multi-agent KV-cache relay pipeline for Qwen2.5-7B."""

    N_LAYERS = 28

    def __init__(self, model, tokenizer, config: PipelineConfig, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

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

    # ── public API ────────────────────────────────────────────────────────

    def run(self, question: str) -> RunResult:
        """Execute the 3-agent pipeline and return the final answer."""
        hop_stats: List[HopStat] = []
        kv_message: Optional[KVMessage] = None
        selection_mask: Optional[SelectionMask] = None
        total_bytes = 0
        answer = ""

        n_agents = self.config.n_agents
        prompts = self.config.system_prompts

        for agent_idx in range(n_agents):
            system_prompt = prompts[agent_idx] if agent_idx < len(prompts) else prompts[-1]
            prompt_text = self._build_prompt(question, system_prompt)
            input_ids = self.tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(self.device)

            is_last = (agent_idx == n_agents - 1)

            # ── decompress + inject KV from previous agent ───────────
            injected_kv_tuple: Optional[tuple] = None
            if kv_message is not None:
                decompressed = self.compressor.decompress(kv_message)   # raw tuple

                if self.selector and selection_mask is not None:
                    decompressed = self.selector.reconstruct(
                        decompressed,
                        selection_mask,
                        strategy=self.config.reconstruction_strategy,
                    )

                injected_kv_tuple = decompressed  # stays as tuple; _forward/_generate convert

            if is_last:
                # ── final agent: generate text ───────────────────────
                answer = self._generate(input_ids, injected_kv_tuple)

            else:
                # ── intermediate agent: forward pass ONLY ─────────────
                raw_kv_tuple = self._forward(input_ids, injected_kv_tuple)

                # layer selection
                tier_info: Optional[Dict[int, int]] = None
                if self.selector:
                    filtered_kv, selection_mask = self.selector.select(raw_kv_tuple)
                    tier_info = selection_mask.tier_per_kept_layer
                    n_transmitted = len(selection_mask.kept_layer_indices)
                else:
                    filtered_kv = raw_kv_tuple
                    selection_mask = None
                    n_transmitted = self.N_LAYERS

                # compression
                kv_message = self.compressor.compress(filtered_kv, tier_info=tier_info)

                # offset correction (stub)
                if self.corrector:
                    kv_message = self.corrector.correct(
                        kv_message,
                        sender_suffix_len=input_ids.shape[1],
                        receiver_suffix_len=input_ids.shape[1],
                    )

                total_bytes += kv_message.compressed_bytes
                if self.verbose:
                    print(f"    [KV] Transmitting {len(kv_message.layers)} layers "
                          f"| Original: {kv_message.original_bytes / 1e6:.2f} MB "
                          f"| Compressed: {kv_message.compressed_bytes / 1e6:.2f} MB "
                          f"| Ratio: {kv_message.compression_ratio:.2f}x")
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

    # ── cache conversion helpers ──────────────────────────────────────────

    @staticmethod
    def _to_tuple(pkv) -> tuple:
        """Convert DynamicCache → plain tuple of (K, V) pairs.

        Uses the stable .to_legacy_cache() API (works in all transformers
        versions that have DynamicCache, from 4.36 through 4.57+).
        Falls through for objects that are already plain tuples.
        """
        if isinstance(pkv, DynamicCache):
            return pkv.to_legacy_cache()   # returns ((k0,v0),(k1,v1),...)
        return pkv  # already a tuple

    @staticmethod
    def _to_dynamic_cache(kv_tuple: tuple) -> DynamicCache:
        """Wrap plain (K, V) tuple → DynamicCache for Transformers injection."""
        return DynamicCache.from_legacy_cache(kv_tuple)

    # ── forward / generate ────────────────────────────────────────────────

    def _forward(self, input_ids: torch.Tensor, injected_kv_tuple: Optional[tuple] = None) -> tuple:
        """model.forward() only — NO text generation. Returns raw (K, V) tuple."""
        kwargs: dict = {"input_ids": input_ids, "use_cache": True}

        if injected_kv_tuple is not None:
            cache = self._to_dynamic_cache(injected_kv_tuple)
            cache_seq_len = injected_kv_tuple[0][0].shape[2]
            kwargs["past_key_values"] = cache
            kwargs["position_ids"] = torch.arange(
                cache_seq_len, cache_seq_len + input_ids.shape[1],
                device=self.device,
            ).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(**kwargs)

        return self._to_tuple(outputs.past_key_values)

    def _generate(self, input_ids: torch.Tensor, injected_kv_tuple: Optional[tuple] = None) -> str:
        """Run generation for the final agent.

        When KV is injected from a prior agent, model.generate() cannot be used
        directly — generate() assumes past_key_values covers a prefix of the
        *same* input_ids, but our KV comes from a *different* agent's prompt.
        This causes internal cache_position trimming to produce an empty tensor,
        crashing in _cache_dependant_input_preparation.

        Solution: prime the cache with one explicit forward() call, then run a
        manual greedy decode loop (one token at a time) using the primed cache.
        When there is no KV to inject, fall back to model.generate() normally.
        """
        eos_id = self.tokenizer.eos_token_id

        if injected_kv_tuple is None:
            # ── No KV injection: standard generate ───────────────────
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=512,
                    do_sample=False,
                )
            new_tokens = output_ids[0, input_ids.shape[1]:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # ── KV injection: forward-prime then manual greedy decode ─────
        cache = self._to_dynamic_cache(injected_kv_tuple)
        cache_seq_len = injected_kv_tuple[0][0].shape[2]
        position_ids = torch.arange(
            cache_seq_len, cache_seq_len + input_ids.shape[1],
            device=self.device,
        ).unsqueeze(0)

        # Step 1 — Prime: forward pass with injected KV + full prompt
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                past_key_values=cache,
                position_ids=position_ids,
                use_cache=True,
            )
        running_cache = out.past_key_values          # natural DynamicCache
        next_logits   = out.logits[:, -1, :]         # logits for the next token

        # Step 2 — Greedy decode one token at a time
        generated: List[int] = []
        for _ in range(512):
            next_token = next_logits.argmax(-1, keepdim=True)  # (1, 1)
            tok_id = next_token.item()
            if tok_id == eos_id:
                break
            generated.append(tok_id)
            with torch.no_grad():
                out = self.model(
                    input_ids=next_token,
                    past_key_values=running_cache,
                    use_cache=True,
                )
            running_cache = out.past_key_values
            next_logits   = out.logits[:, -1, :]

        return self.tokenizer.decode(generated, skip_special_tokens=True)



    def _build_prompt(self, question: str, system_prompt: str) -> str:
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
