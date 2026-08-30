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
from anchor_table import AnchorTable, question_key as make_key, compute_base_kv


# Synthetic (non-eval) few-shot exemplar for the Reasoner: demonstrates
# restating each given number before using it, to counter the model's
# tendency to misread numbers it copies later in its own reasoning.
_REASONER_EXEMPLAR_QUESTION = (
    "A bakery bakes 45 loaves of bread each morning. It sells 12 loaves to a "
    "cafe and donates 8 loaves to a shelter. It sells the rest at $3 per "
    "loaf. How much money does the bakery make from the remaining loaves?"
)
_REASONER_EXEMPLAR_SOLUTION = (
    "Step 1: The bakery bakes 45 loaves (given: 45 loaves baked).\n"
    "Step 2: It sells 12 loaves to a cafe (given: 12 loaves to cafe).\n"
    "Step 3: It donates 8 loaves to a shelter (given: 8 loaves donated).\n"
    "Step 4: Loaves remaining = 45 - 12 - 8 = 25.\n"
    "Step 5: Price per loaf is $3 (given: $3 per loaf).\n"
    "Step 6: Revenue = 25 * 3 = 75.\n"
    "#### 75"
)


# ─── config / result dataclasses ──────────────────────────────────────────────

@dataclass
class PipelineConfig:
    use_layer_selection: bool       # True for Config C/D/E
    compression_mode: str           # 'none'|'uniform_int8'|'uniform_int4'|'adaptive'
    use_offset_correction: bool     # True for Config E only
    reconstruction_strategy: str    # 'zeros'|'nearest'|'interpolate'
    n_agents: int = 3
    outlier_clipping: bool = False
    clip_percentile: float = 99.5
    profile_path: Optional[str] = None
    intermediate_max_new_tokens: int = 200
    # Kept at 512 (not bumped to single_agent's 1536) deliberately: every
    # existing preset's final hop always goes through the manual KV-injection
    # decode loop (_generate's injection branch), never the no-injection
    # model.generate() branch below — that loop hardcodes greedy argmax and
    # ignores generation_kwargs entirely, so a larger budget here buys zero
    # accuracy benefit (no sampling to exploit it) while directly growing
    # peak GPU memory (a longer-lived KV cache on top of the *uncompressed*
    # injected cache for configs like A) and latency. Bumping this to 1536
    # OOM'd Config A on this GPU (~14.56GB, near-zero headroom) after 4
    # samples. Only single_agent's own SingleAgentPipelineConfig.max_new_tokens
    # should use the larger budget, since it actually runs model.generate()
    # with real sampling.
    final_max_new_tokens: int = 512
    generation_kwargs: Dict[str, object] = field(default_factory=lambda: {
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "num_beams": 1,
    })
    print_raw_outputs: bool = False
    anchor_channel_key: str = "solver_to_finalizer"
    _custom_layer_indices: Optional[List[int]] = None  # ablation: override tier selection
    use_reasoner_few_shot: bool = True
    reasoner_few_shot_example: Tuple[str, str] = (
        _REASONER_EXEMPLAR_QUESTION, _REASONER_EXEMPLAR_SOLUTION
    )
    system_prompts: List[str] = field(default_factory=lambda: [
        (
            "You are a mathematical reasoning assistant. Solve the problem step "
            "by step, showing each calculation explicitly. Continue until you "
            "reach the final answer."
        ),
        (
            "You are a critical verification agent. "
            "You have access to a prior agent's reasoning in your context. "
            "Go through each step and check the arithmetic carefully. "
            "If any step is wrong or incomplete, recompute it correctly. "
            "Then state the corrected final answer explicitly as a number."
        ),
        (
            "You are the final answer agent. "
            "You have the full reasoning and verification in your context. "
            "Output exactly one line in this format: The answer is <number>. "
            "Do not add explanation, units, or any other text."
        ),
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
    hop_texts: List[str]            # decoded text per intermediate hop; hop_texts[0] is the Reasoner


# ─── pipeline ─────────────────────────────────────────────────────────────────

class LAKVPipeline:
    """Multi-agent KV-cache relay pipeline for Qwen2.5-7B."""

    def __init__(self, model, tokenizer, config: PipelineConfig, device: str = "cuda",
                 custom_layer_indices: Optional[List[int]] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.N_LAYERS = model.config.num_hidden_layers

        # load profile if provided
        self.profile: Optional[LayerProfile] = None
        if config.profile_path:
            self.profile = LayerProfile.load(config.profile_path)

        # sub-modules
        self.selector: Optional[LayerSelector] = None
        effective_custom = custom_layer_indices or config._custom_layer_indices
        if effective_custom is not None:
            # Ablation: build a synthetic profile with Tier 1 = custom indices, Tier 3 = rest
            self.selector = LayerSelector._from_custom_indices(effective_custom, self.N_LAYERS)
        elif config.use_layer_selection and self.profile:
            self.selector = LayerSelector(self.profile)

        self.verbose = getattr(config, 'verbose', False)

        self.compressor = KVCompressor(
            mode=config.compression_mode,
            profile=self.profile if config.compression_mode == "adaptive" else None,
            outlier_clipping=config.outlier_clipping,
            clip_percentile=config.clip_percentile,
        )
        self.last_run_offset_logs: List[Dict[str, int]] = []

        self.anchor_table: Optional[AnchorTable] = None
        self.corrector: Optional[OffsetCorrector] = None
        if config.use_offset_correction:
            self.anchor_table = AnchorTable(max_size=20, entropy_threshold=0.3, min_confidence=0.5)
            self.corrector = OffsetCorrector(anchor_table=self.anchor_table)

        self._eos_ids = self._get_stop_token_ids()

    def _get_stop_token_ids(self) -> set:
        """All valid end-of-turn token ids for this model.

        The manual greedy-decode loops below can't use model.generate(), which
        normally reads model.generation_config.eos_token_id (often a list, e.g.
        Qwen's chat end token <|im_end|> alongside <|endoftext|>) automatically.
        Checking only tokenizer.eos_token_id misses those, so decoding runs past
        the model's real stop point and starts hallucinating a new chat turn.
        """
        ids = set()
        gen_eos = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        if gen_eos is not None:
            ids.update(gen_eos if isinstance(gen_eos, (list, tuple, set)) else [gen_eos])
        if self.tokenizer.eos_token_id is not None:
            ids.add(self.tokenizer.eos_token_id)
        return ids

    # ── public API ────────────────────────────────────────────────────────

    def run(self, question: str) -> RunResult:
        """Execute the 3-agent pipeline and return the final answer."""
        hop_stats: List[HopStat] = []
        hop_texts: List[str] = []
        kv_message: Optional[KVMessage] = None
        selection_mask: Optional[SelectionMask] = None
        pending_position_offset = 0
        total_bytes = 0
        answer = ""
        q_key = make_key(question)
        self.last_run_offset_logs = []

        # Compute base KV (no-prefix) once per question for anchor table updates
        base_kv: Optional[tuple] = None
        base_hidden: Optional[torch.Tensor] = None
        if self.anchor_table is not None:
            base_kv, base_hidden = compute_base_kv(
                self.model, self.tokenizer, question, self.device)

        n_agents = self.config.n_agents
        prompts = self.config.system_prompts

        for agent_idx in range(n_agents):
            system_prompt = prompts[agent_idx] if agent_idx < len(prompts) else prompts[-1]
            messages = [{"role": "system", "content": system_prompt}]
            if agent_idx == 0 and self.config.use_reasoner_few_shot:
                exemplar_q, exemplar_a = self.config.reasoner_few_shot_example
                messages.append({"role": "user", "content": exemplar_q})
                messages.append({"role": "assistant", "content": exemplar_a})
            messages.append({"role": "user", "content": question})
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            input_ids = self.tokenizer(
                prompt_text, return_tensors="pt", add_special_tokens=False
            )["input_ids"].to(self.device)

            is_last = (agent_idx == n_agents - 1)

            # ── decompress + inject KV from previous agent ───────────
            injected_kv_tuple: Optional[tuple] = None
            if kv_message is not None:
                decompressed = self.compressor.decompress(kv_message, device=self.device)

                if self.selector and selection_mask is not None:
                    decompressed = self.selector.reconstruct(
                        decompressed,
                        selection_mask,
                        strategy=self.config.reconstruction_strategy,
                    )

                injected_kv_tuple = decompressed

            if is_last:
                answer = self._generate(
                    input_ids,
                    injected_kv_tuple,
                    max_new_tokens=self.config.final_max_new_tokens,
                    position_offset=pending_position_offset,
                    receiver_prompt_len=input_ids.shape[1],
                )

            else:
                # Intermediate agent: generate reasoning, KV includes generated tokens
                raw_kv_tuple, agent_hidden, hop_text = self._generate_intermediate_with_hidden(
                    input_ids,
                    injected_kv_tuple,
                    max_new=self.config.intermediate_max_new_tokens,
                    position_offset=pending_position_offset,
                    receiver_prompt_len=input_ids.shape[1],
                )
                hop_texts.append(hop_text)
                pending_position_offset = 0

                # Populate anchor table with this agent's observation
                if self.anchor_table is not None and base_kv is not None:
                    channel_key = (
                        self.config.anchor_channel_key
                        if self.config.n_agents == 2 else f"agent_{agent_idx}"
                    )
                    self.anchor_table.update(
                        q_key, channel_key, base_kv, raw_kv_tuple, agent_hidden,
                        rope_theta=self._get_rope_theta())

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
                kv_message = self.compressor.compress(
                    filtered_kv,
                    tier_info=tier_info,
                    layer_indices=(selection_mask.kept_layer_indices if selection_mask else None),
                )

                # anchor-table offset correction
                if self.corrector and base_hidden is not None:
                    receiver_id = (
                        self.config.anchor_channel_key
                        if self.config.n_agents == 2 else f"agent_{agent_idx + 1}"
                    )
                    sender_seq_len = raw_kv_tuple[0][0].shape[2]
                    receiver_prompt_len = self._prompt_len_for_agent(
                        question, prompts, agent_idx + 1
                    )
                    kv_message, was_corrected = self.corrector.correct(
                        kv_message,
                        sender_seq_len=sender_seq_len,
                        receiver_prompt_len=receiver_prompt_len,
                        question=question,
                        channel_key=receiver_id,
                        query_hidden=base_hidden,
                        device=self.device,
                        rope_theta=self._get_rope_theta(),
                    )
                    if self.corrector.last_offset_log:
                        self.last_run_offset_logs.append(dict(self.corrector.last_offset_log))
                        pending_position_offset = self.corrector.last_offset_log.get("applied_offset", 0)

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

        if self.config.print_raw_outputs:
            print(f"\n[RAW OUTPUT Agent {n_agents-1}] {answer}\n")

        total_orig = sum(h.original_mb for h in hop_stats)
        total_comp = sum(h.compressed_mb for h in hop_stats)

        return RunResult(
            answer=answer,
            hop_stats=hop_stats,
            total_bytes_transmitted=total_bytes,
            total_compressed_mb=total_comp,
            total_original_mb=total_orig,
            overall_compression_ratio=total_orig / max(total_comp, 1e-9),
            hop_texts=hop_texts,
        )

    def _prompt_len_for_agent(self, question: str, prompts: List[str], agent_idx: int) -> int:
        if agent_idx >= self.config.n_agents:
            return 0
        system_prompt = prompts[agent_idx] if agent_idx < len(prompts) else prompts[-1]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"]
        return int(prompt_ids.shape[1])

    def _get_rope_theta(self) -> float:
        """Read rope_theta from model.config, handling the newer configs that
        nest it under a rope_parameters dict instead of exposing it directly."""
        if hasattr(self.model.config, 'rope_theta'):
            return self.model.config.rope_theta
        if hasattr(self.model.config, 'rope_parameters'):
            return self.model.config.rope_parameters.get('rope_theta', 1_000_000.0)
        return 1_000_000.0

    # ── cache conversion helpers ──────────────────────────────────────────

    @staticmethod
    def _to_tuple(pkv) -> tuple:
        """Convert DynamicCache → plain tuple of (K, V) pairs.

        Uses the stable .to_legacy_cache() API (works in all transformers
        versions that have DynamicCache, from 4.36 through 4.57+).
        Falls through for objects that are already plain tuples.
        """
        if isinstance(pkv, tuple):
            return pkv

        if isinstance(pkv, DynamicCache):
            # Older transformers versions
            if hasattr(pkv, "to_legacy_cache"):
                return pkv.to_legacy_cache()  # returns ((k0,v0),(k1,v1),...)

            # Newer transformers versions expose iteration over layer tuples:
            # (keys, values, optional_sliding_window_tensor)
            layers = []
            for layer in pkv:
                if isinstance(layer, tuple):
                    layers.append((layer[0], layer[1]))
                else:
                    layers.append((layer.keys, layer.values))
            return tuple(layers)

        # Generic fallback for cache-like iterables
        if hasattr(pkv, "__iter__"):
            layers = []
            for layer in pkv:
                if isinstance(layer, tuple):
                    layers.append((layer[0], layer[1]))
                else:
                    layers.append((layer.keys, layer.values))
            if layers:
                return tuple(layers)

        return pkv  # already a tuple

    @staticmethod
    def _to_dynamic_cache(kv_tuple: tuple) -> DynamicCache:
        """Wrap plain (K, V) tuple → DynamicCache for Transformers injection.

        Uses explicit per-layer update() — works in all Transformers versions and
        correctly initialises _seen_tokens to the actual sequence length.
        """
        if isinstance(kv_tuple, DynamicCache):
            return kv_tuple
        cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(kv_tuple):
            cache.update(k, v, layer_idx)
        return cache

    # ── forward / generate ────────────────────────────────────────────────

    def _forward(
        self,
        input_ids: torch.Tensor,
        injected_kv_tuple: Optional[tuple] = None,
        position_offset: int = 0,
        receiver_prompt_len: Optional[int] = None,
    ) -> tuple:
        """model.forward() only — NO text generation. Returns raw (K, V) tuple."""
        kwargs: dict = {"input_ids": input_ids, "use_cache": True}

        if injected_kv_tuple is not None:
            cache = self._to_dynamic_cache(injected_kv_tuple)
            cache_seq_len = injected_kv_tuple[0][0].shape[2]
            # Position MUST start after the injected KV, not at receiver_prompt_len.
            # Using receiver_prompt_len caused positions to overlap with the injected
            # cache tokens → RoPE collision → "!!!!" spam outputs.
            position_start = cache_seq_len + position_offset
            kwargs["past_key_values"] = cache
            kwargs["position_ids"] = torch.arange(
                position_start, position_start + input_ids.shape[1],
                device=self.device,
            ).unsqueeze(0)
            kwargs["attention_mask"] = torch.ones(
                (1, cache_seq_len + input_ids.shape[1]),
                dtype=torch.long, device=self.device,
            )

        with torch.no_grad():
            outputs = self.model(**kwargs)

        return self._to_tuple(outputs.past_key_values)

    def _generate(
        self,
        input_ids: torch.Tensor,
        injected_kv_tuple: Optional[tuple] = None,
        max_new_tokens: Optional[int] = None,
        position_offset: int = 0,
        receiver_prompt_len: Optional[int] = None,
    ) -> str:
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
        eos_ids = self._eos_ids
        if max_new_tokens is None:
            max_new_tokens = self.config.final_max_new_tokens

        if injected_kv_tuple is None:
            # ── No KV injection: standard generate ───────────────────
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    **self.config.generation_kwargs,
                )
            new_tokens = output_ids[0, input_ids.shape[1]:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # ── KV injection: forward-prime then manual greedy decode ─────
        cache = self._to_dynamic_cache(injected_kv_tuple)
        cache_seq_len = injected_kv_tuple[0][0].shape[2]
        # Receiver prompt must start AFTER the injected cache (not at receiver_prompt_len).
        position_start = cache_seq_len + position_offset
        prompt_len = input_ids.shape[1]
        position_ids = torch.arange(
            position_start, position_start + prompt_len,
            device=self.device,
        ).unsqueeze(0)
        attention_mask = torch.ones(
            (1, cache_seq_len + prompt_len),
            dtype=torch.long, device=self.device,
        )

        # Step 1 — Prime: forward pass with injected KV + full prompt
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                past_key_values=cache,
                position_ids=position_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
        running_cache = out.past_key_values          # natural DynamicCache
        next_logits   = out.logits[:, -1, :]         # logits for the next token

        # Step 2 — Greedy decode one token at a time with explicit position tracking
        generated: List[int] = []
        cur_pos = position_start + prompt_len
        for _ in range(max_new_tokens):
            next_token = next_logits.argmax(-1, keepdim=True)  # (1, 1)
            tok_id = next_token.item()
            if tok_id in eos_ids:
                break
            generated.append(tok_id)
            with torch.no_grad():
                out = self.model(
                    input_ids=next_token,
                    past_key_values=running_cache,
                    position_ids=torch.tensor([[cur_pos]], device=self.device),
                    use_cache=True,
                )
            running_cache = out.past_key_values
            next_logits   = out.logits[:, -1, :]
            cur_pos += 1

        return self.tokenizer.decode(generated, skip_special_tokens=True)



    def _generate_intermediate(
        self,
        input_ids: torch.Tensor,
        injected_kv_tuple: Optional[tuple] = None,
        max_new: Optional[int] = None,
        position_offset: int = 0,
        receiver_prompt_len: Optional[int] = None,
    ) -> tuple:
        """
        Generate up to max_new tokens for an intermediate agent.
        Returns the full KV tuple INCLUDING the generated tokens,
        so the next agent's context contains this agent's reasoning.
        """
        eos_ids = self._eos_ids
        if max_new is None:
            max_new = self.config.intermediate_max_new_tokens

        if injected_kv_tuple is not None:
            cache = self._to_dynamic_cache(injected_kv_tuple)
            cache_seq_len = injected_kv_tuple[0][0].shape[2]
            position_start = cache_seq_len + position_offset
            prompt_len = input_ids.shape[1]
            position_ids = torch.arange(
                position_start, position_start + prompt_len,
                device=self.device,
            ).unsqueeze(0)
            attention_mask = torch.ones(
                (1, cache_seq_len + prompt_len),
                dtype=torch.long, device=self.device,
            )
            with torch.no_grad():
                out = self.model(
                    input_ids=input_ids,
                    past_key_values=cache,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                )
        else:
            with torch.no_grad():
                out = self.model(input_ids=input_ids, use_cache=True)
            position_start = 0
            prompt_len = input_ids.shape[1]

        running_cache = out.past_key_values
        next_logits = out.logits[:, -1, :]

        cur_pos = position_start + prompt_len
        for _ in range(max_new):
            next_token = next_logits.argmax(-1, keepdim=True)
            tok_id = next_token.item()
            if tok_id in eos_ids:
                break
            with torch.no_grad():
                out = self.model(
                    input_ids=next_token,
                    past_key_values=running_cache,
                    position_ids=torch.tensor([[cur_pos]], device=self.device),
                    use_cache=True,
                )
            running_cache = out.past_key_values
            next_logits = out.logits[:, -1, :]
            cur_pos += 1

        return self._to_tuple(running_cache)

    def _generate_intermediate_with_hidden(
        self,
        input_ids: torch.Tensor,
        injected_kv_tuple: Optional[tuple] = None,
        max_new: Optional[int] = None,
        position_offset: int = 0,
        receiver_prompt_len: Optional[int] = None,
    ) -> tuple:
        """Like _generate_intermediate but also returns last hidden states for anchor embedding."""
        eos_ids = self._eos_ids
        if max_new is None:
            max_new = self.config.intermediate_max_new_tokens

        if injected_kv_tuple is not None:
            cache = self._to_dynamic_cache(injected_kv_tuple)
            cache_seq_len = injected_kv_tuple[0][0].shape[2]
            position_start = cache_seq_len + position_offset
            prompt_len = input_ids.shape[1]
            position_ids = torch.arange(
                position_start, position_start + prompt_len,
                device=self.device,
            ).unsqueeze(0)
            attention_mask = torch.ones(
                (1, cache_seq_len + prompt_len),
                dtype=torch.long, device=self.device,
            )
            with torch.no_grad():
                out = self.model(
                    input_ids=input_ids,
                    past_key_values=cache,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    output_hidden_states=True,
                )
        else:
            with torch.no_grad():
                out = self.model(input_ids=input_ids, use_cache=True,
                                 output_hidden_states=True)
            position_start = 0
            prompt_len = input_ids.shape[1]

        last_hidden = out.hidden_states[-1]  # (1, seq, hidden)
        running_cache = out.past_key_values
        next_logits = out.logits[:, -1, :]

        cur_pos = position_start + prompt_len
        generated_ids: List[int] = []
        for _ in range(max_new):
            next_token = next_logits.argmax(-1, keepdim=True)
            tok_id = next_token.item()
            if tok_id in eos_ids:
                break
            generated_ids.append(tok_id)
            with torch.no_grad():
                out = self.model(
                    input_ids=next_token,
                    past_key_values=running_cache,
                    position_ids=torch.tensor([[cur_pos]], device=self.device),
                    use_cache=True,
                )
            running_cache = out.past_key_values
            next_logits = out.logits[:, -1, :]
            cur_pos += 1

        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return self._to_tuple(running_cache), last_hidden, generated_text

    def _build_prompt(self, question: str, system_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
