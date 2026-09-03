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

from lakv.calibration_profiler import LayerProfile
from lakv.layer_selector import LayerSelector, SelectionMask
from lakv.kv_compressor import KVCompressor, KVMessage
from lakv.offset_corrector import OffsetCorrector
from lakv.anchor_table import AnchorTable, question_key as make_key, compute_base_kv


# Synthetic (non-eval) few-shot exemplars for the Reasoner: demonstrate
# restating each given number before using it, to counter the model's
# tendency to misread/substitute numbers later in its own reasoning. Three
# exemplars covering distinct operation shapes (subtract-then-multiply,
# percentage increase, multi-entity multiplicative scaling) so the pattern
# generalizes rather than just being memorized for one problem shape.
_REASONER_FEWSHOT_EXAMPLES: List[Tuple[str, str]] = [
    (
        "A bakery bakes 45 loaves of bread each morning. It sells 12 loaves to a "
        "cafe and donates 8 loaves to a shelter. It sells the rest at $3 per "
        "loaf. How much money does the bakery make from the remaining loaves?",
        "Step 1: The bakery bakes 45 loaves (given: 45 loaves baked).\n"
        "Step 2: It sells 12 loaves to a cafe (given: 12 loaves to cafe).\n"
        "Step 3: It donates 8 loaves to a shelter (given: 8 loaves donated).\n"
        "Step 4: Loaves remaining = 45 - 12 - 8 = 25.\n"
        "Step 5: Price per loaf is $3 (given: $3 per loaf).\n"
        "Step 6: Revenue = 25 * 3 = 75.\n"
        "#### 75",
    ),
    (
        "A painting costs $200. After restoration it is worth 150% more than "
        "its original price. How much is the painting worth after restoration?",
        "Step 1: The painting's original price is $200 (given: $200 original price).\n"
        "Step 2: The value increase is 150% of the original price (given: 150% more).\n"
        "Step 3: Increase in value = 200 * 1.50 = 300.\n"
        "Step 4: Value after restoration = original price + increase = 200 + 300 = 500.\n"
        "#### 500",
    ),
    (
        "Mia has 5 marbles. Noah has 3 times as many marbles as Mia. Liam has "
        "twice as many marbles as Noah. How many marbles do Mia, Noah, and "
        "Liam have together?",
        "Step 1: Mia has 5 marbles (given: 5 marbles).\n"
        "Step 2: Noah has 3 times as many as Mia (given: 3 times Mia's amount) "
        "= 3 * 5 = 15.\n"
        "Step 3: Liam has twice as many as Noah (given: 2 times Noah's amount) "
        "= 2 * 15 = 30.\n"
        "Step 4: Total = 5 + 15 + 30 = 50.\n"
        "#### 50",
    ),
]


# Reasoner/Verifier/Finalizer system prompts, keyed by dataset. GSM8K's set is
# arithmetic-specific ("mathematical," "calculation," a final number);
# HotpotQA's set is reading-comprehension-specific (identify supporting
# sentences in the given context, verify the answer is actually grounded in
# them, output a short text span instead of a number).
PROMPT_SETS: Dict[str, List[str]] = {
    "gsm8k": [
        (
            "You are a mathematical reasoning assistant. Solve the problem step "
            "by step, showing each calculation explicitly. Before using any "
            "number from the question in a calculation, first restate it "
            "exactly as given in the question, so you don't accidentally "
            "substitute a different value. Continue until you reach the final "
            "answer."
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
    ],
    "hotpotqa": [
        (
            "You are a careful reading-comprehension assistant. Read the given "
            "context passages and the question, identify the specific "
            "sentence(s) that answer it, and state a draft answer with a "
            "brief justification quoting the supporting sentence(s). Before "
            "stating your answer, first quote the exact sentence(s) from the "
            "context word-for-word, so you don't rely on a misremembered or "
            "paraphrased detail."
        ),
        (
            "You are a critical verification agent. "
            "You have access to a prior agent's draft answer and reasoning in "
            "your context. Check whether the draft answer is directly "
            "supported by the context passages given. If it is not supported, "
            "or is contradicted by the passages, correct it. Then state the "
            "corrected answer explicitly."
        ),
        (
            "You are the final answer agent. "
            "You have the full reasoning and verification in your context. "
            "Output exactly one line in this format: The answer is: <answer>. "
            "The answer should be a short word or phrase copied from the "
            "context, not a full sentence. Do not add explanation or any "
            "other text."
        ),
    ],
}


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
    # AnchorTable tuning (only consulted when use_offset_correction=True).
    # graceful_degradation=True (default) means a low-confidence/ambiguous
    # anchor match still gets applied as a blended correction rather than
    # rejected outright. Investigation this session (E/E_int8's accuracy gap
    # vs A, n=15 HotpotQA) found the Verifier hedging away answers the
    # Reasoner got right on corrected hops — plausibly correction noise from
    # exactly this kind of low-confidence blend. Set False to instead fall
    # back to uncorrected relay (raw, unshifted KV — same as if the anchor
    # table had missed) whenever confidence/entropy don't clear the bar
    # below, to isolate whether correction QUALITY (not the plumbing, which
    # is now fixed) is the accuracy bottleneck.
    anchor_graceful_degradation: bool = True
    anchor_min_confidence: float = 0.5
    anchor_entropy_threshold: float = 0.3
    # Absolute L2-distance floor on the best anchor candidate's match quality
    # — a check entropy/min_confidence above CANNOT express (they only
    # measure agreement among candidates, and are trivially "perfect" with
    # just one candidate, which is the common case here). None = disabled.
    # See AnchorTable.__init__ / query_correction for the full reasoning.
    anchor_max_distance: Optional[float] = None
    profile_path: Optional[str] = None
    # Was 200 (a Kaggle OOM workaround that outlived its reason once moved to a
    # 4090 with headroom to spare). Measured this was truncating the Reasoner/
    # Verifier hops mid-derivation on ~85-90% of questions — checked by
    # tokenizing every hop_texts entry across every result file this session
    # and counting how many land within 5 tokens of the cap. 512 is grounded
    # in single_agent's own raw-answer token-length distribution (same verbose
    # numbered-step style, rarely truncated at its 1536 budget): p90=480,
    # p95=533 tokens across 130 samples. It's a ceiling, not a fixed length —
    # generation still stops early via EOS once actually done.
    intermediate_max_new_tokens: int = 512
    # Kept at 512 (not bumped to single_agent's 1536) deliberately: every
    # existing preset's final hop always goes through the manual KV-injection
    # decode loop (_generate's injection branch), which now samples via
    # _sample_next_token when generation_kwargs["do_sample"] is True — but a
    # longer budget still directly grows peak GPU memory (a longer-lived KV
    # cache on top of the *uncompressed* injected cache for configs like A)
    # and latency regardless of sampling. Bumping this to 1536 OOM'd Config A
    # on this GPU (~14.56GB, near-zero headroom) after 4 samples. Only
    # single_agent's own SingleAgentPipelineConfig.max_new_tokens should use
    # the larger budget, since it never carries an injected cache forward
    # with real sampling.
    final_max_new_tokens: int = 512
    # Greedy by default: real eval harnesses only use sampling when paired with
    # self-consistency (multi-sample majority voting), never bare for a single
    # generation — see LAKV_V2_RUN_GUIDE session notes. temperature/top_p still
    # set to Qwen's own recommended values (its generation_config.json) so
    # they're correct whenever do_sample is turned back on (e.g. self-consistency).
    generation_kwargs: Dict[str, object] = field(default_factory=lambda: {
        "do_sample": False,
        "temperature": 0.7,
        "top_p": 0.8,
        "num_beams": 1,
    })
    print_raw_outputs: bool = False
    anchor_channel_key: str = "solver_to_finalizer"
    _custom_layer_indices: Optional[List[int]] = None  # ablation: override tier selection
    # Off by default: the exemplars measurably grow the Reasoner's prompt (and
    # therefore its KV cache — Config A's KV/hop went 37.6MB -> 56.7MB with
    # this on), inflating A-E's reported transmission cost for an accuracy
    # effect that wasn't statistically distinguishable from noise at the
    # sample sizes tested. Mechanism kept available — pass
    # use_reasoner_few_shot=True explicitly to opt back in.
    use_reasoner_few_shot: bool = False
    reasoner_few_shot_examples: List[Tuple[str, str]] = field(
        default_factory=lambda: list(_REASONER_FEWSHOT_EXAMPLES)
    )
    # Which PROMPT_SETS entry to fall back to when system_prompts isn't given
    # explicitly. Only consulted in __post_init__ below — passing
    # system_prompts explicitly (e.g. the two_agent bench prompts in
    # evaluator.py) always wins, dataset is ignored in that case.
    dataset: str = "gsm8k"
    system_prompts: Optional[List[str]] = None

    def __post_init__(self):
        if self.system_prompts is None:
            self.system_prompts = list(PROMPT_SETS[self.dataset])


@dataclass
class HopStat:
    agent_idx: int
    original_mb: float
    compressed_mb: float
    compression_ratio: float
    n_layers_transmitted: int
    n_layers_total: int             # 28
    latency_seconds: float = 0.0    # wall-clock time for this hop's generation call only


@dataclass
class RunResult:
    answer: str
    hop_stats: List[HopStat]
    total_bytes_transmitted: int
    total_compressed_mb: float
    total_original_mb: float
    overall_compression_ratio: float
    hop_texts: List[str]            # decoded text per intermediate hop; hop_texts[0] is the Reasoner
    finalizer_latency_seconds: float = 0.0  # wall-clock time for the last agent's generation call only


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
            self.anchor_table = AnchorTable(
                max_size=20,
                entropy_threshold=config.anchor_entropy_threshold,
                min_confidence=config.anchor_min_confidence,
                graceful_degradation=config.anchor_graceful_degradation,
                max_distance=config.anchor_max_distance,
            )
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
        finalizer_latency = 0.0
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
            if agent_idx == 0 and self.config.use_reasoner_few_shot and self.config.dataset == "gsm8k":
                for exemplar_q, exemplar_a in self.config.reasoner_few_shot_examples:
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
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _t0 = time.perf_counter()
                answer = self._generate(
                    input_ids,
                    injected_kv_tuple,
                    max_new_tokens=self.config.final_max_new_tokens,
                    position_offset=pending_position_offset,
                    receiver_prompt_len=input_ids.shape[1],
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                finalizer_latency = time.perf_counter() - _t0

            else:
                # Intermediate agent: generate reasoning, KV includes generated tokens
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _t0 = time.perf_counter()
                raw_kv_tuple, agent_hidden, hop_text = self._generate_intermediate_with_hidden(
                    input_ids,
                    injected_kv_tuple,
                    max_new=self.config.intermediate_max_new_tokens,
                    position_offset=pending_position_offset,
                    receiver_prompt_len=input_ids.shape[1],
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                hop_latency = time.perf_counter() - _t0
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
                        query_base_kv=base_kv,
                        device=self.device,
                        rope_theta=self._get_rope_theta(),
                    )
                    if self.corrector.last_offset_log:
                        self.last_run_offset_logs.append(dict(self.corrector.last_offset_log))
                    if was_corrected:
                        # Must match anchor_table.py's own target_shift exactly
                        # (target_prompt_len - corrected_seq_len): that's the
                        # amount the corrected cache's RoPE encoding was
                        # shifted by, so it's also the amount the receiver's
                        # NEW tokens need to continue from. The old code used
                        # last_offset_log["applied_offset"] here instead — a
                        # completely different quantity (sender_seq_len -
                        # receiver_prompt_len, computed before the anchor
                        # lookup even ran) that was also applied on cache
                        # MISSES (was_corrected=False), silently shifting an
                        # otherwise-uncorrected, correctly-positioned cache.
                        # This double inconsistency (wrong formula + applied
                        # even without a correction) is the likely cause of
                        # Config E's collapse to near-random accuracy despite
                        # the RoPE-forwarding and reject-gate fixes already
                        # applied this session.
                        corrected_seq_len = kv_message.layers[0].shape[2]
                        pending_position_offset = max(receiver_prompt_len - corrected_seq_len, 0)

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
                    latency_seconds=hop_latency,
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
            finalizer_latency_seconds=finalizer_latency,
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

    def _sample_next_token(self, logits: torch.Tensor) -> torch.Tensor:
        """Pick the next token respecting self.config.generation_kwargs
        (do_sample/temperature/top_p), or greedy argmax when do_sample is
        False. The manual KV-injection decode loops below can't use
        model.generate()'s built-in sampling (they inject cache from a
        different agent's prompt, which generate() can't accept - see the
        docstring on _generate) — this reproduces the same top-p nucleus
        sampling behavior manually so those loops aren't stuck on pure
        greedy regardless of config. Confirmed via hop_texts inspection this
        session: hop 0 (the Reasoner) goes through this same manual loop even
        though it injects nothing, so being permanently greedy here was
        costing accuracy independent of anything about KV relay itself.
        """
        kwargs = self.config.generation_kwargs
        if not kwargs.get("do_sample", False):
            return logits.argmax(-1, keepdim=True)

        temperature = max(float(kwargs.get("temperature", 1.0)), 1e-5)
        top_p = float(kwargs.get("top_p", 1.0))

        probs = torch.softmax(logits.float() / temperature, dim=-1)

        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            # Drop tokens once the cumulative mass *before* them already
            # exceeds top_p, so the token that crosses the threshold is kept.
            drop_mask = (cumulative - sorted_probs) > top_p
            sorted_probs = sorted_probs.masked_fill(drop_mask, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)
            return torch.gather(sorted_idx, -1, sampled_sorted_idx)

        return torch.multinomial(probs, num_samples=1)

    def _generate(
        self,
        input_ids: torch.Tensor,
        injected_kv_tuple: Optional[tuple] = None,
        max_new_tokens: Optional[int] = None,
        position_offset: int = 0,
        receiver_prompt_len: Optional[int] = None,
    ) -> str:
        """Run generation for the final agent.

        When KV is injected from a prior agent, model.generate() can't be called
        directly on the full prompt — its first-call bookkeeping (which tokens
        are "new" vs already covered by past_key_values) assumes input_ids is a
        continuation of whatever produced the cache. When the cache instead came
        from a *different* agent's prompt, that assumption breaks (verified
        against the installed transformers source: it computes
        next_sequence_length = input_ids.shape[1] - past_key_values.get_seq_
        length(), which goes negative here).

        Solution: prime the cache ourselves with one explicit forward() call
        over the full prompt (unavoidable), sample the first token from that,
        then hand the now-self-consistent cache off to generate() for the rest
        of decode — at that point it's just a normal continuation, so
        generate()'s fast path works correctly. Falls back to the fully manual
        per-token loop only for offset-corrected configs (position_offset != 0),
        where generate()'s default position handling doesn't know about the
        custom RoPE-offset trick those configs apply. When there is no KV to
        inject at all, this skips straight to plain model.generate().
        """
        eos_ids = self._eos_ids
        if max_new_tokens is None:
            max_new_tokens = self.config.final_max_new_tokens

        if injected_kv_tuple is None:
            # ── No KV injection: standard generate ───────────────────
            # repetition_penalty forced to 1.0 for the same reason as the
            # fast path in _generate_intermediate_with_hidden — see its
            # comment. Model default (1.05) would otherwise silently apply
            # here but nowhere in the manual loop below.
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    **{"repetition_penalty": 1.0, **self.config.generation_kwargs},
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
        cur_pos = position_start + prompt_len

        if position_offset == 0:
            # position_start == cache_seq_len exactly (no RoPE-offset trick in
            # play), so running_cache's physical length already equals the
            # RoPE position the next token needs. generate() can take over
            # for the rest of decode: sample token 1 ourselves (from the prime
            # step's logits, via the same _sample_next_token used everywhere
            # else), then hand the primed cache off to generate() for tokens
            # 2..max_new_tokens. See _generate_intermediate_with_hidden's
            # matching branch for the full account of why this is safe (and
            # the correction to an earlier, WRONG assumption about omitting
            # attention_mask here — it must be passed, covering the full
            # cache+new-token length, or generate() auto-builds its own
            # length-1 mask that triggers the exact crash this works around).
            first_token = self._sample_next_token(next_logits)
            first_tok_id = first_token.item()
            if first_tok_id in eos_ids or max_new_tokens <= 1:
                generated_ids = [] if first_tok_id in eos_ids else [first_tok_id]
            else:
                handoff_mask = torch.ones((1, cur_pos + 1), dtype=torch.long, device=self.device)
                with torch.no_grad():
                    gen_out = self.model.generate(
                        input_ids=first_token,
                        past_key_values=running_cache,
                        attention_mask=handoff_mask,
                        max_new_tokens=max_new_tokens - 1,
                        use_cache=True,
                        return_dict_in_generate=True,
                        **{"repetition_penalty": 1.0, **self.config.generation_kwargs},
                    )
                generated_ids = gen_out.sequences[0].tolist()
            return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # ── position_offset != 0 (offset-corrected configs): keep the exact
        # manual loop — generate() doesn't know about the custom RoPE-offset
        # trick those configs apply, and silently getting that wrong is
        # exactly the kind of bug that already broke Config E once before. ──
        generated: List[int] = []
        for _ in range(max_new_tokens):
            next_token = self._sample_next_token(next_logits)  # (1, 1)
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
            next_token = self._sample_next_token(next_logits)
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

        if injected_kv_tuple is None:
            # Nothing to inject on this hop (always true for the Reasoner,
            # agent_idx 0) — there's no foreign-prompt cache_position mismatch
            # to work around here, so the manual per-token loop below buys
            # nothing. Let generate() do prefill+decode natively; measured
            # this session: this hop was paying the full manual-loop tax for
            # zero reason, since it was never actually injecting anything.
            # repetition_penalty: Qwen2.5's own generation_config.json defaults
            # this to 1.05. generate() silently inherits it when not overridden
            # here — but _sample_next_token (the manual loop below, still used
            # for Verifier/Finalizer) applies no penalty at all. Force 1.0 so
            # this fast path is truly behaviorally identical to the loop it
            # replaces, not just usually close. Confirmed this mattered: without
            # it, the Reasoner rambled measurably longer (sometimes back past
            # the 512-token cap) and diverged from what the un-penalized
            # Verifier/Finalizer expected, corrupting downstream answers.
            with torch.no_grad():
                gen_out = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    **{"repetition_penalty": 1.0, **self.config.generation_kwargs},
                )
            # hidden_states[0] = the prompt-prefill step (one forward() call
            # over the whole prompt); [-1] = last layer. Same tensor this
            # branch produced before, just read off generate()'s own output
            # instead of a separate hand-rolled forward() call.
            last_hidden = gen_out.hidden_states[0][-1]  # (1, prompt_len, hidden)
            new_tokens = gen_out.sequences[0, input_ids.shape[1]:]
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            return self._to_tuple(gen_out.past_key_values), last_hidden, generated_text

        # ── Real injection (Verifier/Finalizer hops): a manual "prime" step
        # is still unavoidable — generate()'s own first-call bookkeeping
        # (which tokens are "new" vs already in the cache) breaks when the
        # cache didn't come from the same prompt as input_ids (confirmed by
        # reading the installed transformers source: it computes
        # next_sequence_length = input_ids.shape[1] - past_key_values.get_
        # seq_length(), which goes negative here and produces garbage). So we
        # do that one forward() call ourselves, exactly as before. What
        # happens AFTER priming is handled below. ─────────────────────────
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

        last_hidden = out.hidden_states[-1]  # (1, seq, hidden)
        running_cache = out.past_key_values
        next_logits = out.logits[:, -1, :]
        cur_pos = position_start + prompt_len

        if position_offset == 0:
            # Once primed, running_cache's physical length already equals
            # the RoPE position the next token needs (no offset trick in
            # play) — generate() can take over for the rest of decode.
            #
            # CORRECTION to the original version of this comment: omitting
            # attention_mask here does NOT avoid the crash — generate() auto-
            # builds its own default mask sized to match input_ids (length 1),
            # which makes attention_mask.shape[1] == input_ids.shape[1] (1==1)
            # true, wrongly telling _prefill() "input_ids is the FULL sequence,
            # slice it down to just the new part" -> next_sequence_length =
            # 1 - <real cache length> (deeply negative) -> input_ids sliced to
            # 0 elements -> crash deep in q_proj's reshape. Confirmed via an
            # actual run, not just reading the source.
            #
            # Fix: pass attention_mask explicitly, covering the FULL sequence
            # (cache so far + this 1 new token) — length cur_pos + 1, not 1.
            # That makes attention_mask.shape[1] != input_ids.shape[1], so the
            # "full sequence passed, please slice" branch never triggers, and
            # input_ids (just the 1 real new token) is used as-is, correctly.
            first_token = self._sample_next_token(next_logits)
            first_tok_id = first_token.item()
            if first_tok_id in eos_ids:
                generated_ids = []
            elif max_new <= 1:
                # Edge case (max_new is always 512 in practice, never hit):
                # still fold first_token into running_cache via one manual
                # forward, matching what the old per-token loop always did
                # even on its very last iteration — generate() isn't used
                # here since max_new_tokens=0 handling isn't worth depending
                # on for a case that doesn't occur with current configs.
                generated_ids = [first_tok_id]
                with torch.no_grad():
                    out = self.model(
                        input_ids=first_token,
                        past_key_values=running_cache,
                        position_ids=torch.tensor([[cur_pos]], device=self.device),
                        use_cache=True,
                    )
                running_cache = out.past_key_values
            else:
                handoff_mask = torch.ones((1, cur_pos + 1), dtype=torch.long, device=self.device)
                with torch.no_grad():
                    gen_out = self.model.generate(
                        input_ids=first_token,
                        past_key_values=running_cache,
                        attention_mask=handoff_mask,
                        max_new_tokens=max_new - 1,
                        use_cache=True,
                        return_dict_in_generate=True,
                        **{"repetition_penalty": 1.0, **self.config.generation_kwargs},
                    )
                generated_ids = gen_out.sequences[0].tolist()
                running_cache = gen_out.past_key_values
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            return self._to_tuple(running_cache), last_hidden, generated_text

        # ── position_offset != 0 (offset-corrected configs): keep the exact
        # manual loop — see the matching comment in _generate() above. ────
        generated_ids: List[int] = []
        for _ in range(max_new):
            next_token = self._sample_next_token(next_logits)
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
