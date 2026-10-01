"""
Stage 2: adds a KV-relay alternative to the sub-call boundary in
lakv/rlm_repl.py's Stage 1 REPL mechanism. Kept as a separate module, not
a modification of rlm_repl.py, so Stage 1's now-validated behavior (see
docs/PROGRESS_REPORT.md: real-model exact-match went 0/10 -> 0/10 -> 2/10
-> 3/10 -> 4/10 across five rounds of fixes) stays untouched and
re-runnable as a clean baseline.

The architectural change this requires, and why it's unavoidable: Stage
1's root loop (RLMSession) re-tokenizes the ENTIRE conversation from
scratch on every turn (make_model_generate_fn calls apply_chat_template
over the full message history each time) -- there is no persistent cache
for the root to splice anything into. Testing "relay the sub-call's KV
instead of its decoded text" requires the root itself to carry a real,
continuously-extended KV cache across turns instead of restarting from
text each time -- the same continuation pattern this project's main
pipeline (lakv/pipeline.py) and the recursive fan-in prototype
(lakv/recursive_pipeline.py) already use successfully, applied here to a
turn-by-turn REPL loop instead of a fixed number of agent hops.

Two return-channel conditions:
  "text" -- matches Stage 1 in spirit: a sub-call's answer is decoded to
            text and fed back as new tokens, but now via real KV
            continuation for the root's own turns (instead of full
            re-tokenization), so comparing this against "kv" below
            isolates the return-channel variable specifically, not also
            a difference in the root's own generation mechanism.
  "kv"   -- a sub-call's own computed KV cache (its prompt + its
            generated answer) is RoPE-position-shifted (reusing
            AnchorTable._rope_shift_k, the same primitive validated in
            tests/test_recursive_kv_merge.py) and spliced directly onto
            the root's running cache -- the sub-call's answer is never
            re-tokenized as text for the root to read. The sandboxed
            CODE still receives the sub-call's answer as a normal Python
            string (so control flow like `if "x" in result:` keeps
            working, matching Stage 1), but what the ROOT MODEL attends
            to on its next generation is the real relayed representation,
            not a re-encoded copy of the same text its own code can see.
            A minimal chat-template turn marker (no actual content) is
            still needed to prime the root's next turn -- see
            _turn_marker_ids.

All of Stage 1's validated nudges (grounding, exact-repeat detection,
low-exploration confirmation) are ported onto this new KV-continuing
loop using the same wording, not reimplemented from scratch. The yes/no
and stop-when-found instructions live in RLM_SYSTEM_PROMPT itself
(shared, imported from rlm_repl.py), so both stages get them identically.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from transformers import DynamicCache

from lakv.anchor_table import AnchorTable, question_key
from lakv.causal_audit import apply_causal_audit, KVAuditPool
from lakv.confidence import ConfidenceStats, stats_from_logits
from lakv.rlm_repl import (
    CodeSandbox, RLM_SYSTEM_PROMPT, CODE_FENCE_RE, BARE_CALL_RE,
    _references_passages_variable,
)


CHILD_SYSTEM_PROMPT = (
    "Read the given text and respond to the request in it concisely. "
    "You have no memory of any other conversation."
)

# Added 2026-09-24, branch feat/rlm-long-context. RLM_SYSTEM_PROMPT
# (imported above, unedited) has exactly ONE worked example, and it
# demonstrates reading a single passage directly, then answering --
# despite llm_query being described in prose earlier in the same
# prompt, there is no worked example of ever calling it. At the native
# ~10-passage scale this is probably harmless (direct reading is often
# a fine strategy there), but diagnosed as a plausible root cause of
# the long-context leak-rate problem (docs/RLM_LONG_CONTEXT_LOG.md):
# the model's only concrete behavioral template IS "check one passage,
# you're probably done," which is a false affordance once there are
# dozens of passages and genuine search is needed. This adds a SECOND
# worked example demonstrating batch delegation as an explicit
# alternative strategy, rather than only correcting behavior
# after-the-fact via max_direct_reads_before_nudge -- the two are not
# mutually exclusive; see RLMKVSession.__init__'s docstring.
def make_long_context_system_prompt(batch_size: int = 20) -> str:
    """Builds LONG_CONTEXT_SYSTEM_PROMPT (below) with a configurable
    delegation batch size, instead of the originally-hardcoded 20.

    Added 2026-10-01: a real-model dose-response run (n=25/length,
    docs/RLM_LONG_CONTEXT_LOG.md) found `text` decisively beating `kv`
    under the batch_size=20 prompt at BOTH 2000 and 16000 tokens (10/10
    discordant pairs favoring text, zero favoring kv) -- the opposite of
    this branch's founding hypothesis. The gap shrank from -24pts to
    -16pts between those two lengths (some support for "batch_size=20 is
    just mismatched to a ~10-20-passage haystack at 2000 tokens") but did
    NOT close (some support for a persistent, batch-size-driven
    disadvantage specific to kv even where batching is appropriate).
    This function exists to test that directly: hold context length
    fixed and vary batch_size alone. If kv recovers at a small batch_size
    (e.g. 3-5) even at 16000 tokens, that confirms the granularity
    hypothesis (kv's raw, uncompressed splice dilutes relevant signal
    across a large batch in a way text's decode-to-summary step doesn't);
    if kv still loses by a similar margin regardless of batch_size, the
    disadvantage isn't about granularity at all and the explanation is
    still open.
    """
    return RLM_SYSTEM_PROMPT + (
        f"\n\nHere is a SECOND worked example, for when there are MANY "
        f"passages (dozens or more) -- reading them one at a time yourself "
        f"is slow and easy to miss things in. Delegating a BATCH of several "
        f"passages at once to llm_query is a faster way to search:\n\n"
        f"You write, in turn 1:\n"
        f"```python\n"
        f"llm_query('\\n\\n'.join(passages[i] for i in range({batch_size})) + "
        f"'\\n\\nQuestion: ' + question)\n"
        f"```\n\n"
        f"The system reports back: [llm_query result] \"None of these "
        f"passages mention the answer.\"\n\n"
        f"You write, in turn 2:\n"
        f"```python\n"
        f"llm_query('\\n\\n'.join(passages[i] for i in range({batch_size}, "
        f"{2 * batch_size})) + '\\n\\nQuestion: ' + question)\n"
        f"```\n\n"
        f"The system reports back: [llm_query result] \"1932\"\n\n"
        f"You write, in turn 3:\n"
        f"```python\n"
        f"final_answer(\"1932\")\n"
        f"```\n\n"
        f"When there are many passages, prefer this batch-delegation "
        f"strategy over reading passages one at a time yourself."
    )


# Kept as the pre-2026-10-01 default (batch_size=20) for anything that
# imports the constant directly rather than calling the function above.
LONG_CONTEXT_SYSTEM_PROMPT = make_long_context_system_prompt(batch_size=20)


@dataclass
class RLMKVRunResult:
    answer: Optional[str]
    hit_max_turns: bool = False
    llm_query_calls: int = 0
    turn_texts: List[str] = field(default_factory=list)  # decoded root responses
    child_texts: List[str] = field(default_factory=list)  # decoded sub-call answers, for inspection
    audit_logs: List[dict] = field(default_factory=list)  # one entry per spliced child KV, "none" if not auditing
    confidence: Optional[ConfidenceStats] = None  # confidence of the turn that produced the final answer; only set when capture_confidence=True and the session finished (None on hit_max_turns)


class RLMKVSession:
    """Like RLMSession (rlm_repl.py), but the root carries a real,
    continuously-extended KV cache across turns instead of re-tokenizing
    the full conversation each time, and the sub-call boundary can relay
    real KV instead of decoded text (return_channel="kv" vs "text")."""

    def __init__(self, model, tokenizer, device: str = "cuda",
                 return_channel: str = "kv", max_turns: int = 10,
                 child_max_new_tokens: int = 200, root_max_new_tokens: int = 300,
                 causal_audit_mode: str = "none",
                 audit_pool: Optional["KVAuditPool"] = None,
                 audit_generator: Optional[torch.Generator] = None,
                 kv_decision_cue: bool = True,
                 record_audit_pool: Optional["KVAuditPool"] = None,
                 capture_confidence: bool = False,
                 max_direct_reads_before_nudge: Optional[int] = None,
                 system_prompt: Optional[str] = None,
                 repetition_nudge_max_fires: Optional[int] = 1):
        """causal_audit_mode: "none" (real child KV spliced, default) /
        "zeroed" / "random" / "mismatched" -- substitutes the child's KV
        BEFORE it is spliced onto the root's cache, mirroring
        lakv/causal_audit.py's three-tier test applied to this session's
        one splice point (there is no per-hop agent_idx here the way the
        sequential pipeline has one per agent; every child call in a
        session is treated as the same audit point, agent_idx=0). Only
        meaningful when return_channel="kv" -- ignored for "text" (no KV
        is ever spliced there, so there is nothing to substitute).
        "mismatched" requires a pre-built audit_pool (see KVAuditPool) of
        held-out sessions' real child KVs; "zeroed"/"random" need none.

        kv_decision_cue: experimental addition (2026-09-16), only takes
        effect when return_channel="kv". Adds a content-free "consider
        calling final_answer now" nudge after every splice -- see the
        comment at its call site for the real-model observation that
        motivated it. Default True; pass False to reproduce this file's
        exact pre-2026-09-16 behavior for comparison.

        record_audit_pool: when given, every child call's REAL (pre-
        substitution) KV is recorded into this pool instead of being used
        for a substitution -- mirrors LAKVPipeline's record_audit_pool
        (lakv/pipeline.py) exactly, used only by a held-out pre-pass that
        BUILDS a pool for a later 'mismatched' run to consume via
        audit_pool above. causal_audit_mode should be "none" whenever
        this is set -- never pass both record_audit_pool and a non-"none"
        causal_audit_mode on the same session, for the same reason
        LAKVPipeline forbids combining audit_pool and record_audit_pool:
        a 'mismatched' run would silently repopulate its own pool from
        the questions it's scoring.

        capture_confidence: process-level confidence signature (2026-09-18
        research audit -- see lakv/confidence.py's module docstring).
        When True, RLMKVRunResult.confidence is populated with the mean
        top-1 probability / entropy of the ROOT TURN that actually
        produced the final answer (the turn whose code called
        final_answer(...)), generalizing this session's existing
        behavioral signature (turn count, timeout rate, delegation
        calls -- see docs/PROGRESS_REPORT.md 2026-09-17) with a signal
        the other two topologies in this project can also produce.

        max_direct_reads_before_nudge: added 2026-09-24 on branch
        feat/rlm-long-context, opt-in (None = disabled, matching this
        session's pre-existing behavior exactly). Diagnosed directly
        from real-model runs (docs/RLM_LONG_CONTEXT_LOG.md): at a
        ~50-60-passage haystack, 55-60% of sessions never call
        llm_query at all, identically regardless of causal_audit_mode
        -- the root reads a few passages directly, gets lucky finding
        something relevant, and answers without ever exercising the
        audited delegation channel at all. Raising max_turns did not
        move this rate (it isn't a turn-budget problem -- the decision
        not to delegate is made in the first few turns, well before
        the turn budget is ever a constraint). When set to an integer,
        fires a one-time nudge once the session has read that many
        passages directly (touched `passages` as code, not merely
        mentioned it in prose -- see _references_passages_variable)
        WITHOUT ever calling llm_query, encouraging delegation instead
        of continuing to read one passage at a time. Deliberately NOT
        the default even on this branch: at the native ~10-passage
        scale this mechanism was originally built for, reading most or
        all passages directly is often a legitimate, sometimes
        necessary strategy, and this nudge has not been validated
        there -- only pass this for the long-context regime it was
        diagnosed for.

        system_prompt: added 2026-09-24 alongside
        max_direct_reads_before_nudge, same rationale. Defaults to the
        shared RLM_SYSTEM_PROMPT (imported from rlm_repl.py) unchanged
        -- that constant is not edited by this change, so every
        existing caller (including the native ~10-passage regime this
        session class was originally built for) is unaffected. Pass
        LONG_CONTEXT_SYSTEM_PROMPT (below) to add a second worked
        example demonstrating batch delegation via llm_query, not just
        single-passage direct reads -- see that constant's own comment
        for why the ORIGINAL prompt's only worked example may itself be
        a root cause of the leak-rate problem this nudge also targets:
        it's the model's only concrete behavioral template, and it
        demonstrates "read one passage, then answer" with no delegation
        example anywhere, despite llm_query being described in prose
        earlier in the same prompt.

        repetition_nudge_max_fires: added 2026-10-02, diagnosed from a
        free (no-GPU) re-analysis of an already-saved long-context run
        (docs/RLM_LONG_CONTEXT_LOG.md): under return_channel="kv" with
        real, uncorrupted content, 50% of sessions (10/20) repeated the
        EXACT SAME delegation query at least once -- the model gets an
        honest "not found" response and doesn't know how to adapt, so it
        just retries. Under kv_audit_random/kv_audit_mismatched, this
        never happened (0/20 each) -- any well-formed response, even a
        wrong one, seems to prevent the model from feeling stuck enough
        to repeat. The existing repetition nudge (below, in run()) only
        ever fires ONCE per session (see the original
        nudged_about_repetition boolean this replaces), so a session
        that repeats a second, third, or later time gets no further
        help. Default 1 preserves that exact prior behavior unchanged.
        Pass a higher integer (e.g. 3) or None (unlimited) to let it
        keep firing -- this targets a DIFFERENT mechanism than
        max_direct_reads_before_nudge/system_prompt above (search
        efficiency once delegation is already happening, not whether
        delegation happens at all in the first place) and is expected to
        help real/zeroed's timeout rate specifically, independent of the
        confidence-vs-honesty finding those other two are about."""
        if return_channel not in ("text", "kv"):
            raise ValueError(f"Unknown return_channel: {return_channel!r}")
        if causal_audit_mode not in ("none", "zeroed", "random", "mismatched"):
            raise ValueError(f"Unknown causal_audit_mode: {causal_audit_mode!r}")
        if record_audit_pool is not None and causal_audit_mode != "none":
            raise ValueError(
                "record_audit_pool is for a held-out pre-pass that BUILDS a pool -- "
                "causal_audit_mode must be 'none' whenever it's set, never a "
                "substitution mode (that would let a 'mismatched' run repopulate "
                "its own pool from the questions it's scoring)."
            )
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.return_channel = return_channel
        self.max_turns = max_turns
        self.child_max_new_tokens = child_max_new_tokens
        self.root_max_new_tokens = root_max_new_tokens
        self.causal_audit_mode = causal_audit_mode
        self.audit_pool = audit_pool
        self.audit_generator = audit_generator
        self.kv_decision_cue = kv_decision_cue
        self.record_audit_pool = record_audit_pool
        self.capture_confidence = capture_confidence
        self.max_direct_reads_before_nudge = max_direct_reads_before_nudge
        self.system_prompt = system_prompt if system_prompt is not None else RLM_SYSTEM_PROMPT
        self.repetition_nudge_max_fires = repetition_nudge_max_fires
        # Seeded, reproducible donor selection for "mismatched" mode --
        # matches LAKVPipeline's self._audit_rng exactly (lakv/pipeline.py),
        # so this prototype's causal audit is deterministic the same way
        # every other one in this project is, not left to fall back on
        # apply_causal_audit()'s unseeded global-random default.
        import random as _random_module
        self._audit_rng = _random_module.Random(0)
        self._eos_ids = self._get_stop_token_ids()

    def _get_stop_token_ids(self) -> set:
        ids = set()
        gen_eos = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        if gen_eos is not None:
            ids.update(gen_eos if isinstance(gen_eos, (list, tuple, set)) else [gen_eos])
        if self.tokenizer.eos_token_id is not None:
            ids.add(self.tokenizer.eos_token_id)
        return ids

    @staticmethod
    def _to_tuple(pkv) -> tuple:
        if isinstance(pkv, tuple):
            return pkv
        if isinstance(pkv, DynamicCache) and hasattr(pkv, "to_legacy_cache"):
            return pkv.to_legacy_cache()
        layers = []
        for layer in pkv:
            if isinstance(layer, tuple):
                layers.append((layer[0], layer[1]))
            else:
                layers.append((layer.keys, layer.values))
        return tuple(layers)

    @staticmethod
    def _to_dynamic_cache(kv_tuple: tuple) -> DynamicCache:
        cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(kv_tuple):
            cache.update(k, v, layer_idx)
        return cache

    def _get_rope_theta(self) -> float:
        if hasattr(self.model.config, 'rope_theta'):
            return self.model.config.rope_theta
        if hasattr(self.model.config, 'rope_parameters'):
            return self.model.config.rope_parameters.get('rope_theta', 1_000_000.0)
        return 1_000_000.0

    @staticmethod
    def _cache_len(kv: Optional[tuple]) -> int:
        return int(kv[0][0].shape[2]) if kv is not None else 0

    def _extend(self, kv: Optional[tuple], input_ids: torch.Tensor) -> Tuple[tuple, torch.Tensor]:
        """One forward pass appending `input_ids` onto `kv`. Returns the
        extended cache and logits for the final new position, so the
        caller can immediately sample the next token without a second
        forward pass."""
        cache_len = self._cache_len(kv)
        position_ids = torch.arange(
            cache_len, cache_len + input_ids.shape[1], device=self.device
        ).unsqueeze(0)
        attention_mask = torch.ones(
            (1, cache_len + input_ids.shape[1]), dtype=torch.long, device=self.device
        )
        past = self._to_dynamic_cache(kv) if kv is not None else None
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids, past_key_values=past,
                position_ids=position_ids, attention_mask=attention_mask,
                use_cache=True,
            )
        return self._to_tuple(out.past_key_values), out.logits[:, -1, :]

    def _extend_text(self, kv: Optional[tuple], text: str) -> Tuple[tuple, torch.Tensor]:
        ids = self.tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(self.device)
        return self._extend(kv, ids)

    def _turn_marker_ids(self) -> torch.Tensor:
        """Just the chat-template boilerplate that closes the previous
        turn and opens a new assistant turn (e.g. Qwen's
        "<|im_end|>\\n<|im_start|>assistant\\n"), with no actual content
        -- derived from the tokenizer's own template rather than
        hardcoded, so it stays correct if the model/template changes.
        Used to prime the root's next generation after splicing a
        child's KV directly, so the ONLY relayed content is the spliced
        KV itself, not a re-tokenized restatement of it."""
        placeholder = "PLACEHOLDER"
        full = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": placeholder}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        idx = full.index(placeholder)
        marker_text = full[idx + len(placeholder):]
        return self.tokenizer(
            marker_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(self.device)

    def _sample_next_token(self, logits: torch.Tensor, generated_ids: List[int]) -> torch.Tensor:
        """Greedy sampling with a repetition penalty applied against the
        FULL session's generated tokens so far (every turn, not just the
        current one) -- mirrors pipeline.py's _sample_next_token exactly,
        for the same reason it exists there.

        Real-model testing (2026-09-10) found severe repetition loops
        (final_answer("Brooklyn") repeated dozens of times) in BOTH
        return channels, ruling out the KV splice as the cause. Root
        cause: model.generate()'s own repetition_penalty only sees the
        input_ids passed to that specific call -- a handful of new
        tokens per turn in this continuation-based loop, not the several
        turns' worth of history actually sitting in the cache. Stage 1
        never hit this because it re-tokenizes the WHOLE conversation
        every turn, so generate()'s own penalty saw everything each time;
        this session's cache-continuation design loses that for free and
        has to track it manually instead."""
        repetition_penalty = 1.05
        if generated_ids:
            unique_ids = torch.tensor(sorted(set(generated_ids)), device=logits.device, dtype=torch.long)
            seen = logits[0, unique_ids]
            seen = torch.where(seen < 0, seen * repetition_penalty, seen / repetition_penalty)
            logits = logits.clone()
            logits[0, unique_ids] = seen
        return logits.argmax(-1, keepdim=True)  # greedy, matching Stage 1

    def _generate_from_primed(self, kv: tuple, next_logits: torch.Tensor,
                               max_new_tokens: int) -> Tuple[tuple, str, Optional[ConfidenceStats]]:
        """Given an already-primed cache and logits for its next token,
        generate token-by-token (not via model.generate()'s fast path --
        see _sample_next_token's docstring for why that loses session-
        wide repetition tracking in this continuation-based design).

        Stops as soon as a complete action (a closed ```python fence, or
        a recognizable bare final_answer/llm_query call) has been
        produced, rather than running the full max_new_tokens budget.
        Real-model testing (2026-09-10) found that without this, nothing
        stopped the model from free-associating PAST the action it just
        wrote -- inventing a fake "[stdout]" observation and a second
        fake action, all within one uninterrupted generation, before the
        real system ever got to respond. Since this loop's whole design
        extends the cache with whatever gets generated, that invented
        continuation was being baked permanently into the session's real
        notes alongside (and sometimes instead of) the real system
        feedback -- explaining messy output in BOTH return channels
        equally, since the bug lives in this shared root loop, not the
        notes-vs-summary swap being tested."""
        generated: List[int] = []
        confidence_logits: List[torch.Tensor] = []
        running_kv = kv
        logits = next_logits
        cur_pos = self._cache_len(kv)
        for _ in range(max_new_tokens):
            next_token = self._sample_next_token(logits, self._session_generated_ids + generated)
            tok_id = next_token.item()
            if tok_id in self._eos_ids:
                break
            if self.capture_confidence:
                # Raw (pre-repetition-penalty) logits -- same treatment
                # applied identically to every audit condition, so it
                # doesn't bias the real-vs-audited comparison this exists
                # to serve. See lakv/confidence.py's module docstring.
                confidence_logits.append(logits)
            generated.append(tok_id)
            running_kv, logits = self._extend(running_kv, next_token)
            cur_pos += 1
            partial_text = self.tokenizer.decode(generated, skip_special_tokens=True)
            if CODE_FENCE_RE.search(partial_text) or BARE_CALL_RE.search(partial_text):
                break
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        self._session_generated_ids.extend(generated)
        confidence = stats_from_logits(confidence_logits) if self.capture_confidence else None
        return running_kv, text, confidence

    def _run_child(self, passages_context: str) -> Tuple[tuple, str, int]:
        """Compute a sub-call's own KV cache (its prompt + its generated
        answer) as an independent forward pass -- mirrors
        recursive_pipeline.py's _run_child. Also returns the prompt
        length, so the caller can splice only the GENERATED ANSWER
        portion (see splice_child_kv's answer_only_from arg) -- the
        child's own system/user framing ("read this and respond
        concisely") is boilerplate specific to its own throwaway role,
        not meaningful content for the root to attend to, and real-model
        testing (2026-09-10) found splicing it in raw produced confused,
        sometimes blank root output (the root's first sampled token after
        the splice was immediately EOS) -- consistent with the root
        attending to a stranger's unlabeled internal scaffolding rather
        than a clean answer."""
        messages = [
            {"role": "system", "content": CHILD_SYSTEM_PROMPT},
            {"role": "user", "content": passages_context},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        input_ids = self.tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(self.device)
        with torch.no_grad():
            gen_out = self.model.generate(
                input_ids=input_ids, max_new_tokens=self.child_max_new_tokens,
                use_cache=True, return_dict_in_generate=True,
                do_sample=False, repetition_penalty=1.05,
            )
        kv_tuple = self._to_tuple(gen_out.past_key_values)
        prompt_len = int(input_ids.shape[1])
        new_tokens = gen_out.sequences[0, prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return kv_tuple, text, prompt_len

    @staticmethod
    def splice_child_kv(root_kv: tuple, child_kv: tuple, rope_theta: float,
                         answer_only_from: int = 0) -> tuple:
        """Append an independently-computed child KV cache onto the
        root's running cache, RoPE-shifting the child's K tensors to
        start at the root's current length -- the same primitive
        validated in tests/test_recursive_kv_merge.py's merge_child_kv,
        applied here to extend an existing session rather than merging
        fresh siblings into a new aggregator.

        answer_only_from: if > 0, only child_kv[:, :, answer_only_from:, :]
        is spliced (the child's generated answer), dropping its own
        prompt/framing tokens entirely -- see _run_child's docstring for
        why. The dropped prefix still shaped the answer tokens' own K/V
        via attention when the child computed them; it just isn't
        re-included as separate, foreign-framed content in the root's
        own context.

        BUG FIXED 2026-09-16: this used shift=root_len unconditionally,
        which is only correct when answer_only_from == 0 (matching
        merge_child_kv's validated usage in recursive_pipeline.py, which
        always splices a child's KV in full starting at its own position
        0 -- that usage never exercises a non-zero starting offset). Once
        answer_only_from is dropped, the retained answer tokens' own
        RoPE-encoded positions already start at answer_only_from, not 0
        -- shifting by the full root_len re-based them to
        root_len + answer_only_from instead of root_len, a real
        position-drift bug (same class as this project's own documented
        offset-corrector alignment bugs, just newly introduced by this
        file's answer-only truncation, which the original validated
        primitive was never tested against). Root-caused from a real-model
        run (2026-09-16, n=10) where every example that actually crossed
        the child-call boundary showed the "kv" channel degrading into
        word-salad or outright repetition-loop collapse relative to the
        SAME question's "text" channel -- consistent with worse drift on
        longer child prompts (bigger answer_only_from -> bigger error).
        Fix: shift by root_len - answer_only_from, so the retained
        tokens' positions land at exactly root_len, root_len+1, ...,
        matching what a real continuation would look like."""
        root_len = RLMKVSession._cache_len(root_kv)
        merged_layers = []
        for (rk, rv), (ck, cv) in zip(root_kv, child_kv):
            if answer_only_from > 0:
                ck = ck[:, :, answer_only_from:, :]
                cv = cv[:, :, answer_only_from:, :]
            ck_shifted = AnchorTable._rope_shift_k(ck, shift=root_len - answer_only_from, theta=rope_theta)
            merged_layers.append((
                torch.cat([rk, ck_shifted], dim=2),
                torch.cat([rv, cv], dim=2),
            ))
        return tuple(merged_layers)

    def run(self, question: str, passages: List[str]) -> RLMKVRunResult:
        self._session_generated_ids: List[int] = []
        q_key = question_key(question)
        audit_logs: List[dict] = []
        init_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Question: {question}"},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            init_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        input_ids = self.tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(self.device)

        kv, next_logits = self._extend(None, input_ids)
        kv, response, last_turn_confidence = self._generate_from_primed(kv, next_logits, self.root_max_new_tokens)

        pending_child_kv: Dict[str, Optional[tuple]] = {"kv": None, "prompt_len": 0}
        child_texts: List[str] = []

        def llm_query_and_capture(text: str) -> str:
            kv_tuple, decoded_text, prompt_len = self._run_child(text)
            if self.return_channel == "kv":
                pending_child_kv["kv"] = kv_tuple
                pending_child_kv["prompt_len"] = prompt_len
                if self.record_audit_pool is not None:
                    # Held-out pre-pass only (causal_audit_mode is forced
                    # "none" whenever record_audit_pool is set -- see
                    # __init__): record the REAL, full (prompt+answer)
                    # child KV, matching exactly what apply_causal_audit()
                    # substitutes against below -- pool.sample()'s
                    # target_shape check needs the same pre-truncation
                    # shape a 'mismatched' run will later compare against.
                    self.record_audit_pool.add(0, q_key, kv_tuple, question_text=question)
            child_texts.append(decoded_text)
            return decoded_text

        sandbox = CodeSandbox(question, passages, llm_query=llm_query_and_capture)
        touched_passages = False
        nudged_about_grounding = False
        repetition_nudge_fire_count = 0
        nudged_about_over_reading = False
        low_exploration_confirmed = False
        last_code: Optional[str] = None
        passage_touch_turns = 0
        llm_query_calls = 0
        turn_texts: List[str] = [response]

        for _ in range(self.max_turns):
            match = CODE_FENCE_RE.search(response)
            if match is not None:
                code = match.group(1)
            else:
                bare_match = BARE_CALL_RE.search(response)
                if bare_match is not None:
                    func_name, quote, arg = bare_match.groups()
                    code = f"{func_name}({quote}{arg}{quote})"
                else:
                    kv, next_logits = self._extend_text(
                        kv,
                        "No ```python code block found in your last "
                        "response. Write code in a fenced ```python "
                        "block to continue, or call final_answer(...) "
                        "inside one to finish.",
                    )
                    kv, response, last_turn_confidence = self._generate_from_primed(kv, next_logits, self.root_max_new_tokens)
                    turn_texts.append(response)
                    continue

            if _references_passages_variable(code):
                touched_passages = True
                passage_touch_turns += 1
            code_repeated = (last_code is not None and code.strip() == last_code.strip())
            last_code = code

            pending_child_kv["kv"] = None
            pre_call_count = sandbox.llm_query_calls
            turn = sandbox.execute(code)
            if sandbox.llm_query_calls > pre_call_count:
                llm_query_calls += sandbox.llm_query_calls - pre_call_count

            if sandbox.done:
                if passage_touch_turns < 3 and not low_exploration_confirmed:
                    low_exploration_confirmed = True
                    tentative_answer = sandbox.answer
                    sandbox.done = False
                    sandbox.answer = None
                    kv, next_logits = self._extend_text(
                        kv,
                        f"Before finalizing: you have only directly "
                        f"looked at {passage_touch_turns} of "
                        f"{len(passages)} passages so far, and your "
                        f"answer was going to be {tentative_answer!r}. "
                        "If you're confident that's correct, call "
                        "final_answer(...) again to confirm. Otherwise, "
                        "check a few more passages first -- the answer "
                        "may still be sitting in one you haven't read.",
                    )
                    kv, response, last_turn_confidence = self._generate_from_primed(kv, next_logits, self.root_max_new_tokens)
                    turn_texts.append(response)
                    continue

                return RLMKVRunResult(
                    answer=sandbox.answer, hit_max_turns=False,
                    llm_query_calls=llm_query_calls, turn_texts=turn_texts,
                    child_texts=child_texts, audit_logs=audit_logs,
                    confidence=last_turn_confidence,
                )

            if turn.error:
                kv, next_logits = self._extend_text(kv, f"[error]\n{turn.error}")
            elif self.return_channel == "kv" and pending_child_kv["kv"] is not None:
                child_kv_to_splice, audit_log = apply_causal_audit(
                    self.causal_audit_mode, pending_child_kv["kv"],
                    agent_idx=0, question_key=q_key,
                    pool=self.audit_pool, generator=self.audit_generator,
                    rng=self._audit_rng,
                )
                audit_logs.append(audit_log)
                kv = self.splice_child_kv(
                    kv, child_kv_to_splice, self._get_rope_theta(),
                    answer_only_from=pending_child_kv["prompt_len"],
                )
                kv, next_logits = self._extend(kv, self._turn_marker_ids())
                if self.kv_decision_cue:
                    # Real-model testing (2026-09-16) found the "kv" channel
                    # specifically prone to never stopping even after a
                    # sub-call plainly found the answer (one example made 8
                    # calls, the 2nd already correct, and ran out of turns
                    # without ever calling final_answer) -- a plausible
                    # mechanistic reason: "text" channel sees the actual
                    # decoded answer as an explicit "[llm_query result] ..."
                    # cue (see the stdout branch below); "kv" only gets a
                    # bare turn marker after a splice, with nothing textual
                    # signaling that something decisive may have just
                    # arrived. This adds a content-FREE decision cue --
                    # it does not reveal the child's answer (that would
                    # partially reintroduce the text channel's own
                    # mechanism, confounding the very comparison this
                    # channel exists to make) -- only prompts the model to
                    # pause and consider whether it already has enough
                    # information, leaving what "enough" means entirely to
                    # what the spliced KV itself conveys.
                    kv, next_logits = self._extend_text(
                        kv,
                        "A sub-call's answer was just added to your "
                        "context above. If you now have enough information "
                        "to answer the question, call final_answer(...) "
                        "instead of checking more passages.",
                    )
            else:
                kv, next_logits = self._extend_text(kv, f"[stdout]\n{turn.stdout}")

            can_still_nudge_repetition = (
                self.repetition_nudge_max_fires is None
                or repetition_nudge_fire_count < self.repetition_nudge_max_fires
            )
            if code_repeated and can_still_nudge_repetition:
                repetition_nudge_fire_count += 1
                # Escalate the wording once this has fired more than once
                # for the same session -- a real-model finding (2026-10-02,
                # see repetition_nudge_max_fires's docstring) showed the
                # original single-shot version left a session with no
                # further help after its first repeat, which happened in
                # half of real-content long-context sessions.
                if repetition_nudge_fire_count == 1:
                    note = (
                        "Note: that is the exact same code as your last "
                        "attempt, and it will produce the same result "
                        "again. Try a different approach -- e.g. a "
                        "different search term, or just printing each "
                        "passage directly to see what's actually there, "
                        "instead of re-running the same filter."
                    )
                else:
                    note = (
                        f"Note: you have now repeated this exact code "
                        f"{repetition_nudge_fire_count} times -- whatever "
                        "you're trying is not working. Stop and try "
                        "something genuinely different: a batch of "
                        "passages you haven't checked yet, a different "
                        "search term, or reconsider whether the answer "
                        "might require combining information from two "
                        "different passages rather than one."
                    )
                kv, next_logits = self._extend_text(kv, note)
            elif (not touched_passages and not nudged_about_grounding
                    and sandbox.llm_query_calls >= 2):
                nudged_about_grounding = True
                kv, next_logits = self._extend_text(
                    kv,
                    "Note: you have called llm_query more than once "
                    "without ever reading any of the `passages` list "
                    "(e.g. passages[i]) yourself. llm_query has no "
                    "access to `passages` either, so open-ended "
                    "questions with no passage text attached will keep "
                    "getting you unreliable guesses. Try inspecting the "
                    "actual passages directly before asking further "
                    "questions -- the answer may still be sitting in "
                    "one you haven't read.",
                )
            elif (self.max_direct_reads_before_nudge is not None
                    and passage_touch_turns >= self.max_direct_reads_before_nudge
                    and sandbox.llm_query_calls == 0
                    and not nudged_about_over_reading):
                # Opt-in, long-context-regime nudge -- see __init__'s
                # docstring for the full diagnosis. Fires once, the
                # opposite condition to the grounding nudge above (that
                # one fires for too little direct reading; this one for
                # too much with zero delegation attempted at all).
                nudged_about_over_reading = True
                kv, next_logits = self._extend_text(
                    kv,
                    f"Note: you have directly read {passage_touch_turns} "
                    f"passages yourself without ever calling llm_query. "
                    f"With this many passages, reading them one at a "
                    "time yourself is slow and easy to miss things in. "
                    "Try delegating a batch of passages to llm_query "
                    "instead -- e.g. llm_query('\\n\\n'.join(passages[i] "
                    "for i in range(10, 20)) + '\\n\\nQuestion: ' + "
                    "question) -- to search more of them at once.",
                )

            kv, response, last_turn_confidence = self._generate_from_primed(kv, next_logits, self.root_max_new_tokens)
            turn_texts.append(response)

        return RLMKVRunResult(
            answer=None, hit_max_turns=True, llm_query_calls=llm_query_calls,
            turn_texts=turn_texts, child_texts=child_texts, audit_logs=audit_logs,
        )
