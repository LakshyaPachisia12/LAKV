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
from lakv.rlm_repl import (
    CodeSandbox, RLM_SYSTEM_PROMPT, CODE_FENCE_RE, BARE_CALL_RE,
    _references_passages_variable,
)


CHILD_SYSTEM_PROMPT = (
    "Read the given text and respond to the request in it concisely. "
    "You have no memory of any other conversation."
)


@dataclass
class RLMKVRunResult:
    answer: Optional[str]
    hit_max_turns: bool = False
    llm_query_calls: int = 0
    turn_texts: List[str] = field(default_factory=list)  # decoded root responses
    child_texts: List[str] = field(default_factory=list)  # decoded sub-call answers, for inspection
    audit_logs: List[dict] = field(default_factory=list)  # one entry per spliced child KV, "none" if not auditing


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
                 audit_generator: Optional[torch.Generator] = None):
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
        held-out sessions' real child KVs; "zeroed"/"random" need none."""
        if return_channel not in ("text", "kv"):
            raise ValueError(f"Unknown return_channel: {return_channel!r}")
        if causal_audit_mode not in ("none", "zeroed", "random", "mismatched"):
            raise ValueError(f"Unknown causal_audit_mode: {causal_audit_mode!r}")
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
                               max_new_tokens: int) -> Tuple[tuple, str]:
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
        running_kv = kv
        logits = next_logits
        cur_pos = self._cache_len(kv)
        for _ in range(max_new_tokens):
            next_token = self._sample_next_token(logits, self._session_generated_ids + generated)
            tok_id = next_token.item()
            if tok_id in self._eos_ids:
                break
            generated.append(tok_id)
            running_kv, logits = self._extend(running_kv, next_token)
            cur_pos += 1
            partial_text = self.tokenizer.decode(generated, skip_special_tokens=True)
            if CODE_FENCE_RE.search(partial_text) or BARE_CALL_RE.search(partial_text):
                break
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        self._session_generated_ids.extend(generated)
        return running_kv, text

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
        own context."""
        root_len = RLMKVSession._cache_len(root_kv)
        merged_layers = []
        for (rk, rv), (ck, cv) in zip(root_kv, child_kv):
            if answer_only_from > 0:
                ck = ck[:, :, answer_only_from:, :]
                cv = cv[:, :, answer_only_from:, :]
            ck_shifted = AnchorTable._rope_shift_k(ck, shift=root_len, theta=rope_theta)
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
            {"role": "system", "content": RLM_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}"},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            init_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        input_ids = self.tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(self.device)

        kv, next_logits = self._extend(None, input_ids)
        kv, response = self._generate_from_primed(kv, next_logits, self.root_max_new_tokens)

        pending_child_kv: Dict[str, Optional[tuple]] = {"kv": None, "prompt_len": 0}
        child_texts: List[str] = []

        def llm_query_and_capture(text: str) -> str:
            kv_tuple, decoded_text, prompt_len = self._run_child(text)
            if self.return_channel == "kv":
                pending_child_kv["kv"] = kv_tuple
                pending_child_kv["prompt_len"] = prompt_len
            child_texts.append(decoded_text)
            return decoded_text

        sandbox = CodeSandbox(question, passages, llm_query=llm_query_and_capture)
        touched_passages = False
        nudged_about_grounding = False
        nudged_about_repetition = False
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
                    kv, response = self._generate_from_primed(kv, next_logits, self.root_max_new_tokens)
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
                    kv, response = self._generate_from_primed(kv, next_logits, self.root_max_new_tokens)
                    turn_texts.append(response)
                    continue

                return RLMKVRunResult(
                    answer=sandbox.answer, hit_max_turns=False,
                    llm_query_calls=llm_query_calls, turn_texts=turn_texts,
                    child_texts=child_texts, audit_logs=audit_logs,
                )

            if turn.error:
                kv, next_logits = self._extend_text(kv, f"[error]\n{turn.error}")
            elif self.return_channel == "kv" and pending_child_kv["kv"] is not None:
                child_kv_to_splice, audit_log = apply_causal_audit(
                    self.causal_audit_mode, pending_child_kv["kv"],
                    agent_idx=0, question_key=q_key,
                    pool=self.audit_pool, generator=self.audit_generator,
                )
                audit_logs.append(audit_log)
                kv = self.splice_child_kv(
                    kv, child_kv_to_splice, self._get_rope_theta(),
                    answer_only_from=pending_child_kv["prompt_len"],
                )
                kv, next_logits = self._extend(kv, self._turn_marker_ids())
            else:
                kv, next_logits = self._extend_text(kv, f"[stdout]\n{turn.stdout}")

            if code_repeated and not nudged_about_repetition:
                nudged_about_repetition = True
                kv, next_logits = self._extend_text(
                    kv,
                    "Note: that is the exact same code as your last "
                    "attempt, and it will produce the same result "
                    "again. Try a different approach -- e.g. a "
                    "different search term, or just printing each "
                    "passage directly to see what's actually there, "
                    "instead of re-running the same filter.",
                )
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

            kv, response = self._generate_from_primed(kv, next_logits, self.root_max_new_tokens)
            turn_texts.append(response)

        return RLMKVRunResult(
            answer=None, hit_max_turns=True, llm_query_calls=llm_query_calls,
            turn_texts=turn_texts, child_texts=child_texts, audit_logs=audit_logs,
        )
