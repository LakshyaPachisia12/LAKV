"""The RLM state-machine loop (spec sections 5, 9-11) — the core of Phase 1.

One `_run_node` call handles one root-or-recursive invocation: it repeatedly
asks the backend for one JSON action, executes it against a Context, and
feeds the observation back in, until FINAL / a budget limit / a cycle is
hit. `rlm_query` recurses into a fresh `_run_node` over a sub-context built
from the selected chunks, sharing the same RLMBudget (by reference) and
RLMTrace (same instance) as spec section 10 requires.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import List, Optional

from lakv.rlm.actions import Action, parse_action
from lakv.rlm.config import RLMConfig
from lakv.rlm.context import Context
from lakv.rlm.errors import BudgetExceededError, InvalidActionError, RecursionCycleError
from lakv.rlm.llm_backend import InferenceRequest, LLMBackend
from lakv.rlm.metrics import compute_metrics
from lakv.rlm.tracing import RLMTrace, new_invocation_id

_PROMPT_PATH = Path(__file__).parent / "prompts" / "root_system.txt"
_SUB_SYSTEM_PROMPT = (
    "You are a focused sub-model. Answer the question using ONLY the passage "
    "given below. If the passage does not contain the answer, say so plainly. "
    "Be concise."
)


@dataclass
class RLMResult:
    answer: str
    termination_reason: str
    trace: RLMTrace
    metrics: dict
    total_latency_s: float
    n_calls: int


def _load_root_prompt_template() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


class RLMEngine:
    def __init__(self, backend: LLMBackend, config: Optional[RLMConfig] = None):
        self.backend = backend
        self.config = config or RLMConfig()
        self._root_prompt_template = _load_root_prompt_template()

    # ── public entry point ────────────────────────────────────────────────

    def run(self, context_text: str, query: str, tokenizer=None) -> RLMResult:
        trace = RLMTrace()
        t_start = perf_counter()

        root_context = Context(context_text, tokenizer=tokenizer, source_id="root")
        answer, reason = self._run_node(
            context=root_context,
            query=query,
            depth=0,
            parent_invocation_id=None,
            trace=trace,
        )

        total_latency_s = perf_counter() - t_start
        metrics = compute_metrics(trace, total_context_tokens=root_context.length)
        return RLMResult(
            answer=answer,
            termination_reason=reason,
            trace=trace,
            metrics=metrics,
            total_latency_s=total_latency_s,
            n_calls=len(trace.invocation_events()),
        )

    # ── one root-or-recursive invocation ────────────────────────────────

    def _run_node(
        self,
        context: Context,
        query: str,
        depth: int,
        parent_invocation_id: Optional[str],
        trace: RLMTrace,
    ) -> tuple:
        """Returns (answer, termination_reason)."""
        budget = self.config.budget
        history: List[str] = []
        recent_identities: List[tuple] = []
        best_effort_answer: Optional[str] = None
        node_invocation_id = new_invocation_id()

        for iteration in range(self.config.budget.max_root_iterations):
            budget.used_root_iterations += 1

            elapsed = perf_counter() - self._t_start_for_budget(trace)
            if elapsed > budget.max_wall_clock_s:
                return self._finish(best_effort_answer, "timeout", trace)
            if budget.used_total_llm_calls >= budget.max_total_llm_calls:
                return self._finish(best_effort_answer, "budget_exceeded:max_total_llm_calls", trace)
            if budget.used_context_accesses >= budget.max_context_accesses:
                return self._finish(best_effort_answer, "budget_exceeded:max_context_accesses", trace)

            prompt = self._build_root_prompt(context, query, depth, history)
            try:
                budget.used_total_llm_calls += 1
                result = self.backend.generate(InferenceRequest(
                    system_prompt=prompt,
                    user_content="Respond with one JSON action.",
                    max_new_tokens=self.config.root_max_new_tokens,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                ))
            except Exception as e:  # noqa: BLE001 — a failed call must not crash the run
                history.append(f"[model call failed: {e}]")
                continue

            trace.record_invocation(
                invocation_id=node_invocation_id,
                parent_invocation_id=parent_invocation_id,
                depth=depth,
                role="root" if depth == 0 else "recursive_root",
                chunk_ids=[],
                input_token_count=result.input_token_count,
                output_token_count=result.output_token_count,
                model_name=self.backend.model_name,
                dtype=self.backend.dtype,
                device=self.backend.device,
                latency_s=result.latency_s,
            )
            budget.used_generated_tokens += result.output_token_count
            budget.used_input_tokens += result.input_token_count

            try:
                action = parse_action(result.text)
            except InvalidActionError as e:
                history.append(f"[INVALID ACTION: {e}]")
                continue

            identity = action.identity()
            recent_identities.append(identity)
            if len(recent_identities) > self.config.max_repeated_identical_actions:
                recent_identities.pop(0)
            if (
                len(recent_identities) == self.config.max_repeated_identical_actions
                and len(set(recent_identities)) == 1
            ):
                return self._finish(best_effort_answer, "cycle_detected", trace)

            if action.action_type == "final":
                return self._finish(action.args["answer"], "final", trace)

            try:
                observation, note = self._execute_action(
                    action, context, query, depth, node_invocation_id, trace, budget
                )
            except BudgetExceededError as e:
                return self._finish(best_effort_answer, f"budget_exceeded:{e.budget_name}", trace)

            if note is not None:
                best_effort_answer = note
            history.append(f"ACTION: {action.action_type}({action.args})\nOBSERVATION: {observation}")

        return self._finish(best_effort_answer, "max_iterations", trace)

    def _t_start_for_budget(self, trace: RLMTrace) -> float:
        # wall-clock budget is per top-level run(), not per node — approximate
        # via the trace's first event timestamp so every node shares one clock.
        if trace.events:
            return trace.events[0]["timestamp"]
        import time
        return time.time()

    def _finish(self, answer: Optional[str], reason: str, trace: RLMTrace) -> tuple:
        trace.record_termination(reason)
        return (answer if answer is not None else "[no answer produced]", reason)

    # ── action execution ──────────────────────────────────────────────────

    def _execute_action(self, action: Action, context: Context, query: str, depth: int,
                         invocation_id: str, trace: RLMTrace, budget) -> tuple:
        """Returns (observation_text, best_effort_answer_or_None)."""
        a = action.action_type

        if a == "inspect_context":
            if action.args.get("chunk_id"):
                chunk = context.get_chunk(action.args["chunk_id"])
                return (f"chunk {chunk.chunk_id}: tokens[{chunk.token_start}:{chunk.token_end}] "
                        f"len={len(chunk.text)} chars", None)
            return (str(context.metadata()), None)

        if a == "chunk_context":
            chunks = context.chunk(action.args["chunk_size"], action.args.get("overlap", 0), depth=depth)
            preview = ", ".join(f"{c.chunk_id}[{c.token_start}:{c.token_end}]" for c in chunks[:20])
            more = f" (+{len(chunks) - 20} more)" if len(chunks) > 20 else ""
            return (f"created {len(chunks)} chunks: {preview}{more}", None)

        if a == "list_chunks":
            chunks = context.list_chunks()
            return (", ".join(c.chunk_id for c in chunks) or "(no chunks registered yet)", None)

        if a == "get_chunk":
            chunk = context.get_chunk(action.args["chunk_id"])
            budget.used_context_accesses += 1
            trace.record_context_access(
                invocation_id, invocation_id, depth, chunk.chunk_id,
                chunk.token_start, chunk.token_end, "get_chunk",
            )
            return (chunk.text, None)

        if a == "search_context":
            matches = context.search(action.args["pattern"])
            budget.used_context_accesses += 1
            for c in matches:
                trace.record_context_access(
                    invocation_id, invocation_id, depth, c.chunk_id,
                    c.token_start, c.token_end, "search_context", query=action.args["pattern"],
                )
            preview = ", ".join(c.chunk_id for c in matches[:20])
            return (f"{len(matches)} matches: {preview}" if matches else "no matches", None)

        if a == "slice_context":
            budget.used_context_accesses += 1
            text = context.slice(action.args["token_start"], action.args["token_end"])
            trace.record_context_access(
                invocation_id, invocation_id, depth, None,
                action.args["token_start"], action.args["token_end"], "slice_context",
            )
            return (text, None)

        if a == "llm_query":
            return self._do_sub_query(action, context, depth, invocation_id, trace, budget, recursive=False)

        if a == "rlm_query":
            return self._do_sub_query(action, context, depth, invocation_id, trace, budget, recursive=True)

        if a == "aggregate":
            return (f"noted: {action.args['note']}", action.args["note"])

        raise InvalidActionError(f"unhandled action type: {a}")

    def _do_sub_query(self, action: Action, context: Context, depth: int, invocation_id: str,
                       trace: RLMTrace, budget, recursive: bool) -> tuple:
        if budget.used_total_llm_calls >= budget.max_total_llm_calls:
            raise BudgetExceededError("max_total_llm_calls", budget.max_total_llm_calls,
                                       budget.used_total_llm_calls)
        if recursive:
            if depth + 1 > budget.max_depth:
                return (f"[rlm_query rejected: max_depth={budget.max_depth} reached; "
                         f"use llm_query instead]", None)
            if budget.used_recursive_calls >= budget.max_recursive_calls:
                raise BudgetExceededError("max_recursive_calls", budget.max_recursive_calls,
                                           budget.used_recursive_calls)

        chunk_ids = action.args["chunk_ids"]
        sub_query = action.args["query"]
        chunks = [context.get_chunk(cid) for cid in chunk_ids]
        chunk_text = "\n\n".join(c.text for c in chunks)

        budget.used_context_accesses += len(chunks)
        for c in chunks:
            trace.record_context_access(
                invocation_id, invocation_id, depth, c.chunk_id, c.token_start, c.token_end,
                "rlm_query" if recursive else "llm_query", query=sub_query, led_to_recursion=recursive,
            )

        if recursive:
            budget.used_recursive_calls += 1
            sub_context = Context(chunk_text, tokenizer=context.tokenizer,
                                   source_id=f"{context.source_id}/depth{depth+1}")
            answer, _reason = self._run_node(sub_context, sub_query, depth + 1, invocation_id, trace)
            return (answer, None)

        budget.used_total_llm_calls += 1
        result = self.backend.generate(InferenceRequest(
            system_prompt=_SUB_SYSTEM_PROMPT,
            user_content=f"Passage:\n{chunk_text}\n\nQuestion: {sub_query}",
            max_new_tokens=self.config.sub_max_new_tokens,
            do_sample=self.config.do_sample,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        ))
        trace.record_invocation(
            invocation_id=new_invocation_id(),
            parent_invocation_id=invocation_id,
            depth=depth,
            role="sub",
            chunk_ids=chunk_ids,
            input_token_count=result.input_token_count,
            output_token_count=result.output_token_count,
            model_name=self.backend.model_name,
            dtype=self.backend.dtype,
            device=self.backend.device,
            latency_s=result.latency_s,
        )
        budget.used_generated_tokens += result.output_token_count
        budget.used_input_tokens += result.input_token_count
        return (result.text.strip(), None)

    def _build_root_prompt(self, context: Context, query: str, depth: int, history: List[str]) -> str:
        budget = self.config.budget
        execution_state = (
            f"depth={depth}/{budget.max_depth}, "
            f"llm_calls_used={budget.used_total_llm_calls}/{budget.max_total_llm_calls}, "
            f"recursive_calls_used={budget.used_recursive_calls}/{budget.max_recursive_calls}"
        )
        history_text = "\n---\n".join(history[-8:]) if history else "(none yet)"
        return self._root_prompt_template.format(
            query=query,
            context_metadata=context.metadata(),
            execution_state=execution_state,
            history=history_text,
        )
