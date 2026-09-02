# RLM ↔ LAKV interface — Phase 1 status

This documents `lakv/rlm/`, built against the 40-section RLM spec the user
supplied in-session. Per that spec's own section 38 ("do not immediately
write hundreds of lines of code... implement incrementally"), and given this
repo's real constraints (single ~14.56GB GPU, no sharding, capstone
timeline, and an open digit-drift generation-quality bug being fixed in
parallel by someone else), this is **Phase 1 only** — a correct, tested,
inspectable RLM baseline with no KV relay yet. Scope for later phases is
listed at the bottom, not built here.

## Architecture

```
lakv/rlm/
  config.py       RLMConfig, RLMBudget (global, cross-recursion-tree limits)
  errors.py       BudgetExceededError, InvalidActionError, RecursionCycleError, InvalidChunkError
  context.py      Context (token-aware chunking/slice/search/get_chunk), Chunk
  actions.py      structured JSON action schema + parse/validate (no code execution)
  llm_backend.py  LLMBackend protocol, HFBackend (the RLM<->model seam, spec section 17)
  engine.py       RLMEngine — the state-machine loop + recursion
  tracing.py      RLMTrace, ContextAccessEvent, InvocationEvent, TerminationEvent
  metrics.py      compute_metrics(trace) — frequency/revisit/coverage/fan-out/depth
  run.py          CLI (python -m lakv.rlm.run)
  prompts/root_system.txt
```

### Mapping from the spec's suggested tree

The spec asked for `environment.py`, `repl.py`, `executor.py`, `recursion.py`,
`chunking.py` as separate files, plus `policies/` and `benchmarks/rlm/`
directories. Adapted for Phase 1's actual scope:

- `environment.py` + `chunking.py` → merged into `context.py` (one cohesive
  ~200-line module; splitting further added indirection without a second
  implementation to justify it yet).
- `repl.py` / `executor.py` → **not built**. Section 12 explicitly allows a
  controlled structured-action interface instead of unrestricted
  model-generated Python as the starting point; `actions.py` + the
  action-dispatch in `engine.py` is that interface. A sandboxed REPL is real
  security/correctness surface — build it only if Phase 1's fixed action set
  turns out to be too limited for what the model actually needs to express.
- `recursion.py` → folded into `engine.py` (`rlm_query` recurses by calling
  `RLMEngine._run_node` again with `depth+1`, sharing the same `RLMBudget`
  and `RLMTrace` instances — this is the "global budget across the whole
  tree" requirement from spec section 10).
- `policies/baseline.py` → not yet separated out. Spec section 22's three
  baselines exist as: Vanilla = `lakv_v2/pipeline/single_agent.py` (already
  in this repo), Fixed chunk map-reduce = `lakv/rlm_scaffold.py` (built
  earlier this session — one sub-call per HotpotQA paragraph, no adaptive
  exploration), RLM = `lakv/rlm/engine.py` (this module, adaptive). No new
  code needed to have all three baselines; they just live in different
  existing files.
- `benchmarks/rlm/` → not built. Needs the digit-drift bug resolved first
  (any benchmark run through it right now would inherit that noise), and the
  8K-1M+ token range from spec section 24 mostly exceeds this GPU regardless.

## What's implemented

- Depth-bounded recursive state machine (`inspect_context`, `chunk_context`,
  `list_chunks`, `get_chunk`, `search_context`, `slice_context`,
  `llm_query`, `rlm_query`, `aggregate`, `final`), one structured JSON action
  per step, validated against a fixed schema before execution.
- Global cross-recursion budgets: root iterations, total LLM calls,
  recursive calls, depth, generated/input tokens, wall-clock time, context
  accesses — a single `RLMBudget` instance is shared by reference through
  the whole recursion tree, not reset per child.
- Termination paths: `final`, `max_iterations`, every budget type
  (`budget_exceeded:<name>`), `cycle_detected` (N identical consecutive
  actions), `timeout`. A budget hit or malformed action never crashes the
  run — it returns the best available intermediate answer (last `aggregate`
  note, if any) with the termination reason recorded in the trace.
- Token-aware chunk addressing (`token_start`/`token_end`/`char_start`/
  `char_end`/`depth`/`parent_chunk_id`, stable `chunk_NNNNNN` ids) — real
  tokenizer offsets when one is passed to `Context`, a documented
  word-count approximation otherwise (for tests / no-tokenizer use).
- First-class `RLMTrace`: every context access and every model invocation is
  a JSONL-serializable event, with per-chunk revisit counts computed live.
- `compute_metrics()`: access frequency, unique chunks accessed, revisit
  rate, coverage (accessed/total tokens), depth distribution, avg fan-out,
  call counts, token/latency totals.
- 31 tests (`tests/rlm/`), all passing, using a deterministic `MockBackend`
  — no GPU/model download required to validate the engine's control flow
  (budgets, recursion, cycle detection, malformed-action handling, tracing).

## What's explicitly NOT implemented yet

- **REPL/sandboxed code execution** (spec section 13) — using structured
  actions instead, see above.
- **The KV-cache integration point** (spec sections 17-18). `HFBackend`
  wraps plain `model.generate()`; the `InferenceRequest`/`InferenceResult`
  types are the plain version of what the spec asks for, not yet a
  `KVCacheHandle`. Wiring this to `lakv/pipeline.py`'s actual KV-relay
  machinery (manual per-layer injection, RoPE re-rotation, anchor-table
  correction) is a real design task: that code assumes a fixed hop chain,
  not arbitrary RLM sub-call boundaries. Building `LAKVBackend` is the
  concrete next phase, gated on Phase 1's trace data actually showing
  reusable access patterns worth optimizing for (see Hypotheses below) —
  building it speculatively first would risk the same thing this whole
  session has been about: engineering effort spent before there's evidence
  it's justified.
- **Benchmark suite across context sizes** (spec section 24), **synthetic
  long-context tasks** (spec section 25), **visualizations** (spec section
  21) — none built. Phase 1's `tests/rlm/` covers correctness; there is no
  `benchmarks/rlm/` yet because running one meaningfully requires (a) the
  digit-drift fix landing, and (b) a realistic context-size ceiling for this
  GPU, neither of which is settled yet.
- **Parallel/concurrent sub-call dispatch** (spec section 16) — `RLMConfig`
  has a `concurrency` field reserved for this, but `engine.py` currently
  executes everything sequentially. Fine for Phase 1's correctness goal;
  matters once real benchmarks make call-count/latency reduction the point.
- Most of spec section 20's locality metrics (reuse distance, temporal
  reuse, spatial locality, working-set-over-time) — `metrics.py` lists these
  explicitly under `_deferred_metrics` rather than guessing at a
  windowing/binning design before there's a real trace to design against.

## What LAKV can observe today, and what's missing

From a Phase 1 trace (`trace.jsonl`), LAKV research can already ask:
which chunks got accessed, how many times, at what depth, whether an access
led to further recursion, and how many distinct call/invocation nodes the
recursion tree had. What it can NOT yet observe: anything about the actual
KV cache created per call (size, lifetime, reuse, eviction) — those fields
exist as explicit `None`/placeholder values in `InvocationEvent`
(`kv_cache_size_bytes`, `kv_cache_reused`, `kv_cache_recompute_tokens`,
`peak_memory_mb`) precisely so a future `LAKVBackend` can populate them
without changing the trace schema.

## Hypotheses to test next (not yet run — no benchmark exists yet)

1. Does recursive decomposition over the same underlying document (e.g.
   HotpotQA's shared paragraphs across related sub-queries) actually produce
   revisit/reuse in the trace, or does each sub-call end up touching a
   disjoint region of context? This is the load-bearing assumption behind
   EXP-9 in `docs/RESEARCH_PLAN.md`'s RLM addendum — it should be measured
   with `compute_metrics()`'s `revisit_rate`/`chunk_access_frequency` before
   any anchor-table-style reuse mechanism is built for it.
2. What does the recursion-depth vs. call-count tradeoff actually look like
   on this model/task, given `max_depth` defaults to 1? Is depth-1 (root +
   one level of `rlm_query`) enough for HotpotQA-scale contexts, or does the
   model mostly stay at `llm_query` (no recursion at all) unless the context
   is large enough to force it?
3. Per the spec's own section 39 ("critical constraint"): if the trace shows
   no exploitable locality or reuse, that is a valid, reportable negative
   result — it would mean `LAKVBackend` isn't worth building, and that's
   worth knowing before investing in it, not after.

None of these are answered yet. This document intentionally stops at what's
built and measurable, not what's hoped for.
