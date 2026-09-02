# LAKV Research: Experiment Goals + Implementation Plan

---

## Research Narrative (The Paper We Are Writing)

> "In multi-agent LLM pipelines, KV cache relay between agents is expensive and degradation-prone. We show that (1) calibration-driven layer selection counterintuitively *improves* relay accuracy by dropping noisy low-signal layers, (2) tiered quantization preserves this gain at 2x compression, and (3) a shared cross-agent anchor table further reduces relay cost by correcting KV deviations across agents without full retransmission."

Three claims. Each needs an experiment that isolates it.

---

## Experiment Goals (Ordered by Priority)

### EXP-1 — Single-Agent Ceiling (MISSING, CRITICAL)
**Goal:** Establish what Qwen2.5-7B actually scores on GSM8K with no relay at all.
**Why:** Our Config A is only 24%. We don't know if that's the KV handoff hurting it or just bad prompts. We need a true ceiling.
**Config:** `single_agent_baseline` from v2 eval runner, n=200.
**Expected:** ~75–82% (matching KVCOMM's reported 79.6–81.5%).
**Claim enabled:** "Multi-agent relay with our best config (D, 47%) recovers X% of the single-agent ceiling at 2x compression."

---

### EXP-2 — Reproduction + Extended Baseline Table (DONE, needs n=200 rerun clean)
**Goal:** Clean reproduction of Configs A/B_int8/B_int4/C/D on n=200 with single-agent baseline included.
**Current state:** We have n=200 results but no single-agent row.
**Action:** Add single-agent to the same eval run, output a unified table.
**Expected table:**

| Config | Accuracy | Comp Ratio | Layers | MB/hop |
|--------|----------|------------|--------|--------|
| Single-agent (ceiling) | ~78% | — | — | — |
| A (3-agent, no compression) | ~24% | 1.0x | 28 | ~28 |
| B_int8 (uniform INT8) | ~25% | 2.0x | 28 | ~14 |
| B_int4 (uniform INT4) | ~0% | 4.0x | 28 | ~7 |
| C (tier selection only) | ~49% | 1.0x | 20 | ~20 |
| D (tier select + adaptive INT8) | ~47% | 2.0x | 20 | ~10 |

**Key finding to highlight:** C > A by +25pp with *fewer* layers. B_int4 = 0%. D ≈ C accuracy at 2x compression.

---

### EXP-3 — Ablation: Why Does Layer Selection Help? (NEW)
**Goal:** Understand the +25pp gain from Config C. Is it:
  (a) Fewer layers = less noise injected?
  (b) The specific tiers dropped (Tier 3) are the right ones?
  (c) Random layer dropping would do the same thing?
**Configs to add:**
- `C_random_20` — drop 8 random layers (not tier-based), keep 20 — run 3 seeds
- `C_top20` — keep layers 0–19 (naive top-N, no calibration)
- `C_bottom20` — keep layers 8–27 (naive bottom-N)
**Expected:** Tier-based selection beats random and naive top/bottom → validates the calibration signal.
**Claim enabled:** "Calibration-driven tier assignment is necessary; random or positional selection does not reproduce the gain."

---

### EXP-4 — Reconstruction Strategy Ablation (QUICK WIN)
**Goal:** Does `mean_fill` vs `zeros` vs `nearest` matter for dropped layer reconstruction?
**Configs:** C/D with each reconstruction strategy.
**Currently:** everything uses `zeros`. The v2 runner uses `mean_fill`.
**Expected:** mean_fill > zeros, nearest ≈ mean_fill.
**Claim enabled:** Reconstruction strategy is a hyperparameter we have characterized.

---

### EXP-5 — Anchor Table: Hit Rate + Accuracy (NEW, NOVEL)
**Goal:** Show that the shared anchor table (a) accumulates hits over a multi-question run, and (b) questions that get a hit maintain accuracy vs questions that don't.
**Metric to add:** Per-question `anchor_hit: bool`, `anchor_confidence: float`, hit rate @ n=10/50/100/200.
**Setup:** Run D+anchor on n=200 GSM8K sequentially. Plot hit rate vs question index.
**Expected:** Hit rate starts ~0, grows to 30–60% by question 100 as pool warms up.
**Ablation:** Shared pool vs per-agent pool — does sharing across agents (0→1) give faster warmup?
**Claim enabled:** "Cross-agent sharing reduces cold-start questions by X% vs per-agent pools."

---

### EXP-6 — End-to-End Latency (NEEDED FOR PAPER)
**Goal:** Actual wall-clock TTFT numbers, not just MB. The MB proxy isn't enough for a systems paper.
**Metric:** Time from question input to first generated token for each agent, logged per-sample.
**Setup:** Profile pipeline on GPU, separate prefill latency from decode latency.
**Target:** Show anchor table questions have lower TTFT than non-hit questions.
**Note:** Only needed if targeting a systems venue. For NLP venue, MB + accuracy is sufficient.

---

## Current Gaps to Fix Before Running Experiments

### Gap 1 — INT4 total collapse (B_int4 = 0%) — CLOSED, negative finding
Confirmed reproducible across 3 separate runs (Mar 31 n=200, Sep 2 x2 n=5) spanning
5 months and every relevant pipeline fix in between: `B_int4` is 0% every time.
`outlier_clipping=True` is correctly wired end-to-end for this config
(`evaluator.py` PRESETS → `PipelineConfig` → `KVCompressor`,
`lakv/pipeline.py:202`), and `tests/test_kv_compressor_clipping.py` (CPU-only,
synthetic tensors) confirms the clipping math itself works — it measurably
reduces reconstruction MSE. So this is not an implementation bug: **uniform
4-bit KV quantization causes complete generation collapse even with correct
outlier-aware clipping.** Treat as a genuine precision floor and a reportable
negative result, not an open gap. Recommend dropping `B_int4` from quick
iteration runs (it's also consistently the slowest config, ~2x every other
config's latency) and keeping it only in the final full-scale table as the
negative data point.

### Gap 2 — Config A only 24% (single-agent needed for framing)
Without EXP-1, we can't frame the story properly.

### Gap 3 — `OffsetCorrector` is a no-op
Anchor table not implemented → EXP-5 blocked.

### Gap 4 — v2 eval runner doesn't track compression stats
`eval_summary.json` has only accuracy + latency, no MB/compression_ratio fields. Needed for the unified table.

### Gap 5 — No per-agent latency breakdown
Current timing wraps the full pipeline. Need separate solver_latency / finalizer_latency.

---

## Implementation Order

### Phase 1 — Fix & Baseline (run EXP-1 + EXP-2 cleanly)
1. Add `single_agent_baseline` to the old evaluator's run loop (or wire it in alongside configs A-D)
2. Add compression stats (MB, ratio, layers) to v2 eval runner output
3. Run full n=200 eval: single_agent + A + B_int8 + B_int4 + C + D → unified CSV

### Phase 2 — Ablations (EXP-3 + EXP-4)
4. Add `C_random`, `C_top20`, `C_bottom20` to PRESETS in `evaluator.py`
5. Add reconstruction strategy variants to v2 eval runner
6. Run ablations on n=100 (smaller for speed)

### Phase 3 — Anchor Table Implementation (EXP-5)
7. Create `anchor_table.py`:
   - `AnchorEntry`: base_kv (28 layers K+V), per-agent offsets ΔK/ΔV, embedding, metadata
   - `AnchorTable`: query (softmax-weighted retrieval), update (compute + store delta), evict (LFU, max 20)
8. Fill `offset_corrector.py` with real correction logic
9. Wire into `pipeline.py` PipelineConfig + run loop
10. Wire into `lakv_v2/pipeline/two_agent.py`
11. Add `Config_E` (D + anchor) to evaluator PRESETS
12. Add `lakv_v2_anchor` to UnifiedEvalRunner
13. Add per-question hit rate logging to both evaluators

### Phase 4 — Full Experiment Run (EXP-5)
14. Run Config E + lakv_v2_anchor on n=200 with hit rate logging
15. Run shared-pool vs per-agent-pool ablation

---

## Target Results Table for Paper

| Config | Accuracy | vs Ceiling | Comp Ratio | Layers | MB/hop | Novel? |
|--------|----------|------------|------------|--------|--------|--------|
| Single-agent | ~78% | 100% | — | — | — | baseline |
| A (3-agent raw) | ~24% | 31% | 1.0x | 28 | 28 | baseline |
| B_int8 | ~25% | 32% | 2.0x | 28 | 14 | baseline |
| B_int4 | ~0% | 0% | 4.0x | 28 | 7 | negative |
| C (tier select) | ~49% | 63% | 1.0x | 20 | 20 | **ours** |
| D (tier+INT8) | ~47% | 60% | 2.0x | 20 | 10 | **ours** |
| C_random | ~30%? | — | 1.0x | 20 | 20 | ablation |
| E (D+anchor) | ~50%? | 64% | 2.0x+ | 20 | <10 | **ours** |

---

## Files to Modify (Implementation)

| File | Change |
|------|--------|
| `evaluator.py` | Add single_agent_baseline, C_random/C_top20, Config_E to PRESETS |
| `run.py` | Wire single_agent into experiment mode |
| `anchor_table.py` | **CREATE** |
| `offset_corrector.py` | Fill in real correction using AnchorTable |
| `pipeline.py` | Wire AnchorTable into PipelineConfig + run() |
| `lakv_v2/eval_runner.py` | Add compression stats, hit rate logging, anchor config |
| `lakv_v2/pipeline/two_agent.py` | Wire AnchorTable |
| `lakv_v2/cache/anchor_table.py` | **CREATE** (v2 version) |

---

## Success Criteria

The paper is publishable at an NLP workshop (NeurIPS Efficient NLP / ACL SRW) if:
- [ ] EXP-1: Single-agent ceiling confirmed (~75%+)
- [ ] EXP-2: Config D shows ~47% at 2.0x compression reproducibly
- [ ] EXP-3: Tier-based selection beats random by >10pp → calibration justified
- [ ] EXP-5: Anchor table hit rate >30% by question 100, accuracy maintained

Stretch (main conference track):
- [ ] Second dataset (MATH or MMLU subset)
- [ ] Latency numbers (TTFT ms, not just MB)
- [ ] Theoretical bound on correction error vs anchor pool size

---

## Addendum: RLM Integration (`feat/rlm`, branched from `lakshya/hotpotqa`)

### Research Narrative Extension

> Recursive Language Models (Zhang, Kraska, Khattab — MIT CSAIL, arXiv:2512.24601)
> handle near-infinite context by treating the prompt as a REPL-environment
> variable and recursively decomposing it into sub-LM calls. Today every
> recursive call re-prefills its context chunk from scratch, even when
> sibling/ancestor calls already encoded overlapping content. We extend LAKV's
> claim: KV relay's fidelity-preserving compression (layer selection +
> quantization + anchor-table correction) applies directly at a recursive
> call boundary, cutting redundant prefill cost, and RLM's structurally
> repeated context access is a *better* fit for the anchor table's shared-pool
> reuse claim than GSM8K's independent questions ever were.

HotpotQA is already wired into this branch (`lakv/qa_scoring.py`,
`evaluator.py` hotpotqa prompts, `single_agent.py`) — its 10-paragraph
distractor setting is the cheap first testbed before committing to a real
long-context benchmark (RULER/BABILong/LongBench v2), since it already gives
multi-hop, multi-paragraph structure with EM/F1 scoring for free.

### EXP-7 — RLM Text-Only Reproduction (BASELINE, MISSING, CRITICAL)
**Goal:** Implement vanilla RLM (root LM + REPL environment + recursive
sub-LM calls), no KV tricks, on HotpotQA's distractor setting.
**Why:** Need an honest baseline on both axes — accuracy *and* per-call
latency — before claiming KV relay helps either.
**Config:** Root Qwen2.5-7B-Instruct emits REPL code to slice/query the 10
distractor paragraphs; each sub-call is a fresh text-only `generate()` call,
same model.
**Metric:** EM/F1 (via `qa_scoring.py`) + call count + latency per sample.
**Expected:** EM/F1 comparable to or above the `single_agent` HotpotQA
baseline; latency dominated by repeated chunk re-prefill.
**Claim enabled:** "RLM recursion is at least as accurate as flat
single-context QA, but per-sample latency is dominated by redundant
re-prefill" — motivates EXP-8.

---

### EXP-8 — KV-Relay at One Recursion Edge (NEW, CORE THESIS)
**Goal:** Swap exactly one call boundary (root→child) from a fresh text
prompt to LAKV's layer-selected/quantized KV relay; hold everything else
identical to EXP-7.
**Mechanism:** Reuse `layer_selector.py` + `kv_compressor.py` to hand the
child a compressed KV of its assigned chunk instead of re-prefilling raw
text; apply `offset_corrector.py`'s RoPE re-rotation for the position shift
into the child's prompt wrapper (root's REPL wrapper differs from a
sub-call's system+chunk wrapper).
**Metric:** Per-call TTFT/prefill latency (relay vs. re-prefill) at matched
EM/F1.
**Expected:** Relay latency < re-prefill latency at ≤2pp EM/F1 loss,
mirroring Config C/D's tradeoff on GSM8K.
**Claim enabled:** "Cache reuse across a recursive call boundary cuts prefill
cost without a fidelity penalty" — the core RLM+LAKV thesis.

---

### EXP-9 — Anchor Table Reuse Across Overlapping Recursive Queries (NEW, NOVEL)
**Goal:** Unlike GSM8K (independent questions, weak anchor-reuse signal),
HotpotQA's distractor paragraphs are structurally revisited by multiple
sub-queries within and across samples sharing supporting paragraphs. Key the
anchor table by chunk/paragraph id instead of `question_key` and measure hit
rate growth.
**Setup:** Run the EXP-8 config + anchor table over n=200 HotpotQA samples
sequentially; log `anchor_hit`/`anchor_confidence` per recursive call.
**Expected:** Faster/higher hit-rate growth than GSM8K's Config E numbers,
since paragraph reuse here is structural, not incidental.
**Claim enabled:** "Cross-call anchor sharing is most valuable exactly where
recursive decomposition revisits shared context, which RLM-style workloads
do structurally."

---

### EXP-10 — Recursion Depth vs. Error Accumulation (NEW, RISK CHECK)
**Goal:** LAKV's chain relay already showed one bad correction can collapse
accuracy (Config E's pre-fix 4% collapse). A recursion tree compounds this
risk across many branches. Measure EM/F1 as a function of recursion
depth/fan-out with KV relay on, and check whether errors compound faster
than in vanilla RLM.
**Setup:** Force fixed depth (1/2/3 levels) and fan-out (2/4/8 children) on a
controlled long-context task (concatenated HotpotQA contexts, or a
needle-style harness); compare KV-relay vs. text-relay accuracy-decay curves.
**Expected/Risk:** If KV-relay's decay curve is steeper than text-relay's,
compression/anchor confidence thresholds need tightening with depth — a
negative but still publishable finding either way.
**Claim enabled:** Characterizes the safe operating envelope (max
depth/fan-out) for KV-relay recursion.

---

### Current Gaps to Fix Before Running RLM Experiments

**Gap 6 — No RLM scaffold exists yet.** Need a REPL-driving root-call loop
(root LM emits code that queries a context object; sub-LM calls execute
chunk slices) — net-new, not present in `lakv/` or `lakv_v2/`.

**Gap 7 — Recursive call graph doesn't fit the current fixed hop chain.**
`PipelineConfig`'s Reasoner→Verifier→Aggregator / Solver→Finalizer shape is
linear; RLM needs an N-ary/tree-shaped relay entry point instead of 2-3
hardcoded hops.

**Gap 8 — Anchor table is keyed by `(question_key, agent_id)`.** EXP-9 needs
a chunk/paragraph-keyed variant (or a second `AnchorTable` instance) without
breaking the existing GSM8K-keyed usage.

**Gap 9 — No long-context benchmark beyond HotpotQA's 10-paragraph
distractor setting is wired in.** EXP-10 needs either concatenated-context
HotpotQA or a small RULER/BABILong-style harness.

---

### Implementation Order (RLM Phases)

**Phase 5 — RLM Scaffold (EXP-7)**
16. Build a minimal REPL-driving root loop (`lakv/rlm_scaffold.py`) with
    sub-LM calls as plain `generate()` calls.
17. Wire HotpotQA distractor contexts as the REPL "environment" object;
    score with `qa_scoring.py`.
18. Log per-call latency + call count alongside EM/F1.

**Phase 6 — KV Relay at the Recursion Edge (EXP-8)**
19. Add an N-ary/tree-shaped relay entry point (generalize `PipelineConfig`
    or add a parallel `RLMPipelineConfig`).
20. Reuse `layer_selector.py` + `kv_compressor.py` + `offset_corrector.py` at
    the root→child call boundary.
21. Compare relay vs. re-prefill latency and EM/F1 on n=50-100 HotpotQA
    samples first (cheap check before n=200).

**Phase 7 — Anchor Reuse Across Recursion (EXP-9)**
22. Add a chunk/paragraph-keyed anchor variant to `anchor_table.py` (or a new
    `AnchorTable` instance).
23. Run n=200 sequential HotpotQA with hit-rate logging; compare warm-up
    curve to GSM8K Config E.

**Phase 8 — Depth/Fan-out Risk Characterization (EXP-10)**
24. Build a controlled synthetic long-context harness (concatenated
    contexts, fixed depth/fan-out).
25. Run KV-relay vs. text-relay decay curves; report the safe operating
    envelope.

---

### Files to Modify/Create (RLM Work)

| File | Change |
|------|--------|
| `lakv/rlm_scaffold.py` | **CREATE** — REPL-driving root loop + sub-LM call dispatch |
| `lakv/anchor_table.py` | Add chunk/paragraph-keyed variant alongside existing `question_key` usage |
| `lakv/pipeline.py` | Generalize hop chain to an N-ary relay entry point for recursive calls |
| `lakv/qa_scoring.py` | Reuse as-is for HotpotQA EM/F1 scoring of RLM outputs |
| `lakv/evaluator.py` | Add RLM configs (text-only baseline, KV-relay variant) to `PRESETS` |

---

### Success Criteria (RLM Addendum)

- [ ] EXP-7: RLM text-only baseline reproduces HotpotQA EM/F1 near the
      single-agent ceiling, with latency/call-count logged
- [ ] EXP-8: KV-relay at one recursion edge cuts prefill latency at ≤2pp EM/F1 cost
- [ ] EXP-9: Chunk-keyed anchor table shows faster/higher hit-rate growth
      than GSM8K's question-keyed anchor table
- [ ] EXP-10: Depth/fan-out decay curve characterized; safe operating
      envelope documented (even if the finding is negative)
