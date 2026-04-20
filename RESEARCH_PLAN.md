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

### Gap 1 — INT4 total collapse (B_int4 = 0%)
The raw_answer outputs in B_int4 are garbage (`"The!!!!! answer! is! $120,000."`). This is a quantization bug or the 4-bit precision is genuinely too lossy.
**Action:** Check `_quantize` for INT4 — is the clamping/dequantize math correct? Run a roundtrip test. If math is fine, the result stands as a negative finding. If buggy, fix it.

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
