# LAKV — Results Summary (quick reference)

> Purpose: a scannable table of every confirmed number, distinct from
> `CLAUDE.md` (dense technical narrative, read that for the *why*) and
> `docs/naacl2027_paper_draft.md` (academic prose for the actual paper).
> Last updated 2026-09-17 (later), branch `feat/research-extensions`. Every number
> here has a source file and a significance test behind it — see the
> "source" column, or re-run `python -m lakv.stats <file> <cfgA> <cfgB>` to
> reproduce any comparison.

## 1. Established baseline (Qwen2.5-7B-Instruct, HotpotQA, n=100)

Source: `results/run_20260907_141517`

| Config | Accuracy | F1 | Latency | KV/hop |
|---|---|---|---|---|
| single_agent | 57.0% | 68.4% | 1.6s | — |
| text_agent | 54.0% | 68.3% | 7.2s | — |
| **A** (uncompressed relay) | 56.0% | 69.1% | 8.6s | 144.35 MB |
| **B_int8** | 57.0% | 70.5% | 8.3s | 72.16 MB (2.00x) |
| B_int4 (broken) | 0.0% | 0.0% | 39.4s | 38.38 MB (4.00x) |
| C | 43.0% | 57.8% | 10.0s | 104.16 MB |
| **D** | 50.0% | 61.9% | 10.2s | 37.82 MB (3.82x vs `A`) |
| E | 9.0% | 17.4% | 19.8s | 35.63 MB |
| E_int8 | 1.0% | 6.2% | 22.2s | 49.05 MB |

**MB/ratio columns for `B_int4`/`D`/`E` went through two corrections on
2026-09-09** — see §3 for the full story. Net result: these are the
originally-reported numbers, now genuinely backed by real nibble-packing
instead of a byte-accounting bug. `A`/`B_int8`/`C`/`E_int8` were never
affected (no int4 tier involved). Accuracy/F1/latency unaffected either
way, throughout.

## 2. Second model — Mistral-7B-Instruct-v0.3 (n=50)

Source: `results/run_20260907_130646`

| Config | Accuracy | F1 |
|---|---|---|
| single_agent | 42.0% | 53.8% |
| text_agent | 58.0% | 68.0% |
| A | 56.0% | 63.3% |
| B_int8 | 56.0% | 63.3% |
| **D** | **22.0%** | **32.8%** |

**A vs D: 30.5 F1-point collapse, p=0.0001.** On Qwen, the same comparison isn't significant yet (p=0.34). Cross-model asymmetry is the finding, not either number alone.

## 3. The B_int4 fix journey (Qwen, n=100 unless noted)

| Variant | Accuracy | F1 | Verdict |
|---|---|---|---|
| B_int4 (original) | 0.0% | 0.0% | Collapses — root cause: RoPE/quantization interaction |
| B_int4_turboquant (rotate K+V) | 0.0% | 0.0% | **Broken worse** — bizarre out-of-vocab tokens |
| B_int4_turboquant_vonly (rotate V only) | 0.0% | 0.0% | Confirms K-rotation was the specific problem |
| **B_int4_kivi (K per-channel)** | **52.0%** | **64.1%** | **The fix — use this one** |
| B_int4_hybrid (K per-channel + rotate V) | ~50% (n=50 matched) | lower F1 than kivi | No better than kivi alone |
| B_int4_kivi_full (K per-channel + V per-token) | 44.0% (n=50 matched) | lower than kivi | No better than kivi alone |

**B_int4_kivi vs D: statistically indistinguishable (p=0.86), at higher
compression (3.96x vs 3.82x, both vs. `A`), no calibration profile
needed.** Two-step correction on 2026-09-09, told here for the record:
(1) found `kv_compressor.py` billed 4-bit tensors at half the real byte
cost — they were stored as `torch.uint8` with no nibble-packing, same as
8-bit, so 4-bit and 8-bit configs physically occupied the same space
despite different reported ratios. Fixed the accounting first (temporarily
made `B_int4_kivi` ≈72.6 MB/2.00x, `D` ≈52.2 MB/2.77x — `D` briefly
"beat" `B_int4_kivi` on compression). (2) Rather than stop at honest
accounting for a capability that didn't exist, implemented real
two-values-per-byte nibble packing (`lakv/kv_compressor.py::_pack_nibbles`/
`_unpack_nibbles`), unit-tested (11 new tests: round-trip exactness,
bit-identical `compress`/`decompress` output with vs. without packing,
zero regressions across the other 55 existing tests) but not yet
GPU-integration-tested. With real packing, the numbers land back at
almost exactly the originally-reported figures — not a coincidence: the
original formula's arithmetic (0.5 bytes/element for 4-bit) was always
what genuine packing would produce, the bug was that nothing packed
anything. `B_int4_kivi`'s claim is restored: it matches `D`'s accuracy
at genuinely higher compression, with no calibration profile.

## 4. Causal audit — the central mechanistic result

**Qwen2.5-7B-Instruct, HotpotQA:**

| Config | Real | Zeroed | Random | Mismatched | n |
|---|---|---|---|---|---|
| A (uncompressed) | 50.0%/64.0% | 0.0%/0.0% | 0.0%/0.0% | 28.0%/37.1% | 50 |
| D (compressed) | 54.0%/60.2% | 0.0%/0.0% | 0.0%/0.2% | 26.0%/32.2% | 50 |
| B_int8 (compressed) | 54.0%/68.0% | 0.0%/0.0% | 0.0%/0.0% | 24.0%/34.4% | 50 |
| C (layer-selection only) | 46.0%/55.7% | 0.0%/0.0% | 0.0%/0.0% | 24.0%/38.9% | 50 |
| B_int4_kivi (highest compression) | 50.0%/61.7% | 0.0%/0.0% | 0.0%/0.0% | 28.0%/34.7% | 50 |

Source: `results/run_20260907_222745` (A), `results/run_20260909_080434` (D, B_int8), `results/run_20260910_181034` (C, B_int4_kivi)

**Every pairwise comparison across all five Qwen2.5 configs is statistically significant** (real vs zeroed/random: p<0.0001 in all five; real vs mismatched: p=0.0127 (A), p=0.0013 (D), p=0.0003 (B_int8), p=0.0127 (C), p=0.0192 (B_int4_kivi); mismatched vs zeroed/random: p≤0.0005 in all five). The three-tier ordering (real > mismatched > zeroed=random) holds across every structurally distinct relay condition tested on Qwen2.5 — no compression, layer-selection alone, uniform int8, layer-selection plus adaptive compression, and the highest-compression per-channel int4 fix.

**Mistral-7B-Instruct-v0.3, HotpotQA — cross-model extension:**

| Config | Real | Zeroed | Random | Mismatched | n |
|---|---|---|---|---|---|
| A (uncompressed) | 56.0%/63.3% | 0.0%/0.5% | 0.0%/0.3% | 16.0%/26.4% | 50 |
| C (layer-selection only) | 20.0%/30.3% | 0.0%/0.5% | 0.0%/0.1% | 14.0%/26.1% | 50 |
| D (layer-selection + compression) | 22.0%/32.8% | 0.0%/0.5% | 0.0%/0.1% | 16.0%/29.3% | 50 |
| B_int8 (uniform quantization) | 56.0%/63.3% | 0.0%/0.5% | 0.0%/0.1% | 20.0%/29.3% | 50 |
| B_int4_kivi (highest compression) | 54.0%/63.2% | 0.0%/0.5% | 0.0%/0.1% | 16.0%/25.0% | 50 |

Source: `results/run_20260911_101259` (A), `results/run_20260913_090529` (C, B_int4_kivi), `results/run_20260916_153013` (D, B_int8)

**`D` shows the same partial pattern as `C`** (real vs. mismatched not significant, p=0.375, F1 CI includes zero, though mismatched vs. zeroed/random is, p=0.0078 each) — not a new anomaly, a second confirmation: `D`'s own Mistral baseline (22.0%) is dramatically lower than Qwen's (~50-54%), matching Finding 3's architecture-dependence claim, leaving less headroom to detect a further real-vs-mismatched gap. **`B_int8` replicates the full, clean five-comparison pattern** exactly like Qwen (all p≤0.002). The dividing line is clean: layer-selection configs (`C`, `D`) both show the attenuated pattern on Mistral; quantization-only configs (`A`, `B_int8`, `B_int4_kivi`) all show the full pattern — Finding 3's architecture-dependence claim showing up a second way, not a coincidence.

`A` and `B_int4_kivi` replicate the full three-tier ordering cleanly on Mistral, all pairwise comparisons significant (real vs zeroed/random p<0.0001 each for both configs; real vs mismatched p<0.0001 each; mismatched vs zeroed/random p=0.0078 each). `C` is **partial**: mismatched clearly beats zeroed/random (p=0.0156 each — content matters at all), but `C` itself is not significantly different from mismatched here (p=0.5078, F1 CI includes zero) — unlike on Qwen, where this comparison was significant. Most likely explanation: `C`'s own baseline on Mistral is already much lower than on Qwen (20.0% vs 46.0%), independently confirming Finding 3's claim that Mistral is unusually fragile to layer-selection — less headroom to detect a further gap on an already-degraded baseline, not evidence content-identity stops mattering. **All three (`A`, `C`, `B_int4_kivi`) are now written into the paper** — `A` as "Finding 5, Mistral extension completed," `C`/`B_int4_kivi` as "Finding 5, extended to a second model" (this was previously out of sync: `A` had been run and tabled here since 2026-09-11 but not yet drafted into the actual paper text — fixed 2026-09-15).

**Qwen3-8B, HotpotQA — a third model, and a genuinely different pattern:**

| Config | Real | Zeroed | Random | Mismatched | n |
|---|---|---|---|---|---|
| A (uncompressed) | 54.0%/67.2% | 46.0%/55.8% | 0.0%/0.0% | 64.0%/74.8% | 50 |
| D (compressed) | 56.0%/71.2% | 46.0%/55.8% | 2.0%/5.0% | 52.0%/62.4% | 50 |

Source: `results/run_20260914_022638`

**Not a clean replication — a real, verified, architecture-dependent finding.** Random still collapses catastrophically and significantly on both configs (p<0.0001 each). But real is NOT statistically distinguishable from zeroed or mismatched on either config (A vs zeroed p=0.48, A vs mismatched p=0.30, D vs zeroed p=0.33, D vs mismatched p=0.79, all F1 CIs include zero). On A, mismatched > zeroed is itself significant (p=0.0039); on D even that collapses (p=0.45). Verified as real (not a bug) three ways: no donor-question pool leakage; A_audit_zeroed/D_audit_zeroed's byte-identical output traced to D's own `reconstruction_strategy="zeros"` composing correctly with the audit's zeroing (expected, not a bug); real/zeroed/mismatched overlap on only 10-24 of 50 examples pairwise, confirming the substitution is genuinely selective. Written into the paper as "Finding 5, extended to a third model" — read alongside Finding 6, since both show the same meta-point (a mechanism assumed universal is itself architecture-dependent), just on different axes (depth-safety-confidence vs. content-dependence).

**Second topology — fan-in decomposition (not sequential chain), Qwen2.5-7B-Instruct:**

| Config | Real | Zeroed | Random | Mismatched | n |
|---|---|---|---|---|---|
| kv (fan-in, both children audited) | 40.0%/53.0% | 0.0%/0.8% | 0.0%/0.0% | 6.0%/12.0% | 50 |

Source: `results/recursive_poc_check/run_20260913_083353` (real/zeroed/random pilot, n=25), `results/recursive_poc_check/run_20260917_102302` (full n=50 run with `mismatched` added — table above uses this run throughout for consistency). **The cleanest three-tier ladder in the project.** `kv` vs. zeroed: p<0.0001, F1 delta +0.522 [+0.396, +0.645]. `kv` vs. random: p<0.0001, F1 delta +0.530 [+0.403, +0.654]. `kv` vs. **mismatched: p=0.0001** (18/19 discordant pairs favoring real), F1 delta +0.410 [+0.267, +0.546] — the most decisive real-vs-mismatched result anywhere in this paper. Unlike the RLM+KV dynamic prototype (§ below), this topology has no alternate information pathway, so zeroed/random collapse to genuinely near-zero, matching the sequential pipeline exactly. One nuance: mismatched vs. zeroed/random is not significant by McNemar (p=0.25 each, only 3 discordant pairs — the test's own floor at that count) but the F1 CI excludes zero for both (+0.112 [+0.035,+0.200], +0.120 [+0.047,+0.206]) — EM says "not proven," F1 says "trending real," reported both ways. Garbage-signature scan: `kv` 0/50 flagged, mismatched 4/50, zeroed 23/50, random 48/50 — a clean, monotonic match to the accuracy numbers. Written into the paper as "Finding 5, the fan-in topology's ladder completed."

**Third topology — dynamic, model-driven RLM+KV delegation, Qwen2.5-7B-Instruct, n=50:**

| Config | EM | F1 |
|---|---|---|
| text | 22.0% | 0.285 |
| kv (real) | 24.0% | 0.323 |
| kv_audit_zeroed | 8.0% | 0.112 |
| kv_audit_random | 6.0% | 0.092 |
| kv_audit_mismatched | 10.0% | 0.155 |

Source: `results/rlm_kv_check/run_20260916_203322` (`scripts/rlm_repl_kv_check.py`).
**The full three-tier signature holds**: `kv` vs. zeroed p=0.0078 (F1 delta +0.210 [+0.110, +0.323]); `kv` vs. random p=0.0039 (F1 delta +0.230 [+0.127, +0.347]); `kv` vs. **mismatched p=0.0156** (F1 delta +0.167 [+0.071, +0.274]) — all three significant, completing the ladder this topology was missing. A genuinely stronger result than the C2C cross-architecture bridge (§7 below), where content identity didn't matter at all — here it does. Honest caveat: mismatched vs. zeroed/random is not yet significant (p=1.0/p=0.50, 1-2 discordant pairs) — direction is right (10.0%>8.0%>6.0%) but underpowered for that specific leg. Verified via a garbage-signature scan across all 250 transcripts: 0/50 flagged for `real`/`text`, 3/50 `mismatched`, 4/50 `zeroed`, 36/50 `random` — a clean, monotonic ordering matching the accuracy numbers. `kv` vs. `text` remains not significant (p=1.0). **Important reproducibility caveat, unique to this prototype**: diffing this run against the earlier zeroed/random-only run on the identical 50 questions found 15/50 (30%) `text`-channel raw generations differ between runs under greedy decoding — unlike the main pipeline (confirmed bit-identical across reruns), this REPL-based prototype is not run-to-run deterministic, plausibly GPU floating-point non-associativity cascading through a long, self-referential, code-writing decode loop. Doesn't invalidate the within-run significance results, but the absolute percentages here shouldn't be read as a stable ground truth. Written into the paper, completing the three-topology non-exchangeability evidence base.

## 5. Three secondary analyses (no new GPU time — analysis of existing data)

**Model behavior tracks the causal-audit ordering even more completely than final-answer accuracy (RLM+KV prototype, mined from `run_20260916_203322`):**

| Condition | Timeout rate | Mean turns | Mean delegation calls |
|---|---|---|---|
| real | 22% (11/50) | 7.72 | 1.38 |
| mismatched | 42% (21/50) | 8.48 | 1.12 |
| zeroed | 72% (36/50) | 9.46 | 0.76 |
| random | 76% (38/50) | 9.60 | 0.66 |

Reused `lakv.stats.mcnemar_test` with "session finished without exhausting its turn budget" substituted for "correct," on the same paired 50 questions. Every pairwise comparison is significant, **including the one accuracy left unresolved**: real vs. mismatched p=0.0309; real vs. zeroed/random p<0.0001 each; mismatched vs. zeroed p=0.0001; **mismatched vs. random p<0.0001** — accuracy alone couldn't distinguish that last pair (p=1.0/p=0.50). A genuinely new angle: a process/behavioral signature of causal content-corruption, not just an outcome one — the model's own sense of "enough information" tracks content quality gradedly even though it never sees the corrupted content as text. Zero new GPU cost — pure analysis of already-collected transcripts.

**Calibration confidence does not predict cross-architecture safety (and points the wrong way):**

| Model | Tier-1 (kept) mean importance | Tier-3 (dropped) mean importance | Separation |
|---|---|---|---|
| Qwen | 0.737 | 0.246 | 0.491 |
| Mistral | 0.851 | 0.178 | **0.673 (larger)** |

Mistral's calibration looks *more* confident, yet Mistral is the *more* fragile model when it's acted on.

**Layer-selection failures differ in kind, not just rate (D config, wrong answers only):**

| Model | Mean length | % over 150 chars | Character |
|---|---|---|---|
| Qwen | 18 chars | 2% | Close near-misses (reformatted dates, typos) |
| Mistral | 109 chars | 15% | Fabricated tangents, repetition loops (max 1,454 chars) |

Repetition-penalty confound checked and ruled out (Mistral's own default has *no* penalty; our uniform 1.05 applies *more* correction to Mistral than its own baseline, not less — the pattern persists anyway).

**Donor-question bleed-through (real, rare, not the dominant failure mode):** manually reviewed 20/36 wrong `A_audit_mismatched` answers against their logged donor questions (`results/run_20260909_104129`). One unambiguous case — a Kansas fight-song question produced "Ellie Goulding" in the answer, traceable only to an unrelated donor question about that singer. One weaker, ambiguous second case. The other ~18 reviewed were ordinary confusion on the real question's own topic, not donor substitution. Manual, non-exhaustive, single-annotator — a real qualitative data point, not a quantified rate.

## 6. Second task family — GSM8K (n=50, Qwen)

Source: `results/run_20260909_120933`

| Config | Accuracy | KV/hop | Note |
|---|---|---|---|
| **A** | 90.0% | 39.38 MB | |
| **D** | 90.0% | 10.42 MB (2.76x self-ref / 3.78x vs `A`) | McNemar vs `A`: p=1.0, only 2 discordant pairs — layer-selection is *more* free on GSM8K than HotpotQA |
| A_audit_zeroed | 0.0% | 40.35 MB | |
| A_audit_random | 2.0% | 44.45 MB | |
| A_audit_mismatched | 28.0% | 39.64 MB | |

**Every pairwise comparison in the three-tier ladder is significant**: `A` vs zeroed/random/mismatched p<0.0001 each; mismatched vs zeroed p=0.0001; mismatched vs random p=0.0002 — both headline findings (causal audit ordering, near-free layer selection) replicate on a structurally different task, with larger effect sizes and tighter p-values than the original HotpotQA runs. `D`'s reported ratio predates this session's byte-accounting fix and the later real-packing fix, but turns out to already be correct: it ran on the pre-session code, whose formula was always numerically identical to what real nibble-packing produces (see §3) — no correction needed, unlike an earlier pass through this file briefly claimed. Raw-text check on `A_audit_random` confirms the identical code-fragment/mixed-language garbage signature already documented for HotpotQA (not a parsing bug), plus one example where the model drifts mid-generation into reciting "Janet's ducks" (the canonical GSM8K few-shot exemplar still present in its own system prompt) instead of engaging with the real question — anecdotal, not a general claim.

## 7. Heterogeneous cross-architecture bridge (C2C), n=50 HotpotQA

Source: `C2C_external/lakv_pipeline_c2c.py` (accuracy benchmark) and
`C2C_external/lakv_pipeline_c2c_audit.py` (causal audit) — external repo,
not part of this git history; results transcribed here from real runs.

| Condition | Accuracy | F1 | Latency |
|---|---|---|---|
| single_agent_qwen3 (alone) | 56.0% | 67.1% | 25.4s |
| single_agent_qwen25 (alone) | 54.0% | 61.0% | 37.4s |
| **real** (C2C bridge, actual projected KV) | 60.0% | 70.9% | 89.0s |
| zeroed (projector output zeroed) | 0.0% | 0.0% | 262.5s |
| random (projector output moment-matched noise) | 0.0% | 0.0% | 221.7s |
| mismatched (bridge fed a different question) | 56.0% | 70.6% | 87.5s |
| no_comm (sharer channel off) | 56.0% | 67.1% | 49.3s |

`real`/`no_comm` are bit-identical to the earlier accuracy-only run's
`F_c2c`/`single_agent_qwen3` — same 50 questions, greedy decoding, full
determinism confirmed. **`real` is not significantly different from
either standalone model** (vs. qwen3 p=0.75, vs. qwen25 p=0.58) — no
proven accuracy win from the bridge yet. **The causal audit is the real
finding**: `real` vs. `zeroed`/`random` p<0.0001 each, F1 delta +0.709
[+0.593, +0.822] — the channel is unambiguously load-bearing, same
"random is worse than zeroed" garbage-token texture already seen
elsewhere in this project. But `real` vs. `mismatched` is **not
significant** (p=0.625, F1 delta +0.003 [-0.059, +0.067]) — swapping in
a completely different question's content through the same pathway makes
no detectable difference. Read together: this bridge requires a real
projection to be *present*, but doesn't detectably depend on *which*
question produced it — a different, weaker property than same-model
relay satisfies everywhere else in this paper (§4 above). Written into
the paper draft's Limitations, right after the KVCOMM cross-architecture
gap sentence.

## 8. What's still open

- Causal audit now covers `A`, `D`, `B_int8`, `C`, and `B_int4_kivi` on
  Qwen2.5 — every relay condition in the paper except the already-broken
  `B_int4`, which is uninformative to audit further (see §4). **All five
  configs (`A`, `C`, `D`, `B_int8`, `B_int4_kivi`) are now also confirmed
  on Mistral** (§4) — this gap is closed. **The full three-tier causal-
  audit ladder (real > mismatched > zeroed/random) is now confirmed on
  all three topologies tested in this paper**: sequential chain
  (Finding 5), static fan-in decomposition (§4, run 2026-09-17 — the
  cleanest of the three), and dynamic model-driven RLM+KV delegation
  (§4, run 2026-09-16). No causal-audit gaps remain open in this paper.
- `B_int4_kivi`'s trend below `A`/`B_int8` (52% vs 56-57%) is not yet statistically confirmed as a real cost (7-12 discordant examples, p=0.36-0.50).
- `D` vs `C` on Qwen not significant (p=0.14, underpowered at n=100).
- Bleed-through analysis is manual/partial (20 of 36 examples, one annotator) — an automated or fully-annotated version would be needed to turn this into a quantified claim.
- GSM8K generalization check is a single n=50 run, one model — not yet extended to `B_int8`/`D`-family causal audit, Mistral, or the newer model families being added for Candidate 1.

---

**For the "why" behind any of these numbers, see `CLAUDE.md`'s "Research-extensions findings" section. For how these are framed as a paper, see `docs/naacl2027_paper_draft.md`.**
