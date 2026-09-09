# LAKV — Project Reference

## What this project is

LAKV studies whether relaying **KV cache** between agents in a multi-agent LLM
pipeline (Reasoner → Verifier → Finalizer, all Qwen2.5-7B-Instruct) beats
relaying **plain text** between the same agents — on both accuracy and latency.
The core pitch: KV-relay should avoid recomputing a shared prefix, so it
should be faster than re-processing text from scratch, while achieving
comparable accuracy. Layer selection and quantization are studied as a way to
shrink the relayed KV further, on top of that.

Primary dataset: **HotpotQA** (multi-hop QA, distractor config — each item
bundles its own 10 candidate passages, no retrieval needed). GSM8K was the
original dataset before an earlier pivot; some code paths (few-shot exemplars,
numeric-grounding checks) are GSM8K-specific and gated off for HotpotQA.

Active branch: `lakshya/hotpotqa`. Model: `Qwen/Qwen2.5-7B-Instruct`, bf16, on
a single RTX 4090.

**Publication push (branch `feat/research-extensions`, off `lakshya/hotpotqa`,
started 2026-09-07):** extends this project toward an ACL/NAACL (ARR)
submission — adds paired significance testing, a second model family
(Mistral-7B-Instruct-v0.3) to check generalization, a causal audit of
whether relayed KV carries real content, and a rotation-based (TurboQuant-
inspired) int4 quantization attempt. See "Research-extensions findings"
below for what's been statistically confirmed so far. Not yet merged to
`lakshya/hotpotqa` — if you're reading this on a different branch, that
section may not apply to what's currently checked out.

## Architecture

- `run.py` — CLI entry point (`--mode calibrate|sanity|experiment`).
- `lakv/pipeline.py` — `LAKVPipeline`, the 3-agent KV-relay pipeline. Also
  home to `PipelineConfig` (every tunable knob for a config).
- `lakv/evaluator.py` — `Evaluator` + `PRESETS` dict (the named configs below).
- `lakv/layer_selector.py` — drops low-importance layers per the calibration
  profile; reconstructs dropped layers via `zeros`/`nearest`/`interpolate`.
- `lakv/kv_compressor.py` — per-head min-max quantization (int8/int4/adaptive
  tier mix).
- `lakv/offset_corrector.py` + `lakv/anchor_table.py` — the KVCOMM-derived
  offset-correction mechanism (config E). See "E is a documented negative
  result" below before touching this.
- `lakv/calibration_profiler.py` — produces the per-layer tier-assignment
  profile (`profiles/<run>/qwen_<dataset>.json`) that layer selection and
  adaptive compression read.
- `lakv_v2/pipeline/single_agent.py`, `text_agent.py` — the two non-KV
  baselines. `text_agent` relays literal decoded text instead of KV cache,
  same role/prompt structure as config A, for an apples-to-apples comparison.
- `lakv/stats.py` (branch `feat/research-extensions`) — paired McNemar test +
  bootstrap F1 CI + Monte-Carlo power estimate for comparing two configs from
  the same `experiment_results.json` (or two different files, same example
  indices — used for old-vs-new / cross-model comparisons too). Run via
  `python -m lakv.stats <file> <cfgA> <cfgB> ...`.
- `lakv/causal_audit.py` (same branch) — substitutes the KV handed to the
  next agent with zeroed/moment-matched-random/held-out-mismatched content,
  to test whether a config's accuracy comes from the real relayed content or
  just from having *some* non-empty cache. Reachable via config names
  `<base>_audit_zeroed` / `_audit_random` / `_audit_mismatched`, base in
  `("A", "D", "B_int8")` — see `lakv/evaluator.py::AUDIT_MODE_SUFFIXES`.
- `lakv/kv_compressor.py`'s int4 variants for fixing `B_int4`'s collapse
  (same branch, chronological order tried): `uniform_int4_rotated` (Hadamard
  rotation of both K/V, TurboQuant/PolarQuant-inspired — **broken**, produces
  out-of-distribution tokens from disrupting K's RoPE encoding, preset
  `B_int4_turboquant`, kept only as a documented negative result);
  `uniform_int4_rotated_v_only` (isolating diagnostic, confirmed the K/RoPE
  diagnosis, preset `B_int4_turboquant_vonly`); **`uniform_int4_kivi_k_channel`
  (the actual fix — K quantized per-channel instead of per-head, KIVI-
  inspired, preset `B_int4_kivi`, use this one)**; `uniform_int4_hybrid`
  (KIVI's K fix + rotated V, preset `B_int4_hybrid` — tested equivalent to
  `B_int4_kivi` with no benefit, don't use, kept for the record). See
  "Research-extensions findings" below for the real-model results.

## Configs (`lakv/evaluator.py::PRESETS`)

| Config | Layer selection | Compression | Offset correction | Notes |
|---|---|---|---|---|
| `single_agent` | — | — | — | one shot, no relay at all |
| `text_agent` | — | — | — | relays decoded **text**, not KV |
| `A` | no | none | no | uncompressed KV relay, 28/28 layers |
| `B_int8` / `B_int4` | no | uniform int8 / int4 | no | compression only |
| `C` / `C_nearest` / `C_interpolate` | yes | none | no | layer selection only, varying reconstruction |
| `D` / `D_nearest` / `D_interpolate` | yes | adaptive (int8/int4 tier mix) | no | layer selection + compression |
| `E` / `E_int8` | yes | adaptive / uniform int8 | **yes** | adds anchor-based position correction on top of D/B_int8-equivalent settings |
| `E_strict`, `E_int8_strict`, `E_nodelta` | yes | (matches E) | yes | diagnostic-only ablations of E, not results to report |

All KV-relay configs (`A`–`E`) share the same Reasoner/Verifier/Finalizer
prompts as `text_agent`, so any accuracy/latency delta between a KV config and
`text_agent` is attributable to the relay mechanism, not prompt differences.

## Established results (n=100, HotpotQA, as of 2026-09-04)

| Config | Accuracy | F1 | Latency | KV/hop |
|---|---|---|---|---|
| single_agent | 57.0% | 68.4% | 1.5s | — |
| text_agent | 54.0% | 68.3% | 6.7s | — |
| **A** | 57.0% | 69.6% | 7.8s | 143.98 MB |
| **B_int8** | 56.0% | 68.6% | 8.0s | 72.07 MB (2.00x) |
| B_int4 | 0.0% | 0.0% | 35.0s | ~76.7 MB (2.00x, corrected — see below) — **collapses, see below** |
| C | 47.0% | 59.8% | 10.6s | 104.26 MB |
| C_nearest / C_interpolate | 33.0% / 36.0% | 48.3% / 44.7% | 9.2s / 8.5s | ~103 MB |
| **D** | 44.0% | 58.3% | 9.6s | ~52.0 MB (2.00x self-ref / ~2.77x vs `A`, corrected — see below) |
| D_nearest / D_interpolate | 34.0% / 31.0% | 47.2% / 47.0% | 8.8s / 8.7s | ~37.5 MB, uncorrected (not yet recomputed for this row) |
| E / E_int8 | 12.0% / 4.0% | 21.2% / 10.9% | 21.6s / 22.3s | ~36-49 MB, uncorrected (not yet recomputed for this row) |

**MB/ratio columns above predate the 2026-09-09 int4 byte-accounting fix
(see "Research-extensions findings" below) for every row involving int4 —
`B_int4`/`D`/`D_nearest`/`D_interpolate`/`E` are corrected or flagged as
not yet recomputed; `B_int8`/`A`/`C` were never affected. Accuracy/F1/
latency in this whole table are unaffected either way (compression byte
accounting is independent of generation).**

**Bottom line: `D` and `B_int8` are the strongest defensible KV-relay
results.** `A` matches `single_agent`'s accuracy and is close to (but ~15%
slower than) `text_agent`'s latency — the "avoid recomputation" win is real
but small, since decode time dominates total latency far more than prefill for
this task (hops run 100-500+ generated tokens). `D` trades ~13 accuracy points
for a ~2.77x smaller relayed cache (vs `A`, corrected); `B_int8` gets 2x compression for
essentially no accuracy cost.

## Research-extensions findings (branch `feat/research-extensions`, as of 2026-09-07)

The table above (2026-09-04) predates a real bug fix (`repetition_penalty`
was being force-disabled on both decode paths instead of applied
consistently — see git history on this branch) and predates any paired
significance testing. Refreshed n=100 HotpotQA/Qwen numbers, same profile,
current code: `single_agent` 57.0%/68.4%, `text_agent` 54.0%/68.3%, `A`
56.0%/69.1%, `B_int8` 57.0%/70.5%, `B_int4` 0.0%/0.0% (collapses as originally
coded — **fixed variant `B_int4_kivi` reaches 52.0%/64.1%, see finding 4/5
below, don't stop reading at this row**), `C` 43.0%/57.8%, `D` 50.0%/61.9%,
`E` 9.0%/17.4%, `E_int8` 1.0%/6.2%.

**Important honesty note:** D's move (44%→50%) and E's move (12%→9%) from
the repetition_penalty fix are NOT statistically significant against the old
numbers (paired McNemar, same 100 examples: D p=0.21, E p=0.58, E_int8
p=0.375 — see `lakv/stats.py`). The fix demonstrably changed *which*
examples came out right, but whether it's a net real improvement/regression
isn't established at this n. Don't cite "the fix improved D" as fact.

**Three claims that ARE statistically solid enough to build a paper around**
(see `lakv/stats.py` output for exact p-values/CIs; re-derive before citing,
don't just trust this summary if more data has come in since):

1. **Uniform 8-bit KV quantization is statistically indistinguishable from
   uncompressed relay.** Confirmed on both Qwen (`A` vs `B_int8`: p=1.0, only
   3 discordant examples out of 100) and Mistral-7B-Instruct-v0.3 (`A` vs
   `B_int8`: 0 discordant examples — literally identical per-example
   outcomes). Not an underpowered null result — the effect is tight enough
   at n=100 that this is a real near-zero-cost finding.
2. **The anchor-based offset-correction mechanism (config `E`) fails
   catastrophically and this is not explained by the repetition_penalty
   confound.** `D` vs `E`: p≈0.00003 (old file) and p≈0.00003 (new file,
   even after the fix) — E's raw hop_texts in the new run still show
   duplicated phrases and hallucinated finalizer content, same failure
   signature as before the fix. See "Known issues" below for the full bug
   history behind this conclusion.
3. **Layer-selection's accuracy cost is architecture-dependent, not a fixed
   property of "drop ~29% of layers."** On Mistral-7B-Instruct-v0.3 (n=50,
   `results/run_20260907_130646`), `A` vs `D` collapsed by 30.5 F1 points
   (p=0.0001, overwhelming) after dropping 9/32 layers (28%). On Qwen (n=100,
   this file), the same comparison isn't even statistically significant yet
   (`A` vs `D`: p=0.34) — but the claim survives regardless: the most
   generous reading of Qwen's own uncertainty (F1 CI upper bound +16.4
   points) is still less than half of Mistral's confirmed collapse.
   Supporting citation found in a literature search this session: "No Free
   Swap: Protocol-Dependent Layer Redundancy in Transformers" (arXiv
   2605.16234) finds layer redundancy is architecture/protocol-dependent
   across Qwen3-8B/Llama-3.1-8B/Mistral-7B — consistent with, not just
   coincidentally matching, this finding.
4. **`B_int4`'s total collapse (0.0%/0.0%) is a fixable RoPE-interaction bug
   in the quantization SCHEME, not an intrinsic 4-bit ceiling.** Root-caused
   this session, not just patched: K is cached post-RoPE, and RoPE mixes
   channel pairs by a position-dependent amount, which per-head (whole-range)
   min-max quantization is defenseless against at only 16 quantization
   levels. Confirmed two ways before trusting it: (a) a real-model test of
   naive Hadamard-rotation quantization (`B_int4_turboquant`, TurboQuant/
   PolarQuant-inspired) produced bizarre out-of-distribution tokens (literal
   `.HttpServletResponse`, random CJK characters) — a failure SHAPE
   inconsistent with ordinary quantization noise, consistent with the
   rotation scrambling RoPE's paired-dimension structure specifically; (b) an
   isolating diagnostic (`B_int4_turboquant_vonly`, K untouched, only V
   rotated) reverted to ORDINARY `B_int4`-style repetition-loop garbage
   (`"2 2 2 2"`, `"0 0 0 0"`) instead of the bizarre pattern — direct
   real-model confirmation that rotating K specifically was the problem, not
   int4 precision in general. Fix: `B_int4_kivi` (KIVI-inspired, Liu et al.
   ICML'24 — quantize K per-channel instead of per-head, V unchanged) went
   from 0.0%/0.0% to **52.0%/64.1% at n=100** (confirmed, real, trustworthy
   number — first seen at 30.0%/39.4% at n=10, which was correctly flagged
   as too small to trust; the n=100 run resolved it, no corruption at either
   scale). `B_int4_hybrid` (K per-channel + V also rotated) was built to test
   whether rotating V on top adds anything — it doesn't (n=10): 0 discordant
   pairs vs `B_int4_kivi` (p=1.0, identical per-example outcomes), F1
   slightly *lower* (37.5% vs 39.4%) from one example where the extra
   rotation produced a more verbose, lower-precision answer with no accuracy
   benefit. **Use `B_int4_kivi`, not `B_int4_hybrid` — simpler, same
   accuracy, better F1.** `B_int4_kivi_full` (K per-channel + V per-token,
   KIVI's other asymmetric half) is built and unit-tested but not yet run on
   GPU — open question whether V's own axis fix adds anything on top of K's.
5. **`B_int4_kivi` (52.0%/64.1%, n=100) is statistically indistinguishable
   from `D` (50.0%/61.9%)** — McNemar p=0.86, F1 delta +2.2 pts [-8.2,
   +12.8] — **and needs none of D's machinery** (no calibration profile, no
   layer-selection tiering, just a fixed quantization scheme). Also not
   significantly different from `A` (56.0%, p=0.50) or `B_int8` (57.0%,
   p=0.36) — but don't overclaim this as "free" the way `B_int8` vs `A` is:
   those comparisons had 0-3 discordant examples (an extremely tight match);
   `B_int4_kivi` vs `A`/`B_int8` has 7-12 discordant examples and point
   estimates that trend real (52% vs 56-57%) — the honest statement is "not
   yet distinguishable from a real cost at this n," not "no cost."
   **CORRECTED 2026-09-09 — the "compresses harder than `D`" half of this
   finding was WRONG, root cause found via `feat/lakv_nvidia`:**
   `kv_compressor.py::compress()` billed 4-bit layers at 0.5 bytes/element,
   but `_quantize()`/`_quantize_per_channel()`/`_quantize_per_token()` all
   store the result as `torch.uint8` regardless of `bits` — no nibble-
   packing exists anywhere in this file, so a 4-bit value has always taken
   the same one full byte an 8-bit value does. Fixed to bill
   `k_q.nbytes + v_q.nbytes` directly. Corrected numbers (exact, recomputed
   from stored per-hop shapes, no GPU rerun needed since byte accounting is
   independent of generation): `B_int4_kivi` is **~72.6 MB/hop (≈2.00x vs
   `A`)**, not the previously-reported 36.43 MB (3.99x). `D` is **~52.2
   MB/hop (≈2.00x self-referential to its own transmitted layers, ≈2.77x
   vs `A`)**, not the previously-reported 37.82 MB (2.76x) — `D`'s number
   happens to land close to its old (wrong) value because its real
   advantage was always the layers it drops entirely, a genuine saving the
   old bug didn't touch, coincidentally offsetting the int4-tier
   undercounting. **Net effect: `D` now produces a ~39% SMALLER cache than
   `B_int4_kivi` (52.2 MB vs 72.6 MB) — the exact opposite of what was
   previously claimed.** `B_int4_kivi`'s real, defensible selling point is
   "matches `D`'s accuracy with `B_int8`-level compression and no
   calibration profile," not "higher compression than `D`." See
   `scripts/correct_compression_ratios.py` for the exact recomputation
   method (pure architecture/shape math — compressed size never depended
   on tensor values, so no rerun was needed to get exact, not approximate,
   corrected numbers). **`B_int4_kivi_full` (adds per-token V quantization
   on top of K's fix) does NOT improve on `B_int4_kivi`** — tested at n=50
   (`results/run_20260907_203253`), same idx range as a matched subset of
   `B_int4_kivi`'s n=100 run: 44.0% vs 50.0%, McNemar p=0.51 (not
   significant, but trending toward plain `B_int4_kivi` being better, not
   worse). Its compression is also still WORSE after correction (≈1.94x vs
   `B_int4_kivi`'s ≈2.00x, corrected from the old 3.75x-vs-3.99x framing)
   for a real, mechanical reason unrelated to the packing bug: per-token
   quantization stores a scale/zero-point pair per sequence position, and
   sequence length (hundreds to low thousands of tokens) vastly exceeds K's
   per-channel grouping (128 channels) — real overhead, no accuracy
   benefit. Combined with `B_int4_hybrid`'s earlier null result (rotating V
   didn't help either), this is now TWO independent, differently-motivated
   attempts to improve V that both failed — strong, replicated evidence
   that K's per-channel fix alone accounts for the whole recovery, not a
   coincidence. `B_int4_kivi` (plain, K-only) remains the config to report
   and use — just not with the "beats `D`'s compression" claim.**

6. **Causal audit: config `A`'s relayed KV demonstrably carries real,
   specific content — not just a generic non-empty cache.** Ran on `A`
   (uncompressed relay), n=50, HotpotQA: `A` 50.0%/64.0%, `A_audit_zeroed`
   0.0%/0.0%, `A_audit_random` (moment-matched noise) 0.0%/0.0%,
   `A_audit_mismatched` (a real, different held-out question's KV) 28.0%/
   37.1%. **Every pairwise comparison in the three-tier ladder is
   statistically significant**: `A` vs zeroed p<0.0001, `A` vs random
   p<0.0001, `A` vs mismatched p=0.0127 (borderline at an earlier n=20 pilot,
   confirmed here), mismatched vs zeroed/random p=0.0001 each. This is the
   mechanistic validation underneath every other finding in this document —
   direct proof that accuracy comes from the specific transmitted content,
   not merely from the receiving agent having *some* cache to attend over.
   Qualitative texture confirms it: `A_audit_mismatched`'s wrong answers are
   coherent, grammatical, plausible-sounding (real structure, wrong
   question); `A_audit_zeroed`'s are garbled English; `A_audit_random`'s are
   pure noise (code fragments, mixed-language garbage) — **worse than
   zeroed**, plausibly because a zero vector is "quiet" to attention
   (weak key signal, easy to down-weight) while realistic-magnitude random
   noise looks like real signal and actively misdirects attention, a small
   but genuine and explainable mechanistic aside worth a sentence in the
   paper. **CONFIRMED, extended to `D`/`B_int8` (n=50 each,
   `results/run_20260909_080434`): the exact same three-tier ordering holds
   under compression, and MORE decisively than on uncompressed `A`.**
   `D`: 54.0%/60.2% vs `D_audit_zeroed`/`D_audit_random` 0.0%/0.0%,0.2% vs
   `D_audit_mismatched` 26.0%/32.2%. `B_int8`: 54.0%/68.0% vs zeroed/random
   0.0%/0.0% vs mismatched 24.0%/34.4%. **All 10 pairwise comparisons across
   both configs are significant** — real vs zeroed/random p<0.0001 each
   (both configs); real vs mismatched p=0.0013 (`D`), p=0.0003 (`B_int8`) —
   tighter than `A`'s own p=0.0127; mismatched vs zeroed/random p=0.0002-
   0.0005. Raw text confirms the same failure signatures as `A`: zeroed →
   garbled English, random → multilingual/code garbage (worse than zeroed),
   mismatched → coherent, plausible, wrong. **This is the decision-gate
   result for the non-exchangeability framing: content-identity
   non-exchangeability holds under compression, not just uncompressed
   relay — the framing does NOT need to narrow.**
7. **A calibration signal's own confidence does not predict downstream
   layer-selection safety — and points the wrong way.** Tested directly
   (not assumed): mean separation between tier-1/2 (kept) and tier-3
   (dropped) importance scores is 0.491 for Qwen, 0.673 for Mistral —
   Mistral's calibration looks *more* decisive, yet Mistral is the model
   that collapses harder when that ranking is acted on (finding 3). Rules
   out "noisy calibration signal" as the explanation for the cross-model
   gap; supports the sharper reading that a technique's internal confidence
   is not evidence of safety. Single comparison, two models — a real,
   falsified hypothesis test, not a validated general predictor.
8. **Layer-selection failures differ in KIND across architectures, not just
   rate.** Qwen's `D` wrong answers: mean 18 chars, 2% exceed 150 chars,
   overwhelmingly close near-misses (reformatted dates, dropped honorifics,
   single-character typos). Mistral's: mean 109 chars, 15% exceed 150 chars,
   max 1,454 — includes fluent fabricated tangents (a full invented
   biography of an unrelated person) and degenerate repetition loops, which
   Qwen's failure set essentially doesn't show. Checked the obvious
   confound before trusting this: is a uniform `repetition_penalty=1.05`
   under-penalizing Mistral relative to its own tuned defaults? Checked
   both models' real HuggingFace `generation_config.json` — Qwen's own
   default *is* 1.05 (exact match); Mistral's specifies none at all,
   meaning our setting applies *more* correction for Mistral than its own
   baseline calls for, the opposite of what the confound would need. The
   long-tail failure pattern persists anyway — weakens rather than
   supports a decoding-hyperparameter explanation.

**Not yet statistically established, don't overclaim these:** `D` vs `C` on
Qwen (p=0.14, only 32% power at n=100 — would need ~n=300 for 82% power);
`A` vs `D` on Qwen alone without the cross-model framing (p=0.34, 15%
power); whether `B_int4_kivi`'s trend below `A`/`B_int8` is a real cost or
just underpowered noise (see finding 5 above).

9. **Content bleed-through from the donor question is real but rare, not
   the dominant failure mode.** Ran `A_audit_mismatched` again
   (`results/run_20260909_104129`, n=50 — bit-identical accuracy/F1 to the
   earlier run, as expected: this pipeline is fully deterministic under
   greedy decoding plus a seeded donor-selection RNG) specifically to get
   `donor_question` logged, then manually compared 20 of 36 wrong answers
   against both the real question and the logged donor question(s). Found
   one unambiguous case: real question about a Kansas university fight
   song, donor question about the singer Ellie Goulding, wrong prediction
   literally contains "Ellie Goulding" — an entity with zero connection to
   the real question, traceable only to the donor. One weaker, more
   ambiguous second case (a sports-figure name appearing near a
   racing-themed donor). **The other ~18 examples reviewed show ordinary
   confusion on the REAL question's own topic — near-misses, flipped
   comparisons, generic hallucination — not donor-topic substitution.**
   Honest read: bleed-through is a real, demonstrable phenomenon (this
   isn't a null result), but it is not what's driving most of
   `A_audit_mismatched`'s accuracy loss in this sample — that's still
   better explained by the receiving agent working from wrong/irrelevant
   real content in general, only occasionally manifesting as a literal
   named-entity leak. This was a manual review of a subset (20/36), not an
   automated, exhaustive, or statistically tested classification — treat as
   a qualitative, exploratory finding, not a quantified rate.

**Still open / in progress on this branch:** Orthogonal Backfill was
deliberately scoped out (not attempted) rather than left open — see
`docs/naacl2027_paper_draft.md`'s Limitations section for the reasoning
(its real formula needs attention weights this pipeline's fast `sdpa`
decode path doesn't expose, and shipping an approximated version under time
pressure was judged not worth the risk given this project's own evidence
that unfaithful reproductions can actively mislead). Causal audit not yet
run on `C` or the `B_int4` family.

## Known issues / settled questions (read before re-investigating)

- **Reconstruction strategy: `zeros` beats `nearest`/`interpolate`, by a lot**
  (10-16 points, confirmed at n=100 across both C and D families). This is
  the opposite of the intuitive guess. Likely reason: different transformer
  layers have distinct, non-interchangeable K/V projection weights, so a
  neighboring layer's real content is *confidently wrong* signal, not a
  helpful approximation — zeros just gets down-weighted by attention instead.
  **Don't switch `D` off `zeros` without new evidence.**
- **`E`/`E_int8` (offset correction) is a documented negative result**, not
  an open bug to keep chasing. Four real, structurally distinct bugs were
  found and fixed this session (in order): (1) `offset_corrector.py` matched
  corrected layers by list position instead of real layer index once layer
  selection drops any layer, scrambling which attention layer received which
  cache; (2) the correction discarded the real transmitted KV and substituted
  reconstructed content instead (first from a different anchor question, then
  from the query's own recomputed no-prefix KV) — the actual KVCOMM reference
  design adds the delta *on top of* the real relayed content, never replaces
  it; (3) delta was combined with the real content at the wrong alignment
  point; (4) `AnchorTable.update()`'s `prompt_seq_len` parameter — which
  marks where an agent's reasoning begins — was never passed at the call
  site, so the delta computation compared misaligned windows (offset by
  however many reasoning tokens were generated). After all four fixes, `E`
  still only reaches 12.0%/21.2% F1 at n=100, well below `D`. Read as: the
  anchor-delta-transfer technique (adapted from
  [KVCOMM](https://github.com/FastMAS/KVCOMM), NeurIPS'25) does not reliably
  transfer across semantically unrelated HotpotQA questions, even correctly
  implemented. It may have been more viable on the original GSM8K target
  (more homogeneous question structure) — never tested.
- **`B_int4` (uniform 4-bit, no layer-selection tiering) collapses to pure
  noise output — RESOLVED on `feat/research-extensions`, root cause was NOT
  "4 bits is just too lossy."** The original hypothesis in this section
  (forcing every layer to 4 bits with no int8 safety net) turned out to be
  wrong, or at least not the dominant factor — the real cause is that K is
  cached post-RoPE, and per-head (whole-range) min-max quantization at only
  16 levels is defenseless against RoPE's position-dependent channel mixing.
  Use `B_int4_kivi` (K quantized per-channel instead) — see "Research-
  extensions findings" below for the full diagnosis and real-model numbers
  (0.0%/0.0% → 30.0%/39.4% at n=10, needs a larger run to confirm the exact
  figure but the qualitative fix — coherent output instead of corrupted
  garbage — is solid).
- **`A` vs `text_agent` latency gap** (~7.8s vs ~6.7s) is real but doesn't
  close further with the current decode-loop optimization
  (`position_offset == 0` hops fast-path to `model.generate()` instead of a
  manual per-token loop) — decode time dominates total latency on this task,
  so avoiding prefill recomputation only ever buys back a small fraction.
- **Attention backend**: `sdpa` for sanity/experiment modes,`eager` only for
  calibration (needs `output_attentions`, which `sdpa` doesn't return).
- **Single-process pipeline**: nothing ever actually serializes/transmits a
  `KVMessage` over a wire — compression/reconstruction happen in-memory in one
  Python process. Byte/compression-ratio accounting is still meaningful (it's
  what a real distributed deployment would need to send), but don't expect
  smaller-KV configs to be *faster* in this single-GPU setup — layer
  selection/compression save bandwidth, not compute (the model still runs all
  28 transformer layers regardless of what's in their injected cache).

## Workflow conventions

- **The user runs all GPU/model/long-running commands themselves** — hand
  back the exact command, don't execute training/eval/calibration runs via
  Bash. Fine to run fast, read-only, CPU-only checks (compiling, importing,
  CPU dry-run tests with dummy tensors, inspecting result JSON) directly.
- Verify file paths (profile paths, result dirs) against the actual
  filesystem before handing back a command — timestamps get mistyped easily.
- `run.py --mode experiment` saves incrementally
  (`experiment_results.partial.json`) and supports `--resume_dir` — safe to
  leave long runs unattended.
- Before trusting any new accuracy/latency number from a code change, check
  the raw `hop_texts`/`raw_answer` for corruption artifacts (repetition
  loops, hitting the token cap) rather than reading the summary table alone —
  this session found multiple bugs that would have been missed by only
  looking at the top-line numbers.
