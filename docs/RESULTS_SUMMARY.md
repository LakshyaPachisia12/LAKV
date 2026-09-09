# LAKV — Results Summary (quick reference)

> Purpose: a scannable table of every confirmed number, distinct from
> `CLAUDE.md` (dense technical narrative, read that for the *why*) and
> `docs/naacl2027_paper_draft.md` (academic prose for the actual paper).
> Last updated 2026-09-09, branch `feat/research-extensions`. Every number
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
| B_int4 (broken) | 0.0% | 0.0% | 39.4s | ~76.8 MB (2.00x, corrected) |
| C | 43.0% | 57.8% | 10.0s | 104.16 MB |
| **D** | 50.0% | 61.9% | 10.2s | ~52.2 MB (2.00x self-ref / 2.77x vs `A`, corrected) |
| E | 9.0% | 17.4% | 19.8s | ~49.1 MB (2.00x, corrected) |
| E_int8 | 1.0% | 6.2% | 22.2s | 49.05 MB |

**MB/ratio columns for `B_int4`/`D`/`E` corrected 2026-09-09** — see §3 note
below for the bug and exact recomputation method. `A`/`B_int8`/`C`/`E_int8`
were never affected (no int4 tier involved). Accuracy/F1/latency unaffected
either way.

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

**B_int4_kivi vs D: statistically indistinguishable (p=0.86), no calibration
profile needed.** **CORRECTED 2026-09-09:** the "higher compression" half
of this claim was wrong — `kv_compressor.py` billed 4-bit tensors at half
the real byte cost (they're stored as `torch.uint8` with no nibble-packing,
same as 8-bit). Fixed; exact recomputation (pure shape/architecture math,
no GPU rerun needed) gives `B_int4_kivi` ≈72.6 MB/hop (≈2.00x vs `A`) and
`D` ≈52.2 MB/hop (≈2.00x self-ref / ≈2.77x vs `A`) — **`D` is now ~39%
smaller than `B_int4_kivi`, the opposite of what was previously reported.**
`B_int4_kivi`'s real, defensible claim: matches `D`'s accuracy at
`B_int8`-level compression with no calibration profile — not "beats `D`'s
compression." See `scripts/correct_compression_ratios.py`.

## 4. Causal audit — the central mechanistic result

| Config | Real | Zeroed | Random | Mismatched | n |
|---|---|---|---|---|---|
| A (uncompressed) | 50.0%/64.0% | 0.0%/0.0% | 0.0%/0.0% | 28.0%/37.1% | 50 |
| D (compressed) | 54.0%/60.2% | 0.0%/0.0% | 0.0%/0.2% | 26.0%/32.2% | 50 |
| B_int8 (compressed) | 54.0%/68.0% | 0.0%/0.0% | 0.0%/0.0% | 24.0%/34.4% | 50 |

Source: `results/run_20260907_222745` (A), `results/run_20260909_080434` (D, B_int8)

**Every pairwise comparison across all three configs is statistically significant** (real vs zeroed/random: p<0.0001 in all three; real vs mismatched: p=0.0127 (A), p=0.0013 (D), p=0.0003 (B_int8); mismatched vs zeroed/random: p≤0.0005 in all three). The three-tier ordering (real > mismatched > zeroed=random) holds regardless of compression — this is the mechanistic proof underneath every other claim in this project.

## 5. Two secondary analyses (no new GPU time — analysis of existing data)

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
| **D** | 90.0% | ~14.37 MB (~2.00x self-ref / ~2.74x vs `A`, corrected) | McNemar vs `A`: p=1.0, only 2 discordant pairs — layer-selection is *more* free on GSM8K than HotpotQA |
| A_audit_zeroed | 0.0% | 40.35 MB | |
| A_audit_random | 2.0% | 44.45 MB | |
| A_audit_mismatched | 28.0% | 39.64 MB | |

**Every pairwise comparison in the three-tier ladder is significant**: `A` vs zeroed/random/mismatched p<0.0001 each; mismatched vs zeroed p=0.0001; mismatched vs random p=0.0002 — both headline findings (causal audit ordering, near-free layer selection) replicate on a structurally different task, with larger effect sizes and tighter p-values than the original HotpotQA runs. `D`'s reported ratio in the raw run predates the 2026-09-09 int4 byte-accounting fix (run started 12:09, fix landed 12:40) — corrected above. Raw-text check on `A_audit_random` confirms the identical code-fragment/mixed-language garbage signature already documented for HotpotQA (not a parsing bug), plus one example where the model drifts mid-generation into reciting "Janet's ducks" (the canonical GSM8K few-shot exemplar still present in its own system prompt) instead of engaging with the real question — anecdotal, not a general claim.

## 7. What's still open

- Causal audit not yet run on `C` or the `B_int4` family.
- `B_int4_kivi`'s trend below `A`/`B_int8` (52% vs 56-57%) is not yet statistically confirmed as a real cost (7-12 discordant examples, p=0.36-0.50).
- `D` vs `C` on Qwen not significant (p=0.14, underpowered at n=100).
- Bleed-through analysis is manual/partial (20 of 36 examples, one annotator) — an automated or fully-annotated version would be needed to turn this into a quantified claim.
- GSM8K generalization check is a single n=50 run, one model — not yet extended to `B_int8`/`D`-family causal audit, Mistral, or the newer model families being added for Candidate 1.

---

**For the "why" behind any of these numbers, see `CLAUDE.md`'s "Research-extensions findings" section. For how these are framed as a paper, see `docs/naacl2027_paper_draft.md`.**
