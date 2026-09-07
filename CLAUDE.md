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
| B_int4 | 0.0% | 0.0% | 35.0s | 38.34 MB (4.00x) — **collapses, see below** |
| C | 47.0% | 59.8% | 10.6s | 104.26 MB |
| C_nearest / C_interpolate | 33.0% / 36.0% | 48.3% / 44.7% | 9.2s / 8.5s | ~103 MB |
| **D** | 44.0% | 58.3% | 9.6s | 37.69 MB (2.76x) |
| D_nearest / D_interpolate | 34.0% / 31.0% | 47.2% / 47.0% | 8.8s / 8.7s | ~37.5 MB |
| E / E_int8 | 12.0% / 4.0% | 21.2% / 10.9% | 21.6s / 22.3s | ~36-49 MB |

**Bottom line: `D` and `B_int8` are the strongest defensible KV-relay
results.** `A` matches `single_agent`'s accuracy and is close to (but ~15%
slower than) `text_agent`'s latency — the "avoid recomputation" win is real
but small, since decode time dominates total latency far more than prefill for
this task (hops run 100-500+ generated tokens). `D` trades ~13 accuracy points
for a 2.76x smaller relayed cache; `B_int8` gets 2x compression for
essentially no accuracy cost.

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
  noise output.** Checked the quantize/dequantize math — it's shared,
  bit-width-generic code that works fine for `D`'s adaptive int4 subset, so
  probably not a code bug. Most likely explanation: forcing *every* layer
  (including whichever ones calibration flagged as too important for
  aggressive compression) down to 4 bits, with no int8 safety net for
  sensitive layers, is simply too lossy. Not independently confirmed via the
  calibration profile's tier distribution — worth checking if this needs a
  firmer answer.
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
