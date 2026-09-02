# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

LAKV ("Layer-Adaptive KV-cache relay") is a research codebase studying multi-agent
LLM reasoning where agents relay **KV-cache state** (instead of decoded text)
between hops. The thesis: relaying compressed/selected KV cache across
cooperating agents preserves more reasoning fidelity than relaying text, up to
some compression ratio. Model under test: `Qwen/Qwen2.5-7B-Instruct`. Benchmark:
GSM8K (grade-school math word problems), loaded via `datasets.load_dataset("gsm8k", "main")`.

Single GPU only (no sharding), bf16, `attn_implementation="eager"` everywhere
(required because calibration needs `output_attentions=True`, which sdpa/flash
don't support).

For paper-style claims and experiment numbering (EXP-1..N, target accuracies),
see `RESEARCH_PLAN.md`. For step-by-step run commands, see `README.md` — it's
kept current and this file doesn't duplicate it.

## Repo layout — two pipeline generations coexist

- **v1** (repo root): `pipeline.py` (`LAKVPipeline`), `evaluator.py`, `run.py`,
  `run_best.py`, `calibration_profiler.py`, `layer_selector.py`,
  `kv_compressor.py`, `offset_corrector.py`, `anchor_table.py`. A
  heterogeneous 3-agent relay: Reasoner → Verifier → Aggregator, each with a
  distinct prompt, KV passed hop-to-hop through layer selection + quantization
  + offset correction.
- **v2** (`lakv_v2/`): a 2-agent shared-prefix redesign — Solver → Finalizer —
  reusing v1's compression/selection/anchor-table modules. Key files:
  `lakv_v2/benchmark.py` (5-phase runner), `lakv_v2/eval_runner.py`,
  `lakv_v2/sanity_test.py`, `lakv_v2/pipeline/{shared_prefix_pipeline.py,
  single_agent.py, two_agent.py, text_agent.py}`,
  `lakv_v2/cache/{aligner.py, compressor.py, reconstruct.py, selector.py}`,
  `lakv_v2/config/prompts.yaml`, `lakv_v2/tests/validation_suite.py`.

When making a change, check whether it needs to land in both generations —
they duplicate concepts (KV injection, anchor table, offset correction) but
are separate code paths. `lakv_v2/pipeline/shared_prefix_pipeline.py`'s decode
loops are still hardcoded greedy and have **not** received the sampling fix
that landed in v1's `pipeline.py` — don't assume a v1 fix propagated to v2.

`LAKV_V2_RUN_GUIDE.md` is an older/shorter v2 guide that may have drifted from
`README.md` (e.g. config naming like `old_lakv_A` vs README's `v2_full_prefix`)
— prefer `README.md` when the two disagree, and flag the drift if it matters
to the task at hand.

Not on this branch: HotpotQA support (lives on unmerged branch
`lakshya/hotpotqa`, which also reorganizes the codebase into folders). GSM8K
is the only dataset here.

## Core mechanism

**AnchorTable** (`anchor_table.py`, class `AnchorTable`) is a shared,
cross-agent pool of KV *deltas* for the shared question content: `delta =
actual_kv - base_kv`, where `base_kv` is the KV of the bare question with no
system prefix. One agent's observed correction can be reused by another agent
for the same question instead of recomputing/retransmitting full KV.

- `update()` stores per-`(question_key, agent_id)` offsets, aligning/de-rotating
  via `_rope_shift_k` before diffing. New anchors are admitted only when they
  pass both a length check and a similarity-entropy check (`_should_add_new_anchor`,
  implementing the paper's Eq. 5).
- `query_correction()` retrieves a similarity-weighted blend of candidate
  deltas, computes a confidence from blend entropy, and re-rotates the result
  forward via `_rope_shift_k` to the receiving agent's actual position.
- `graceful_degradation` (default on): ambiguous/low-confidence matches used
  to return `None`, forcing callers to fall back to *raw, uncorrected* KV
  (wrong RoPE position — this collapsed Config E's accuracy to ~4% before the
  fix). Now a best-effort blended correction is applied instead of bailing out.

**Manual KV-injection decode loop** (`pipeline.py`, `LAKVPipeline._generate`):
because injecting a foreign `past_key_values` cache means `model.generate()`
can't be used, generation is hand-rolled. `position_ids` for injected tokens
start at `cache_seq_len + position_offset` (not `receiver_prompt_len`) to
avoid RoPE-position collisions. `_sample_next_token` implements real
temperature/top-p sampling (mirroring what `generate()` would do) rather than
hardcoded argmax — falls back to argmax only when `do_sample=False`.
`lakv_v2/pipeline/shared_prefix_pipeline.py` has an equivalent but *still-greedy*
decode loop (see caveat above).

**RoPE re-rotation** (`anchor_table.py::_rope_shift_k`): shifts a key
tensor's rotary angle by a constant `shift * inv_freq(theta)`. Used to (a)
de-rotate observed KV back to base position before diffing in `update()`, and
(b) re-rotate a corrected key forward to the receiver's position in
`query_correction()`. A prior bug: `OffsetCorrector.correct()` failed to
forward `target_prompt_len` into `query_correction`, so re-rotation silently
never ran (root cause of the Config E collapse mentioned above). Also watch
for `rope_theta` living under `config.rope_parameters` on newer `transformers`
versions with Qwen2Config — use the `_get_rope_theta()` helper, don't assume
a flat attribute.

**EOS handling**: manual decode loops don't get `model.generate()`'s
automatic multi-EOS handling, so `pipeline.py::_get_stop_token_ids()` collects
both `model.generation_config.eos_token_id` (a list, includes Qwen's
`<|im_end|>`) and `tokenizer.eos_token_id`. A loop that checks only the latter
will run past end-of-turn and hallucinate a new chat turn on almost every hop
— this was a real bug, not a hypothetical.

## Agent types (what to compare against what)

- `single_agent.py` (`SingleAgentPipeline`, v2): one plain `model.generate()`
  call, no relay — the accuracy ceiling baseline.
- `text_agent.py` (`TextAgentPipeline`, v2): same 3-agent roles/prompts as
  KV-relay Config A, but agents pass **decoded text** forward and each calls a
  fresh `model.generate()` — no KV injection, no AnchorTable. This is the
  control that tests LAKV's actual thesis (KV relay vs. just relaying text).
  As of the last check, KV-relay underperformed this baseline pre the
  sampling fix — re-verify current numbers before citing results.
- `two_agent.py` / `pipeline.py` (`LAKVPipeline`) / `shared_prefix_pipeline.py`:
  the real KV-relay pipelines — inject `past_key_values` across agents through
  the mechanism above.

## Running experiments

Entry points: `run.py --mode calibrate|sanity|experiment` (v1), `run_best.py`
(v1, narrower A/C/D runner), `lakv_v2/sanity_test.py`, `lakv_v2/benchmark.py
--phase 1..5|all|v1` (v2). No `config.py`/`config.yaml` for model params beyond
`lakv_v2/config/prompts.yaml` (system prompts) — everything else is a CLI flag
(`--model`, `--task`, `--n_samples`, `--n_calibration`, `--configs`/`--rows`,
`--profile_path`, `--device`, `--arch`, `--use_offset_correction`).

`evaluator.py`'s `PRESETS`/`ABLATION_CONFIGS` define the configs referenced
throughout (A, B_int8, B_int4, C_nearest/interpolate, D_nearest, E, E_int8,
single_agent, text_agent, plus random/top/bottom-N ablations). Calibration
writes `LayerProfile` JSON (Tier 1/2/3 per layer) to `profiles/` (git-ignored);
results land in `results/run_<timestamp>/{experiment_results.json,
per_hop_stats.json, results_table.csv}` (git-ignored).

## GPU / memory constraints — read before changing generation budgets

Target hardware: ~14.56GB GPU, near-zero headroom. Eager attention is
O(seq_len²) and OOMs on long cumulative KV-relay context. Concretely:
- `final_max_new_tokens` was cut from 1536 → 512 in `PipelineConfig`
  (`pipeline.py`) because the final hop's manual decode loop ignores
  `generation_kwargs` entirely — a bigger budget only grew memory/latency with
  zero accuracy benefit, and 1536 OOM'd Config A after 4 samples.
- `evaluator.py::run_experiment` wraps each sample's `pipe.run()` in
  `try/except (torch.OutOfMemoryError, RuntimeError containing "out of
  memory")`, calls `torch.cuda.empty_cache()`, and records
  `oom_skipped=True` for that sample rather than losing the whole config's
  run. Don't remove this — Config A OOM'd 4/25 samples before it existed.
- Before proposing to raise any `max_new_tokens`, batch size, or n_samples for
  a config that runs the manual decode loop, check whether the increase
  actually changes generation behavior (many are ignored by the manual loop)
  and whether it fits the ~14.56GB budget.
- Prefer cheap, targeted reruns (single config, small `n_samples`) over full
  sweeps when validating a fix — full runs are expensive on this hardware and
  full re-verification isn't usually necessary to confirm a bug is fixed.

## Known non-bugs

Low GSM8K digit-retrieval accuracy in some configs traces to a genuine model
reasoning weakness (Qwen2.5-7B struggling to retrieve/track specific digits
through relayed context), not a precision or pipeline bug. Don't spend time
re-chasing this as if it were a code defect unless new evidence points
elsewhere.

## Testing

- `test_kv_compressor_clipping.py` (repo root): real pytest test, CPU-only,
  synthetic tensors — tests INT4 outlier-aware clipping in `kv_compressor.py`.
- `lakv_v2/tests/validation_suite.py`: **not** a pytest suite — a manual
  `ValidationSuite` class needing GPU, checking cache-handoff parity,
  quantization roundtrip, and layer-drop sensitivity. Run it explicitly, it
  won't show up in `pytest` collection.
- `lakv_v2/sanity_test.py`, `lakv_v2/diagnostic_test.py`,
  `scripts/diagnostics/*.py`: manual smoke-test/diagnostic scripts, not
  automated tests.

## Working conventions for this repo

- Verify claims (accuracy numbers, "the bug is fixed", config behavior)
  against a real run or the current code before stating them externally —
  don't extrapolate from past results or memory.
- Keep git history clean; this repo has a track record of small, well-scoped
  fix commits (one bug per commit) — follow that pattern rather than bundling
  unrelated changes.
- If a task involves the Kaggle-hosted training/eval environment, confirm
  Kaggle actually has the latest local code before assuming a fix is live
  there — local commits don't auto-sync.
- Before proposing a rerun, prefer the smallest sufficient one (single config,
  few samples) over a full sweep, given the GPU cost/time constraints above.
