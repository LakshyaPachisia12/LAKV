# LAKV — Layer-Adaptive KV-cache relay for multi-agent LLM reasoning

LAKV studies how much of a language model's KV cache you can compress, select, and
relay between cooperating agents on a math-reasoning task (GSM8K) before accuracy
degrades — and whether a learned "anchor table" of KV corrections can recover some
of that lost accuracy. The repo contains **two pipeline generations**:

- **v1** (root-level files: `pipeline.py`, `evaluator.py`, `run.py`, …) — a
  heterogeneous **3-agent relay** (Reasoner → Verifier → Aggregator by default),
  each with its own system prompt, KV cache passed hop-to-hop through layer
  selection + quantization + anchor-table offset correction.
- **v2** (`lakv_v2/`) — a **2-agent shared-prefix design** (Solver → Finalizer)
  where both agents start from a literal shared prompt prefix, eliminating the
  inter-agent positional-offset problem v1 has to correct for, while reusing the
  same compression/selection/anchor-table modules.

Everything runs against **Qwen2.5-7B-Instruct** on GSM8K by default and needs a
CUDA GPU for any real (non-trivial) run — see [Hardware requirements](#hardware-requirements).

---

## Table of contents

- [LAKV — Layer-Adaptive KV-cache relay for multi-agent LLM reasoning](#lakv--layer-adaptive-kv-cache-relay-for-multi-agent-llm-reasoning)
  - [Table of contents](#table-of-contents)
  - [Repo layout](#repo-layout)
  - [Environment setup](#environment-setup)
  - [Hardware requirements](#hardware-requirements)
  - [Quickstart](#quickstart)
  - [v1 pipeline — full command reference](#v1-pipeline--full-command-reference)
    - [1. Calibration (required once, before any layer-selection or compression config)](#1-calibration-required-once-before-any-layer-selection-or-compression-config)
    - [2. Sanity check](#2-sanity-check)
    - [3. Full experiment](#3-full-experiment)
    - [v1 config presets](#v1-config-presets)
    - [Alternate v1 entry point: `run_best.py`](#alternate-v1-entry-point-run_bestpy)
  - [v2 pipeline — full command reference](#v2-pipeline--full-command-reference)
    - [1. Sanity check](#1-sanity-check)
    - [2. Full benchmark (5 phases)](#2-full-benchmark-5-phases)
  - [Running the test suite](#running-the-test-suite)
  - [Output locations](#output-locations)
  - [Architecture summary: v1 vs v2](#architecture-summary-v1-vs-v2)
  - [Troubleshooting](#troubleshooting)

---

## Repo layout

```
LAKV/
├── run.py                     # v1 CLI entry point (calibrate / sanity / experiment)
├── run_best.py                 # v1 "best config" runner (3-agent relay, config A/C/D)
├── pipeline.py                 # v1 LAKVPipeline — N-agent KV relay orchestrator
├── evaluator.py                # v1 Evaluator — runs PRESETS across a dataset, saves results
├── calibration_profiler.py     # Per-layer importance profiling → LayerProfile JSON (Tier 1/2/3)
├── layer_selector.py           # Drops/reconstructs Tier-3 layers based on a LayerProfile
├── kv_compressor.py            # INT8 / INT4 per-head min-max quantization (+ outlier clipping)
├── offset_corrector.py         # v1-only: inter-agent prompt-length position offset correction
├── anchor_table.py             # Shared cross-agent/cross-question KV-delta correction pool
├── test_kv_compressor_clipping.py   # Standalone pytest: INT4 outlier-clipping accuracy test
├── requirements.txt
│
└── lakv_v2/                    # v2 — shared-prefix 2-agent pipeline (independent of v1 above)
    ├── sanity_test.py          # Quick 3-question, multi-config sanity check
    ├── benchmark.py            # 5-phase GSM8K benchmark runner (single-agent → full v2 matrix)
    ├── eval_runner.py          # Lower-level v2 eval harness (used programmatically)
    ├── diagnostic_test.py      # Ad-hoc debugging script for cache/position-id issues
    ├── pipeline/
    │   ├── shared_prefix_pipeline.py   # SharedPrefixPipeline — the v2 Solver→Finalizer relay
    │   ├── single_agent.py             # Absolute baseline: one agent, no relay, no compression
    │   └── two_agent.py
    ├── cache/                  # v2-side cache alignment/selection/compression helpers
    ├── utils/parsing.py
    └── tests/validation_suite.py

results/    # generated experiment output (git-ignored)
profiles/   # generated calibration profiles + plots (git-ignored)
```

---

## Environment setup

Requires **Python 3.10+** (developed/tested on 3.12) and a CUDA-capable GPU for
real runs (see below).

```bash
# 1. Clone / cd into the repo
cd LAKV

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (Git Bash / WSL / Linux / macOS):
source .venv/bin/activate

# 3. Install PyTorch with CUDA support FIRST (pick the command for your CUDA version
#    from https://pytorch.org/get-started/locally/ — the line below is for CUDA 12.1;
#    plain `pip install torch` will give you a CPU-only build that cannot run these
#    models at usable speed)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 4. Install everything else
pip install -r requirements.txt
```

`requirements.txt` covers: `torch`, `transformers`, `datasets` (HuggingFace, used to
load GSM8K), `numpy`, `scipy`, `matplotlib` (calibration plots), `tqdm`, `pytest`.

The first run of any script will download **Qwen2.5-7B-Instruct** (~15 GB) from the
Hugging Face Hub into your HF cache (`~/.cache/huggingface` by default, or
`$HF_HOME` if set) and the **GSM8K** dataset. If the model is gated or you hit
rate limits, log in first:

```bash
pip install huggingface_hub
huggingface-cli login
```

---

## Hardware requirements

- **GPU strongly required.** Qwen2.5-7B-Instruct in bf16 is ~15 GB of weights
  alone; every pipeline here also runs multi-agent forward passes and greedy
  decoding on top of that. On CPU this is technically possible but impractically
  slow (expect minutes per generated token) and may not fit in RAM on a machine
  with less than ~32 GB.
- All scripts accept `--device` (default `cuda`); pass `--device cpu` only for
  quick import/wiring smoke tests, not real evaluation.
- Single-GPU is assumed throughout (`device_map=<device>` loads the whole model
  onto one device — there's no multi-GPU sharding here).

---

## Quickstart

The fastest way to confirm your environment works end-to-end, without running a
full benchmark:

```bash
# v1: calibrate once (needed for layer-selection/compression configs), then a
# 3-sample sanity check comparing Config A (no compression) vs Config D
# (layer selection + adaptive compression).
python run.py --mode calibrate --n_calibration 50
python run.py --mode sanity --profile_path profiles/run_<timestamp>/qwen_gsm8k.json

# v2: a 3-question sanity check across all default v2 transfer-mode configs
python lakv_v2/sanity_test.py
```

(`run.py --mode calibrate` prints the exact `profile_path` to use next — copy it
from its output.)

---

## v1 pipeline — full command reference

### 1. Calibration (required once, before any layer-selection or compression config)

Profiles each of the model's 28 layers for importance and assigns a Tier
(1 = keep in INT8, 2 = keep in INT4, 3 = drop), saving the result as a
timestamped JSON + diagnostic plots.

```bash
python run.py --mode calibrate \
  --model Qwen/Qwen2.5-7B-Instruct \
  --task gsm8k \
  --n_calibration 50 \
  --profile_dir profiles \
  --device cuda
```
Output: `profiles/run_<timestamp>/qwen_gsm8k.json` (+ `layer_signals.png`,
`score_scatter.png`). The command prints the exact `--profile_path` to pass to
the next two modes.

### 2. Sanity check

Runs 3 fixed GSM8K questions through Config A and Config D (see [preset
table](#v1-config-presets) below) and prints per-sample outputs — use this to
confirm nothing crashes before committing to a full run.

```bash
python run.py --mode sanity \
  --profile_path profiles/run_<timestamp>/qwen_gsm8k.json \
  --device cuda \
  --print_raw_outputs
```

### 3. Full experiment

Runs every config in `--configs` across `--n_samples` GSM8K test questions,
checkpointing every 5 samples, and writes `experiment_results.json`,
`results_table.csv`, and `per_hop_stats.json`.

```bash
python run.py --mode experiment \
  --profile_path profiles/run_<timestamp>/qwen_gsm8k.json \
  --n_samples 100 \
  --output_dir results \
  --device cuda \
  --arch legacy
```

Default `--configs` (all run unless overridden):
```
single_agent  A  B_int8  B_int4  C  D  E  E_int8
```

Run a subset explicitly:
```bash
python run.py --mode experiment \
  --profile_path profiles/run_<timestamp>/qwen_gsm8k.json \
  --configs A B_int4 E E_int8 \
  --n_samples 50
```

Resume an interrupted run (reads the partial checkpoint from that run's output
folder and continues):
```bash
python run.py --mode experiment \
  --profile_path profiles/run_<timestamp>/qwen_gsm8k.json \
  --resume_dir results/run_<timestamp>
```

Use the 2-agent architecture variant instead of the default 3-agent relay:
```bash
python run.py --mode experiment \
  --profile_path profiles/run_<timestamp>/qwen_gsm8k.json \
  --arch two_agent
```

Full `run.py` flag reference:

| Flag                  | Default                                     | Meaning                                            |
| --------------------- | ------------------------------------------- | -------------------------------------------------- |
| `--model`             | `Qwen/Qwen2.5-7B-Instruct`                  | HF model id                                        |
| `--task`              | `gsm8k`                                     | Dataset tag (used for profile naming)              |
| `--mode`              | `calibrate`                                 | `calibrate` \| `sanity` \| `experiment`            |
| `--n_samples`         | `100`                                       | GSM8K test questions to evaluate (experiment mode) |
| `--n_calibration`     | `50`                                        | GSM8K train questions used for calibration         |
| `--configs`           | `single_agent A B_int8 B_int4 C D E E_int8` | Which presets to run                               |
| `--profile_dir`       | `profiles`                                  | Base dir for new calibration profiles              |
| `--profile_path`      | *(required for sanity/experiment)*          | Path to an existing `LayerProfile` JSON            |
| `--output_dir`        | `results`                                   | Base dir for experiment results                    |
| `--resume_dir`        | `None`                                      | Existing run folder to resume                      |
| `--device`            | `cuda`                                      | `cuda` \| `cpu`                                    |
| `--arch`              | `legacy`                                    | `legacy` (3-agent) \| `two_agent`                  |
| `--print_raw_outputs` | off                                         | Print raw generated text per sample                |

### v1 config presets

Defined in `evaluator.py::PRESETS`:

| Preset          | Layer selection | Compression                       | Offset correction | Reconstruction                         |
| --------------- | --------------- | --------------------------------- | ----------------- | -------------------------------------- |
| `single_agent`  | —               | —                                 | —                 | (baseline: one agent, no relay at all) |
| `A`             | off             | none                              | off               | zeros                                  |
| `B_int8`        | off             | uniform INT8                      | off               | zeros                                  |
| `B_int4`        | off             | uniform INT4                      | off               | zeros                                  |
| `C`             | on              | none                              | off               | zeros                                  |
| `C_nearest`     | on              | none                              | off               | nearest                                |
| `C_interpolate` | on              | none                              | off               | interpolate                            |
| `D`             | on              | adaptive (Tier1→INT8, Tier2→INT4) | off               | zeros                                  |
| `D_nearest`     | on              | adaptive                          | off               | nearest                                |
| `E`             | on              | adaptive                          | **on**            | zeros                                  |
| `E_int8`        | on              | uniform INT8                      | **on**            | zeros                                  |

Configs with layer selection or adaptive compression require `--profile_path`
from a prior calibration run. `E` / `E_int8` additionally build a real
`AnchorTable` (`pipeline.py`, gated by `use_offset_correction`) that learns
cross-question KV-drift corrections as the run progresses.

There are also ablation presets (random/fixed layer-index selections) available
via `--configs C_random_20_s0 C_top20 C_bottom20 ...` — see
`evaluator.py::ABLATION_CONFIGS`.

### Alternate v1 entry point: `run_best.py`

A narrower runner for just the "best" configs (A, C, D) against the full GSM8K
test set:

```bash
python run_best.py \
  --profile_path profiles/run_<timestamp>/qwen_gsm8k.json \
  --output_dir results/best_run \
  --device cuda
```

---

## v2 pipeline — full command reference

v2 does not require calibration to get started — `v2_full_prefix` / `v2_tail_only`
run with no profile at all. Layer-selection/compression rows do need one, produced
by the same `run.py --mode calibrate` command above (v1 and v2 share
`calibration_profiler.py`).

### 1. Sanity check

```bash
python lakv_v2/sanity_test.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --device cuda
```

Tests `v2_TEXT_ONLY`, `v2_TEXT_VERIFIER`, `v2_A_full`, `v2_A_tail` against 3 fixed
GSM8K questions each and checks for malformed output. Add `--profile_path` to
also test `v2_C_full` (layer selection). Add `--use_offset_correction` to enable
the `AnchorTable` and print its hit/admission call counts after each config.

```bash
python lakv_v2/sanity_test.py \
  --profile_path profiles/run_<timestamp>/qwen_gsm8k.json \
  --use_offset_correction
```

### 2. Full benchmark (5 phases)

```bash
python lakv_v2/benchmark.py \
  --profile_path profiles/run_<timestamp>/qwen_gsm8k.json \
  --output_dir results/v2_run_<timestamp> \
  --phase all
```

Phases (each writes its own `*_summaries.json` / `*_records.json` / `*_table.csv`
into `--output_dir`):

| Phase                          | Rows                                                    | Needs `--profile_path`? |
| ------------------------------ | ------------------------------------------------------- | ----------------------- |
| 1 — Baselines                  | `single_agent`, `text_2agent`                           | no                      |
| 2 — V2 Full/Tail KV            | `v2_full_prefix`, `v2_tail_only`                        | no                      |
| 3 — V2 Layer Selection         | `v2_layer_select`                                       | yes                     |
| 4 — V2 Selection + Compression | `v2_compressed`, `v2_compressed_tail`                   | yes                     |
| 5 — Full comparison matrix     | all of the above + more                                 | yes                     |
| `v1` (`--phase v1`)            | `v1_full_kv` (frozen v1 reference, for comparison only) | no                      |

Run one phase at a time:
```bash
python lakv_v2/benchmark.py --profile_path <path> --phase 1 --n_phase1 20
python lakv_v2/benchmark.py --profile_path <path> --phase 2 --n_phase2 20
```

Override the row list for a custom comparison:
```bash
python lakv_v2/benchmark.py --profile_path <path> \
  --rows v2_full_prefix v2_tail_only single_agent \
  --n_phase1 30 --phase 1
```

Enable `AnchorTable` offset correction on v2 rows (off by default):
```bash
python lakv_v2/benchmark.py --profile_path <path> \
  --phase 1 --n_phase1 5 --use_offset_correction --debug
```
`--debug` prints a per-sample block (question / solver reasoning / relay stats /
anchor hit-or-miss / finalizer answer / verdict) as it runs.

Full `lakv_v2/benchmark.py` flag reference:

| Flag                      | Default                      | Meaning                                   |
| ------------------------- | ---------------------------- | ----------------------------------------- |
| `--model`                 | `Qwen/Qwen2.5-7B-Instruct`   | HF model id                               |
| `--profile_path`          | `None`                       | Required for layer-select/compressed rows |
| `--device`                | `cuda`                       | `cuda` \| `cpu`                           |
| `--output_dir`            | `results/v2_run_<timestamp>` | Where results are written                 |
| `--phase`                 | `all`                        | `1`\|`2`\|`3`\|`4`\|`5`\|`all`\|`v1`      |
| `--n_phase1..5`           | `20/20/20/20/100`            | Sample counts per phase                   |
| `--print_raw`             | off                          | Print raw model outputs per sample        |
| `--debug`                 | off                          | Print full per-sample debug block         |
| `--rows`                  | `None`                       | Override row list for a custom run        |
| `--use_offset_correction` | off                          | Enable `AnchorTable` on v2 rows           |

---

## Running the test suite

```bash
# INT4 outlier-clipping accuracy test (synthetic tensors, CPU-only, no model/GPU)
pytest test_kv_compressor_clipping.py -v

# Or run it directly for a printed before/after accuracy report
python test_kv_compressor_clipping.py

# KVCompressor's built-in sanity check (none/INT8/INT4 round-trip on random data,
# CPU-only, no model/GPU)
python -c "from kv_compressor import KVCompressor; KVCompressor.run_sanity_check(device='cpu')"
```

`lakv_v2/tests/validation_suite.py` is **not** a pytest file — it's a
`ValidationSuite` class (cache-handoff parity, quantization round-trip, layer-drop
sensitivity) meant to be instantiated with a real loaded model/tokenizer and run
programmatically, so it needs a GPU:

```bash
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from lakv_v2.tests.validation_suite import ValidationSuite
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct', device_map='cuda')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
ValidationSuite(model, tokenizer, device='cuda').run_all()
"
```

---

## Output locations

Both `results/` and `profiles/` are **git-ignored** (generated artifacts, not
source):

- `profiles/run_<timestamp>/qwen_gsm8k.json` — calibration output (`LayerProfile`),
  plus `layer_signals.png` and `score_scatter.png` diagnostic plots.
- `results/run_<timestamp>/` (v1) or `results/v2_run_<timestamp>/` (v2) —
  `experiment_results.json` / `*_records.json` (full per-sample data),
  `results_table.csv` / `*_table.csv` (summary table), `per_hop_stats.json`
  (v1 only — per-hop compression/latency stats), and
  `experiment_results.partial.json` (v1's resumable checkpoint file, written
  every 5 samples — `checkpoint_every` is a parameter of
  `Evaluator.run_experiment`, not currently exposed as a `run.py` CLI flag).

---

## Architecture summary: v1 vs v2

|                                | v1 (`pipeline.py`)                                                                                           | v2 (`lakv_v2/pipeline/shared_prefix_pipeline.py`)                                                                                                    |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Agent count                    | 3 by default (configurable via `n_agents`)                                                                   | 2, fixed (Solver, Finalizer)                                                                                                                         |
| Prompts                        | Each agent gets its **own distinct** system prompt (reasoner / verifier / aggregator)                        | **One shared** system prompt/prefix for both agents                                                                                                  |
| Position handling              | `OffsetCorrector` corrects for inter-agent prompt-length mismatch (`raw_offset = sender_len - receiver_len`) | No `OffsetCorrector` at all — Finalizer's `position_ids` are an exact continuation from `cache_seq_len`, since both agents share one prefix          |
| Layer selection / quantization | `layer_selector.py` + `kv_compressor.py`                                                                     | **Same modules**, imported directly — not reimplemented                                                                                              |
| Calibration                    | `calibration_profiler.py`                                                                                    | **Same module**                                                                                                                                      |
| Cross-question KV correction   | `anchor_table.py`, gated by `use_offset_correction`                                                          | **Same module**, gated the same way — construct an `AnchorTable` and pass it into `SharedPrefixConfig(anchor_table=..., use_offset_correction=True)` |

v2 is not a bugfixed rewrite of v1 — it's a structurally different experiment
(fewer, homogeneous-prompt agents) that deliberately reuses v1's compression,
selection, calibration, and anchor-table modules unchanged.

---

## Troubleshooting

- **`CUDA out of memory`** — lower `--n_samples`/`--n_phase*`, close other GPU
  processes, or fall back to `--device cpu` for a (very slow) correctness check.
- **Gated/rate-limited model download** — run `huggingface-cli login` (see
  [Environment setup](#environment-setup)).
- **`--profile_path is required for ...`** — any config with layer selection or
  adaptive compression (`C*`, `D*`, `E*`, `v2_layer_select`, `v2_compressed*`)
  needs a calibration profile; run `python run.py --mode calibrate` first.
- **Malformed / `!!!!`-spam output** — this was a real historical bug (fixed in
  the `cec7714` commit: explicit `position_ids` in every greedy-decode step,
  correct `cache_seq_len`-based position offsets). If you see it again on a
  config that previously worked, check that any local KV-injection changes
  still pass explicit `position_ids` on every decode step.
- **No GPU available right now** — you can still validate wiring/logic changes
  without a model: `python -m py_compile <file>.py`, `python -c "import <module>"`,
  and the CPU-only tests under [Running the test suite](#running-the-test-suite)
  all work with no GPU and no model download.
