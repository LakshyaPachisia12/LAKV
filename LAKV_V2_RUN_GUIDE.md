# LAKV-v2 Run Guide

This document outlines the step-by-step commands to run the LAKV-v2 experiments on the Lab System. It ensures we first leverage your existing, validated Calibration Profiler to extract precise layer importance, and then feed it into the new LAKV-v2 Evaluation Runner.

**Prerequisites:** `pip install accelerate` — recent `transformers` requires it for
`device_map=` and it's missing from `requirements.txt` on some setups. Also make sure
`run.py`'s `load_gsm8k()` points at `openai/gsm8k` (not the bare `gsm8k` short-name) —
current `datasets`/`huggingface_hub` versions no longer resolve legacy canonical
dataset names without a namespace.

## Step 1: Calibration (Generating the Profile)

Yes! The calibration script is completely untouched and kept intact because it correctly measures attention importance, offset variance, and effective rank beautifully without modifications.

```bash
# Run calibration on GSM8K to profile layer importance
python run.py --mode calibrate --n_calibration 50
```

*Note: This will output the `profile_path` pointing to a `.json` file inside the `profiles/run_<timestamp>/` directory. Copy that JSON path for the next step.*

## Step 2: Running the Evaluation

### Option A (recommended): plain CLI, no script needed

For comparing the single-agent baseline against the original (root-level) pipeline's
presets — `A`, `B_int8`, `B_int4`, `C`, `D`, `E`, `E_int8` — plus `text_agent`, just use
`run.py`'s own flags. This is the same entry point `--mode calibrate` and `--mode sanity`
use, and it works identically on Windows/PowerShell, Linux, or a Kaggle notebook:

```bash
python run.py --mode experiment \
  --profile_path "<YOUR_PROFILE_PATH>" \
  --configs single_agent A B_int8 B_int4 C D E E_int8 text_agent \
  --n_samples 100 \
  --output_dir results
```

*Replace `<YOUR_PROFILE_PATH>` with the path printed at the end of Step 1
(e.g. `profiles/run_XXXXXXXX_XXXXXX/qwen_gsm8k.json`). Results land in
`results/run_<timestamp>/`.*

### Option B: `UnifiedEvalRunner`, only if you need the `lakv_v2`-exclusive pipelines

`lakv_v2_full`, `lakv_v2_selected`, and `lakv_v2_selected_int8` (the new `TwoAgentPipeline`
variants) only exist in `lakv_v2/eval_runner.py`, not in `run.py`'s preset list — use this
path when you want those compared side-by-side against the baseline and the old pipeline
in one run. There's no CLI for this one, so save it as a script and run that (an inline
`python -c "..."` heredoc doesn't translate across shells):

```python
# eval_lakv_v2.py
from lakv_v2.eval_runner import UnifiedEvalRunner
from run import load_model, load_gsm8k

model, tokenizer = load_model("Qwen/Qwen2.5-7B-Instruct", "cuda")
dataset = load_gsm8k("test", n=100)

runner = UnifiedEvalRunner(model, tokenizer, device="cuda", profile_path="<YOUR_PROFILE_PATH>")

configs = [
    "single_agent_baseline",
    "old_lakv_A",
    "lakv_v2_full",
    "lakv_v2_selected",
    "lakv_v2_selected_int8",
]
runner.run_eval(dataset, configs, output_dir="results/lakv_v2_new_run")
```

```bash
python eval_lakv_v2.py
```

Note: `text_agent` is not yet wired into `UnifiedEvalRunner.select_pipeline()`, so it's
only reachable via Option A.

## Step 3: Run Validation Suite (Optional but Recommended)

Before initiating a large benchmark, you can explicitly test if the new specific Qwen2.5 cache alignment mathematically verifies correct parity against an uninterrupted string execution. 

```bash
python -m lakv_v2.tests.validation_suite
```
*It will evaluate and print `PASS`/`FAIL` for Cache Parity metrics, Quantization math, and Layer drop bounds.*

---

## Explanation of Implemented Configs

These are the setups executed in the evaluation array:

1. **`single_agent_baseline`**: 
   - Utilizes the `SingleAgentPipeline` doing one standard generation without any KV caching. 
   - Establishes the absolute maximum score and base latency against the unmodified Qwen2.5.

2. **`old_lakv_A`**: 
   - Imports your preserved `pipeline.py` and old evaluator setups. 
   - Recreates exactly the old 3-agent orchestration to display the performance gap natively.

3. **`lakv_v2_full`**: 
   - Leverages the new `TwoAgentPipeline` cleanly aligned with exact RoPE metrics from the `CacheAligner`. 
   - Transmits *all* layers to measure isolated prompt handoff improvements without compression penalties.

4. **`lakv_v2_selected`**: 
   - Leverages the `TwoAgentPipeline`.
   - Passes the `profile_path` into the runner, cleanly extracting your `Tier 1` and `Tier 2` layer indices from the calibration JSON, discarding the weak bottom tiers seamlessly. 
   - Reconstructs dropped layers using `mean_fill` interpolation instead of zeros to secure float bounds.

5. **`lakv_v2_selected_int8`**: 
   - Incorporates cleanly aligned layer mapping and tier selection identical to the above.
   - Pushes selected caches into the new `CompressorV2` wrapper running strict `uniform_int8` before transmission. Tests how well 8-bit precision maintains logic fidelity when dropped layers securely interpolate.
