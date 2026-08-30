# LAKV-v2 Run Guide

This document outlines the step-by-step commands to run the LAKV-v2 experiments on the Lab System. It ensures we first leverage your existing, validated Calibration Profiler to extract precise layer importance, and then feed it into the new LAKV-v2 Evaluation Runner.

## Step 1: Calibration (Generating the Profile)

Yes! The calibration script is completely untouched and kept intact because it correctly measures attention importance, offset variance, and effective rank beautifully without modifications.

```bash
# Run calibration on GSM8K to profile layer importance
python run.py --mode calibrate --n_calibration 50
```

*Note: This will output the `profile_path` pointing to a `.json` file inside the `profiles/run_<timestamp>/` directory. Copy that JSON path for the next step.*

## Step 2: Running LAKV-v2 Evaluation

We can run the custom evaluation suite which directly compares the single agent baseline, your original framework, and our newly aligned systems simultaneously. 

Run this command from the root directory:

```bash
# Replace <YOUR_PROFILE_PATH> with the exact path generated in Step 1 (e.g., profiles/run_XXXXXXXX_XXXXXX/qwen_gsm8k.json)
python -c "
from lakv_v2.eval_runner import UnifiedEvalRunner
from run import load_model, load_gsm8k

# Initialize model
model, tokenizer = load_model('Qwen/Qwen2.5-7B-Instruct', 'cuda')

# Fetch your dataset
dataset = load_gsm8k('test', n=100)

# Initialize the v2 evaluator using the calibration profile
runner = UnifiedEvalRunner(model, tokenizer, device='cuda', profile_path='<YOUR_PROFILE_PATH>')

# Exact configs requested
configs = [
    'single_agent_baseline', 
    'old_lakv_A', 
    'lakv_v2_full', 
    'lakv_v2_selected', 
    'lakv_v2_selected_int8'
]

# Run experiments!
runner.run_eval(dataset, configs, output_dir='results/lakv_v2_new_run')
"
```

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
