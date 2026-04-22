"""
LAKV — Main entry point.

Usage:
  python run.py --mode calibrate
      → saves to profiles/run_<timestamp>/

  python run.py --mode sanity --profile_path profiles/run_<timestamp>/qwen_gsm8k.json
      → 3-sample sanity check (Config A vs D)

  python run.py --mode experiment --profile_path profiles/run_<timestamp>/qwen_gsm8k.json
      → full evaluation, results saved to results/run_<timestamp>/
"""

import argparse
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_model(model_name: str, device: str):
    """Load Qwen2.5-7B in bfloat16 with eager attention for calibration support."""
    print(f"[run] Loading model: {model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def load_gsm8k(split: str = "test", n: int = None):
    """Return list of {question, answer} dicts from GSM8K."""
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main")
    data = []
    for item in ds[split]:
        answer = item["answer"].split("####")[-1].strip()
        data.append({"question": item["question"], "answer": answer})
    if n is not None:
        data = data[:n]
    return data


def mode_calibrate(args):
    from calibration_profiler import CalibrationProfiler, plot_signals, plot_score_scatter

    # New timestamped folder for every calibration run
    profile_dir = Path(args.profile_dir) / f"run_{_timestamp()}"
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_path = profile_dir / "qwen_gsm8k.json"
    print(f"[run] Profile will be saved to: {profile_path}")

    model, tokenizer = load_model(args.model, args.device)
    questions = [e["question"] for e in load_gsm8k("train", n=args.n_calibration)]

    profiler = CalibrationProfiler(model, tokenizer, device=args.device)
    profile = profiler.run_calibration(questions, save_path=str(profile_path),
                                       dataset_name=args.task)

    plot_signals(profile, str(profile_dir / "layer_signals.png"))
    plot_score_scatter(profile, str(profile_dir / "score_scatter.png"))

    print("\n[run] Tier assignments:")
    for i, t in enumerate(profile.tier_assignment):
        tag = {1: "INT8 (keep)", 2: "INT4 (keep)", 3: "DROP"}[t]
        print(f"  Layer {i:2d}: Tier {t}  \u2192  {tag}")

    print(f'\n[run] Next steps:')
    print(f'  Sanity : python run.py --mode sanity    --profile_path "{profile_path}"')
    print(f'  Eval   : python run.py --mode experiment --profile_path "{profile_path}"')


def mode_sanity(args):
    from evaluator import Evaluator
    from kv_compressor import KVCompressor
    
    if not args.profile_path:
        raise ValueError("--profile_path is required for --mode sanity. "
                         "Run calibrate first and use the printed path.")
                         
    # Run compressor sanity check FIRST before pipeline sanity
    print("Step 0: Verifying KVCompressor before running pipeline...")
    KVCompressor.run_sanity_check(device=args.device)
    print()

    model, tokenizer = load_model(args.model, args.device)
    dataset = load_gsm8k("test", n=3)
    Evaluator(model, tokenizer, device=args.device).run_sanity_check(
        profile_path=args.profile_path, dataset=dataset, arch=args.arch, print_raw_outputs=args.print_raw_outputs)


def mode_experiment(args):
    from evaluator import Evaluator
    if not args.profile_path:
        raise ValueError("--profile_path is required for --mode experiment.")

    if args.resume_dir:
        output_dir = Path(args.resume_dir)
        if not output_dir.exists():
            raise ValueError(f"--resume_dir does not exist: {output_dir}")
        print(f"[run] Resuming run in: {output_dir}")
        resume = True
    else:
        output_dir = Path(args.output_dir) / f"run_{_timestamp()}"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[run] Results will be saved to: {output_dir}")
        resume = False

    model, tokenizer = load_model(args.model, args.device)
    dataset = load_gsm8k("test", n=args.n_samples)
    Evaluator(model, tokenizer, device=args.device).run_experiment(
        dataset=dataset,
        profile_path=args.profile_path,
        configs_to_run=args.configs,
        n_samples=args.n_samples,
        output_dir=str(output_dir),
        resume=resume,
        arch=args.arch,
        print_raw_outputs=args.print_raw_outputs,
    )


def main():
    parser = argparse.ArgumentParser(description="LAKV — KV Cache Compression Pipeline")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--task", default="gsm8k")
    parser.add_argument("--mode", choices=["calibrate", "sanity", "experiment"],
                        default="calibrate")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--n_calibration", type=int, default=50)
    parser.add_argument("--configs", nargs="+",
                        default=["single_agent", "A", "B_int8", "B_int4", "C", "D"])
    parser.add_argument("--profile_dir", default="profiles",
                        help="Base dir for calibration profiles (timestamped sub-folder auto-created)")
    parser.add_argument("--profile_path", default=None,
                        help="Explicit path to LayerProfile JSON (required for sanity/experiment)")
    parser.add_argument("--output_dir", default="results",
                        help="Base dir for experiment results (timestamped sub-folder auto-created)")
    parser.add_argument("--resume_dir", default=None,
                        help="Path to an existing run folder to resume (e.g. results/run_20260420_123456)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arch", choices=["legacy", "two_agent"], default="legacy",
                        help="Architecture mode for multi-agent configs (default: legacy)")
    parser.add_argument("--print_raw_outputs", action="store_true", help="Print raw generated outputs")
    args = parser.parse_args()

    if args.mode == "calibrate":
        mode_calibrate(args)
    elif args.mode == "sanity":
        mode_sanity(args)
    elif args.mode == "experiment":
        mode_experiment(args)


if __name__ == "__main__":
    main()
