"""CLI entry point for the RLM engine (spec section 32, scoped to Phase 1).

Usage:
    python -m lakv.rlm.run --context path/to/file.txt --query "..." \
        --max-depth 1 --max-steps 20 --chunk-size 512 --trace \
        --output-dir results/rlm_run

Model loading mirrors run.py's load_model() (same defaults: bfloat16, eager
attention, trust_remote_code) so behavior stays consistent with the rest of
this repo's CLI conventions — deliberately not importing run.py directly
since it's a script, not a package, matching how scripts/diagnostics/*.py
already duplicate this small amount of loading logic rather than reaching
into run.py's internals.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from lakv.rlm.config import RLMBudget, RLMConfig
from lakv.rlm.engine import RLMEngine
from lakv.rlm.llm_backend import HFBackend


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model_name: str, device: str, dtype: str, attn_implementation: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}[dtype]
    resolved_device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
    if resolved_device != device:
        print(f"[rlm.run] CUDA not available, falling back to CPU.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=resolved_device,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer, resolved_device


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LAKV RLM engine — Phase 1")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--context", required=True, help="Path to a text file to use as context")
    p.add_argument("--query", required=True)
    p.add_argument("--max-depth", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=20, help="max root iterations")
    p.add_argument("--max-calls", type=int, default=40)
    p.add_argument("--max-tokens", type=int, default=8192, help="max generated tokens (budget)")
    p.add_argument("--chunk-size", type=int, default=512, help="hint only — the model still "
                    "chooses its own chunk_size via chunk_context actions; this is not enforced")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--attn_implementation", choices=["eager", "sdpa"], default="eager")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--trace", action="store_true", help="write trace.jsonl to --output-dir")
    p.add_argument("--output-dir", default=None)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    set_seeds(args.seed)

    context_text = Path(args.context).read_text(encoding="utf-8")

    model, tokenizer, resolved_device = load_model(
        args.model, args.device, args.dtype, args.attn_implementation
    )
    backend = HFBackend(model, tokenizer, model_name=args.model, device=resolved_device)

    budget = RLMBudget(
        max_root_iterations=args.max_steps,
        max_total_llm_calls=args.max_calls,
        max_depth=args.max_depth,
        max_generated_tokens=args.max_tokens,
    )
    config = RLMConfig(
        do_sample=args.do_sample,
        temperature=args.temperature,
        model_name=args.model,
        device=resolved_device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        budget=budget,
        seed=args.seed,
    )

    engine = RLMEngine(backend, config)
    result = engine.run(context_text, args.query, tokenizer=tokenizer)

    print(f"\n[ANSWER] {result.answer}")
    print(f"[TERMINATION] {result.termination_reason}")
    print(f"[CALLS] {result.n_calls}  [LATENCY] {result.total_latency_s:.2f}s")
    print(f"[METRICS] {json.dumps(result.metrics, indent=2, default=str)}")

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(config.as_dict(), indent=2, default=str))
        (out / "metrics.json").write_text(json.dumps(result.metrics, indent=2, default=str))
        (out / "answer.txt").write_text(result.answer)
        if args.trace:
            result.trace.to_jsonl(str(out / "trace.jsonl"))
        print(f"[rlm.run] Saved outputs to {out}/")

    return result


if __name__ == "__main__":
    main()
