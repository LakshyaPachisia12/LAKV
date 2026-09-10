"""
Small manual-inspection check for the RLM+KV-relay prototype
(lakv/recursive_pipeline.py) -- NOT a scored evaluation. Runs a handful of
real HotpotQA examples through both return-channel conditions ("kv" vs
"text") and prints everything: per-child findings, final answers, and the
gold answer, so a human can eyeball whether recursive_kv produces coherent
output at all before committing to a larger scored run.

The CPU-only mechanics (RoPE fan-in merge correctness) are already verified
in tests/test_recursive_kv_merge.py -- this script is the first real-model
step after that, deliberately small (n=5, default) since the point is
sanity-checking output quality, not measuring accuracy yet.

Usage:
    python scripts/recursive_poc_check.py --n 5
    python scripts/recursive_poc_check.py --n 5 --n_children 2
"""

import argparse
import sys
from pathlib import Path

# Running as `python scripts/recursive_poc_check.py` only puts scripts/ on
# sys.path, not the project root -- so `run.py` and the `lakv` package
# aren't importable without this (same fix tests/test_*.py already use).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run import load_model
from lakv.recursive_pipeline import (
    RecursiveKVPipeline, RecursivePipelineConfig, load_hotpotqa_structured,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--n_children", type=int, default=2)
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_name, device="cuda")
    data = load_hotpotqa_structured(split=args.split, n=args.n)

    for channel in ("text", "kv"):
        print(f"\n{'=' * 70}\nRETURN CHANNEL: {channel}\n{'=' * 70}")
        config = RecursivePipelineConfig(n_children=args.n_children, return_channel=channel)
        pipeline = RecursiveKVPipeline(model, tokenizer, config, device="cuda")

        for i, item in enumerate(data):
            result = pipeline.run(item["question"], item["passages"])
            print(f"\n--- Example {i} ---")
            print(f"Question: {item['question']}")
            print(f"Gold answer: {item['answer']}")
            for j, (text, stat) in enumerate(zip(result.child_texts, result.hop_stats)):
                print(f"  Child {j} (seq_len={stat.seq_len}): {text[:300]!r}")
            print(f"Final answer ({channel}): {result.answer[:300]!r}")


if __name__ == "__main__":
    main()
