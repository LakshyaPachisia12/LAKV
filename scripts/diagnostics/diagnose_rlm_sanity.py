"""
RLM scaffold sanity check — EXP-7 first look (docs/RESEARCH_PLAN.md, RLM addendum).

Standalone, no full evaluator/PRESETS wiring yet (that's a later step once this
is confirmed to work at all). Loads a handful of HotpotQA distractor-setting
samples WITH their 10 paragraphs kept separate (unlike run.py's load_hotpotqa,
which flattens them into one string for single_agent/text_agent), runs each
through lakv.rlm_scaffold.RLMPipeline, and reports EM/F1 plus call-count and
per-role (sub vs. root) latency.

Run:
    python scripts/diagnostics/diagnose_rlm_sanity.py

Reuses `model`/`tokenizer` from notebook globals if present (Kaggle-paste
convention matching the other scripts in this folder); otherwise loads
Qwen2.5-7B-Instruct fresh the same way run.py does.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from lakv.rlm_scaffold import RLMPipeline, RLMPipelineConfig
from lakv.qa_scoring import exact_match_score, f1_score

N_SAMPLES = 5


def load_hotpotqa_paragraphs(split: str = "validation", n: int = N_SAMPLES):
    """Like run.py's load_hotpotqa, but keeps the 10 distractor paragraphs as
    a list instead of flattening them into one context string — the RLM
    scaffold needs per-chunk boundaries to dispatch sub-calls over."""
    from datasets import load_dataset
    ds = load_dataset("hotpot_qa", "distractor")
    data = []
    for item in ds[split]:
        titles = item["context"]["title"]
        sentences = item["context"]["sentences"]
        paragraphs = [f"[{t}] " + " ".join(s) for t, s in zip(titles, sentences)]
        data.append({
            "question": item["question"],
            "answer": item["answer"],
            "paragraphs": paragraphs,
        })
        if len(data) >= n:
            break
    return data


if __name__ == "__main__":
    if "model" not in globals() or "tokenizer" not in globals():
        from transformers import AutoModelForCausalLM, AutoTokenizer
        MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
        print(f"[rlm_sanity] Loading {MODEL_NAME} fresh...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="eager",
            trust_remote_code=True,
        )
        model.eval()
    else:
        print("[rlm_sanity] Reusing model/tokenizer already in globals.")

    samples = load_hotpotqa_paragraphs(n=N_SAMPLES)
    pipe = RLMPipeline(model, tokenizer, RLMPipelineConfig())

    n_correct = 0
    total_calls = 0
    total_sub_latency = 0.0
    total_root_latency = 0.0

    for i, s in enumerate(samples):
        print("\n" + "=" * 100)
        print(f"SAMPLE {i}  ({len(s['paragraphs'])} paragraphs)")
        print(f"[QUESTION] {s['question']}")
        print(f"[GOLD] {s['answer']}")

        result = pipe.run(s["paragraphs"], s["question"])

        em = exact_match_score(result.answer, s["answer"])
        f1 = f1_score(result.answer, s["answer"])
        n_correct += int(em)
        total_calls += result.n_calls
        total_sub_latency += result.sub_call_latency_s
        total_root_latency += result.root_call_latency_s

        print(f"[PREDICTED] {result.answer}")
        print(f"[EM] {em}  [F1] {f1:.3f}")
        print(f"[CALLS] {result.n_calls}  (sub: {result.n_calls - 1}, root: 1)")
        print(f"[LATENCY] total={result.total_latency_s:.2f}s "
              f"sub={result.sub_call_latency_s:.2f}s root={result.root_call_latency_s:.2f}s")
        for j, sa in enumerate(result.sub_answers):
            print(f"  sub[{j}]: {sa[:120]}")

    n = len(samples)
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"EM accuracy: {n_correct}/{n} ({100 * n_correct / n:.1f}%)")
    print(f"Total calls across {n} samples: {total_calls} "
          f"(avg {total_calls / n:.1f} calls/sample)")
    print(f"Total sub-call latency: {total_sub_latency:.1f}s  "
          f"Total root-call latency: {total_root_latency:.1f}s")
    print("\nThis is EXP-7's baseline: text-only recursion, no KV relay. Compare")
    print("EM/F1 here against single_agent's HotpotQA accuracy (flat, no")
    print("decomposition) before moving to EXP-8 (KV relay at one call boundary).")
