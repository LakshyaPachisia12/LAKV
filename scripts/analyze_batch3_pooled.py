"""
Pools two or more rlm_long_context_sweep.py runs at the SAME context
length/config (different --seed each, so the runs are independent
filler-haystack draws over the same 25 scored questions -- see
docs/RLM_LONG_CONTEXT_LOG.md's final entry) into one paired McNemar test
+ bootstrap F1 CI, via lakv.stats (idx-matched, not list-position).

Each input run's own idx range is 0..n-1; naively concatenating would
collide (idx 0 from run A would overwrite/mis-pair with idx 0 from run
B). This offsets each run's idx by (run_index * 10000) before pooling,
so every record across every run gets a globally unique idx and
_align_by_idx pairs kv[i] with text[i] correctly within each run without
ever crossing runs.

No GPU required -- pure post-hoc analysis of already-saved JSON.

Usage:
    python scripts/analyze_batch3_pooled.py \\
        results/rlm_long_context_sweep/run_20261001_155505/sweep_results.json \\
        results/rlm_long_context_sweep/run_<new_seed1_run>/sweep_results.json \\
        --length 16000
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lakv.stats import mcnemar_test, bootstrap_f1_diff_ci


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_files", nargs="+", help="Paths to sweep_results.json files to pool")
    parser.add_argument("--length", type=int, required=True,
                         help="Context length key to pool (must be present in every run's "
                              "'records' dict, e.g. 16000)")
    args = parser.parse_args()

    pooled_kv, pooled_text = [], []
    for run_idx, path in enumerate(args.run_files):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        length_key = str(args.length)
        if length_key not in data["records"]:
            raise ValueError(f"{path}: length {args.length} not found, "
                              f"available: {list(data['records'].keys())}")
        kv_recs = data["records"][length_key]["kv"]
        text_recs = data["records"][length_key]["text"]
        offset = run_idx * 10000
        for r in kv_recs:
            pooled_kv.append({**r, "idx": r["idx"] + offset})
        for r in text_recs:
            pooled_text.append({**r, "idx": r["idx"] + offset})
        print(f"[{path}] seed={data['args'].get('seed')} n={len(kv_recs)} "
              f"kv_acc={100 * sum(r['correct'] for r in kv_recs) / len(kv_recs):.1f}% "
              f"text_acc={100 * sum(r['correct'] for r in text_recs) / len(text_recs):.1f}%")

    print(f"\nPooled n = {len(pooled_kv)} (from {len(args.run_files)} run(s))")
    pooled_kv_acc = 100 * sum(r["correct"] for r in pooled_kv) / len(pooled_kv)
    pooled_text_acc = 100 * sum(r["correct"] for r in pooled_text) / len(pooled_text)
    print(f"Pooled kv accuracy:   {pooled_kv_acc:.1f}%")
    print(f"Pooled text accuracy: {pooled_text_acc:.1f}%")

    result = mcnemar_test(pooled_kv, pooled_text)
    print(f"\nMcNemar (kv vs text, pooled): kv-only={result.n_a_only} text-only={result.n_b_only} "
          f"both_correct={result.n_both_correct} both_wrong={result.n_both_wrong} "
          f"n_discordant={result.n_discordant} p={result.p_value:.4f}")

    diff, lo, hi = bootstrap_f1_diff_ci(pooled_kv, pooled_text)
    print(f"Bootstrap F1 diff (kv - text): {diff:+.3f} [{lo:+.3f}, {hi:+.3f}]"
          f"  ({'excludes zero' if lo > 0 or hi < 0 else 'includes zero'})")

    print(f"\n{'SIGNIFICANT' if result.p_value < 0.05 else 'NOT significant'} at alpha=0.05 "
          f"(n_discordant={result.n_discordant})")


if __name__ == "__main__":
    main()
