"""
Print paired McNemar + F1-bootstrap comparisons for every pair among a set of
configs in one results file.

Built for the dropped-KV causal-audit control (A vs A_audit_dropped vs
A_audit_mismatched), but works for any set of config names that are present
in the same results dir and share the same scored example indices.

Supports two result-file shapes, auto-detected:
  - run.py's sequential-pipeline output, experiment_results.json:
    {cfg: {"summary": ..., "per_sample": [...]}}, each per_sample record
    already has a "correct" field.
  - scripts/recursive_poc_check.py's fan-in output, transcripts.json:
    {"args": ..., "summary": ..., "records": {cfg: [...]}}, each record
    has "em" instead of "correct", and isn't wrapped in a per-cfg
    "per_sample" key. load_results() reshapes this into the same
    {cfg: {"per_sample": [...]}} shape the rest of the script already
    expects, copying "em" into a "correct" key on each record -- the
    sequential-pipeline path below is completely unchanged by this.

Reuses lakv.stats.mcnemar_test / bootstrap_f1_diff_ci directly rather than
reimplementing the math -- that module is already the project's one
significance-testing tool, and this script is just a convenience wrapper
that prints the three-way comparison table and validates pairing first.

Usage:
    python scripts/paired_audit.py results/run_20261008_215649
    python scripts/paired_audit.py results/run_20261009_120000 --configs A A_audit_dropped A_audit_mismatched
    python scripts/paired_audit.py results/recursive_poc_check/run_20260917_102302 --configs kv kv_audit_zeroed kv_audit_random kv_audit_mismatched
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

# Running this as `python scripts/paired_audit.py` does not put the repo
# root on sys.path, so the `lakv` package import below would fail -- same
# fix every other script in this directory already needs and applies.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lakv.stats import mcnemar_test, bootstrap_f1_diff_ci


def _reshape_fan_in_records(raw: dict) -> dict:
    """Reshape scripts/recursive_poc_check.py's transcripts.json ("records":
    {cfg: [...]}, each record using "em") into the same {cfg: {"per_sample":
    [...]}} shape experiment_results.json already uses, with a "correct"
    key added to each record (copied from "em", not renamed -- the original
    "em" key is left in place too, in case anything else wants it)."""
    reshaped = {}
    for cfg, records in raw["records"].items():
        per_sample = []
        for record in records:
            record = dict(record)
            record["correct"] = record["em"]
            per_sample.append(record)
        reshaped[cfg] = {"per_sample": per_sample}
    return reshaped


def load_results(results_dir: str) -> dict:
    """Load either file shape this script supports, auto-detected by
    filename -- see the module docstring. experiment_results.json (the
    sequential pipeline's own format) is tried first and returned as-is,
    completely unchanged from before; transcripts.json (the fan-in
    prototype's format) is detected by its "records" key and reshaped by
    _reshape_fan_in_records() into the same shape before being returned."""
    seq_path = Path(results_dir) / "experiment_results.json"
    if seq_path.exists():
        with open(seq_path, encoding="utf-8") as f:
            return json.load(f)

    fan_in_path = Path(results_dir) / "transcripts.json"
    if fan_in_path.exists():
        with open(fan_in_path, encoding="utf-8") as f:
            raw = json.load(f)
        if "records" not in raw:
            raise ValueError(f"{fan_in_path} does not have the expected 'records' key "
                              f"-- found top-level keys: {sorted(raw.keys())}")
        return _reshape_fan_in_records(raw)

    raise FileNotFoundError(
        f"Neither experiment_results.json nor transcripts.json found in {results_dir}"
    )


def check_same_examples(results: dict, configs: list) -> None:
    """Confirm every config has the same idx set, and that question/gold text
    is identical across configs for every shared idx, before any comparison
    is trusted. Raises immediately if anything doesn't match -- silently
    comparing misaligned examples is exactly the class of bug this guards
    against (see lakv/stats.py's own idx-based alignment for why)."""
    idx_sets = {}
    for cfg in configs:
        if cfg not in results:
            raise ValueError(f"Config {cfg!r} not found in this results file. "
                              f"Available configs: {sorted(results.keys())}")
        idx_sets[cfg] = {s["idx"] for s in results[cfg]["per_sample"]}

    first_cfg = configs[0]
    first_idx_set = idx_sets[first_cfg]
    for cfg in configs[1:]:
        if idx_sets[cfg] != first_idx_set:
            missing = first_idx_set - idx_sets[cfg]
            extra = idx_sets[cfg] - first_idx_set
            raise ValueError(
                f"Config {cfg!r} does not have the same idx set as {first_cfg!r}. "
                f"Missing from {cfg!r}: {sorted(missing)}. Extra in {cfg!r}: {sorted(extra)}."
            )

    by_idx = {
        cfg: {s["idx"]: s for s in results[cfg]["per_sample"]}
        for cfg in configs
    }
    for idx in sorted(first_idx_set):
        questions = {by_idx[cfg][idx]["question"] for cfg in configs}
        golds = {by_idx[cfg][idx]["gold"] for cfg in configs}
        if len(questions) > 1:
            raise ValueError(f"idx={idx}: question text differs across configs {configs}")
        if len(golds) > 1:
            raise ValueError(f"idx={idx}: gold answer differs across configs {configs}")

    print(f"OK: all {len(configs)} configs share the same {len(first_idx_set)} example "
          f"indices, with identical question/gold text for every one.\n")


def print_pair_comparison(results: dict, cfg_a: str, cfg_b: str) -> None:
    per_sample_a = results[cfg_a]["per_sample"]
    per_sample_b = results[cfg_b]["per_sample"]

    mc = mcnemar_test(per_sample_a, per_sample_b)
    f1_diff, ci_lo, ci_hi = bootstrap_f1_diff_ci(per_sample_a, per_sample_b)

    sig = "SIGNIFICANT" if mc.p_value < 0.05 else "not significant"
    ci_note = "excludes zero" if (ci_lo > 0 or ci_hi < 0) else "includes zero"

    print(f"{cfg_a} vs {cfg_b}")
    print(f"  discordant pairs: {mc.n_discordant}  "
          f"({cfg_a}-only={mc.n_a_only}, {cfg_b}-only={mc.n_b_only})")
    print(f"  both correct: {mc.n_both_correct}   both wrong: {mc.n_both_wrong}")
    print(f"  McNemar exact two-sided p = {mc.p_value:.6f}  ({sig})")
    print(f"  F1 diff ({cfg_a} minus {cfg_b}): {f1_diff * 100:+.1f} pts, "
          f"95% CI [{ci_lo * 100:+.1f}, {ci_hi * 100:+.1f}] pts  ({ci_note})")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", help="Directory containing experiment_results.json")
    parser.add_argument(
        "--configs", nargs="+",
        default=["A", "A_audit_dropped", "A_audit_mismatched"],
        help="Config names to compare, pairwise, against each other.",
    )
    args = parser.parse_args()

    results = load_results(args.results_dir)
    check_same_examples(results, args.configs)

    print("=" * 70)
    print("PAIRED COMPARISONS")
    print("=" * 70)
    for cfg_a, cfg_b in itertools.combinations(args.configs, 2):
        print_pair_comparison(results, cfg_a, cfg_b)


if __name__ == "__main__":
    main()
