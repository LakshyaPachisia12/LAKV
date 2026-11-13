"""Did the pipeline edits on this branch change the paper's behavior?  (UNTESTED ON GPU.)

Compares a NEW run's per-sample outputs for one config (default A) against the
baseline run that produced the paper's chain numbers, aligned by `idx`:

    python scripts/check_regression_vs_baseline.py NEW_experiment_results.json [BASELINE.json] [--config A]
        [--baseline-config A_nopass]

--baseline-config names the config to read from the BASELINE file when it differs
from --config, e.g. `--config A_hide100 --baseline-config A_nopass` (the relay-necessity
curve's 100%-hidden level must reproduce the baseline's A_nopass arm).

BASELINE defaults to results/run_20261009_113254/experiment_results.json (Qwen2.5-7B,
HotpotQA, n=100; see docs/matched_donor_design.md, GPU step 0, for how to produce a
comparable NEW run). Only idx present in BOTH files are compared, so an n=20 NEW run is
checked against the first 20 baseline samples.

A sample "differs" if its raw_answer text or its parsed `predicted` answer differs.
Exit status: 0 = every compared sample identical; 1 = at least one difference, no
overlapping samples, or mismatched questions; 2 = config missing from a file.
hop_texts (Reasoner/Verifier text) are reported as a separate diagnostic: they do not
change the exit status, but a difference there with identical final answers still means
the intermediate behavior moved.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_BASELINE = "results/run_20261009_113254/experiment_results.json"
SHOW = 5


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare_runs(new: dict, base: dict, config: str = "A", baseline_config: str = None) -> dict:
    """Pure comparison of two loaded experiment_results dicts. `config` is read from
    `new`; `baseline_config` (default: same as `config`) is read from `base`."""
    baseline_config = baseline_config or config
    for name, d, c in (("new", new, config), ("baseline", base, baseline_config)):
        if c not in d or "per_sample" not in d.get(c, {}):
            raise KeyError(f"config {c!r} not found in the {name} results")
    by_idx = {s["idx"]: s for s in base[baseline_config]["per_sample"]}
    compared = identical = 0
    differences, question_mismatch, hop_text_diffs = [], [], []
    only_new = 0
    for s in new[config]["per_sample"]:
        b = by_idx.get(s["idx"])
        if b is None:
            only_new += 1
            continue
        if s.get("question") != b.get("question"):
            question_mismatch.append(s["idx"])
            continue
        compared += 1
        raw_same = (s.get("raw_answer") or "") == (b.get("raw_answer") or "")
        pred_same = s.get("predicted") == b.get("predicted")
        if raw_same and pred_same:
            identical += 1
        else:
            differences.append({"idx": s["idx"], "new_predicted": s.get("predicted"),
                                "base_predicted": b.get("predicted"),
                                "new_raw_answer": s.get("raw_answer"), "base_raw_answer": b.get("raw_answer")})
        if s.get("hop_texts") != b.get("hop_texts"):
            hop_text_diffs.append(s["idx"])
    return {"config": config, "baseline_config": baseline_config, "n_compared": compared, "n_identical": identical,
            "n_differing": len(differences), "differences": differences,
            "n_only_in_new": only_new, "question_mismatch_idx": question_mismatch,
            "hop_texts_differ_idx": hop_text_diffs}


def exit_code(result: dict) -> int:
    if result["n_compared"] == 0 or result["question_mismatch_idx"] or result["n_differing"]:
        return 1
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("new_results")
    ap.add_argument("baseline_results", nargs="?", default=DEFAULT_BASELINE)
    ap.add_argument("--config", default="A")
    ap.add_argument("--baseline-config", default=None,
                    help="config to read from the baseline file if its name differs from --config")
    args = ap.parse_args(argv)
    try:
        res = compare_runs(_load(args.new_results), _load(args.baseline_results), args.config,
                           args.baseline_config)
    except KeyError as e:
        print(f"ERROR: {e}")
        return 2
    bc = f" (baseline config {res['baseline_config']})" if res["baseline_config"] != res["config"] else ""
    print(f"config {res['config']}{bc}: new={args.new_results}  baseline={args.baseline_results}")
    print(f"n compared: {res['n_compared']}   n identical (raw_answer and predicted): {res['n_identical']}   "
          f"n differing: {res['n_differing']}")
    if res["n_only_in_new"]:
        print(f"  ({res['n_only_in_new']} NEW samples have no baseline counterpart and were not compared)")
    if res["question_mismatch_idx"]:
        print(f"ERROR: question text differs at idx {res['question_mismatch_idx'][:10]} - not the same examples")
    for d in res["differences"][:SHOW]:
        print(f"  idx {d['idx']}: predicted new={d['new_predicted']!r} base={d['base_predicted']!r}")
        print(f"      new  raw_answer: {(d['new_raw_answer'] or '')[:200]!r}")
        print(f"      base raw_answer: {(d['base_raw_answer'] or '')[:200]!r}")
    if len(res["differences"]) > SHOW:
        print(f"  ... and {len(res['differences']) - SHOW} more")
    if res["hop_texts_differ_idx"]:
        print(f"DIAGNOSTIC: Reasoner/Verifier hop_texts differ on {len(res['hop_texts_differ_idx'])} sample(s), "
              f"idx {res['hop_texts_differ_idx'][:10]} (does not affect the exit status)")
    rc = exit_code(res)
    print("RESULT:", "PASS (identical)" if rc == 0 else "FAIL (behavior differs or nothing to compare)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
