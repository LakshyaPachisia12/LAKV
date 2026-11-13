"""Sanity readout for relay-necessity-curve runs.   *** UNTESTED ON GPU (logic unit-tested on CPU) ***

    python scripts/check_hide_run.py results/<dir>/run_<ts>/experiment_results.json [--expect_seed 0]

For every hide config in the file (A_hide20/50/80, A_hidegold1/2, with their _audit_dropped
partner) it checks, from the per-sample `hide` log written by the evaluator:
  * REAL and DROPPED arms hid exactly the same passages for every idx (a hard requirement);
  * gold axis: exactly `level` gold passages (and nothing else) hidden for every question;
  * fractional axis: for 10-passage items exactly level/10 passages hidden, never more gold
    than the item has, and exactly one gold hidden at 50%;
  * one hide_seed throughout (and equal to --expect_seed if given);
  * the run records dataset provenance and hide_seed in _run_metadata (warning only).
Diagnostics (never gate): realized mean gold hidden vs nominal, and per-config rates of empty
answers / unparsed answers, to spot degenerate generations at heavy hiding.
Exit status 1 if any hard check fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DROP = "_audit_dropped"


def _hide_cfgs(results):
    return [c for c, v in results.items() if not c.startswith("_") and isinstance(v, dict)
            and any(s.get("hide") for s in v.get("per_sample", []))]


def check_hide_results(results: dict, expect_seed=None) -> dict:
    cfgs = _hide_cfgs(results)
    violations, warnings, report = [], [], {}
    seeds = set()
    for c in cfgs:
        ps = [s for s in results[c]["per_sample"] if not s.get("skipped")]
        hides = [s["hide"] for s in ps if s.get("hide")]
        if len(hides) != len(ps):
            violations.append(f"{c}: {len(ps) - len(hides)} samples have no hide log")
        seeds.update(h["hide_seed"] for h in hides)
        axis, level = hides[0]["axis"], hides[0]["level"]
        n_gold = sum(h["n_gold_hidden"] for h in hides) / max(len(hides), 1)
        nominal = float(level) if axis == "gold" else 2.0 * level / 100.0
        empty = sum(1 for s in ps if not (s.get("raw_answer") or "").strip()) / max(len(ps), 1)
        unparsed = sum(1 for s in ps if s.get("predicted") in (None, "")) / max(len(ps), 1)
        report[c] = {"axis": axis, "level": level, "n": len(ps), "mean_gold_hidden": n_gold,
                     "nominal_gold_hidden": nominal, "frac_empty_answer": empty, "frac_unparsed_answer": unparsed}
        if axis == "gold":
            bad = [i for i, h in enumerate(hides) if h["n_gold_hidden"] != level or h["n_hidden"] != level]
            if bad:
                violations.append(f"{c}: {len(bad)} questions did not hide exactly {level} gold passage(s) and nothing else")
        else:
            for i, h in enumerate(hides):
                if h["n_passages"] == 10 and h["n_hidden"] != level // 10:
                    violations.append(f"{c}: question {i} hid {h['n_hidden']} of 10 passages (expected {level // 10})")
                    break
            if any(h["n_gold_hidden"] > h["n_gold"] for h in hides):
                violations.append(f"{c}: a question hid more gold passages than it has")
            if level == 50 and any(h["n_passages"] == 10 and h["n_gold_hidden"] != 1 for h in hides):
                violations.append(f"{c}: hide50 must hide exactly one gold passage on every 10-passage question")
            if abs(n_gold - nominal) > 0.25:
                warnings.append(f"{c}: mean gold hidden {n_gold:.2f} is far from nominal {nominal:.2f}")
    # real and dropped must have hidden identical passages
    pairs = []
    for c in cfgs:
        if c.endswith(DROP):
            continue
        d = c + DROP
        if d not in results:
            violations.append(f"{c}: no matching {d} arm in this file")
            continue
        a = {s["idx"]: s["hide"]["hidden_idx"] for s in results[c]["per_sample"] if s.get("hide")}
        b = {s["idx"]: s["hide"]["hidden_idx"] for s in results[d]["per_sample"] if s.get("hide")}
        if set(a) != set(b):
            violations.append(f"{c} vs {d}: different question sets ({len(a)} vs {len(b)})")
        diff = [i for i in set(a) & set(b) if a[i] != b[i]]
        if diff:
            violations.append(f"{c} vs {d}: hidden passages differ on {len(diff)} questions (first idx {sorted(diff)[:5]})")
        pairs.append((c, d, len(set(a) & set(b)) - len(diff)))
    if len(seeds) > 1:
        violations.append(f"more than one hide_seed in one file: {sorted(seeds)}")
    if expect_seed is not None and seeds and seeds != {expect_seed}:
        violations.append(f"hide_seed {sorted(seeds)} != expected {expect_seed}")
    md = results.get("_run_metadata", {})
    if md.get("hide_seed") is None:
        warnings.append("_run_metadata has no hide_seed")
    if not md.get("dataset_provenance"):
        warnings.append("_run_metadata has no dataset_provenance (run.py records it for HotpotQA)")
    return {"configs": report, "pairs_identical": pairs, "violations": violations, "warnings": warnings,
            "seeds": sorted(seeds), "n_hide_configs": len(cfgs)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results_json")
    ap.add_argument("--expect_seed", type=int, default=None)
    args = ap.parse_args(argv)
    res = check_hide_results(json.loads(Path(args.results_json).read_text(encoding="utf-8")), args.expect_seed)
    print(json.dumps(res, indent=2, default=str))
    if res["n_hide_configs"] == 0:
        print("FAIL: no hide configs found in this file")
        return 1
    print("RESULT:", "PASS" if not res["violations"] else "FAIL: " + "; ".join(res["violations"]))
    return 0 if not res["violations"] else 1


if __name__ == "__main__":
    sys.exit(main())
