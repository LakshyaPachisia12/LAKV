"""Readout for the matched-donor audit modes (UNTESTED ON GPU; logic unit-tested on CPU).

    python scripts/check_matched_donors.py results/<run>/experiment_results.json

Prints, for whichever of these configs are present:
  * self-replay validation: A vs A_audit_self_replay (identical outputs, paired
    McNemar on the common subset, numeric cache difference of rebuild vs real)
  * donor-log invariants for every matched-donor arm: skips, pad_positions,
    length ratios (length_matched), and the same-question donors' own checks
    (hop0_differs_from_greedy, defends_distractor)
Exit status 1 if a hard invariant is violated (padding > 0, a length_matched
ratio outside +-5%), so it can gate the next stage of the GPU checklist.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

NEW_MODES = ("length_matched", "matched_passages", "matched_wrong_answer", "self_replay")


def _logs(per_sample):
    for s in per_sample:
        for l in s.get("audit_logs", []):
            yield s, l


def summarize_self_replay(results: dict, base: str = "A", replay: str = "A_audit_self_replay") -> dict:
    from lakv.stats import mcnemar_test
    pa, pb = results[base]["per_sample"], results[replay]["per_sample"]
    by = {s["idx"]: s for s in pb}
    pairs = [(a, by[a["idx"]]) for a in pa if a["idx"] in by and not by[a["idx"]].get("skipped")]
    identical = sum(1 for a, b in pairs if (a.get("raw_answer") or "").strip() == (b.get("raw_answer") or "").strip())
    out = {"n_pairs": len(pairs), "identical_answer_frac": identical / len(pairs) if pairs else None,
           "accuracy_base": results[base]["summary"]["accuracy"], "accuracy_replay": results[replay]["summary"]["accuracy"]}
    if pairs:
        m = mcnemar_test(pa, pb)
        out.update(mcnemar_p=m.p_value, n_discordant=m.n_discordant)
    per_hop = {}
    for _s, l in _logs(pb):
        r = l.get("rebuild_vs_real")
        if r and r.get("comparable"):
            per_hop.setdefault(l["agent_idx"], []).append(r)
    out["rebuild_vs_real"] = {
        hop: {"n": len(v), "min_cosine_worst": min(x["min_cosine"] for x in v),
              "max_abs_diff_worst": max(x["max_abs_diff"] for x in v),
              "max_abs_diff_median": st.median(x["max_abs_diff"] for x in v),
              "max_rel_l2_worst": max(x["max_rel_l2"] for x in v)}
        for hop, v in sorted(per_hop.items())}
    out["not_comparable"] = sum(1 for _s, l in _logs(pb)
                                if l.get("rebuild_vs_real") and not l["rebuild_vs_real"].get("comparable"))
    return out


def check_donor_logs(results: dict, cfg: str) -> dict:
    ps = results[cfg]["per_sample"]
    n_skipped = sum(1 for s in ps if s.get("skipped"))
    logs = [l for _s, l in _logs(ps) if l.get("mode") in NEW_MODES]
    scored = [l for l in logs if not l.get("skipped")]
    violations = []
    pad_bad = [l for l in scored if l.get("pad_positions", 0) != 0 or l.get("truncated_positions", 0) != 0]
    if pad_bad:
        violations.append(f"{len(pad_bad)} substitutions had padding/truncation")
    out = {"n_samples": len(ps), "n_skipped": n_skipped, "n_substitutions": len(scored),
           "skip_reasons": sorted({s.get("skip_reason") for s in ps if s.get("skipped")})}
    ratios = [l["len_ratio"] for l in scored if "len_ratio" in l]
    if ratios:
        out["len_ratio_min"], out["len_ratio_max"] = min(ratios), max(ratios)
    if any(l.get("mode") == "length_matched" for l in scored):
        bad = [r for r in ratios if abs(r - 1) > 0.05 + 1e-9]
        if bad:
            violations.append(f"{len(bad)} length_matched donors outside +-5%")
    for key in ("hop0_differs_from_greedy",):
        vals = [l[key] for l in scored if l.get(key) is not None]
        if vals:
            out[key + "_rate"] = sum(bool(v) for v in vals) / len(vals)
    dd = [x for l in scored for x in (l.get("defends_distractor") or [])]
    if dd:
        out["defends_distractor_rate"] = sum(bool(x) for x in dd) / len(dd)
    out["violations"] = violations
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_json")
    args = ap.parse_args()
    results = json.loads(Path(args.results_json).read_text(encoding="utf-8"))
    rc = 0
    if "A_audit_self_replay" in results and "A" in results:
        print("== self-replay validation ==")
        print(json.dumps(summarize_self_replay(results), indent=2, default=str))
    for cfg in results:
        if cfg.startswith("_"):
            continue
        if any(cfg.endswith("_audit_" + m) for m in NEW_MODES):
            r = check_donor_logs(results, cfg)
            print(f"== {cfg} ==")
            print(json.dumps(r, indent=2, default=str))
            if r["violations"]:
                rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
