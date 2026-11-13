"""Readout for the matched-donor audit modes (UNTESTED ON GPU; logic unit-tested on CPU).

    python scripts/check_matched_donors.py results/<run>/experiment_results.json

Prints, for whichever of these configs are present:
  * self-replay validation: A vs A_audit_self_replay. GATE (decides the exit
    status): (1) at least --min_agree of --expect_pairs final answers agree
    (default 19 of 20; "agree" = the parsed `predicted` answers are equal after
    answer normalization), and (2) no substitution had not_comparable > 0, i.e. the
    rebuilt cache has the same length as the real one (otherwise the cache-length
    rule is wrong). Cosine similarity / max-abs-diff / rel-L2 of rebuilt vs real
    caches, raw-text identity and McNemar are printed as DIAGNOSTICS and never gate.
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
    agree = sum(1 for a, b in pairs if answers_agree(a, b))
    out = {"n_pairs": len(pairs), "n_answers_agree": agree,
           "identical_answer_frac": identical / len(pairs) if pairs else None,  # diagnostic: raw text identical
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
    out["n_rebuild_logs"] = sum(len(v) for v in per_hop.values()) + out["not_comparable"]
    return out


def answers_agree(a: dict, b: dict) -> bool:
    """Final answers agree: parsed `predicted` equal after answer normalization
    (two unparsed answers, both None, count as agreeing)."""
    from lakv.qa_scoring import normalize_answer
    na = normalize_answer(a.get("predicted") or "")
    nb = normalize_answer(b.get("predicted") or "")
    return na == nb


def self_replay_gate(summary: dict, expect_pairs: int = 20, min_agree: int = 19) -> dict:
    """The self-replay pass/fail decision. Only answer agreement and cache-length
    comparability gate; cosine and the other numbers are diagnostics."""
    reasons = []
    if summary["not_comparable"] > 0:
        reasons.append(f"not_comparable = {summary['not_comparable']} > 0: the rebuilt cache length differs from the "
                       f"real cache, so the cache-length rule (cache_gen_len) is wrong")
    if summary["n_pairs"] != expect_pairs:
        reasons.append(f"{summary['n_pairs']} comparable samples, expected {expect_pairs} "
                       f"(skipped or missing samples, or a different --n_samples)")
    if summary["n_answers_agree"] < min_agree:
        reasons.append(f"only {summary['n_answers_agree']} of {summary['n_pairs']} final answers agree "
                       f"(need >= {min_agree})")
    if summary["n_rebuild_logs"] < 2 * summary["n_pairs"]:
        reasons.append(f"only {summary['n_rebuild_logs']} rebuild logs for {summary['n_pairs']} samples "
                       f"(expected 2 per sample)")
    return {"pass": not reasons, "reasons": reasons, "rule": f">= {min_agree} of {expect_pairs} answers agree "
            f"AND not_comparable == 0 (cosine is diagnostic only)"}


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
    ap.add_argument("--expect_pairs", type=int, default=20,
                    help="self-replay: number of samples that must be compared (default 20)")
    ap.add_argument("--min_agree", type=int, default=19,
                    help="self-replay: minimum number of agreeing final answers (default 19)")
    args = ap.parse_args()
    results = json.loads(Path(args.results_json).read_text(encoding="utf-8"))
    rc = 0
    if "A_audit_self_replay" in results and "A" in results:
        summ = summarize_self_replay(results)
        gate = self_replay_gate(summ, args.expect_pairs, args.min_agree)
        print("== self-replay validation ==")
        print(f"GATE: {'PASS' if gate['pass'] else 'FAIL'}   rule: {gate['rule']}")
        print(f"      answers agree: {summ['n_answers_agree']} of {summ['n_pairs']}   "
              f"not_comparable: {summ['not_comparable']}")
        for r in gate["reasons"]:
            print(f"      FAIL REASON: {r}")
        diag = {k: v for k, v in summ.items() if k not in ("n_answers_agree", "not_comparable")}
        print("DIAGNOSTICS (non-gating; cosine / abs diff / rel L2 of rebuilt vs real caches, raw-text identity, McNemar):")
        print(json.dumps(diag, indent=2, default=str))
        if not gate["pass"]:
            rc = 1
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
