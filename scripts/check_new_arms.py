"""Checker for the corruption (real_/lm_ operators), fan-in and RLM+KV length-matched arms.

Usage:  python scripts/check_new_arms.py <experiment_results.json | transcripts.json> [--max_skip 0.25]
Exit 0 = all invariants hold; 1 = a violation (printed). UNTESTED ON GPU (CPU tests only).

Checks
  experiment_results.json (chain, run.py):
    corruption arms  -> every audit log's n_modified / length_after equal the operator's definition,
                        a crc32 sample_seed and cache_checksum are present, lm_ arms carry a donor log
                        with pad_positions == truncated_positions == 0.
  transcripts.json from recursive_poc_check.py (fan-in) and rlm_repl_kv_check.py (delegation):
    length_matched   -> pad_positions == 0 and truncated_positions == 0, donor length within 5% of the
                        target (fan-in: whole child cache; RLM: answer region, via real_answer_len),
                        RLM logs carry donor_prompt_len; skip rate reported (flag above --max_skip).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lakv.cache_corruption import is_corruption_mode, modified_positions, parse_corruption_mode

TOL = 0.05


def check_chain(d, problems):
    summary = {}
    for cfg, body in d.items():
        if not isinstance(body, dict) or "per_sample" not in body:
            continue
        n, n_corr = 0, 0
        for s in body["per_sample"]:
            for log in s.get("audit_logs", []):
                mode = log.get("mode", "")
                if not is_corruption_mode(mode):
                    continue
                n_corr += 1
                target, op, pct = parse_corruption_mode(mode)
                m, start, end = modified_positions(op, log["length_before"], pct)
                want_len = {"trunc": log["length_before"] - m, "pad": log["length_before"] + m}.get(op, log["length_before"])
                if log["n_modified"] != m or log["length_after"] != want_len or (log["modified_start"], log["modified_end"]) != (start, end):
                    problems.append(f"{cfg} idx {s['idx']}: operator output does not match its definition: {log}")
                if "sample_seed" not in log or "cache_checksum" not in log:
                    problems.append(f"{cfg} idx {s['idx']}: missing sample_seed/cache_checksum")
                if target == "lm":
                    dn = log.get("donor", {})
                    if dn.get("pad_positions") != 0 or dn.get("truncated_positions") != 0:
                        problems.append(f"{cfg} idx {s['idx']}: lm donor was length-adjusted: {dn}")
                    if abs(dn.get("len_ratio", 1.0) - 1.0) > TOL + 1e-9:
                        problems.append(f"{cfg} idx {s['idx']}: lm donor outside 5%: {dn.get('len_ratio')}")
            n += 1
        summary[cfg] = (n, n_corr)
    return summary


def check_transcripts(d, problems, max_skip):
    summary = {}
    for chan, recs in d["records"].items():
        if "length_matched" not in chan:
            continue
        skipped = sum(1 for r in recs if r.get("skipped"))
        subs = 0
        for r in recs:
            for log in r.get("audit_logs", []):
                if log.get("mode") != "length_matched" or log.get("skipped"):
                    continue
                subs += 1
                if log.get("pad_positions") != 0 or log.get("truncated_positions") != 0:
                    problems.append(f"{chan} idx {r['idx']}: donor was length-adjusted: {log}")
                target = log.get("real_answer_len", log.get("target_cache_len"))
                if target and abs(log["donor_cache_len"] - target) > TOL * target + 1e-9:
                    problems.append(f"{chan} idx {r['idx']}: donor length {log['donor_cache_len']} not within 5% of {target}")
                if "real_answer_len" in log and "donor_prompt_len" not in log:
                    problems.append(f"{chan} idx {r['idx']}: RLM log without donor_prompt_len")
        rate = skipped / max(len(recs), 1)
        summary[chan] = (len(recs), skipped, subs, rate)
        if rate > max_skip:
            problems.append(f"{chan}: skip rate {rate:.0%} above {max_skip:.0%}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--max_skip", type=float, default=0.25)
    args = ap.parse_args()
    d = json.loads(Path(args.path).read_text(encoding="utf-8"))
    problems = []
    if "records" in d:
        summary = check_transcripts(d, problems, args.max_skip)
        for c, (n, sk, subs, rate) in summary.items():
            print(f"{c}: {n} records, {sk} skipped ({rate:.0%}), {subs} substitutions")
    else:
        summary = check_chain(d, problems)
        for c, (n, ncorr) in summary.items():
            if ncorr:
                print(f"{c}: {n} samples, {ncorr} corruption substitutions checked")
    for p in problems:
        print("VIOLATION:", p)
    print("RESULT:", "PASS" if not problems else "FAIL")
    sys.exit(0 if not problems else 1)


if __name__ == "__main__":
    main()
