"""How much of the `mismatched` audit arm was zero-padded? (CPU only, existing data.)

Older result files (before the instrumentation in lakv/causal_audit.py) log,
for every mismatched substitution, only `shape_resized` and the donor's full
question text. They do NOT store the donor's or target's cache length. This
script reconstructs what it can:

EXACT
  * shape_resized fraction (donor length != target length).
  * target cache length per hop = hop_stats[hop].original_mb*1e6 / bytes_per_position
    (bytes_per_position = layers * 2 * kv_heads * head_dim * 2 bytes, bf16).
    Hop 0 is never injected, so hop 0's cache length is the target for the
    agent_idx=1 substitution; hop 1's cache (as produced in this run) is the
    target for agent_idx=2. Integer divisibility is checked as a sanity test.
  * donor PROMPT length: tokenizer + the pipeline's own chat template applied
    to the logged donor question text.

ESTIMATED (labelled as such)
  * donor generated length is not stored anywhere. Hard bounds: 0..512 tokens
    per hop (intermediate_max_new_tokens). Point estimate: the median
    generated length of the scored questions in the same run
    (gen0 = L0 - P0, gen1 = L1 - L0 - P1; both exact per scored question).
  * donor cache length at agent_idx=1 = P0_donor + gen0;
    at agent_idx=2 = P0_donor + gen0 + P1_donor + gen1 (donor's Verifier cache
    in the unaudited chain).

Usage:  python scripts/analyze_mismatched_padding.py [--out report.md]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# run id -> (label, tokenizer repo, layers, kv_heads, head_dim)
MODELS = {
    28: ("Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct", 4, 128),
    32: ("Mistral-7B-v0.3", "mistralai/Mistral-7B-Instruct-v0.3", 8, 128),
    36: ("Qwen3-8B", "Qwen/Qwen3-8B", 8, 128),
}
RUNS = [
    ("20261009_113254", "primary"), ("20261011_150444", "donor draw"), ("20261011_184238", "donor draw"),
    ("20261011_141020", "primary"), ("20261011_234807", "donor draw"), ("20261012_093630", "donor draw"),
    ("20261011_221317", "primary"),
]
GEN_CAP = 512  # PipelineConfig.intermediate_max_new_tokens


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out", default=None)
    ap.add_argument("--local_files_only", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from lakv.pipeline import PROMPT_SETS
    prompts = PROMPT_SETS["hotpotqa"]
    toks = {}

    def prompt_len(tok, agent_idx, blob):
        sp = prompts[agent_idx] if agent_idx < len(prompts) else prompts[-1]
        text = tok.apply_chat_template(
            [{"role": "system", "content": sp}, {"role": "user", "content": blob}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return len(tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0])

    lines = ["# Mismatched-audit padding analysis (existing data, CPU only)", "",
             "EXACT = computed from stored bytes/flags. EST = depends on the donor's generated length, "
             "which no result file stores (bounds and a median-based estimate are given).", ""]
    summary_rows = []
    for run, kind in RUNS:
        d = json.loads((Path(args.results_dir) / f"run_{run}" / "experiment_results.json").read_text(encoding="utf-8"))
        cfg = next(c for c in d if c.endswith("_audit_mismatched"))
        ps = d[cfg]["per_sample"]
        layers = int(d[cfg]["summary"]["mean_layers_transmitted"])
        label, repo, kvh, hd = MODELS[layers]
        bpp = layers * 2 * kvh * hd * 2
        if repo not in toks:
            try:
                toks[repo] = AutoTokenizer.from_pretrained(repo, local_files_only=args.local_files_only)
            except Exception:
                # newer tokenizer.json formats fail to load with older `tokenizers`;
                # the slow (vocab.json + merges.txt) tokenizer gives the same token ids.
                toks[repo] = AutoTokenizer.from_pretrained(repo, local_files_only=args.local_files_only,
                                                           use_fast=False)
        tok = toks[repo]

        L0, L1, P0, P1, G0, G1, ok = [], [], [], [], [], [], []
        divisible = 0
        for s in ps:
            b0, b1 = s["hop_stats"][0]["original_mb"] * 1e6, s["hop_stats"][1]["original_mb"] * 1e6
            divisible += (round(b0) % bpp == 0) + (round(b1) % bpp == 0)
            l0, l1 = round(b0) // bpp, round(b1) // bpp
            p0, p1 = prompt_len(tok, 0, s["question"]), prompt_len(tok, 1, s["question"])
            L0.append(l0); L1.append(l1); P0.append(p0); P1.append(p1)
            G0.append(l0 - p0)            # hop 0 is never injected: exact
            G1.append(l1 - l0 - p1)       # injected cache has length l0 (matched): exact
            ok.append(bool(s["correct"]))
        g0m, g1m = st.median(G0), st.median(G1)

        for agent_idx in (1, 2):
            tgt, rows = [], []
            for s, l0, l1 in zip(ps, L0, L1):
                log = next(l for l in s["audit_logs"] if l.get("mode") == "mismatched" and l["agent_idx"] == agent_idx)
                donor = log["donor_question"]
                dp0, dp1 = prompt_len(tok, 0, donor), prompt_len(tok, 1, donor)
                target = l0 if agent_idx == 1 else l1
                lo = dp0 if agent_idx == 1 else dp0 + dp1
                hi = lo + GEN_CAP * agent_idx
                est = lo + (g0m if agent_idx == 1 else g0m + g1m)
                # plausible range from observed generated-length percentiles
                plo = lo + (pct(G0, .05) if agent_idx == 1 else pct(G0, .05) + pct(G1, .05))
                phi = lo + (pct(G0, .95) if agent_idx == 1 else pct(G0, .95) + pct(G1, .95))
                rows.append(dict(target=target, lo=lo, hi=hi, est=est, plo=plo, phi=phi,
                                 resized=log["shape_resized"]))
            n = len(rows)
            cert_pad = sum(r["target"] > r["hi"] for r in rows)
            cert_trunc = sum(r["target"] < r["lo"] for r in rows)
            est_pad = sum(r["target"] > r["est"] for r in rows)
            est_trunc = sum(r["target"] < r["est"] for r in rows)
            plaus_pad = sum(r["target"] > r["phi"] for r in rows)
            plaus_trunc = sum(r["target"] < r["plo"] for r in rows)
            padfrac = [max(r["target"] - r["est"], 0) / r["target"] for r in rows]
            padfrac_hi = [max(r["target"] - r["plo"], 0) / r["target"] for r in rows]
            padfrac_lo = [max(r["target"] - r["phi"], 0) / r["target"] for r in rows]
            summary_rows.append(dict(
                run=run, kind=kind, model=label, agent_idx=agent_idx, n=n,
                resized_exact=sum(r["resized"] for r in rows) / n,
                target_mean=st.mean(r["target"] for r in rows),
                donor_est_mean=st.mean(r["est"] for r in rows),
                cert_pad=cert_pad / n, cert_trunc=cert_trunc / n,
                est_pad=est_pad / n, est_trunc=est_trunc / n,
                plaus_pad=plaus_pad / n, plaus_trunc=plaus_trunc / n,
                padfrac_est=st.mean(padfrac), padfrac_lo=st.mean(padfrac_lo), padfrac_hi=st.mean(padfrac_hi),
            ))
        lines.append(f"- {label} run_{run} ({kind}): n={len(ps)}, bytes/position={bpp}, "
                     f"cache lengths integer-divisible: {divisible}/{2*len(ps)}; scored-question generated "
                     f"tokens hop0 median {g0m} (p5-p95 {pct(G0,.05)}-{pct(G0,.95)}), hop1 median {g1m} "
                     f"(p5-p95 {pct(G1,.05)}-{pct(G1,.95)}).")

    hdr = ["run", "kind", "model", "agent_idx", "resized (EXACT)", "target len mean (EXACT)", "donor len mean (EST)",
           "certainly padded", "certainly truncated", "padded (EST)", "truncated (EST)",
           "padded (plausible p5-95)", "truncated (plausible)", "pad fraction EST [lo,hi]"]
    lines += ["", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in summary_rows:
        lines.append(
            f"| {r['run']} | {r['kind']} | {r['model']} | {r['agent_idx']} | {r['resized_exact']:.0%} | "
            f"{r['target_mean']:.0f} | {r['donor_est_mean']:.0f} | {r['cert_pad']:.0%} | {r['cert_trunc']:.0%} | "
            f"{r['est_pad']:.0%} | {r['est_trunc']:.0%} | {r['plaus_pad']:.0%} | {r['plaus_trunc']:.0%} | "
            f"{r['padfrac_est']:.1%} [{r['padfrac_lo']:.1%}, {r['padfrac_hi']:.1%}] |")
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
