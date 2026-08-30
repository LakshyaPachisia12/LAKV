"""
LAKV-v2 Benchmark Runner
========================

Implements the 5-phase evaluation plan:

  Phase 1: Single-agent baseline (n=20) — establish ceiling
  Phase 2: v2 Config A_full  (full KV, no selection, no compression)
  Phase 3: v2 Config C_full  (layer selection, no compression)
  Phase 4: v2 Config D_full  (layer selection + adaptive compression)
  Phase 5: Full benchmark (n=100) on best configs + baselines

Rows evaluated:

  single_agent        — SingleAgentPipeline (ceiling)
  text_2agent         — TextTwoAgentPipeline (text relay, no KV)
  v1_A_legacy         — Old pipeline Config A (historical comparison)
  v2_A_full           — v2, full KV, no selection, no compression
  v2_A_tail           — v2, tail KV, no selection, no compression
  v2_C_full           — v2, full KV, layer selection, no compression
  v2_D_full           — v2, full KV, layer selection, adaptive compression
  v2_D_tail           — v2, tail KV, layer selection, adaptive compression

Usage:
  python lakv_v2/benchmark.py \\
      --profile_path profiles/run_<ts>/qwen_gsm8k.json \\
      --output_dir results/v2_run_<ts> \\
      --phase all

  # Or step by step:
  python lakv_v2/benchmark.py --profile_path ... --phase 1
  python lakv_v2/benchmark.py --profile_path ... --phase 2 --output_dir results/v2_run_<ts>
"""

from __future__ import annotations

import argparse
import csv
import copy
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# ── path setup ─────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from lakv_v2.pipeline.shared_prefix_pipeline import (
    SharedPrefixConfig,
    SharedPrefixPipeline,
    SharedPrefixResult,
    DEFAULT_FINALIZER_SUFFIX,
    VERIFY_FINALIZER_SUFFIX,
)
from lakv_v2.pipeline.single_agent import SingleAgentPipeline, SingleAgentPipelineConfig
from lakv.anchor_table import AnchorTable


# ── answer parsing ─────────────────────────────────────────────────────────────

def extract_answer(text: str) -> Optional[str]:
    """Extract numeric answer. Prefer #### format; fall back progressively."""
    if not text:
        return None
    # Strip non-numeric chars between digits (e.g. LaTeX braces) but NOT spaces —
    # stripping spaces causes "4 × 7" → "47" which breaks extraction.
    cleaned = re.sub(r"(?<=\d)[^\d,.\-\s]+(?=\d)", "", text)
    for candidate in (text, cleaned):
        m = re.search(r"####\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)", candidate)
        if m:
            return m.group(1).replace(",", "").rstrip(".")
    for candidate in (text, cleaned):
        m = re.search(r"(?im)^\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)\s*\.?\s*$", candidate)
        if m:
            return m.group(1).replace(",", "").rstrip(".")
    for candidate in (text, cleaned):
        m = re.search(r"(?i)answer\s*(?:is|=|:)?\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)", candidate)
        if m:
            return m.group(1).replace(",", "").rstrip(".")
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", cleaned)
    return numbers[-1].replace(",", "").rstrip(".") if numbers else None


def is_malformed(text: str) -> bool:
    if not text:
        return True
    if re.search(r"[!?.]{4,}", text):
        return True
    # Detect long runs of a single non-space character (digit spam / symbol spam)
    if re.search(r"(.)\1{5,}", text):
        return True
    return False


# ── pipeline factories ─────────────────────────────────────────────────────────

def _load_profile(profile_path: str):
    from lakv.calibration_profiler import LayerProfile
    return LayerProfile.load(profile_path)


def _build_selector(profile_path: str):
    from lakv.layer_selector import LayerSelector
    profile = _load_profile(profile_path)
    return LayerSelector(profile)


def build_pipeline(row_name: str, model, tokenizer, profile_path: str, device: str,
                   debug: bool = False, use_offset_correction: bool = False):
    """
    Factory: returns (pipeline, pipe_type).

    pipe_type is one of: 'single' | 'text2agent' | 'v1' | 'v2'

    use_offset_correction: mirrors v1 PipelineConfig.use_offset_correction. Off by
    default so existing behavior (no AnchorTable, no correction) doesn't silently
    change. Only applies to 'v2' rows — single/text2agent/v1 rows are unaffected
    (v1's own use_offset_correction stays hardcoded False below; frozen row).
    """
    if row_name == "single_agent":
        cfg = SingleAgentPipelineConfig(
            system_prompt=(
                "You are a mathematical reasoning assistant. "
                "Solve carefully and return the final answer as #### [number]."
            ),
        )
        return SingleAgentPipeline(model, tokenizer, cfg, device), "single"

    if row_name == "text_2agent":
        from run_best import TextTwoAgentPipeline
        return TextTwoAgentPipeline(model, tokenizer, device), "text2agent"

    if row_name in ("v1_full_kv", "v1_A_legacy"):
        # V1 reference: 3-agent relay, full KV, no selection.
        # Frozen — do not modify pipeline.py for this row.
        from lakv.pipeline import LAKVPipeline, PipelineConfig
        cfg = PipelineConfig(
            use_layer_selection=False,
            compression_mode="none",
            use_offset_correction=False,
            reconstruction_strategy="zeros",
            n_agents=3,
        )
        return LAKVPipeline(model, tokenizer, cfg, device), "v1"

    # ── v2 configs ────────────────────────────────────────────────────────
    selector = None

    # ── v2 protocol rows (renamed from legacy letters) ────────────────────
    if row_name == "v2_full_prefix":
        cfg = SharedPrefixConfig(
            transfer_mode="full",
            use_layer_selection=False,
            compression_mode="none",
            solver_max_new_tokens=512,
            finalizer_max_new_tokens=48,
        )

    elif row_name == "v2_tail_only":
        cfg = SharedPrefixConfig(
            transfer_mode="tail",
            use_layer_selection=False,
            compression_mode="none",
            solver_max_new_tokens=512,
            finalizer_max_new_tokens=48,
        )

    elif row_name == "v2_layer_select":
        cfg = SharedPrefixConfig(
            transfer_mode="full",
            use_layer_selection=True,
            compression_mode="none",
            reconstruction_strategy="nearest",
            profile_path=profile_path,
            solver_max_new_tokens=512,
            finalizer_max_new_tokens=48,
        )

    elif row_name == "v2_compressed":
        cfg = SharedPrefixConfig(
            transfer_mode="full",
            use_layer_selection=True,
            compression_mode="adaptive",
            reconstruction_strategy="nearest",
            profile_path=profile_path,
            solver_max_new_tokens=512,
            finalizer_max_new_tokens=48,
        )

    elif row_name == "v2_compressed_tail":
        cfg = SharedPrefixConfig(
            transfer_mode="tail",
            use_layer_selection=True,
            compression_mode="adaptive",
            reconstruction_strategy="nearest",
            profile_path=profile_path,
            solver_max_new_tokens=512,
            finalizer_max_new_tokens=48,
        )

    # ── ablations / legacy names (kept for backward compatibility) ─────────
    elif row_name in ("v2_A_full", "v2_full_prefix_legacy"):
        cfg = SharedPrefixConfig(
            transfer_mode="full",
            use_layer_selection=False,
            compression_mode="none",
            solver_max_new_tokens=512,
            finalizer_max_new_tokens=48,
        )

    elif row_name == "v2_full_prefix_96":
        cfg = SharedPrefixConfig(
            transfer_mode="full",
            use_layer_selection=False,
            compression_mode="none",
            solver_max_new_tokens=96,
            finalizer_max_new_tokens=48,
        )

    elif row_name in ("v2_A_tail",):
        cfg = SharedPrefixConfig(
            transfer_mode="tail",
            use_layer_selection=False,
            compression_mode="none",
            solver_max_new_tokens=512,
            finalizer_max_new_tokens=48,
        )

    elif row_name in ("v2_C_full", "v2_layer_select_nearest"):
        cfg = SharedPrefixConfig(
            transfer_mode="full",
            use_layer_selection=True,
            compression_mode="none",
            reconstruction_strategy="nearest",
            profile_path=profile_path,
            solver_max_new_tokens=512,
            finalizer_max_new_tokens=48,
        )

    elif row_name in ("v2_D_full",):
        cfg = SharedPrefixConfig(
            transfer_mode="full",
            use_layer_selection=True,
            compression_mode="adaptive",
            reconstruction_strategy="nearest",
            profile_path=profile_path,
            solver_max_new_tokens=512,
            finalizer_max_new_tokens=48,
        )

    elif row_name in ("v2_D_tail",):
        cfg = SharedPrefixConfig(
            transfer_mode="tail",
            use_layer_selection=True,
            compression_mode="adaptive",
            reconstruction_strategy="nearest",
            profile_path=profile_path,
            solver_max_new_tokens=512,
            finalizer_max_new_tokens=48,
        )

    elif row_name in ("v2_A_verify", "v2_full_prefix_verify"):
        cfg = SharedPrefixConfig(
            transfer_mode="full",
            use_layer_selection=False,
            compression_mode="none",
            finalizer_suffix=VERIFY_FINALIZER_SUFFIX,
            solver_max_new_tokens=512,
            finalizer_max_new_tokens=48,
        )

    else:
        raise ValueError(f"Unknown row: {row_name!r}")

    cfg.debug = debug
    cfg.use_offset_correction = use_offset_correction
    if use_offset_correction:
        # Fresh AnchorTable per row so hit/admission stats are scoped to this
        # row's dataset pass, not shared across unrelated rows/configs.
        cfg.anchor_table = AnchorTable(max_size=20, entropy_threshold=0.3, verbose=debug)
    return SharedPrefixPipeline(model, tokenizer, cfg, device), "v2"


# ── single sample evaluation ───────────────────────────────────────────────────

def eval_sample(
    pipeline, pipe_type: str, question: str, gold: str, tokenizer
) -> dict:
    t0 = time.time()

    if pipe_type == "single":
        raw_answer = pipeline.run(question)
        latency = time.time() - t0
        return _make_record(
            question, gold, raw_answer, latency, tokenizer,
            transfer_mode="-", compression_ratio=0.0,
            n_layers_transferred=0, original_mb=0.0, compressed_mb=0.0,
            prefix_len=0, solver_gen_len=0,
        )

    if pipe_type == "text2agent":
        out = pipeline.run(question)
        raw_answer = out["answer"]
        latency = time.time() - t0
        return _make_record(
            question, gold, raw_answer, latency, tokenizer,
            transfer_mode="text", compression_ratio=0.0,
            n_layers_transferred=0, original_mb=0.0, compressed_mb=0.0,
            prefix_len=0, solver_gen_len=0,
        )

    if pipe_type == "v1":
        r = pipeline.run(question)
        latency = time.time() - t0
        n_hops = max(len(r.hop_stats), 1)
        return _make_record(
            question, gold, r.answer, latency, tokenizer,
            transfer_mode="v1_full", compression_ratio=r.overall_compression_ratio,
            n_layers_transferred=sum(h.n_layers_transmitted for h in r.hop_stats) // n_hops,
            original_mb=r.total_original_mb, compressed_mb=r.total_compressed_mb,
            prefix_len=0, solver_gen_len=0,
        )

    # v2
    r: SharedPrefixResult = pipeline.run(question)
    latency = time.time() - t0
    return _make_record(
        question, gold, r.answer, latency, tokenizer,
        transfer_mode=r.transfer_mode,
        compression_ratio=r.compression_ratio,
        n_layers_transferred=r.n_layers_transferred,
        original_mb=r.original_mb,
        compressed_mb=r.compressed_mb,
        prefix_len=r.prefix_len,
        solver_gen_len=r.solver_gen_len,
        solver_reasoning=r.solver_reasoning,
        cache_seq_len=r.cache_seq_len,
        anchor_hit=r.anchor_hit,
        anchor_confidence=r.anchor_confidence,
    )


def _make_record(
    question, gold, raw_answer, latency, tokenizer,
    transfer_mode, compression_ratio, n_layers_transferred,
    original_mb, compressed_mb, prefix_len, solver_gen_len,
    solver_reasoning="",
    cache_seq_len: int = 0,
    anchor_hit: bool = False,
    anchor_confidence: float = 0.0,
) -> dict:
    pred = extract_answer(raw_answer)
    correct = pred is not None and pred.strip() == gold.strip()
    malformed = is_malformed(raw_answer)
    gen_tokens = len(tokenizer(raw_answer or "", add_special_tokens=False)["input_ids"])
    return {
        "question": question,
        "gold": gold,
        "raw_answer": raw_answer,
        "predicted": pred,
        "correct": correct,
        "parse_failed": pred is None,
        "malformed": malformed,
        "latency_s": round(latency, 3),
        "transfer_mode": transfer_mode,
        "compression_ratio": round(compression_ratio, 4),
        "n_layers_transferred": n_layers_transferred,
        "original_mb": round(original_mb, 4),
        "compressed_mb": round(compressed_mb, 4),
        "prefix_len": prefix_len,
        "solver_gen_len": solver_gen_len,
        "generated_tokens": gen_tokens,
        "solver_reasoning": solver_reasoning,
        "cache_seq_len": cache_seq_len,
        "anchor_hit": anchor_hit,
        "anchor_confidence": anchor_confidence,
    }


_DEBUG_SEP = "─" * 62


def _print_debug_block(idx: int, rec: dict) -> None:
    """Print one-sample debug block. Pipeline has already printed the top half;
    this function prints the verdict line and closing separator."""
    mark = "✓" if rec["correct"] else "✗"
    verdict = "CORRECT" if rec["correct"] else "INCORRECT"
    print(f"  Parsed   : {rec['predicted']}   Gold: {rec['gold']}   {mark} {verdict}")
    print(_DEBUG_SEP)


# ── row evaluation ─────────────────────────────────────────────────────────────

def eval_row(
    row_name: str,
    pipeline,
    pipe_type: str,
    dataset: List[dict],
    tokenizer,
    print_raw: bool = False,
    debug: bool = False,
) -> Tuple[dict, List[dict]]:
    records = []
    for idx, sample in enumerate(dataset):
        rec = eval_sample(pipeline, pipe_type, sample["question"], sample["answer"], tokenizer)
        records.append(rec)
        if debug and pipe_type == "v2":
            # pipeline already printed the top half inside run() when cfg.debug=True
            _print_debug_block(idx, rec)
        elif print_raw:
            print(f"  Q: {sample['question'][:60]}...")
            print(f"  Raw: {rec['raw_answer'][:120]}")
            print(f"  Pred={rec['predicted']} Gold={sample['answer']} ✓={rec['correct']}")
            print()

    n = len(records)
    summary = {
        "config": row_name,
        "n_samples": n,
        "accuracy": sum(r["correct"] for r in records) / n if n else 0.0,
        "parse_failures": sum(r["parse_failed"] for r in records),
        "malformed_rate": sum(r["malformed"] for r in records) / n if n else 0.0,
        "mean_latency_s": sum(r["latency_s"] for r in records) / n if n else 0.0,
        "mean_compression_ratio": sum(r["compression_ratio"] for r in records) / n if n else 0.0,
        "mean_layers_transferred": sum(r["n_layers_transferred"] for r in records) / n if n else 0.0,
        "mean_original_mb": sum(r["original_mb"] for r in records) / n if n else 0.0,
        "mean_compressed_mb": sum(r["compressed_mb"] for r in records) / n if n else 0.0,
        "mean_generated_tokens": sum(r["generated_tokens"] for r in records) / n if n else 0.0,
        "mean_solver_gen_len": sum(r["solver_gen_len"] for r in records) / n if n else 0.0,
    }
    return summary, records


# ── stage runner ───────────────────────────────────────────────────────────────

def run_stage(
    stage_name: str,
    row_names: List[str],
    dataset: List[dict],
    model,
    tokenizer,
    profile_path: str,
    device: str,
    output_dir: Path,
    print_raw: bool = False,
    debug: bool = False,
    use_offset_correction: bool = False,
) -> Dict[str, dict]:
    print(f"\n{'='*64}")
    print(f"  {stage_name}  (n={len(dataset)})")
    print(f"{'='*64}")

    all_summaries = {}
    all_records = {}

    for row_name in row_names:
        print(f"\n[{row_name}] Building pipeline...")
        try:
            pipeline, pipe_type = build_pipeline(
                row_name, model, tokenizer, profile_path, device, debug=debug,
                use_offset_correction=use_offset_correction,
            )
        except Exception as e:
            print(f"  ✗ Failed to build pipeline: {e}")
            continue

        print(f"[{row_name}] Running {len(dataset)} samples...")
        try:
            summary, records = eval_row(
                row_name, pipeline, pipe_type, dataset, tokenizer,
                print_raw=print_raw, debug=debug,
            )
        except Exception as e:
            import traceback
            print(f"  ✗ Pipeline error: {e}")
            traceback.print_exc()
            continue

        all_summaries[row_name] = summary
        all_records[row_name] = records
        _print_row(summary)

        anchor_table = getattr(pipeline.config, "anchor_table", None) if hasattr(pipeline, "config") else None
        if anchor_table is not None:
            print(f"  [AnchorTable summary for {row_name}] {anchor_table.call_counts()}")

    # Save stage results
    stage_key = stage_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    _save_json(output_dir / f"{stage_key}_summaries.json", all_summaries)
    _save_json(output_dir / f"{stage_key}_records.json", all_records)
    _save_csv(output_dir / f"{stage_key}_table.csv", all_summaries)

    print(f"\n[Saved] {output_dir / stage_key}_*.json/.csv")
    return all_summaries


# ── printing / saving ──────────────────────────────────────────────────────────

def _print_row(s: dict) -> None:
    acc_str = f"{s['accuracy']*100:5.1f}%"
    pf_str  = f"PF={s['parse_failures']:3d}"
    mf_str  = f"Mal={s['malformed_rate']*100:4.1f}%"
    lat_str = f"Lat={s['mean_latency_s']:5.1f}s"
    ly_str  = f"Ly={s['mean_layers_transferred']:4.0f}/28"
    cr_str  = f"CR={s['mean_compression_ratio']:5.2f}x"
    print(f"  {s['config']:<22} | {acc_str} | {pf_str} | {mf_str} | {lat_str} | {ly_str} | {cr_str}")


def _print_summary_table(all_stages: Dict[str, Dict[str, dict]]) -> None:
    print(f"\n{'='*80}")
    print("  FINAL SUMMARY")
    print(f"{'='*80}")
    hdr = f"{'Config':<22} | {'n':>4} | {'Acc':>6} | {'PFail':>5} | {'Malform':>7} | {'Lat(s)':>6} | {'Ly/28':>5} | {'Comp':>5}"
    print(hdr)
    print("-" * len(hdr))
    seen = {}
    for stage_name, summaries in all_stages.items():
        for row_name, s in summaries.items():
            if row_name not in seen:
                seen[row_name] = s
                print(f"  {s['config']:<22} | {s['n_samples']:>4} | "
                      f"{s['accuracy']*100:>5.1f}% | {s['parse_failures']:>5} | "
                      f"{s['malformed_rate']*100:>6.1f}% | {s['mean_latency_s']:>6.1f} | "
                      f"{s['mean_layers_transferred']:>5.0f} | {s['mean_compression_ratio']:>5.2f}x")


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def _save_csv(path: Path, summaries: Dict[str, dict]) -> None:
    if not summaries:
        return
    fields = [
        "config", "n_samples", "accuracy", "parse_failures", "malformed_rate",
        "mean_latency_s", "mean_compression_ratio", "mean_layers_transferred",
        "mean_original_mb", "mean_compressed_mb", "mean_generated_tokens", "mean_solver_gen_len",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for s in summaries.values():
            w.writerow([s.get(field, "") for field in fields])


# ── dataset loader ─────────────────────────────────────────────────────────────

def load_gsm8k(n: int = None) -> List[dict]:
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main")
    data = []
    for item in ds["test"]:
        answer = item["answer"].split("####")[-1].strip()
        data.append({"question": item["question"], "answer": answer})
    return data[:n] if n is not None else data


def load_model(model_name: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[benchmark] Loading model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


# ── main ───────────────────────────────────────────────────────────────────────

# Family 1 — baselines (no profile needed)
PHASE1_ROWS = ["single_agent", "text_2agent"]

# Family 2 — V2 full/tail KV (no profile needed)
PHASE2_ROWS = ["v2_full_prefix", "v2_tail_only"]

# Family 2b — V2 layer selection (profile required)
PHASE3_ROWS = ["v2_layer_select"]

# Family 2c — V2 selection + compression (profile required)
PHASE4_ROWS = ["v2_compressed", "v2_compressed_tail"]

# Full comparison matrix (profile required for v2_layer_select / v2_compressed)
PHASE5_ROWS = [
    "single_agent",
    "text_2agent",
    "v2_full_prefix",
    "v2_tail_only",
    "v2_layer_select",
    "v2_compressed",
]

# V1 reference — run separately, never mix into PHASE5 accuracy table
PHASE_V1_ROWS = ["v1_full_kv"]


def main():
    parser = argparse.ArgumentParser(description="LAKV-v2 Benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--profile_path", default=None,
                        help="Path to LayerProfile JSON (required for layer_select/compressed rows)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (auto-timestamped if omitted)")
    parser.add_argument("--phase", default="all",
                        choices=["1", "2", "3", "4", "5", "all", "v1"],
                        help="Which phase to run (default: all). Use 'v1' for V1 reference family.")
    parser.add_argument("--n_phase1", type=int, default=20)
    parser.add_argument("--n_phase2", type=int, default=20)
    parser.add_argument("--n_phase3", type=int, default=20)
    parser.add_argument("--n_phase4", type=int, default=20)
    parser.add_argument("--n_phase5", type=int, default=100)
    parser.add_argument("--print_raw", action="store_true",
                        help="Print raw model outputs for each sample")
    parser.add_argument("--debug", action="store_true",
                        help="Print per-sample debug block: question, solver, relay stats, anchor, finalizer, verdict")
    parser.add_argument("--rows", nargs="+", default=None,
                        help="Override row list for a custom run")
    parser.add_argument("--use_offset_correction", action="store_true",
                        help="Enable AnchorTable cross-question KV correction on v2 rows "
                             "(off by default; mirrors v1 PipelineConfig.use_offset_correction)")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.output_dir or f"results/v2_run_{ts}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"[benchmark] Output dir: {out}")

    model, tokenizer = load_model(args.model, args.device)
    dataset = load_gsm8k(n=args.n_phase5)

    all_stages = {}

    def _run(stage_name, rows, n):
        summaries = run_stage(
            stage_name, rows, dataset[:n],
            model, tokenizer, args.profile_path, args.device,
            out, print_raw=args.print_raw, debug=args.debug,
            use_offset_correction=args.use_offset_correction,
        )
        all_stages[stage_name] = summaries
        return summaries

    custom_rows = args.rows

    if args.phase in ("1", "all"):
        rows = custom_rows or PHASE1_ROWS
        _run(f"Phase 1 — Baselines (n={args.n_phase1})", rows, args.n_phase1)

    if args.phase in ("2", "all"):
        rows = custom_rows or PHASE2_ROWS
        _run(f"Phase 2 — V2 Full/Tail KV (n={args.n_phase2})", rows, args.n_phase2)

    if args.phase in ("3", "all"):
        rows = custom_rows or PHASE3_ROWS
        _run(f"Phase 3 — V2 Layer Selection (n={args.n_phase3})", rows, args.n_phase3)

    if args.phase in ("4", "all"):
        rows = custom_rows or PHASE4_ROWS
        _run(f"Phase 4 — V2 Selection + Compression (n={args.n_phase4})", rows, args.n_phase4)

    if args.phase in ("5", "all"):
        rows = custom_rows or PHASE5_ROWS
        _run(f"Phase 5 — Full V2 Benchmark (n={args.n_phase5})", rows, args.n_phase5)

    if args.phase == "v1":
        # V1 reference family — separate output, not mixed with V2 results
        rows = custom_rows or PHASE_V1_ROWS
        n = args.n_phase1
        _run(f"V1 Reference — {rows} (n={n})", rows, n)
        print("\n[NOTE] V1 reference results are for architectural comparison only.")
        print("       Do not mix these numbers with V2 Family accuracy tables.")

    _print_summary_table(all_stages)
    _save_json(out / "all_summaries.json", all_stages)
    print(f"\n[benchmark] Complete. All results in: {out}/")


if __name__ == "__main__":
    main()
