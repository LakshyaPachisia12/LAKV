"""
Benchmark-first runner for LAKV v1.5 experiments.

Stages:
  1) n=20 smoke on all required rows
  2) n=50 shortlist comparison
  3) n=100 final benchmark on best rows

Required rows:
  - single_agent
  - TEXT_2AGENT
  - A_legacy
  - A_2agent
  - C_legacy
  - C_2agent
  - D_legacy
  - D_2agent
  - best_anchor_2agent
"""

import argparse
import copy
import csv
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

from evaluator import PRESETS, extract_answer
from lakv_v2.pipeline.single_agent import SingleAgentPipeline, SingleAgentPipelineConfig
from pipeline import LAKVPipeline, PipelineConfig
from run import load_gsm8k, load_model


TWO_AGENT_PROMPTS = [
    (
        "You are a careful mathematical reasoning agent.\n"
        "Solve step-by-step internally.\n"
        "Be accurate."
    ),
    (
        "You have access to previous reasoning memory.\n"
        "Use it to solve the problem.\n\n"
        "Return ONLY:\n\n"
        "#### number"
    ),
]

FINAL_FORMAT_PROMPT = (
    "You are the final answer agent. "
    "Return ONLY one line in this format: #### number"
)

DETERMINISTIC_KWARGS = {
    "do_sample": False,
    "temperature": 0.0,
    "top_p": 1.0,
    "num_beams": 1,
}

STRICT_FORMAT_RE = re.compile(r"(?im)^\s*####\s*\$?\s*-?\d[\d,]*(?:\.\d+)?\s*\.?\s*$")


class TextTwoAgentPipeline:
    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def run(self, question: str) -> Dict[str, str]:
        solver_prompt = TWO_AGENT_PROMPTS[0]
        finalizer_prompt = TWO_AGENT_PROMPTS[1]

        solver_messages = [
            {"role": "system", "content": solver_prompt},
            {"role": "user", "content": question},
        ]
        solver_text = self.tokenizer.apply_chat_template(
            solver_messages, tokenize=False, add_generation_prompt=True
        )
        solver_ids = self.tokenizer(
            solver_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(self.device)

        with torch.no_grad():
            solver_out = self.model.generate(
                input_ids=solver_ids,
                max_new_tokens=64,
                **DETERMINISTIC_KWARGS,
            )
        solver_new = solver_out[0, solver_ids.shape[1]:]
        solver_reasoning = self.tokenizer.decode(solver_new, skip_special_tokens=True)

        final_user = (
            f"Problem:\n{question}\n\n"
            f"Previous reasoning memory:\n{solver_reasoning}"
        )
        final_messages = [
            {"role": "system", "content": finalizer_prompt},
            {"role": "user", "content": final_user},
        ]
        final_text = self.tokenizer.apply_chat_template(
            final_messages, tokenize=False, add_generation_prompt=True
        )
        final_ids = self.tokenizer(
            final_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(self.device)

        with torch.no_grad():
            final_out = self.model.generate(
                input_ids=final_ids,
                max_new_tokens=16,
                **DETERMINISTIC_KWARGS,
            )
        final_new = final_out[0, final_ids.shape[1]:]
        answer = self.tokenizer.decode(final_new, skip_special_tokens=True)

        return {
            "answer": answer,
            "solver_reasoning": solver_reasoning,
        }


# torch is imported lazily to avoid accidental startup overhead before model loading.
import torch


def build_lakv_config(name: str, profile_path: str) -> PipelineConfig:
    if name.startswith("A_"):
        cfg = copy.deepcopy(PRESETS["A"])
    elif name.startswith("C_"):
        cfg = copy.deepcopy(PRESETS["C"])
    elif name.startswith("D_"):
        cfg = copy.deepcopy(PRESETS["D"])
    elif name == "best_anchor_2agent":
        cfg = copy.deepcopy(PRESETS["D"])
        cfg.use_offset_correction = True
    else:
        raise ValueError(f"Unsupported LAKV config name: {name}")

    if cfg.use_layer_selection or cfg.compression_mode == "adaptive":
        cfg.profile_path = profile_path

    cfg.generation_kwargs = dict(DETERMINISTIC_KWARGS)

    if name.endswith("_2agent") or name == "best_anchor_2agent":
        cfg.n_agents = 2
        cfg.system_prompts = list(TWO_AGENT_PROMPTS)
        cfg.intermediate_max_new_tokens = 64
        cfg.final_max_new_tokens = 16
    else:
        # Keep legacy architecture, but enforce strict final formatting for benchmark discipline.
        cfg.n_agents = 3
        prompts = list(cfg.system_prompts)
        if prompts:
            prompts[-1] = FINAL_FORMAT_PROMPT
        cfg.system_prompts = prompts
        cfg.intermediate_max_new_tokens = 64
        cfg.final_max_new_tokens = 16

    return cfg


def is_malformed_output(raw_answer: str) -> bool:
    return not bool(STRICT_FORMAT_RE.search((raw_answer or "").strip()))


def safe_mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def evaluate_row(row_name: str, pipeline, pipe_type: str, dataset: List[dict], tokenizer) -> Tuple[dict, List[dict]]:
    per_sample = []

    for idx, sample in enumerate(dataset):
        question = sample["question"]
        gold = str(sample["answer"]).strip()

        anchor_hit = False
        compression_ratio = 0.0
        layers_sent = 0.0

        t0 = time.time()
        if pipe_type == "single":
            raw_answer = pipeline.run(question)
        elif pipe_type == "text2agent":
            out = pipeline.run(question)
            raw_answer = out["answer"]
        else:
            hit_before = 0
            if getattr(pipeline, "anchor_table", None) is not None:
                hit_before = len(pipeline.anchor_table.hit_log())

            run_result = pipeline.run(question)
            raw_answer = run_result.answer
            compression_ratio = run_result.overall_compression_ratio
            if run_result.hop_stats:
                layers_sent = sum(h.n_layers_transmitted for h in run_result.hop_stats) / len(run_result.hop_stats)

            if getattr(pipeline, "anchor_table", None) is not None:
                hit_after = len(pipeline.anchor_table.hit_log())
                if hit_after > hit_before:
                    new_hits = pipeline.anchor_table.hit_log()[hit_before:hit_after]
                    anchor_hit = any(entry.get("hit", False) for entry in new_hits)

        latency = time.time() - t0

        pred = extract_answer(raw_answer)
        correct = (pred is not None) and (pred.strip() == gold)
        parse_failed = pred is None
        malformed = is_malformed_output(raw_answer)
        generated_tokens = len(tokenizer(raw_answer or "", add_special_tokens=False)["input_ids"])

        sample_row = {
            "idx": idx,
            "question": question,
            "gold": gold,
            "predicted": pred,
            "raw_answer": raw_answer,
            "correct": correct,
            "parse_failed": parse_failed,
            "malformed": malformed,
            "latency_s": latency,
            "compression_ratio": compression_ratio,
            "layers_sent": layers_sent,
            "anchor_hit": anchor_hit,
            "generated_tokens": generated_tokens,
        }

        if hasattr(pipeline, "last_run_offset_logs") and pipeline.last_run_offset_logs:
            sample_row["offset_log"] = pipeline.last_run_offset_logs[-1]

        per_sample.append(sample_row)

    n = len(per_sample)
    parse_failures = sum(1 for s in per_sample if s["parse_failed"])
    malformed_count = sum(1 for s in per_sample if s["malformed"])

    summary = {
        "config": row_name,
        "accuracy": safe_mean([1.0 if s["correct"] else 0.0 for s in per_sample]),
        "parse_failures": parse_failures,
        "malformed_output_rate": malformed_count / n if n else 0.0,
        "mean_latency": safe_mean([s["latency_s"] for s in per_sample]),
        "compression_ratio": safe_mean([s["compression_ratio"] for s in per_sample]),
        "layers_sent": safe_mean([s["layers_sent"] for s in per_sample]),
        "anchor_hit_rate": safe_mean([1.0 if s["anchor_hit"] else 0.0 for s in per_sample]),
        "avg_generated_tokens": safe_mean([float(s["generated_tokens"]) for s in per_sample]),
        "n_samples": n,
    }
    return summary, per_sample


def write_stage_csv(path: Path, summaries: Dict[str, dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "config",
            "accuracy",
            "parse_failures",
            "malformed_output_rate",
            "mean_latency",
            "compression_ratio",
            "layers_sent",
            "anchor_hit_rate",
            "avg_generated_tokens",
        ])
        for cfg_name, s in summaries.items():
            writer.writerow([
                cfg_name,
                f"{s['accuracy']:.4f}",
                int(s["parse_failures"]),
                f"{s['malformed_output_rate']:.4f}",
                f"{s['mean_latency']:.4f}",
                f"{s['compression_ratio']:.4f}",
                f"{s['layers_sent']:.4f}",
                f"{s['anchor_hit_rate']:.4f}",
                f"{s['avg_generated_tokens']:.2f}",
            ])


def print_prompt_validation(rows: List[str], dataset: List[dict], model, tokenizer, profile_path: str, device: str) -> None:
    print("\n=== Prompt Validation (5 samples) ===")
    for row in rows:
        print(f"\n[{row}]")
        pipeline, pipe_type = build_pipeline(row, model, tokenizer, profile_path, device)
        for i, sample in enumerate(dataset[:5]):
            if pipe_type == "single":
                raw = pipeline.run(sample["question"])
            elif pipe_type == "text2agent":
                raw = pipeline.run(sample["question"])["answer"]
            else:
                raw = pipeline.run(sample["question"]).answer
            print(f"Q{i}: {raw.strip()}")


def build_pipeline(row_name: str, model, tokenizer, profile_path: str, device: str):
    if row_name == "single_agent":
        cfg = SingleAgentPipelineConfig(
            system_prompt=(
                "You are a mathematical reasoning assistant. "
                "Return ONLY one line in this format: #### number"
            )
        )
        return SingleAgentPipeline(model, tokenizer, cfg, device), "single"

    if row_name == "TEXT_2AGENT":
        return TextTwoAgentPipeline(model, tokenizer, device), "text2agent"

    cfg = build_lakv_config(row_name, profile_path)
    return LAKVPipeline(model, tokenizer, cfg, device), "lakv"


def run_stage(
    stage_name: str,
    row_names: List[str],
    dataset: List[dict],
    model,
    tokenizer,
    profile_path: str,
    device: str,
) -> Tuple[Dict[str, dict], Dict[str, List[dict]]]:
    print(f"\n=== {stage_name}: n={len(dataset)} ===")
    summaries = {}
    details = {}
    for row in row_names:
        pipeline, pipe_type = build_pipeline(row, model, tokenizer, profile_path, device)
        summary, per_sample = evaluate_row(row, pipeline, pipe_type, dataset, tokenizer)
        summaries[row] = summary
        details[row] = per_sample
        print(
            f"{row:<20} | Acc: {summary['accuracy']*100:>5.1f}% | "
            f"ParseFail: {summary['parse_failures']:>3d} | "
            f"Malformed: {summary['malformed_output_rate']*100:>5.1f}% | "
            f"Lat: {summary['mean_latency']:.2f}s"
        )
    return summaries, details


def choose_stage2_rows(stage1: Dict[str, dict], pause_two_agent: bool, deprioritize_anchor: bool) -> List[str]:
    allowed = list(stage1.keys())
    if pause_two_agent:
        allowed = [r for r in allowed if not r.endswith("_2agent") and r != "best_anchor_2agent"]
    if deprioritize_anchor:
        allowed = [r for r in allowed if r != "best_anchor_2agent"]

    ranked = sorted(allowed, key=lambda r: stage1[r]["accuracy"], reverse=True)
    fixed = ["single_agent", "TEXT_2AGENT", "C_legacy"]
    if not pause_two_agent and "C_2agent" in ranked:
        fixed.append("C_2agent")

    rows = []
    for r in fixed + ranked:
        if r not in rows:
            rows.append(r)
        if len(rows) >= 6:
            break
    return rows


def choose_stage3_rows(stage2: Dict[str, dict]) -> List[str]:
    ranked = sorted(stage2.keys(), key=lambda r: stage2[r]["accuracy"], reverse=True)
    rows = ["single_agent", "TEXT_2AGENT"]
    for r in ranked:
        if r not in rows:
            rows.append(r)
        if len(rows) >= 5:
            break
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run staged LAKV v1.5 benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B")
    parser.add_argument("--profile_path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="results/best_run")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(args.model, args.device)
    dataset = load_gsm8k("test", n=100)

    all_rows = [
        "single_agent",
        "TEXT_2AGENT",
        "A_legacy",
        "A_2agent",
        "C_legacy",
        "C_2agent",
        "D_legacy",
        "D_2agent",
        "best_anchor_2agent",
    ]

    print_prompt_validation(
        ["single_agent", "TEXT_2AGENT", "C_legacy", "C_2agent"],
        dataset,
        model,
        tokenizer,
        args.profile_path,
        args.device,
    )

    stage1_summaries, stage1_details = run_stage(
        "Stage 1 (smoke)",
        all_rows,
        dataset[:20],
        model,
        tokenizer,
        args.profile_path,
        args.device,
    )
    write_stage_csv(output_dir / "stage1_n20.csv", stage1_summaries)

    global_parse_rate = (
        sum(s["parse_failures"] for s in stage1_summaries.values())
        / max(sum(s["n_samples"] for s in stage1_summaries.values()), 1)
    )
    pause_two_agent = (
        stage1_summaries["C_2agent"]["accuracy"]
        < (stage1_summaries["C_legacy"]["accuracy"] - 0.05)
    )
    anchor_deprioritize = stage1_summaries["best_anchor_2agent"]["anchor_hit_rate"] < 0.05

    decisions = {
        "global_parse_rate": global_parse_rate,
        "pause_two_agent": pause_two_agent,
        "anchor_deprioritize": anchor_deprioritize,
    }

    if global_parse_rate > 0.10:
        print("\n[STOP] Parse failure rate > 10%. Fix parser/output contract before continuing.")
        with open(output_dir / "decisions.json", "w", encoding="utf-8") as f:
            json.dump(decisions, f, indent=2)
        with open(output_dir / "stage1_details.json", "w", encoding="utf-8") as f:
            json.dump(stage1_details, f, indent=2)
        return

    stage2_rows = choose_stage2_rows(stage1_summaries, pause_two_agent, anchor_deprioritize)
    stage2_summaries, stage2_details = run_stage(
        "Stage 2 (shortlist)",
        stage2_rows,
        dataset[:50],
        model,
        tokenizer,
        args.profile_path,
        args.device,
    )
    write_stage_csv(output_dir / "stage2_n50.csv", stage2_summaries)

    stage3_rows = choose_stage3_rows(stage2_summaries)
    stage3_summaries, stage3_details = run_stage(
        "Stage 3 (final)",
        stage3_rows,
        dataset[:100],
        model,
        tokenizer,
        args.profile_path,
        args.device,
    )
    write_stage_csv(output_dir / "stage3_n100.csv", stage3_summaries)

    with open(output_dir / "decisions.json", "w", encoding="utf-8") as f:
        json.dump(decisions, f, indent=2)
    with open(output_dir / "stage1_details.json", "w", encoding="utf-8") as f:
        json.dump(stage1_details, f, indent=2)
    with open(output_dir / "stage2_details.json", "w", encoding="utf-8") as f:
        json.dump(stage2_details, f, indent=2)
    with open(output_dir / "stage3_details.json", "w", encoding="utf-8") as f:
        json.dump(stage3_details, f, indent=2)

    print("\nBenchmark complete.")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
