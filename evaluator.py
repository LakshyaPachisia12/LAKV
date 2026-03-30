"""
LAKV Module 6: Evaluator

Runs all pipeline configurations on a dataset, collects per-sample metrics,
and produces the results table suitable for a paper.
"""

import csv
import json
import re
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from pipeline import LAKVPipeline, PipelineConfig, RunResult


# ── config presets ────────────────────────────────────────────────────────────

CONFIGS: Dict[str, PipelineConfig] = {
    "A": PipelineConfig(
        use_layer_selection=False, compression_mode="none",
        use_offset_correction=False, reconstruction_strategy="nearest",
        n_agents=3,
    ),
    "B_int8": PipelineConfig(
        use_layer_selection=False, compression_mode="uniform_int8",
        use_offset_correction=False, reconstruction_strategy="nearest",
        n_agents=3,
    ),
    "B_int4": PipelineConfig(
        use_layer_selection=False, compression_mode="uniform_int4",
        use_offset_correction=False, reconstruction_strategy="nearest",
        n_agents=3,
    ),
    "C": PipelineConfig(
        use_layer_selection=True, compression_mode="none",
        use_offset_correction=False, reconstruction_strategy="nearest",
        n_agents=3,
    ),
    "D": PipelineConfig(
        use_layer_selection=True, compression_mode="adaptive",
        use_offset_correction=False, reconstruction_strategy="nearest",
        n_agents=3,
    ),
}

PRESETS = CONFIGS


def extract_answer(text: str) -> Optional[str]:
    number_pat = r"[+-]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"

    # Try "The answer is X" pattern first
    match = re.search(rf"[Tt]he answer is\s*\$?({number_pat})", text)
    if match:
        return match.group(1).replace(",", "")
    # Try #### pattern (GSM8K standard)
    match = re.search(rf"####\s*\$?({number_pat})", text)
    if match:
        return match.group(1).replace(",", "")
    # Fallback: last number in response
    numbers = re.findall(number_pat, text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def answers_match(predicted: Optional[str], gold: str) -> bool:
    if predicted is None:
        return False
    try:
        return abs(float(predicted.strip()) - float(gold.strip())) < 1e-3
    except Exception:
        return predicted.strip() == gold.strip()


class Evaluator:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def run_experiment(self, dataset, profile_path, configs_to_run=None,
                       n_samples=100, output_dir="results/"):
        if configs_to_run is None:
            configs_to_run = ["A", "B_int8", "C", "D"]

        samples = dataset[:n_samples]
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        all_results = {}

        for cfg_name in configs_to_run:
            base_cfg = CONFIGS[cfg_name]
            preset = replace(base_cfg)
            if preset.use_layer_selection or preset.compression_mode == "adaptive":
                preset.profile_path = profile_path
            pipe = LAKVPipeline(self.model, self.tokenizer, preset, self.device)
            print(f"\n{'='*60}\nRunning Config {cfg_name}\n{'='*60}")

            correct = 0; per_sample = []
            tot_comp = tot_orig = tot_ratio = tot_layers = tot_lat = 0.0

            for i, s in enumerate(tqdm(samples, desc=f"Config {cfg_name}")):
                t0 = time.time()
                r = pipe.run(s["question"])
                elapsed = time.time() - t0
                pred = extract_answer(r.answer)
                gold = str(s["answer"]).strip()
                ok = answers_match(pred, gold)
                print(f"  Sample {i} | time={elapsed:.1f}s | answer={r.answer[:50]}")
                if ok: correct += 1
                nh = max(len(r.hop_stats), 1)
                per_sample.append({"idx":i,"question":s["question"],"gold":gold,
                    "predicted":pred,"raw_answer":r.answer,"correct":ok,
                    "compressed_mb":r.total_compressed_mb,"original_mb":r.total_original_mb,
                    "compression_ratio":r.overall_compression_ratio,"latency_s":elapsed,
                    "hop_stats":[asdict(h) for h in r.hop_stats]})
                tot_comp += r.total_compressed_mb/nh
                tot_orig += r.total_original_mb/nh
                tot_ratio += r.overall_compression_ratio
                tot_layers += sum(h.n_layers_transmitted for h in r.hop_stats)/nh
                tot_lat += elapsed

            n = len(samples)
            summary = {"config":cfg_name,"accuracy":correct/n if n else 0,
                "mean_compressed_mb":tot_comp/n if n else 0,
                "mean_compression_ratio":tot_ratio/n if n else 0,
                "mean_layers_transmitted":tot_layers/n if n else 0,
                "mean_latency_seconds":tot_lat/n if n else 0,
                "n_correct":correct,"n_samples":n}
            all_results[cfg_name] = {"summary":summary,"per_sample":per_sample}

        self._print_table(all_results)
        with open(out/"experiment_results.json","w") as f:
            json.dump(all_results, f, indent=2, default=str)
        self._save_csv(all_results, out/"results_table.csv")
        hop_data = {c:[s["hop_stats"] for s in d["per_sample"]] for c,d in all_results.items()}
        with open(out/"per_hop_stats.json","w") as f:
            json.dump(hop_data, f, indent=2, default=str)
        print(f"\n[Evaluator] Results saved to {out}/")
        return all_results

    def run_sanity_check(self, profile_path, dataset=None, n_samples=3):
        if dataset is None:
            dataset = [{"question":"What is 15 + 27?","answer":"42"},
                {"question":"A train travels 60 miles in 2 hours. Speed in mph?","answer":"30"},
                {"question":"Apples cost $2 each. How much for 5?","answer":"10"}]
        samples = dataset[:n_samples]
        cfg_a = replace(CONFIGS["A"])
        cfg_d = replace(CONFIGS["D"])
        cfg_d.profile_path = profile_path

        pipe_a = LAKVPipeline(self.model, self.tokenizer, cfg_a, self.device)
        pipe_d = LAKVPipeline(self.model, self.tokenizer, cfg_d, self.device)

        print(f"\n{'='*80}\nSanity Check — Config A vs Config D\n{'='*80}")
        for i, s in enumerate(samples):
            r_a = pipe_a.run(s["question"])
            r_d = pipe_d.run(s["question"])

            pred_a = extract_answer(r_a.answer)
            pred_d = extract_answer(r_d.answer)
            gold = str(s["answer"]).strip()

            print(f"\nSample {i}")
            print(f"  Q: {s['question']}")
            print(f"  Gold: {gold}")
            print(f"  A answer: {r_a.answer[:150]}")
            print(f"  A extracted: {pred_a} | match={answers_match(pred_a, gold)}")
            print(f"  D answer: {r_d.answer[:150]}")
            print(f"  D extracted: {pred_d} | match={answers_match(pred_d, gold)}")
            print(
                f"  KV sizes | A: {r_a.total_compressed_mb:.2f} MB "
                f"({r_a.overall_compression_ratio:.2f}x) "
                f"| D: {r_d.total_compressed_mb:.2f} MB ({r_d.overall_compression_ratio:.2f}x)"
            )

    @staticmethod
    def _print_table(all_results):
        labels = {
            "A": "A (raw)",
            "B_int8": "B_int8",
            "B_int4": "B_int4",
            "C": "C (sel)",
            "D": "D (sel+cmp)",
        }
        order = ["A", "B_int8", "B_int4", "C", "D"]

        print("\nConfig       | Accuracy | KV/hop (MB) | Comp Ratio | Layers Sent | Latency(s)")
        print("-------------|----------|-------------|------------|-------------|----------")
        for cfg_name in order:
            if cfg_name not in all_results:
                continue
            s = all_results[cfg_name]["summary"]
            layers = int(round(s["mean_layers_transmitted"]))
            print(
                f"{labels[cfg_name]:<12} | "
                f"{s['accuracy']*100:>6.1f}% | "
                f"{s['mean_compressed_mb']:>9.2f} MB | "
                f"{s['mean_compression_ratio']:>8.2f}x | "
                f"{layers:>6d}/28    | "
                f"{s['mean_latency_seconds']:>7.1f}s"
            )

    @staticmethod
    def _save_csv(all_results, path):
        order = ["A", "B_int8", "B_int4", "C", "D"]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Config", "Accuracy", "KV/hop (MB)", "Comp Ratio", "Layers Sent", "Latency(s)"])
            for cfg_name in order:
                if cfg_name not in all_results:
                    continue
                s = all_results[cfg_name]["summary"]
                w.writerow([
                    cfg_name,
                    f"{s['accuracy']*100:.1f}%",
                    f"{s['mean_compressed_mb']:.2f}",
                    f"{s['mean_compression_ratio']:.2f}x",
                    f"{int(round(s['mean_layers_transmitted']))}/28",
                    f"{s['mean_latency_seconds']:.1f}s",
                ])
