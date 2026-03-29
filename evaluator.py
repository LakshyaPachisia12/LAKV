"""
LAKV Module 6: Evaluator

Runs all pipeline configurations on a dataset, collects per-sample metrics,
and produces the results table suitable for a paper.
"""

import csv
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from pipeline import LAKVPipeline, PipelineConfig, RunResult


# ── config presets ────────────────────────────────────────────────────────────

PRESETS: Dict[str, PipelineConfig] = {
    "A": PipelineConfig(
        use_layer_selection=False, compression_mode="none",
        use_offset_correction=False, reconstruction_strategy="zeros",
    ),
    "B_int8": PipelineConfig(
        use_layer_selection=False, compression_mode="uniform_int8",
        use_offset_correction=False, reconstruction_strategy="zeros",
    ),
    "B_int4": PipelineConfig(
        use_layer_selection=False, compression_mode="uniform_int4",
        use_offset_correction=False, reconstruction_strategy="zeros",
    ),
    "C": PipelineConfig(
        use_layer_selection=True, compression_mode="none",
        use_offset_correction=False, reconstruction_strategy="zeros",
    ),
    # "D": PipelineConfig(
    #     use_layer_selection=True, compression_mode="none",
    #     use_offset_correction=False, reconstruction_strategy="zeros",
    # ),
    # "D": PipelineConfig(
    #     use_layer_selection=True, compression_mode="adaptive",
    #     use_offset_correction=False, reconstruction_strategy="zeros",
    # ),
    "D": PipelineConfig(
        use_layer_selection=True, compression_mode="adaptive",
        use_offset_correction=False, reconstruction_strategy="zeros",
    ),
}


def extract_answer(text: str) -> Optional[str]:
    """Extract numerical answer from model output (GSM8K format)."""
    match = re.search(r"####\s*(-?\d[\d,]*(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return numbers[-1].replace(",", "") if numbers else None


class Evaluator:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def run_experiment(self, dataset, profile_path, configs_to_run=None,
                       n_samples=100, output_dir="results/"):
        if configs_to_run is None:
            configs_to_run = ["A", "B_int8", "B_int4", "C", "D"]

        samples = dataset[:n_samples]
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        all_results = {}

        for cfg_name in configs_to_run:
            preset = PRESETS[cfg_name]
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
                ok = pred is not None and pred.strip() == gold
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
        for cfg_name in ("A","D"):
            preset = PRESETS[cfg_name]
            if preset.use_layer_selection or preset.compression_mode == "adaptive":
                preset.profile_path = profile_path
            pipe = LAKVPipeline(self.model, self.tokenizer, preset, self.device)
            print(f"\n{'='*60}\nSanity Check — Config {cfg_name}\n{'='*60}")
            for i,s in enumerate(samples):
                r = pipe.run(s["question"])
                pred = extract_answer(r.answer)
                print(f"  Q{i}: {s['question']}")
                print(f"  → answer: {r.answer[:200]}...")
                print(f"  → extracted: {pred}  |  gold: {s['answer']}")
                print(f"  → KV: {r.total_compressed_mb:.2f} MB (ratio {r.overall_compression_ratio:.2f}x)\n")

    @staticmethod
    def _print_table(all_results):
        labels = {"A":"A (raw)","B_int8":"B_int8","B_int4":"B_int4","C":"C (sel)","D":"D (sel+cmp)"}
        hdr = f"{'Config':<13}| {'Accuracy':>8} | {'KV/hop (MB)':>11} | {'Comp Ratio':>10} | {'Layers Sent':>11} | {'Latency(s)':>10}"
        print(f"\n{hdr}\n{'-'*len(hdr)}")
        for c,d in all_results.items():
            s=d["summary"]; lb=labels.get(c,c)
            print(f"{lb:<13}| {s['accuracy']*100:>7.1f}% | {s['mean_compressed_mb']:>8.2f} MB | "
                  f"{s['mean_compression_ratio']:>8.2f}x | {s['mean_layers_transmitted']:>5.0f}/28     | {s['mean_latency_seconds']:>8.1f}s")

    @staticmethod
    def _save_csv(all_results, path):
        with open(path,"w",newline="") as f:
            w=csv.writer(f)
            w.writerow(["config","accuracy","mean_compressed_mb","mean_compression_ratio",
                "mean_layers_transmitted","mean_latency_seconds","n_correct","n_samples"])
            for c,d in all_results.items():
                s=d["summary"]
                w.writerow([c,f"{s['accuracy']:.4f}",f"{s['mean_compressed_mb']:.4f}",
                    f"{s['mean_compression_ratio']:.4f}",f"{s['mean_layers_transmitted']:.1f}",
                    f"{s['mean_latency_seconds']:.2f}",s["n_correct"],s["n_samples"]])
