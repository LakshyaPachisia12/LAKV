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

import torch
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
    "C_nearest": PipelineConfig(
        use_layer_selection=True, compression_mode="none",
        use_offset_correction=False, reconstruction_strategy="nearest",
    ),
    "C_interpolate": PipelineConfig(
        use_layer_selection=True, compression_mode="none",
        use_offset_correction=False, reconstruction_strategy="interpolate",
    ),
    "D": PipelineConfig(
        use_layer_selection=True, compression_mode="adaptive",
        use_offset_correction=False, reconstruction_strategy="zeros",
    ),
    "D_nearest": PipelineConfig(
        use_layer_selection=True, compression_mode="adaptive",
        use_offset_correction=False, reconstruction_strategy="nearest",
    ),
    "E": PipelineConfig(
        use_layer_selection=True, compression_mode="adaptive",
        use_offset_correction=True, reconstruction_strategy="zeros",
    ),
}


# ── ablation presets (require custom layer indices, built at runtime) ─────────
# These are registered by name; Evaluator builds them after loading the profile.
ABLATION_CONFIGS = {
    "C_random_20_s0": {"type": "random_select", "n_keep": 20, "seed": 0},
    "C_random_20_s1": {"type": "random_select", "n_keep": 20, "seed": 1},
    "C_random_20_s2": {"type": "random_select", "n_keep": 20, "seed": 2},
    "C_top20":        {"type": "fixed_select",  "indices": list(range(20))},
    "C_bottom20":     {"type": "fixed_select",  "indices": list(range(8, 28))},
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

    # ── pipeline factories ────────────────────────────────────────────────

    def _build_pipeline(self, cfg_name: str, profile_path: Optional[str]):
        """Return a pipeline for cfg_name. Handles standard presets, ablations, and single-agent."""
        import random as _random
        from layer_selector import LayerSelector
        from calibration_profiler import LayerProfile

        if cfg_name == "single_agent":
            from lakv_v2.pipeline.single_agent import SingleAgentPipeline, SingleAgentPipelineConfig
            return SingleAgentPipeline(self.model, self.tokenizer,
                                       SingleAgentPipelineConfig(), self.device), "single"

        if cfg_name in ABLATION_CONFIGS:
            spec = ABLATION_CONFIGS[cfg_name]
            if spec["type"] == "random_select":
                rng = _random.Random(spec["seed"])
                indices = sorted(rng.sample(range(28), spec["n_keep"]))
            else:
                indices = spec["indices"]

            # Build a fake profile where Tier 3 = dropped, Tier 1 = kept
            profile = LayerProfile.load(profile_path) if profile_path else None
            preset = PipelineConfig(
                use_layer_selection=True,
                compression_mode="none",
                use_offset_correction=False,
                reconstruction_strategy="zeros",
                profile_path=profile_path,
                _custom_layer_indices=indices,
            )
            pipe = LAKVPipeline(self.model, self.tokenizer, preset, self.device,
                                custom_layer_indices=indices)
            return pipe, "multi"

        # Standard preset
        preset = PRESETS[cfg_name]
        if preset.use_layer_selection or preset.compression_mode == "adaptive":
            preset.profile_path = profile_path
        pipe = LAKVPipeline(self.model, self.tokenizer, preset, self.device)
        return pipe, "multi"

    def run_experiment(self, dataset, profile_path, configs_to_run=None,
                       n_samples=100, output_dir="results/"):
        if configs_to_run is None:
            configs_to_run = ["single_agent", "A", "B_int8", "B_int4", "C", "D"]

        samples = dataset[:n_samples]
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        all_results = {}

        for cfg_name in configs_to_run:
            pipe, pipe_type = self._build_pipeline(cfg_name, profile_path)
            print(f"\n{'='*60}\nRunning Config {cfg_name}\n{'='*60}")

            correct = 0; per_sample = []
            tot_comp = tot_orig = tot_ratio = tot_layers = tot_lat = 0.0

            for i, s in enumerate(tqdm(samples, desc=f"Config {cfg_name}")):
                t0 = time.time()

                if pipe_type == "single":
                    raw_answer = pipe.run(s["question"])
                    elapsed = time.time() - t0
                    pred = extract_answer(raw_answer)
                    gold = str(s["answer"]).strip()
                    ok = pred is not None and pred.strip() == gold
                    if ok: correct += 1
                    per_sample.append({"idx":i,"question":s["question"],"gold":gold,
                        "predicted":pred,"raw_answer":raw_answer,"correct":ok,
                        "compressed_mb":0.0,"original_mb":0.0,
                        "compression_ratio":0.0,"latency_s":elapsed,"hop_stats":[]})
                    tot_lat += elapsed
                else:
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
        hdr = f"{'Config':<22}| {'Accuracy':>8} | {'KV/hop (MB)':>11} | {'Comp Ratio':>10} | {'Layers Sent':>11} | {'Latency(s)':>10}"
        print(f"\n{hdr}\n{'-'*len(hdr)}")
        for c, d in all_results.items():
            s = d["summary"]
            is_single = (c == "single_agent")
            comp_str = "      —   " if is_single else f"{s['mean_compressed_mb']:>8.2f} MB"
            ratio_str = "        — " if is_single else f"{s['mean_compression_ratio']:>8.2f}x"
            layer_str = "   —      " if is_single else f"{s['mean_layers_transmitted']:>5.0f}/28    "
            print(f"{c:<22}| {s['accuracy']*100:>7.1f}% | {comp_str} | "
                  f"{ratio_str} | {layer_str} | {s['mean_latency_seconds']:>8.1f}s")

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
