"""
LAKV-v2 Module: Eval Runner
Executes evaluation loops cleanly comparing Old LAKV vs Baseline vs LAKV-v2.
Retains per-sample failure logs.
"""

import json
import time
from pathlib import Path
from tqdm import tqdm

from lakv_v2.pipeline.single_agent import SingleAgentPipeline, SingleAgentPipelineConfig
from lakv_v2.pipeline.two_agent import TwoAgentPipeline, TwoAgentPipelineConfig
from lakv_v2.cache.selector import LayerSelectorV2
from lakv_v2.cache.compressor import CompressorV2
from lakv_v2.cache.reconstruct import ReconstructorV2
from lakv_v2.utils.parsing import extract_answer

# Import old pipeline for direct comparison
from pipeline import LAKVPipeline, PipelineConfig as OldPipelineConfig
from evaluator import PRESETS as OLD_PRESETS

class UnifiedEvalRunner:
    def __init__(self, model, tokenizer, device="cuda", profile_path=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Load from your original beautiful calibration profile JSON
        self.tier_1_2_indices = list(range(19)) # Fallback
        if profile_path:
            from calibration_profiler import LayerProfile
            profile = LayerProfile.load(profile_path)
            self.tier_1_2_indices = [i for i, t in enumerate(profile.tier_assignment) if t in (1, 2)]
            print(f"[LAKV-v2] Loaded calibration from {profile_path}")
            print(f"[LAKV-v2] Selecting {len(self.tier_1_2_indices)} layers based on Tier 1 & 2.")

    def select_pipeline(self, mode: str):
        if mode == "single_agent_baseline":
            cfg = SingleAgentPipelineConfig()
            p = SingleAgentPipeline(self.model, self.tokenizer, cfg, self.device)
            return p, "single"
            
        elif mode == "old_lakv_A":
            preset = OLD_PRESETS["A"]
            p = LAKVPipeline(self.model, self.tokenizer, preset, self.device)
            return p, "old"
            
        elif mode == "lakv_v2_full":
            cfg = TwoAgentPipelineConfig()
            p = TwoAgentPipeline(self.model, self.tokenizer, cfg, self.device)
            return p, "v2_full"
            
        elif mode == "lakv_v2_selected":
            cfg = TwoAgentPipelineConfig()
            cfg.selector = LayerSelectorV2(keep_indices=self.tier_1_2_indices, reconstructor=ReconstructorV2("mean_fill"))
            p = TwoAgentPipeline(self.model, self.tokenizer, cfg, self.device)
            return p, "v2_selected"
            
        elif mode == "lakv_v2_selected_int8":
            cfg = TwoAgentPipelineConfig()
            cfg.selector = LayerSelectorV2(keep_indices=self.tier_1_2_indices, reconstructor=ReconstructorV2("mean_fill"))
            cfg.compressor = CompressorV2(mode="uniform_int8")
            p = TwoAgentPipeline(self.model, self.tokenizer, cfg, self.device)
            return p, "v2_selected_int8"

        elif mode == "lakv_v2_anchor":
            from anchor_table import AnchorTable
            cfg = TwoAgentPipelineConfig()
            cfg.selector = LayerSelectorV2(keep_indices=self.tier_1_2_indices, reconstructor=ReconstructorV2("mean_fill"))
            cfg.compressor = CompressorV2(mode="uniform_int8")
            cfg.anchor_table = AnchorTable(max_size=20, entropy_threshold=0.3)
            p = TwoAgentPipeline(self.model, self.tokenizer, cfg, self.device)
            return p, "v2_anchor"

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def run_eval(self, dataset, configs_to_run, output_dir="lakv_v2_results/"):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        summary = {}
        failure_log = []
        per_sample_all = {}

        for cfg_name in configs_to_run:
            print(f"\n[{cfg_name}] Starting Evaluation...")
            pipe, pipe_type = self.select_pipeline(cfg_name)

            correct = 0
            latencies = []
            compressed_mbs = []
            original_mbs = []
            ratios = []
            layers_sent_list = []
            per_sample = []

            for i, s in enumerate(tqdm(dataset, desc=cfg_name)):
                question = s["question"]
                gold = str(s["answer"]).strip()

                compressed_mb = 0.0
                original_mb = 0.0
                ratio = 0.0
                layers_sent = 0
                anchor_hit = False
                anchor_confidence = 0.0

                if pipe_type == "single":
                    t0 = time.time()
                    raw_answer = pipe.run(question)
                    latency = time.time() - t0
                elif pipe_type == "old":
                    t0 = time.time()
                    r = pipe.run(question)
                    latency = time.time() - t0
                    raw_answer = r.answer
                    compressed_mb = r.total_compressed_mb
                    original_mb = r.total_original_mb
                    ratio = r.overall_compression_ratio
                    nh = max(len(r.hop_stats), 1)
                    layers_sent = sum(h.n_layers_transmitted for h in r.hop_stats) / nh
                else:
                    # v2 pipeline
                    r = pipe.run(question)
                    raw_answer = r["answer"]
                    latency = r["latency"]
                    cs = r.get("compression_stats", {})
                    compressed_mb = cs.get("compressed_mb", 0.0)
                    original_mb = cs.get("original_mb", 0.0)
                    ratio = cs.get("ratio", 0.0)
                    layers_sent = cs.get("layers_sent", 0)
                    anchor_hit = r.get("anchor_hit", False)
                    anchor_confidence = r.get("anchor_confidence", 0.0)

                pred = extract_answer(raw_answer)
                ok = (pred is not None) and (pred.strip() == gold)
                if ok:
                    correct += 1
                else:
                    failure_log.append({
                        "config": cfg_name, "question_id": i,
                        "question": question, "gold": gold,
                        "predicted": pred, "raw_output": raw_answer,
                    })

                latencies.append(latency)
                compressed_mbs.append(compressed_mb)
                original_mbs.append(original_mb)
                ratios.append(ratio)
                layers_sent_list.append(layers_sent)
                per_sample.append({
                    "idx": i, "question": question, "gold": gold,
                    "predicted": pred, "raw_answer": raw_answer, "correct": ok,
                    "latency_s": latency, "compressed_mb": compressed_mb,
                    "original_mb": original_mb, "compression_ratio": ratio,
                    "layers_sent": layers_sent,
                    "anchor_hit": anchor_hit, "anchor_confidence": anchor_confidence,
                })

            n = len(dataset)
            acc = correct / n
            mean_lat = sum(latencies) / n
            mean_comp = sum(compressed_mbs) / n
            mean_orig = sum(original_mbs) / n
            mean_ratio = sum(ratios) / n
            mean_layers = sum(layers_sent_list) / n
            n_hits = sum(1 for s in per_sample if s["anchor_hit"])
            hit_rate = n_hits / n

            anchor_note = f" | AnchorHits: {n_hits}/{n} ({hit_rate*100:.1f}%)" if any(
                s["anchor_hit"] for s in per_sample) else ""
            print(f"[{cfg_name}] Acc: {acc*100:.1f}% | "
                  f"Latency: {mean_lat:.2f}s | "
                  f"KV: {mean_comp:.2f}MB (ratio {mean_ratio:.2f}x) | "
                  f"Layers: {mean_layers:.1f}{anchor_note}")

            summary[cfg_name] = {
                "accuracy": acc, "n_correct": correct, "n_samples": n,
                "mean_latency_s": mean_lat,
                "mean_compressed_mb": mean_comp,
                "mean_original_mb": mean_orig,
                "mean_compression_ratio": mean_ratio,
                "mean_layers_sent": mean_layers,
                "anchor_hit_rate": hit_rate,
                "n_anchor_hits": n_hits,
            }
            per_sample_all[cfg_name] = per_sample

        # ── print summary table ───────────────────────────────────────────
        hdr = f"{'Config':<28}| {'Accuracy':>8} | {'KV(MB)':>8} | {'Ratio':>7} | {'Layers':>7} | {'Lat(s)':>8}"
        print(f"\n{hdr}\n{'-'*len(hdr)}")
        for cfg_name, s in summary.items():
            is_single = (cfg_name == "single_agent_baseline")
            comp = "      —" if is_single else f"{s['mean_compressed_mb']:>7.2f}"
            rat  = "     — " if is_single else f"{s['mean_compression_ratio']:>6.2f}x"
            lay  = "     — " if is_single else f"{s['mean_layers_sent']:>6.1f}"
            print(f"{cfg_name:<28}| {s['accuracy']*100:>7.1f}% | {comp} | {rat} | {lay} | {s['mean_latency_s']:>7.2f}s")

        with open(out / "eval_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        with open(out / "failure_logs.json", "w") as f:
            json.dump(failure_log, f, indent=2)
        with open(out / "per_sample_results.json", "w") as f:
            json.dump(per_sample_all, f, indent=2)

        print(f"\n[Eval] Results saved to {output_dir}")
