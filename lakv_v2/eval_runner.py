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
            
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def run_eval(self, dataset, configs_to_run, output_dir="lakv_v2_results/"):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        
        summary = {}
        failure_log = []

        for cfg_name in configs_to_run:
            print(f"\n[{cfg_name}] Starting Evaluation...")
            pipe, pipe_type = self.select_pipeline(cfg_name)
            
            correct = 0
            latencies = []
            
            for i, s in enumerate(tqdm(dataset, desc=cfg_name)):
                question = s["question"]
                gold = str(s["answer"]).strip()
                
                # Execute depending on pipeline style
                if pipe_type == "single":
                    t0 = time.time()
                    raw_answer = pipe.run(question)
                    latency = time.time() - t0
                elif pipe_type == "old":
                    t0 = time.time()
                    r = pipe.run(question)
                    latency = time.time() - t0
                    raw_answer = r.answer
                else: 
                    # v2 pipeline
                    r = pipe.run(question)
                    raw_answer = r["answer"]
                    latency = r["latency"]

                pred = extract_answer(raw_answer)
                ok = (pred is not None) and (pred.strip() == gold)
                
                if ok: 
                    correct += 1
                else:
                    failure_log.append({
                        "config": cfg_name,
                        "question_id": i,
                        "question": question,
                        "gold": gold,
                        "predicted": pred,
                        "raw_output": raw_answer
                    })
                    
                latencies.append(latency)

            acc = correct / len(dataset)
            mean_lat = sum(latencies)/len(latencies)
            print(f"[{cfg_name}] Acc: {acc*100:.1f}%, Latency: {mean_lat:.2f}s")
            
            summary[cfg_name] = {
                "accuracy": acc,
                "n_correct": correct,
                "n_samples": len(dataset),
                "mean_latency_s": mean_lat
            }

        # Save results
        with open(out / "eval_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
            
        with open(out / "failure_logs.json", "w") as f:
            json.dump(failure_log, f, indent=2)
            
        print(f"\n[Eval] Check {output_dir} for results and extensive failure logs.")
