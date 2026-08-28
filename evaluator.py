"""
LAKV Module 6: Evaluator

Runs all pipeline configurations on a dataset, collects per-sample metrics,
and produces the results table suitable for a paper.
"""

import csv
import copy
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from pipeline import LAKVPipeline, PipelineConfig, RunResult


TWO_AGENT_BENCH_PROMPTS = [
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


# ── config presets ────────────────────────────────────────────────────────────

PRESETS: Dict[str, Optional[PipelineConfig]] = {
    # single_agent is not a PipelineConfig preset — it uses a different
    # pipeline class (SingleAgentPipeline / SingleAgentPipelineConfig from
    # lakv_v2.pipeline.single_agent) with no relay, compression, or layer
    # selection at all. It's listed here (value=None) purely so it shows up
    # in PRESETS for discoverability/enumeration; _build_pipeline() special-
    # cases cfg_name == "single_agent" BEFORE ever indexing into PRESETS, so
    # this entry is never deepcopy'd or passed through _apply_arch(). Do not
    # replace None with a real PipelineConfig — SingleAgentPipelineConfig has
    # an incompatible shape (no n_agents/system_prompts/etc.) and would break
    # here if the special-case above it were ever removed.
    "single_agent": None,
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
        outlier_clipping=True,
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
        outlier_clipping=True,
    ),
    "D_nearest": PipelineConfig(
        use_layer_selection=True, compression_mode="adaptive",
        use_offset_correction=False, reconstruction_strategy="nearest",
        outlier_clipping=True,
    ),
    "E": PipelineConfig(
        use_layer_selection=True, compression_mode="adaptive",
        use_offset_correction=True, reconstruction_strategy="zeros",
        outlier_clipping=True,
    ),
    "E_int8": PipelineConfig(
        use_layer_selection=True, compression_mode="uniform_int8",
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
    """Extract numeric answer, preferring explicit final-answer markers with robust fallbacks.

    Priority order: #### N  >  \\boxed{N}  >  "answer is/=/: N"  >  a bare-number
    line  >  last number in the text. The first three patterns anchor on an
    explicit "this is the final answer" marker, so they are checked (in that
    order) BEFORE any digit-repair cleanup runs — this prevents unrelated
    numbers elsewhere in the reasoning from ever outranking a clearly marked
    final answer.
    """
    if not text:
        return None

    # Repair split-digit corruption like "1!2!0" -> "120": only merge digits
    # separated by punctuation/symbols with NO whitespace and NO letters in
    # between. This intentionally does NOT merge two distinct numbers that
    # appear in ordinary prose (e.g. "4 apples and 7 oranges"), since that gap
    # contains letters/spaces and won't match — merging those was the bug that
    # turned "\boxed{7}" plus an earlier "4" into a spurious "47".
    cleaned = re.sub(r"(?<=\d)[^\w\s.,\-]+(?=\d)", "", text)

    for candidate in (text, cleaned):
        match = re.search(r"####\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)", candidate)
        if match:
            return match.group(1).replace(",", "").rstrip(".")

    for candidate in (text, cleaned):
        match = list(re.finditer(r"\\boxed\{\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)\s*\}", candidate))
        if match:
            return match[-1].group(1).replace(",", "").rstrip(".")

    for candidate in (text, cleaned):
        match = re.search(r"(?i)answer\s*(?:is|=|:)?\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)", candidate)
        if match:
            return match.group(1).replace(",", "").rstrip(".")

    for candidate in (text, cleaned):
        match = re.search(r"(?im)^\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)\s*\.?\s*$", candidate)
        if match:
            return match.group(1).replace(",", "").rstrip(".")

    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", cleaned)
    return numbers[-1].replace(",", "").rstrip(".") if numbers else None


def extract_numbers(text: str) -> List[str]:
    """Extract every number in text, normalized (commas stripped) for comparison."""
    if not text:
        return []
    return [n.replace(",", "") for n in re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)]


def has_numeric_grounding(question: str, reasoning_text: str, early_fraction: float = 0.3) -> bool:
    """Cheap regex sanity check: does the model's early reasoning reference at
    least one number actually mentioned in the question?

    This is a pipeline-health signal, not a correctness check. It is designed
    to catch gross misreads cheaply — wrong question reaching the model,
    garbled/corrupted generation, a KV relay hop handing over the wrong
    context — the kind of bug that otherwise takes hours of manual JSON
    archaeology to spot. It deliberately does NOT try to be a precise
    verifier: a question with no numbers, or reasoning that references a real
    question number in different shorthand (e.g. "50k" for "50,000"), is not
    flagged — false positives would make the signal useless to act on.
    """
    q_numbers = set(extract_numbers(question))
    if not q_numbers:
        return True  # nothing to ground against — don't flag

    if not reasoning_text:
        return False

    cutoff = max(1, int(len(reasoning_text) * early_fraction))
    early_numbers = set(extract_numbers(reasoning_text[:cutoff]))
    return bool(q_numbers & early_numbers)


def is_malformed(text: str) -> bool:
    if not text:
        return True
    # Too many repeated punctuations
    if re.search(r"[!?.]{4,}", text):
        return True
    # Or split digits
    if re.search(r"(?<=\d)[^\d,\.\-\s]+(?=\d)", text):
        return True
    return False


class Evaluator:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    # ── pipeline factories ────────────────────────────────────────────────

    @staticmethod
    def _apply_arch(preset: PipelineConfig, arch: str) -> None:
        if arch == "legacy":
            return
        if arch != "two_agent":
            raise ValueError(f"Unknown arch: {arch}")

        preset.n_agents = 2
        preset.system_prompts = list(TWO_AGENT_BENCH_PROMPTS)
        preset.intermediate_max_new_tokens = 64
        preset.final_max_new_tokens = 16

    def _build_pipeline(self, cfg_name: str, profile_path: Optional[str], arch: str = "legacy"):
        """Return a pipeline for cfg_name. Handles standard presets, ablations, and single-agent."""
        import random as _random

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
            self._apply_arch(preset, arch)
            pipe = LAKVPipeline(self.model, self.tokenizer, preset, self.device,
                                custom_layer_indices=indices)
            return pipe, "multi"

        # Standard preset
        preset = copy.deepcopy(PRESETS[cfg_name])
        self._apply_arch(preset, arch)
        if preset.use_layer_selection or preset.compression_mode == "adaptive":
            preset.profile_path = profile_path
        pipe = LAKVPipeline(self.model, self.tokenizer, preset, self.device)
        return pipe, "multi"

    def run_experiment(self, dataset, profile_path, configs_to_run=None,
                       n_samples=100, output_dir="results/", checkpoint_every=5,
                       resume=False, arch: str = "legacy", print_raw_outputs: bool = False):
        if configs_to_run is None:
            # E (layer selection + adaptive compression + anchor-table offset
            # correction) was defined in PRESETS but missing from this default
            # list, so a plain `--mode experiment` run with no --configs never
            # exercised it. Added so it runs head-to-head with the others.
            configs_to_run = ["single_agent", "A", "B_int8", "B_int4", "C", "D", "E", "E_int8"]

        samples = dataset[:n_samples]
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)

        partial_path = out / "experiment_results.partial.json"
        all_results = {}

        # ── resume: load prior partial checkpoint ─────────────────────────────
        if resume and partial_path.exists():
            with open(partial_path) as f:
                all_results = json.load(f)
            completed = [c for c, d in all_results.items() if d["summary"].get("status") == "completed"]
            print(f"[Evaluator] Resuming — already completed: {completed or 'none'}")

        def _write_partial_file():
            with open(partial_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

        def _update_running_summary(cfg_name: str, per_sample: List[dict], correct: int):
            n_done = len(per_sample)
            if n_done == 0:
                summary = {
                    "config": cfg_name,
                    "accuracy": 0.0,
                    "mean_compressed_mb": 0.0,
                    "mean_compression_ratio": 0.0,
                    "mean_layers_transmitted": 0.0,
                    "mean_latency_seconds": 0.0,
                    "parse_failures": 0,
                    "malformed_output_rate": 0.0,
                    "numeric_grounding_failures": 0,
                    "n_correct": 0,
                    "n_samples": 0,
                    "n_expected": len(samples),
                    "status": "running",
                }
                all_results[cfg_name] = {"summary": summary, "per_sample": per_sample}
                return

            mean_latency = sum(x["latency_s"] for x in per_sample) / n_done
            mean_comp = sum(x["compressed_mb"] for x in per_sample) / n_done
            mean_ratio = sum(x["compression_ratio"] for x in per_sample) / n_done
            parse_failures = sum(1 for x in per_sample if x["predicted"] is None)
            malformed_rate = sum(1 for x in per_sample if x["is_malformed"]) / n_done
            grounding_failures = sum(1 for x in per_sample if x.get("numeric_grounding_failure"))

            layers_vals = []
            for x in per_sample:
                hs = x.get("hop_stats", [])
                if hs:
                    layers_vals.append(sum(h["n_layers_transmitted"] for h in hs) / max(len(hs), 1))
            mean_layers = (sum(layers_vals) / len(layers_vals)) if layers_vals else 0.0

            summary = {
                "config": cfg_name,
                "accuracy": correct / n_done,
                "mean_compressed_mb": mean_comp,
                "mean_compression_ratio": mean_ratio,
                "mean_layers_transmitted": mean_layers,
                "mean_latency_seconds": mean_latency,
                "parse_failures": parse_failures,
                "malformed_output_rate": malformed_rate,
                "numeric_grounding_failures": grounding_failures,
                "n_correct": correct,
                "n_samples": n_done,
                "n_expected": len(samples),
                "status": "running",
            }
            all_results[cfg_name] = {"summary": summary, "per_sample": per_sample}

        try:
            for cfg_name in configs_to_run:
                # Skip configs already completed in a prior run
                if resume and cfg_name in all_results and all_results[cfg_name]["summary"].get("status") == "completed":
                    print(f"[Evaluator] Skipping {cfg_name} (already completed)")
                    continue

                pipe, pipe_type = self._build_pipeline(cfg_name, profile_path, arch=arch)
                pipe.config.print_raw_outputs = print_raw_outputs
                print(f"\n{'='*60}\nRunning Config {cfg_name}\n{'='*60}")

                # Restore partial progress for this config if resuming mid-config
                if resume and cfg_name in all_results and all_results[cfg_name]["summary"].get("status") == "running":
                    per_sample = all_results[cfg_name]["per_sample"]
                    correct = sum(1 for x in per_sample if x["correct"])
                    start_idx = len(per_sample)
                    print(f"[Evaluator] Resuming {cfg_name} from sample {start_idx}/{len(samples)}")
                else:
                    per_sample = []
                    correct = 0
                    start_idx = 0

                tot_comp = sum(x["compressed_mb"] for x in per_sample)
                tot_orig = sum(x.get("original_mb", 0.0) for x in per_sample)
                tot_ratio = sum(x["compression_ratio"] for x in per_sample)
                tot_layers = sum(
                    sum(h["n_layers_transmitted"] for h in x["hop_stats"]) / max(len(x["hop_stats"]), 1)
                    for x in per_sample if x.get("hop_stats")
                )
                tot_lat = sum(x["latency_s"] for x in per_sample)

                remaining = samples[start_idx:]
                for i_rel, s in enumerate(tqdm(remaining, desc=f"Config {cfg_name}", initial=start_idx, total=len(samples))):
                    i = start_idx + i_rel
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
                            "is_malformed":is_malformed(raw_answer),
                            "numeric_grounding_failure": not has_numeric_grounding(s["question"], raw_answer),
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
                        # r.hop_texts[0] is the Reasoner's full raw decoded text (hop 1),
                        # now captured separately from the Aggregator's terse final answer
                        # (r.answer) — grounding checks against reasoning text instead of
                        # a one-line "The answer is N." string.
                        reasoner_text = r.hop_texts[0] if r.hop_texts else None
                        per_sample.append({"idx":i,"question":s["question"],"gold":gold,
                            "predicted":pred,"raw_answer":r.answer,"correct":ok,
                            "is_malformed":is_malformed(r.answer),
                            "reasoner_text":reasoner_text,"hop_texts":r.hop_texts,
                            "numeric_grounding_failure": not has_numeric_grounding(
                                s["question"], reasoner_text if reasoner_text else r.answer),
                            "compressed_mb":r.total_compressed_mb,"original_mb":r.total_original_mb,
                            "compression_ratio":r.overall_compression_ratio,"latency_s":elapsed,
                            "hop_stats":[asdict(h) for h in r.hop_stats]})
                        tot_comp += r.total_compressed_mb/nh
                        tot_orig += r.total_original_mb/nh
                        tot_ratio += r.overall_compression_ratio
                        tot_layers += sum(h.n_layers_transmitted for h in r.hop_stats)/nh
                        tot_lat += elapsed

                    if checkpoint_every and ((i + 1) % checkpoint_every == 0):
                        _update_running_summary(cfg_name, per_sample, correct)
                        _write_partial_file()

                n = len(samples)
                summary = {"config":cfg_name,"accuracy":correct/n if n else 0,
                    "mean_compressed_mb":tot_comp/n if n else 0,
                    "mean_compression_ratio":tot_ratio/n if n else 0,
                    "mean_layers_transmitted":tot_layers/n if n else 0,
                    "mean_latency_seconds":tot_lat/n if n else 0,
                    "parse_failures":sum(1 for x in per_sample if x["predicted"] is None),
                    "malformed_output_rate":sum(1 for x in per_sample if x["is_malformed"])/n if n else 0,
                    "numeric_grounding_failures":sum(1 for x in per_sample if x.get("numeric_grounding_failure")),
                    "n_correct":correct,"n_samples":n,
                    "n_expected":n,
                    "status":"completed"}
                all_results[cfg_name] = {"summary":summary,"per_sample":per_sample}
                _write_partial_file()
        except KeyboardInterrupt:
            print("\n[Evaluator] Interrupted. Saving partial progress...")
            _write_partial_file()
            print(f"[Evaluator] Checkpoint saved to {partial_path}")
            print(f"[Evaluator] Resume with: --resume --output_dir {out}")
            raise

        self._print_table(all_results)
        with open(out/"experiment_results.json","w") as f:
            json.dump(all_results, f, indent=2, default=str)
        self._save_csv(all_results, out/"results_table.csv")
        hop_data = {c:[s["hop_stats"] for s in d["per_sample"]] for c,d in all_results.items()}
        with open(out/"per_hop_stats.json","w") as f:
            json.dump(hop_data, f, indent=2, default=str)
        print(f"\n[Evaluator] Results saved to {out}/")
        return all_results

    def run_sanity_check(self, profile_path, dataset=None, n_samples=3, arch: str = "legacy", print_raw_outputs: bool = False):
        if dataset is None:
            dataset = [{"question":"What is 15 + 27?","answer":"42"},
                {"question":"A train travels 60 miles in 2 hours. Speed in mph?","answer":"30"},
                {"question":"Apples cost $2 each. How much for 5?","answer":"10"}]
        samples = dataset[:n_samples]
        for cfg_name in ("A","D"):
            preset = copy.deepcopy(PRESETS[cfg_name])
            self._apply_arch(preset, arch)
            if preset.use_layer_selection or preset.compression_mode == "adaptive":
                preset.profile_path = profile_path
            pipe = LAKVPipeline(self.model, self.tokenizer, preset, self.device)
            pipe.config.print_raw_outputs = print_raw_outputs
            print(f"\n{'='*60}\nSanity Check — Config {cfg_name}\n{'='*60}")
            for i,s in enumerate(samples):
                r = pipe.run(s["question"])
                pred = extract_answer(r.answer)
                grounding_failed = not has_numeric_grounding(s["question"], r.answer)
                print(f"  Q{i}: {s['question']}")
                print(f"  → answer: {r.answer[:200]}...")
                print(f"  → extracted: {pred}  |  gold: {s['answer']}")
                print(f"  → numeric grounding: {'FAILED — check for misread/corruption' if grounding_failed else 'ok'}")
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
            n = s.get("n_samples", 0)
            gf = s.get("numeric_grounding_failures", 0)
            if n:
                print(f"{'':<22}  numeric grounding: {gf}/{n} flagged ({gf/n*100:.1f}%)")

    @staticmethod
    def _save_csv(all_results, path):
        with open(path,"w",newline="") as f:
            w=csv.writer(f)
            w.writerow(["config","accuracy","mean_compressed_mb","mean_compression_ratio",
                "mean_layers_transmitted","mean_latency_seconds","parse_failures","malformed_output_rate",
                "numeric_grounding_failures","n_correct","n_samples"])
            for c,d in all_results.items():
                s=d["summary"]
                w.writerow([c,f"{s['accuracy']:.4f}",f"{s['mean_compressed_mb']:.4f}",
                    f"{s['mean_compression_ratio']:.4f}",f"{s['mean_layers_transmitted']:.1f}",
                    f"{s['mean_latency_seconds']:.2f}",s["parse_failures"],f"{s['malformed_output_rate']:.4f}",
                    s.get("numeric_grounding_failures", 0),s["n_correct"],s["n_samples"]])
