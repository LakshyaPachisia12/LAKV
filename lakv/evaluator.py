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

from lakv.pipeline import LAKVPipeline, PipelineConfig, PROMPT_SETS, RunResult
from lakv.qa_scoring import extract_qa_answer, exact_match_score, f1_score
from lakv.causal_audit import KVAuditPool


TWO_AGENT_BENCH_PROMPTS = {
    "gsm8k": [
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
    ],
    "hotpotqa": [
        (
            "You are a careful reading-comprehension agent.\n"
            "Read the given context passages and answer the question "
            "internally.\n"
            "Be accurate and use only the given context."
        ),
        (
            "You have access to previous reasoning memory.\n"
            "Use it to answer the question.\n\n"
            "Return ONLY:\n\n"
            "The answer is: <short answer>"
        ),
    ],
}


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
    # text_agent: same 3-agent role structure/prompts as Config A, but agents
    # communicate via literal decoded text instead of KV injection - see
    # lakv_v2.pipeline.text_agent.TextAgentPipeline. Also not a PipelineConfig;
    # _build_pipeline special-cases it the same way as single_agent above.
    "text_agent": None,
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
    # Phase 3 of the research-extensions plan: identical to B_int4 in every
    # setting (outlier_clipping stays on, so this is a clean single-variable
    # comparison) except the quantization scheme itself — rotates each head's
    # K/V by a fixed Hadamard matrix before quantizing (TurboQuant/PolarQuant-
    # inspired, see lakv/kv_compressor.py's rotation section for exactly what
    # is and isn't a faithful reproduction of that paper). Tests whether
    # B_int4's total collapse (CLAUDE.md: 0.0%/0.0%) is a quantization-scheme
    # artifact or a real 4-bit ceiling for this model.
    "B_int4_turboquant": PipelineConfig(
        use_layer_selection=False, compression_mode="uniform_int4_rotated",
        use_offset_correction=False, reconstruction_strategy="zeros",
        outlier_clipping=True,
    ),
    # Diagnostic pair, not a "real" config to report (same convention as
    # E_strict/E_nodelta above) — B_int4_turboquant produced wildly out-of-
    # distribution decoded tokens on a real n=100 HotpotQA run (literal Java
    # class names, random CJK characters), a failure SHAPE inconsistent with
    # ordinary quantization noise. Leading hypothesis: rotating K interacts
    # badly with RoPE, which is already applied to K before it's cached (V
    # has no RoPE). This isolates that: only V is rotated, K goes through
    # the exact same code path as plain B_int4 (verified bit-identical by
    # tests/test_kv_compressor_rotation.py). If this config's failure mode
    # looks like B_int4's ordinary degradation instead of turboquant's
    # bizarre one, that confirms the RoPE-interaction hypothesis.
    "B_int4_turboquant_vonly": PipelineConfig(
        use_layer_selection=False, compression_mode="uniform_int4_rotated_v_only",
        use_offset_correction=False, reconstruction_strategy="zeros",
        outlier_clipping=True,
    ),
    # KIVI-inspired (Liu et al., ICML'24) alternative to the rotation
    # approach above: instead of rotating K (which produced bizarre
    # out-of-distribution tokens, plausibly from disrupting K's RoPE
    # encoding), quantize K per-channel instead of per-head. V is unchanged
    # from plain B_int4 (not implicated in the RoPE-interaction failure).
    # See lakv/kv_compressor.py's _quantize_per_channel docstring for the
    # full reasoning and what's simplified relative to the original paper.
    "B_int4_kivi": PipelineConfig(
        use_layer_selection=False, compression_mode="uniform_int4_kivi_k_channel",
        use_offset_correction=False, reconstruction_strategy="zeros",
        outlier_clipping=True,
    ),
    # Recombination of the two validated pieces above, not new untested
    # math: K uses KIVI's per-channel grouping (uniform_int4_kivi_k_channel),
    # V uses the Hadamard rotation (uniform_int4_rotated_v_only). Built to
    # test whether V benefits from rotation ON TOP OF K's RoPE-safe fix, or
    # whether that's unnecessary complexity once K alone is fixed — compare
    # against B_int4_kivi specifically, don't assume this wins by default.
    "B_int4_hybrid": PipelineConfig(
        use_layer_selection=False, compression_mode="uniform_int4_hybrid",
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
    "D_interpolate": PipelineConfig(
        use_layer_selection=True, compression_mode="adaptive",
        use_offset_correction=False, reconstruction_strategy="interpolate",
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
    # Diagnostic pair, not a "real" config to report — isolates whether E's
    # accuracy gap vs A is caused by low-confidence anchor corrections being
    # applied anyway (graceful_degradation=True, the default) rather than
    # rejected in favor of uncorrected relay. Identical to E/E_int8 in every
    # other respect. See PipelineConfig.anchor_graceful_degradation.
    "E_strict": PipelineConfig(
        use_layer_selection=True, compression_mode="adaptive",
        use_offset_correction=True, reconstruction_strategy="zeros",
        outlier_clipping=True, anchor_graceful_degradation=False,
    ),
    "E_int8_strict": PipelineConfig(
        use_layer_selection=True, compression_mode="uniform_int8",
        use_offset_correction=True, reconstruction_strategy="zeros",
        anchor_graceful_degradation=False,
    ),
    # No longer just a diagnostic — after fixing query_correction to apply
    # the delta on top of the REAL relayed KV (matching the actual KVCOMM
    # reference design) instead of a substitute reconstruction, this is now
    # a legitimate config in its own right: D's real content, plus a pure
    # RoPE position-shift, with the anchor-transferred delta switched off.
    # Useful to isolate delta's own contribution once E's baseline (delta on)
    # is trustworthy. See PipelineConfig.anchor_delta_scale.
    "E_nodelta": PipelineConfig(
        use_layer_selection=True, compression_mode="adaptive",
        use_offset_correction=True, reconstruction_strategy="zeros",
        outlier_clipping=True, anchor_delta_scale=0.0,
    ),
}


# ── causal-audit configs (Phase 1 of the research-extensions plan) ────────────
# Not separate PRESETS entries (the brief explicitly calls for parametrized
# plumbing, not nine near-duplicate pipelines) — instead, a name suffix on an
# existing base config dispatches to the same preset with causal_audit_mode
# set. "mismatched" additionally requires a pre-built KVAuditPool — see
# Evaluator._build_audit_pool and the pre-pass in run_experiment.
AUDIT_BASE_CONFIGS = ("A", "D", "B_int8")
AUDIT_MODE_SUFFIXES = {
    "_audit_zeroed": "zeroed",
    "_audit_random": "random",
    "_audit_mismatched": "mismatched",
}
AUDIT_MISMATCHED_POOL_SIZE = 20


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


def score_sample(dataset_name: str, raw_text: str, gold) -> tuple:
    """Dispatch to the right extraction/comparison for this dataset.

    GSM8K: numeric extract_answer() + strict string equality (unchanged
    behavior). HotpotQA (or any non-gsm8k dataset): text-span
    extract_qa_answer() + normalized exact-match, plus token-F1 as a partial-
    credit signal that GSM8K's numeric answers have no equivalent for.
    Returns (predicted, correct, f1_or_none).
    """
    gold_s = str(gold).strip()
    if dataset_name == "gsm8k":
        pred = extract_answer(raw_text)
        ok = pred is not None and pred.strip() == gold_s
        return pred, ok, None

    pred = extract_qa_answer(raw_text)
    ok = exact_match_score(pred, gold_s)
    f1 = f1_score(pred, gold_s)
    return pred, ok, f1


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
        preset.system_prompts = list(TWO_AGENT_BENCH_PROMPTS[preset.dataset])
        preset.intermediate_max_new_tokens = 64
        preset.final_max_new_tokens = 16

    def _build_pipeline(self, cfg_name: str, profile_path: Optional[str], arch: str = "legacy",
                        dataset: str = "gsm8k", greedy: bool = False,
                        audit_pools: Optional[Dict[str, KVAuditPool]] = None,
                        record_audit_pool: Optional[KVAuditPool] = None):
        """Return a pipeline for cfg_name. Handles standard presets, ablations, single-agent,
        and causal-audit variants (<base>_audit_zeroed / _audit_random / _audit_mismatched,
        base in AUDIT_BASE_CONFIGS — see Phase 1 of the research-extensions plan).

        greedy=True forces do_sample=False on every config, overriding each
        pipeline's own sampling defaults (do_sample=True, temperature=0.6,
        top_p=0.95) — useful as a deterministic accuracy control run, since
        every preset otherwise samples per commit 1890b6a.

        audit_pools: {base_config_name: KVAuditPool}, required when cfg_name
        is a "*_audit_mismatched" variant — built ahead of time by
        run_experiment's held-out pre-pass, never by this method.
        record_audit_pool: when given, cfg_name is built as a PLAIN (non-
        audited) pipeline that records its real relayed KV into this pool
        instead of scoring anything — used only by _build_audit_pool's own
        pre-pass. Mutually exclusive with cfg_name being an audit variant.
        """
        import random as _random

        if cfg_name == "single_agent":
            from lakv_v2.pipeline.single_agent import SingleAgentPipeline, SingleAgentPipelineConfig
            cfg = SingleAgentPipelineConfig(dataset=dataset)
            if greedy:
                cfg.do_sample = False
            return SingleAgentPipeline(self.model, self.tokenizer, cfg, self.device), "single"

        if cfg_name == "text_agent":
            from lakv_v2.pipeline.text_agent import TextAgentPipeline, TextAgentPipelineConfig
            cfg = TextAgentPipelineConfig(dataset=dataset)
            if greedy:
                cfg.do_sample = False
            return TextAgentPipeline(self.model, self.tokenizer, cfg, self.device), "text"

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
                dataset=dataset,
            )
            self._apply_arch(preset, arch)
            if greedy:
                preset.generation_kwargs = dict(preset.generation_kwargs, do_sample=False)
            pipe = LAKVPipeline(self.model, self.tokenizer, preset, self.device,
                                custom_layer_indices=indices)
            return pipe, "multi"

        # Causal-audit variant? Strip a known suffix to find the base preset
        # name and which substitution mode to apply — see AUDIT_MODE_SUFFIXES.
        base_name = cfg_name
        audit_mode = "none"
        for suffix, mode in AUDIT_MODE_SUFFIXES.items():
            if cfg_name.endswith(suffix):
                base_name, audit_mode = cfg_name[: -len(suffix)], mode
                break

        if base_name not in PRESETS or PRESETS[base_name] is None:
            raise ValueError(f"Unknown config: {cfg_name!r}")

        # Standard preset. PRESETS entries were built at module-load time with
        # the gsm8k default, so system_prompts is already populated by
        # PipelineConfig.__post_init__ — reassign directly rather than
        # clearing it back to None, since __post_init__ only fills a None.
        preset = copy.deepcopy(PRESETS[base_name])
        preset.dataset = dataset
        preset.system_prompts = list(PROMPT_SETS[dataset])
        preset.causal_audit_mode = audit_mode
        self._apply_arch(preset, arch)
        if preset.use_layer_selection or preset.compression_mode == "adaptive":
            preset.profile_path = profile_path
        if greedy:
            preset.generation_kwargs = dict(preset.generation_kwargs, do_sample=False)

        audit_pool = None
        if audit_mode == "mismatched":
            if not audit_pools or base_name not in audit_pools:
                raise ValueError(
                    f"{cfg_name!r} requires a pre-built KVAuditPool for base config "
                    f"{base_name!r}, but none was provided — this should have been built "
                    f"by run_experiment's held-out pre-pass before reaching _build_pipeline."
                )
            audit_pool = audit_pools[base_name]

        pipe = LAKVPipeline(self.model, self.tokenizer, preset, self.device,
                            audit_pool=audit_pool, record_audit_pool=record_audit_pool)
        return pipe, "multi"

    def _build_audit_pool(self, base_cfg_name: str, profile_path: Optional[str],
                          held_out_samples: List[dict], arch: str = "legacy",
                          dataset_name: str = "gsm8k", greedy: bool = False) -> KVAuditPool:
        """Run base_cfg_name (plain, causal_audit_mode='none') over held_out_samples,
        recording every hop's REAL relayed KV into a fresh KVAuditPool.

        held_out_samples must never overlap with the examples actually being
        scored in the same run — see run_experiment, which slices them from
        beyond the scored n_samples range specifically so "mismatched" mode
        can never sample a question's own content back to itself.
        """
        pool = KVAuditPool(max_size_per_agent=len(held_out_samples))
        pipe, pipe_type = self._build_pipeline(base_cfg_name, profile_path, arch=arch,
                                               dataset=dataset_name, greedy=greedy,
                                               record_audit_pool=pool)
        assert pipe_type == "multi", (
            f"_build_audit_pool only supports KV-relay base configs (got {base_cfg_name!r}, "
            f"pipe_type={pipe_type!r}) — single_agent/text_agent have no KV handoff to record."
        )
        for s in held_out_samples:
            pipe.run(s["question"])
        return pool

    def run_experiment(self, dataset, profile_path, configs_to_run=None,
                       n_samples=100, output_dir="results/", checkpoint_every=5,
                       resume=False, arch: str = "legacy", print_raw_outputs: bool = False,
                       dataset_name: str = "gsm8k", greedy: bool = False):
        if configs_to_run is None:
            # E (layer selection + adaptive compression + anchor-table offset
            # correction) was defined in PRESETS but missing from this default
            # list, so a plain `--mode experiment` run with no --configs never
            # exercised it. Added so it runs head-to-head with the others.
            configs_to_run = ["single_agent", "A", "B_int8", "B_int4", "C", "D", "E", "E_int8"]

        samples = dataset[:n_samples]

        # ── causal-audit pre-pass: build a held-out KVAuditPool for every
        # base config that has a "*_audit_mismatched" variant requested. Must
        # happen BEFORE the main loop below, and must draw from examples
        # beyond samples (dataset[n_samples:...]), never from samples itself
        # — the whole point of "mismatched" mode is that its substitute
        # content is never the current question's own. If the caller didn't
        # load enough extra examples for this (see run.py, which reserves
        # AUDIT_MISMATCHED_POOL_SIZE extra when a mismatched config is
        # requested), fail loudly here rather than building an
        # under-sized/degenerate pool silently.
        audit_pools: Dict[str, KVAuditPool] = {}
        mismatched_bases = sorted({
            cfg[: -len("_audit_mismatched")]
            for cfg in configs_to_run if cfg.endswith("_audit_mismatched")
        })
        if mismatched_bases:
            held_out = dataset[n_samples:n_samples + AUDIT_MISMATCHED_POOL_SIZE]
            if len(held_out) < AUDIT_MISMATCHED_POOL_SIZE:
                raise ValueError(
                    f"Need {AUDIT_MISMATCHED_POOL_SIZE} held-out examples beyond "
                    f"n_samples={n_samples} to build a causal-audit pool for "
                    f"{mismatched_bases}, but only {len(held_out)} are available "
                    f"(dataset has {len(dataset)} total). Load more examples — "
                    f"e.g. n_samples + {AUDIT_MISMATCHED_POOL_SIZE} from the dataset loader."
                )
            for base in mismatched_bases:
                print(f"[Evaluator] Building causal-audit KV pool for '{base}' "
                      f"from {len(held_out)} held-out examples (never among the {n_samples} scored)...")
                audit_pools[base] = self._build_audit_pool(
                    base, profile_path, held_out, arch=arch,
                    dataset_name=dataset_name, greedy=greedy)

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
                    "mean_f1": None,
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
            f1_vals = [x["f1"] for x in per_sample if x.get("f1") is not None]
            mean_f1 = (sum(f1_vals) / len(f1_vals)) if f1_vals else None

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
                "mean_f1": mean_f1,
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

                pipe, pipe_type = self._build_pipeline(cfg_name, profile_path, arch=arch,
                                                        dataset=dataset_name, greedy=greedy,
                                                        audit_pools=audit_pools)
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

                    try:
                        is_gsm8k = dataset_name == "gsm8k"
                        gold = str(s["answer"]).strip()
                        if pipe_type == "single":
                            raw_answer = pipe.run(s["question"])
                            elapsed = time.time() - t0
                            pred, ok, f1 = score_sample(dataset_name, raw_answer, gold)
                            if ok: correct += 1
                            per_sample.append({"idx":i,"question":s["question"],"gold":gold,
                                "predicted":pred,"raw_answer":raw_answer,"correct":ok,"f1":f1,
                                "is_malformed":is_malformed(raw_answer) if is_gsm8k else False,
                                "numeric_grounding_failure": (not has_numeric_grounding(s["question"], raw_answer)) if is_gsm8k else False,
                                "compressed_mb":0.0,"original_mb":0.0,
                                "compression_ratio":0.0,"latency_s":elapsed,"hop_stats":[]})
                            tot_lat += elapsed
                        elif pipe_type == "text":
                            r = pipe.run(s["question"])
                            elapsed = time.time() - t0
                            pred, ok, f1 = score_sample(dataset_name, r.answer, gold)
                            if ok: correct += 1
                            reasoner_text = r.hop_texts[0] if r.hop_texts else None
                            # Reuse the KV-config fields (compressed_mb/original_mb/
                            # compression_ratio) to carry the text payload size, so
                            # the existing summary table/CSV need no schema changes
                            # and text_agent's payload shows up directly comparable
                            # to A-E's KV/hop MB in the same column. total_bytes is
                            # the sum of both inter-agent handoffs (Reasoner->
                            # Verifier, Verifier->Finalizer); mb below is per-hop
                            # mean to match how KV configs report per-hop MB.
                            nh = max(len(r.hop_bytes), 1)
                            total_mb = r.total_bytes / 1e6
                            per_sample.append({"idx":i,"question":s["question"],"gold":gold,
                                "predicted":pred,"raw_answer":r.answer,"correct":ok,"f1":f1,
                                "is_malformed":is_malformed(r.answer) if is_gsm8k else False,
                                "reasoner_text":reasoner_text,"hop_texts":r.hop_texts,
                                "numeric_grounding_failure": (not has_numeric_grounding(
                                    s["question"], reasoner_text if reasoner_text else r.answer)) if is_gsm8k else False,
                                "compressed_mb":total_mb,"original_mb":total_mb,
                                "compression_ratio":1.0,"latency_s":elapsed,"hop_stats":[],
                                "hop_latencies_s":r.hop_latencies})
                            tot_comp += total_mb/nh
                            tot_orig += total_mb/nh
                            tot_ratio += 1.0
                            tot_lat += elapsed
                        else:
                            r = pipe.run(s["question"])
                            elapsed = time.time() - t0
                            pred, ok, f1 = score_sample(dataset_name, r.answer, gold)
                            if ok: correct += 1
                            nh = max(len(r.hop_stats), 1)
                            # r.hop_texts[0] is the Reasoner's full raw decoded text (hop 1),
                            # now captured separately from the Aggregator's terse final answer
                            # (r.answer) — grounding checks against reasoning text instead of
                            # a one-line "The answer is N." string.
                            reasoner_text = r.hop_texts[0] if r.hop_texts else None
                            per_sample.append({"idx":i,"question":s["question"],"gold":gold,
                                "predicted":pred,"raw_answer":r.answer,"correct":ok,"f1":f1,
                                "is_malformed":is_malformed(r.answer) if is_gsm8k else False,
                                "reasoner_text":reasoner_text,"hop_texts":r.hop_texts,
                                "numeric_grounding_failure": (not has_numeric_grounding(
                                    s["question"], reasoner_text if reasoner_text else r.answer)) if is_gsm8k else False,
                                "compressed_mb":r.total_compressed_mb,"original_mb":r.total_original_mb,
                                "compression_ratio":r.overall_compression_ratio,"latency_s":elapsed,
                                "hop_stats":[asdict(h) for h in r.hop_stats],
                                "finalizer_latency_s":r.finalizer_latency_seconds,
                                "offset_logs":list(pipe.last_run_offset_logs),
                                "audit_logs":list(getattr(pipe, "last_run_audit_logs", []))})
                            tot_comp += r.total_compressed_mb/nh
                            tot_orig += r.total_original_mb/nh
                            tot_ratio += r.overall_compression_ratio
                            tot_layers += sum(h.n_layers_transmitted for h in r.hop_stats)/nh
                            tot_lat += elapsed
                    except (torch.OutOfMemoryError, RuntimeError) as e:
                        if isinstance(e, RuntimeError) and not isinstance(e, torch.OutOfMemoryError) \
                                and "out of memory" not in str(e).lower():
                            raise
                        elapsed = time.time() - t0
                        torch.cuda.empty_cache()
                        gold = str(s["answer"]).strip()
                        print(f"  [OOM] Config {cfg_name} sample {i} skipped (out of memory), "
                              f"continuing with next sample: {e}")
                        per_sample.append({"idx":i,"question":s["question"],"gold":gold,
                            "predicted":None,"raw_answer":None,"correct":False,"f1":0.0 if dataset_name != "gsm8k" else None,
                            "is_malformed":True,"numeric_grounding_failure":True,
                            "compressed_mb":0.0,"original_mb":0.0,
                            "compression_ratio":0.0,"latency_s":elapsed,"hop_stats":[],
                            "oom_skipped":True})
                        tot_lat += elapsed

                    if checkpoint_every and ((i + 1) % checkpoint_every == 0):
                        _update_running_summary(cfg_name, per_sample, correct)
                        _write_partial_file()

                n = len(samples)
                f1_vals = [x["f1"] for x in per_sample if x.get("f1") is not None]
                summary = {"config":cfg_name,"accuracy":correct/n if n else 0,
                    "mean_compressed_mb":tot_comp/n if n else 0,
                    "mean_compression_ratio":tot_ratio/n if n else 0,
                    "mean_layers_transmitted":tot_layers/n if n else 0,
                    "mean_latency_seconds":tot_lat/n if n else 0,
                    "parse_failures":sum(1 for x in per_sample if x["predicted"] is None),
                    "malformed_output_rate":sum(1 for x in per_sample if x["is_malformed"])/n if n else 0,
                    "numeric_grounding_failures":sum(1 for x in per_sample if x.get("numeric_grounding_failure")),
                    "mean_f1": (sum(f1_vals)/len(f1_vals)) if f1_vals else None,
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

    def run_sanity_check(self, profile_path, dataset=None, n_samples=3, arch: str = "legacy",
                         print_raw_outputs: bool = False, dataset_name: str = "gsm8k",
                         greedy: bool = False):
        if dataset is None:
            dataset = [{"question":"What is 15 + 27?","answer":"42"},
                {"question":"A train travels 60 miles in 2 hours. Speed in mph?","answer":"30"},
                {"question":"Apples cost $2 each. How much for 5?","answer":"10"}]
        samples = dataset[:n_samples]
        for cfg_name in ("A","D"):
            preset = copy.deepcopy(PRESETS[cfg_name])
            preset.dataset = dataset_name
            preset.system_prompts = list(PROMPT_SETS[dataset_name])
            self._apply_arch(preset, arch)
            if preset.use_layer_selection or preset.compression_mode == "adaptive":
                preset.profile_path = profile_path
            if greedy:
                preset.generation_kwargs = dict(preset.generation_kwargs, do_sample=False)
            pipe = LAKVPipeline(self.model, self.tokenizer, preset, self.device)
            pipe.config.print_raw_outputs = print_raw_outputs
            print(f"\n{'='*60}\nSanity Check — Config {cfg_name}\n{'='*60}")
            for i,s in enumerate(samples):
                r = pipe.run(s["question"])
                pred, ok, f1 = score_sample(dataset_name, r.answer, s["answer"])
                print(f"  Q{i}: {s['question']}")
                print(f"  → answer: {r.answer[:200]}...")
                print(f"  → extracted: {pred}  |  gold: {s['answer']}  |  correct: {ok}"
                      + (f"  |  f1: {f1:.2f}" if f1 is not None else ""))
                if dataset_name == "gsm8k":
                    grounding_failed = not has_numeric_grounding(s["question"], r.answer)
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
            # n_layers_total varies by model (28 for Qwen2.5-7B, 32 for e.g.
            # Llama-3.1-8B) — read it from this config's own hop_stats rather
            # than hardcoding, so the table stays correct once a second model
            # is evaluated (see Phase 0b of the research-extensions plan).
            n_layers_total = None
            for sample in d.get("per_sample", []):
                hs = sample.get("hop_stats")
                if hs:
                    n_layers_total = hs[0].get("n_layers_total")
                    break
            total_str = f"{n_layers_total}" if n_layers_total is not None else "?"
            layer_str = "   —      " if is_single else f"{s['mean_layers_transmitted']:>5.0f}/{total_str:<6}"
            print(f"{c:<22}| {s['accuracy']*100:>7.1f}% | {comp_str} | "
                  f"{ratio_str} | {layer_str} | {s['mean_latency_seconds']:>8.1f}s")
            n = s.get("n_samples", 0)
            mean_f1 = s.get("mean_f1")
            if mean_f1 is not None:
                print(f"{'':<22}  mean F1: {mean_f1*100:.1f}%")
            gf = s.get("numeric_grounding_failures", 0)
            if n and mean_f1 is None:
                print(f"{'':<22}  numeric grounding: {gf}/{n} flagged ({gf/n*100:.1f}%)")

    @staticmethod
    def _save_csv(all_results, path):
        with open(path,"w",newline="") as f:
            w=csv.writer(f)
            w.writerow(["config","accuracy","mean_compressed_mb","mean_compression_ratio",
                "mean_layers_transmitted","mean_latency_seconds","parse_failures","malformed_output_rate",
                "numeric_grounding_failures","mean_f1","n_correct","n_samples"])
            for c,d in all_results.items():
                s=d["summary"]
                mean_f1 = s.get("mean_f1")
                w.writerow([c,f"{s['accuracy']:.4f}",f"{s['mean_compressed_mb']:.4f}",
                    f"{s['mean_compression_ratio']:.4f}",f"{s['mean_layers_transmitted']:.1f}",
                    f"{s['mean_latency_seconds']:.2f}",s["parse_failures"],f"{s['malformed_output_rate']:.4f}",
                    s.get("numeric_grounding_failures", 0),
                    f"{mean_f1:.4f}" if mean_f1 is not None else "",
                    s["n_correct"],s["n_samples"]])
