"""Provenance record for a run: what produced these numbers.

Before this module, experiment_results.json stored no model, dataset, dtype,
attention backend, decoding settings, seeds, git commit or library versions
(only a GSM8K smoke-test folder had a run_metadata.json). Everything here is
best-effort and never raises: a metadata failure must not kill a long GPU run,
so any field that cannot be determined is recorded as None or as an
"error: ..." string rather than being guessed.

Pure functions, CPU-only, testable with fake model objects.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=_REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def git_info() -> Dict[str, Any]:
    """Commit hash, branch, and whether the working tree had uncommitted
    changes to tracked files (a dirty tree means the hash alone does not
    reproduce the run)."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": (bool(status) if status is not None else None),
    }


def library_versions() -> Dict[str, Optional[str]]:
    """Versions of the libraries that determine numerics and data."""
    out: Dict[str, Optional[str]] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("torch", "transformers", "datasets", "accelerate", "numpy", "scipy"):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            out[name] = None
    try:
        import torch
        out["cuda"] = torch.version.cuda
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        out["cuda"] = out["gpu"] = None
    return out


def file_sha256(path: Optional[str]) -> Optional[str]:
    """Hash of the calibration profile used, so the exact layer tiers are
    identifiable even though profiles are not tracked in git."""
    if not path:
        return None
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except Exception:
        return None


def model_info(model: Any) -> Dict[str, Any]:
    """dtype / attention backend / layer count / the model's own
    generation_config defaults (which single_agent and text_agent inherit,
    since they do not set repetition_penalty themselves)."""
    info: Dict[str, Any] = {}
    try:
        info["dtype"] = str(getattr(model, "dtype", None))
        cfg = getattr(model, "config", None)
        info["attn_implementation"] = getattr(cfg, "_attn_implementation", None)
        info["num_hidden_layers"] = getattr(cfg, "num_hidden_layers", None)
        info["model_name_or_path"] = getattr(cfg, "_name_or_path", None)
        gc = getattr(model, "generation_config", None)
        info["model_generation_config_defaults"] = {
            "repetition_penalty": getattr(gc, "repetition_penalty", None),
            "temperature": getattr(gc, "temperature", None),
            "top_p": getattr(gc, "top_p", None),
            "do_sample": getattr(gc, "do_sample", None),
        }
    except Exception as e:  # pragma: no cover - defensive
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def decoding_for_config(cfg_name: str) -> Dict[str, Any]:
    """Effective token budgets and decoding settings of one config name.

    KV-relay configs carry them in PipelineConfig; single_agent/text_agent
    have their own dataclasses and set no repetition_penalty (they inherit
    the model default, reported separately under model_info).
    """
    try:
        from lakv.evaluator import PRESETS, parse_audit_config_name, AUDIT_MODE_SUFFIXES
        from lakv.evaluator import ABLATION_CONFIGS, LAYER_BAND_CONFIGS
        from lakv.evaluator import REGION_CONFIGS
        from lakv.necessity_curve import split_hide_suffix
        full_name = cfg_name
        cfg_name, hide = split_hide_suffix(cfg_name, AUDIT_MODE_SUFFIXES)   # _hide100 -> _nopass
        region_spec = REGION_CONFIGS.get(cfg_name)
        base, _pool, audit_mode, nopass = parse_audit_config_name(region_spec["base"] if region_spec else cfg_name)
        d: Dict[str, Any] = {"base": base, "audit_mode": audit_mode, "receiver_sees_passages": not nopass}
        if hide is not None:
            d["hide"] = {"axis": hide.axis, "level": hide.level, "suffix": hide.suffix,
                         "receivers_see": "reduced view; the sender (agent 0) sees all passages"}
        if full_name != cfg_name and hide is None:
            d["alias_of"] = cfg_name
        if base in ("single_agent", "single_agent_matched"):
            from lakv_v2.pipeline.single_agent import SingleAgentPipelineConfig as C
            c = C()
            d.update(final_max_new_tokens=512 if base.endswith("_matched") else c.max_new_tokens,
                     intermediate_max_new_tokens=None, do_sample=c.do_sample,
                     repetition_penalty="model default (not set by this pipeline)")
        elif base in ("text_agent", "text_agent_matched"):
            from lakv_v2.pipeline.text_agent import TextAgentPipelineConfig as C
            c = C()
            d.update(final_max_new_tokens=512 if base.endswith("_matched") else c.final_max_new_tokens,
                     intermediate_max_new_tokens=c.intermediate_max_new_tokens, do_sample=c.do_sample,
                     repetition_penalty="model default (not set by this pipeline)")
        elif cfg_name in ABLATION_CONFIGS or cfg_name in LAYER_BAND_CONFIGS or base in PRESETS and PRESETS[base] is not None:
            p = PRESETS.get(base)
            if p is None:  # ablation configs are built at runtime
                from lakv.pipeline import PipelineConfig
                p = PipelineConfig(use_layer_selection=True, compression_mode="none",
                                   use_offset_correction=False, reconstruction_strategy="zeros")
                spec = ABLATION_CONFIGS.get(cfg_name) or LAYER_BAND_CONFIGS.get(cfg_name)
                d["ablation_spec"] = spec
                d["receiver_sees_passages"] = (spec or {}).get("receiver_sees_passages", True)
            d.update(final_max_new_tokens=p.final_max_new_tokens,
                     intermediate_max_new_tokens=p.intermediate_max_new_tokens,
                     compression_mode=p.compression_mode, use_layer_selection=p.use_layer_selection,
                     use_offset_correction=p.use_offset_correction,
                     reconstruction_strategy=p.reconstruction_strategy, n_agents=p.n_agents,
                     generation_kwargs=dict(p.generation_kwargs),
                     repetition_penalty=p.generation_kwargs.get("repetition_penalty"))
        if region_spec:
            d["region_spec"] = region_spec
        return d
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def dataset_provenance(ds_split, *, hf_id: str, config: str, split: str, samples) -> Dict[str, Any]:
    """Identify exactly which HuggingFace data produced a run (UNTESTED ON GPU; CPU-tested with fakes).

    Best-effort, never raises. Records: the datasets library fingerprint of the split, the
    cache revision directory (datasets caches hub datasets under .../<config>/<version>/<revision>/,
    so that folder name is the hub revision hash), row count, the dataset info version, and two
    content digests over the examples actually loaded (question+answer text; gold titles), which
    identify the data offline even when the cache path is unavailable. Only the last three
    components of cache file paths are kept (no user directory names).
    """
    import hashlib
    out: Dict[str, Any] = {"hf_id": hf_id, "config": config, "split": split}
    try:
        try:
            out["num_rows_split"] = len(ds_split)
        except Exception:
            out["num_rows_split"] = None
        out["fingerprint"] = getattr(ds_split, "_fingerprint", None)
        info = getattr(ds_split, "info", None)
        out["info_version"] = str(getattr(info, "version", None)) if info is not None else None
        out["builder_name"] = getattr(info, "builder_name", None) if info is not None else None
        files = getattr(ds_split, "cache_files", None) or []
        names = [f.get("filename") for f in files if isinstance(f, dict) and f.get("filename")]
        out["cache_files"] = ["/".join(Path(n).parts[-3:]) for n in names]
        out["cache_revision_dirs"] = sorted({Path(n).parent.name for n in names})
        try:
            import datasets as _ds
            out["datasets_version"] = _ds.__version__
        except Exception:
            out["datasets_version"] = None
        samples = list(samples)
        out["n_loaded"] = len(samples)
        qa = json.dumps([[x.get("question"), x.get("answer")] for x in samples], ensure_ascii=False, sort_keys=True)
        out["questions_answers_sha256"] = hashlib.sha256(qa.encode("utf-8")).hexdigest()
        gt = json.dumps([x.get("gold_titles") for x in samples], ensure_ascii=False)
        out["gold_titles_sha256"] = hashlib.sha256(gt.encode("utf-8")).hexdigest()
    except Exception as e:  # pragma: no cover - defensive
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def collect_run_metadata(
    *,
    mode: Optional[str] = None,
    model_name: Optional[str] = None,
    dataset: Optional[str] = None,
    split: Optional[str] = None,
    n_samples: Optional[int] = None,
    configs: Optional[Iterable[str]] = None,
    arch: Optional[str] = None,
    greedy: Optional[bool] = None,
    profile_path: Optional[str] = None,
    audit_donor_seed: Optional[int] = None,
    audit_pool_size: Optional[int] = None,
    model: Any = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the provenance record. Keys from the original run_metadata.json
    (timestamp, git_commit, model, dataset, split, n_samples, configs, arch,
    greedy, profile_path, datasets_version, transformers_version,
    torch_version) are kept with the same names for compatibility."""
    configs = list(configs) if configs is not None else None
    libs = library_versions()
    md: Dict[str, Any] = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "mode": mode,
        **git_info(),
        "model": model_name,
        "dataset": dataset,
        "split": split,
        "n_samples": n_samples,
        "configs": configs,
        "arch": arch,
        "greedy": greedy,
        "profile_path": profile_path,
        "profile_sha256": file_sha256(profile_path),
        "datasets_version": libs.get("datasets"),
        "transformers_version": libs.get("transformers"),
        "torch_version": libs.get("torch"),
        "libraries": libs,
        # No global torch/numpy/random seed is set anywhere in this code base;
        # decoding is greedy (do_sample=False). Randomness that does exist is
        # explicit and recorded here:
        "global_seed": None,
        "audit_donor_seed": audit_donor_seed,
        "audit_pool_size": audit_pool_size,
    }
    if model is not None:
        md["model_info"] = model_info(model)
        md["dtype"] = md["model_info"].get("dtype")
        md["attn_implementation"] = md["model_info"].get("attn_implementation")
    if configs:
        md["decoding_per_config"] = {c: decoding_for_config(c) for c in configs}
    if extra:
        md.update(extra)
    return md
