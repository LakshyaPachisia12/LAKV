"""Run provenance record (CPU-only; fake model object, no weights loaded)."""
import json
from types import SimpleNamespace

from lakv.run_metadata import (collect_run_metadata, decoding_for_config, file_sha256,
                               library_versions, model_info)

REQUIRED = ["model", "dataset", "dtype", "attn_implementation", "decoding_per_config",
            "audit_donor_seed", "global_seed", "git_commit", "libraries", "profile_sha256"]


def fake_model(layers=32, rep=None):
    return SimpleNamespace(
        dtype="torch.bfloat16",
        config=SimpleNamespace(_attn_implementation="sdpa", num_hidden_layers=layers,
                               _name_or_path="fake/model"),
        generation_config=SimpleNamespace(repetition_penalty=rep, temperature=None, top_p=None, do_sample=None),
    )


def test_all_required_fields_present_and_json_serialisable(tmp_path):
    prof = tmp_path / "p.json"; prof.write_text("{}")
    md = collect_run_metadata(mode="experiment", model_name="fake/model", dataset="hotpotqa",
                              split="validation", n_samples=100, configs=["A", "D", "text_agent"],
                              arch="legacy", greedy=False, profile_path=str(prof),
                              audit_donor_seed=3, model=fake_model())
    for k in REQUIRED:
        assert k in md, k
    assert md["audit_donor_seed"] == 3 and md["global_seed"] is None
    assert md["dtype"] == "torch.bfloat16" and md["attn_implementation"] == "sdpa"
    assert md["profile_sha256"] == file_sha256(str(prof)) and len(md["profile_sha256"]) == 64
    json.dumps(md, default=str)  # must serialise


def test_backward_compatible_keys_kept():
    md = collect_run_metadata(model_name="m", dataset="gsm8k", n_samples=3, configs=["A"])
    for k in ("timestamp", "git_commit", "model", "dataset", "split", "n_samples", "configs",
              "arch", "greedy", "profile_path", "datasets_version", "transformers_version", "torch_version"):
        assert k in md, k


def test_kv_config_budgets_and_repetition_penalty():
    d = decoding_for_config("A")
    assert d["final_max_new_tokens"] == 512 and d["intermediate_max_new_tokens"] == 512
    assert d["repetition_penalty"] == 1.05
    assert decoding_for_config("A_audit_dropped")["audit_mode"] == "dropped"
    assert decoding_for_config("A_nopass")["receiver_sees_passages"] is False


def test_baselines_report_model_default_repetition_penalty():
    # single_agent/text_agent set no repetition_penalty themselves; the record
    # must say so instead of claiming 1.05.
    for name in ("single_agent", "text_agent"):
        assert "model default" in decoding_for_config(name)["repetition_penalty"]
    assert decoding_for_config("single_agent")["final_max_new_tokens"] == 1536
    assert decoding_for_config("text_agent_matched")["final_max_new_tokens"] == 512
    assert model_info(fake_model(rep=None))["model_generation_config_defaults"]["repetition_penalty"] is None


def test_never_raises_on_bad_inputs():
    assert "error" in decoding_for_config("no_such_config") or "base" in decoding_for_config("no_such_config")
    assert file_sha256("/definitely/missing") is None
    assert "torch" in library_versions()
    model_info(object())  # no attributes at all
