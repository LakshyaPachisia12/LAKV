"""The held-out mismatched-pool pre-pass must treat `<cfg>_maskzero` exactly like `<cfg>`
(regression for the overnight-6 crash where `A_nopass_audit_mismatched_maskzero` built no pool)."""
import pytest

from lakv.evaluator import Evaluator, AUDIT_MISMATCHED_POOL_SIZE


class _Stop(Exception):
    pass


def _prepass(monkeypatch, configs):
    ev = Evaluator(model=None, tokenizer=None, device="cpu")
    built = []

    def fake_build(base, profile_path, held_out, **kw):
        built.append((base, [s["question"] for s in held_out]))
        return object()

    def stop(*a, **k):
        raise _Stop

    monkeypatch.setattr(ev, "_build_audit_pool", fake_build)
    monkeypatch.setattr(ev, "_build_donor_sources", stop)
    n = 3
    dataset = [{"question": f"q{i}", "answer": "a"} for i in range(n + AUDIT_MISMATCHED_POOL_SIZE)]
    with pytest.raises(_Stop):
        ev.run_experiment(dataset, None, configs_to_run=configs, n_samples=n, output_dir="unused_out")
    return built


def test_pool_prepass_recognises_maskzero_suffix(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    plain = _prepass(monkeypatch, ["A_nopass_audit_mismatched"])
    masked = _prepass(monkeypatch, ["A_nopass_audit_mismatched_maskzero"])
    assert plain and plain == masked                       # same base, same held-out questions
    assert plain[0][0] == "A_nopass"
    both = _prepass(monkeypatch, ["A_nopass_audit_mismatched", "A_nopass_audit_mismatched_maskzero"])
    assert len(both) == 1                                  # one pool per base, not two


def test_no_pool_for_non_mismatched_maskzero(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert _prepass(monkeypatch, ["A_nopass_audit_real_pad10_maskzero"]) == []
