"""CPU unit tests for the dose-response corruption operators (lakv/cache_corruption.py)
and their wiring (apply_causal_audit dispatch, evaluator config-name parsing)."""
import sys
import zlib
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lakv.cache_corruption import (
    apply_corruption, apply_corruption_mode, cache_checksum, is_corruption_mode, modified_positions,
    parse_corruption_mode, stable_seed, uses_length_matched_donor,
)
from lakv.causal_audit import DonorUnavailable, NoMatchedDonor, apply_causal_audit
from lakv.evaluator import parse_audit_config_name


def make_kv(L=100, layers=3, heads=2, dim=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return tuple((torch.randn(1, heads, L, dim, generator=g) + 1.0,
                  torch.randn(1, heads, L, dim, generator=g) + 1.0) for _ in range(layers))


def length(kv):
    return kv[0][0].shape[2]


@pytest.mark.parametrize("pct,m_trunc,m_pad", [(10, 10, 11), (25, 25, 33), (50, 50, 100)])
def test_output_lengths(pct, m_trunc, m_pad):
    kv = make_kv(100)
    assert length(apply_corruption(kv, "trunc", pct)[0]) == 100 - m_trunc
    out, info = apply_corruption(kv, "pad", pct)
    assert length(out) == 100 + m_pad and info["n_modified"] == m_pad
    # zero positions are fraction p of the FINAL cache (the old arm's pad-fraction definition)
    assert abs(m_pad / (100 + m_pad) - pct / 100) < 0.01
    assert length(apply_corruption(kv, "fill", pct)[0]) == 100
    assert length(apply_corruption(kv, "fillhead", pct)[0]) == 100
    for (k, v) in apply_corruption(kv, "pad", pct)[0]:
        assert k.shape[:2] == kv[0][0].shape[:2] and k.shape[3] == kv[0][0].shape[3]


def test_trunc_keeps_prefix_exactly():
    kv = make_kv(80)
    out, info = apply_corruption(kv, "trunc", 25)
    assert length(out) == 60
    for (k, v), (ko, vo) in zip(kv, out):
        assert torch.equal(ko, k[:, :, :60]) and torch.equal(vo, v[:, :, :60])
    assert (info["modified_start"], info["modified_end"]) == (60, 80)  # removed range of the input


def test_pad_appends_zeros_and_keeps_real_content():
    kv = make_kv(60)
    out, info = apply_corruption(kv, "pad", 25)
    m = info["n_modified"]
    assert m == 20 and length(out) == 80
    for (k, v), (ko, vo) in zip(kv, out):
        assert torch.equal(ko[:, :, :60], k) and torch.equal(vo[:, :, :60], v)
        assert torch.count_nonzero(ko[:, :, 60:]) == 0 and torch.count_nonzero(vo[:, :, 60:]) == 0
    assert (info["modified_start"], info["modified_end"]) == (60, 80)


def test_fill_zeros_exactly_the_tail_range():
    kv = make_kv(100)
    out, info = apply_corruption(kv, "fill", 25)
    assert (info["modified_start"], info["modified_end"], info["n_modified"]) == (75, 100, 25)
    for (k, v), (ko, vo) in zip(kv, out):
        assert torch.equal(ko[:, :, :75], k[:, :, :75]) and torch.equal(vo[:, :, :75], v[:, :, :75])
        assert torch.count_nonzero(ko[:, :, 75:]) == 0 and torch.count_nonzero(vo[:, :, 75:]) == 0
    assert info["modified_max_abs_K_before"] > 0 and info["modified_max_abs_K_after"] == 0.0


def test_fillhead_zeros_exactly_the_head_range():
    kv = make_kv(100)
    out, info = apply_corruption(kv, "fillhead", 50)
    assert (info["modified_start"], info["modified_end"], info["n_modified"]) == (0, 50, 50)
    for (k, v), (ko, vo) in zip(kv, out):
        assert torch.count_nonzero(ko[:, :, :50]) == 0 and torch.count_nonzero(vo[:, :, :50]) == 0
        assert torch.equal(ko[:, :, 50:], k[:, :, 50:]) and torch.equal(vo[:, :, 50:], v[:, :, 50:])


@pytest.mark.parametrize("op", ["trunc", "pad", "fill", "fillhead"])
def test_input_never_mutated(op):
    kv = make_kv(50)
    snap = tuple((k.clone(), v.clone()) for k, v in kv)
    apply_corruption(kv, op, 50)
    for (k, v), (ks, vs) in zip(kv, snap):
        assert torch.equal(k, ks) and torch.equal(v, vs)


def test_trunc_never_empties_cache():
    kv = make_kv(3)
    assert length(apply_corruption(kv, "trunc", 99)[0]) >= 1


def test_determinism_checksum_and_seed():
    a, ia = apply_corruption_mode("real_pad25", make_kv(70), "qkey", 1, audit_seed=3)
    b, ib = apply_corruption_mode("real_pad25", make_kv(70), "qkey", 1, audit_seed=3)
    assert ia == ib and ia["cache_checksum"] == ib["cache_checksum"] == cache_checksum(a)
    assert ia["sample_seed"] == zlib.crc32(b"corrupt|3|qkey|real_pad25|1") & 0x7FFFFFFF  # crc32, process independent
    # different question / hop / mode / audit seed give different seeds; different content a different checksum
    assert stable_seed("corrupt", 3, "other", "real_pad25", 1) != ia["sample_seed"]
    _, ic = apply_corruption_mode("real_pad25", make_kv(70, seed=1), "qkey", 1, audit_seed=3)
    assert ic["cache_checksum"] != ia["cache_checksum"]
    _, id_ = apply_corruption_mode("real_fill25", make_kv(70), "qkey", 1, audit_seed=3)
    assert id_["cache_checksum"] != ia["cache_checksum"]


def test_mode_parsing():
    assert parse_corruption_mode("real_fillhead25") == ("real", "fillhead", 25)
    assert parse_corruption_mode("lm_trunc50") == ("lm", "trunc", 50)
    assert uses_length_matched_donor("lm_pad10") and not uses_length_matched_donor("real_pad10")
    for bad in ("real_pad0", "real_pad100", "foo", "length_matched", "mismatched", "real_zap25", ""):
        assert not is_corruption_mode(bad)
    with pytest.raises(ValueError):
        parse_corruption_mode("real_pad0")
    with pytest.raises(ValueError):
        parse_corruption_mode("mismatched")


def test_config_name_parsing_new_and_old_names_unchanged():
    assert parse_audit_config_name("A_nopass_audit_real_pad25") == ("A", "A_nopass", "real_pad25", True)
    assert parse_audit_config_name("A_audit_lm_fillhead50") == ("A", "A", "lm_fillhead50", False)
    assert parse_audit_config_name("A_nopass_audit_lm_trunc10") == ("A", "A_nopass", "lm_trunc10", True)
    # pre-existing names unchanged
    assert parse_audit_config_name("A") == ("A", "A", "none", False)
    assert parse_audit_config_name("A_audit_length_matched") == ("A", "A", "length_matched", False)
    assert parse_audit_config_name("A_nopass_audit_mismatched") == ("A", "A_nopass", "mismatched", True)
    assert parse_audit_config_name("A_nopass_audit_dropped") == ("A", "A_nopass", "dropped", True)
    assert parse_audit_config_name("B_int8_audit_zeroed") == ("B_int8", "B_int8", "zeroed", False)
    assert parse_audit_config_name("text_agent_audit_mismatched")[2] == "mismatched"


class _FakeDonorSource:
    def __init__(self, donor):
        self.donor = donor
        self.calls = 0

    def fetch(self, agent_idx, question_key, real_kv, rng=None, **ctx):
        self.calls += 1
        return self.donor, {"agent_idx": agent_idx, "donor_cache_len": length(self.donor),
                            "pad_positions": 0, "truncated_positions": 0}


class _NoDonorSource:
    def fetch(self, agent_idx, question_key, real_kv, rng=None, **ctx):
        raise NoMatchedDonor("no donor within 5% of target length", {"mode": "length_matched"})


def test_dispatch_real_target_uses_real_cache_no_donor():
    real = make_kv(100)
    out, log = apply_causal_audit("real_fill25", real, agent_idx=1, question_key="q", audit_seed=0)
    assert log["mode"] == "real_fill25" and log["target"] == "real" and "donor" not in log
    assert length(out) == 100 and torch.count_nonzero(out[0][0][:, :, 75:]) == 0
    assert torch.equal(out[0][0][:, :, :75], real[0][0][:, :, :75])


def test_dispatch_lm_target_corrupts_the_donor_unmodified_then_operator():
    real, donor = make_kv(100, seed=0), make_kv(97, seed=5)
    src = _FakeDonorSource(donor)
    out, log = apply_causal_audit("lm_trunc25", real, agent_idx=2, question_key="q", donor_source=src, audit_seed=1)
    assert src.calls == 1 and log["donor"]["pad_positions"] == 0 and log["donor"]["donor_cache_len"] == 97
    assert length(out) == 97 - round(0.25 * 97)
    assert torch.equal(out[0][0], donor[0][0][:, :, : length(out)])          # prefix of the DONOR, not the real cache
    assert not torch.equal(out[0][0], real[0][0][:, :, : length(out)])


def test_dispatch_lm_requires_donor_source_and_propagates_skip():
    real = make_kv(60)
    with pytest.raises(ValueError):
        apply_causal_audit("lm_pad10", real, agent_idx=1, question_key="q")
    with pytest.raises(DonorUnavailable) as ei:   # evaluator turns this into a skipped sample
        apply_causal_audit("lm_pad10", real, agent_idx=1, question_key="q", donor_source=_NoDonorSource())
    assert ei.value.log["skipped"] is True


def test_zero_percent_style_identity_for_unaffected_modes():
    # pre-existing modes still dispatch (regression guard on the dispatcher)
    real = make_kv(40)
    out, log = apply_causal_audit("zeroed", real, agent_idx=1, question_key="q")
    assert log == {"mode": "zeroed"} and torch.count_nonzero(out[0][0]) == 0
    out, log = apply_causal_audit("none", real, agent_idx=1, question_key="q")
    assert out is real
