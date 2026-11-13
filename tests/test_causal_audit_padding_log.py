"""Mismatched-audit instrumentation: logs how a donor cache was length-matched.

Logging only - these tests also pin that substitution behavior is unchanged.
"""
import torch

from lakv.causal_audit import (KVAuditPool, _match_seq_len, apply_causal_audit,
                               describe_seq_len_handling)


def kv(seq, layers=2, fill=1.0):
    return tuple((torch.full((1, 2, seq, 4), fill), torch.full((1, 2, seq, 4), -fill)) for _ in range(layers))


def test_describe_exact_truncated_padded():
    e = describe_seq_len_handling(10, 10)
    assert (e["handling"], e["pad_positions"], e["pad_fraction"], e["truncated_positions"]) == ("exact", 0, 0.0, 0)
    t = describe_seq_len_handling(12, 10)
    assert (t["handling"], t["truncated_positions"], t["pad_fraction"]) == ("truncated", 2, 0.0)
    assert abs(t["truncated_fraction"] - 2 / 12) < 1e-9
    p = describe_seq_len_handling(6, 10)
    assert (p["handling"], p["pad_positions"], p["pad_fraction"]) == ("zero_padded", 4, 0.4)
    assert p["donor_cache_len"] == 6 and p["target_cache_len"] == 10


def test_description_agrees_with_what_match_seq_len_actually_does():
    for donor_len, target_len in [(10, 10), (12, 10), (6, 10)]:
        donor = torch.ones(1, 2, donor_len, 4)
        out, resized = _match_seq_len(donor, torch.Size([1, 2, target_len, 4]))
        d = describe_seq_len_handling(donor_len, target_len)
        assert out.shape[2] == target_len
        assert resized == (d["handling"] != "exact")
        assert int((out.abs().sum(dim=(0, 1, 3)) == 0).sum()) == d["pad_positions"]


def _pool():
    pool = KVAuditPool(max_size_per_agent=5)
    pool.add(1, "donorA", kv(6), question_text="donor question A")
    pool.donor_heldout_offsets = {"donorA": 7}
    return pool


def test_mismatched_log_records_donor_and_padding():
    pool = _pool()
    real = kv(10)
    out, log = apply_causal_audit("mismatched", real, 1, "mine", pool=pool)
    assert log["mode"] == "mismatched" and log["donor_question"] == "donor question A"
    assert log["handling"] == "zero_padded" and log["pad_positions"] == 4 and log["pad_fraction"] == 0.4
    assert log["donor_cache_len"] == 6 and log["target_cache_len"] == 10
    assert log["donor_question_key"] == "donorA" and log["donor_pool_order"] == 0
    assert log["donor_heldout_offset"] == 7 and log["shape_resized"] is True
    # behavior unchanged: tail is zeros, head is the donor's real values
    assert torch.all(out[0][0][:, :, 6:, :] == 0) and torch.all(out[0][0][:, :, :6, :] == 1.0)


def test_sample_keeps_original_three_tuple_interface():
    pool = _pool()
    res = pool.sample(1, "mine", torch.Size([1, 2, 10, 4]), "cpu")
    assert len(res) == 3 and res[2] == "donor question A"


def test_log_is_json_serialisable():
    import json
    _, log = apply_causal_audit("mismatched", kv(10), 1, "mine", pool=_pool())
    json.dumps(log)


def test_other_modes_unchanged():
    _, log = apply_causal_audit("zeroed", kv(4), 1, "mine")
    assert log == {"mode": "zeroed"}
