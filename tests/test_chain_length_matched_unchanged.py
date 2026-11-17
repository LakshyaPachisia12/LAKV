"""Regression guard: the existing chain length_matched arm (DiskKVAuditPool + LengthMatchedSource
through apply_causal_audit) behaves exactly as on exp/matched-donor after the fan-in / RLM / corruption work."""
import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lakv.causal_audit import DiskKVAuditPool, NoMatchedDonor, apply_causal_audit
from lakv.donor_traces import LengthMatchedSource
from lakv.evaluator import AUDIT_MODE_SUFFIXES, parse_audit_config_name


def make_kv(L, seed=0):
    g = torch.Generator().manual_seed(seed)
    return tuple((torch.randn(1, 2, L, 8, generator=g), torch.randn(1, 2, L, 8, generator=g)) for _ in range(2))


# the exact log keys the arm emitted before the new work (see exp/matched-donor, causal_audit.py)
EXPECTED_LOG_KEYS = {
    "agent_idx", "skipped", "donor_question_key", "donor_question", "donor_pool_order", "donor_cache_len",
    "target_cache_len", "len_ratio", "handling", "pad_positions", "truncated_positions", "tolerance",
    "n_eligible", "pool_size", "mode",
}


@pytest.fixture
def pool(tmp_path):
    p = DiskKVAuditPool(tmp_path / "p")
    for i, L in enumerate((100, 103, 150)):
        p.add(1, f"d{i}", make_kv(L, seed=i), question_text=f"d{i}")
    p.mark_complete()
    return p


def test_chain_log_keys_and_donor_unmodified(pool):
    out, log = apply_causal_audit("length_matched", make_kv(101, seed=9), agent_idx=1, question_key="s",
                                  donor_source=LengthMatchedSource(pool), rng=random.Random(0))
    assert set(log) == EXPECTED_LOG_KEYS            # in particular no donor_prompt_len on the chain path
    assert log["handling"] in ("exact", "unmodified_within_tolerance")
    assert log["pad_positions"] == 0 and log["truncated_positions"] == 0
    seed = {"d0": 0, "d1": 1}[log["donor_question_key"]]
    ref = make_kv(log["donor_cache_len"], seed=seed)
    for (k, v), (kr, vr) in zip(out, ref):
        assert torch.equal(k, kr) and torch.equal(v, vr)


def test_chain_donor_choice_unchanged_for_shared_seeded_rng(pool):
    # the chain still draws with the pipeline's single seeded stream: same seed -> same sequence of donors
    def draws(seed):
        rng = random.Random(seed)
        return [apply_causal_audit("length_matched", make_kv(101), agent_idx=1, question_key="s",
                                   donor_source=LengthMatchedSource(pool), rng=rng)[1]["donor_question_key"]
                for _ in range(8)]
    assert draws(0) == draws(0)
    assert random.Random(0).choice(sorted(["d0", "d1"])) in ("d0", "d1")


def test_chain_skip_and_names_unchanged(pool):
    with pytest.raises(NoMatchedDonor):
        apply_causal_audit("length_matched", make_kv(300), agent_idx=1, question_key="s",
                           donor_source=LengthMatchedSource(pool), rng=random.Random(0))
    assert AUDIT_MODE_SUFFIXES["_audit_length_matched"] == "length_matched"
    assert parse_audit_config_name("A_nopass_audit_length_matched") == ("A", "A_nopass", "length_matched", True)
    assert parse_audit_config_name("A_audit_self_replay") == ("A", "A", "self_replay", False)
