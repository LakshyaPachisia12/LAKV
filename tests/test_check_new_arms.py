"""The new-arm checker accepts correct logs and flags violations (CPU only)."""
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lakv.cache_corruption import apply_corruption_mode


def run_checker(tmp_path, payload, *extra):
    f = tmp_path / "x.json"
    f.write_text(json.dumps(payload), encoding="utf-8")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_new_arms.py"), str(f), *extra],
                       capture_output=True, text=True)
    return r.returncode, r.stdout


def kv(L):
    return tuple((torch.randn(1, 2, L, 8), torch.randn(1, 2, L, 8)) for _ in range(2))


def chain_payload(break_it=False):
    _, log = apply_corruption_mode("real_pad25", kv(60), "q", 1)
    log = {"mode": "real_pad25", **log}
    if break_it:
        log["n_modified"] += 1
    return {"A_nopass_audit_real_pad25": {"per_sample": [{"idx": 0, "audit_logs": [log]}]}}


def test_chain_corruption_logs_pass_and_fail(tmp_path):
    assert run_checker(tmp_path, chain_payload())[0] == 0
    code, out = run_checker(tmp_path, chain_payload(break_it=True))
    assert code == 1 and "VIOLATION" in out


def rlm_payload(donor_len=40, pad=0, skipped=0):
    recs = [{"idx": 0, "audit_logs": [{"mode": "length_matched", "donor_cache_len": donor_len, "real_answer_len": 41,
                                       "pad_positions": pad, "truncated_positions": 0, "donor_prompt_len": 100}]}]
    recs += [{"idx": 1 + i, "skipped": True, "audit_logs": []} for i in range(skipped)]
    return {"records": {"kv_audit_length_matched": recs}}


def test_rlm_transcripts(tmp_path):
    assert run_checker(tmp_path, rlm_payload())[0] == 0
    assert run_checker(tmp_path, rlm_payload(donor_len=30))[0] == 1          # outside 5%
    assert run_checker(tmp_path, rlm_payload(pad=3))[0] == 1                 # padded donor
    assert run_checker(tmp_path, rlm_payload(skipped=3))[0] == 1             # 75% skipped > 25%
    assert run_checker(tmp_path, rlm_payload(skipped=3), "--max_skip", "0.9")[0] == 0
