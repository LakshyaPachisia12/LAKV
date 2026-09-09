"""
Retroactively corrects compressed_mb / compression_ratio for existing
result files affected by the int4 byte-accounting bug (kv_compressor.py's
compress(): quantized tensors are always stored as torch.uint8 -- no
nibble-packing exists anywhere in this file -- but the old byte accounting
used bytes_per_el = 0.5 for bits==4, fabricating a packing benefit that
never physically happened. A 4-bit value occupied the same one full byte
an 8-bit value did; the fix (already applied to lakv/kv_compressor.py)
bills k_q.nbytes + v_q.nbytes directly instead of a bits-derived formula.

This script does NOT need a GPU or the real model: compressed size is a
pure function of tensor shape and which layers were assigned which bits,
not of the actual K/V values. For each hop of each affected sample, it:
  1. Backs out the real seq_len from that hop's stored (unaffected)
     original_mb, using known Qwen2.5-7B-Instruct KV-cache dims
     (num_key_value_heads=4, head_dim=128, confirmed via
     model.config in this codebase's calibration_profiler.py).
  2. Reconstructs synthetic K/V tensors of that exact shape and re-runs
     the real KVCompressor.compress() (now fixed) to get the correct
     compressed_bytes for that hop -- exact, not an approximation, since
     compressed size depends only on shape/bits, never on values.
  3. Recomputes each sample's overall compressed_mb/compression_ratio as
     the sum/ratio across its corrected hops, and each config's summary
     mean_compressed_mb/mean_compression_ratio as the mean across samples.

For adaptive-mode configs (D, D_nearest, D_interpolate, E), it needs the
calibration profile's tier_assignment (which layer got int8 vs int4).
The three Qwen/HotpotQA profiles that predate/coincide with every run
corrected here have IDENTICAL layer-by-layer tier_assignment (verified
before writing this script), so using any one of them is exact, not a
guess -- see PROFILE_PATH below.

Run directly (CPU-only, no GPU, no model weights needed):
    python scripts/correct_compression_ratios.py
Prints a before/after table; does not modify any file on disk.
"""

import json
from pathlib import Path

import torch

from lakv.kv_compressor import KVCompressor

REPO_ROOT = Path(__file__).resolve().parent.parent
N_KV_HEADS = 4
HEAD_DIM = 128
BF16_BYTES = 2
PROFILE_PATH = REPO_ROOT / "profiles" / "run_20260904_115736" / "qwen_hotpotqa.json"

# (result file, config name, compressor mode, is_adaptive)
TARGETS = [
    ("results/run_20260907_201157/experiment_results.json", "B_int4_kivi",
     "uniform_int4_kivi_k_channel", False),
    ("results/run_20260907_141517/experiment_results.json", "B_int4",
     "uniform_int4", False),
    ("results/run_20260907_141517/experiment_results.json", "D",
     "adaptive", True),
    ("results/run_20260907_141517/experiment_results.json", "E",
     "adaptive", True),
    ("results/run_20260909_080434/experiment_results.json", "D",
     "adaptive", True),
    ("results/run_20260907_203253/experiment_results.json", "B_int4_kivi_full",
     "uniform_int4_kivi_full", False),
]


def load_tier_assignment():
    with open(PROFILE_PATH, encoding="utf-8") as f:
        p = json.load(f)
    return {i: t for i, t in enumerate(p["tier_assignment"])}


def n_el_from_original_mb(original_mb, n_layers_transmitted):
    # original_mb always uses real bf16 nbytes (unaffected by the bug):
    # original_bytes = n_layers * 2 (K & V) * n_kv_heads * head_dim * seq_len * 2 bytes
    original_bytes = original_mb * 1e6
    per_layer_per_tensor_bytes = original_bytes / (n_layers_transmitted * 2)
    seq_len = per_layer_per_tensor_bytes / (N_KV_HEADS * HEAD_DIM * BF16_BYTES)
    return round(seq_len)


def corrected_compressed_mb(original_mb, n_layers_transmitted, mode, adaptive_tier_info=None):
    seq_len = n_el_from_original_mb(original_mb, n_layers_transmitted)
    if adaptive_tier_info is not None:
        transmitted = [i for i, t in adaptive_tier_info.items() if t != 3][:n_layers_transmitted]
        layer_indices = transmitted
    else:
        layer_indices = list(range(n_layers_transmitted))

    pkv = tuple(
        (torch.randn(1, N_KV_HEADS, seq_len, HEAD_DIM, dtype=torch.bfloat16),
         torch.randn(1, N_KV_HEADS, seq_len, HEAD_DIM, dtype=torch.bfloat16))
        for _ in layer_indices
    )
    comp = KVCompressor(mode=mode, profile=(object() if adaptive_tier_info is not None else None))
    msg = comp.compress(pkv, tier_info=adaptive_tier_info, layer_indices=layer_indices)
    return msg.compressed_bytes / 1e6, seq_len


def main():
    tier_assignment = load_tier_assignment()
    print(f"{'File / Config':<70} {'old MB':>8} {'new MB':>8} {'old x':>7} {'new x':>7}")
    print("-" * 104)

    for rel_path, cfg_name, mode, is_adaptive in TARGETS:
        full_path = REPO_ROOT / rel_path
        if not full_path.exists():
            print(f"{rel_path} [{cfg_name}]: FILE NOT FOUND, skipping")
            continue
        with open(full_path, encoding="utf-8") as f:
            data = json.load(f)
        if cfg_name not in data:
            print(f"{rel_path} [{cfg_name}]: CONFIG NOT IN FILE, skipping")
            continue

        cfg = data[cfg_name]
        old_mean_mb = cfg["summary"]["mean_compressed_mb"]
        old_mean_ratio = cfg["summary"]["mean_compression_ratio"]

        new_compressed_mb_per_sample = []
        new_ratio_per_sample = []
        for sample in cfg["per_sample"]:
            total_new_compressed_bytes = 0.0
            total_original_bytes = 0.0
            for hop in sample["hop_stats"]:
                tier_info = tier_assignment if is_adaptive else None
                new_mb, _ = corrected_compressed_mb(
                    hop["original_mb"], hop["n_layers_transmitted"], mode, tier_info
                )
                total_new_compressed_bytes += new_mb * 1e6
                total_original_bytes += hop["original_mb"] * 1e6
            new_compressed_mb_per_sample.append(total_new_compressed_bytes / 1e6)
            new_ratio_per_sample.append(total_original_bytes / max(total_new_compressed_bytes, 1e-9))

        new_mean_mb = sum(new_compressed_mb_per_sample) / len(new_compressed_mb_per_sample)
        new_mean_ratio = sum(new_ratio_per_sample) / len(new_ratio_per_sample)

        label = f"{rel_path.split('/')[1]} [{cfg_name}]"
        print(f"{label:<70} {old_mean_mb:8.2f} {new_mean_mb:8.2f} {old_mean_ratio:6.2f}x {new_mean_ratio:6.2f}x")


if __name__ == "__main__":
    main()
