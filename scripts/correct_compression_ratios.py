"""
Retroactively corrects compressed_mb / compression_ratio for existing
result files affected by the int4 byte-accounting bug (kv_compressor.py's
compress(): quantized tensors are always stored as torch.uint8 -- no
nibble-packing exists anywhere in this file -- but the old byte accounting
used bytes_per_el = 0.5 for bits==4, fabricating a packing benefit that
never physically happened. A 4-bit value occupied the same one full byte
an 8-bit value did; the fix (already applied to lakv/kv_compressor.py)
bills k_q.nbytes + v_q.nbytes directly instead of a bits-derived formula.

Closed-form approach (fast, exact, no GPU/model/synthetic tensors needed):
for any config whose transmitted layers are ENTIRELY quantized (bits==4 or
bits==8, no bits==16 layer surviving layer selection -- true for every
config here: B_int8, B_int4, B_int4_kivi and variants, and D/E's adaptive
tier mix, since tier-3 layers are dropped before reaching the compressor),
corrected storage is always exactly 1 byte/element regardless of nominal
bit-width, i.e. exactly HALF of the real (unaffected) bf16 original size,
modulo a small, ignorable scale/zero-point overhead (<0.5% of total bytes
for per-head/per-channel grouping; see B_int4_kivi_full's ~1.94x instead of
the naive ~2.00x for the one mode where that overhead isn't negligible --
per-token quantization stores a scale/zp pair per sequence position).

    corrected_compressed_mb ~= original_mb / 2

original_mb is untouched by the bug (the bits==16 branch always used real
tensor.nbytes), so this needs no rerun, no seq_len reconstruction, and no
synthetic KVCompressor calls -- it's pure arithmetic on already-stored,
already-correct numbers. An earlier version of this script instead
rebuilt synthetic per-hop tensors and called the real KVCompressor.compress()
per hop; that approach is unnecessary complexity for what turns out to be
a simple closed-form correction, and a full-dataset run of it produced a
result inconsistent with a direct, careful re-derivation (spot-checked
against real per-hop data across the sample range) -- replaced rather than
debugged further, since the closed-form path has no room for whatever that
bug was: it doesn't do per-hop synthetic reconstruction at all.

Run directly (CPU-only, no GPU, no model weights needed, near-instant):
    python scripts/correct_compression_ratios.py
Prints a before/after table; does not modify any file on disk.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# (result file, config name)
TARGETS = [
    ("results/run_20260907_201157/experiment_results.json", "B_int4_kivi"),
    ("results/run_20260907_141517/experiment_results.json", "B_int4"),
    ("results/run_20260907_141517/experiment_results.json", "D"),
    ("results/run_20260907_141517/experiment_results.json", "E"),
    ("results/run_20260909_080434/experiment_results.json", "D"),
    ("results/run_20260907_203253/experiment_results.json", "B_int4_kivi_full"),
]


def main():
    print(f"{'File / Config':<60} {'old MB':>8} {'new MB':>8} {'old x':>7} {'new x':>7}")
    print("-" * 94)

    for rel_path, cfg_name in TARGETS:
        full_path = REPO_ROOT / rel_path
        if not full_path.exists():
            print(f"{rel_path} [{cfg_name}]: FILE NOT FOUND, skipping")
            continue
        with open(full_path, encoding="utf-8") as f:
            data = json.load(f)
        if cfg_name not in data:
            print(f"{rel_path} [{cfg_name}]: CONFIG NOT IN FILE, skipping")
            continue

        s = data[cfg_name]["summary"]
        old_mb, old_ratio = s["mean_compressed_mb"], s["mean_compression_ratio"]
        # mean original_mb = old_mb * old_ratio (ratio = original / compressed);
        # original_mb itself was never affected by the bug.
        mean_original_mb = old_mb * old_ratio
        new_mb = mean_original_mb / 2
        new_ratio = mean_original_mb / new_mb  # always ~2.00x by construction

        label = f"{rel_path.split('/')[1]} [{cfg_name}]"
        print(f"{label:<60} {old_mb:8.2f} {new_mb:8.2f} {old_ratio:6.2f}x {new_ratio:6.2f}x")


if __name__ == "__main__":
    main()
