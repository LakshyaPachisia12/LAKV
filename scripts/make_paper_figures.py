"""
Generates the two headline figures for docs/naacl2027_paper_draft.md from
existing result files -- no GPU, no new experiments, just plotting numbers
already established and verified earlier in this project.

Figure 1: the causal audit three-tier ordering (real > mismatched >
zeroed/random), one panel per configuration (A, D, B_int8) -- the paper's
central mechanistic result (Results, Finding 5).

Figure 2: the B_int4 fix journey -- collapse, a broken rotation-based
attempt, the diagnostic that isolated the cause, the working fix, and two
value-side variants that didn't improve on it (Results, Finding 4).

Run directly (CPU-only, reads existing JSON, no model/GPU involved):
    python scripts/make_paper_figures.py
Saves to docs/figures/fig1_causal_audit.png and
docs/figures/fig2_b_int4_journey.png.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = REPO_ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Brand-neutral, colorblind-safe palette: real=blue, mismatched=amber,
# zeroed=grey, random=red -- consistent ordering/colors across both panels
# of Figure 1 so a reader can compare configs at a glance.
COLORS = {
    "Real": "#2166AC",
    "Mismatched": "#F4A582",
    "Zeroed": "#999999",
    "Random": "#B2182B",
}


def load_summary(path, cfg):
    with open(REPO_ROOT / path) as f:
        d = json.load(f)
    return d[cfg]["summary"]


def make_figure_1():
    """Causal audit: A (results/run_20260907_222745), D and B_int8
    (results/run_20260909_080434) -- verified numbers from this session."""
    configs = [
        ("A", "results/run_20260907_222745/experiment_results.json",
         {"Real": "A", "Mismatched": "A_audit_mismatched",
          "Zeroed": "A_audit_zeroed", "Random": "A_audit_random"}),
        ("D", "results/run_20260909_080434/experiment_results.json",
         {"Real": "D", "Mismatched": "D_audit_mismatched",
          "Zeroed": "D_audit_zeroed", "Random": "D_audit_random"}),
        ("B_int8", "results/run_20260909_080434/experiment_results.json",
         {"Real": "B_int8", "Mismatched": "B_int8_audit_mismatched",
          "Zeroed": "B_int8_audit_zeroed", "Random": "B_int8_audit_random"}),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    conditions = ["Real", "Mismatched", "Zeroed", "Random"]

    for ax, (title, path, cfg_map) in zip(axes, configs):
        accs = [load_summary(path, cfg_map[c])["accuracy"] * 100 for c in conditions]
        bars = ax.bar(conditions, accs, color=[COLORS[c] for c in conditions],
                       edgecolor="black", linewidth=0.6)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, acc + 1.5, f"{acc:.0f}%",
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 65)
        ax.tick_params(axis="x", rotation=20, labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Accuracy (%)", fontsize=10)
    fig.suptitle(
        "Figure 1: Causal audit — relayed KV carries real, specific content\n"
        "(real > mismatched > zeroed = random; every pairwise comparison significant, p<0.05)",
        fontsize=10.5, y=1.03,
    )
    fig.tight_layout()
    out = FIG_DIR / "fig1_causal_audit.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def make_figure_2():
    """B_int4 fix journey -- all from n=100 (kivi/hybrid matched-subset
    numbers noted in the label since those two are n=50)."""
    steps = [
        ("B_int4\n(original)", 0.0, "Collapses"),
        ("B_int4_\nturboquant", 0.0, "Broken worse\n(rotate K+V)"),
        ("B_int4_\nturboquant_vonly", 0.0, "Diagnostic\n(rotate V only)"),
        ("B_int4_kivi\n(the fix)", 52.0, "K per-channel\n— matches D"),
        ("B_int4_\nhybrid", 50.0, "+rotate V:\nno better*"),
        ("B_int4_kivi_\nfull", 44.0, "+V per-token:\nno better*"),
    ]
    labels = [s[0] for s in steps]
    accs = [s[1] for s in steps]
    notes = [s[2] for s in steps]
    colors = ["#B2182B", "#B2182B", "#999999", "#2166AC", "#F4A582", "#F4A582"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(labels, accs, color=colors, edgecolor="black", linewidth=0.6)
    for bar, acc, note in zip(bars, accs, notes):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 1.5, f"{acc:.0f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, -6, note,
                ha="center", va="top", fontsize=7.5, color="#444444")

    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_ylim(-14, 62)
    ax.axhline(50.0, color="black", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.text(5.5, 51, "D (50.0%)", fontsize=8, style="italic", ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=8.5)
    fig.suptitle(
        "Figure 2: Diagnosing and fixing B_int4's collapse\n"
        "(*hybrid/kivi_full compared on a matched 50-example subset against kivi's own n=100 run)",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    out = FIG_DIR / "fig2_b_int4_journey.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    make_figure_1()
    make_figure_2()
