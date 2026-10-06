"""
Figure 4 for docs/latex/main_rlm_focus.tex (the RLM-focused paper draft
only -- does not touch scripts/make_paper_figures.py or any figure used
by docs/latex/main.tex, the original paper). No GPU, no new experiments --
reads already-saved result JSON and plots numbers already verified
elsewhere in this project.

The new paper's actual headline claim -- that the three-tier causal
ordering (real > mismatched > zeroed/random) holds across three
structurally distinct topologies, not only the sequential chain -- has
no figure anywhere in the draft, only a table. This is that figure: one
panel per topology (sequential chain, static fan-in, dynamic RLM+KV
delegation), same real/mismatched/zeroed/random bars as the original
paper's Figure 1, so a reader can visually compare the topology
progression the same way Figure 1 lets them compare configs.

Run directly (CPU-only, reads existing JSON, no model/GPU involved):
    python scripts/make_figure4_topology_ladder.py
Saves to docs/figures/fig4_topology_ladder.png.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = REPO_ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Same brand-neutral, colorblind-safe palette as the original paper's
# Figure 1, so the two figures read as one consistent visual language.
COLORS = {
    "Real": "#2166AC",
    "Mismatched": "#F4A582",
    "Zeroed": "#999999",
    "Random": "#B2182B",
}

CONDITIONS = ["Real", "Mismatched", "Zeroed", "Random"]


def sequential_accuracies():
    """A (uncompressed relay), sequential chain -- same source as the
    original paper's Figure 1, panel A."""
    path = REPO_ROOT / "results/run_20260907_222745/experiment_results.json"
    with open(path) as f:
        d = json.load(f)
    keys = {"Real": "A", "Mismatched": "A_audit_mismatched",
            "Zeroed": "A_audit_zeroed", "Random": "A_audit_random"}
    return [d[keys[c]]["summary"]["accuracy"] * 100 for c in CONDITIONS]


def transcripts_accuracies(path, keys):
    """Shared reader for the two topology-generalization scripts
    (recursive_poc_check.py / rlm_repl_kv_check.py), whose summary JSON
    schema is {channel: {n_correct, n, mean_f1}} rather than the main
    pipeline's {cfg: {summary: {accuracy}}}."""
    with open(REPO_ROOT / path) as f:
        d = json.load(f)
    summary = d["summary"]
    out = []
    for c in CONDITIONS:
        s = summary[keys[c]]
        out.append(100.0 * s["n_correct"] / s["n"])
    return out


def fan_in_accuracies():
    """Fan-in topology, n=50, every child audited -- the cleanest
    three-tier ladder in the project (results/recursive_poc_check/
    run_20260917_102302)."""
    path = "results/recursive_poc_check/run_20260917_102302/transcripts.json"
    keys = {"Real": "kv", "Mismatched": "kv_audit_mismatched",
            "Zeroed": "kv_audit_zeroed", "Random": "kv_audit_random"}
    return transcripts_accuracies(path, keys)


def rlm_kv_accuracies():
    """Dynamic, model-driven RLM+KV delegation, n=50, full ladder
    including mismatched (results/rlm_kv_check/run_20260916_203322)."""
    path = "results/rlm_kv_check/run_20260916_203322/transcripts.json"
    keys = {"Real": "kv", "Mismatched": "kv_audit_mismatched",
            "Zeroed": "kv_audit_zeroed", "Random": "kv_audit_random"}
    return transcripts_accuracies(path, keys)


def make_figure_4():
    panels = [
        ("Sequential chain\n(config A)", sequential_accuracies()),
        ("Static fan-in\n(cleanest ladder)", fan_in_accuracies()),
        ("Dynamic RLM+KV\ndelegation", rlm_kv_accuracies()),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)

    for ax, (title, accs) in zip(axes, panels):
        bars = ax.bar(CONDITIONS, accs, color=[COLORS[c] for c in CONDITIONS],
                       edgecolor="black", linewidth=0.6)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, acc + 1.5, f"{acc:.0f}%",
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(60, max(accs) + 12))
        ax.tick_params(axis="x", rotation=20, labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Accuracy (%)", fontsize=10)
    fig.suptitle(
        "The causal-audit ladder across three structurally\n"
        "distinct topologies -- real content beats mismatched content\n"
        "significantly in every topology tested, not only a fixed pipeline",
        fontsize=10.5, y=1.06,
    )
    fig.tight_layout()
    out = FIG_DIR / "fig4_topology_ladder.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    make_figure_4()
