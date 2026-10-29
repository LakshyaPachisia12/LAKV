"""Figure 1 for docs/latex/main_v6.tex: wrong cache vs. no cache vs. right cache.

Numbers are Qwen2.5-7B-Instruct, HotpotQA, n=100 (docs/latex/table_no_relay_v5.tex
rows "chain" and "fan-in"). Output is a column-width vector PDF (+ PNG preview).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "latex", "figures")

# Okabe-Ito colour-blind-safe palette
C_REAL, C_NONE, C_WRONG = "#0072B2", "#8C8C8C", "#D55E00"
INK = "#222222"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})

fig = plt.figure(figsize=(3.15, 3.85))

# ---------------------------------------------------------------- schematic
ax = fig.add_axes([0, 0.62, 1, 0.38])
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.axis("off")


def box(x, y, w, h, text, fc, ec=INK, tc=INK, fs=7, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=2",
                                fc=fc, ec=ec, lw=0.8, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", linespacing=1.15)


box(2, 40, 22, 14, "Agent 1\n(sender)", "#F2F2F2", bold=True)
box(76, 40, 22, 14, "Agent 2\n(receiver)", "#F2F2F2", bold=True)
ax.add_patch(FancyArrowPatch((25.5, 47), (74.5, 47), arrowstyle="-|>", mutation_scale=9,
                             lw=1.2, color=INK))
ax.text(50, 51.5, "relayed KV cache  =  ?", ha="center", va="center", fontsize=7.4,
        fontweight="bold")
ax.text(50, 41.5, "we swap in one of three:", ha="center", va="center", fontsize=6.6,
        style="italic", color="#555555")

chips = [(2, "RIGHT", "sender's own\ncache", C_REAL),
         (35.5, "NONE", "nothing sent\n(no-message ctrl)", C_NONE),
         (69, "WRONG", "another question's\nreal cache", C_WRONG)]
for x, name, sub, col in chips:
    ax.add_patch(FancyBboxPatch((x, 3), 29, 26, boxstyle="round,pad=0.4,rounding_size=2",
                                fc="white", ec=col, lw=1.4))
    ax.text(x + 14.5, 23.5, name, ha="center", va="center", fontsize=8, fontweight="bold",
            color=col)
    ax.text(x + 14.5, 11.5, sub, ha="center", va="center", fontsize=6.1, color=INK,
            linespacing=1.15)

# ------------------------------------------------------------------- bars
bx = fig.add_axes([0.13, 0.135, 0.85, 0.37])
groups = [
    ("Sequential chain", "receiver already holds\nthe evidence", [56.0, 59.0, 29.0]),
    ("Fan-in", "cache is the receiver's\nonly route to evidence", [37.0, 25.0, 8.0]),
]
w, gap = 0.26, 1.0
for gi, (_, _, vals) in enumerate(groups):
    for bi, (v, col) in enumerate(zip(vals, (C_REAL, C_NONE, C_WRONG))):
        x = gi * gap + (bi - 1) * (w + 0.03)
        bx.bar(x, v, w, color=col, zorder=3)
        bx.text(x, v + 1.6, f"{v:.0f}", ha="center", va="bottom", fontsize=7, fontweight="bold",
                color=col if col != C_NONE else "#555555")

bx.set_xticks([0, gap])
bx.set_xticklabels([t for t, _, _ in groups], fontsize=7.4, fontweight="bold")
for gi, (_, sub, _) in enumerate(groups):
    bx.text(gi * gap, -19, sub, ha="center", va="top", fontsize=6.0, style="italic",
            color="#555555", linespacing=1.15, clip_on=False)
bx.set_ylim(0, 92)
bx.set_xlim(-0.55, gap + 0.55)
bx.set_ylabel("Accuracy (%)", fontsize=7)
bx.set_yticks([0, 20, 40, 60])
bx.tick_params(axis="y", labelsize=6.5, length=2)
bx.tick_params(axis="x", length=0, pad=2)
bx.yaxis.grid(True, lw=0.4, color="#DDDDDD", zorder=0)
for s_ in ("top", "right"):
    bx.spines[s_].set_visible(False)

notes = [
    (0, "wrong < none (p<1e-4)", "real ≈ none (p=0.68)"),
    (gap, "wrong < none (p<1e-4)", "real ≥ none (p=0.050)"),
]
for x, l1, l2 in notes:
    bx.text(x, 86, l1, ha="center", va="center", fontsize=5.9, color=C_WRONG, fontweight="bold")
    bx.text(x, 78, l2, ha="center", va="center", fontsize=5.9, color="#555555", fontweight="bold")


os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "fig1_wrong_vs_none.pdf"))
fig.savefig(os.path.join(OUT, "fig1_wrong_vs_none.png"), dpi=300)
print("wrote", os.path.abspath(OUT))
