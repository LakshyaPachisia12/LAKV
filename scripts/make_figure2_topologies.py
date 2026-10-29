"""Figure 2 for docs/latex/main_v6.tex: the three relay topologies and where the
receiver has an independent path to the evidence. Output: full-width vector PDF.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "latex", "figures")
C_REAL, C_WRONG, C_OK, INK = "#0072B2", "#D55E00", "#2E7D32", "#222222"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5})

fig = plt.figure(figsize=(6.3, 2.25))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 126)
ax.set_ylim(0, 45)
ax.axis("off")


def box(x, y, w, h, text, fc="#F4F4F4", ec=INK, fs=7.2, bold=True, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc, linespacing=1.1)


def arrow(p0, p1, col=C_REAL, lw=1.6, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=8, lw=lw,
                                 color=col, ls=ls, shrinkA=0, shrinkB=0))


def tag(x, y, text, col):
    ax.text(x, y, text, ha="center", va="center", fontsize=6.4, color="white",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc=col, ec="none"))


# ---------------------------------------------------------------- panel titles
for x, t in [(20.5, "(a) Sequential chain"), (63, "(b) Static fan-in"), (105.5, "(c) Model-driven delegation")]:
    ax.text(x, 42, t, ha="center", va="center", fontsize=8.2, fontweight="bold")
for x in (41.5, 84):
    ax.plot([x, x], [3, 40], color="#CCCCCC", lw=0.7)

# ------------------------------------------------------------------ (a) chain
box(0.8, 22, 12.4, 8, "Reasoner", fs=6.6)
box(15.3, 22, 11.4, 8, "Verifier", fs=6.6)
box(28.8, 22, 12.2, 8, "Finalizer", fs=6.6)
arrow((12.9, 26), (15.1, 26))
arrow((26.9, 26), (29.1, 26))
ax.text(14, 33.2, "cache", ha="center", fontsize=6.2, color=C_REAL, fontweight="bold")
ax.text(28, 33.2, "cache", ha="center", fontsize=6.2, color=C_REAL, fontweight="bold")
for x in (21, 35):
    arrow((x, 19.5), (x, 21.6), col=C_OK, lw=1.2)
ax.text(28, 14.5, "passages already in\neach receiver's own prompt", ha="center", va="center",
        fontsize=6.4, color=C_OK, linespacing=1.15)
tag(21, 7.5, "other route to evidence", C_OK)

# ----------------------------------------------------------------- (b) fan-in
box(44.5, 29, 15, 8, "Child 1\npassages 1–5", fs=6.8)
box(44.5, 15, 15, 8, "Child 2\npassages 6–10", fs=6.8)
box(70, 21, 13, 10, "Aggregator", fs=7.2)
arrow((60, 33), (69.6, 28.5))
arrow((60, 19), (69.6, 23.5))
ax.text(72, 35.5, "caches concatenated", ha="center", fontsize=6.2, color=C_REAL,
        fontweight="bold")
ax.text(76.5, 17.5, "sees only the\nquestion + cache", ha="center", va="center", fontsize=6.4,
        color=C_WRONG, linespacing=1.15)
tag(63, 7.5, "cache is the only route", C_WRONG)

# ------------------------------------------------------------- (c) delegation
box(88, 20, 17, 12, "Root model\n+ code REPL", fs=7.0)
box(111, 24, 13, 8, "Sub-call", fs=7.0)
arrow((105.4, 29), (110.6, 29), col="#777777", lw=1.2)
ax.text(108, 33.0, "delegate", ha="center", fontsize=5.8, color="#777777")
arrow((110.6, 26), (105.4, 26), col=C_REAL)
ax.text(116, 20.3, "answer cache\nspliced onto root", ha="center", va="center", fontsize=5.9,
        color=C_REAL, fontweight="bold", linespacing=1.1)
ax.text(105, 14.5, "root can also read the\npassages directly in code", ha="center", va="center",
        fontsize=6.4, color=C_OK, linespacing=1.15)
tag(105.5, 7.5, "other route to evidence", C_OK)

# legend strip
ax.text(1.5, 1.2, "blue arrow = relayed KV cache (the slot we audit)    green = independent path to the "
        "source passages", fontsize=6.0, color="#555555", va="center")

os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "fig2_topologies.pdf"))
fig.savefig(os.path.join(OUT, "fig2_topologies.png"), dpi=300)
print("ok")
