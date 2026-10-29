"""Page-1 headline figure (single column, plain figure[t]) for docs/latex/main_v8.tex.

Grouped bars of exact-match accuracy for real / dropped (no cache) / mismatched
(wrong cache) in the sequential chain and the fan-in. Qwen2.5-7B-Instruct,
HotpotQA, n=100. Numbers are copied from the paper's tables (Table 2 and the
standard rows of the passage-access table); update only if those change.

Built at printed size (3.0 x 2.2 in) so all text is >= 7 pt as printed. Colours
match Figure 2 (real = blue, mismatched = orange, dropped = neutral grey).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "latex", "figures")
BLUE, ORANGE, GREY, INK = "#0072B2", "#D55E00", "#9A9A9A", "#222222"
FS = 7.0
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FS})

GROUPS = ["Sequential chain", "Fan-in"]
# (real, dropped, mismatched), exact match %, Qwen2.5-7B, HotpotQA, n=100
DATA = {
    "Sequential chain": (56.0, 59.0, 29.0),
    "Fan-in": (37.0, 25.0, 8.0),
}
COLORS = [BLUE, GREY, ORANGE]
NAMES = ["real", "dropped (no cache)", "mismatched"]

fig, ax = plt.subplots(figsize=(3.0, 2.2))
fig.subplots_adjust(left=0.15, right=0.99, top=0.83, bottom=0.13)
w = 0.26
for gi, g in enumerate(GROUPS):
    for bi, (v, c) in enumerate(zip(DATA[g], COLORS)):
        x = gi + (bi - 1) * (w + 0.02)
        ax.bar(x, v, w, color=c, edgecolor=INK if c == GREY else "none", linewidth=0.5, zorder=3)
        ax.text(x, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=FS, color=INK if c == GREY else c,
                fontweight="bold")
ax.set_xticks(range(len(GROUPS)))
ax.set_xticklabels(GROUPS, fontsize=FS)
ax.set_ylim(0, 72)
ax.set_xlim(-0.55, len(GROUPS) - 0.45)
ax.set_yticks([0, 20, 40, 60])
ax.set_ylabel("Exact match (%)", fontsize=FS)
ax.yaxis.grid(True, lw=0.4, color="#E3E3E3", zorder=0)
ax.tick_params(axis="x", length=0, pad=3)
ax.tick_params(axis="y", labelsize=FS, length=2)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# legend row above the plot
ax_leg = fig.add_axes([0, 0.88, 1, 0.1])
ax_leg.axis("off")
ax_leg.set_xlim(0, 3.0)
ax_leg.set_ylim(0, 1)
x = 0.12
for name, c in zip(NAMES, COLORS):
    ax_leg.add_patch(plt.Rectangle((x, 0.2), 0.12, 0.6, fc=c, ec=INK if c == GREY else "none", lw=0.5))
    ax_leg.text(x + 0.17, 0.5, name, fontsize=FS, va="center")
    x += 0.17 + 0.058 * len(name) + 0.12

os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "fig_headline.pdf"))
fig.savefig(os.path.join(OUT, "fig_headline.png"), dpi=250)
print("ok")
