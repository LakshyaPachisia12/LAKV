"""Compression figure (appendix) for docs/latex/main_v8.tex: does the
real-vs-mismatched gap survive compression? Data: Appendix Table (causal audit
ladder), accuracy %, n=50. Compression ratios are measured on Qwen2.5-7B
(A = 1x, int8 = 2.00x, KIVI-style int4 = 3.96x; vs A's 144.35 MB/hop).
Naive int4 (3.96-4.00x) collapses to 0.0% accuracy and cannot be audited
(stated in the main text and not repeated in the figure).

Built at printed size (3.0 in wide, single column) so all text is >= 7 pt.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "latex", "figures")
BLUE, ORANGE, INK = "#0072B2", "#D55E00", "#222222"
FS = 7.0
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FS})

CONFIGS = ["none\n1.00×", "int8\n2.00×", "int4\n3.96×"]   # int4 = KIVI-style
DATA = {  # model -> (real, mismatched) per config
    "Qwen2.5-7B": ([50.0, 54.0, 50.0], [28.0, 24.0, 28.0]),
    "Mistral-7B": ([56.0, 56.0, 54.0], [16.0, 20.0, 16.0]),
}

fig, axes = plt.subplots(1, 2, figsize=(3.0, 2.55), sharey=True)
fig.subplots_adjust(left=0.15, right=0.99, top=0.70, bottom=0.20, wspace=0.10)
w = 0.38
for ax, (model, (real, mism)) in zip(axes, DATA.items()):
    xs = range(len(CONFIGS))
    for i in xs:
        ax.bar(i - w / 2, real[i], w, color=BLUE, zorder=3)
        ax.bar(i + w / 2, mism[i], w, color=ORANGE, zorder=3)
        ax.text(i - w / 2, real[i] + 1.5, f"{real[i]:.0f}", ha="center", va="bottom", fontsize=FS, color=BLUE,
                fontweight="bold")
        ax.text(i + w / 2, mism[i] + 1.5, f"{mism[i]:.0f}", ha="center", va="bottom", fontsize=FS, color=ORANGE,
                fontweight="bold")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(CONFIGS, fontsize=FS)
    ax.set_title(model, fontsize=FS, fontweight="bold", pad=3)
    ax.set_ylim(0, 72)
    ax.set_yticks([0, 20, 40, 60])
    ax.yaxis.grid(True, lw=0.4, color="#E3E3E3", zorder=0)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.tick_params(axis="y", labelsize=FS, length=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("Exact match (%)", fontsize=FS)

fig.text(0.02, 0.985, "Compression keeps the real vs. wrong gap", fontsize=7.6, fontweight="bold",
         va="top", ha="left")
ax_leg = fig.add_axes([0, 0.79, 1, 0.09])
ax_leg.axis("off")
ax_leg.set_xlim(0, 1)
ax_leg.set_ylim(0, 1)
ax_leg.add_patch(plt.Rectangle((0.03, 0.15), 0.05, 0.7, color=BLUE))
ax_leg.text(0.10, 0.5, "real cache", fontsize=FS, va="center")
ax_leg.add_patch(plt.Rectangle((0.40, 0.15), 0.05, 0.7, color=ORANGE))
ax_leg.text(0.47, 0.5, "wrong cache (mismatched)", fontsize=FS, va="center")
fig.text(0.02, 0.015, "n=50 per bar; int4 = KIVI-style per-channel keys", fontsize=FS, color="#555555",
         va="bottom", ha="left")

os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "fig3_compression.pdf"))
fig.savefig(os.path.join(OUT, "fig3_compression.png"), dpi=250)
print("ok")
