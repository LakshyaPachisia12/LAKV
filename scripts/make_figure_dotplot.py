"""Figure 2 (single column) for docs/latex/main_v8.tex: dot-arrow plot of real /
dropped (no cache) / mismatched (wrong cache) accuracy per model and topology.
Data: docs/latex/table_no_relay (Table 2 of the paper) and the passage-access
table. When reruns land, update ROWS and re-run.

Built at printed size in inches (3.0 in wide), so all text is >= 7 pt as
printed. The per-row "real vs none" notes of the earlier version were dropped
(they are in the tables); the arrow p-value stays.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "latex", "figures")
BLUE, ORANGE, INK, GREEN, PURPLE = "#0072B2", "#D55E00", "#222222", "#1B8A5A", "#7B3294"
FS = 7.0
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FS})

# label, sublabel, n, real, dropped, mismatched, p(dropped vs mismatched), kind: "sole" | "redundant" | "null"
ROWS = [
    ("Qwen2.5 fan-in", "cache is only route", 100, 37.0, 25.0, 8.0, "p=7.6e-5", "sole"),
    ("Qwen2.5 chain", "passages hidden", 100, 52.0, 20.0, 7.0, "p=1e-3", "sole"),
    ("Qwen2.5 fan-in", "reads passages too", 100, 55.0, 57.0, 30.0, "p=1e-5", "redundant"),
    ("Qwen2.5 chain", "passages in prompt", 100, 56.0, 59.0, 29.0, "p<1e-4", "redundant"),
    ("Mistral-7B chain", "passages in prompt", 100, 53.0, 49.0, 26.0, "p=2e-4", "redundant"),
    ("Qwen3-8B chain", "non-thinking", 100, 63.0, 62.0, 62.0, "n.s.", "null"),
]

W, H = 3.0, 3.42
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

X0, SCALE = 0.12, 2.8 / 68.0       # inches of x for accuracy 0 and per accuracy point


def xin(v):
    return X0 + v * SCALE


PITCH, Y_FIRST = 0.38, 2.50
AXIS_Y = Y_FIRST - 5 * PITCH - 0.30

# header
ax.text(0.04, H - 0.04, "The wrong KV cache is worse than no cache", fontsize=7.6, fontweight="bold",
        va="top", ha="left")
ax.text(0.04, H - 0.26, "arrow: no cache → wrong cache (exact McNemar p)", fontsize=FS, color=ORANGE,
        va="top", ha="left")
ly = H - 0.52
ax.scatter([0.11], [ly], s=30, color=BLUE, edgecolor="white", linewidth=0.5, zorder=5)
ax.text(0.19, ly, "real cache", fontsize=FS, va="center")
ax.scatter([1.07], [ly], s=26, marker="s", facecolor="white", edgecolor=INK, linewidth=1.1, zorder=5)
ax.text(1.15, ly, "no cache", fontsize=FS, va="center")
ax.scatter([1.85], [ly], s=34, marker="X", color=ORANGE, linewidth=0.2, edgecolor="white", zorder=5)
ax.text(1.93, ly, "wrong cache", fontsize=FS, va="center")

# gridlines + axis
for v in (0, 20, 40, 60):
    ax.plot([xin(v), xin(v)], [AXIS_Y, Y_FIRST + 0.22], color="#E3E3E3", lw=0.4, zorder=0)
    ax.text(xin(v), AXIS_Y - 0.05, f"{v}", fontsize=FS, ha="center", va="top")
ax.plot([xin(0), xin(68)], [AXIS_Y, AXIS_Y], color=INK, lw=0.6)
ax.text(xin(34), 0.02, "HotpotQA exact-match accuracy (%)", fontsize=FS, ha="center", va="bottom")

for i, (lab, sub, n, real, none, wrong, p, kind) in enumerate(ROWS):
    yc = Y_FIRST - i * PITCH
    ym = yc - 0.07            # marker line
    yl = yc + 0.13            # label line
    if kind == "sole":
        ax.add_patch(Rectangle((0.0, yc - 0.18), W, 0.38, fc="#E6F0F8", ec="none", zorder=0))
    grey = kind == "null"
    c_real = "#9AA7B2" if grey else BLUE
    c_none = "#9A9A9A" if grey else INK
    c_wrong = "#C9A58C" if grey else ORANGE
    if not grey:
        ax.annotate("", xy=(xin(wrong), ym), xytext=(xin(none), ym),
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.3, shrinkA=3.2, shrinkB=3.8), zorder=3)
    ax.scatter([xin(real)], [ym], s=30, color=c_real, zorder=5, edgecolor="white", linewidth=0.5)
    ax.scatter([xin(none)], [ym], s=26, marker="s", facecolor="white", edgecolor=c_none, linewidth=1.1, zorder=5)
    ax.scatter([xin(wrong)], [ym], s=34, marker="X", color=c_wrong, zorder=5, linewidth=0.2, edgecolor="white")
    if i == 0:  # text relay reference point in the sole-route fan-in
        ax.scatter([xin(51.0)], [ym], s=34, marker="D", color=PURPLE, zorder=5, edgecolor="white", linewidth=0.5)
        ax.text(xin(51.0) + 0.08, ym, "text relay", fontsize=FS, color=PURPLE, fontweight="bold",
                va="center", ha="left")
    # label line: bold name, then plain sublabel; p-value right-aligned
    t = ax.text(0.04, yl, lab, fontsize=FS, fontweight="bold", va="center", ha="left", color=INK)
    fig.canvas.draw()
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    w_in = bb.width / fig.dpi
    ax.text(0.04 + w_in + 0.06, yl, sub, fontsize=FS, va="center", ha="left", color="#555555", style="italic")
    ax.text(W - 0.04, yl, p, fontsize=FS, va="center", ha="right", color="#8A8A8A" if grey else ORANGE,
            fontweight="bold")

os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "fig2_dotplot.pdf"))
fig.savefig(os.path.join(OUT, "fig2_dotplot.png"), dpi=250)
print("ok")
