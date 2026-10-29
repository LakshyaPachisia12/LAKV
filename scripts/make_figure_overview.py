"""Figure 1 (full width, plain figure* float at the top of page 1) for
docs/latex/main_v8.tex.

Left: the five audit conditions. Right: the three relay topologies, each tagged
with whether the receiver has a redundant route to the evidence. All
coordinates are in inches and the figure is built at its printed size (6.3 in
wide), so every piece of text is >= 7 pt as printed. The compressed-relay strip
of the earlier version was dropped to keep text at 7 pt.

Framing note: the dropped (no-cache) control follows Zhang & Emu (2026)'s
no-message baseline, so it is tagged "no-cache control", not "new".
Output: vector PDF + PNG preview.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "latex", "figures")
BLUE, ORANGE, GREEN, INK, GREY = "#0072B2", "#D55E00", "#1B8A5A", "#222222", "#555555"
FS = 7.0       # body text in the figure (pt, as printed)
FSB = 7.6      # panel titles
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FS, "hatch.linewidth": 0.5})

W, H = 6.3, 3.12
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def rbox(x, y, w, h, fc="white", ec=INK, lw=0.8, ls="-", r=0.05, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def txt(x, y, s, fs=FS, ha="center", va="center", bold=False, italic=False, color=INK, z=6, rot=0):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=color, zorder=z, rotation=rot,
            fontweight="bold" if bold else "normal", style="italic" if italic else "normal",
            linespacing=1.1)


def arrow(p0, p1, color=INK, lw=0.9, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, lw=lw, color=color,
                                 shrinkA=0, shrinkB=0, zorder=3, connectionstyle=f"arc3,rad={rad}"))


def agent(x, yc, w, label, h=0.3):
    rbox(x, yc - h / 2, w, h, fc="#F3F3F3")
    txt(x + w / 2, yc, label, bold=True)


def cache_block(x, yc, w=0.26, h=0.2):
    """The audit point: one block striped with the four substitutes."""
    y = yc - h / 2
    seg = w / 4.0
    ax.add_patch(Rectangle((x, y), seg, h, fc=BLUE, ec="none", zorder=4))
    ax.add_patch(Rectangle((x + seg, y), seg, h, fc="white", ec="none", zorder=4))
    ax.add_patch(Rectangle((x + 2 * seg, y), seg, h, fc="white", ec=GREY, hatch="....", lw=0, zorder=4))
    ax.add_patch(Rectangle((x + 3 * seg, y), seg, h, fc=ORANGE, ec="none", zorder=4))
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=INK, lw=0.8, zorder=5))


def qp(x, y):
    """'question + passages' sticker (stacked yellow cards); (x, y) = lower-left."""
    for d in (0.05, 0.025, 0.0):
        ax.add_patch(Rectangle((x + d, y + d), 0.3, 0.17, fc="#FFF3C4", ec="#B38B00", lw=0.6, zorder=3))
    txt(x + 0.15, y + 0.085, "Q+P", fs=FS - 0.6, bold=True, color="#7A5C00", z=8)


def pill(x_right, y, s, color):
    w = 0.062 * len(s) + 0.14
    rbox(x_right - w, y - 0.1, w, 0.2, fc="white", ec=color, lw=0.9, r=0.04)
    txt(x_right - w / 2, y, s, bold=True, color=color)


# ============================================================ LEFT: five conditions
LX, LW = 0.04, 1.96
rbox(LX, 0.04, LW, H - 0.08, fc="white", ec=INK, lw=0.9, r=0.07)
txt(LX + LW / 2, 2.95, "Five audit conditions", fs=FSB, bold=True)

# sender -> [striped cache] -> receiver
rbox(0.14, 2.43, 0.58, 0.28, fc="#F3F3F3")
txt(0.43, 2.57, "sender", bold=True)
arrow((0.72, 2.57), (0.86, 2.57))
cache_block(0.87, 2.57)
arrow((1.13, 2.57), (1.27, 2.57))
rbox(1.27, 2.43, 0.65, 0.28, fc="#F3F3F3")
txt(1.595, 2.57, "receiver", bold=True)
txt(LX + LW / 2, 2.25, "striped block = one of:", italic=True, color=GREY)

items = [("real", "own cache", "blue"), ("zeroed", "all zeros", "white"),
         ("random", "matched noise", "hatch"), ("mismatched", "another question's", "orange")]
y0, dy = 1.97, 0.33
for i, (name, desc, kind) in enumerate(items):
    y = y0 - i * dy
    kw = dict(ec=INK, lw=0.6, zorder=4)
    fc = {"blue": BLUE, "white": "white", "hatch": "white", "orange": ORANGE}[kind]
    ax.add_patch(Rectangle((0.14, y - 0.09), 0.24, 0.18, fc=fc, hatch="...." if kind == "hatch" else None, **kw))
    txt(0.46, y + 0.075, name, ha="left", bold=True)
    txt(0.46, y - 0.075, desc, ha="left", color=GREY)

bx = 1.74
ytop, ybot = y0 + 0.15, y0 - 3 * dy - 0.15
ax.plot([bx - 0.05, bx, bx, bx - 0.05], [ytop, ytop, ybot, ybot], color=INK, lw=0.7, zorder=3)
txt(bx + 0.13, (ytop + ybot) / 2, "audit of Cheng et al.", rot=90, fs=FS)

# dropped (no-cache control)
rbox(0.1, 0.1, 1.84, 0.72, fc="#EAF6F0", ec=GREEN, lw=1.2, ls=(0, (3, 2)), r=0.06)
ax.add_patch(Rectangle((0.17, 0.57), 0.24, 0.18, fc="white", ec=GREEN, lw=0.9, ls=(0, (2, 1.2)), zorder=4))
txt(0.29, 0.66, "∅", color=GREEN, bold=True)
txt(0.49, 0.705, "dropped", ha="left", bold=True, color=GREEN)
txt(0.49, 0.575, "no cache at all", ha="left", color=GREY)
pill(1.88, 0.38, "no-cache control", GREEN)
txt(0.17, 0.19, "asks: wrong worse than none?", ha="left", italic=True, color=GREEN)

# ============================================================ RIGHT: topologies
RX, RR = 2.12, 6.25
# ---- (a) chain
txt(RX, 3.0, "(a) Sequential chain", fs=FSB, ha="left", bold=True)
pill(RR, 3.0, "redundant route: receiver reads passages", GREEN)
ay, aw, g = 2.5, 0.85, 0.52
xs = [RX + i * (aw + g) for i in range(3)]
for name, x in zip(("Reasoner", "Verifier", "Finalizer"), xs):
    agent(x, ay, aw, name)
    qp(x + (aw - 0.33) / 2, ay + 0.2)
for i in range(2):
    x_end = xs[i] + aw
    arrow((x_end, ay), (x_end + 0.1, ay))
    cache_block(x_end + 0.1, ay)
    arrow((x_end + 0.36, ay), (xs[i + 1] - 0.0, ay))
arrow((xs[2] + aw, ay), (xs[2] + aw + 0.1, ay))
txt(xs[2] + aw + 0.13, ay, "answer", ha="left")
ax.plot([RX, RR], [2.17, 2.17], color="#DDDDDD", lw=0.8, zorder=1)

# ---- (b) fan-in
txt(RX, 2.07, "(b) Static fan-in", fs=FSB, ha="left", bold=True)
pill(RR, 2.07, "sole route: cache is the only path to passages", ORANGE)
cw = 1.34
agent(RX, 1.68, cw, "Child 1: passages 1–5")
agent(RX, 1.28, cw, "Child 2: passages 6–10")
for yy in (1.68, 1.28):
    arrow((RX + cw, yy), (RX + cw + 0.1, yy))
    cache_block(RX + cw + 0.1, yy)
mx = RX + cw + 0.5
rbox(mx, 1.2, 0.72, 0.56, fc="#F8F8F8", ec=GREY, lw=0.7)
txt(mx + 0.36, 1.48, "RoPE-shift\n+ concat.")
arrow((RX + cw + 0.36, 1.68), (mx, 1.58), rad=-0.1)
arrow((RX + cw + 0.36, 1.28), (mx, 1.38), rad=0.1)
ax_x = mx + 0.72 + 0.08
rbox(ax_x, 1.2, 0.94, 0.56, fc="#F3F3F3")
txt(ax_x + 0.47, 1.58, "Aggregator", bold=True)
txt(ax_x + 0.47, 1.36, "question only", color=GREY)
arrow((mx + 0.72, 1.48), (ax_x, 1.48))
arrow((ax_x + 0.94, 1.48), (ax_x + 1.03, 1.48))
txt(ax_x + 1.05, 1.48, "answer", ha="left")
ax.plot([RX, RR], [1.1, 1.1], color="#DDDDDD", lw=0.8, zorder=1)

# ---- (c) RLM+KV
txt(RX, 1.0, "(c) RLM+KV delegation", fs=FSB, ha="left", bold=True)
pill(RR, 1.0, "redundant route: root keeps passages", GREEN)
qp(RX, 0.58)
rbox(RX + 0.42, 0.44, 1.56, 0.5, fc="#F3F3F3")
txt(RX + 1.2, 0.8, "Root model", bold=True)
txt(RX + 1.2, 0.58, "writes code, delegates", color=GREY)
arrow((RX + 1.98, 0.69), (RX + 2.62, 0.69))
txt(RX + 2.3, 0.82, "delegate", color=GREY)
agent(RX + 2.62, 0.69, 0.82, "Sub-call")
arrow((RX + 3.44, 0.69), (RX + 3.55, 0.69))
cache_block(RX + 3.55, 0.69)
arrow((RX + 3.68, 0.59), (RX + 3.68, 0.4), lw=0.8)
rbox(RX + 0.7, 0.2, 3.4, 0.2, fc="#F8F8F8", ec=GREY, lw=0.7, r=0.04)
txt(RX + 2.4, 0.3, "RoPE-shift + splice into the root's cache")
arrow((RX + 0.7, 0.3), (RX + 0.7, 0.44), lw=0.8)

# ---- legend
qp(RX, 0.02)
txt(RX + 0.42, 0.105, "= question and passages in the agent's own prompt", ha="left", color=GREY)

os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "fig1_overview.pdf"))
fig.savefig(os.path.join(OUT, "fig1_overview.png"), dpi=200)
print("ok")
