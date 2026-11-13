"""Relay-necessity curve: table + figures from a manifest.   *** UNTESTED ON GPU ***

    python scripts/make_necessity_curve.py --manifest docs/necessity_curve_manifest.example.json \
        --out_dir results/necessity_curve

Writes (to --out_dir):
  necessity_curve_table.csv / .json   one row per level (+ seed-sensitivity rows, Holm p, slopes)
  necessity_curve.png                 MAIN figure, 2x2:
        (a) PRIMARY   gold axis: real-minus-dropped (pp, 95% CI) vs gold passages hidden (0/1/2)
        (b) secondary fractional axis: the same vs % of passages hidden
        (c) overlay of both axes on a common x = realized mean gold passages hidden
        (d) real-minus-dropped vs the dropped arm's own accuracy (receiver-only accuracy)
  necessity_curve_accuracy.png        real and dropped accuracy (Wilson 95% CI) on both axes

Reused runs (level 0 and 100%) are drawn with square markers, new runs with circles. p-values
shown are Holm-adjusted over the 7 distinct level tests; raw p-values are in the table.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Okabe-Ito colour-blind-safe palette
BLUE, ORANGE, VERM, GREY, SKY = "#0072B2", "#E69F00", "#D55E00", "#6b6b6b", "#56B4E9"


def _series(rows, axis):
    pts = sorted([r for r in rows if r["axis"] == axis], key=lambda r: r["level"])
    return pts


def _xval(r, axis):
    return r["level"] if axis == "frac" else r["level"]


def _fmt_p(p):
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def _diff_panel(ax, pts, axis, color, xlabel, title, ymin, ymax):
    xs = [r["level"] for r in pts]
    ys = [r["diff_pp"] for r in pts]
    lo = [r["diff_pp"] - r["diff_ci_low_pp"] for r in pts]
    hi = [r["diff_ci_high_pp"] - r["diff_pp"] for r in pts]
    ax.axhline(0, color=GREY, lw=0.8, ls="--")
    ax.errorbar(xs, ys, yerr=[lo, hi], color=color, lw=1.8, capsize=4, zorder=2)
    for r in pts:
        ax.plot(r["level"], r["diff_pp"], marker="s" if r["reused"] else "o", ms=8, color=color,
                mfc=color, mec="black", mew=0.8, zorder=3)
        # a point whose CI reaches below zero is labelled BELOW its error bar so the text never sits on the line
        below = r["diff_ci_low_pp"] < 0 and r["diff_pp"] <= 0
        ax.annotate(f"{r['diff_pp']:+.0f} pp\nHolm {_fmt_p(r['holm_p'])}",
                    (r["level"], r["diff_ci_low_pp"] if below else r["diff_pp"]), textcoords="offset points",
                    xytext=(10, -26) if below else (10, 8), fontsize=7.5)
    span = (max(xs) - min(xs)) or 1.0
    ax.set_xlim(min(xs) - 0.06 * span, max(xs) + 0.30 * span)   # room for the labels on the right
    ax.set_xlabel(xlabel); ax.set_ylabel("real minus dropped accuracy (pp)")
    ax.set_title(title, fontsize=10, loc="left"); ax.set_ylim(ymin, ymax)
    ax.grid(alpha=0.25)


def make_figures(table: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = table["rows"]
    gold, frac = _series(rows, "gold"), _series(rows, "frac")
    allv = [r["diff_ci_low_pp"] for r in rows] + [r["diff_ci_high_pp"] for r in rows]
    ymin, ymax = min(allv + [0]) - 16, max(allv + [0]) + 8      # extra room below for labels of negative points

    fig, axs = plt.subplots(2, 2, figsize=(12.5, 9), constrained_layout=True)
    a, b, c, d = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]
    sl = table["slopes"]
    gtitle = "(a) PRIMARY: hide gold passages (distractors stay visible)"
    if "gold" in sl:
        s = sl["gold"]; gtitle += f"\nslope {s['slope']*100:+.1f} pp per gold passage [{s['ci_low']*100:+.1f}, {s['ci_high']*100:+.1f}]"
    _diff_panel(a, gold, "gold", BLUE, "gold passages hidden from the receivers (of 2)", gtitle, ymin, ymax)
    a.set_xticks([r["level"] for r in gold])
    ftitle = "(b) secondary: hide a share of all passages"
    if "frac" in sl:
        s = sl["frac"]; ftitle += f"\nslope {s['slope']*100:+.1f} pp per 100% hidden [{s['ci_low']*100:+.1f}, {s['ci_high']*100:+.1f}]"
    _diff_panel(b, frac, "frac", ORANGE, "% of passages hidden from the receivers", ftitle, ymin, ymax)
    b.set_xticks([r["level"] for r in frac])

    # (c) overlay on a common x = realized mean gold passages hidden
    c.axhline(0, color=GREY, lw=0.8, ls="--")
    for pts, color, lab, mfc_none in ((frac, ORANGE, "fractional axis (secondary)", True), (gold, BLUE, "gold axis (PRIMARY)", False)):
        xs = [r["mean_gold_hidden"] for r in pts]; ys = [r["diff_pp"] for r in pts]
        lo = [r["diff_pp"] - r["diff_ci_low_pp"] for r in pts]; hi = [r["diff_ci_high_pp"] - r["diff_pp"] for r in pts]
        c.errorbar(xs, ys, yerr=[lo, hi], color=color, lw=1.8 if not mfc_none else 1.2, capsize=3, label=lab,
                   marker="o", ms=7, mfc="white" if mfc_none else color, mec=color)
    c.set_xlabel("mean gold passages hidden per question (realized where logged, else nominal)")
    c.set_ylabel("real minus dropped accuracy (pp)"); c.set_ylim(ymin, ymax)
    c.set_title("(c) both axes on a common 'information lacked' scale", fontsize=10, loc="left")
    c.legend(fontsize=8, loc="upper left"); c.grid(alpha=0.25)

    # (d) difference vs the dropped arm's own accuracy
    d.axhline(0, color=GREY, lw=0.8, ls="--")
    for pts, color, mk, lab in ((gold, BLUE, "o", "gold axis"), (frac, ORANGE, "^", "fractional axis")):
        xs = [100 * r["acc_dropped"] for r in pts]; ys = [r["diff_pp"] for r in pts]
        d.plot(xs, ys, color=color, lw=1.0, alpha=0.6)
        d.scatter(xs, ys, color=color, marker=mk, s=55, edgecolor="black", linewidth=0.7, label=lab, zorder=3)
        for r, x, y in zip(pts, xs, ys):
            # gold labels up-left, fractional labels down-right, so coinciding points stay readable
            gold_pt = r["axis"] == "gold"
            d.annotate(f"gold {r['level']}" if gold_pt else f"{r['level']}%", (x, y), textcoords="offset points",
                       xytext=(-8, 7) if gold_pt else (8, -13), ha="right" if gold_pt else "left",
                       fontsize=7.5, color=color)
    d.set_xlabel("dropped-arm (receiver-only) accuracy (%)"); d.set_ylabel("real minus dropped accuracy (pp)")
    d.set_title("(d) the benefit grows as the receiver alone does worse", fontsize=10, loc="left")
    d.legend(fontsize=8); d.grid(alpha=0.25)
    x0, x1 = d.get_xlim(); d.set_xlim(x0, x1 + 4)                # keep the right-most label inside the axes
    y0, y1 = d.get_ylim(); d.set_ylim(y0 - 2, y1 + 4)            # and the top label
    ns = sorted({r["n_pairs"] for r in rows})
    fig.suptitle("Relay necessity: Qwen2.5-7B-Instruct, HotpotQA, 3-agent chain "
                 f"(n = {', '.join(str(x) for x in ns)} paired questions per level; squares = reused runs)", fontsize=11)
    fig.savefig(out_dir / "necessity_curve.png", dpi=170)
    plt.close(fig)

    # accuracy figure
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True, sharey=True)
    for ax, pts, name, xl in ((axs[0], gold, "gold axis (PRIMARY)", "gold passages hidden"),
                              (axs[1], frac, "fractional axis (secondary)", "% of passages hidden")):
        xs = [r["level"] for r in pts]
        for key, lo_k, hi_k, color, ls, lab in (("acc_real", "acc_real_ci_low", "acc_real_ci_high", VERM, "-", "real (cache)"),
                                                ("acc_dropped", "acc_dropped_ci_low", "acc_dropped_ci_high", SKY, "--", "dropped (no cache)")):
            ys = [100 * r[key] for r in pts]
            ax.errorbar(xs, ys, yerr=[[100 * (r[key] - r[lo_k]) for r in pts], [100 * (r[hi_k] - r[key]) for r in pts]],
                        color=color, ls=ls, marker="o", capsize=3, label=lab)
        ax.set_title(name, fontsize=10, loc="left"); ax.set_xlabel(xl); ax.set_xticks(xs); ax.grid(alpha=0.25)
    axs[0].set_ylabel("accuracy (%)"); axs[0].legend(fontsize=8)
    fig.savefig(out_dir / "necessity_curve_accuracy.png", dpi=170)
    plt.close(fig)


def write_table(table: dict, out_dir: Path) -> None:
    rows = table["rows"] + [dict(r, axis=r["axis"] + " (sensitivity)") for r in table["sensitivity"]]
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out_dir / "necessity_curve_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
    (out_dir / "necessity_curve_table.json").write_text(json.dumps(table, indent=2, default=str), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", default="results/necessity_curve")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--no_figures", action="store_true", help="write the table only")
    args = ap.parse_args(argv)
    from lakv.necessity_stats import build_table
    mpath = Path(args.manifest)
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    table = build_table(manifest, base=mpath.parent, n_boot=args.n_boot)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    write_table(table, out)
    print(f"{'axis':5s} {'level':>5s} {'n':>4s} {'real':>6s} {'drop':>6s} {'diff pp [95% CI]':>24s} {'p(McNemar)':>11s} {'Holm p':>8s}  gold_hidden")
    for r in sorted(table["rows"], key=lambda r: (r["axis"], r["level"])):
        print(f"{r['axis']:5s} {r['level']:5d} {r['n_pairs']:4d} {100*r['acc_real']:5.1f}% {100*r['acc_dropped']:5.1f}% "
              f"{r['diff_pp']:+7.1f} [{r['diff_ci_low_pp']:+6.1f},{r['diff_ci_high_pp']:+6.1f}] {r['mcnemar_p']:11.3g} "
              f"{r['holm_p']:8.3g}  {r['mean_gold_hidden']:.2f} ({r['gold_hidden_source']})")
    for ax, s in table["slopes"].items():
        print(f"slope[{ax}]: {s['slope']*100:+.2f} pp [{s['ci_low']*100:+.2f}, {s['ci_high']*100:+.2f}]  ({s['unit']}; bootstrap p={s['bootstrap_p_two_sided']:.3f})")
    for r in table["sensitivity"]:
        print(f"sensitivity {r.get('label','')}: diff {r['diff_pp']:+.1f} pp; vs reference seed: "
              f"{r.get('diff_of_diffs_pp', float('nan')):+.1f} pp [{r.get('diff_of_diffs_ci_low_pp', float('nan')):+.1f}, "
              f"{r.get('diff_of_diffs_ci_high_pp', float('nan')):+.1f}]")
    if not args.no_figures:
        make_figures(table, out)
        print(f"wrote {out/'necessity_curve.png'} and {out/'necessity_curve_accuracy.png'}")
    print(f"wrote {out/'necessity_curve_table.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
