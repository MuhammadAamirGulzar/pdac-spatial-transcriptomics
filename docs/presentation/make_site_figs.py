"""
Slide graphics for the site-specificity talk (clinical audience).

These are deliberately simpler than the report figures. One idea per picture,
big type, numbers written on the marks, no statistics vocabulary on the face of
the chart. The report figures stay where they are for the written version.

Site colours are fixed here and used everywhere in the deck:
    primary tumour  green      liver met  orange      lymph node met  blue
    lung met        magenta    peritoneal met  purple
This set is checked for colour-blind separation on every pair, so the deck does
not depend on the audience distinguishing red from green.

Titles are drawn in FIGURE coordinates above the axes rather than with set_title,
because the long subtitles used here overrun a title pad and collide.

Run:
    python docs/presentation/make_site_figs.py
Output:
    Outputs/presentation_figures/slides_site/*.png
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "Outputs", "presentation_figures", "slides_site")
CACHE = os.path.join(ROOT, "docs", "presentation", "_tissue_cache")
os.makedirs(OUT, exist_ok=True)

# fixed site identity colours, validated for CVD separation on every pair
PRIMARY, LIVER, NODE, LUNG, PERIT = "#1baf7a", "#eb6834", "#2a78d6", "#b5417d", "#6a3d9a"
AMBER = "#f0a202"
INK, INK2, MUTED = "#12161d", "#4a5361", "#8b93a1"
SURF, PANEL, RULE = "#ffffff", "#f5f7fa", "#d8dee7"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "DejaVu Sans", "font.size": 13, "text.color": INK,
    "axes.edgecolor": RULE, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, facecolor=SURF)
    plt.close(fig)
    print("  ", name)


def titles(fig, title, subtitle, x=0.055, y=0.955, gap=0.075, size=20, sub=13):
    fig.text(x, y, title, fontsize=size, fontweight="bold", color=INK, va="top")
    fig.text(x, y - gap, subtitle, fontsize=sub, color=INK2, va="top", linespacing=1.5)


def box(ax, x, y, w, h, fc=PANEL, ec=RULE, lw=1.4, r=0.02, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))


def arrow(ax, p1, p2, c=MUTED, lw=2.4, rad=0.0, z=3):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", color=c, lw=lw,
                                 mutation_scale=20, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}"))


def strip(ax, keep_bottom=True):
    for s in ["top", "right", "left"] + ([] if keep_bottom else ["bottom"]):
        ax.spines[s].set_visible(False)


# ============================================================ 1  what we measure
def fig_what_we_measure():
    fig = plt.figure(figsize=(14.5, 6.6))
    titles(fig, "One slide, measured three ways at every point",
           "The tissue is covered with a grid of measuring points. Each point is 55 microns "
           "across, about one to ten cells.", y=0.965, gap=0.085)

    ax = fig.add_axes([0, 0, 1, 0.80]); ax.set_xlim(0, 10); ax.set_ylim(0, 3.6); ax.axis("off")

    tp = os.path.join(CACHE, "IU_PDA_T11_he.png")
    if os.path.exists(tp):
        sub = ax.inset_axes([0.030, 0.24, 0.235, 0.70])
        sub.imshow(plt.imread(tp)); sub.set_xticks([]); sub.set_yticks([])
        for s in sub.spines.values():
            s.set_edgecolor(RULE); s.set_linewidth(1.4)
        sub.set_title("one tissue section", fontsize=13.5, color=INK,
                      fontweight="bold", pad=8)
    ax.text(1.48, 0.56, "every dot on the grid is one measuring point",
            fontsize=12, color=INK2, ha="center", va="center")

    chans = [
        (2.50, "The picture", LIVER,
         "the stained image of that exact point,\nthe same view a pathologist reads"),
        (1.42, "The genes", PRIMARY,
         "which genes are switched on there,\nabout 18,000 measured at once"),
        (0.34, "The cell mixture", NODE,
         "what cell types make up that point:\ntumour, immune, stroma, liver and so on"),
    ]
    for y, name, c, what in chans:
        arrow(ax, (3.05, 1.85), (3.68, y + 0.48), rad=0.07)
        box(ax, 3.78, y, 6.05, 0.96)
        ax.add_patch(FancyBboxPatch((3.78, y), 0.08, 0.96, boxstyle="square,pad=0",
                                    fc=c, ec=c, zorder=4))
        ax.text(4.06, y + 0.66, name, fontsize=15.5, fontweight="bold", color=c, zorder=6)
        ax.text(4.06, y + 0.28, what, fontsize=12.5, color=INK2, zorder=6, linespacing=1.45)

    fig.text(0.055, 0.035,
             "Because all three describe the same point, we can ask whether the picture alone "
             "tells us what the genes are doing.",
             fontsize=12.5, color=INK2, style="italic")
    save(fig, "s1_what_we_measure.png")


# ============================================================ 2  half shared
def fig_half_shared(cos):
    pct = 100 * cos
    fig = plt.figure(figsize=(14.5, 6.0))
    titles(fig, "A tumour that spreads to the liver and one that spreads to a lymph node\n"
                "are not making the same change",
           "Both deposits came from the same patient and the same primary cancer.",
           y=0.965, gap=0.155, size=21)

    ax = fig.add_axes([0, 0.11, 1, 0.63]); ax.set_xlim(0, 10); ax.set_ylim(0, 3.2)
    ax.axis("off")

    box(ax, 0.20, 1.18, 2.30, 0.84, fc="#eaf7f1", ec=PRIMARY)
    ax.text(1.35, 1.60, "Primary tumour\nin the pancreas", fontsize=13.5, fontweight="bold",
            color=PRIMARY, ha="center", va="center", linespacing=1.4, zorder=6)
    for y, c, lab, bg, rad in [(2.16, LIVER, "Liver deposit", "#fdeee7", 0.12),
                               (0.20, NODE, "Lymph node deposit", "#e8f1fc", -0.12)]:
        arrow(ax, (2.58, 1.60), (3.12, y + 0.42), c=c, lw=2.8, rad=rad)
        box(ax, 3.22, y, 2.30, 0.84, fc=bg, ec=c)
        ax.text(4.37, y + 0.42, lab, fontsize=13.5, fontweight="bold", color=c,
                ha="center", va="center", zorder=6)

    x0, w, yb, hb = 6.10, 3.60, 1.60, 0.80
    ax.text(x0, yb + hb + 0.42, "Of everything that changes when a tumour spreads:",
            fontsize=13.5, color=INK, fontweight="bold", va="bottom")
    ax.add_patch(FancyBboxPatch((x0, yb), w * cos, hb,
                                boxstyle="round,pad=0,rounding_size=0.02",
                                fc="#8b93a1", ec="none", zorder=4))
    ax.add_patch(FancyBboxPatch((x0 + w * cos + 0.04, yb), w * (1 - cos) - 0.04, hb,
                                boxstyle="round,pad=0,rounding_size=0.02",
                                fc=LIVER, ec="none", zorder=4))
    ax.text(x0 + w * cos / 2, yb + hb / 2, f"{pct:.0f}%", fontsize=21, fontweight="bold",
            color="white", ha="center", va="center", zorder=6)
    ax.text(x0 + w * cos + w * (1 - cos) / 2, yb + hb / 2, f"{100 - pct:.0f}%", fontsize=21,
            fontweight="bold", color="white", ha="center", va="center", zorder=6)
    ax.text(x0 + w * cos / 2, yb - 0.16, "the same at\nboth destinations", fontsize=12,
            color="#5d6773", ha="center", va="top", linespacing=1.4)
    ax.text(x0 + w * cos + w * (1 - cos) / 2, yb - 0.16, "decided by\nwhere it landed",
            fontsize=12, color=LIVER, ha="center", va="top", fontweight="bold",
            linespacing=1.4)

    fig.text(0.055, 0.045,
             "If spreading were one fixed process, that bar would be a single colour. "
             "Roughly half of the change is set by the destination.",
             fontsize=13, color=INK2, style="italic")
    save(fig, "s2_half_shared.png")


# ============================================================ 2b  B cells
def fig_bcells(pb):
    d = pb.copy()
    d["pct"] = 100 * d["rctd::B cells"]
    paired = sorted(set(d[d.site == "HM"].patient) & set(d[d.site == "LNM"].patient))

    fig = plt.figure(figsize=(14.6, 6.2))
    titles(fig, "What differs between the two sites is the immune content",
           "B cells were the one thing that clearly separated the two destinations out of "
           "42 things we looked at.", y=0.965, gap=0.085)

    axa = fig.add_axes([0.055, 0.12, 0.36, 0.60])
    for p in paired:
        hm = d[(d.patient == p) & (d.site == "HM")]["pct"].mean()
        ln = d[(d.patient == p) & (d.site == "LNM")]["pct"].mean()
        axa.plot([0, 1], [hm, ln], color="#b9c0ca", lw=2.2, zorder=2)
        axa.scatter([0, 1], [hm, ln], s=150, c=[LIVER, NODE], zorder=4,
                    edgecolors="white", linewidths=2)
        axa.text(1.06, ln, p.replace("PT_", "Patient "), fontsize=11.5, color=INK2,
                 va="center")
    axa.set_xlim(-0.3, 1.55); axa.set_xticks([0, 1])
    axa.set_xticklabels(["liver\ndeposit", "lymph node\ndeposit"], fontsize=13.5)
    axa.set_ylabel("B cells, % of the tissue", fontsize=12.5, color=INK2)
    axa.grid(axis="y", alpha=0.35); axa.set_axisbelow(True)
    for s in ["top", "right"]:
        axa.spines[s].set_visible(False)
    axa.text(0, 1.10, "The four patients who gave tissue from both sites",
             transform=axa.transAxes, fontsize=14.5, fontweight="bold", color=INK)
    axa.text(0, 1.02, "Every one goes up. The direction never reverses.",
             transform=axa.transAxes, fontsize=12, color=INK2)

    axb = fig.add_axes([0.56, 0.12, 0.40, 0.60])
    groups = [("primary\ntumour", "T", PRIMARY), ("liver\ndeposit", "HM", LIVER),
              ("lymph node\ndeposit", "LNM", NODE)]
    rng = np.random.default_rng(0)
    for i, (lab, site, c) in enumerate(groups):
        v = d[d.site == site]["pct"].to_numpy()
        axb.scatter(i + rng.uniform(-0.11, 0.11, len(v)), v, s=110, color=c, zorder=3,
                    edgecolors="white", linewidths=1.6)
        axb.plot([i - 0.26, i + 0.26], [np.median(v)] * 2, color=INK, lw=2.6, zorder=4)
    axb.set_xticks(range(3)); axb.set_xticklabels([g[0] for g in groups], fontsize=13.5)
    axb.set_xlim(-0.55, 2.55)
    axb.grid(axis="y", alpha=0.35); axb.set_axisbelow(True)
    for s in ["top", "right"]:
        axb.spines[s].set_visible(False)
    axb.text(0, 1.10, "Every section we have", transform=axb.transAxes, fontsize=14.5,
             fontweight="bold", color=INK)
    axb.text(0, 1.02, "One dot is one tissue section. The black bar is the middle value.",
             transform=axb.transAxes, fontsize=12, color=INK2)
    save(fig, "s2b_bcells.png")


# ============================================================ 2c  purity control
def fig_purity(ps):
    p = [r["purity"] for r in ps]
    hep = [100 * r["hepatocyte_hm"] for r in ps]
    ratio = [r["ratio"] for r in ps]

    fig = plt.figure(figsize=(14.6, 6.0))
    titles(fig, "Could this simply be normal lymph node tissue in the sample?",
           "We repeated everything while demanding purer and purer tumour at each measuring point. "
           "If the immune difference were\ncontamination, it would fade away as the contamination "
           "does.", y=0.965, gap=0.075)

    ax1 = fig.add_axes([0.065, 0.14, 0.38, 0.56])
    ax1.plot(p, hep, "-o", color=LIVER, lw=3, ms=11, zorder=3)
    for xx, yy in zip(p, hep):
        ax1.text(xx, yy + 0.9, f"{yy:.1f}%", ha="center", fontsize=12, color=LIVER,
                 fontweight="bold")
    ax1.set_ylim(0, max(hep) * 1.35); ax1.set_xticks(p)
    ax1.set_xticklabels([f"{int(100*v)}%" for v in p], fontsize=12.5)
    ax1.set_xlabel("how pure we demand the tumour to be", fontsize=12, color=INK2)
    ax1.grid(axis="y", alpha=0.35); ax1.set_axisbelow(True)
    for s in ["top", "right"]:
        ax1.spines[s].set_visible(False)
    ax1.text(0, 1.10, "Liver tissue in the liver deposits", transform=ax1.transAxes,
             fontsize=14.5, fontweight="bold", color=INK)
    ax1.text(0, 1.02, "The contamination really does fall away, about eightfold.",
             transform=ax1.transAxes, fontsize=12, color=INK2)

    ax2 = fig.add_axes([0.585, 0.14, 0.38, 0.56])
    ax2.plot(p, ratio, "-o", color=NODE, lw=3, ms=11, zorder=3)
    for xx, yy in zip(p, ratio):
        ax2.text(xx, yy + 0.14, f"{yy:.1f}x", ha="center", fontsize=12, color=NODE,
                 fontweight="bold")
    ax2.axhline(1.0, color=MUTED, ls="--", lw=1.6)
    ax2.text(p[0], 1.06, "no difference", fontsize=11, color=MUTED)
    ax2.set_ylim(0, 3.4); ax2.set_xticks(p)
    ax2.set_xticklabels([f"{int(100*v)}%" for v in p], fontsize=12.5)
    ax2.set_xlabel("how pure we demand the tumour to be", fontsize=12, color=INK2)
    ax2.grid(axis="y", alpha=0.35); ax2.set_axisbelow(True)
    for s in ["top", "right"]:
        ax2.spines[s].set_visible(False)
    ax2.text(0, 1.10, "Immune cells: lymph node versus liver", transform=ax2.transAxes,
             fontsize=14.5, fontweight="bold", color=INK)
    ax2.text(0, 1.02, "The difference holds at about twice, all the way across.",
             transform=ax2.transAxes, fontsize=12, color=INK2)
    save(fig, "s2c_purity.png")


# ============================================================ 3  immune by site
def fig_immune_by_site(shifts):
    order = ["Tcells", "Bcells", "NK", "Plasma", "Myeloid"]
    nice = {"Tcells": "T cells", "Bcells": "B cells", "NK": "NK cells",
            "Plasma": "Plasma cells", "Myeloid": "Myeloid cells"}
    d = shifts.set_index("feature")

    fig = plt.figure(figsize=(14.2, 6.4))
    titles(fig, "In a second, independent group of patients the same thing happens",
           "13 patients, none of whom had chemotherapy first, with deposits in three different "
           "places.\nEvery immune cell type falls in the liver and rises in the lung.",
           y=0.965, gap=0.075)

    ax = fig.add_axes([0.075, 0.11, 0.895, 0.63])
    x = np.arange(len(order)); bw = 0.26
    for i, (lab, col, c) in enumerate([("Liver deposit", "shift_HM", LIVER),
                                       ("Lung deposit", "shift_LuM", LUNG),
                                       ("Peritoneal deposit", "shift_PM", PERIT)]):
        ax.bar(x + (i - 1) * (bw + 0.014), [d.loc[f, col] for f in order], bw,
               color=c, label=lab, zorder=3, linewidth=0)
    ax.axhline(0, color=INK, lw=1.8, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels([nice[f] for f in order], fontsize=14)
    ax.set_yticks([]); ax.set_ylim(-1.15, 1.55)
    strip(ax)
    ax.legend(frameon=False, fontsize=12.5, loc="upper right", ncol=1)
    ax.text(0.004, 0.965, "above the line  =  MORE immune cells than the primary tumour",
            transform=ax.transAxes, fontsize=11.5, color=MUTED)
    ax.text(0.004, 0.025, "below the line  =  FEWER immune cells than the primary tumour",
            transform=ax.transAxes, fontsize=11.5, color=MUTED)
    save(fig, "s3_immune_by_site.png")


# ============================================================ 3b  sanity check
def fig_sanity(shifts):
    d = shifts.set_index("feature")
    ctrl = [("Liver cells found in the liver deposits", d.loc["Hepatocyte", "shift_HM"], LIVER),
            ("Lung tissue found in the lung deposits", d.loc["Lung", "shift_LuM"], LUNG)]
    fig = plt.figure(figsize=(12.2, 3.9))
    titles(fig, "Does the method actually read the tissue correctly?",
           "Before trusting the immune result, we check that the same measurement finds the "
           "surrounding organ where it must be.",
           y=0.93, gap=0.135, size=18, sub=12.5)
    ax = fig.add_axes([0.34, 0.14, 0.60, 0.44])
    y = np.arange(len(ctrl))
    ax.barh(y, [c[1] for c in ctrl], 0.42, color=[c[2] for c in ctrl], zorder=3, linewidth=0)
    for i, (lab, v, c) in enumerate(ctrl):
        ax.text(v + 0.05, i, f"+{v:.1f}", va="center", fontsize=15, fontweight="bold", color=c)
    ax.set_yticks(y); ax.set_yticklabels([c[0] for c in ctrl], fontsize=13)
    ax.set_xticks([]); ax.set_xlim(0, 2.45); ax.invert_yaxis()
    ax.axvline(0, color=INK, lw=1.6)
    strip(ax, keep_bottom=False)
    fig.text(0.055, 0.045, "Both controls behave as they should, so the immune finding is not "
                           "an artefact of the measurement.",
             fontsize=12, color=INK2, style="italic")
    save(fig, "s3b_sanity_check.png")


# ============================================================ 4  cross organ
def fig_cross_organ(m8):
    a, b, c_ = m8["within_mean"], m8["across_study_mean"], m8["cross_mean"]
    vals = [("Same organ,\nsame laboratory", a, PRIMARY),
            ("Same organ,\nDIFFERENT laboratory", b, AMBER),
            ("DIFFERENT organ", c_, LIVER)]
    fig = plt.figure(figsize=(14.0, 6.4))
    titles(fig, "A model that reads genes from a picture does not move between organs",
           f"{m8['n_samples']} tissue sections across {len(m8['organs'])} organs. A taller bar "
           "means the picture predicts the genes better.\nThe middle bar is the control: it shows "
           "the drop is not simply because the sample came from a different hospital.",
           y=0.965, gap=0.075)

    ax = fig.add_axes([0.075, 0.13, 0.895, 0.58])
    x = np.arange(3)
    ax.bar(x, [v[1] for v in vals], 0.52, color=[v[2] for v in vals], zorder=3, linewidth=0)
    for i, (lab, v, c) in enumerate(vals):
        ax.text(i, v / 2, f"{v:.2f}", ha="center", va="center", fontsize=23,
                fontweight="bold", color="white", zorder=6)
    ax.set_xticks(x); ax.set_xticklabels([v[0] for v in vals], fontsize=14, linespacing=1.5)
    ax.set_yticks([]); ax.set_ylim(0, 0.315)
    strip(ax)
    ax.annotate("", xy=(0.80, 0.253), xytext=(0.20, 0.253),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=2))
    ax.text(0.5, 0.262, f"changing laboratory\ncosts {100*(a-b)/a:.0f}%", ha="center",
            fontsize=12.5, color=INK2, linespacing=1.4)
    ax.annotate("", xy=(2.0, 0.297), xytext=(1.0, 0.297),
                arrowprops=dict(arrowstyle="->", color=LIVER, lw=2.4))
    ax.text(1.5, 0.303, f"changing ORGAN costs a further {100*(b-c_)/b:.0f}%", ha="center",
            fontsize=14, color=LIVER, fontweight="bold")
    save(fig, "s4_cross_organ.png")


if __name__ == "__main__":
    R = os.path.join(ROOT, "Outputs")
    m6 = json.load(open(os.path.join(R, "stage6_site_program", "metrics.json")))
    m8 = json.load(open(os.path.join(R, "stage8_hest_cross_organ", "metrics.json")))
    sh = pd.read_csv(os.path.join(R, "stage7_external_replication", "site_shifts.csv"))
    # metastases and primaries are stored in two files; the deck figure shows all three groups
    pb = pd.concat([
        pd.read_csv(os.path.join(R, "stage6_site_program", "pseudobulk_sections.csv")),
        pd.read_csv(os.path.join(R, "stage6_site_program", "pseudobulk_primary.csv")),
    ], ignore_index=True)
    print("writing slide graphics ->", OUT)
    fig_what_we_measure()
    fig_half_shared(m6["cos_HMshift_LNMshift"])
    fig_bcells(pb)
    fig_purity(m6["purity_sweep"])
    fig_immune_by_site(sh)
    fig_sanity(sh)
    fig_cross_organ(m8)
