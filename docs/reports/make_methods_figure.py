"""
Methodology figure for the manuscript. Four panels:

  a  the three datasets and what each one can answer
  b  how a tissue section becomes numbers (the three measurement channels)
  c  the three safeguards, each tied to the mistake it prevents
  d  the analysis logic, and how one result generated the next question

Drawn rather than photographed so a reader can follow the whole design without
opening any code.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEST = os.path.join(ROOT, "docs", "reports", "figures")
os.makedirs(DEST, exist_ok=True)

INK, INK2, MUTED = "#12161d", "#4a5361", "#8b93a1"
SURF, PANEL, RULE = "#ffffff", "#f5f7fa", "#d8dee7"
C_T, C_HM, C_LNM, C_OTHER = "#1baf7a", "#eb6834", "#2a78d6", "#7a5cc4"
C_WARN = "#b3541e"

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})


def box(ax, x, y, w, h, fc=PANEL, ec=RULE, lw=1.1, r=0.012, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))


def txt(ax, x, y, s, size=9, w="normal", c=INK, ha="left", va="center", z=5, style="normal"):
    ax.text(x, y, s, fontsize=size, fontweight=w, color=c, ha=ha, va=va, zorder=z,
            style=style, linespacing=1.45)


def arrow(ax, p1, p2, c=MUTED, lw=1.5, style="-|>", z=3, rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, color=c, lw=lw,
                                 mutation_scale=12, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}"))


def panel_tag(ax, x, y, letter):
    ax.text(x, y, letter, fontsize=13, fontweight="bold", color=INK, ha="left", va="top")


fig = plt.figure(figsize=(13.6, 15.2), facecolor=SURF)
gs = fig.add_gridspec(4, 1, height_ratios=[1.02, 1.24, 1.0, 1.12], hspace=0.14,
                      left=0.035, right=0.975, top=0.975, bottom=0.02)

# ============================================================== a  DATASETS
ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 10); ax.set_ylim(0, 3.1); ax.axis("off")
panel_tag(ax, -0.15, 3.05, "a")
txt(ax, 0.35, 2.92, "Three datasets, each answering a different question", size=11.5, w="bold")

specs = [
    (0.35, C_T, "Discovery cohort", "GSE272362 (Khaliq 2024)",
     ["30 sections, 13 patients", "primary 10 · liver met 12", "lymph node 5 · normal 3",
      "91,496 spots"],
     "Do metastases at two sites\nrun the same program?"),
    (3.55, C_HM, "Replication cohort", "GSE274557 (Maitra 2025)",
     ["55 sections, 13 patients", "all treatment naive", "primary · liver · lung · peritoneum",
      "11 patients gave 2+ met sites"],
     "Does it hold with three\nsites and no chemotherapy?"),
    (6.75, C_LNM, "Generalisation cohort", "HEST-1k (Jaume 2024)",
     ["156 sections, 7 organs", "bowel, prostate, kidney, brain,", "breast, lymph node, pancreas",
      "paired H&E tiles"],
     "Does tissue context govern\nexpression outside pancreas?"),
]
for x, c, title, src, bullets, q in specs:
    box(ax, x, 0.30, 2.85, 2.35, fc=PANEL)
    ax.add_patch(FancyBboxPatch((x, 2.34), 2.85, 0.31,
                                boxstyle="round,pad=0,rounding_size=0.012", fc=c, ec=c, zorder=3))
    txt(ax, x + 0.14, 2.495, title, size=9.6, w="bold", c="white", z=6)
    txt(ax, x + 0.14, 2.16, src, size=8.3, c=INK2, z=6)
    for i, b in enumerate(bullets):
        txt(ax, x + 0.14, 1.90 - i * 0.235, b, size=8.4, c=INK, z=6)
    txt(ax, x + 0.14, 0.60, q, size=8.4, c=c, w="bold", z=6)

# ============================================================== b  MEASUREMENT
ax = fig.add_subplot(gs[1]); ax.set_xlim(0, 10); ax.set_ylim(0, 3.6); ax.axis("off")
panel_tag(ax, -0.15, 3.55, "b")
txt(ax, 0.35, 3.44, "How one tissue section becomes numbers", size=11.5, w="bold")

box(ax, 0.35, 1.35, 1.55, 1.42, fc="#fdf0f4", ec="#e8b9c8")
txt(ax, 1.12, 2.58, "Tissue section", size=8.8, w="bold", ha="center")
for i in range(5):
    for jj in range(4):
        ax.add_patch(Circle((0.62 + jj * 0.33, 1.66 + i * 0.19), 0.072,
                            fc="#e187a8", ec="white", lw=.7, zorder=4))
txt(ax, 1.12, 1.16, "each dot = one spot\n(55 µm, 1–10 cells)", size=7.4, c=INK2, ha="center")

# The three channels are measured in PARALLEL on the same spot, so they are stacked
# with a fan of arrows. Drawing them in a row with one arrow through all three
# would read as a sequential pipeline, which is not what happens.
chans = [
    (2.55, "H&E image", "224 x 224 tile at each spot",
     "UNI2-h foundation model", "1,536 numbers per spot", C_LNM),
    (1.70, "Gene expression", "~18,000 genes at each spot",
     "scVI latent / signature scores", "50 numbers or signature scores", C_T),
    (0.85, "Cell composition", "deconvolution of the same spot",
     "RCTD, the published assay", "fraction of each of 15 cell types", C_HM),
]
for y, name, what, how, out, c in chans:
    arrow(ax, (1.95, 2.06), (2.62, y + 0.38), c=MUTED, rad=0.06)
    box(ax, 2.70, y, 7.05, 0.76, fc=PANEL)
    ax.add_patch(FancyBboxPatch((2.70, y), 0.055, 0.76,
                                boxstyle="square,pad=0", fc=c, ec=c, zorder=4))
    txt(ax, 2.92, y + 0.53, name, size=9.2, w="bold", c=c, z=6)
    txt(ax, 2.92, y + 0.22, what, size=8.1, c=INK2, z=6)
    txt(ax, 5.30, y + 0.55, "processed with", size=7.2, c=MUTED, z=6)
    txt(ax, 5.30, y + 0.28, how, size=8.4, w="bold", c=INK, z=6)
    txt(ax, 7.75, y + 0.55, "gives", size=7.2, c=MUTED, z=6)
    txt(ax, 7.75, y + 0.28, out, size=8.1, c=INK2, z=6)
txt(ax, 0.35, 0.58,
    "All three describe the SAME spot, so any one can be tested against the others.\n"
    "H&E tiles and expression are paired in all three datasets; cell composition\n"
    "is available for the discovery cohort.",
    size=8.0, c=INK2, style="italic", va="top")

# ============================================================== c  SAFEGUARDS
ax = fig.add_subplot(gs[2]); ax.set_xlim(0, 10); ax.set_ylim(0, 3.0); ax.axis("off")
panel_tag(ax, -0.15, 2.95, "c")
txt(ax, 0.35, 2.82, "Three safeguards, each matched to a mistake it prevents",
    size=11.5, w="bold")

guards = [
    (0.35, "If we counted spots",
     "91,496 spots come from only\n17 metastasis sections. Spots in\none section are one tumour\nmeasured many times.",
     "Every test uses one value per\nSECTION. 9 liver vs 5 nodal,\nnever 91,496."),
    (3.55, "If we compared whole sections",
     "A lymph node is an immune organ.\nComparing whole sections would\nrediscover anatomy, not cancer\nbiology.",
     "Keep only tumour-rich spots, then\nrepeat at rising purity. A tissue\nartefact must fade. Ours did not."),
    (6.75, "If we compared different people",
     "12 liver sections against 5 nodal\nalso compares 12 people against 5,\nand in HEST each organ often\ncomes from one laboratory.",
     "Four patients gave both sites, so\npatient cancels. In HEST, swap\nSTUDY with organ held fixed."),
]
for x, wrong, why, fix in guards:
    box(ax, x, 0.18, 2.85, 2.42, fc="#fff8f1", ec="#f0d3b4")
    txt(ax, x + 0.13, 2.40, "RISK", size=7.4, w="bold", c=C_WARN)
    txt(ax, x + 0.13, 2.20, wrong, size=9, w="bold", c=INK)
    txt(ax, x + 0.13, 1.72, why, size=8.1, c=INK2)
    ax.plot([x + 0.13, x + 2.72], [1.30, 1.30], color="#f0d3b4", lw=1)
    txt(ax, x + 0.13, 1.14, "WHAT WE DID", size=7.4, w="bold", c="#1f7a4d")
    txt(ax, x + 0.13, 0.66, fix, size=8.1, c=INK)

# ============================================================== d  ANALYSIS
ax = fig.add_subplot(gs[3]); ax.set_xlim(0, 10); ax.set_ylim(0, 3.4); ax.axis("off")
panel_tag(ax, -0.15, 3.35, "d")
txt(ax, 0.35, 3.22, "What was asked, and how each answer produced the next question",
    size=11.5, w="bold")

steps = [
    (0.35, "1", "Is the metastatic\ndirection real biology,\nor just organ tissue?",
     "Composition separates liver\nfrom primary at 0.82, but\nstill 0.78 with liver and\ntumour channels removed,\nand only 0.59 for lymph node.",
     "Mostly ORGAN identity", C_WARN),
    (2.75, "2", "Do the two metastatic\nsites share one\nprogram?",
     "Alignment of the two shifts\naway from primary:\ncosine +0.55.\nB cells differ, d = +2.03,\nq = 0.042, 4 of 4 patients.",
     "Only about half shared", C_LNM),
    (5.15, "3", "Does that hold with\nmore sites and no\nchemotherapy?",
     "Three site pairs in an\nindependent cohort:\n−0.26, +0.09, −0.37.\nLiver immune cold,\nlung immune hot.",
     "Replicates, and stronger", C_T),
    (7.55, "4", "If tissue context rules\nexpression, can an image\nmodel cross organs?",
     "Same organ, new study:\n0.235 → 0.199.\nNew organ: → 0.125.\nThe loss is organ,\nnot laboratory.",
     "It cannot. Confirmed", C_HM),
]
for x, n, q, res, verdict, c in steps:
    box(ax, x, 0.30, 2.15, 2.62, fc=PANEL)
    ax.add_patch(Circle((x + 0.22, 2.70), 0.115, fc=c, ec=c, zorder=5))
    txt(ax, x + 0.22, 2.70, n, size=8.6, w="bold", c="white", ha="center", z=6)
    txt(ax, x + 0.42, 2.70, "OBJECTIVE", size=7.2, w="bold", c=INK2, z=6)
    txt(ax, x + 0.12, 2.28, q, size=8.5, w="bold", c=INK, z=6)
    txt(ax, x + 0.12, 1.42, res, size=7.9, c=INK2, z=6)
    ax.add_patch(FancyBboxPatch((x + 0.10, 0.42), 1.95, 0.30,
                                boxstyle="round,pad=0,rounding_size=0.01",
                                fc=c, ec=c, alpha=.13, zorder=4))
    txt(ax, x + 0.20, 0.57, verdict, size=8.3, w="bold", c=c, z=6)
    if x < 7:
        arrow(ax, (x + 2.18, 1.60), (x + 2.38, 1.60), c=MUTED, lw=1.6)

fig.savefig(os.path.join(DEST, "fig_methods.png"), dpi=200,
            bbox_inches="tight", facecolor=SURF)
plt.close(fig)
print("wrote", os.path.join(DEST, "fig_methods.png"))
