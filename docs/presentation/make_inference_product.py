"""
THE INFERENCE PRODUCT  — what the clinic would receive (Phase C vision).

Left: the 5-step pipeline as plain-language cards
      routine H&E  ->  spot patches  ->  foundation model  ->  trained predictor  ->  risk map
Right: the REAL output on a held-out primary tumour — a predicted metastasis-risk /
       leaving-score heatmap produced from H&E ALONE, on a patient the model never
       trained on (leave-one-patient-out = the honest new-patient stand-in).

Honest caveat banner is baked in: today the map reads the abundance-linked signal.

Reads Outputs/stage3_phase_b/phase_b_scores.csv.  tcga env python.
PPTX_SLIDE=1 -> slide layout.
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import _tissue as T

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(ROOT))
FIGDIR = os.path.join(PROJ, "Outputs", "presentation_figures")
SLIDES = os.path.join(FIGDIR, "slides")
PHASEB = os.path.join(PROJ, "Outputs", "stage3_phase_b", "phase_b_scores.csv")
SCORES = os.path.join(PROJ, "Outputs", "stage1a_leaving_program", "leaving_program_scores.csv")
SLIDE = os.environ.get("PPTX_SLIDE", "0") == "1"

BLUE = "#2C5F8A"; ORANGE = "#E07B54"; DARK = "#333333"; LIGHT = "#D9E8F5"; GREY = "#8D8D8D"
SAMPLE = "IU_PDA_T4"

STEPS = [
    ("1", "Routine H&E slide", "The pink-and-purple slide from the pathology lab.\nNo molecular test needed.", BLUE),
    ("2", "Cut into spot patches", "The slide is tiled into small 224px tiles,\none per tissue location.", ORANGE),
    ("3", "Foundation model", "A large pathology-pretrained image network\nturns each tile into a numeric fingerprint.", BLUE),
    ("4", "Trained predictor", "Our model — taught by gene data during training —\nreads each fingerprint. No genes needed now.", ORANGE),
    ("5", "Metastasis-risk map", "A heatmap over the whole tumour flagging the\nregions scored 'primed to leave'.", BLUE),
]


def build():
    df = pd.read_csv(PHASEB)
    d = df[(df["sample"] == SAMPLE) & (df["predictor"] == df["predictor"].iloc[0])].copy()
    sc = pd.read_csv(SCORES)[["patch_stem", "barcode"]]
    d = d.merge(sc, on="patch_stem")
    img, scale = T.render_wsi(SAMPLE, max_dim=1300)
    xy = T.wsi_spot_xy(SAMPLE, d["barcode"].astype(str).tolist())
    d = d[d["barcode"].astype(str).isin(xy)]
    px = d["barcode"].astype(str).map(lambda b: xy[b][0] * scale)
    py = d["barcode"].astype(str).map(lambda b: xy[b][1] * scale)
    risk = d["pred_raw_smooth"].values

    fig = plt.figure(figsize=(17.6, 8.8))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.25], wspace=0.05,
                  left=0.02, right=0.98, top=0.9, bottom=0.06)

    # ---- left: pipeline cards
    axL = fig.add_subplot(gs[0]); axL.axis("off"); axL.set_xlim(0, 1); axL.set_ylim(0, 1)
    n = len(STEPS); h = 0.15; gap = (1 - n * h) / (n + 1)
    for i, (num, title, body, col) in enumerate(STEPS):
        y = 1 - gap - (i + 1) * h - i * gap
        card = FancyBboxPatch((0.08, y), 0.84, h, boxstyle="round,pad=0.008,rounding_size=0.02",
                              fc="white", ec=col, lw=2)
        axL.add_patch(card)
        axL.add_patch(plt.Circle((0.15, y + h / 2), 0.032, color=col))
        axL.text(0.15, y + h / 2, num, ha="center", va="center", color="white",
                 fontsize=14, fontweight="bold")
        axL.text(0.22, y + h * 0.68, title, ha="left", va="center", fontsize=13,
                 fontweight="bold", color=col)
        axL.text(0.22, y + h * 0.28, body, ha="left", va="center", fontsize=9.5, color=DARK)
        if i < n - 1:
            axL.annotate("", xy=(0.5, y - gap * 0.15), xytext=(0.5, y),
                         arrowprops=dict(arrowstyle="-|>", color=GREY, lw=2))
    axL.text(0.5, 0.975, "How the clinic would use it", ha="center", fontsize=15,
             fontweight="bold", color=DARK)

    # ---- right: real risk map on held-out tumour
    axR = fig.add_subplot(gs[1]); axR.imshow(img)
    sctr = axR.scatter(px, py, c=risk, s=9, cmap="RdYlBu_r",
                       vmin=np.nanpercentile(risk, 3), vmax=np.nanpercentile(risk, 97),
                       linewidths=0, alpha=0.9)
    axR.set_title(f"Predicted risk map from H&E alone — {SAMPLE.replace('IU_PDA_','')} "
                  f"(patient held out of training)", fontsize=13, fontweight="bold", color=BLUE)
    axR.set_xticks([]); axR.set_yticks([])
    for sp in axR.spines.values(): sp.set_visible(False)
    cb = plt.colorbar(sctr, ax=axR, fraction=0.045, pad=0.02)
    cb.set_label("lower  ←  predicted leaving risk  →  higher", fontsize=10)
    cb.outline.set_visible(False)

    # honest banner
    fig.text(0.5, 0.02,
             "This is the real output for a patient the model never saw. Honest caveat: today the map reads the "
             "tumour-abundance-linked signal; the finer confound-free state is not yet readable from H&E on new patients.",
             ha="center", fontsize=11, color=DARK, style="italic",
             bbox=dict(boxstyle="round,pad=0.4", fc=LIGHT, ec="none"))

    if not SLIDE:
        fig.suptitle("The inference product — a metastasis-risk map from a routine slide (Phase C vision)",
                     fontsize=16, fontweight="bold", color=DARK, y=0.965)
    out = os.path.join(SLIDES if SLIDE else FIGDIR, "figI_inference_product.png")
    fig.savefig(out, dpi=135, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("saved", out)


if __name__ == "__main__":
    build()
