"""
THE TUMOUR ECOSYSTEM  — what cells live in each tissue, and which of them travel
with the leaving program.  Uses the RCTD cell-composition deconvolution (derived
from the same ST counts) to add the "cells" link in WSI -> spot -> cells -> genes.

Three panels:
  A  mean cell make-up per sample (stacked) — the metastases carry hepatocytes,
     the primaries are tumour + CAF + immune.
  B  which cell types track the confound-free leaving score across the 4 primaries
     (Spearman) — cancer-associated fibroblasts / matrix cells ride with it.
  C  the leading partner cell painted on a real primary-tumour slide, next to the
     leaving score — they co-localise (invasion is a tumour+stroma neighbourhood).

Reads _gene_cache/rctd_all.csv (built by _cache_rctd.py).  tcga env python.
PPTX_SLIDE=1 -> slide layout.
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr
import _tissue as T

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(ROOT))
FIGDIR = os.path.join(PROJ, "Outputs", "presentation_figures")
SLIDES = os.path.join(FIGDIR, "slides")
SCORES = os.path.join(PROJ, "Outputs", "stage1a_leaving_program", "leaving_program_scores.csv")
RCTD_CSV = os.path.join(ROOT, "_gene_cache", "rctd_all.csv")
SLIDE = os.environ.get("PPTX_SLIDE", "0") == "1"

CELLS = ["B cells", "C1Q-TAM", "CD4+ cells", "CD8-NK cells", "DCs", "Endothelial cells",
         "FCN1-TAM", "Hepatocytes", "iCAF", "myCAF", "Normal Epithelial cells",
         "Proliferative T cells", "PVL", "SPP1-TAM", "Tumor Epithelial cells"]
# plain-language groups + colours
GROUP = {
    "Tumor Epithelial cells": ("Tumour (malignant)", "#E07B54"),
    "myCAF": ("Fibroblasts (myCAF)", "#6B8E23"), "iCAF": ("Fibroblasts (iCAF)", "#9ACD32"),
    "PVL": ("Perivascular", "#8FBC8F"), "Endothelial cells": ("Blood vessels", "#B0728A"),
    "C1Q-TAM": ("Macrophage (C1Q)", "#4F86C6"), "SPP1-TAM": ("Macrophage (SPP1)", "#6FA8DC"),
    "FCN1-TAM": ("Macrophage (FCN1)", "#9FC5E8"), "DCs": ("Dendritic", "#3D85C6"),
    "CD4+ cells": ("T cell (CD4)", "#C27BA0"), "CD8-NK cells": ("T/NK (CD8)", "#A64D79"),
    "Proliferative T cells": ("T cell (prolif.)", "#D5A6BD"), "B cells": ("B cell", "#674EA7"),
    "Normal Epithelial cells": ("Normal epithelium", "#CCCCCC"), "Hepatocytes": ("Liver (hepatocyte)", "#666666"),
}
BLUE = "#2C5F8A"; ORANGE = "#E07B54"; DARK = "#333333"


def build():
    rctd = pd.read_csv(RCTD_CSV)
    sc = pd.read_csv(SCORES)[["patch_stem", "leaving_score", "leaving_score_resid", "tumor_frac"]]
    order = ["IU_PDA_T1", "IU_PDA_T3", "IU_PDA_T4", "IU_PDA_T11", "IU_PDA_HM11", "IU_PDA_HM13"]

    fig = plt.figure(figsize=(17.6, 9.2))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.15, 1], height_ratios=[1, 1],
                  hspace=0.42, wspace=0.22, left=0.06, right=0.99, top=0.9, bottom=0.14)

    # ---- A: stacked composition per sample
    axA = fig.add_subplot(gs[:, 0])
    comp = rctd.groupby("sample")[CELLS].mean().reindex(order)
    # order cells by overall abundance for a readable stack
    cell_order = comp.mean().sort_values(ascending=False).index.tolist()
    bottom = np.zeros(len(order))
    labels_seen = []
    for c in cell_order:
        vals = comp[c].values * 100
        name, col = GROUP[c]
        axA.barh(range(len(order)), vals, left=bottom, color=col, edgecolor="white", lw=0.5,
                 label=name if name not in labels_seen else None)
        labels_seen.append(name)
        bottom += vals
    axA.set_yticks(range(len(order)))
    axA.set_yticklabels([s.replace("IU_PDA_", "") + ("  (metastasis)" if "HM" in s else "  (primary)")
                         for s in order], fontsize=11)
    axA.invert_yaxis(); axA.set_xlabel("% of cells (average over spots)", fontsize=11, labelpad=8)
    axA.set_xlim(0, 100)
    axA.set_title("A · Who lives in each tissue", fontsize=14, fontweight="bold", color=BLUE, loc="left")
    axA.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)
    for sp in ["top", "right"]: axA.spines[sp].set_visible(False)

    # ---- B: cell-type vs leaving score, RAW vs confound-free (PT pooled)
    axB = fig.add_subplot(gs[0, 1])
    pt = rctd[rctd["sample"].str.contains("_T")].merge(sc, on="patch_stem")
    raw_c, res_c = {}, {}
    for c in CELLS:
        if pt[c].std() > 0:
            raw_c[c] = spearmanr(pt[c], pt["leaving_score"]).correlation
            res_c[c] = spearmanr(pt[c], pt["leaving_score_resid"]).correlation
    raw_s = pd.Series(raw_c).sort_values()
    yy = np.arange(len(raw_s)); hh = 0.4
    cols = ["#C0392B" if v > 0 else "#2C5F8A" for v in raw_s.values]
    axB.barh(yy + hh / 2, raw_s.values, hh, color=cols, label="raw leaving score")
    axB.barh(yy - hh / 2, [res_c[c] for c in raw_s.index], hh, color="#CCCCCC",
             label="confound-free (abundance removed)")
    axB.axvline(0, color="#888", lw=1)
    axB.set_yticks(yy); axB.set_yticklabels([GROUP[c][0] for c in raw_s.index], fontsize=8.5)
    axB.set_xlabel("Spearman correlation with leaving score", fontsize=10)
    axB.set_title("B · Raw score = tumour-cell content; remove it and no cell type is left",
                  fontsize=12.5, fontweight="bold", color=BLUE, loc="left")
    axB.legend(fontsize=8.5, loc="lower right", frameon=False)
    for sp in ["top", "right"]: axB.spines[sp].set_visible(False)

    # ---- C: tumour-cell fraction vs RAW leaving score on a real PT slide (they co-locate)
    samp = "IU_PDA_T4"
    img, scale = T.render_wsi(samp, max_dim=1100)
    d = rctd[rctd["sample"] == samp].merge(
        pd.read_csv(SCORES)[["patch_stem", "barcode", "leaving_score"]], on="patch_stem")
    xy = T.wsi_spot_xy(samp, d["barcode"].astype(str).tolist())
    d = d[d["barcode"].astype(str).isin(xy)]
    px = d["barcode"].astype(str).map(lambda b: xy[b][0] * scale)
    py = d["barcode"].astype(str).map(lambda b: xy[b][1] * scale)

    gsC = gs[1, 1].subgridspec(1, 2, wspace=0.05)
    axC1 = fig.add_subplot(gsC[0]); axC1.imshow(img)
    axC1.scatter(px, py, c=d["Tumor Epithelial cells"].values * 100, s=6, cmap="Oranges",
                 vmax=np.nanpercentile(d["Tumor Epithelial cells"].values * 100, 97), linewidths=0)
    axC1.set_title(f"Tumour-cell % on {samp.replace('IU_PDA_','')}", fontsize=10.5, fontweight="bold")
    axC2 = fig.add_subplot(gsC[1]); axC2.imshow(img)
    m = np.nanpercentile(np.abs(d["leaving_score"]), 97)
    axC2.scatter(px, py, c=d["leaving_score"].values, s=6, cmap="RdBu_r",
                 vmin=-m, vmax=m, linewidths=0)
    axC2.set_title("Raw leaving score", fontsize=10.5, fontweight="bold")
    for ax in (axC1, axC2):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
    fig.text(0.75, 0.075, "C · The predictable part of the score simply IS tumour content — the two maps match",
             ha="center", fontsize=11, color=DARK, style="italic")

    if not SLIDE:
        fig.suptitle("The tumour ecosystem — the cells behind the leaving program",
                     fontsize=17, fontweight="bold", color=DARK, y=0.965)
    out = os.path.join(SLIDES if SLIDE else FIGDIR, "figC_cell_ecosystem.png")
    fig.savefig(out, dpi=135, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("saved", out)
    pd.DataFrame({"raw": raw_c, "resid": res_c}).to_csv(
        os.path.join(ROOT, "_gene_cache", "cell_leaving_corr.csv"))
    print("tumour-cell vs raw:", round(raw_c["Tumor Epithelial cells"], 3),
          "| max |resid|:", round(max(abs(v) for v in res_c.values()), 3))


if __name__ == "__main__":
    build()
