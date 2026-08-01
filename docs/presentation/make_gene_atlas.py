"""
GENE-EXPRESSION WSI ATLAS  (discovery-framed, for a pathologist audience)

Shows REAL marker-gene expression from the lab count matrices painted spot-by-spot
onto the real whole-slide H&E, so the audience can see the biology partition the
tissue: tumour/epithelial identity vs stroma/ECM vs invasion program vs the leaving
score. Every panel is the SAME tissue in the SAME orientation.

Two figures:
  figG_atlas_{sample}.png    - 8-panel marker atlas on one primary tumour
  figG_liver_genes_HM.png    - hepatocyte genes prove the liver confound (HM11)

Run with the tcga env python.  PPTX_SLIDE=1 -> slide-mode (bigger fonts, no title).
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import _tissue as T
import _gene_expr as G

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(ROOT))
FIGDIR = os.path.join(PROJ, "Outputs", "presentation_figures")
SLIDES = os.path.join(FIGDIR, "slides")
SCORES = os.path.join(PROJ, "Outputs", "stage1a_leaving_program", "leaving_program_scores.csv")
os.makedirs(FIGDIR, exist_ok=True); os.makedirs(SLIDES, exist_ok=True)
SLIDE = os.environ.get("PPTX_SLIDE", "0") == "1"

BLUE = "#2C5F8A"; ORANGE = "#E07B54"; DARK = "#333333"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#888888"})

# genes to paint + plain-language role.  cmap: sequential for expression.
EXPR_CMAP = "magma"
PANELS = [
    ("EPCAM",    "Tumour / epithelial identity"),
    ("KRT8",     "Epithelial cytokeratin (tumour)"),
    ("COL1A1",   "Collagen — desmoplastic stroma"),
    ("SPARC",    "Matrix remodelling stroma"),
    ("SERPINE1", "Invasion / plasminogen program"),
    ("S100A4",   "Metastasis-associated (EMT)"),
    ("POSTN",    "Periostin — invasive front matrix"),
]


def _load(sample):
    sc = pd.read_csv(SCORES)
    sc = sc[sc["sample"] == sample].copy()
    sc["barcode"] = sc["barcode"].astype(str)
    genes = [g for g, _ in PANELS]
    expr = G.extract(sample, genes)
    xy = T.wsi_spot_xy(sample, sc["barcode"].tolist())
    sc = sc[sc["barcode"].isin(xy)].copy()
    sc["px"] = sc["barcode"].map(lambda b: xy[b][0])
    sc["py"] = sc["barcode"].map(lambda b: xy[b][1])
    sc = sc.join(expr, on="barcode")
    img, scale = T.render_wsi(sample, max_dim=1500)
    return sc, img, scale


def _panel(ax, img, scale, sc, values, title, cmap, vmin=None, vmax=None,
           diverging=False, dot=7):
    ax.imshow(img)
    x = sc["px"].values * scale; y = sc["py"].values * scale
    v = np.asarray(values, float)
    if diverging:
        m = np.nanpercentile(np.abs(v), 97) or 1.0
        vmin, vmax = -m, m
    else:
        if vmin is None: vmin = 0.0
        if vmax is None:
            vmax = np.nanpercentile(v, 98)
            if vmax <= 0: vmax = np.nanmax(v) if np.nanmax(v) > 0 else 1.0
    sctr = ax.scatter(x, y, c=v, s=dot, cmap=cmap, vmin=vmin, vmax=vmax,
                      linewidths=0, alpha=0.9)
    ax.set_title(title, fontsize=13 if SLIDE else 11, fontweight="bold", color=DARK, pad=4)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    cb = plt.colorbar(sctr, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=8); cb.outline.set_visible(False)
    cb.set_label("low → high", fontsize=8)
    return sctr


def atlas(sample):
    sc, img, scale = _load(sample)
    tag = sample.replace("IU_PDA_", "")
    fig = plt.figure(figsize=(17.5, 8.6))
    gs = GridSpec(2, 4, figure=fig, hspace=0.16, wspace=0.06,
                  left=0.015, right=0.985, top=0.90 if not SLIDE else 0.94, bottom=0.02)

    # panel 0: plain H&E reference
    ax0 = fig.add_subplot(gs[0, 0]); ax0.imshow(img)
    ax0.set_title(f"{tag} — routine H&E slide", fontsize=13 if SLIDE else 11,
                  fontweight="bold", color=BLUE, pad=4)
    ax0.set_xticks([]); ax0.set_yticks([])
    for sp in ax0.spines.values(): sp.set_visible(False)

    # 7 gene panels
    slots = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
    present = [g for g, _ in PANELS if g in sc.columns]
    for (g, role), (r, c) in zip(PANELS, slots):
        ax = fig.add_subplot(gs[r, c])
        if g in sc.columns:
            _panel(ax, img, scale, sc, sc[g].values, f"{g} — {role}", EXPR_CMAP)
        else:
            ax.imshow(img); ax.set_title(f"{g} (not detected)", fontsize=10, color="#999")
            ax.set_xticks([]); ax.set_yticks([])

    if not SLIDE:
        fig.suptitle(f"Reading the biology directly off the slide — {tag} primary tumour\n"
                     f"every dot is one Visium spot, coloured by that gene's expression (real lab counts)",
                     fontsize=15, fontweight="bold", color=DARK, y=0.985)
    out = os.path.join(SLIDES if SLIDE else FIGDIR, f"figG_atlas_{tag}.png")
    fig.savefig(out, dpi=135, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("saved", out)


def liver_genes(sample="IU_PDA_HM11"):
    """Hepatocyte genes light up ONLY in the liver zone -> the confound made visible."""
    genes = ["ALB", "APOA1", "HP", "TTR"]
    sc = pd.read_csv(SCORES)  # PT-only file; HM not here, so build coords from qc
    # HM has no leaving score; just need barcodes+coords -> use qc metrics barcodes
    q = pd.read_csv(os.path.join(PROJ, "dataset", "ST", "scVI_counts", f"{sample}_qc_metrics.csv"))
    q["barcode"] = q["barcode"].astype(str)
    expr = G.extract(sample, genes)
    xy = T.wsi_spot_xy(sample, q["barcode"].tolist())
    q = q[q["barcode"].isin(xy)].copy()
    q["px"] = q["barcode"].map(lambda b: xy[b][0]); q["py"] = q["barcode"].map(lambda b: xy[b][1])
    q = q.join(expr, on="barcode")
    img, scale = T.render_wsi(sample, max_dim=1500)
    tag = sample.replace("IU_PDA_", "")

    fig = plt.figure(figsize=(17.5, 4.6))
    gs = GridSpec(1, 5, figure=fig, wspace=0.06, left=0.01, right=0.99,
                  top=0.84, bottom=0.03)
    ax0 = fig.add_subplot(gs[0, 0]); ax0.imshow(img)
    ax0.set_title(f"{tag} — liver metastasis H&E", fontsize=13 if SLIDE else 11,
                  fontweight="bold", color=BLUE, pad=4)
    ax0.set_xticks([]); ax0.set_yticks([])
    for sp in ax0.spines.values(): sp.set_visible(False)
    roles = {"ALB": "Albumin", "APOA1": "Apolipoprotein A1", "HP": "Haptoglobin", "TTR": "Transthyretin"}
    for g, c in zip(genes, range(1, 5)):
        ax = fig.add_subplot(gs[0, c])
        _panel(ax, img, scale, q, q[g].values, f"{g} — {roles[g]}", "viridis")
    if not SLIDE:
        fig.suptitle("Liver-cell genes light up exactly where the liver is — the confound, proven in the raw data",
                     fontsize=15, fontweight="bold", color=DARK, y=0.99)
    out = os.path.join(SLIDES if SLIDE else FIGDIR, f"figG_liver_genes_{tag}.png")
    fig.savefig(out, dpi=135, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("saved", out)


if __name__ == "__main__":
    for s in ["IU_PDA_T11", "IU_PDA_T4"]:
        atlas(s)
    liver_genes("IU_PDA_HM11")
