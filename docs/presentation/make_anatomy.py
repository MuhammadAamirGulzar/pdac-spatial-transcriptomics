"""
ANATOMY OF A SPOT  — the WSI -> spot -> cells -> genes drill-down, made concrete.

Takes two real, tumour-rich spots from the SAME primary tumour (patient 11):
  * an INVASIVE spot   (high confound-free leaving score)
  * a BULKY spot       (equally tumour-rich, but low leaving score)
and walks each one down the full chain a pathologist can verify:
  whole slide -> zoomed tissue -> the exact 224px H&E patch ->
  its cell make-up (RCTD) -> its marker-gene readout (real lab counts).

The two spots have almost the same tumour content, so any difference is the
leaving/invasion biology, not "how much tumour" — the core message of the project.

Run with tcga env python.  PPTX_SLIDE=1 -> slide layout.
"""
import os, numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, FancyArrowPatch
from PIL import Image
import _tissue as T
import _gene_expr as G

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(ROOT))
FIGDIR = os.path.join(PROJ, "Outputs", "presentation_figures")
SLIDES = os.path.join(FIGDIR, "slides")
SCORES = os.path.join(PROJ, "Outputs", "stage1a_leaving_program", "leaving_program_scores.csv")
PATCH = os.path.join(PROJ, "dataset", ".png patches", ".png patches")
RCTD = os.path.join(PROJ, "dataset", "Cell Embedding Extraction", "RCTD")
SLIDE = os.environ.get("PPTX_SLIDE", "0") == "1"

CELLS = ["B cells", "C1Q-TAM", "CD4+ cells", "CD8-NK cells", "DCs", "Endothelial cells",
         "FCN1-TAM", "Hepatocytes", "iCAF", "myCAF", "Normal Epithelial cells",
         "Proliferative T cells", "PVL", "SPP1-TAM", "Tumor Epithelial cells"]
BLUE = "#2C5F8A"; ORANGE = "#E07B54"; DARK = "#333333"; GREEN = "#2E8B57"; RED = "#C0392B"
GREY = "#8D8D8D"

INVASION = ["SERPINE1", "S100A4", "POSTN", "MMP14", "COL1A1", "FN1", "LAMC2"]
EPITH = ["EPCAM", "KRT8", "VIM"]

SAMPLE = "IU_PDA_T11"
HERO = {                                   # patch_stem -> label/colour/role
    "invasive": ("IU_PDA_T11_patch-000845_22_48", GREEN, "INVASIVE spot"),
    "bulky":    ("IU_PDA_T11_patch-001591_22_90", BLUE,  "BULKY spot"),
}


def load_rctd(sample, stem):
    v = torch.load(os.path.join(RCTD, sample, stem + ".pt"), weights_only=True).numpy()
    return pd.Series(v, index=CELLS)


def neighbourhood_mean(sc, expr, bc_by_stem, stem, genes, radius=2):
    """Mean expression over the hero spot + its Visium neighbours (grid distance
    <= radius in row/col), to damp single-spot dropout. Returns Series over genes."""
    r0, c0 = sc.loc[stem, "row"], sc.loc[stem, "col"]
    near = sc[(sc["row"] - r0).abs() <= radius * 2]
    near = near[((near["row"] - r0).abs() + (near["col"] - c0).abs()) <= radius * 2]
    bcs = [b for b in near["barcode"].astype(str) if b in expr.index]
    sub = expr.loc[bcs]
    return sub.reindex(columns=genes).mean()


def build():
    sc = pd.read_csv(SCORES); sc = sc[sc["sample"] == SAMPLE].copy()
    sc["barcode"] = sc["barcode"].astype(str)
    expr = G.extract(SAMPLE, INVASION + EPITH)
    med = expr.median()                    # sample-wide median per gene (reference)
    img, scale = T.render_wsi(SAMPLE, max_dim=1600)

    stems = {k: v[0] for k, v in HERO.items()}
    sc = sc.set_index("patch_stem")
    bc = {k: sc.loc[stems[k], "barcode"] for k in stems}
    xy = T.wsi_spot_xy(SAMPLE, list(bc.values()))
    pos = {k: xy[bc[k]] for k in bc}       # full-res px

    fig = plt.figure(figsize=(17.6, 9.6))
    gs = GridSpec(4, 3, figure=fig, width_ratios=[1.35, 1, 1], height_ratios=[1, 1, 1, 1.15],
                  hspace=0.42, wspace=0.28, left=0.035, right=0.975, top=0.9, bottom=0.055)

    # ---- left column: whole slide with both spots ringed (spans rows 0-2)
    axw = fig.add_subplot(gs[0:3, 0]); axw.imshow(img)
    for k in stems:
        x, y = pos[k][0] * scale, pos[k][1] * scale
        col = HERO[k][1]
        axw.add_patch(Circle((x, y), 34, fill=False, ec=col, lw=3.5))
        axw.annotate(HERO[k][2], (x, y), xytext=(x + 44, y),
                     fontsize=12, fontweight="bold", color=col,
                     va="center", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, alpha=0.9))
    axw.set_title(f"{SAMPLE.replace('IU_PDA_','')} — one primary tumour, two spots",
                  fontsize=13, fontweight="bold", color=BLUE)
    axw.set_xticks([]); axw.set_yticks([])
    for sp in axw.spines.values(): sp.set_visible(False)

    # ---- for each hero spot: zoom | patch | (cells+genes below)
    for ci, k in enumerate(["invasive", "bulky"]):
        stem = stems[k]; col = HERO[k][1]
        r = sc.loc[stem]
        x, y = pos[k]
        # zoom crop ~ 1500 full-res px box
        half = 900
        crop, box = T.crop_wsi(SAMPLE, x - half, y - half, x + half, y + half, out_px=520)
        x0, y0, _, _, s = box
        axz = fig.add_subplot(gs[0, ci + 1]); axz.imshow(crop)
        axz.add_patch(Circle(((x - x0) * s, (y - y0) * s), crop.shape[0] * 0.055,
                             fill=False, ec=col, lw=2.6))
        axz.set_title(f"{HERO[k][2]}  ·  zoom", fontsize=12, fontweight="bold", color=col)
        axz.set_xticks([]); axz.set_yticks([])
        for sp in axz.spines.values(): sp.set_edgecolor(col); sp.set_linewidth(2)

        # the exact 224px patch
        axp = fig.add_subplot(gs[1, ci + 1])
        patch = Image.open(os.path.join(PATCH, SAMPLE, stem + ".png")).convert("RGB")
        axp.imshow(patch)
        axp.set_title("its 224px H&E patch", fontsize=11, color=DARK)
        axp.set_xticks([]); axp.set_yticks([])
        for sp in axp.spines.values(): sp.set_edgecolor(col); sp.set_linewidth(2)
        # score annotation under patch
        axp.set_xlabel(f"tumour {r['tumor_frac']*100:.0f}%  ·  raw {r['leaving_score']:+.1f}"
                       f"  ·  confound-free {r['leaving_score_resid']:+.1f}",
                       fontsize=9.5, color=col, fontweight="bold")

        # cell composition (RCTD) — top cell types
        axc = fig.add_subplot(gs[2, ci + 1])
        comp = load_rctd(SAMPLE, stem).sort_values(ascending=False).head(6)[::-1]
        barcols = [ORANGE if c == "Tumor Epithelial cells" else
                   ("#6B8E23" if "CAF" in c else GREY) for c in comp.index]
        axc.barh(range(len(comp)), comp.values * 100, color=barcols)
        axc.set_yticks(range(len(comp))); axc.set_yticklabels(comp.index, fontsize=9)
        axc.set_xlabel("% of cells in spot", fontsize=9)
        axc.set_title("cell make-up (RCTD)", fontsize=11, color=DARK, fontweight="bold")
        axc.tick_params(axis="x", labelsize=8)
        for sp in ["top", "right"]: axc.spines[sp].set_visible(False)

    # ---- bottom row (spans all 3 cols): marker-gene readout, both spots vs sample median
    axg = fig.add_subplot(gs[3, :])
    genes = INVASION + EPITH
    xidx = np.arange(len(genes)); w = 0.28
    inv_nb = neighbourhood_mean(sc, expr, bc, stems["invasive"], genes)
    blk_nb = neighbourhood_mean(sc, expr, bc, stems["bulky"], genes)
    inv_v = [inv_nb.get(g, 0) for g in genes]
    blk_v = [blk_nb.get(g, 0) for g in genes]
    med_v = [med.get(g, 0) for g in genes]
    axg.bar(xidx - w, inv_v, w, color=GREEN, label="Invasive spot")
    axg.bar(xidx, blk_v, w, color=BLUE, label="Bulky spot")
    axg.bar(xidx + w, med_v, w, color="#CCCCCC", label="Tumour median (reference)")
    for i, g in enumerate(genes):
        axg.axvline(i + 0.5, color="#EEEEEE", lw=0.8, zorder=0)
    axg.axvspan(-0.5, len(INVASION) - 0.5, color=ORANGE, alpha=0.06)
    axg.axvspan(len(INVASION) - 0.5, len(genes) - 0.5, color=BLUE, alpha=0.06)
    axg.text(len(INVASION) / 2 - 0.5, axg.get_ylim()[1] * 0.92, "INVASION / MATRIX program",
             ha="center", fontsize=11, color=ORANGE, fontweight="bold")
    axg.text(len(INVASION) + len(EPITH) / 2 - 0.5, axg.get_ylim()[1] * 0.92, "EPITHELIAL identity",
             ha="center", fontsize=11, color=BLUE, fontweight="bold")
    axg.set_xticks(xidx); axg.set_xticklabels(genes, fontsize=10, fontweight="bold")
    axg.set_ylabel("expression\n(log CP10k)", fontsize=10)
    axg.set_title("Marker-gene readout (spot + immediate neighbours) — the invasive region switches ON "
                  "the invasion/matrix program (real lab counts)",
                  fontsize=12, fontweight="bold", color=DARK)
    axg.legend(fontsize=10, loc="upper right", frameon=False)
    for sp in ["top", "right"]: axg.spines[sp].set_visible(False)

    if not SLIDE:
        fig.suptitle("Anatomy of a spot:  whole slide  →  tissue  →  H&E patch  →  cells  →  genes",
                     fontsize=17, fontweight="bold", color=DARK, y=0.965)
    out = os.path.join(SLIDES if SLIDE else FIGDIR, "figH_anatomy_of_a_spot.png")
    fig.savefig(out, dpi=135, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("saved", out)


if __name__ == "__main__":
    build()
