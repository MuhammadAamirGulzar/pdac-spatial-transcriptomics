"""
Expanded Stage-0 LIVER-CONFOUND slides (splits old fig2 A/B/C into biology-linked slides).
Uses REAL whole-slide H&E (render_wsi) + a zoom-lens into the liver-cell-rich zone,
real spot patches, and hepatocyte marker genes as the biological proof.

Figures (Outputs/presentation_figures/[slides/]):
  figL1_contamination_where.png   how much liver + WSI zoom-lens on the liver-rich zone
  figL2_liver_proof.png           spots + cell composition + hepatocyte genes (the proof)
  figL3_generalization.png        2 mets don't resemble each other (panel B)
  figL4_axis_is_liver.png         axis = liver content, removable by tumour-only (panel C)

Run: "C:/Users/datai/anaconda3/envs/tcga/python.exe" docs/presentation/make_liver_confound.py
     (PPTX_SLIDE=1 for caption-free slide versions)
"""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, ConnectionPatch
from PIL import Image
import _tissue

ROOT = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(os.path.dirname(ROOT))
SLIDE = os.environ.get("PPTX_SLIDE") == "1"
OUT = os.path.join(PROJ, "Outputs", "presentation_figures", "slides" if SLIDE else "")
os.makedirs(OUT, exist_ok=True)
PATCH_DIR = os.path.join(PROJ, "dataset", ".png patches", ".png patches")
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white", "axes.facecolor": "white"})
PT_C, HM_C = "#4878b0", "#d1604a"
SEQ = "magma"

bio = json.load(open(os.path.join(PROJ, "Outputs/stage0_confound/liver_biology.json")))
hm = pd.read_csv(os.path.join(PROJ, "Outputs/stage0_confound/hm11_spots.csv"))
lead = pd.read_csv(os.path.join(PROJ, "Outputs/stage1a_leaving_program/leaving_program_scores.csv"))
hmfr = pd.read_csv(os.path.join(ROOT, "_cache_hm_fracs.csv"))   # HM11+HM13 hepatocyte/tumour frac

def caption(fig, text, y=0.008):
    if SLIDE: return
    fig.text(0.5, y, text, ha="center", va="bottom", fontsize=9.4, color="#333", wrap=True,
             bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f2", ec="#cccccc"))

def patch_img(sample, stem):
    p = os.path.join(PATCH_DIR, sample, stem + ".png")
    return np.asarray(Image.open(p).convert("RGB")) if os.path.exists(p) else np.full((100, 100, 3), 240, np.uint8)

# ============================================================ FIG L1 - how much & where
def figL1():
    C = bio["counts"]
    fig = plt.figure(figsize=(19, 8.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.85, 1.25, 1.15], wspace=0.14,
                          left=0.05, right=0.98, top=0.86, bottom=0.12)

    # (a) violin PT vs HM with counts
    ax = fig.add_subplot(gs[0, 0])
    pt_h = lead["hepatocyte_frac"].values
    hm_h = hmfr["hepatocyte_frac"].values
    parts = ax.violinplot([pt_h, hm_h], showmeans=True, showextrema=False)
    for b, c in zip(parts["bodies"], [PT_C, HM_C]): b.set_facecolor(c); b.set_alpha(0.6)
    parts["cmeans"].set_color("black")
    ax.set_xticks([1, 2]); ax.set_xticklabels([f"Primary tumour\nn={C['n_PT']:,} spots\n(T1,T3,T4,T11)",
                                               f"Liver metastasis\nn={C['n_HM']:,} spots\n(HM11,HM13)"], fontsize=9)
    ax.set_ylabel("Hepatocyte (liver-cell) fraction per spot")
    ax.text(1, pt_h.mean()+0.04, f"mean {pt_h.mean():.3f}", ha="center", fontsize=9)
    ax.text(2, hm_h.mean()+0.04, f"mean {hm_h.mean():.2f}", ha="center", fontsize=9, fontweight="bold", color=HM_C)
    ax.set_title("(a) Liver-cell content per spot", fontsize=12.5, fontweight="bold")

    # (b) HM11 whole-slide overview coloured by hepatocyte fraction + box
    axo = fig.add_subplot(gs[0, 1])
    wsi, scale = _tissue.render_wsi("IU_PDA_HM11", max_dim=2000)
    axo.imshow(wsi)
    d = hm.dropna(subset=["imagecol", "imagerow"])
    sc = axo.scatter(d["imagecol"]*scale, d["imagerow"]*scale, c=d["hepatocyte_frac"],
                     cmap=SEQ, vmin=0, vmax=1, s=6, marker="h", linewidths=0)
    box = bio["hm11_zoom_box"]
    bx = [box["x0"]*scale, box["y0"]*scale, (box["x1"]-box["x0"])*scale, (box["y1"]-box["y0"])*scale]
    axo.add_patch(Rectangle((bx[0], bx[1]), bx[2], bx[3], fill=False, ec="#00e5ff", lw=3))
    axo.set_xticks([]); axo.set_yticks([]); axo.set_aspect("equal")
    fig.colorbar(sc, ax=axo, fraction=0.046, pad=0.02).set_label("hepatocyte fraction", fontsize=9)
    axo.set_title("(b) HM11 metastasis — real H&E, spots = liver-cell content", fontsize=12, fontweight="bold")

    # (c) ZOOM LENS into the liver-rich box (sharp full-res crop) + high-hepa spots
    axz = fig.add_subplot(gs[0, 2])
    crop, (cx0, cy0, cx1, cy1, s) = _tissue.crop_wsi("IU_PDA_HM11", box["x0"], box["y0"], box["x1"], box["y1"], out_px=1100)
    axz.imshow(crop)
    inb = d[(d["imagecol"].between(cx0, cx1)) & (d["imagerow"].between(cy0, cy1))]
    scz = axz.scatter((inb["imagecol"]-cx0)*s, (inb["imagerow"]-cy0)*s, c=inb["hepatocyte_frac"],
                      cmap=SEQ, vmin=0, vmax=1, s=90, marker="h", linewidths=0.4, edgecolors="white")
    axz.set_xticks([]); axz.set_yticks([]); axz.set_aspect("equal")
    axz.set_title(f"(c) ZOOM: liver-cell-rich zone\n{bio['hm11_zoom_n_high_hepa']} spots with hepatocyte fraction > 0.4",
                  fontsize=12, fontweight="bold")
    for sp in axz.spines.values(): sp.set_edgecolor("#00e5ff"); sp.set_linewidth(3)
    # lens connector lines from box (b) to zoom (c)
    for (xb, yb), (xz, yz) in [((bx[0]+bx[2], bx[1]), (0, 0)), ((bx[0]+bx[2], bx[1]+bx[3]), (0, crop.shape[0]))]:
        con = ConnectionPatch(xyA=(xz, yz), coordsA=axz.transData, xyB=(xb, yb), coordsB=axo.transData,
                              color="#00e5ff", lw=1.5, ls="--")
        fig.add_artist(con)

    fig.suptitle("Liver-cell contamination of the metastasis tissue — how much, and exactly where",
                 fontsize=17, fontweight="bold", y=0.955)
    caption(fig, f"(a) Across ALL spots, liver metastasis spots average {hm_h.mean():.2f} hepatocyte (liver-cell) content vs {pt_h.mean():.3f} in primary tumour — "
                 f"a tissue-of-origin confound (PT n={C['n_PT']:,} from 4 slides, HM n={C['n_HM']:,} from 2 slides). "
                 "(b) On the real HM11 whole-slide, spots are coloured by liver-cell content; the cyan box marks a liver-rich pocket. "
                 "(c) Zoomed in, those spots sit on tissue that looks like liver parenchyma — the metastatic tumour is physically embedded in normal liver.", y=0.004)
    plt.savefig(os.path.join(OUT, "figL1_contamination_where.png"), dpi=125, bbox_inches="tight")
    plt.close(); print("FIG L1 done")

# ============================================================ FIG L2 - the biological proof
def figL2():
    GENES = ["ALB", "TTR", "APOA1", "HP", "FGB", "SERPINA1"]
    ex = bio["examples"]["HM_high"] + bio["examples"]["PT_low"]
    tags = ["HM · liver-rich", "HM · liver-rich", "PT · tumour", "PT · tumour"]
    cols = [HM_C, HM_C, PT_C, PT_C]
    fig = plt.figure(figsize=(19, 9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0], hspace=0.42, wspace=0.30,
                          left=0.05, right=0.97, top=0.88, bottom=0.10)

    # top row: 4 example spot patches with composition
    for i, (e, tag, c) in enumerate(zip(ex, tags, cols)):
        axp = fig.add_subplot(gs[0, i])
        axp.imshow(patch_img(e["sample"], e["patch_stem"]))
        axp.set_xticks([]); axp.set_yticks([])
        for sp in axp.spines.values(): sp.set_edgecolor(c); sp.set_linewidth(4)
        axp.set_title(f"{tag}", fontsize=11.5, fontweight="bold", color=c)
        axp.set_xlabel(f"liver-cell {int(e['hepatocyte_frac']*100)}%   ·   tumour {int(e['tumor_frac']*100)}%",
                       fontsize=9.5, color=c, fontweight="bold")

    # bottom-left: hepatocyte marker gene bars for the 4 spots
    axg = fig.add_subplot(gs[1, 0:2])
    x = np.arange(len(GENES)); w = 0.2
    for j, (e, c) in enumerate(zip(ex, cols)):
        vals = [e.get(f"g_{g}", 0.0) for g in GENES]
        axg.bar(x + (j-1.5)*w, vals, w, color=c, alpha=0.55 + 0.15*(j % 2))
    axg.set_xticks(x); axg.set_xticklabels(GENES, fontsize=10)
    axg.set_ylabel("hepatocyte-gene expression\n(CP10k log)")
    axg.set_title("(a) The 'liver' spots strongly express hepatocyte genes; tumour spots do not",
                  fontsize=12, fontweight="bold")
    axg.plot([], [], color=HM_C, lw=6, label="HM liver-rich spots"); axg.plot([], [], color=PT_C, lw=6, label="PT tumour spots (≈ 0)")
    axg.legend(fontsize=9, loc="upper right")

    # bottom-right: RCTD hepatocyte frac vs liver-gene score (HM11), r
    axs = fig.add_subplot(gs[1, 2:4])
    v = hm.dropna(subset=["hepa_gene_score"])
    axs.scatter(v["hepatocyte_frac"], v["hepa_gene_score"], s=6, alpha=0.25, color=HM_C, linewidths=0)
    m, b = np.polyfit(v["hepatocyte_frac"], v["hepa_gene_score"], 1)
    xs = np.array([0, 1]); axs.plot(xs, m*xs+b, color="black", lw=2)
    axs.set_xlabel("RCTD hepatocyte fraction (cell deconvolution)")
    axs.set_ylabel("hepatocyte marker-gene score")
    axs.set_title(f"(b) The cell call agrees with the genes\nHM11: r = {bio['corr_rctd_vs_livergenes_HM11']:.2f}  (n={len(v):,} spots)",
                  fontsize=12, fontweight="bold")

    fig.suptitle("Proof it is really liver: the same spots carry liver CELLS and liver GENES",
                 fontsize=17, fontweight="bold", y=0.965)
    caption(fig, "Two liver-rich metastasis spots (red) and two primary-tumour spots (blue), each with its real H&E patch and cell composition. "
                 "(a) The metastasis spots strongly express canonical hepatocyte genes — ALB (albumin), TTR, APOA1, HP, FGB, SERPINA1 — which the tumour spots essentially lack. "
                 f"(b) Across all HM11 spots the cell-deconvolution hepatocyte fraction and the liver-gene score agree almost perfectly (r={bio['corr_rctd_vs_livergenes_HM11']:.2f}): the 'liver cells' call is real biology, not a deconvolution artefact.", y=0.004)
    plt.savefig(os.path.join(OUT, "figL2_liver_proof.png"), dpi=125, bbox_inches="tight")
    plt.close(); print("FIG L2 done")

# ============================================================ FIG L3 - generalization (panel B)
def figL3():
    fig, ax = plt.subplots(figsize=(13, 7.4)); fig.subplots_adjust(top=0.80, bottom=0.18, left=0.1, right=0.95)
    gen = [("HM11", 0.0, HM_C), ("HM13", 0.0, HM_C), ("T1", 0.961, PT_C),
           ("T3", 0.93, PT_C), ("T4", 1.0, PT_C), ("T11", 0.984, PT_C)]
    ax.bar(range(len(gen)), [g[1] for g in gen], color=[g[2] for g in gen])
    ax.axhline(0.5, ls="--", color="gray", lw=1); ax.text(5.0, 0.52, "chance", fontsize=9, color="gray")
    ax.set_xticks(range(len(gen))); ax.set_xticklabels([g[0] for g in gen], fontsize=11)
    ax.set_ylabel("Recovered when held out (accuracy)"); ax.set_ylim(0, 1.12)
    for i, g in enumerate(gen):
        ax.text(i, g[1]+0.02, f"{g[1]:.2f}", ha="center", fontsize=10, fontweight="bold", color=g[2])
    ax.annotate("the two metastases are NOT\nrecognised from each other (0.00)", xy=(0.5, 0.0), xytext=(1.3, 0.42),
                ha="center", fontsize=11, color=HM_C, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=HM_C, lw=1.6))
    fig.suptitle("With only 2 metastasis patients, one cannot predict the other",
                 fontsize=16, fontweight="bold", y=0.94)
    caption(fig, "Leave-one-sample-out test: hold out each sample and ask if the model recognises its tumour type from the rest. "
                 "Primary tumours generalise (93–100%), but each liver metastasis is NOT recovered from the other (0%) — there is no transferable, "
                 "tumour-intrinsic metastasis signature to learn from just two mets. This is why we target an intra-primary program instead.", y=0.02)
    plt.savefig(os.path.join(OUT, "figL3_generalization.png"), dpi=125, bbox_inches="tight")
    plt.close(); print("FIG L3 done")

# ============================================================ FIG L4 - axis is liver (panel C)
def figL4():
    C = bio["counts"]
    fig, ax = plt.subplots(figsize=(13, 7.4)); fig.subplots_adjust(top=0.80, bottom=0.2, left=0.12, right=0.95)
    vals = [bio["corr_axis_hepatocyte_all"], bio["corr_axis_hepatocyte_tumoronly"]]
    ax.bar([0, 1], vals, color=["#9e9e9e", "#2e7d32"], width=0.55)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"ALL spots\nn={C['n_total']:,}\n(all 6 samples)",
                        f"TUMOUR-ONLY spots\nn={C['n_tumor_only']:,}\n(Tumor fraction ≥ 0.5)"], fontsize=11)
    ax.set_ylabel("corr( metastasis axis , liver-cell content )")
    for i, v in enumerate(vals): ax.text(i, v+0.012, f"{v:.2f}", ha="center", fontweight="bold", fontsize=13)
    ax.annotate("confound removed", xy=(1, vals[1]), xytext=(0.55, 0.30), fontsize=11, color="#2e7d32",
                fontweight="bold", arrowprops=dict(arrowstyle="-|>", color="#2e7d32", lw=1.6))
    fig.suptitle("The 'metastasis' axis is mostly liver content — and it is removable",
                 fontsize=16, fontweight="bold", y=0.94)
    caption(fig, f"The naive HM-minus-PT 'metastasis direction' correlates 0.40 with liver-cell content across all {C['n_total']:,} spots (6 samples). "
                 f"Restricting to the {C['n_tumor_only']:,} tumour-dominated spots (RCTD Tumor ≥ 0.5) — i.e. dropping the liver-heavy non-tumour spots — collapses that correlation to 0.07. "
                 "So most of the apparent 'metastasis' signal was the liver the tumour sits in. CONCLUSION: define the target INSIDE the primary tumour (the leaving program) to avoid the confound entirely.", y=0.02)
    plt.savefig(os.path.join(OUT, "figL4_axis_is_liver.png"), dpi=125, bbox_inches="tight")
    plt.close(); print("FIG L4 done")

if __name__ == "__main__":
    figL1(); figL2(); figL3(); figL4()
    print("DONE ->", OUT)
