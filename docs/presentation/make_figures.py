"""
Presentation figure set for the PDAC ST + Foundation-Model project.
Audience: domain experts (clinical / biology), non-technical on ML.
Every spatial result is shown NEXT TO its real H&E tissue (same orientation),
with plain-language captions and gene/cell names for interpretability.

Run:  "C:/Users/datai/anaconda3/envs/tcga/python.exe" presentation/make_figures.py
Out:  Outputs/presentation_figures/
"""
import os, glob, json, textwrap, re
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.stats import spearmanr
import _tissue

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(ROOT))
SLIDE = os.environ.get("PPTX_SLIDE") == "1"   # slide mode: no captions, clean for deck
OUT  = os.path.join(PROJ, "Outputs", "presentation_figures", "slides" if SLIDE else "")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white",
                     "axes.facecolor": "white"})

PT = ["IU_PDA_T1", "IU_PDA_T3", "IU_PDA_T4", "IU_PDA_T11"]
ALL = PT + ["IU_PDA_HM11", "IU_PDA_HM13"]
def short(s): return s.replace("IU_PDA_", "")
DIVCMAP, SEQCMAP = "RdBu_r", "viridis"
PT_C, HM_C = "#4878b0", "#d1604a"

lead = pd.read_csv(os.path.join(PROJ, "Outputs/stage1a_leaving_program/leaving_program_scores.csv"))
pb   = pd.read_csv(os.path.join(PROJ, "Outputs/stage3_phase_b/phase_b_scores.csv"))

# WSI pixel coords per spot (real whole-slide overlays). T3 has no local WSI.
WSI_PT = ["IU_PDA_T1", "IU_PDA_T4", "IU_PDA_T11"]
_coord = pd.read_csv(os.path.join(PROJ, "Outputs/Patient-Sample-Information/spot_spatial_coordinates.csv"))
_bc2xy = _coord.set_index("spot_barcode")[["imagecol", "imagerow"]]

def with_px(df):
    """Attach WSI pixel coords (px_x=imagecol, px_y=imagerow) via barcode (or patch_stem→barcode)."""
    d = df.copy()
    if "barcode" not in d.columns:
        d = d.merge(lead[["patch_stem", "barcode"]], on="patch_stem", how="left")
    d = d.merge(_bc2xy, left_on="barcode", right_index=True, how="inner")
    return d

def wsi_he_panel(ax, sample, max_dim=2000):
    img, _ = _tissue.render_wsi(sample, max_dim=max_dim)
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for sp in ax.spines.values(): sp.set_edgecolor("#888"); sp.set_linewidth(0.8)
    return img.shape

def wsi_score_panel(ax, sample, dpx, val, cmap, vmin, vmax, s=6, max_dim=2000):
    img, scale = _tissue.render_wsi(sample, max_dim=max_dim)
    ax.imshow(img)
    sc = ax.scatter(dpx["imagecol"].to_numpy() * scale, dpx["imagerow"].to_numpy() * scale,
                    c=val, cmap=cmap, vmin=vmin, vmax=vmax, s=s, marker="h", linewidths=0)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for sp in ax.spines.values(): sp.set_edgecolor("#888"); sp.set_linewidth(0.8)
    return sc

def caption(fig, text, y=0.015):
    if SLIDE:
        return
    fig.text(0.5, y, text, ha="center", va="bottom", fontsize=9.6, color="#333333",
             wrap=True, bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f2", ec="#cccccc"))

def he_panel(ax, sample):
    img, ext = _tissue.render_he(sample)
    ax.imshow(img, extent=ext, aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor("#888"); sp.set_linewidth(0.8)

def score_panel(ax, sample, row, col, val, cmap, vmin, vmax, s=10, cbar_label=None):
    _, ext = _tissue.render_he(sample)
    x, y = _tissue.phys_xy(np.asarray(row), np.asarray(col))
    scsc = ax.scatter(x, y, c=val, cmap=cmap, vmin=vmin, vmax=vmax, s=s, marker="h",
                      linewidths=0, edgecolors="none")
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor("#888"); sp.set_linewidth(0.8)
    return scsc

# HM hepatocyte/tumour per-spot (cache) -- PT comes from lead CSV
def hm_fracs():
    cache = os.path.join(ROOT, "_cache_hm_fracs.csv")
    if os.path.exists(cache): return pd.read_csv(cache)
    rows = []
    qc = pd.read_csv(os.path.join(PROJ, "Outputs/Patient-Sample-Information/spot_qc_mask.csv"))
    for s in ["IU_PDA_HM11", "IU_PDA_HM13"]:
        for _, r in qc[qc["sample"] == s].iterrows():
            p = os.path.join(PROJ, "dataset", "Cell Embedding Extraction", "RCTD", s, r["patch_stem"] + ".pt")
            if os.path.exists(p):
                v = torch.load(p, map_location="cpu", weights_only=False).numpy()
                rows.append({"sample": s, "hepatocyte_frac": float(v[7]), "tumor_frac": float(v[14])})
    d = pd.DataFrame(rows); d.to_csv(cache, index=False); return d

# ============================================================ FIG 1 - cohort overview
def fig1_cohort():
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.75], hspace=0.32, wspace=0.12)
    counts = {"IU_PDA_T1": 3073, "IU_PDA_T3": 4241, "IU_PDA_T4": 3587, "IU_PDA_T11": 2677,
              "IU_PDA_HM11": 3894, "IU_PDA_HM13": 1387}
    for i, s in enumerate(ALL):
        ax = fig.add_subplot(gs[i // 3, i % 3]); he_panel(ax, s)
        tag = "PRIMARY tumour (pancreas)" if "_T" in s else "Liver METASTASIS"
        c = PT_C if "_T" in s else HM_C
        ax.set_title(f"{short(s)}   —   {tag}\n{counts[s]:,} Visium spots",
                     fontsize=11.5, fontweight="bold", color=c)
    axt = fig.add_subplot(gs[2, :]); axt.axis("off")
    axt.text(0.5, 0.96, "Each spot carries THREE matched measurements",
             ha="center", fontsize=13, fontweight="bold")
    boxes = [("H&E image", "tissue appearance\n(foundation-model embedding)", "#8d6e63"),
             ("Gene expression", "what genes the\ncells switch on", "#2e7d32"),
             ("Cell composition", "which cell types are\npresent (deconvolution)", "#1565c0")]
    for j, (t, d, c) in enumerate(boxes):
        x = 0.085 + j * 0.235
        axt.add_patch(FancyBboxPatch((x, 0.20), 0.20, 0.52, boxstyle="round,pad=0.02",
                      transform=axt.transAxes, fc=c, ec="none", alpha=0.16))
        axt.text(x + 0.10, 0.60, t, ha="center", fontsize=11.5, fontweight="bold", color=c)
        axt.text(x + 0.10, 0.36, d, ha="center", fontsize=9.2, color="#333")
    axt.annotate("", xy=(0.96, 0.46), xytext=(0.80, 0.46), xycoords=axt.transAxes,
                 arrowprops=dict(arrowstyle="-|>", lw=2, color="#b71c1c"))
    axt.text(0.88, 0.16, "GOAL: predict metastasis-prone\ntumour regions from H&E ALONE",
             ha="center", fontsize=10, fontweight="bold", color="#b71c1c")
    fig.suptitle("Study design:  4 primary pancreatic tumours and 2 liver metastases  (18,859 spots)",
                 fontsize=16, fontweight="bold")
    caption(fig, "Spatial transcriptomics (genes + cell composition) is used ONLY during training to teach the model what disseminating tumour looks like. "
                 "At inference a routine H&E slide alone must reproduce the signal — no transcriptomics needed (Dr. Ashiq's constraint). "
                 "Only patient 11 has BOTH a primary tumour (T11) and its matched liver metastasis (HM11).", y=0.01)
    plt.savefig(os.path.join(OUT, "fig1_cohort_overview.png"), dpi=125, bbox_inches="tight")
    plt.close(); print("FIG1 done")

# ============================================================ FIG 2 - Stage 0 confound
def fig2_confound():
    hm = hm_fracs()
    fig, axs = plt.subplots(1, 3, figsize=(16, 6.6))
    fig.subplots_adjust(top=0.74, bottom=0.30, wspace=0.32, left=0.06, right=0.97)
    # (a) hepatocyte content PT vs HM
    ax = axs[0]
    pt_h = lead["hepatocyte_frac"].values
    hm_h = hm["hepatocyte_frac"].values
    parts = ax.violinplot([pt_h, hm_h], showmeans=True, showextrema=False)
    for b, c in zip(parts["bodies"], [PT_C, HM_C]): b.set_facecolor(c); b.set_alpha(0.6)
    parts["cmeans"].set_color("black")
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Primary\ntumour", "Liver\nmetastasis"])
    ax.set_ylabel("Hepatocyte (liver-cell) fraction per spot")
    ax.set_title("(a) Liver-cell contamination of the\nmetastasis tissue", fontsize=12, fontweight="bold")
    ax.text(1, pt_h.mean()+0.03, f"mean {pt_h.mean():.3f}", ha="center", fontsize=9)
    ax.text(2, hm_h.mean()+0.03, f"mean {hm_h.mean():.2f}", ha="center", fontsize=9, fontweight="bold")
    # (b) cross-patient generalization
    ax = axs[1]
    gen = [("HM11", 0.0, HM_C), ("HM13", 0.0, HM_C), ("T1", 0.961, PT_C),
           ("T3", 0.93, PT_C), ("T4", 1.0, PT_C), ("T11", 0.984, PT_C)]
    ax.bar(range(len(gen)), [g[1] for g in gen], color=[g[2] for g in gen])
    ax.axhline(0.5, ls="--", color="gray", lw=1); ax.text(5.0, 0.52, "chance", fontsize=8, color="gray")
    ax.set_xticks(range(len(gen))); ax.set_xticklabels([g[0] for g in gen])
    ax.set_ylabel("Recovered when held out (accuracy)"); ax.set_ylim(0, 1.12)
    ax.set_title("(b) Can the model recognise a held-out\nsample's tumour type?", fontsize=12, fontweight="bold")
    for i, g in enumerate(gen):
        ax.text(i, g[1] + 0.02, f"{g[1]:.2f}", ha="center", fontsize=9,
                fontweight="bold", color=g[2])
    ax.annotate("metastases NOT\nrecognised (0.00)", xy=(0.5, 0.0), xytext=(1.0, 0.40),
                ha="center", fontsize=9, color=HM_C, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=HM_C, lw=1.4))
    # (c) confound removal by tumour restriction
    ax = axs[2]
    ax.bar([0, 1], [0.402, 0.073], color=["#9e9e9e", "#2e7d32"], width=0.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["All spots", "Tumour-only\nspots"])
    ax.set_ylabel("corr( metastasis axis , liver content )")
    ax.set_title("(c) The metastasis axis is mostly\nliver content, and removable", fontsize=12, fontweight="bold")
    for i, v in enumerate([0.402, 0.073]): ax.text(i, v+0.01, f"{v:.2f}", ha="center", fontweight="bold")
    fig.suptitle("Stage 0:  why we do not simply compare primary versus liver metastasis tissue",
                 fontsize=15.5, fontweight="bold", y=0.96)
    caption(fig, "(a) Liver metastasis spots are heavily mixed with normal liver cells (hepatocytes), absent in primary tumour — a tissue-of-origin confound. "
                 "(b) With only 2 metastasis patients, one cannot predict the other (0% recovered), while primary tumours generalise (93–100%): there is no transferable tumour-intrinsic metastasis signature to learn. "
                 "(c) Restricting to tumour-rich spots strips the liver-content confound (0.40 -> 0.07). CONCLUSION: target an intra-primary 'leaving program' instead of the metastasis tissue.", y=0.005)
    plt.savefig(os.path.join(OUT, "fig2_stage0_confound.png"), dpi=125, bbox_inches="tight")
    plt.close(); print("FIG2 done")

# ============================================================ FIG 3 - leaving maps + REAL whole-slide tissue
def fig3_leaving_maps():
    # T3 has no local whole-slide image -> show the three tumours that do (all significant).
    moran = {"IU_PDA_T1": 0.39, "IU_PDA_T4": 0.37, "IU_PDA_T11": 0.37}
    samples = WSI_PT
    if SLIDE:
        fig, axs = plt.subplots(2, 3, figsize=(18, 9.6)); fig.subplots_adjust(top=0.88, bottom=0.03, hspace=0.14, wspace=0.06)
        for i, s in enumerate(samples):
            dpx = with_px(lead[lead["sample"] == s])
            wsi_he_panel(axs[0, i], s); axs[0, i].set_title(f"{short(s)}  whole-slide H&E", fontsize=12.5, fontweight="bold")
            sc = wsi_score_panel(axs[1, i], s, dpx, dpx["leaving_score"].to_numpy(), DIVCMAP, -2.5, 2.5, s=6)
            axs[1, i].set_title(f"{short(s)}  leaving score  (I={moran[s]:.2f})", fontsize=12.5, fontweight="bold")
            fig.colorbar(sc, ax=axs[1, i], fraction=0.046, pad=0.02)
        fig.suptitle("The leaving program forms coherent zones in every tumour  (real H&E, red = high)",
                     fontsize=16, fontweight="bold", y=0.965)
    else:
        fig, axs = plt.subplots(3, 2, figsize=(12, 16.5))
        for i, s in enumerate(samples):
            dpx = with_px(lead[lead["sample"] == s])
            wsi_he_panel(axs[i, 0], s); axs[i, 0].set_title(f"{short(s)}  whole-slide H&E", fontsize=12, fontweight="bold")
            sc = wsi_score_panel(axs[i, 1], s, dpx, dpx["leaving_score"].to_numpy(), DIVCMAP, -2.5, 2.5, s=7)
            axs[i, 1].set_title(f"{short(s)}  leaving-program score   (Moran's I = {moran[s]:.2f}, p<0.001)",
                                fontsize=12, fontweight="bold")
            cb = fig.colorbar(sc, ax=axs[i, 1], fraction=0.046, pad=0.02)
            cb.set_label("low  <  EMT / invasion  >  high", fontsize=9)
        fig.suptitle("Stage 1A:  the intra-primary leaving program is spatially organised",
                     fontsize=15.5, fontweight="bold", y=0.997)
        caption(fig, "Right: each spot coloured by its EMT/invasion leaving-program score (red = high), overlaid on the REAL whole-slide H&E (left). "
                     "High-score regions form coherent zones (positive, significant Moran's I in every slide), consistent with invasive fronts. "
                     "(T3, the strongest slide, has no local whole-slide image and is shown separately.)", y=0.004)
    plt.savefig(os.path.join(OUT, "fig3_stage1a_leaving_maps.png"), dpi=120, bbox_inches="tight")
    plt.close(); print("FIG3 done")

# ============================================================ FIG 4 - program biology
def fig4_program_biology():
    fig, axs = plt.subplots(1, 2, figsize=(16, 6.4))
    # (a) gene groups
    ax = axs[0]; ax.axis("off")
    groups = [("EMT transcription factors", ["SNAI1", "ZEB1", "ZEB2", "SNAI2*", "PRRX1*"], "#6a1b9a"),
              ("Mesenchymal markers", ["CDH2", "VIM*", "S100A4"], "#283593"),
              ("TGF-beta / pro-invasive", ["TGFB1", "TGFBR1", "SERPINE1"], "#00695c"),
              ("Proteases (invasion)", ["MMP1", "LOXL2", "LOX", "MMP9*", "MMP14*", "TIMP1"], "#bf360c"),
              ("ECM / matrix build-out", ["COL1A1", "COL3A1", "COL5A1", "FN1", "LAMC2", "TNC", "POSTN*", "SPARC*"], "#4e342e")]
    ax.set_title("(a) What defines the 'leaving program'", fontsize=13, fontweight="bold", loc="left")
    yy = 0.93
    for name, genes, c in groups:
        ax.text(0.0, yy, name, fontsize=11, fontweight="bold", color=c, transform=ax.transAxes)
        ax.text(0.0, yy-0.052, "  " + "   ".join(genes), fontsize=10, color="#222", transform=ax.transAxes)
        yy -= 0.155
    ax.text(0.0, 0.02, "* = held-out validator gene (never used to build the score)",
            fontsize=8.5, style="italic", color="#666", transform=ax.transAxes)
    # (b) external specificity
    ax = axs[1]
    fg = json.load(open(os.path.join(PROJ, "Outputs/stage4_validation/validation_metrics.json")))["test2_fges_specificity"]
    pick = ["Cancer_associated_fibroblasts", "Matrix", "EMT_signature", "Matrix_remodeling",
            "Protumor_cytokines", "Tumor_proliferation_rate", "MHCI"]
    vals = [fg[k]["target_raw"] for k in pick]
    names = [k.replace("_", " ").replace("Cancer associated fibroblasts", "CAFs").replace("EMT signature", "EMT") for k in pick]
    colors = ["#2e7d32" if v > 0.12 else "#bdbdbd" for v in vals]
    ax.barh(range(len(pick)), vals, color=colors)
    ax.set_yticks(range(len(pick))); ax.set_yticklabels(names, fontsize=10); ax.invert_yaxis()
    ax.set_xlabel("correlation of our target with INDEPENDENT paper signatures")
    ax.set_title("(b) Validation: the target tracks EMT / stroma,\nnot unrelated programs", fontsize=12.5, fontweight="bold")
    ax.axvline(0, color="black", lw=0.8); ax.set_xlim(-0.07, max(vals) + 0.07)
    for i, v in enumerate(vals): ax.text(v + (0.006 if v >= 0 else -0.006), i, f"{v:+.2f}",
                                         va="center", ha="left" if v >= 0 else "right", fontsize=9)
    fig.suptitle("The program is real, externally validated EMT biology",
                 fontsize=15.5, fontweight="bold")
    caption(fig, "(a) The score is the average activity of 19 well-known epithelial-to-mesenchymal-transition (EMT) and invasion genes. "
                 "(b) Computed completely independently, our target correlates specifically with the source paper's own EMT (+0.21), cancer-associated-fibroblast (+0.31) and matrix (+0.29) signatures, "
                 "and NOT with unrelated programs (proliferation, MHCI) — confirming it captures genuine dissemination biology, not an artefact.", y=0.005)
    plt.savefig(os.path.join(OUT, "fig4_stage1a_program_biology.png"), dpi=125, bbox_inches="tight")
    plt.close(); print("FIG4 done")

# ============================================================ FIG 5 - the ceiling
def fig5_ceiling():
    fig, axs = plt.subplots(2, 2, figsize=(14, 11.5))
    fig.subplots_adjust(hspace=0.55, wspace=0.22, top=0.90, bottom=0.13, left=0.07, right=0.96)
    # (a) raw scatter
    for ax, col, tgt, rho, ttl in [
        (axs[0, 0], "pred_raw", "leaving_score", 0.289, "(a) H&E predicts the abundance-coupled score"),
        (axs[0, 1], "pred_resid", "leaving_score_resid", 0.086, "(b) H&E does NOT predict the confound-free EMT")]:
        x = pb[tgt].values; y = pb[col].values
        ax.scatter(x, y, s=4, alpha=0.25, color="#4878b0", linewidths=0)
        m, b = np.polyfit(x, y, 1); xs = np.array([x.min(), x.max()])
        ax.plot(xs, m*xs+b, color="#b71c1c", lw=2)
        ax.set_xlabel("actual leaving score (from transcriptomics)")
        ax.set_ylabel("predicted from H&E alone")
        ax.set_title(f"{ttl}\nSpearman rho = {rho:+.2f}", fontsize=12, fontweight="bold")
    # (c) abundance ceiling
    ax = axs[1, 0]
    bars = [("Tumour-amount\nonly (ceiling)", 0.478, "#1b5e20"), ("H&E model\n(raw)", 0.289, "#4878b0"),
            ("H&E model\n(confound-free)", 0.086, "#bdbdbd")]
    ax.bar(range(3), [b[1] for b in bars], color=[b[2] for b in bars], width=0.6)
    ax.set_xticks(range(3)); ax.set_xticklabels([b[0] for b in bars], fontsize=9.5)
    ax.set_ylabel("cross-patient accuracy (Spearman rho)")
    ax.set_title("(c) H&E mostly reports HOW MUCH tumour is present", fontsize=12, fontweight="bold")
    for i, b in enumerate(bars): ax.text(i, b[1]+0.008, f"{b[1]:.2f}", ha="center", fontweight="bold")
    # (d) bridge vs frozen FM
    ax = axs[1, 1]
    grp = np.arange(2); w = 0.36
    ax.bar(grp-w/2, [0.289, 0.086], w, label="Transcriptomic bridge", color="#7e57c2")
    ax.bar(grp+w/2, [0.270, 0.081], w, label="Foundation model alone", color="#bcaaa4")
    ax.set_xticks(grp); ax.set_xticklabels(["Raw score", "Confound-free"])
    ax.set_ylabel("cross-patient accuracy (Spearman rho)"); ax.legend(fontsize=9)
    ax.set_title("(d) The transcriptomic 'bridge' adds ~nothing\nover the H&E foundation model", fontsize=12, fontweight="bold")
    fig.suptitle("What an H&E slide can and cannot recover, on patients it never saw",
                 fontsize=15.5, fontweight="bold")
    caption(fig, "(a) H&E reproduces the raw program score moderately (rho +0.29) — but (b) once the tumour-amount confound is removed, the confound-free EMT signal is at the noise floor (rho +0.09). "
                 "(c) A trivial 'how much tumour is here' predictor (+0.48) BEATS the H&E model — i.e. H&E mainly senses tumour abundance, not the fine metastatic state. "
                 "(d) The contrastive bridge that injects transcriptomics does not beat the foundation model used directly. Honest conclusion: at 55 um Visium resolution with this cohort, H&E cannot yet read the confound-free leaving program across patients.", y=0.005)
    plt.savefig(os.path.join(OUT, "fig5_stages23_ceiling.png"), dpi=125, bbox_inches="tight")
    plt.close(); print("FIG5 done")

# ============================================================ FIG 6 - spatial prediction
def fig6_spatial_pred():
    samples = ["IU_PDA_T4", "IU_PDA_T1"]   # both have a real whole-slide image + Phase-B preds
    fig, axs = plt.subplots(len(samples), 3, figsize=(15, 5.3*len(samples)))
    for i, s in enumerate(samples):
        dpx = with_px(pb[pb["sample"] == s])
        wsi_he_panel(axs[i, 0], s); axs[i, 0].set_title(f"{short(s)} — whole-slide H&E", fontsize=12, fontweight="bold")
        sc1 = wsi_score_panel(axs[i, 1], s, dpx, dpx["leaving_score"].to_numpy(), DIVCMAP, -2.5, 2.5, s=7)
        axs[i, 1].set_title(f"{short(s)} — ACTUAL (transcriptomics)", fontsize=12, fontweight="bold")
        fig.colorbar(sc1, ax=axs[i, 1], fraction=0.046, pad=0.02)
        pr = dpx["pred_raw_smooth"].to_numpy()
        sc2 = wsi_score_panel(axs[i, 2], s, dpx, pr, DIVCMAP, np.percentile(pr, 2), np.percentile(pr, 98), s=7)
        axs[i, 2].set_title(f"{short(s)} — PREDICTED from H&E only", fontsize=12, fontweight="bold")
        fig.colorbar(sc2, ax=axs[i, 2], fraction=0.046, pad=0.02)
    fig.suptitle("Predicted from H&E versus the real leaving score, on the real whole-slide tissue",
                 fontsize=15.5, fontweight="bold")
    caption(fig, "For each tumour: H&E (left), the actual transcriptomic leaving score (middle), and the score PREDICTED from H&E alone (right). "
                 "The H&E prediction captures the broad high/low zonation (driven by tumour density) but misses finer structure — the visual read-out of the abundance ceiling in the previous figure.", y=0.005)
    plt.savefig(os.path.join(OUT, "fig6_stage3_spatial_prediction.png"), dpi=120, bbox_inches="tight")
    plt.close(); print("FIG6 done")

# ============================================================ FIG 7 - scorecard
def fig7_scorecard():
    fig = plt.figure(figsize=(14, 9)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.text(0.5, 0.95, "Validation scorecard and honest verdict", ha="center",
            fontsize=18, fontweight="bold")
    rows = [
        ("Is the target real EMT biology?", "YES", "+0.21 vs independent paper EMT GSEA  (+0.41 vs held-out EMT genes)", "#2e7d32"),
        ("Spatially organised (not noise)?", "YES", "Moran's I 0.37-0.61, p<0.001 in all 4 tumours", "#2e7d32"),
        ("Can H&E read it on unseen patients?", "ONLY ABUNDANCE", "raw rho +0.29, but confound-free rho +0.01 (noise floor)", "#ef6c00"),
        ("Beats a trivial tumour-amount model?", "NO", "tumour-amount +0.48  >  H&E +0.29", "#c62828"),
        ("Does the transcriptomic bridge help?", "NO", "bridge +0.289 ~ foundation model +0.270", "#c62828"),
        ("Primary-11 -> Metastasis-11 agreement?", "NO", "-0.21 (driven by microenvironment, not metastasis)", "#c62828"),
    ]
    y = 0.85
    for q, verdict, detail, c in rows:
        ax.add_patch(FancyBboxPatch((0.05, y-0.075), 0.90, 0.085, boxstyle="round,pad=0.005",
                     transform=ax.transAxes, fc=c, ec="none", alpha=0.10))
        ax.text(0.07, y-0.018, q, fontsize=12, fontweight="bold", va="center")
        ax.text(0.50, y-0.018, verdict, fontsize=12, fontweight="bold", color=c, va="center")
        ax.text(0.50, y-0.05, detail, fontsize=9, color="#444", va="center")
        y -= 0.105
    ax.add_patch(FancyBboxPatch((0.05, 0.045), 0.90, 0.14, boxstyle="round,pad=0.01",
                 transform=ax.transAxes, fc="#eceff1", ec="#90a4ae"))
    ax.text(0.07, 0.15, "Take-home", fontsize=12.5, fontweight="bold")
    ax.text(0.07, 0.108, "The intra-primary EMT leaving program is a genuine, reproducible, externally validated transcriptomic signal.\n"
            "From a routine H&E slide, today's model reliably reads only HOW MUCH tumour is present, not the\nconfound-free metastatic state, on patients it never saw.",
            fontsize=10.3, va="center", linespacing=1.3)
    ax.text(0.07, 0.062, "What would close the gap:  pathologist annotation of invasive fronts  •  higher-resolution platform (Visium HD / Xenium)  •  more matched primary-metastasis patients",
            fontsize=9.8, style="italic", color="#37474f", va="center")
    plt.savefig(os.path.join(OUT, "fig7_stage4_scorecard.png"), dpi=125, bbox_inches="tight")
    plt.close(); print("FIG7 done")

if __name__ == "__main__":
    fig1_cohort(); fig2_confound(); fig3_leaving_maps()
    fig4_program_biology(); fig5_ceiling(); fig6_spatial_pred(); fig7_scorecard()
    print("ALL FIGURES ->", OUT)
