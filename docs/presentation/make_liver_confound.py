"""
Stage-0 LIVER-CONFOUND slides (v2), biology-linked and on CLEAN whole-slide H&E.
  figLsum   how-much (violin) + axis-is-liver (removable) on ONE slide
  figLwhere_<sample>  clean WSI overview + LIVER zoom + NON-LIVER zoom (tissue clear,
                      regions boxed, thin unfilled rings so tissue stays visible)
                      for HM11, HM13 and the primary tumour with the most liver
  figL2     proof: cells + genes (HM vs PT group means, r=0.93)
  figL3     generalization: 2 mets don't resemble each other

Run: "C:/Users/datai/anaconda3/envs/tcga/python.exe" docs/presentation/make_liver_confound.py
     (PPTX_SLIDE=1 for caption-free slide versions)
"""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import _tissue

ROOT = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(os.path.dirname(ROOT))
SLIDE = os.environ.get("PPTX_SLIDE") == "1"
OUT = os.path.join(PROJ, "Outputs", "presentation_figures", "slides" if SLIDE else "")
os.makedirs(OUT, exist_ok=True)
S0 = os.path.join(PROJ, "Outputs", "stage0_confound")
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white", "axes.facecolor": "white"})
PT_C, HM_C = "#4878b0", "#d1604a"; SEQ = "magma"
LIVER_EC, TUM_EC = "#00e5ff", "#7CFC00"

bio = json.load(open(os.path.join(S0, "liver_biology.json")))
zoom = json.load(open(os.path.join(S0, "liver_zoom.json")))
lead = pd.read_csv(os.path.join(PROJ, "Outputs/stage1a_leaving_program/leaving_program_scores.csv"))
hmfr = pd.read_csv(os.path.join(ROOT, "_cache_hm_fracs.csv"))
def short(s): return s.replace("IU_PDA_", "")

def caption(fig, text, y=0.01):
    if SLIDE: return
    fig.text(0.5, y, text, ha="center", va="bottom", fontsize=9.4, color="#333", wrap=True,
             bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f2", ec="#cccccc"))

def spots(sample):
    return pd.read_csv(os.path.join(S0, f"spots_{sample}.csv")).dropna(subset=["imagecol", "imagerow"])

# ============================================================ FIG Lsum - how much + removable
def figLsum():
    C = bio["counts"]
    fig, axs = plt.subplots(1, 2, figsize=(16, 7.6)); fig.subplots_adjust(top=0.84, bottom=0.20, wspace=0.28, left=0.08, right=0.96)
    # (a) violin PT vs HM
    ax = axs[0]
    pt_h = lead["hepatocyte_frac"].values; hm_h = hmfr["hepatocyte_frac"].values
    parts = ax.violinplot([pt_h, hm_h], showmeans=True, showextrema=False)
    for b, c in zip(parts["bodies"], [PT_C, HM_C]): b.set_facecolor(c); b.set_alpha(0.6)
    parts["cmeans"].set_color("black")
    ax.set_xticks([1, 2]); ax.set_xticklabels([f"Primary tumour\nn={C['n_PT']:,} spots\n(T1,T3,T4,T11)",
                                               f"Liver metastasis\nn={C['n_HM']:,} spots\n(HM11,HM13)"], fontsize=9.5)
    ax.set_ylabel("Hepatocyte (liver-cell) fraction per spot")
    ax.text(1, pt_h.mean()+0.04, f"mean {pt_h.mean():.3f}", ha="center", fontsize=9)
    ax.text(2, hm_h.mean()+0.04, f"mean {hm_h.mean():.2f}", ha="center", fontsize=9, fontweight="bold", color=HM_C)
    ax.set_title("(a) How much liver contamination", fontsize=12.5, fontweight="bold")
    # (b) axis is liver, removable
    ax = axs[1]
    vals = [bio["corr_axis_hepatocyte_all"], bio["corr_axis_hepatocyte_tumoronly"]]
    ax.bar([0, 1], vals, color=["#9e9e9e", "#2e7d32"], width=0.55)
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"ALL spots\nn={C['n_total']:,}\n(6 samples)",
                                               f"TUMOUR-ONLY\nn={C['n_tumor_only']:,}\n(Tumor ≥ 0.5)"], fontsize=9.5)
    ax.set_ylabel("corr( metastasis axis , liver content )")
    for i, v in enumerate(vals): ax.text(i, v+0.012, f"{v:.2f}", ha="center", fontweight="bold", fontsize=13)
    ax.annotate("confound\nremoved", xy=(1, vals[1]), xytext=(0.55, 0.30), fontsize=10.5, color="#2e7d32",
                fontweight="bold", ha="center", arrowprops=dict(arrowstyle="-|>", color="#2e7d32", lw=1.6))
    ax.set_title("(b) That contamination IS the 'metastasis' axis — and it is removable", fontsize=11.5, fontweight="bold")
    fig.suptitle("Liver-cell contamination: how much, and why it fakes a 'metastasis' signal",
                 fontsize=16, fontweight="bold", y=0.95)
    caption(fig, f"(a) Liver metastasis spots average {hm_h.mean():.2f} liver-cell content vs {pt_h.mean():.3f} in primary tumour — a tissue-of-origin confound (PT n={C['n_PT']:,} from 4 slides, HM n={C['n_HM']:,} from 2). "
                 f"(b) The naive HM-minus-PT 'metastasis direction' correlates 0.40 with that liver content across all {C['n_total']:,} spots; restricting to the {C['n_tumor_only']:,} tumour-dominated spots collapses it to 0.07. "
                 "Most of the apparent metastasis signal was the liver the tumour sits in — so we target a program INSIDE the primary tumour instead.", y=0.02)
    plt.savefig(os.path.join(OUT, "figLsum_howmuch_removable.png"), dpi=125, bbox_inches="tight")
    plt.close(); print("FIG Lsum done")

# ============================================================ FIG Lwhere - WSI + clean zooms
def rings(ax, d, cx0, cy0, s, ec, lw=1.1, r=7):
    for _, row in d.iterrows():
        ax.add_patch(Circle(((row["imagecol"]-cx0)*s, (row["imagerow"]-cy0)*s), r,
                            fill=False, ec=ec, lw=lw))

def _box_around(d, span=0.16):
    """Small box around the single highest-hepatocyte spot (fallback when no cluster)."""
    r = d.nlargest(1, "hepatocyte_frac").iloc[0]
    fw = d["imagecol"].max() - d["imagecol"].min(); fh = d["imagerow"].max() - d["imagerow"].min()
    sx, sy = span*fw, span*fh
    return dict(x0=float(r["imagecol"]-sx/2), y0=float(r["imagerow"]-sy/2),
                x1=float(r["imagecol"]+sx/2), y1=float(r["imagerow"]+sy/2))

def figLwhere(sample, note_liver, note_nonliver, liver_thresh=0.4,
              liver_title="LIVER-cell-rich zone", is_pt=False):
    d = spots(sample)
    z = zoom[sample]
    wsi, scale = _tissue.render_wsi(sample, max_dim=2000)
    lb = z.get("liver_box") or _box_around(d)
    nb = z.get("nonliver_box")
    fig = plt.figure(figsize=(19, 8.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.0], wspace=0.10,
                          left=0.02, right=0.995, top=0.85, bottom=0.10)
    # overview
    axo = fig.add_subplot(gs[0, 0]); axo.imshow(wsi)
    sc = axo.scatter(d["imagecol"]*scale, d["imagerow"]*scale, c=d["hepatocyte_frac"],
                     cmap=SEQ, vmin=0, vmax=1, s=5, marker="h", linewidths=0)
    axo.set_xticks([]); axo.set_yticks([]); axo.set_aspect("equal")
    fig.colorbar(sc, ax=axo, fraction=0.046, pad=0.02).set_label("hepatocyte fraction", fontsize=9)
    sub = f"max liver-cell fraction here ≈ {d['hepatocyte_frac'].max()*100:.0f}%" if is_pt else "spots = liver-cell content"
    axo.set_title(f"{short(sample)} — real whole-slide H&E\n({sub})", fontsize=12, fontweight="bold")
    for b, ec in [(lb, LIVER_EC), (nb, TUM_EC)]:
        if b:
            axo.add_patch(Rectangle((b["x0"]*scale, b["y0"]*scale), (b["x1"]-b["x0"])*scale,
                          (b["y1"]-b["y0"])*scale, fill=False, ec=ec, lw=3))
    # liver (or top-liver) zoom
    axl = fig.add_subplot(gs[0, 1])
    crop, (x0, y0, x1, y1, s) = _tissue.crop_wsi(sample, lb["x0"], lb["y0"], lb["x1"], lb["y1"], out_px=1000)
    axl.imshow(crop)
    inb = d[(d["imagecol"].between(x0, x1)) & (d["imagerow"].between(y0, y1)) & (d["hepatocyte_frac"] > liver_thresh)]
    rings(axl, inb, x0, y0, s, LIVER_EC, lw=1.2, r=6)
    axl.set_xlabel(note_liver, fontsize=9.5, color="#0097a7")
    axl.set_xticks([]); axl.set_yticks([])
    for sp in axl.spines.values(): sp.set_edgecolor(LIVER_EC); sp.set_linewidth(3)
    axl.set_title(liver_title, fontsize=12, fontweight="bold", color="#0097a7")
    # non-liver zoom
    axn = fig.add_subplot(gs[0, 2])
    if nb:
        crop, (x0, y0, x1, y1, s) = _tissue.crop_wsi(sample, nb["x0"], nb["y0"], nb["x1"], nb["y1"], out_px=1000)
        axn.imshow(crop)
        inb = d[(d["imagecol"].between(x0, x1)) & (d["imagerow"].between(y0, y1)) & (d["tumor_frac"] > 0.5)]
        rings(axn, inb, x0, y0, s, TUM_EC, lw=1.2, r=6)
        axn.set_xlabel(note_nonliver, fontsize=9.5, color="#558b2f")
    axn.set_xticks([]); axn.set_yticks([])
    for sp in axn.spines.values(): sp.set_edgecolor(TUM_EC); sp.set_linewidth(3)
    axn.set_title("Tumour / non-liver zone", fontsize=12, fontweight="bold", color="#558b2f")

    if is_pt:
        fig.suptitle(f"{short(sample)} (PRIMARY tumour) — essentially NO liver cells anywhere",
                     fontsize=16, fontweight="bold", y=0.95)
        caption(fig, f"The primary tumour sits in the pancreas, not the liver: across all {len(d):,} spots the maximum liver-cell fraction is only ~{d['hepatocyte_frac'].max()*100:.0f}% (mean {d['hepatocyte_frac'].mean():.3f}). "
                     "Left: every spot is dark (near-zero liver content). Middle/right: zoomed tumour tissue is malignant/desmoplastic with no liver parenchyma. "
                     "This is the clean contrast to the metastases — the liver confound exists ONLY in the metastasis slides, which is why the target is defined inside the primary tumour.", y=0.02)
    else:
        fig.suptitle(f"{short(sample)} (liver METASTASIS) — where the liver cells are, on the real tissue",
                     fontsize=16, fontweight="bold", y=0.95)
        caption(fig, f"Left: the real {short(sample)} whole-slide, each spot coloured by liver-cell content (cyan box = liver-rich zone, green box = tumour zone). "
                     "Middle: zoomed into the liver-rich zone — the tissue has the cords-and-sinusoids look of normal liver (thin cyan rings mark spots with hepatocyte fraction > 0.4). "
                     "Right: zoomed into a tumour zone — dense malignant/desmoplastic tissue with essentially no liver cells. Rings are unfilled so the underlying H&E stays visible.", y=0.02)
    plt.savefig(os.path.join(OUT, f"figLwhere_{short(sample)}.png"), dpi=125, bbox_inches="tight")
    plt.close(); print(f"FIG Lwhere {sample} done")

# ============================================================ FIG L2 - proof (fixed bars)
def figL2():
    GENES = ["ALB", "TTR", "APOA1", "HP", "FGB", "SERPINA1"]
    hm = pd.read_csv(os.path.join(S0, "hm11_spots.csv"))
    pt = pd.read_csv(os.path.join(S0, "t1_spots.csv"))
    hm_liver = hm[hm["hepatocyte_frac"] > 0.4]; pt_tum = pt[pt["tumor_frac"] > 0.5]
    ex = bio["examples"]["HM_high"] + bio["examples"]["PT_low"]
    tags = ["HM · liver-rich", "HM · liver-rich", "PT · tumour", "PT · tumour"]
    cols = [HM_C, HM_C, PT_C, PT_C]
    fig = plt.figure(figsize=(19, 9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0], hspace=0.42, wspace=0.30,
                          left=0.05, right=0.97, top=0.88, bottom=0.10)
    for i, (e, tag, c) in enumerate(zip(ex, tags, cols)):
        axp = fig.add_subplot(gs[0, i])
        p = os.path.join(PROJ, "dataset", ".png patches", ".png patches", e["sample"], e["patch_stem"] + ".png")
        from PIL import Image
        if os.path.exists(p): axp.imshow(np.asarray(Image.open(p).convert("RGB")))
        axp.set_xticks([]); axp.set_yticks([])
        for sp in axp.spines.values(): sp.set_edgecolor(c); sp.set_linewidth(4)
        axp.set_title(tag, fontsize=11.5, fontweight="bold", color=c)
        axp.set_xlabel(f"liver-cell {int(e['hepatocyte_frac']*100)}%   ·   tumour {int(e['tumor_frac']*100)}%",
                       fontsize=9.5, color=c, fontweight="bold")
    # (a) GROUP-MEAN gene bars: HM liver-rich vs PT tumour (two clear colours)
    axg = fig.add_subplot(gs[1, 0:2])
    x = np.arange(len(GENES)); w = 0.38
    hm_v = [hm_liver[f"g_{g}"].mean() if f"g_{g}" in hm_liver else 0 for g in GENES]
    pt_v = [pt_tum[f"g_{g}"].mean() if f"g_{g}" in pt_tum else 0 for g in GENES]
    axg.bar(x - w/2, hm_v, w, color=HM_C, label=f"HM liver-rich spots (n={len(hm_liver):,})")
    axg.bar(x + w/2, pt_v, w, color=PT_C, label=f"PT tumour spots (n={len(pt_tum):,})")
    axg.set_xticks(x); axg.set_xticklabels(GENES, fontsize=10.5)
    axg.set_ylabel("hepatocyte-gene expression\n(CP10k log, group mean)")
    axg.set_title("(a) Liver-rich spots express hepatocyte genes; tumour spots ≈ 0", fontsize=12, fontweight="bold")
    axg.legend(fontsize=9.5, loc="upper right")
    # (b) RCTD hepatocyte vs liver-gene score
    axs = fig.add_subplot(gs[1, 2:4])
    v = hm.dropna(subset=["hepa_gene_score"])
    axs.scatter(v["hepatocyte_frac"], v["hepa_gene_score"], s=6, alpha=0.25, color=HM_C, linewidths=0)
    m, b = np.polyfit(v["hepatocyte_frac"], v["hepa_gene_score"], 1)
    xs = np.array([0, 1]); axs.plot(xs, m*xs+b, color="black", lw=2)
    axs.set_xlabel("RCTD hepatocyte fraction (cell deconvolution)")
    axs.set_ylabel("hepatocyte marker-gene score")
    axs.set_title(f"(b) Cell call agrees with genes — HM11: r = {bio['corr_rctd_vs_livergenes_HM11']:.2f}  (n={len(v):,})",
                  fontsize=12, fontweight="bold")
    fig.suptitle("Proof it is really liver: the same spots carry liver CELLS and liver GENES",
                 fontsize=17, fontweight="bold", y=0.965)
    caption(fig, "Two liver-rich metastasis spots (red) and two primary-tumour spots (blue) with their real H&E and cell composition. "
                 "(a) Group means over ALL liver-rich HM spots vs ALL PT tumour spots: the liver spots strongly express hepatocyte genes — ALB, TTR, APOA1, HP, FGB, SERPINA1 — which tumour spots essentially lack. "
                 f"(b) Across HM11 the cell-deconvolution hepatocyte fraction and the liver-gene score agree almost perfectly (r={bio['corr_rctd_vs_livergenes_HM11']:.2f}): the liver call is real biology.", y=0.004)
    plt.savefig(os.path.join(OUT, "figL2_liver_proof.png"), dpi=125, bbox_inches="tight")
    plt.close(); print("FIG L2 done")

# ============================================================ FIG L3 - generalization
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
    caption(fig, "Leave-one-sample-out: hold out each sample and ask if the model recognises its tumour type from the rest. "
                 "Primary tumours generalise (93–100%), but each liver metastasis is NOT recovered from the other (0%) — there is no transferable, "
                 "tumour-intrinsic metastasis signature to learn from just two mets.", y=0.02)
    plt.savefig(os.path.join(OUT, "figL3_generalization.png"), dpi=125, bbox_inches="tight")
    plt.close(); print("FIG L3 done")

if __name__ == "__main__":
    figLsum(); figL2(); figL3()
    figLwhere("IU_PDA_HM11", "liver parenchyma: cords + sinusoids", "dense tumour / desmoplasia")
    figLwhere("IU_PDA_HM13", "liver parenchyma at the metastasis edge", "malignant tumour tissue")
    pt = zoom["PT_most_liver"]
    figLwhere(pt, "even the 'most liver' spot here is barely any", "typical primary-tumour tissue",
              liver_thresh=0.1, liver_title="Highest-liver spot (still ≈ 0)", is_pt=True)
    print("DONE ->", OUT, "| PT with most liver:", pt)
