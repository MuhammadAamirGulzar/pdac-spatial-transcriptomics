"""
Presentation figures explaining the leaving PROGRAM -> SCORE -> CONFOUND -> TARGET,
grounded in real IU_PDA_T3 spots, plus the gene-level importance breakdown.

Produces (Outputs/presentation_figures/):
  figA_leaving_worked_spots.png   3 real T3 spots (A/B/C) on H&E + the confound scatter
  figB_leaving_gene_importance.png which genes drive the confound-free target

Run: "C:/Users/datai/anaconda3/envs/tcga/python.exe" docs/presentation/make_leaving_explainer.py
(set PPTX_SLIDE=1 for caption-free slide versions -> slides/ subdir)
"""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from PIL import Image
import _tissue

ROOT = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(os.path.dirname(ROOT))
SLIDE = os.environ.get("PPTX_SLIDE") == "1"
OUT = os.path.join(PROJ, "Outputs", "presentation_figures", "slides" if SLIDE else "")
os.makedirs(OUT, exist_ok=True)
PATCH_DIR = os.path.join(PROJ, "dataset", ".png patches", ".png patches")
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white", "axes.facecolor": "white"})
DIV = "RdBu_r"

lead = pd.read_csv(os.path.join(PROJ, "Outputs/stage1a_leaving_program/leaving_program_scores.csv"))
gi = json.load(open(os.path.join(PROJ, "Outputs/stage1a_leaving_program/gene_importance.json")))
def short(s): return s.replace("IU_PDA_", "")

def caption(fig, text, y=0.01):
    if SLIDE: return
    fig.text(0.5, y, text, ha="center", va="bottom", fontsize=9.4, color="#333", wrap=True,
             bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f2", ec="#cccccc"))

# Three real example spots on IU_PDA_T4 — the sample HAS a real whole-slide image
# (T3's WSI is not on this machine). imagerow/imagecol are WSI pixel coords.
WSAMPLE = "IU_PDA_T4"
SPOTS = [
    dict(tag="A", stem="IU_PDA_T4_patch-002734_71_5",   imgrow=2699,  imgcol=3682,
         tf=0.37, raw=+3.38, resid=+4.03, read="Invasive front\n(true positive)"),
    dict(tag="B", stem="IU_PDA_T4_patch-003397_24_82",  imgrow=13105, imgcol=14763,
         tf=0.18, raw=+2.16, resid=+3.59, read="Few cells, strong\nprogram (rescued)"),
    dict(tag="C", stem="IU_PDA_T4_patch-001352_55_109", imgrow=16769, imgcol=7468,
         tf=0.82, raw=+0.12, resid=-2.46, read="Bulky tumour core\n(not invasive)"),
]
COLA, COLB, COLC = "#2e7d32", "#1565c0", "#c62828"
SPOT_COL = {"A": COLA, "B": COLB, "C": COLC}
PATCH_PNG = os.path.join(PATCH_DIR, WSAMPLE)

# ============================================================ FIG A - worked spots
def figA():
    s = WSAMPLE
    d = lead[lead["sample"] == s]
    conf = np.corrcoef(d["tumor_frac"], d["leaving_score"])[0, 1]
    wsi, scale = _tissue.render_wsi(s, max_dim=2000)
    fig = plt.figure(figsize=(19.5, 10.2))
    gs = fig.add_gridspec(3, 3, width_ratios=[1.55, 0.72, 1.15], height_ratios=[1, 1, 1],
                          hspace=0.42, wspace=0.16, left=0.02, right=0.985, top=0.87, bottom=0.11)

    # (1) REAL whole-slide H&E with the 3 spots marked
    axh = fig.add_subplot(gs[:, 0])
    axh.imshow(wsi); axh.set_xticks([]); axh.set_yticks([]); axh.set_aspect("equal")
    for sp in SPOTS:
        x, y = sp["imgcol"] * scale, sp["imgrow"] * scale
        axh.add_patch(Circle((x, y), 34, fill=False, ec=SPOT_COL[sp["tag"]], lw=3.5))
        axh.annotate(sp["tag"], (x, y - 52), color="white", fontsize=15, fontweight="bold",
                     ha="center", va="center",
                     bbox=dict(boxstyle="circle,pad=0.25", fc=SPOT_COL[sp["tag"]], ec="white", lw=1.5))
    axh.set_title(f"{short(s)} tumour — REAL whole-slide H&E", fontsize=14, fontweight="bold")

    # (2) the three ACTUAL 224px spot patches (pathologist can validate by eye)
    for i, sp in enumerate(SPOTS):
        axp = fig.add_subplot(gs[i, 1])
        pf = os.path.join(PATCH_PNG, sp["stem"] + ".png")
        if os.path.exists(pf):
            axp.imshow(np.asarray(Image.open(pf).convert("RGB")))
        axp.set_xticks([]); axp.set_yticks([])
        for spine in axp.spines.values(): spine.set_edgecolor(SPOT_COL[sp["tag"]]); spine.set_linewidth(4)
        axp.set_title(f"{sp['tag']}  ·  tumour {int(sp['tf']*100)}%", fontsize=11,
                      fontweight="bold", color=SPOT_COL[sp["tag"]])
        axp.set_xlabel(f"raw {sp['raw']:+.1f}   →   resid {sp['resid']:+.1f}", fontsize=9.5,
                       color=SPOT_COL[sp["tag"]], fontweight="bold")
    fig.text(0.485, 0.885, "actual spot tissue\n(55µm H&E patch)", ha="center", fontsize=9.5,
             style="italic", color="#555")

    # (3) the confound scatter: raw score vs tumour fraction
    axs = fig.add_subplot(gs[0:2, 2])
    axs.scatter(d["tumor_frac"], d["leaving_score"], s=5, alpha=0.16, color="#888", linewidths=0)
    m, b = np.polyfit(d["tumor_frac"], d["leaving_score"], 1)
    xs = np.array([0, d["tumor_frac"].max()])
    axs.plot(xs, m*xs+b, color="#b71c1c", lw=2)
    for sp in SPOTS:
        axs.scatter([sp["tf"]], [sp["raw"]], s=150, color=SPOT_COL[sp["tag"]], edgecolor="white", zorder=5)
        axs.annotate(sp["tag"], (sp["tf"], sp["raw"]), color=SPOT_COL[sp["tag"]], fontsize=13,
                     fontweight="bold", xytext=(7, 4), textcoords="offset points")
    axs.set_xlabel("tumour fraction (RCTD)"); axs.set_ylabel("raw leaving score (z)")
    axs.set_title(f"The CONFOUND: raw score rises with\ntumour amount  (corr = {conf:.2f})",
                  fontsize=12, fontweight="bold")

    # (4) per-spot values bar
    axb = fig.add_subplot(gs[2, 2])
    xpos = np.arange(3); w = 0.26
    tf = [sp["tf"] for sp in SPOTS]; raw = [sp["raw"] for sp in SPOTS]; res = [sp["resid"] for sp in SPOTS]
    axb.bar(xpos - w, tf, w, label="tumour frac", color="#bdbdbd")
    axb.bar(xpos, raw, w, label="raw score", color="#9575cd")
    axb.bar(xpos + w, res, w, label="resid (target)", color="#ef6c00")
    axb.axhline(0, color="black", lw=0.8)
    axb.set_xticks(xpos); axb.set_xticklabels([sp["tag"] for sp in SPOTS], fontsize=11, fontweight="bold")
    axb.set_ylabel("value"); axb.legend(fontsize=8, loc="lower left")
    axb.set_title("Residual splits A/B (invasive) from C (bulk)", fontsize=11, fontweight="bold")

    fig.suptitle(f"Program → Score → Confound → Target, on three real {short(s)} spots",
                 fontsize=18, fontweight="bold", y=0.955)
    caption(fig,
        f"Left: the REAL whole-slide H&E of {short(s)} with three spots marked. Middle: each spot's actual 55µm H&E patch — a pathologist can judge invasiveness by eye (A/B fibrous, infiltrative edges; C a dense tumour sheet). "
        f"Right-top: the confound — the RAW score climbs with tumour amount (red line), so a dense spot scores high just by having more tumour. "
        f"Right-bottom: A (invasive front) & B (few but strongly-invasive cells) stay HIGH after removing abundance, while C ({int(SPOTS[2]['tf']*100)}% tumour, bulky core) flips NEGATIVE — the residual is what we actually model.", y=0.004)
    plt.savefig(os.path.join(OUT, "figA_leaving_worked_spots.png"), dpi=125, bbox_inches="tight")
    plt.close(); print("FIG A done")

# ============================================================ FIG B - gene importance
MODULE = {  # gene -> (module label, colour)
    "SNAI1":("EMT TF","#6a1b9a"),"ZEB1":("EMT TF","#6a1b9a"),"ZEB2":("EMT TF","#6a1b9a"),
    "CDH2":("Mesenchymal","#283593"),"S100A4":("Mesenchymal","#283593"),
    "TGFB1":("TGF-β / pro-invasive","#00695c"),"TGFBR1":("TGF-β / pro-invasive","#00695c"),
    "SERPINE1":("TGF-β / pro-invasive","#00695c"),
    "MMP1":("Protease","#bf360c"),"LOXL2":("Protease","#bf360c"),"LOX":("Protease","#bf360c"),"TIMP1":("Protease","#bf360c"),
    "COL1A1":("ECM","#4e342e"),"COL3A1":("ECM","#4e342e"),"COL5A1":("ECM","#4e342e"),
    "FN1":("ECM","#4e342e"),"LAMC2":("ECM","#4e342e"),"TNC":("ECM","#4e342e"),"ITGB6":("ECM","#4e342e"),
}
def figB():
    core = pd.DataFrame(gi["core_ranked"]).sort_values("corr_resid")
    fig, axs = plt.subplots(1, 2, figsize=(17, 8.2))
    fig.subplots_adjust(top=0.86, bottom=0.16, wspace=0.30, left=0.09, right=0.97)

    # (a) 19 genes ranked by corr with confound-free target, coloured by module
    ax = axs[0]
    cols = [MODULE[g][1] for g in core["gene"]]
    ax.barh(range(len(core)), core["corr_resid"], color=cols)
    ax.set_yticks(range(len(core))); ax.set_yticklabels(core["gene"], fontsize=9.5)
    ax.set_xlabel("correlation with confound-free target (avg of 4 tumours)")
    ax.set_title("(a) Which genes DRIVE the target\nECM / protease / TGF-β lead; classic EMT-TFs trail",
                 fontsize=12, fontweight="bold")
    for i, v in enumerate(core["corr_resid"]):
        ax.text(v + 0.004, i, f"{v:.2f}", va="center", fontsize=8)
    seen = {}
    for g in core["gene"]:
        lab, c = MODULE[g]; seen[lab] = c
    handles = [plt.Line2D([0],[0], marker="s", ls="", mfc=c, mec=c, label=l) for l, c in seen.items()]
    ax.legend(handles=handles, fontsize=8.5, loc="lower right", title="module")
    ax.set_xlim(0, core["corr_resid"].max() + 0.06)

    # (b) abundance-driven vs tumour-intrinsic: raw vs resid correlation
    ax = axs[1]
    cr = pd.DataFrame(gi["core_ranked"])
    ax.plot([0, 0.65], [0, 0.65], ls="--", color="#999", lw=1)
    for _, r in cr.iterrows():
        c = MODULE[r["gene"]][1]
        ax.scatter(r["corr_raw"], r["corr_resid"], s=70, color=c, edgecolor="white", zorder=4)
        ax.annotate(r["gene"], (r["corr_raw"], r["corr_resid"]), fontsize=8,
                    xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel("corr with RAW score  (abundance-coupled)")
    ax.set_ylabel("corr with RESID target  (tumour-intrinsic)")
    ax.set_title("(b) Above line = tumour-intrinsic (TNC, LAMC2, ITGB6)\nFar-right, below = abundance markers (COL1A1, FN1)",
                 fontsize=12, fontweight="bold")
    ax.text(0.55, 0.05, "collagen / FN1 =\nHOW MUCH tumour", fontsize=8.5, color="#4e342e", ha="center")
    ax.text(0.12, 0.42, "intrinsic\ninvasion", fontsize=8.5, color="#00695c", ha="center")

    fig.suptitle("Not all 19 genes contribute equally — and the residual re-weights them",
                 fontsize=16, fontweight="bold", y=0.955)
    caption(fig,
        "(a) Averaged over all 4 tumours, the confound-free target is driven by SECRETED ECM / protease / TGF-β genes (SERPINE1, S100A4, COL5A1, LAMC2, FN1), not the classic EMT transcription factors (SNAI1, ZEB1) — expected at 55µm Visium where TFs are low-count and noisy. "
        "(b) Collagens & FN1 correlate strongly with the RAW score but drop after residualising (they track tumour amount), whereas TNC, LAMC2, ITGB6 sit ABOVE the line (more linked to the confound-free invasive state). "
        "Data-driven across all ~17,900 genes, the top hits ARE the panel genes; epithelial markers (CDH1, EPCAM) sit at ~0, so at this resolution the signal is an added invasion/ECM program, not a clean epithelial shut-off.", y=0.004)
    plt.savefig(os.path.join(OUT, "figB_leaving_gene_importance.png"), dpi=125, bbox_inches="tight")
    plt.close(); print("FIG B done")

if __name__ == "__main__":
    figA(); figB()
    print("DONE ->", OUT)
