"""
Worked-example figure: makes the method traceable and verifiable for a pathologist.
Tumour T3. Left: where two real tumour spots sit. Middle: their actual H&E (judge the
morphology by eye). Right: across ALL tumour spots, the high score group expresses the
EMT/invasion genes more than the low score group (group level removes single spot noise).
Each spot also carries an independent paper's EMT score, which agrees.
"""
import os, csv
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import _tissue

ROOT = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(os.path.dirname(ROOT))
SLIDE = os.environ.get("PPTX_SLIDE") == "1"
OUT = os.path.join(PROJ, "Outputs", "presentation_figures", "slides" if SLIDE else "")
os.makedirs(OUT, exist_ok=True)
PATCH = os.path.join(PROJ, "dataset", ".png patches", ".png patches", "IU_PDA_T3")
BLUE, ORANGE = "#2C5F8A", "#E07B54"

HIGH = dict(stem="IU_PDA_T3_patch-000842_38_102", row=38, col=102, leaving=+2.26, emt_pct=69)
LOW  = dict(stem="IU_PDA_T3_patch-000403_17_73", row=17, col=73, leaving=-0.58, emt_pct=28)
MES = ["SNAI1", "ZEB1", "CDH2", "S100A4", "TGFB1", "MMP1", "LOXL2", "FN1", "COL1A1", "TNC"]
EPI = ["CDH1", "EPCAM"]; GENES = MES + EPI

def group_means():
    lead = pd.read_csv(os.path.join(PROJ, "Outputs/stage1a_leaving_program/leaving_program_scores.csv"))
    d = lead[(lead["sample"] == "IU_PDA_T3") & (lead["tumor_frac"] >= 0.5)]
    hi = set(d[d["leaving_score"] >= d["leaving_score"].quantile(0.75)]["barcode"])
    lo = set(d[d["leaving_score"] <= d["leaving_score"].quantile(0.25)]["barcode"])
    f = os.path.join(PROJ, "dataset", "ST", "scVI_counts", "IU_PDA_T3.csv")
    with open(f, newline="") as fh:
        rd = csv.reader(fh); header = [h.strip('"') for h in next(rd)]
        hidx = [i - 1 for i, h in enumerate(header) if h in hi]
        lidx = [i - 1 for i, h in enumerate(header) if h in lo]
        res, want = {}, set(GENES)
        for rv in rd:
            g = rv[0].strip('"')
            if g in want:
                arr = np.array(rv[1:], dtype=float); z = (arr - arr.mean()) / (arr.std() + 1e-9)
                res[g] = (float(z[hidx].mean()), float(z[lidx].mean())); want.discard(g)
                if not want: break
    return res

def patch_img(stem): return np.asarray(Image.open(os.path.join(PATCH, stem + ".png")).convert("RGB"))

def spot_px():
    """imagerow/imagecol for the HIGH/LOW spots on the real T3 WSI (via barcode)."""
    lead = pd.read_csv(os.path.join(PROJ, "Outputs/stage1a_leaving_program/leaving_program_scores.csv"))
    coord = pd.read_csv(os.path.join(PROJ, "Outputs/Patient-Sample-Information/spot_spatial_coordinates.csv"))
    coord = coord[coord["image"] == "IU_PDA_T3"].set_index("spot_barcode")
    bc = lead.set_index("patch_stem")["barcode"]
    out = {}
    for tag, sp in [("HIGH", HIGH), ("LOW", LOW)]:
        b = bc.get(sp["stem"])
        if b in coord.index:
            out[tag] = (float(coord.loc[b, "imagecol"]), float(coord.loc[b, "imagerow"]))
    return out

def build():
    res = group_means()
    wsi, scale = _tissue.render_wsi("IU_PDA_T3", max_dim=2000)
    px = spot_px()
    fig = plt.figure(figsize=(17.2, 9.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 0.6, 1.5], height_ratios=[1, 1],
                          hspace=0.30, wspace=0.26, left=0.035, right=0.99, top=0.83, bottom=0.06)

    axm = fig.add_subplot(gs[:, 0]); axm.imshow(wsi); axm.set_aspect("equal")
    for tag, sp, c, lab in [("HIGH", HIGH, ORANGE, "high"), ("LOW", LOW, BLUE, "low")]:
        if tag not in px: continue
        x, y = px[tag][0]*scale, px[tag][1]*scale
        axm.scatter([x], [y], s=520, facecolors="none", edgecolors=c, linewidths=3.4)
        axm.annotate(lab, (x, y), xytext=(15, 13), textcoords="offset points",
                     fontsize=12.5, fontweight="bold", color=c)
    axm.set_xticks([]); axm.set_yticks([])
    axm.set_title("1.  Where the two spots sit in tumour T3  (real H&E)", fontsize=13.5, fontweight="bold", color=BLUE)

    for r, sp, c, name in [(0, HIGH, ORANGE, "HIGH score spot"), (1, LOW, BLUE, "LOW score spot")]:
        axp = fig.add_subplot(gs[r, 1]); axp.imshow(patch_img(sp["stem"]))
        axp.set_xticks([]); axp.set_yticks([])
        for s in axp.spines.values(): s.set_color(c); s.set_linewidth(3.5)
        ttl = "2.  Its real H&E" if r == 0 else ""
        axp.set_title(f"{ttl}\n{name}\nscore {sp['leaving']:+.2f}   |   paper EMT {sp['emt_pct']}th pct",
                      fontsize=10.5, fontweight="bold", color=c)

    axg = fig.add_subplot(gs[:, 2])
    y = np.arange(len(GENES)); w = 0.4
    hi_v = [res[g][0] for g in GENES]; lo_v = [res[g][1] for g in GENES]
    axg.barh(y - w/2, hi_v, w, color=ORANGE, label="High score spots")
    axg.barh(y + w/2, lo_v, w, color=BLUE, label="Low score spots")
    axg.set_yticks(y); axg.set_yticklabels(GENES, fontsize=11); axg.invert_yaxis()
    axg.axvline(0, color="black", lw=0.8); axg.axhline(len(MES) - 0.5, color="#bbbbbb", lw=1, ls="--")
    axg.text(axg.get_xlim()[1], len(MES)/2 - 0.5, " EMT / invasion", rotation=90,
             va="center", ha="left", fontsize=10, color=ORANGE, fontweight="bold")
    axg.text(axg.get_xlim()[1], len(MES) + 0.5, " epithelial", rotation=90,
             va="center", ha="left", fontsize=10, color=BLUE, fontweight="bold")
    axg.set_xlabel("average gene level (z within slide)", fontsize=11)
    axg.legend(loc="lower right", fontsize=10.5)
    axg.set_title("3.  Across all tumour spots, high score spots switch the EMT genes on",
                  fontsize=13, fontweight="bold", color=BLUE)

    fig.suptitle("Reading one result end to end:  tissue  >  H&E  >  genes  >  score  >  outside confirmation",
                 fontsize=17, fontweight="bold", color=BLUE, y=0.955)
    fig.text(0.035, 0.88, "A pathologist can judge the H&E by eye, then check that the molecular profile and an independent paper's EMT score "
             "point the same way. The score is not a black box.", fontsize=11.5, color="#333333")
    fig.savefig(os.path.join(OUT, "fig_worked_example.png"), dpi=125, bbox_inches="tight")
    plt.close(); print("worked example rebuilt")

if __name__ == "__main__":
    build()
