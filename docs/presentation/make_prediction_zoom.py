"""
Per-sample H&E-prediction slides on REAL whole-slide tissue, with zooms into a
region where transcriptomics and the H&E-only prediction AGREE and one where they
DISAGREE (thin rings on the spots so the tissue stays visible).

One figure per sample: WSI | ACTUAL (transcriptomics) | PREDICTED (H&E only)
                       + AGREE zoom (actual|pred) + DISAGREE zoom (actual|pred).

Run: "C:/Users/datai/anaconda3/envs/tcga/python.exe" docs/presentation/make_prediction_zoom.py
     (PPTX_SLIDE=1 for slide versions)
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import _tissue

ROOT = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(os.path.dirname(ROOT))
SLIDE = os.environ.get("PPTX_SLIDE") == "1"
OUT = os.path.join(PROJ, "Outputs", "presentation_figures", "slides" if SLIDE else "")
os.makedirs(OUT, exist_ok=True)
DIV = "RdBu_r"; AGREE_EC, DIS_EC = "#00c853", "#d50000"
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white", "axes.facecolor": "white"})
def short(s): return s.replace("IU_PDA_", "")

pb = pd.read_csv(os.path.join(PROJ, "Outputs/stage3_phase_b/phase_b_scores.csv"))
lead = pd.read_csv(os.path.join(PROJ, "Outputs/stage1a_leaving_program/leaving_program_scores.csv"))
coord = pd.read_csv(os.path.join(PROJ, "Outputs/Patient-Sample-Information/spot_spatial_coordinates.csv"))

def caption(fig, text, y=0.01):
    if SLIDE: return
    fig.text(0.5, y, text, ha="center", va="bottom", fontsize=9.4, color="#333", wrap=True,
             bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f2", ec="#cccccc"))

def prep(sample):
    d = pb[pb["sample"] == sample].merge(lead[["patch_stem", "barcode"]], on="patch_stem", how="left")
    c = coord[coord["image"] == sample][["spot_barcode", "imagerow", "imagecol"]]
    d = d.merge(c, left_on="barcode", right_on="spot_barcode", how="inner").dropna(subset=["imagecol"])
    az = (d["leaving_score"] - d["leaving_score"].mean()) / (d["leaving_score"].std() + 1e-9)
    pz = (d["pred_raw_smooth"] - d["pred_raw_smooth"].mean()) / (d["pred_raw_smooth"].std() + 1e-9)
    d = d.assign(az=az.values, pz=pz.values, concord=(az*pz).values)
    return d

def dense_box(sub, d, span=0.16):
    if len(sub) < 6: return None
    fw = d["imagecol"].max()-d["imagecol"].min(); fh = d["imagerow"].max()-d["imagerow"].min()
    H, xe, ye = np.histogram2d(sub["imagecol"], sub["imagerow"], bins=12)
    i, j = np.unravel_index(np.argmax(H), H.shape)
    cx = 0.5*(xe[i]+xe[i+1]); cy = 0.5*(ye[j]+ye[j+1])
    return dict(x0=float(cx-span*fw/2), y0=float(cy-span*fh/2), x1=float(cx+span*fw/2), y1=float(cy+span*fh/2))

def score_on_wsi(ax, wsi, scale, d, col, vmin=-2.5, vmax=2.5, s=6):
    ax.imshow(wsi)
    sc = ax.scatter(d["imagecol"]*scale, d["imagerow"]*scale, c=d[col], cmap=DIV, vmin=vmin, vmax=vmax,
                    s=s, marker="h", linewidths=0)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    return sc

def zoom_pair(fig, gs_a, gs_p, sample, box, d, ec, label):
    """Two crops (actual|pred) of the same region with thin rings coloured by score."""
    for gsx, col, sub in [(gs_a, "az", "actual"), (gs_p, "pz", "predicted")]:
        ax = fig.add_subplot(gsx)
        crop, (x0, y0, x1, y1, s) = _tissue.crop_wsi(sample, box["x0"], box["y0"], box["x1"], box["y1"], out_px=760)
        ax.imshow(crop)
        inb = d[(d["imagecol"].between(x0, x1)) & (d["imagerow"].between(y0, y1))]
        import matplotlib.cm as cm
        norm = plt.Normalize(-2.5, 2.5)
        for _, r in inb.iterrows():
            ax.add_patch(Circle(((r["imagecol"]-x0)*s, (r["imagerow"]-y0)*s), 9, fill=False,
                                ec=cm.RdBu_r(norm(r[col])), lw=1.8))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor(ec); sp.set_linewidth(3)
        ax.set_title(f"{label} · {sub}", fontsize=10.5, fontweight="bold", color=ec)

def fig_sample(sample):
    d = prep(sample)
    wsi, scale = _tissue.render_wsi(sample, max_dim=2000)
    agree = d[(d["concord"] > 0) & (d["az"].abs() > 0.6) & (d["pz"].abs() > 0.6)]
    dis = d[(d["concord"] < 0) & (d["az"].abs() > 0.7) & (d["pz"].abs() > 0.7)]
    abox = dense_box(agree, d); dbox = dense_box(dis, d)
    rho = np.corrcoef(d["az"], d["pz"])[0, 1]

    fig = plt.figure(figsize=(19, 10.5))
    gs = fig.add_gridspec(2, 12, height_ratios=[1.15, 1.0], hspace=0.26, wspace=0.5,
                          left=0.02, right=0.98, top=0.88, bottom=0.10)
    # top: WSI | actual | predicted
    axw = fig.add_subplot(gs[0, 0:4]); axw.imshow(wsi); axw.set_xticks([]); axw.set_yticks([]); axw.set_aspect("equal")
    axw.set_title(f"{short(sample)} — real whole-slide H&E", fontsize=12.5, fontweight="bold")
    for b, ec in [(abox, AGREE_EC), (dbox, DIS_EC)]:
        if b: axw.add_patch(Rectangle((b["x0"]*scale, b["y0"]*scale), (b["x1"]-b["x0"])*scale,
                            (b["y1"]-b["y0"])*scale, fill=False, ec=ec, lw=3))
    axa = fig.add_subplot(gs[0, 4:8]); sc = score_on_wsi(axa, wsi, scale, d, "az")
    axa.set_title(f"{short(sample)} — ACTUAL (transcriptomics)", fontsize=12.5, fontweight="bold")
    fig.colorbar(sc, ax=axa, fraction=0.046, pad=0.02)
    axp = fig.add_subplot(gs[0, 8:12]); sc = score_on_wsi(axp, wsi, scale, d, "pz")
    axp.set_title(f"{short(sample)} — PREDICTED from H&E only", fontsize=12.5, fontweight="bold")
    fig.colorbar(sc, ax=axp, fraction=0.046, pad=0.02)
    # bottom: agree pair | disagree pair
    if abox: zoom_pair(fig, gs[1, 0:3], gs[1, 3:6], sample, abox, d, AGREE_EC, "AGREE")
    if dbox: zoom_pair(fig, gs[1, 6:9], gs[1, 9:12], sample, dbox, d, DIS_EC, "DISAGREE")

    fig.suptitle(f"{short(sample)}:  H&E-only prediction vs the real leaving score  (Spearman-like ρ = {rho:.2f})",
                 fontsize=16, fontweight="bold", y=0.955)
    caption(fig, "Top: the real whole-slide, the ACTUAL transcriptomic leaving score, and the score PREDICTED from H&E alone (red high, blue low). "
                 "Green box = a region where the two AGREE; red box = a region where they DISAGREE. Bottom: those regions zoomed on the real tissue, thin rings coloured by score (left ring set = actual, right = predicted). "
                 "The prediction captures broad high/low zonation (tracking tumour density) but misses finer structure — visible where the rings flip colour between the actual and predicted crops.", y=0.008)
    plt.savefig(os.path.join(OUT, f"figP_{short(sample)}.png"), dpi=125, bbox_inches="tight")
    plt.close(); print(f"FIG P {sample} done  (agree box {'ok' if abox else 'none'}, dis box {'ok' if dbox else 'none'})")

if __name__ == "__main__":
    for s in ["IU_PDA_T4", "IU_PDA_T1"]:
        fig_sample(s)
    print("DONE ->", OUT)
