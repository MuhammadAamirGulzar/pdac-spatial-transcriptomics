"""
Prep for the 'where is the liver' WSI+zoom slides.
For HM11, HM13 and each PT: per-spot hepatocyte/tumour fraction (RCTD) + WSI coords;
finds a LIVER-rich zoom box and a NON-liver (tumour) zoom box per sample via 2D-density
peak; picks the PT sample with the most liver content.

Run: "C:/Users/datai/anaconda3/envs/tcga/python.exe" docs/presentation/_liver_zoom_prep.py
Out: Outputs/stage0_confound/liver_zoom.json + per-sample spot CSVs
"""
import os, json
import numpy as np, pandas as pd, torch
import _tissue

ROOT = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(os.path.dirname(ROOT))
RCTD = os.path.join(PROJ, "dataset", "Cell Embedding Extraction", "RCTD")
QC = pd.read_csv(os.path.join(PROJ, "Outputs/Patient-Sample-Information/spot_qc_mask.csv"))
COORD = pd.read_csv(os.path.join(PROJ, "Outputs/Patient-Sample-Information/spot_spatial_coordinates.csv"))
HEPA_IDX, TUMOR_IDX = 7, 14
ALL = ["IU_PDA_HM11", "IU_PDA_HM13", "IU_PDA_T1", "IU_PDA_T3", "IU_PDA_T4", "IU_PDA_T11"]

def table(sample):
    rows = []
    for _, r in QC[QC["sample"] == sample].iterrows():
        p = os.path.join(RCTD, sample, r["patch_stem"] + ".pt")
        if os.path.exists(p):
            v = torch.load(p, map_location="cpu", weights_only=False).numpy()
            rows.append({"patch_stem": r["patch_stem"], "barcode": r["barcode"],
                         "hepatocyte_frac": float(v[HEPA_IDX]), "tumor_frac": float(v[TUMOR_IDX])})
    d = pd.DataFrame(rows)
    c = COORD[COORD["image"] == sample][["spot_barcode", "imagerow", "imagecol"]]
    return d.merge(c, left_on="barcode", right_on="spot_barcode", how="left").dropna(subset=["imagecol"])

def dense_box(d, sub, span_frac=0.16):
    """Box (full-res coords) centred on the densest cluster of the `sub` spots."""
    if len(sub) < 5: return None
    fw = d["imagecol"].max() - d["imagecol"].min()
    fh = d["imagerow"].max() - d["imagerow"].min()
    # 2D histogram peak of the subset
    nb = 12
    H, xe, ye = np.histogram2d(sub["imagecol"], sub["imagerow"], bins=nb)
    i, j = np.unravel_index(np.argmax(H), H.shape)
    cx = 0.5 * (xe[i] + xe[i+1]); cy = 0.5 * (ye[j] + ye[j+1])
    sx = span_frac * fw; sy = span_frac * fh
    b = dict(x0=float(cx - sx/2), y0=float(cy - sy/2), x1=float(cx + sx/2), y1=float(cy + sy/2))
    inb = sub[(sub["imagecol"].between(b["x0"], b["x1"])) & (sub["imagerow"].between(b["y0"], b["y1"]))]
    b["n"] = int(len(inb))
    return b

out = {}
for s in ALL:
    d = table(s)
    d.to_csv(os.path.join(PROJ, f"Outputs/stage0_confound/spots_{s}.csv"), index=False)
    liver = d[d["hepatocyte_frac"] > 0.4]
    tum = d[(d["tumor_frac"] > 0.5) & (d["hepatocyte_frac"] < 0.05)]
    out[s] = {
        "n": len(d), "hepa_mean": float(d["hepatocyte_frac"].mean()),
        "n_liver_gt0.4": int(len(liver)), "n_liver_gt0.2": int((d["hepatocyte_frac"] > 0.2).sum()),
        "liver_box": dense_box(d, liver, 0.16),
        "nonliver_box": dense_box(d, tum, 0.16),
    }
    print(f"{s}: {len(d)} spots, hepa_mean={d['hepatocyte_frac'].mean():.4f}, "
          f"liver>0.4: {len(liver)}, liver>0.2: {(d['hepatocyte_frac']>0.2).sum()}")

# pick PT with most liver
pts = ["IU_PDA_T1", "IU_PDA_T3", "IU_PDA_T4", "IU_PDA_T11"]
pt_pick = max(pts, key=lambda s: out[s]["n_liver_gt0.2"])
out["PT_most_liver"] = pt_pick
print(f"\nPT sample with most liver content: {pt_pick} "
      f"({out[pt_pick]['n_liver_gt0.2']} spots with hepatocyte>0.2)")

json.dump(out, open(os.path.join(PROJ, "Outputs/stage0_confound/liver_zoom.json"), "w"), indent=2)
print("saved -> Outputs/stage0_confound/liver_zoom.json")
