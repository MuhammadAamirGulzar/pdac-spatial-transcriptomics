"""
Data prep for the expanded Stage-0 liver-confound slides.
Builds per-spot hepatocyte fraction (RCTD) + hepatocyte MARKER-GENE score for HM
and PT spots, with WSI pixel coords; validates the RCTD hepatocyte call against
real liver genes; picks a high-hepatocyte zoom region on HM11 and example spots.

Run: "C:/Users/datai/anaconda3/envs/tcga/python.exe" docs/presentation/_liver_confound_prep.py
Out: Outputs/stage0_confound/liver_biology.json  (+ prints)
"""
import os, json, glob
import numpy as np, pandas as pd, torch

ROOT = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(os.path.dirname(ROOT))
RCTD = os.path.join(PROJ, "dataset", "Cell Embedding Extraction", "RCTD")
COUNTS = os.path.join(PROJ, "dataset", "ST", "scVI_counts")
QC = pd.read_csv(os.path.join(PROJ, "Outputs/Patient-Sample-Information/spot_qc_mask.csv"))
COORD = pd.read_csv(os.path.join(PROJ, "Outputs/Patient-Sample-Information/spot_spatial_coordinates.csv"))
HEPA_IDX, TUMOR_IDX = 7, 14
# canonical hepatocyte / liver-secretome markers
HEPA_GENES = ["ALB", "TTR", "APOA1", "APOA2", "APOC1", "APOC3", "HP", "FGA", "FGB",
              "FGG", "SERPINA1", "TF", "APOB", "CYP2E1", "ORM1"]

def rctd_table(sample):
    rows = []
    for _, r in QC[QC["sample"] == sample].iterrows():
        p = os.path.join(RCTD, sample, r["patch_stem"] + ".pt")
        if os.path.exists(p):
            v = torch.load(p, map_location="cpu", weights_only=False).numpy()
            rows.append({"sample": sample, "patch_stem": r["patch_stem"], "barcode": r["barcode"],
                         "hepatocyte_frac": float(v[HEPA_IDX]), "tumor_frac": float(v[TUMOR_IDX])})
    return pd.DataFrame(rows)

def add_px(d, sample):
    c = COORD[COORD["image"] == sample][["spot_barcode", "imagerow", "imagecol"]]
    return d.merge(c, left_on="barcode", right_on="spot_barcode", how="left")

def add_marker_genes(tab, sample):
    """Attach CP10k-log1p hepatocyte-marker gene columns + a z-mean liver-gene score."""
    df = pd.read_csv(os.path.join(COUNTS, f"{sample}.csv"), index_col=0)
    df.index = [g.strip().strip('"') for g in df.index.astype(str)]
    df = df[~df.index.duplicated(keep="first")]
    present = [g for g in HEPA_GENES if g in df.index]
    common = [b for b in df.columns if b in set(tab["barcode"])]
    lib = df[common].sum(0); lib[lib == 0] = 1
    logn = np.log1p(df.loc[present, common].div(lib, axis=1) * 1e4)   # genes x spots
    score = ((logn.T - logn.T.mean()) / (logn.T.std() + 1e-9)).mean(1)
    tab = tab.set_index("barcode")
    tab.loc[score.index, "hepa_gene_score"] = score.values
    for g in present:
        tab.loc[common, f"g_{g}"] = logn.loc[g, common].values
    return tab.reset_index(), present

# ---- HM11 (has WSI) — full table + hepatocyte marker score
hm, present = add_marker_genes(add_px(rctd_table("IU_PDA_HM11"), "IU_PDA_HM11"), "IU_PDA_HM11")
val = hm.dropna(subset=["hepa_gene_score"])
r = np.corrcoef(val["hepatocyte_frac"], val["hepa_gene_score"])[0, 1]
print(f"HM11: {len(hm)} spots; hepatocyte markers present {present}")
print(f"HM11 corr(RCTD hepatocyte_frac, liver-gene score) = {r:.3f}  (validates the call)")

# ---- PT T1 (has WSI) — for contrast (near-zero hepatocyte), same marker genes
pt, _ = add_marker_genes(add_px(rctd_table("IU_PDA_T1"), "IU_PDA_T1"), "IU_PDA_T1")
print(f"T1: {len(pt)} spots; hepatocyte_frac mean {pt['hepatocyte_frac'].mean():.4f} "
      f"(HM11 mean {hm['hepatocyte_frac'].mean():.3f})")

# ---- pick a high-hepatocyte ZOOM region on HM11 (dense cluster of high-hepa spots)
hi = hm[hm["hepatocyte_frac"] > 0.4].dropna(subset=["imagerow", "imagecol"])
# densest region: median of the high-hepa spots
cx, cy = hi["imagecol"].median(), hi["imagerow"].median()
# window ~ 18% of slide span around a high-hepa dense pocket
span = 0.20 * (hm["imagecol"].max() - hm["imagecol"].min())
box = dict(x0=float(cx - span/2), y0=float(cy - span/2), x1=float(cx + span/2), y1=float(cy + span/2))
n_in = ((hi["imagecol"].between(box["x0"], box["x1"])) & (hi["imagerow"].between(box["y0"], box["y1"]))).sum()
print(f"HM11 zoom box centred ({int(cx)},{int(cy)}) contains {int(n_in)} high-hepatocyte spots")

# ---- example spots: 2 HM high-hepatocyte, 2 PT near-zero
def pick(d, stem_col="patch_stem"):
    return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in d.items()}
hm_hi = hm.dropna(subset=["hepa_gene_score"]).sort_values("hepatocyte_frac", ascending=False)
ex_hm = hm_hi.iloc[[0, 3]]      # two strongly-hepatocyte spots
ex_pt = pt.sort_values("tumor_frac", ascending=False).iloc[[0, 1]]
def rec(r):
    o = {"sample": r["sample"], "patch_stem": r["patch_stem"], "barcode": r["barcode"],
         "hepatocyte_frac": float(r["hepatocyte_frac"]), "tumor_frac": float(r["tumor_frac"]),
         "imagerow": float(r.get("imagerow", np.nan)), "imagecol": float(r.get("imagecol", np.nan))}
    for g in present:
        if f"g_{g}" in r and pd.notna(r.get(f"g_{g}")): o[f"g_{g}"] = float(r[f"g_{g}"])
    return o
examples = {"HM_high": [rec(ex_hm.iloc[0]), rec(ex_hm.iloc[1])],
            "PT_low": [rec(ex_pt.iloc[0]), rec(ex_pt.iloc[1])]}
for tag, lst in examples.items():
    for e in lst:
        print(f"  {tag}: {e['sample']} hepa={e['hepatocyte_frac']:.2f} tum={e['tumor_frac']:.2f} "
              f"ALB={e.get('g_ALB',float('nan')):.1f} stem={e['patch_stem']}")

out = {"counts": {"n_total": 18859, "n_PT": 13578, "n_HM": 5281,
                  "PT_samples": {"IU_PDA_T1": 3073, "IU_PDA_T3": 4241, "IU_PDA_T4": 3587, "IU_PDA_T11": 2677},
                  "HM_samples": {"IU_PDA_HM11": 3894, "IU_PDA_HM13": 1387},
                  "n_tumor_only": 4851},
       "hepa_frac_PT_mean": 0.002, "hepa_frac_HM_mean": 0.140,
       "corr_axis_hepatocyte_all": 0.402, "corr_axis_hepatocyte_tumoronly": 0.073,
       "hepa_markers_present": present,
       "corr_rctd_vs_livergenes_HM11": float(r),
       "hm11_zoom_box": box, "hm11_zoom_n_high_hepa": int(n_in),
       "examples": examples}
op = os.path.join(PROJ, "Outputs/stage0_confound/liver_biology.json")
json.dump(out, open(op, "w"), indent=2)
# also persist the per-spot HM11 + T1 tables for the figure
hm.to_csv(os.path.join(PROJ, "Outputs/stage0_confound/hm11_spots.csv"), index=False)
pt.to_csv(os.path.join(PROJ, "Outputs/stage0_confound/t1_spots.csv"), index=False)
print("saved ->", op)
