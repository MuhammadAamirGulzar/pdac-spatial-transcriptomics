"""
STAGE 1B - Patient-11 matched anchor: PT11 -> HM11 resemblance (laptop, no GPU).

SECONDARY corroboration axis from REVIEW_PLAN.md, REVISED by Stage 0:
the cohort-wide mu_HM - mu_PT direction is a liver confound and does NOT generalize
across the 2 HM patients, so mu_HM is demoted to a PATIENT-11-ONLY anchor.

Question (Stage-4 corroboration): within patient 11, do PT11 tumour spots that score
high on the Stage-1A *leaving program* also look transcriptomically more like the
established metastasis HM11?  Agreement of two INDEPENDENTLY-derived axes (1A = EMT
gene signature; 1B = transcriptome-wide PT11->HM11 direction) is the headline.

Design choices (so this is honest, not circular):
- Patient 11 ONLY: PT11 = IU_PDA_T11, HM11 = IU_PDA_HM11.  No cross-patient generalization.
- TUMOUR-DOMINATED spots only (RCTD Tumor frac >= 0.5) -> strips bulk hepatocyte/liver
  admixture (Stage 0: confound corr 0.40 -> 0.07 within tumour-only).
- Representation = NON-batch-corrected log-norm gene expression (NOT scVI: scVI batch-
  corrects PT11 vs HM11 and would erase the very axis we want).  Transcriptome-wide HVGs,
  deliberately NOT the 1A EMT gene set, so 1A and 1B are independent.
- Direction Delta = z-centroid(HM11 tumour) - z-centroid(PT11 tumour); resemblance_i = z_i . Delta_hat.
- Confound audit: corr(resemblance, hepatocyte frac); liver-gene-excluded variant; top drivers.

Run:
    "C:/Users/datai/anaconda3/envs/tcga/python.exe" stage1b_pt11_hm11_anchor.py

Outputs -> Outputs/stage1b_pt11_anchor/
    pt11_hm11_resemblance.csv, metrics.json, summary.txt,
    heatmap_pt11_resemblance.png, leaving_vs_resemblance.png,
    pt_vs_hm_separation.png, driver_genes.png
"""

import os, json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------- paths
from _cohort import (ROOT, RCTD_DIR, HEPATOCYTE_IDX, TUMOR_IDX, OUT_TAG,
                     out_dir, banner, load_spot_embeddings)

COUNTS_DIR = os.path.join(ROOT, "dataset", "ST", "scVI_counts")
QC_CSV     = os.path.join(ROOT, "Outputs", "Patient-Sample-Information", "spot_qc_mask.csv")
S1A_CSV    = os.path.join(ROOT, "Outputs", "stage1a_leaving_program" + OUT_TAG,
                          "leaving_program_scores.csv")
OUT_DIR    = out_dir("stage1b_pt11_anchor")
banner("STAGE 1B - PT11/HM11 anchor")

PT11, HM11 = "IU_PDA_T11", "IU_PDA_HM11"
TUMOR_THRESH = 0.50
N_HVG = 2000
RNG = np.random.default_rng(0)

# liver / hepatocyte genes -> excluded in the confound-controlled variant
LIVER_GENES = {"ALB", "APOA1", "APOA2", "APOB", "APOC1", "APOC3", "APOH", "APOM",
               "HP", "TTR", "TF", "FGA", "FGB", "FGG", "SERPINA1", "ALDOB", "ORM1",
               "ORM2", "AHSG", "CYP2E1", "CYP3A4", "ASGR1", "ASGR2", "HPX", "GC",
               "RBP4", "APOC2", "FGL1", "CPS1", "ARG1", "MT1H"}

# ----------------------------------------------------------------------------- load helpers
def load_lognorm(sample):
    """Return (lognorm spots x genes, genes array, barcodes array)."""
    mat = pd.read_csv(os.path.join(COUNTS_DIR, sample + ".csv"), index_col=0)
    mat = mat[~mat.index.duplicated(keep="first")]
    genes = mat.index.to_numpy()
    barcodes = mat.columns.to_numpy()
    counts = mat.to_numpy(dtype=np.float64).T
    del mat
    lib = counts.sum(1, keepdims=True); lib[lib == 0] = 1.0
    return np.log1p(counts / lib * 1e4), genes, barcodes

def rctd_fracs(sample, barcodes, bc2stem):
    """Return (tumor_frac, hepatocyte_frac, patch_stem, row, col) aligned to barcodes.

    Reads the sample's cell embeddings in one consolidated load rather than one
    torch.load per spot (see _cohort.load_spot_embeddings)."""
    stems, rows, cols = [], [], []
    for b in barcodes:
        st = bc2stem.get((sample, b))
        stems.append(st)
        if st is None:
            rows.append(-1); cols.append(-1)
        else:
            p = st.split("_"); rows.append(int(p[-2])); cols.append(int(p[-1]))

    V, has = load_spot_embeddings(
        RCTD_DIR, sample, [st if st is not None else "" for st in stems])
    has &= np.array([st is not None for st in stems])
    # Widen to float64 to match the `float(v[i])` this replaced -- under NumPy 2's
    # NEP-50 rules the weak np.nan scalar does not promote a float32 array.
    tf = np.where(has, V[:, TUMOR_IDX].astype(np.float64), np.nan)
    hf = np.where(has, V[:, HEPATOCYTE_IDX].astype(np.float64), np.nan)
    return (tf, hf, np.array(stems, dtype=object), np.array(rows), np.array(cols))

def hex_morans_I(score, rc, n_perm=999):
    pos = {(int(r), int(c)): i for i, (r, c) in enumerate(rc)}
    offs = [(0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    rows, cols = [], []
    for (r, c), i in pos.items():
        for dr, dc in offs:
            j = pos.get((r + dr, c + dc))
            if j is not None: rows.append(i); cols.append(j)
    n = len(score)
    if not rows: return float("nan"), float("nan")
    rows = np.array(rows); cols = np.array(cols)
    z = score - score.mean(); denom = (z ** 2).sum(); W = len(rows)
    I_obs = (n / W) * (z[rows] * z[cols]).sum() / denom
    perm = np.array([(n / W) * ((zp := RNG.permutation(z))[rows] * zp[cols]).sum() / (zp ** 2).sum()
                     for _ in range(n_perm)])
    return float(I_obs), float((1 + (perm >= I_obs).sum()) / (n_perm + 1))

# ----------------------------------------------------------------------------- load
qc = pd.read_csv(QC_CSV)
qc = qc[qc["sample"].isin([PT11, HM11])]
bc2stem = {(s, b): st for s, b, st in zip(qc["sample"], qc["barcode"], qc["patch_stem"])}

print("loading PT11 ..."); LP, gP, bP = load_lognorm(PT11)
print("loading HM11 ..."); LH, gH, bH = load_lognorm(HM11)

# shared genes, aligned
shared = np.intersect1d(gP, gH)
iP = {g: k for k, g in enumerate(gP)}; iH = {g: k for k, g in enumerate(gH)}
LP = LP[:, [iP[g] for g in shared]]; LH = LH[:, [iH[g] for g in shared]]
genes = shared
print(f"shared genes: {len(genes)}")

tfP, hfP, stemP, rowP, colP = rctd_fracs(PT11, bP, bc2stem)
tfH, hfH, stemH, rowH, colH = rctd_fracs(HM11, bH, bc2stem)

# tumour-dominated restriction
mP = tfP >= TUMOR_THRESH
mH = tfH >= TUMOR_THRESH
print(f"tumour-dominated spots: PT11 {mP.sum()}/{len(mP)} | HM11 {mH.sum()}/{len(mH)}")

# ----------------------------------------------------------------------------- resemblance axis
def build_axis(exclude_liver):
    keep = np.array([g not in LIVER_GENES for g in genes]) if exclude_liver else np.ones(len(genes), bool)
    Xp, Xh = LP[mP][:, keep], LH[mH][:, keep]
    # HVG on combined tumour-dominated spots
    comb = np.vstack([Xp, Xh])
    var = comb.var(0)
    hvg = np.argsort(var)[::-1][:min(N_HVG, keep.sum())]
    mu, sd = comb[:, hvg].mean(0), comb[:, hvg].std(0) + 1e-9
    zp = (Xp[:, hvg] - mu) / sd
    zh = (Xh[:, hvg] - mu) / sd
    delta = zh.mean(0) - zp.mean(0)
    delta /= (np.linalg.norm(delta) + 1e-9)
    res_pt = zp @ delta            # PT11 tumour spots' HM11-resemblance
    res_hm = zh @ delta            # HM11 tumour spots (separation sanity)
    gnames = genes[keep][hvg]
    return res_pt, res_hm, delta, gnames

res_pt, res_hm, delta, hvg_names = build_axis(exclude_liver=False)
res_pt_nl, res_hm_nl, _, _       = build_axis(exclude_liver=True)

# ----------------------------------------------------------------------------- merge Stage-1A scores (PT11)
s1a = pd.read_csv(S1A_CSV)
s1a = s1a[s1a["sample"] == PT11][["patch_stem", "leaving_score", "leaving_score_resid"]]

pt = pd.DataFrame({
    "patch_stem": stemP[mP], "sample": PT11,
    "row": rowP[mP], "col": colP[mP],
    "tumor_frac": tfP[mP], "hepatocyte_frac": hfP[mP],
    "hm11_resemblance": res_pt, "hm11_resemblance_noliver": res_pt_nl,
})
pt = pt.merge(s1a, on="patch_stem", how="left")
pt.to_csv(os.path.join(OUT_DIR, "pt11_hm11_resemblance.csv"), index=False)

# ----------------------------------------------------------------------------- metrics
def safe_corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 10 else float("nan")

I_res, p_res = hex_morans_I(pt["hm11_resemblance"].to_numpy(),
                            pt[["row", "col"]].to_numpy()) if (pt["row"] >= 0).all() else (np.nan, np.nan)

# top driver genes (most positive = HM11-like, most negative = PT11-like)
order = np.argsort(delta)
top_hm = [(hvg_names[i], float(delta[i])) for i in order[::-1][:20]]
top_pt = [(hvg_names[i], float(delta[i])) for i in order[:20]]
n_liver_in_top = sum(1 for g, _ in top_hm if g in LIVER_GENES)

metrics = {
    "n_pt11_tumour": int(mP.sum()), "n_hm11_tumour": int(mH.sum()),
    "tumor_thresh": TUMOR_THRESH, "n_hvg": int(len(hvg_names)),
    "HEADLINE_corr_leaving_vs_resemblance": safe_corr(pt["leaving_score"], pt["hm11_resemblance"]),
    "HEADLINE_corr_leavingResid_vs_resemblance": safe_corr(pt["leaving_score_resid"], pt["hm11_resemblance"]),
    "corr_leaving_vs_resemblance_noliver": safe_corr(pt["leaving_score"], pt["hm11_resemblance_noliver"]),
    "corr_leavingResid_vs_resemblance_noliver": safe_corr(pt["leaving_score_resid"], pt["hm11_resemblance_noliver"]),
    "confound_corr_resemblance_vs_hepatocyte": safe_corr(pt["hm11_resemblance"], pt["hepatocyte_frac"]),
    "confound_corr_resemblance_vs_tumorfrac": safe_corr(pt["hm11_resemblance"], pt["tumor_frac"]),
    "corr_resemblance_full_vs_noliver": safe_corr(pt["hm11_resemblance"], pt["hm11_resemblance_noliver"]),
    "morans_I_resemblance": I_res, "morans_p_resemblance": p_res,
    "pt11_resemblance_mean": float(res_pt.mean()), "hm11_resemblance_mean": float(res_hm.mean()),
    "n_liver_genes_in_top20_HM": n_liver_in_top,
    "top_HM_like_genes": top_hm, "top_PT_like_genes": top_pt,
}
with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# ----------------------------------------------------------------------------- plots
# spatial heatmap of resemblance on PT11
fig, ax = plt.subplots(figsize=(6, 6))
sc = ax.scatter(pt["col"], -pt["row"], c=pt["hm11_resemblance"], cmap="viridis", s=16)
ax.set_title("PT11 tumour spots: HM11 resemblance"); ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(sc, ax=ax, fraction=0.046)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "heatmap_pt11_resemblance.png"), dpi=130); plt.close()

# headline: 1A leaving vs 1B resemblance
fig, axs = plt.subplots(1, 2, figsize=(11, 5))
for ax, col, lab in [(axs[0], "leaving_score", "1A leaving score (raw)"),
                     (axs[1], "leaving_score_resid", "1A leaving score (residualized)")]:
    r = safe_corr(pt[col], pt["hm11_resemblance"])
    ax.scatter(pt[col], pt["hm11_resemblance"], s=8, alpha=0.4, color="#8338ec")
    ax.set_xlabel(lab); ax.set_ylabel("HM11 resemblance (1B)")
    ax.set_title(f"two independent axes  r={r:.3f}")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "leaving_vs_resemblance.png"), dpi=130); plt.close()

# PT vs HM separation sanity (direction is defined to separate; just a check)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(res_pt, bins=50, alpha=0.6, density=True, label="PT11 tumour", color="#5b9bd5")
ax.hist(res_hm, bins=50, alpha=0.6, density=True, label="HM11 tumour", color="#d96459")
ax.set_xlabel("projection onto PT11->HM11 axis"); ax.legend()
ax.set_title("Patient-11 anchor axis (tumour-dominated spots)")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "pt_vs_hm_separation.png"), dpi=130); plt.close()

# driver genes
fig, axs = plt.subplots(1, 2, figsize=(11, 6))
for ax, data, title, color in [(axs[0], top_hm, "HM11-like (high resemblance)", "#d96459"),
                               (axs[1], top_pt[::-1], "PT11-like (low resemblance)", "#5b9bd5")]:
    names = [g for g, _ in data]; vals = [v for _, v in data]
    ax.barh(range(len(names)), vals, color=color)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.set_title(title); ax.axvline(0, color="k", lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "driver_genes.png"), dpi=130); plt.close()

# ----------------------------------------------------------------------------- verdict
M = metrics
v = []
v.append("STAGE 1B - PATIENT-11 MATCHED ANCHOR (PT11 -> HM11) - VERDICT\n" + "=" * 60)
v.append(f"Tumour-dominated spots: PT11 {M['n_pt11_tumour']} | HM11 {M['n_hm11_tumour']}  "
         f"(Tumor frac >= {TUMOR_THRESH}), {M['n_hvg']} HVGs, scVI NOT used (batch-correction would erase axis).")
v.append("")
v.append("HEADLINE - agreement of two INDEPENDENT axes (1A EMT signature vs 1B transcriptome anchor):")
v.append(f"   corr(1A leaving raw,   1B resemblance) = {M['HEADLINE_corr_leaving_vs_resemblance']:.3f}")
v.append(f"   corr(1A leaving resid, 1B resemblance) = {M['HEADLINE_corr_leavingResid_vs_resemblance']:.3f}")
v.append(f"   liver-gene-excluded: raw {M['corr_leaving_vs_resemblance_noliver']:.3f} | "
         f"resid {M['corr_leavingResid_vs_resemblance_noliver']:.3f}")
v.append("   -> POSITIVE = high intra-PT11 leaving-program spots also look more like HM11. Corroboration.")
v.append("")
v.append("CONFOUND AUDIT:")
v.append(f"   corr(resemblance, hepatocyte frac) = {M['confound_corr_resemblance_vs_hepatocyte']:.3f}  (want ~0)")
v.append(f"   corr(resemblance, tumor frac)      = {M['confound_corr_resemblance_vs_tumorfrac']:.3f}")
v.append(f"   liver genes in top-20 HM-like drivers = {M['n_liver_genes_in_top20_HM']}/20")
v.append(f"   corr(full axis, liver-excluded axis) = {M['corr_resemblance_full_vs_noliver']:.3f}")
v.append(f"   Moran's I (resemblance on PT11) = {M['morans_I_resemblance']:.3f} p={M['morans_p_resemblance']:.3f}")
v.append("")
v.append("Top HM11-like drivers: " + ", ".join(g for g, _ in M["top_HM_like_genes"][:12]))
v.append("Top PT11-like drivers: " + ", ".join(g for g, _ in M["top_PT_like_genes"][:12]))
v.append("")
v.append("USE: Stage-4 qualitative corroboration ONLY (NOT a Phase B training direction).")
v.append("Saved: Outputs/stage1b_pt11_anchor/pt11_hm11_resemblance.csv")
txt = "\n".join(v)
open(os.path.join(OUT_DIR, "summary.txt"), "w").write(txt)
print("\n" + txt + f"\n\nSaved -> {OUT_DIR}")
