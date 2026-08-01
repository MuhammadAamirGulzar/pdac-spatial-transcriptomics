"""
STAGE 0 - Confound diagnostic (gene space).  Laptop, no GPU.

Question this answers: is the HM-vs-PT signal we plan to exploit (mu_HM - mu_PT)
actually metastatic biology, or is it just "tumour-in-liver vs tumour-in-pancreas"
(hepatocyte content / tissue of residence)?  If the latter, the Phase B direction
vector is a confound and must be restricted to tumour-only spots / residualised.

Run:
    "C:/Users/datai/anaconda3/envs/tcga/python.exe" stage0_confound_diagnostic.py

Outputs -> Outputs/stage0_confound/
    metrics.json, hepatocyte_by_sample.png, delta_projection.png,
    proj_vs_hepatocyte.png, umap_scvi.png (if umap-learn installed), summary.txt
"""

import os, json, glob
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------- paths
from _cohort import (ROOT, RCTD_DIR, HEPATOCYTE_IDX, TUMOR_IDX,
                     patients_of, out_dir, banner, boxplot,
                     load_spot_embeddings)

SCVI_DIR = os.path.join(ROOT, "dataset", "Gene Embedding Extraction", "scvi_latent_pt_embeddings")
QC_CSV   = os.path.join(ROOT, "Outputs", "Patient-Sample-Information", "spot_qc_mask.csv")
OUT_DIR  = out_dir("stage0_confound")
banner("STAGE 0 - confound diagnostic")

TUMOR_THRESH   = 0.50  # "tumour-dominated" spot = Tumor Epithelial fraction >= this

# ----------------------------------------------------------------------------- load
qc = pd.read_csv(QC_CSV)
qc["group"] = np.where(qc["sample"].str.contains("HM"), "HM", "PT")

# Per sample rather than per spot: one consolidated read of each modality instead
# of two .pt round-trips for every one of the ~20k (soon ~91k) spots.  A spot is
# kept only when BOTH modalities have a tensor, exactly as the per-file
# os.path.exists() pair did.
scvi_blocks, rctd_blocks, idx_blocks = [], [], []
for sample, grp in qc.groupby("sample", sort=False):
    stems = grp["patch_stem"].tolist()
    Xg, ok_g = load_spot_embeddings(SCVI_DIR, sample, stems)
    Xc, ok_c = load_spot_embeddings(RCTD_DIR, sample, stems)
    keep = ok_g & ok_c          # both modalities present, as the old exists() pair required
    scvi_blocks.append(Xg[keep])
    rctd_blocks.append(Xc[keep])
    idx_blocks.append(np.asarray(grp.index)[keep])
    print(f"  {sample:14s} {int(keep.sum()):5d}/{len(stems):5d} spots with both modalities")

# Restore qc's original row order, which the per-sample grouping broke.
kept = np.concatenate(idx_blocks)
order = np.argsort(kept, kind="stable")
kept_idx = kept[order]

df = qc.loc[kept_idx].reset_index(drop=True).copy()
X_scvi = np.vstack(scvi_blocks).astype(np.float64)[order]    # (N,50)
X_rctd = np.vstack(rctd_blocks).astype(np.float64)[order]    # (N,15)
hepato = X_rctd[:, HEPATOCYTE_IDX]
tumor  = X_rctd[:, TUMOR_IDX]
df["hepatocyte_frac"] = hepato
df["tumor_frac"]      = tumor
y = (df["group"].values == "HM").astype(int)         # 1 = HM
samples = df["sample"].values
df["patient"] = patients_of(samples)
print(f"Loaded {len(df)} spots | HM={int(y.sum())} PT={int((1-y).sum())}")
print(f"Folds = {df['patient'].nunique()} patients (leave-one-patient-out): "
      f"{ {p: sorted(g['sample'].unique().tolist()) for p, g in df.groupby('patient')} }")

results = {"n_spots": int(len(df)),
           "n_HM": int(y.sum()), "n_PT": int((1 - y).sum()),
           "tumor_thresh": TUMOR_THRESH,
           "rctd_variant": os.path.basename(RCTD_DIR),
           "cv": "leave-one-patient-out",
           "patients": {p: sorted(g["sample"].unique().tolist())
                        for p, g in df.groupby("patient")}}

# ----------------------------------------------------------------------------- helpers
def loso_accuracy(X, y, samples):
    """Leave-one-PATIENT-out: train on 4 patients, predict the held-out patient.

    Sample-level holdout leaks -- IU_PDA_T11 and IU_PDA_HM11 are the matched
    primary/metastasis pair from patient PT_11, so holding out one leaves the
    other in training and the 'cross-patient' claim is void.

    The PT_11 fold is two-class (its own PT and HM spots); the rest are
    single-class.  We therefore pool out-of-fold predictions across all folds
    and take balanced accuracy = mean(TPR, TNR) on the pooled OOF vector, which
    is well defined regardless of per-fold class composition."""
    pat = patients_of(samples)
    per = {}
    oof = np.full(len(y), -1, dtype=int)
    for p in np.unique(pat):
        te = pat == p
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(sc.transform(X[tr]), y[tr])
        pred = clf.predict(sc.transform(X[te]))
        oof[te] = pred
        per[str(p)] = {"acc": float((pred == y[te]).mean()),
                       "n": int(te.sum()),
                       "samples": sorted(set(samples[te].tolist()))}
    ok = oof >= 0
    if ok.sum() and len(np.unique(y[ok])) == 2:
        tpr = float((oof[ok][y[ok] == 1] == 1).mean())
        tnr = float((oof[ok][y[ok] == 0] == 0).mean())
        bal = float(np.mean([tpr, tnr]))
    else:
        tpr = tnr = bal = float("nan")
    return {"per_patient": per, "tpr_HM": tpr, "tnr_PT": tnr}, bal

# ---- (A) Can scVI separate HM/PT across held-out patients?  all spots vs tumour-only
for tag, sub in [("scvi_all", np.ones(len(df), bool)),
                 ("scvi_tumoronly", tumor >= TUMOR_THRESH)]:
    if sub.sum() < 50 or len(np.unique(y[sub])) < 2:
        results[tag] = {"note": "insufficient spots/classes", "n": int(sub.sum())}
        continue
    per, bal = loso_accuracy(X_scvi[sub], y[sub], samples[sub])
    results[tag] = {"n": int(sub.sum()), "loso_patient": per, "balanced_acc": bal}
    print(f"[{tag}] n={sub.sum()} balanced LOPO acc={bal:.3f} "
          f"(TPR_HM={per['tpr_HM']:.3f} TNR_PT={per['tnr_PT']:.3f})")

# ---- (B) RCTD composition alone, and RCTD WITHOUT hepatocyte (isolates liver driver)
keep_no_hep = [i for i in range(X_rctd.shape[1]) if i != HEPATOCYTE_IDX]
for tag, X in [("rctd_all", X_rctd), ("rctd_no_hepatocyte", X_rctd[:, keep_no_hep])]:
    per, bal = loso_accuracy(X, y, samples)
    results[tag] = {"loso_patient": per, "balanced_acc": bal}
    print(f"[{tag}] balanced LOPO acc={bal:.3f} "
          f"(TPR_HM={per['tpr_HM']:.3f} TNR_PT={per['tnr_PT']:.3f})")

# ---- (C) How much of mu_HM - mu_PT (scVI) is just hepatocyte content?
delta = X_scvi[y == 1].mean(0) - X_scvi[y == 0].mean(0)
delta /= (np.linalg.norm(delta) + 1e-9)
proj = X_scvi @ delta
r_all = float(np.corrcoef(proj, hepato)[0, 1])
to = tumor >= TUMOR_THRESH
r_tumor = float(np.corrcoef(proj[to], hepato[to])[0, 1]) if to.sum() > 10 else float("nan")
results["delta_vs_hepatocyte"] = {
    "pearson_r_all_spots": r_all,
    "pearson_r_tumoronly": r_tumor,
    "hepato_frac_PT_mean": float(hepato[y == 0].mean()),
    "hepato_frac_HM_mean": float(hepato[y == 1].mean()),
    "tumor_frac_PT_mean":  float(tumor[y == 0].mean()),
    "tumor_frac_HM_mean":  float(tumor[y == 1].mean()),
}
print(f"corr(delta-projection, hepatocyte_frac): all={r_all:.3f} tumour-only={r_tumor:.3f}")

# ----------------------------------------------------------------------------- plots
# hepatocyte fraction by sample
fig, ax = plt.subplots(figsize=(8, 4))
order = sorted(df["sample"].unique(), key=lambda s: ("HM" not in s, s))
data = [df.loc[df["sample"] == s, "hepatocyte_frac"].values for s in order]
bp = boxplot(ax, data, order, patch_artist=True)
for patch, s in zip(bp["boxes"], order):
    patch.set_facecolor("#d96459" if "HM" in s else "#5b9bd5")
ax.set_ylabel("RCTD hepatocyte fraction"); ax.set_title("Hepatocyte content by sample (red=HM, blue=PT)")
plt.xticks(rotation=30, ha="right"); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "hepatocyte_by_sample.png"), dpi=130); plt.close()

# delta projection distribution
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(proj[y == 0], bins=60, alpha=0.6, label="PT", color="#5b9bd5", density=True)
ax.hist(proj[y == 1], bins=60, alpha=0.6, label="HM", color="#d96459", density=True)
ax.set_xlabel("projection onto (mu_HM - mu_PT)"); ax.legend()
ax.set_title("scVI metastasis-direction projection, PT vs HM")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "delta_projection.png"), dpi=130); plt.close()

# proj vs hepatocyte scatter
fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter(hepato, proj, c=tumor, s=4, cmap="viridis")
ax.set_xlabel("hepatocyte fraction"); ax.set_ylabel("delta projection")
ax.set_title(f"Confound check  (r_all={r_all:.2f})"); plt.colorbar(sc, label="tumor frac")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "proj_vs_hepatocyte.png"), dpi=130); plt.close()

# optional UMAP
try:
    import umap
    emb = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit_transform(
        StandardScaler().fit_transform(X_scvi))
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (title, col, cmap) in zip(axs.ravel(), [
            ("sample", pd.factorize(df["sample"])[0], "tab10"),
            ("group HM=1", y, "coolwarm"),
            ("tumor frac", tumor, "viridis"),
            ("hepatocyte frac", hepato, "magma")]):
        p = ax.scatter(emb[:, 0], emb[:, 1], c=col, s=3, cmap=cmap); ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(p, ax=ax, fraction=0.046)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "umap_scvi.png"), dpi=130); plt.close()
    results["umap"] = "saved"
except Exception as e:
    results["umap"] = f"skipped ({e})"

# ----------------------------------------------------------------------------- save + verdict
with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
    json.dump(results, f, indent=2)

verdict = []
verdict.append("STAGE 0 CONFOUND DIAGNOSTIC - VERDICT\n" + "=" * 44)
verdict.append(f"cell modality : {os.path.basename(RCTD_DIR)}   CV : leave-one-patient-out "
               f"({df['patient'].nunique()} patients, {df['sample'].nunique()} samples)")
verdict.append(f"HM hepatocyte frac mean = {hepato[y==1].mean():.3f}  vs  PT = {hepato[y==0].mean():.3f}")
verdict.append(f"HM tumour frac mean     = {tumor[y==1].mean():.3f}  vs  PT = {tumor[y==0].mean():.3f}")
verdict.append(f"corr(metastasis-direction, hepatocyte) : all spots = {r_all:.3f} | tumour-only = {r_tumor:.3f}")
sa = results.get("scvi_all", {}).get("balanced_acc")
st = results.get("scvi_tumoronly", {}).get("balanced_acc")
verdict.append(f"scVI HM/PT LOPO balanced acc : all = {sa} | tumour-only = {st}")
verdict.append(f"RCTD LOPO acc : all = {results['rctd_all']['balanced_acc']:.3f} | "
               f"no-hepatocyte = {results['rctd_no_hepatocyte']['balanced_acc']:.3f}")
verdict.append("")
verdict.append("READ: if r_all is high AND drops sharply tumour-only -> mu_HM axis is a")
verdict.append("liver-content confound; Phase B must restrict to tumour-only / residualise.")
verdict.append("If RCTD-no-hepatocyte acc collapses vs RCTD-all -> hepatocyte IS the driver.")
txt = "\n".join(verdict)
open(os.path.join(OUT_DIR, "summary.txt"), "w").write(txt)
print("\n" + txt + f"\n\nSaved -> {OUT_DIR}")
