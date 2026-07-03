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
ROOT     = os.path.dirname(os.path.abspath(__file__))
SCVI_DIR = os.path.join(ROOT, "dataset", "Gene Embedding Extraction", "scvi_latent_pt_embeddings")
RCTD_DIR = os.path.join(ROOT, "dataset", "Cell Embedding Extraction", "RCTD")
QC_CSV   = os.path.join(ROOT, "Outputs", "Patient-Sample-Information", "spot_qc_mask.csv")
OUT_DIR  = os.path.join(ROOT, "Outputs", "stage0_confound")
os.makedirs(OUT_DIR, exist_ok=True)

HEPATOCYTE_IDX = 7    # RCTD alphabetical order
TUMOR_IDX      = 14
TUMOR_THRESH   = 0.50  # "tumour-dominated" spot = Tumor Epithelial fraction >= this

# ----------------------------------------------------------------------------- load
qc = pd.read_csv(QC_CSV)
qc["group"] = np.where(qc["sample"].str.contains("HM"), "HM", "PT")

scvi, rctd, kept_idx = [], [], []
for i, r in qc.iterrows():
    gp = os.path.join(SCVI_DIR, r["sample"], r["patch_stem"] + ".pt")
    cp = os.path.join(RCTD_DIR, r["sample"], r["patch_stem"] + ".pt")
    if not (os.path.exists(gp) and os.path.exists(cp)):
        continue
    scvi.append(torch.load(gp, map_location="cpu", weights_only=False).numpy())
    rctd.append(torch.load(cp, map_location="cpu", weights_only=False).numpy())
    kept_idx.append(i)
    if len(kept_idx) % 4000 == 0:
        print(f"  loaded {len(kept_idx)} spots...")

df = qc.loc[kept_idx].reset_index(drop=True).copy()
X_scvi = np.vstack(scvi).astype(np.float64)          # (N,50)
X_rctd = np.vstack(rctd).astype(np.float64)          # (N,15)
hepato = X_rctd[:, HEPATOCYTE_IDX]
tumor  = X_rctd[:, TUMOR_IDX]
df["hepatocyte_frac"] = hepato
df["tumor_frac"]      = tumor
y = (df["group"].values == "HM").astype(int)         # 1 = HM
samples = df["sample"].values
print(f"Loaded {len(df)} spots | HM={int(y.sum())} PT={int((1-y).sum())}")

results = {"n_spots": int(len(df)),
           "n_HM": int(y.sum()), "n_PT": int((1 - y).sum()),
           "tumor_thresh": TUMOR_THRESH}

# ----------------------------------------------------------------------------- helpers
def loso_accuracy(X, y, samples):
    """Leave-one-SAMPLE-out: train on 5 samples, predict the held-out one.
    Held-out sample is single-class, so we report per-sample accuracy (how often
    its known label is recovered) and the balanced mean over HM vs PT samples."""
    per = {}
    for s in np.unique(samples):
        te = samples == s
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(sc.transform(X[tr]), y[tr])
        pred = clf.predict(sc.transform(X[te]))
        per[str(s)] = float((pred == y[te]).mean())
    pt = [v for k, v in per.items() if "HM" not in k]
    hm = [v for k, v in per.items() if "HM" in k]
    bal = float(np.mean([np.mean(pt), np.mean(hm)])) if pt and hm else float("nan")
    return per, bal

# ---- (A) Can scVI separate HM/PT across held-out patients?  all spots vs tumour-only
for tag, sub in [("scvi_all", np.ones(len(df), bool)),
                 ("scvi_tumoronly", tumor >= TUMOR_THRESH)]:
    if sub.sum() < 50 or len(np.unique(y[sub])) < 2:
        results[tag] = {"note": "insufficient spots/classes", "n": int(sub.sum())}
        continue
    per, bal = loso_accuracy(X_scvi[sub], y[sub], samples[sub])
    results[tag] = {"n": int(sub.sum()), "per_sample": per, "balanced_acc": bal}
    print(f"[{tag}] n={sub.sum()} balanced LOSO acc={bal:.3f}")

# ---- (B) RCTD composition alone, and RCTD WITHOUT hepatocyte (isolates liver driver)
keep_no_hep = [i for i in range(X_rctd.shape[1]) if i != HEPATOCYTE_IDX]
for tag, X in [("rctd_all", X_rctd), ("rctd_no_hepatocyte", X_rctd[:, keep_no_hep])]:
    per, bal = loso_accuracy(X, y, samples)
    results[tag] = {"per_sample": per, "balanced_acc": bal}
    print(f"[{tag}] balanced LOSO acc={bal:.3f}")

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
bp = ax.boxplot(data, labels=order, patch_artist=True)
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
verdict.append(f"HM hepatocyte frac mean = {hepato[y==1].mean():.3f}  vs  PT = {hepato[y==0].mean():.3f}")
verdict.append(f"corr(metastasis-direction, hepatocyte) : all spots = {r_all:.3f} | tumour-only = {r_tumor:.3f}")
sa = results.get("scvi_all", {}).get("balanced_acc")
st = results.get("scvi_tumoronly", {}).get("balanced_acc")
verdict.append(f"scVI HM/PT LOSO balanced acc : all = {sa} | tumour-only = {st}")
verdict.append(f"RCTD LOSO acc : all = {results['rctd_all']['balanced_acc']:.3f} | "
               f"no-hepatocyte = {results['rctd_no_hepatocyte']['balanced_acc']:.3f}")
verdict.append("")
verdict.append("READ: if r_all is high AND drops sharply tumour-only -> mu_HM axis is a")
verdict.append("liver-content confound; Phase B must restrict to tumour-only / residualise.")
verdict.append("If RCTD-no-hepatocyte acc collapses vs RCTD-all -> hepatocyte IS the driver.")
txt = "\n".join(verdict)
open(os.path.join(OUT_DIR, "summary.txt"), "w").write(txt)
print("\n" + txt + f"\n\nSaved -> {OUT_DIR}")
