"""
STAGE 5 -- decompose the metastatic axis into a SHARED, site-independent component
and site-specific components, characterise its biology, and test whether it is a
better H&E target than anything tried so far.

Motivation
----------
Stage 0 (full cohort) established two things:
  * liver mets and lymph-node mets are each separable from primary on the tumour
    transcriptome at ~0.62 balanced accuracy (null 0.43), and
  * the two directions are only PARTLY the same -- cosine 0.466, and projected onto
    the same primary spots their scores correlate only rho +0.32.

So "metastatic" is not one axis. If a site-independent core exists it is the only
part that could generalise, and it is the only part worth asking H&E to predict.
Stage 1a showed the ST leaving-programme target does not track the metastatic axis
at all, and Stage 3's comparison showed H&E predicts the leaving programme (+0.116)
but not met_resemblance (-0.043) -- so a better-defined target is exactly what is
missing.

What this does
--------------
1. Fit metastasis-vs-primary on tumour-dominated spots in scVI space, separately
   for HM and for LNM, leave-one-patient-out.
2. Decompose:  shared      = the direction both agree on (normalised w_HM + w_LNM)
                site-spec  = what each has that the other does not (w_HM - w_LNM)
   Report the geometry so the split is auditable, not assumed.
3. Score every PRIMARY spot on the shared axis, out-of-fold.
4. Characterise the shared axis biologically against the 27 Bagaev fges signatures
   (and contrast with the site-specific axis), which says what it actually measures.
5. Test predictability from H&E (UNI2-h) on the primaries that have patches.

Run:
    COHORT=full python 04_Models/stage5_shared_met_axis.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("COHORT", "full")
from _cohort import ROOT, PATIENT_OF, SITE_OF, ALL_SAMPLES, COHORT

if COHORT != "full":
    sys.exit("requires COHORT=full")

FULL = os.path.join(ROOT, "dataset", "full_cohort")
OUT_DIR = os.path.join(ROOT, "Outputs", "stage5_shared_met_axis")
os.makedirs(OUT_DIR, exist_ok=True)
TUMOR_THRESH = 0.50

print("=" * 74)
print("STAGE 5 - shared vs site-specific metastatic axis")
print("=" * 74)

# ------------------------------------------------------------------ load
lat = []
for f in sorted(os.listdir(os.path.join(FULL, "gene_scvi"))):
    if not f.endswith("_scvi_latent.csv"):
        continue
    s = f.replace("_scvi_latent.csv", "")
    d = pd.read_csv(os.path.join(FULL, "gene_scvi", f))
    d["sample"], d["patient"], d["site"] = s, PATIENT_OF[s], SITE_OF[s]
    lat.append(d)
lat = pd.concat(lat, ignore_index=True)
SC = [c for c in lat.columns if c.startswith("scvi_")]

tf = pd.concat([pd.read_csv(os.path.join(FULL, "rctd", f"{s}_rctd_fullfinal.csv"))
                [["barcode", "Tumor Epithelial cells"]] for s in ALL_SAMPLES],
               ignore_index=True)
tf.columns = ["barcode", "tumor_frac"]
lat = lat.merge(tf, on="barcode", how="left")
td = lat[lat.tumor_frac >= TUMOR_THRESH].reset_index(drop=True)
print(f"tumour-dominated spots: {len(td)}  {dict(td.site.value_counts())}")


def fit_axis(met_sites, train_mask=None):
    """Logistic met-vs-primary weight vector in scVI space (standardised)."""
    sub = td[td.site.isin(list(met_sites) + ["T"])]
    if train_mask is not None:
        sub = sub[train_mask(sub)]
    y = sub.site.isin(met_sites).to_numpy().astype(int)
    if len(np.unique(y)) < 2:
        return None
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    clf.fit(sub[SC].to_numpy(np.float64), y)
    w = clf[-1].coef_.ravel()
    return w / (np.linalg.norm(w) + 1e-12)


# ------------------------------------------------------------------ geometry
w_hm, w_ln = fit_axis(("HM",)), fit_axis(("LNM",))
cos = float(w_hm @ w_ln)
shared = w_hm + w_ln
shared /= np.linalg.norm(shared)
sitespec = w_hm - w_ln
sitespec /= np.linalg.norm(sitespec)
print(f"\nGEOMETRY  cos(w_HM, w_LNM) = {cos:+.3f}")
print(f"  shared axis   captures  {50*(1+cos):.0f}% of each site's direction")
print(f"  shared . w_HM = {shared@w_hm:+.3f}   shared . w_LNM = {shared@w_ln:+.3f}")
print(f"  shared . site-specific = {shared@sitespec:+.3f}  (orthogonal by construction)")

# ------------------------------------------------------------------ OOF scores
prim = td[td.site == "T"].copy()
for name in ("shared", "sitespec", "hm", "lnm"):
    prim[f"score_{name}"] = np.nan
for pat in sorted(prim.patient.unique()):
    keep = lambda s, p=pat: s.patient != p
    a, b = fit_axis(("HM",), keep), fit_axis(("LNM",), keep)
    if a is None or b is None:
        continue
    sh = a + b; sh /= np.linalg.norm(sh)
    ss = a - b; ss /= np.linalg.norm(ss)
    m = (prim.patient == pat).to_numpy()
    X = prim.loc[m, SC].to_numpy(np.float64)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    prim.loc[m, "score_shared"] = Xs @ sh
    prim.loc[m, "score_sitespec"] = Xs @ ss
    prim.loc[m, "score_hm"] = Xs @ a
    prim.loc[m, "score_lnm"] = Xs @ b
print(f"\nscored {prim.score_shared.notna().sum()} primary spots out-of-fold "
      f"({prim.patient.nunique()} patients)")
print(f"  rho(shared, site-specific) on primaries = "
      f"{spearmanr(prim.score_shared, prim.score_sitespec).statistic:+.3f}")

# ------------------------------------------------------------------ biology
fg = []
for s in ALL_SAMPLES:
    p = os.path.join(FULL, "fges", f"{s}_fges.csv")
    if os.path.exists(p):
        fg.append(pd.read_csv(p))
fg = pd.concat(fg, ignore_index=True)
SIG = [c for c in fg.columns if c != "barcode"]
pf = prim.merge(fg, on="barcode", how="inner")
print(f"\nBIOLOGY of the shared axis  (n={len(pf)} primary spots, {len(SIG)} Bagaev signatures)")
rows = []
for s in SIG:
    rows.append(dict(signature=s,
                     rho_shared=spearmanr(pf.score_shared, pf[s]).statistic,
                     rho_sitespec=spearmanr(pf.score_sitespec, pf[s]).statistic))
bio = pd.DataFrame(rows).sort_values("rho_shared", ascending=False)
bio.to_csv(os.path.join(OUT_DIR, "shared_axis_signatures.csv"), index=False)
print("  top 8 POSITIVE on the shared (site-independent) axis:")
for _, r in bio.head(8).iterrows():
    print(f"    {r.signature:38s} {r.rho_shared:+.3f}   (site-specific {r.rho_sitespec:+.3f})")
print("  top 5 NEGATIVE:")
for _, r in bio.tail(5).iloc[::-1].iterrows():
    print(f"    {r.signature:38s} {r.rho_shared:+.3f}   (site-specific {r.rho_sitespec:+.3f})")

# ------------------------------------------------------------------ H&E test
print("\nIS THE SHARED AXIS PREDICTABLE FROM H&E?")
import torch
FM = os.path.join(ROOT, "dataset", "Feature Extraction Embeddings", "UNI2-h")
qc = pd.read_csv(os.path.join(ROOT, "Outputs", "Patient-Sample-Information",
                              "spot_qc_mask.csv"))[["patch_stem", "sample", "barcode"]]
pv = prim.merge(qc, on=["sample", "barcode"], how="inner")
vec = {}
for s in sorted(pv["sample"].unique()):
    p = os.path.join(FM, f"{s}_uni2h.pt")
    if os.path.exists(p):
        d = torch.load(p, map_location="cpu", weights_only=False)
        vec.update(dict(zip(d["patch_names"], d["embeddings"].numpy())))
pv = pv[pv.patch_stem.isin(vec) & pv.score_shared.notna()]
print(f"  primaries with H&E: {sorted(pv['sample'].unique())}  ({len(pv)} spots, "
      f"{pv.patient.nunique()} patients)")

res = {}
if pv.patient.nunique() >= 3:
    X = np.stack([vec[s] for s in pv.patch_stem]).astype(np.float32)
    g = pv.patient.to_numpy()
    for tgt in ["score_shared", "score_sitespec", "score_hm", "score_lnm"]:
        y = pv[tgt].to_numpy(np.float64)
        pred = np.full(len(y), np.nan)
        for pat in sorted(set(g)):
            tr, te = g != pat, g == pat
            m = make_pipeline(StandardScaler(), Ridge(alpha=100.0)).fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        ok = np.isfinite(pred)
        r = float(spearmanr(y[ok], pred[ok]).statistic)
        res[tgt] = r
        print(f"    H&E -> {tgt:16s} held-out rho = {r:+.3f}")

json.dump({"cos_hm_lnm": cos, "he": res,
           "n_primary_spots": int(len(prim)),
           "top_shared": bio.head(10).to_dict("records"),
           "bottom_shared": bio.tail(10).to_dict("records")},
          open(os.path.join(OUT_DIR, "metrics.json"), "w"), indent=2)
prim.drop(columns=SC).to_csv(os.path.join(OUT_DIR, "primary_axis_scores.csv"), index=False)
print(f"\nSaved -> {OUT_DIR}")
