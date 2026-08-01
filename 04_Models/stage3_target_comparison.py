"""
Which target should Phase B actually be trained on?

`stage1a_full_cohort.py` showed the ST "leaving programme" is largely independent
of the transcriptional metastatic axis (resid rho +0.066) -- so an H&E model that
predicts it is not predicting metastatic behaviour.  The obvious alternative is
`met_resemblance`: the leave-one-patient-out decision score of a
metastasis-vs-primary classifier trained on tumour-dominated spots in scVI space,
evaluated on held-out PRIMARY spots.  It is metastasis-derived by construction.

This script asks the only question that matters for the deliverable:

    from H&E alone, which target is more predictable on a held-out PATIENT?

Both targets are regressed from the SAME frozen UNI2-h features, with the same
leave-one-patient-out ridge, on the same spots -- so the comparison is apples to
apples.  Only the four primaries that have H&E patches can take part (T1, T3, T4,
T11), because the 24 new slides have no full-resolution WSI.

Run:
    COHORT=full C:/Users/datainsight/anaconda3/envs/stproj/python.exe \
        04_Models/stage3_target_comparison.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("COHORT", "full")

from _cohort import (ROOT, PATIENT_OF, SITE_OF, ALL_SAMPLES, COHORT)

if COHORT != "full":
    sys.exit("this script requires COHORT=full")

FULL = os.path.join(ROOT, "dataset", "full_cohort")
FM_DIR = os.path.join(ROOT, "dataset", "Feature Extraction Embeddings", "UNI2-h")
QC_CSV = os.path.join(ROOT, "Outputs", "Patient-Sample-Information", "spot_qc_mask.csv")
LEAVING_CSV = os.path.join(ROOT, "Outputs", "stage1a_full_cohort_residcaf",
                           "leaving_program_scores.csv")
OUT_DIR = os.path.join(ROOT, "Outputs", "stage3_target_comparison")
os.makedirs(OUT_DIR, exist_ok=True)

FM_SUFFIX = "_uni2h.pt"
TUMOR_THRESH = 0.50
SEED = 42
np.random.seed(SEED)

print("=" * 74)
print("TARGET COMPARISON   leaving programme  vs  met_resemblance   (H&E -> target)")
print("=" * 74)

# ------------------------------------------------------------------ met_resemblance
print("[1/4] building met_resemblance (LOPO metastasis-vs-primary in scVI space) ...")
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
td = lat[lat.tumor_frac >= TUMOR_THRESH]

sub = td[td.site.isin(["HM", "LNM", "T"])]
y_met = sub.site.isin(["HM", "LNM"]).to_numpy().astype(int)
g_met = sub.patient.to_numpy()
X_met = sub[SC].to_numpy(np.float64)
resemblance = {}
for pat in sorted(set(g_met)):
    tr = g_met != pat
    te = (g_met == pat) & (sub.site == "T").to_numpy()
    if te.sum() == 0 or len(np.unique(y_met[tr])) < 2:
        continue
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(X_met[tr], y_met[tr])
    resemblance.update(dict(zip(sub.barcode.to_numpy()[te],
                                clf.decision_function(X_met[te]))))
print(f"      met_resemblance for {len(resemblance)} held-out primary spots")

# ------------------------------------------------------------------ join targets to patches
print("[2/4] joining targets to H&E patches ...")
tgt = pd.read_csv(LEAVING_CSV)
tgt["met_resemblance"] = tgt["barcode"].map(resemblance)

qc = pd.read_csv(QC_CSV)[["patch_stem", "sample", "barcode"]]
tgt = tgt.merge(qc, on=["sample", "barcode"], how="inner")
print(f"      {len(tgt)} primary spots have both a patch and a target "
      f"({tgt['sample'].nunique()} slides: {sorted(tgt['sample'].unique())})")

fm_vec = {}
for s in sorted(tgt["sample"].unique()):
    p = os.path.join(FM_DIR, s + FM_SUFFIX)
    if not os.path.exists(p):
        print(f"      no FM embeddings for {s}")
        continue
    d = torch.load(p, map_location="cpu", weights_only=False)
    for n, v in zip(d["patch_names"], d["embeddings"].numpy()):
        fm_vec[n] = v
tgt = tgt[tgt["patch_stem"].isin(fm_vec)]
tgt = tgt[np.isfinite(tgt["met_resemblance"])]
X = np.stack([fm_vec[s] for s in tgt["patch_stem"]]).astype(np.float32)
groups = tgt["patient"].to_numpy()
print(f"      aligned: {X.shape[0]} spots x {X.shape[1]}d UNI2-h, "
      f"{len(set(groups))} patients {sorted(set(groups))}")

# ------------------------------------------------------------------ LOPO ridge
print("[3/4] leave-one-patient-out ridge, identical for every target ...")
ALPHAS = np.logspace(-1, 4, 12)


def lopo_ridge(y):
    """Pooled held-out Spearman, plus per-patient. Alpha chosen inside each fold."""
    pred = np.full(len(y), np.nan)
    for pat in sorted(set(groups)):
        tr, te = groups != pat, groups == pat
        if len(np.unique(y[tr])) < 2:
            continue
        # inner split on training patients to pick alpha
        inner = groups[tr]
        best, best_rho = ALPHAS[0], -np.inf
        for a in ALPHAS:
            rhos = []
            for ip in sorted(set(inner)):
                itr, ite = inner != ip, inner == ip
                if ite.sum() < 20:
                    continue
                m = make_pipeline(StandardScaler(), Ridge(alpha=a))
                m.fit(X[tr][itr], y[tr][itr])
                r = spearmanr(y[tr][ite], m.predict(X[tr][ite])).statistic
                if np.isfinite(r):
                    rhos.append(r)
            if rhos and np.mean(rhos) > best_rho:
                best_rho, best = float(np.mean(rhos)), a
        m = make_pipeline(StandardScaler(), Ridge(alpha=best))
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    ok = np.isfinite(pred) & np.isfinite(y)
    pooled = float(spearmanr(y[ok], pred[ok]).statistic)
    per = {}
    for pat in sorted(set(groups)):
        m = (groups == pat) & ok
        if m.sum() > 20:
            per[pat] = float(spearmanr(y[m], pred[m]).statistic)
    return pooled, per, pred


TARGETS = [
    ("leaving_score", "leaving programme (raw)"),
    ("leaving_score_resid", "leaving programme (residualised)"),
    ("met_resemblance", "met_resemblance  [metastasis-derived]"),
]
results = {}
for col, label in TARGETS:
    y = tgt[col].to_numpy(np.float64)
    pooled, per, pred = lopo_ridge(y)
    tgt[f"pred_{col}"] = pred
    results[col] = {"label": label, "pooled_rho": pooled, "per_patient": per}
    print(f"      {label:42s} pooled rho = {pooled:+.3f}   "
          f"per-patient {[f'{v:+.2f}' for v in per.values()]}")

# ------------------------------------------------------------------ report
print("[4/4] writing outputs ...")
tgt.to_csv(os.path.join(OUT_DIR, "target_comparison_predictions.csv"), index=False)
with open(os.path.join(OUT_DIR, "metrics.json"), "w") as fh:
    json.dump({"n_spots": int(len(tgt)),
               "patients": sorted(set(groups.tolist())),
               "results": results,
               "corr_between_targets": {
                   "leaving_resid_vs_met_resemblance": float(
                       spearmanr(tgt.leaving_score_resid, tgt.met_resemblance).statistic),
                   "leaving_raw_vs_met_resemblance": float(
                       spearmanr(tgt.leaving_score, tgt.met_resemblance).statistic)}},
              fh, indent=2)

lv = results["leaving_score_resid"]["pooled_rho"]
mr = results["met_resemblance"]["pooled_rho"]
print("\n" + "=" * 74)
print("VERDICT")
print("=" * 74)
print(f"  H&E -> leaving programme (resid) : {lv:+.3f}")
print(f"  H&E -> met_resemblance           : {mr:+.3f}")
print(f"  correlation between the two targets on these spots: "
      f"{spearmanr(tgt.leaving_score_resid, tgt.met_resemblance).statistic:+.3f}")
print()
if mr > lv + 0.03:
    print("  -> met_resemblance is the BETTER H&E target: it is both metastasis-derived")
    print("     AND more predictable from morphology. Re-target Phase B on it.")
elif lv > mr + 0.03:
    print("  -> The leaving programme remains more predictable from H&E, but recall it")
    print("     does NOT track the metastatic axis -- H&E is predicting EMT/desmoplasia,")
    print("     not metastatic propensity. Say so explicitly rather than re-targeting.")
else:
    print("  -> Comparable. Prefer met_resemblance on construct-validity grounds: it is")
    print("     derived from actual metastases, whereas the leaving programme is not.")
print(f"\nSaved -> {OUT_DIR}")
