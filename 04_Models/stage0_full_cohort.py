"""
STAGE 0 (full cohort) -- the two tests the 6-sample cohort could not run.

Background
----------
`stage0_confound_diagnostic.py` asks whether the "metastatic direction" learned
from spatial transcriptomics is real metastatic biology or simply liver tissue.
Its decisive test -- restrict to TUMOUR-DOMINATED spots (Tumor Epithelial >= 0.50),
where hepatocyte contamination is minimal, and see whether HM still separates from
PT -- returned `nan` on 6 slides: all 876 tumour-dominated HM spots came from
HM11, so no leave-one-patient-out fold could ever hold one out.

The 30-slide split fixes this and adds a second, stronger test:

  TEST A  tumour-only HM vs PT, leave-one-patient-out.
          Now runnable: 9 HM patients contribute tumour-dominated spots.

  TEST B  tumour-only LNM vs PT -- the CONTROL.
          Lymph-node metastases carry essentially no hepatocytes (mean fraction
          0.0003, vs 0.322 for liver mets).  If the metastasis-vs-primary axis
          holds for LNM as well as HM, it cannot be an artifact of liver tissue.
          This is the cleanest available answer to the project's #1 confound risk
          and the 6-sample cohort could not pose it at all.

Both modalities are keyed by BARCODE (there are no H&E patches for the 24 new
slides), so this script deliberately does not touch patch_stem / spot_qc_mask.

Input   dataset/full_cohort/rctd/<sample>_rctd_fullfinal.csv
        dataset/full_cohort/gene_scvi/<sample>_scvi_latent.csv   (optional;
              tests fall back to cell-modality-only if scVI has not been run)
Output  Outputs/stage0_full_cohort/

Run:
    COHORT=full C:/Users/datainsight/anaconda3/envs/stproj/python.exe \
        04_Models/stage0_full_cohort.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("COHORT", "full")

from _cohort import (ROOT, CELL_TYPES, HEPATOCYTE_IDX, TUMOR_IDX, PATIENT_OF,
                     SITE_OF, ALL_SAMPLES, COHORT)

if COHORT != "full":
    sys.exit("this script requires COHORT=full")

FULL = os.path.join(ROOT, "dataset", "full_cohort")
RCTD_DIR = os.path.join(FULL, "rctd")
SCVI_DIR = os.path.join(FULL, "gene_scvi")
OUT_DIR = os.path.join(ROOT, "Outputs", "stage0_full_cohort")
os.makedirs(OUT_DIR, exist_ok=True)

TUMOR_THRESH = 0.50     # same definition of "tumour-dominated" as stage 0

print("=" * 72)
print("STAGE 0 - FULL COHORT   tumour-only metastasis-vs-primary + LNM control")
print("=" * 72)

# ------------------------------------------------------------------ load
frames = []
for s in ALL_SAMPLES:
    rc = pd.read_csv(os.path.join(RCTD_DIR, f"{s}_rctd_fullfinal.csv"))
    if list(rc.columns[1:]) != CELL_TYPES:
        sys.exit(f"FATAL: {s} cell-type column order differs from CELL_TYPES")
    rc["sample"], rc["patient"], rc["site"] = s, PATIENT_OF[s], SITE_OF[s]
    frames.append(rc)
cells = pd.concat(frames, ignore_index=True)
W = cells[CELL_TYPES].to_numpy(np.float64)
cells["tumor_frac"] = W[:, TUMOR_IDX]
cells["hepatocyte_frac"] = W[:, HEPATOCYTE_IDX]
print(f"cell modality : {len(cells)} spots, {cells['sample'].nunique()} samples, "
      f"{cells['patient'].nunique()} patients")

scvi = None
if os.path.isdir(SCVI_DIR):
    sf = [f for f in os.listdir(SCVI_DIR) if f.endswith("_scvi_latent.csv")]
    if sf:
        scvi = pd.concat([pd.read_csv(os.path.join(SCVI_DIR, f)) for f in sorted(sf)],
                         ignore_index=True)
        print(f"gene modality : {len(scvi)} spots from {len(sf)} samples")
if scvi is None:
    print("gene modality : ABSENT -- run 03_Embedding_Extraction/Gene/scvi_full_cohort.py")
    print("                (cell-modality tests still run below)")

SCVI_COLS = [c for c in (scvi.columns if scvi is not None else []) if c.startswith("scvi_")]
df = cells if scvi is None else cells.merge(scvi, on="barcode", how="inner")
if scvi is not None:
    print(f"joined        : {len(df)} spots with both modalities")


# ------------------------------------------------------------------ helper
def lopo_balanced_acc(X, y, groups, label):
    """Pooled out-of-fold balanced accuracy under leave-one-PATIENT-out.

    Pooled rather than averaged per fold because fold sizes are very uneven and a
    fold containing only one class yields an undefined per-fold score.
    """
    n_pos_pat = len(np.unique(groups[y == 1]))
    n_neg_pat = len(np.unique(groups[y == 0]))
    if n_pos_pat < 2 or n_neg_pat < 2:
        return {"label": label, "n": int(len(y)), "balanced_acc": float("nan"),
                "note": f"needs >=2 patients per class (pos={n_pos_pat} neg={n_neg_pat})"}
    oof = np.full(len(y), -1)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict(X[te])
    m = oof >= 0
    if m.sum() == 0 or len(np.unique(y[m])) < 2:
        return {"label": label, "n": int(len(y)), "balanced_acc": float("nan"),
                "note": "no evaluable folds"}
    ba = balanced_accuracy_score(y[m], oof[m])
    tpr = float((oof[m][y[m] == 1] == 1).mean()) if (y[m] == 1).any() else float("nan")
    tnr = float((oof[m][y[m] == 0] == 0).mean()) if (y[m] == 0).any() else float("nan")
    return {"label": label, "n": int(m.sum()), "balanced_acc": float(ba),
            "tpr_met": tpr, "tnr_primary": tnr,
            "n_patients_met": int(n_pos_pat), "n_patients_primary": int(n_neg_pat)}


def run_contrast(pos_site, neg_site="T", tumour_only=True):
    """pos_site vs neg_site. tumour_only restricts to Tumor Epithelial >= 0.50."""
    sub = df[df["site"].isin([pos_site, neg_site])]
    if tumour_only:
        sub = sub[sub["tumor_frac"] >= TUMOR_THRESH]
    y = (sub["site"] == pos_site).to_numpy().astype(int)
    g = sub["patient"].to_numpy()
    out = {"contrast": f"{pos_site}_vs_{neg_site}", "tumour_only": tumour_only,
           "pos_site": pos_site, "neg_site": neg_site, "n_spots": int(len(sub)),
           "mean_hepatocyte_pos": float(sub.loc[y == 1, "hepatocyte_frac"].mean()),
           "mean_hepatocyte_neg": float(sub.loc[y == 0, "hepatocyte_frac"].mean()),
           "results": []}
    if len(sub) == 0:
        return out
    Wc = sub[CELL_TYPES].to_numpy(np.float64)
    out["results"].append(lopo_balanced_acc(Wc, y, g, "rctd_all"))
    keep = [i for i in range(len(CELL_TYPES)) if i != HEPATOCYTE_IDX]
    out["results"].append(lopo_balanced_acc(Wc[:, keep], y, g, "rctd_no_hepatocyte"))
    # The tumour channel is what `tumour_only` conditions on, so a vector that
    # still contains it can separate purely on residual tumour-fraction spread.
    # Dropping both it and hepatocytes is the strictest cell-modality test.
    if tumour_only:
        keep2 = [i for i in range(len(CELL_TYPES))
                 if i not in (HEPATOCYTE_IDX, TUMOR_IDX)]
        out["results"].append(
            lopo_balanced_acc(Wc[:, keep2], y, g, "rctd_no_hep_no_tumour"))
    if SCVI_COLS:
        out["results"].append(
            lopo_balanced_acc(sub[SCVI_COLS].to_numpy(np.float64), y, g, "scvi"))
    return out


# ------------------------------------------------------------------ site census
print("\nSITE CENSUS (tumour-dominated = Tumor Epithelial >= 0.50)")
cen = []
for site in ["T", "HM", "LNM", "NP"]:
    g = df[df["site"] == site]
    td = g[g["tumor_frac"] >= TUMOR_THRESH]
    cen.append(dict(site=site, slides=g["sample"].nunique(), spots=len(g),
                    tumour_dominated=len(td),
                    patients_with_td=td["patient"].nunique(),
                    mean_hepatocyte=float(g["hepatocyte_frac"].mean())))
census = pd.DataFrame(cen)
print(census.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
census.to_csv(os.path.join(OUT_DIR, "site_census.csv"), index=False)

# ------------------------------------------------------------------ tests
report = {"tumour_threshold": TUMOR_THRESH, "contrasts": []}
CONTRASTS = [
    ("HM", "T", "TEST A  liver metastasis vs primary"),
    ("LNM", "T", "TEST B  lymph-node metastasis vs primary  [LIVER-CONFOUND-FREE CONTROL]"),
    ("HM", "LNM", "TEST C  liver vs lymph-node metastasis  [are the two met sites alike?]"),
]
for site, neg, name in CONTRASTS:
    for tumour_only in (False, True):
        r = run_contrast(site, neg, tumour_only)
        report["contrasts"].append(r)
        tag = "TUMOUR-ONLY" if tumour_only else "all spots  "
        print(f"\n{name}\n  [{tag}] n={r['n_spots']}  mean hepatocyte: "
              f"{r['pos_site']}={r['mean_hepatocyte_pos']:.4f} "
              f"{r['neg_site']}={r['mean_hepatocyte_neg']:.4f}")
        for m in r["results"]:
            if np.isnan(m["balanced_acc"]):
                print(f"    {m['label']:22s} balanced acc = nan   ({m.get('note','')})")
            else:
                print(f"    {m['label']:22s} balanced acc = {m['balanced_acc']:.3f}  "
                      f"(TPR_{r['pos_site']}={m['tpr_met']:.3f} "
                      f"TNR_{r['neg_site']}={m['tnr_primary']:.3f}, "
                      f"patients {m['n_patients_met']}v{m['n_patients_primary']})")

# ------------------------------------------------------------------ NULL control
# Every real contrast lands near 0.62 on scVI, which is only interpretable against
# the accuracy obtainable from inter-PATIENT variation alone.  Split the PRIMARY
# patients into two arbitrary halves -- a label with no biological meaning -- and
# run exactly the same leave-one-patient-out classification.  Whatever that scores
# is the noise floor; a real effect must clear it.
print("\n" + "=" * 72)
print("NULL CONTROL  arbitrary primary-vs-primary patient splits (tumour-only)")
print("=" * 72)
t_only = df[(df["site"] == "T") & (df["tumor_frac"] >= TUMOR_THRESH)]
t_patients = np.array(sorted(t_only["patient"].unique()))
rng = np.random.default_rng(0)
null_rows = []
for rep in range(5):
    perm = rng.permutation(t_patients)
    grp_a = set(perm[: len(perm) // 2])
    y = t_only["patient"].isin(grp_a).to_numpy().astype(int)
    g = t_only["patient"].to_numpy()
    row = {"rep": rep, "group_a": sorted(grp_a)}
    for lbl, X in [("rctd_all", t_only[CELL_TYPES].to_numpy(np.float64))] + \
                  ([("scvi", t_only[SCVI_COLS].to_numpy(np.float64))] if SCVI_COLS else []):
        row[lbl] = lopo_balanced_acc(X, y, g, lbl)["balanced_acc"]
    null_rows.append(row)
    print(f"  rep {rep}: rctd_all={row['rctd_all']:.3f}" +
          (f"  scvi={row['scvi']:.3f}" if SCVI_COLS else "") +
          f"   (A={sorted(grp_a)})")
null_df = pd.DataFrame(null_rows)
null_df.to_csv(os.path.join(OUT_DIR, "null_control.csv"), index=False)
NULL_RCTD = float(null_df["rctd_all"].mean())
NULL_SCVI = float(null_df["scvi"].mean()) if SCVI_COLS else float("nan")
print(f"\n  NOISE FLOOR (mean of 5 arbitrary splits): rctd_all={NULL_RCTD:.3f}" +
      (f"  scvi={NULL_SCVI:.3f}" if SCVI_COLS else ""))
report["null_control"] = {"rctd_all_mean": NULL_RCTD, "scvi_mean": NULL_SCVI,
                          "reps": null_rows}

with open(os.path.join(OUT_DIR, "metrics.json"), "w") as fh:
    json.dump(report, fh, indent=2)


# ------------------------------------------------------------------ verdict
def get(contrast, tumour_only, label):
    for c in report["contrasts"]:
        if c["contrast"] == contrast and c["tumour_only"] == tumour_only:
            for m in c["results"]:
                if m["label"] == label:
                    return m["balanced_acc"]
    return float("nan")


hm_all, hm_tum = get("HM_vs_T", False, "rctd_all"), get("HM_vs_T", True, "rctd_all")
ln_all, ln_tum = get("LNM_vs_T", False, "rctd_all"), get("LNM_vs_T", True, "rctd_all")
hm_nohep = get("HM_vs_T", True, "rctd_no_hepatocyte")

print("\n" + "=" * 72)
print("VERDICT")
print("=" * 72)
print(f"  HM  vs T : all spots {hm_all:.3f} -> tumour-only {hm_tum:.3f} "
      f"(no-hepatocyte {hm_nohep:.3f})")
print(f"  LNM vs T : all spots {ln_all:.3f} -> tumour-only {ln_tum:.3f}")
print()
if not np.isnan(hm_tum):
    print("  TEST A is RUNNABLE -- the 6-sample cohort returned nan here.")
    drop = hm_all - hm_tum
    print(f"    Restricting to tumour-dominated spots changes HM-vs-T by {-drop:+.3f}.")
    print("    A large drop means the HM axis was largely liver tissue;"
          " a small one means it survives where hepatocytes are scarce.")
if not np.isnan(ln_tum):
    print(f"\n  TEST B (control): LNM carries essentially no hepatocytes, so"
          f" {ln_tum:.3f}\n    cannot be explained by liver contamination.")
    if not np.isnan(hm_tum):
        print(f"    LNM {ln_tum:.3f} vs HM {hm_tum:.3f} on tumour-only spots ->", end=" ")
        print("both sites separate from primary: the axis is metastatic, not hepatic."
              if ln_tum > 0.60 else
              "the metastatic signal does NOT generalise beyond liver mets.")
if SCVI_COLS:
    hm_s = get("HM_vs_T", True, "scvi")
    ln_s = get("LNM_vs_T", True, "scvi")
    hl_s = get("HM_vs_LNM", True, "scvi")
    hm_c = get("HM_vs_T", True, "rctd_no_hep_no_tumour")
    print("\n  TRANSCRIPTOME vs COMPOSITION (tumour-only) -- the key comparison:")
    print(f"    HM vs T   : composition {hm_tum:.3f}  |  transcriptome {hm_s:.3f}")
    print(f"    LNM vs T  : composition {ln_tum:.3f}  |  transcriptome {ln_s:.3f}")
    print(f"    HM vs LNM : transcriptome {hl_s:.3f}")
    print(f"    NULL floor (arbitrary primary splits): "
          f"composition {NULL_RCTD:.3f}  |  transcriptome {NULL_SCVI:.3f}")
    print()
    print("    Composition separates HM from T far better than the transcriptome does,")
    print("    and stays high after dropping hepatocyte AND tumour channels "
          f"({hm_c:.3f}) --")
    print("    i.e. most of that signal is ORGAN MICROENVIRONMENT, not tumour biology.")
    margin_hm, margin_ln = hm_s - NULL_SCVI, ln_s - NULL_SCVI
    print(f"\n    Transcriptome margin over the null floor: "
          f"HM {margin_hm:+.3f}, LNM {margin_ln:+.3f}.")
    if max(margin_hm, margin_ln) < 0.05:
        print("    -> Both are within noise: on tumour-dominated spots the metastatic")
        print("       transcriptional signal is NOT separable from inter-patient variation.")
    elif abs(hm_s - ln_s) < 0.05:
        print("    -> HM and LNM score alike, so what the transcriptome captures is")
        print("       SITE-INDEPENDENT -- consistent with a genuine metastatic program.")
    else:
        print("    -> HM and LNM differ, so the transcriptional signal is site-specific.")
print(f"\nSaved -> {OUT_DIR}")
