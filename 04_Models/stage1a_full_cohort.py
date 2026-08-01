"""
STAGE 1A (full cohort) -- the leaving-program target on all 10 primaries, and the
test the project has never been able to run properly.

Two parts:

  PART 1  Rebuild the "leaving programme" target on 10 primary slides (was 4).
          Identical scoring to `stage1a_leaving_program.py` -- same CORE/HELDOUT
          gene sets, same Seurat AddModuleScore with expression-matched controls,
          same within-sample z and residualisation -- but keyed by BARCODE, since
          the 24 new slides have no H&E patches.

  PART 2  THE CONVERGENCE TEST.  Stage 0 (full cohort) established that the tumour
          TRANSCRIPTOME separates metastasis from primary at ~0.62 balanced
          accuracy, equally for liver and lymph-node mets, i.e. a real
          site-independent metastatic axis.  The project's entire premise is that
          some PRIMARY spots are already "metastatic in behaviour".  So:

            train  metastasis-vs-primary on tumour-dominated spots (scVI latents),
                   leave-one-PATIENT-out
            apply  the held-out patient's PRIMARY spots -> `met_resemblance`
            ask    does met_resemblance track the leaving-programme score?

          A positive correlation is direct evidence that the ST-derived leaving
          programme identifies primary tumour that resembles metastasis. Near zero
          means the leaving programme and the metastatic axis are different things
          -- which would undercut the target Phase B is trained on.

          Because the cohort now has TWO metastatic sites, the same test is run
          with HM-only and LNM-only training. Agreement between them means the
          resemblance is metastatic rather than organ-specific.

Run:
    COHORT=full C:/Users/datainsight/anaconda3/envs/stproj/python.exe \
        04_Models/stage1a_full_cohort.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("COHORT", "full")

from _cohort import (ROOT, CELL_TYPES, HEPATOCYTE_IDX, ICAF_IDX, MYCAF_IDX,
                     TUMOR_IDX, PATIENT_OF, SITE_OF, PT_SAMPLES, ALL_SAMPLES,
                     RESID_ON, COHORT)

if COHORT != "full":
    sys.exit("this script requires COHORT=full")

FULL = os.path.join(ROOT, "dataset", "full_cohort")
COUNTS_DIR = os.path.join(FULL, "scVI_counts")
RCTD_DIR = os.path.join(FULL, "rctd")
COORDS_DIR = os.path.join(FULL, "coords")
SCVI_DIR = os.path.join(FULL, "gene_scvi")
OUT_DIR = os.path.join(ROOT, "Outputs", "stage1a_full_cohort" +
                       ("_residcaf" if RESID_ON == "tumor_caf" else ""))
os.makedirs(OUT_DIR, exist_ok=True)

TUMOR_THRESH = 0.50

# ---- gene sets: identical to stage1a_leaving_program.py ----
EMT_CORE = ["SNAI1", "ZEB1", "ZEB2", "CDH2", "S100A4", "TGFBR1", "TGFB1",
            "MMP1", "LOXL2", "ITGB6", "LAMC2", "SERPINE1"]
ECM_PANEL = ["COL1A1", "COL3A1", "COL5A1", "FN1", "LOX", "TNC", "TIMP1"]
CORE = EMT_CORE + ECM_PANEL
HELDOUT = ["VIM", "SNAI2", "PRRX1", "MMP9", "MMP14", "POSTN", "SPARC"]

RNG = np.random.default_rng(0)
N_BINS, N_CTRL = 25, 100

print("=" * 74)
print(f"STAGE 1A - FULL COHORT   [{len(PT_SAMPLES)} primaries | RESID_ON={RESID_ON}]")
print("=" * 74)


def add_module_score(lognorm, all_genes, sig_genes, bins):
    """Seurat AddModuleScore: mean(signature) - mean(expression-matched controls)."""
    gidx = {g: i for i, g in enumerate(all_genes)}
    sig = [g for g in sig_genes if g in gidx]
    sig_i = [gidx[g] for g in sig]
    ctrl_i = set()
    for g in sig:
        pool = np.where(bins == bins[gidx[g]])[0]
        pool = pool[~np.isin(pool, sig_i)]
        if len(pool):
            ctrl_i.update(RNG.choice(pool, size=min(N_CTRL, len(pool)),
                                     replace=False).tolist())
    ctrl_i = np.array(sorted(ctrl_i))
    ctrl_mean = lognorm[:, ctrl_i].mean(axis=1) if len(ctrl_i) else 0.0
    return lognorm[:, sig_i].mean(axis=1) - ctrl_mean, sig


def hex_morans_I(score, rc, n_perm=999):
    pos = {(int(r), int(c)): i for i, (r, c) in enumerate(rc)}
    offs = [(0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    rows, cols = [], []
    for (r, c), i in pos.items():
        for dr, dc in offs:
            j = pos.get((r + dr, c + dc))
            if j is not None:
                rows.append(i); cols.append(j)
    if not rows:
        return float("nan"), float("nan")
    rows, cols = np.array(rows), np.array(cols)
    n, W = len(score), len(rows)
    z = score - score.mean()
    denom = (z ** 2).sum()
    I_obs = (n / W) * (z[rows] * z[cols]).sum() / denom
    perm = np.empty(n_perm)
    for k in range(n_perm):
        zp = RNG.permutation(z)
        perm[k] = (n / W) * (zp[rows] * zp[cols]).sum() / (zp ** 2).sum()
    return float(I_obs), float((1 + (perm >= I_obs).sum()) / (n_perm + 1))


# ============================================================ PART 1: the target
rows = []
for sample in PT_SAMPLES:
    print(f"[{sample}] scoring ...", flush=True)
    mat = pd.read_csv(os.path.join(COUNTS_DIR, sample + ".csv"), index_col=0)
    mat = mat[~mat.index.duplicated(keep="first")]
    genes = mat.index.to_numpy()
    barcodes = mat.columns.to_numpy()
    counts = mat.to_numpy(dtype=np.float64).T
    del mat

    lib = counts.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    lognorm = np.log1p(counts / lib * 1e4)
    del counts

    gene_avg = lognorm.mean(axis=0)
    bins = pd.qcut(pd.Series(gene_avg).rank(method="first"),
                   N_BINS, labels=False).to_numpy()

    score_core, used_core = add_module_score(lognorm, genes, CORE, bins)
    score_emt, _ = add_module_score(lognorm, genes, EMT_CORE, bins)
    score_ecm, _ = add_module_score(lognorm, genes, ECM_PANEL, bins)
    score_hout, _ = add_module_score(lognorm, genes, HELDOUT, bins)

    gidx = {g: i for i, g in enumerate(genes)}
    Xc = lognorm[:, [gidx[g] for g in used_core]]
    Xc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-9)
    pc1 = PCA(n_components=1, random_state=0).fit_transform(Xc)[:, 0]
    if np.corrcoef(pc1, score_core)[0, 1] < 0:
        pc1 = -pc1
    del lognorm

    rc = pd.read_csv(os.path.join(RCTD_DIR, f"{sample}_rctd_fullfinal.csv")).set_index("barcode")
    co = pd.read_csv(os.path.join(COORDS_DIR, f"{sample}_coords.csv")).set_index("barcode")
    W = rc.reindex(barcodes)[CELL_TYPES].to_numpy(np.float64)

    sdf = pd.DataFrame({
        "sample": sample, "patient": PATIENT_OF[sample], "barcode": barcodes,
        "row": co.reindex(barcodes)["row"].to_numpy(),
        "col": co.reindex(barcodes)["col"].to_numpy(),
        "tumor_frac": W[:, TUMOR_IDX], "hepatocyte_frac": W[:, HEPATOCYTE_IDX],
        "caf_frac": W[:, ICAF_IDX] + W[:, MYCAF_IDX],
        "score_core": score_core, "score_emt": score_emt, "score_ecm": score_ecm,
        "score_pca": pc1, "score_heldout": score_hout,
    })
    sdf["leaving_score"] = ((sdf.score_core - sdf.score_core.mean())
                            / (sdf.score_core.std() + 1e-9))

    # residualise within sample on abundance (and stroma, if RESID_ON=tumor_caf)
    conf = ["tumor_frac"] + (["caf_frac"] if RESID_ON == "tumor_caf" else [])
    A = np.column_stack([np.ones(len(sdf))] + [sdf[c].to_numpy() for c in conf])
    ok = np.isfinite(A).all(1)
    for src, dst in [("leaving_score", "leaving_score_resid"),
                     ("score_heldout", "heldout_resid")]:
        y = sdf[src].to_numpy(np.float64)
        r = np.full(len(y), np.nan)
        beta, *_ = np.linalg.lstsq(A[ok], y[ok], rcond=None)
        r[ok] = y[ok] - A[ok] @ beta
        sdf[dst] = r

    I, p = hex_morans_I(sdf.leaving_score.to_numpy(),
                        sdf[["row", "col"]].to_numpy())
    print(f"    n={len(sdf)}  Moran's I(core)={I:.3f} p={p:.3f}  "
          f"core~heldout={np.corrcoef(sdf.score_core, sdf.score_heldout)[0,1]:+.3f}  "
          f"core~CAF={np.corrcoef(sdf.score_core, sdf.caf_frac)[0,1]:+.3f}")
    sdf["morans_I"] = I
    rows.append(sdf)

target = pd.concat(rows, ignore_index=True)
target.to_csv(os.path.join(OUT_DIR, "leaving_program_scores.csv"), index=False)
print(f"\nTarget built: {len(target)} primary spots across "
      f"{target['sample'].nunique()} slides / {target['patient'].nunique()} patients")


# ============================================================ PART 2: convergence
print("\n" + "=" * 74)
print("CONVERGENCE TEST  does the leaving programme mark metastasis-like primary?")
print("=" * 74)

scvi_files = sorted(f for f in os.listdir(SCVI_DIR) if f.endswith("_scvi_latent.csv"))
if not scvi_files:
    sys.exit("no scVI latents -- run 03_Embedding_Extraction/Gene/scvi_full_cohort.py")
lat = []
for f in scvi_files:
    s = f.replace("_scvi_latent.csv", "")
    d = pd.read_csv(os.path.join(SCVI_DIR, f))
    d["sample"], d["patient"], d["site"] = s, PATIENT_OF[s], SITE_OF[s]
    lat.append(d)
lat = pd.concat(lat, ignore_index=True)
SC = [c for c in lat.columns if c.startswith("scvi_")]

# tumour fraction per barcode, for the tumour-dominated restriction
tf = []
for s in ALL_SAMPLES:
    d = pd.read_csv(os.path.join(RCTD_DIR, f"{s}_rctd_fullfinal.csv"))
    tf.append(pd.DataFrame({"barcode": d["barcode"],
                            "tumor_frac": d["Tumor Epithelial cells"]}))
lat = lat.merge(pd.concat(tf, ignore_index=True), on="barcode", how="left")
lat_td = lat[lat.tumor_frac >= TUMOR_THRESH]
print(f"tumour-dominated spots: {len(lat_td)}  "
      f"({dict(lat_td.site.value_counts())})")


def resemblance(met_sites, tag):
    """Leave-one-patient-out metastasis-vs-primary; score held-out PRIMARY spots."""
    sub = lat_td[lat_td.site.isin(list(met_sites) + ["T"])]
    y = sub.site.isin(met_sites).to_numpy().astype(int)
    g = sub.patient.to_numpy()
    X = sub[SC].to_numpy(np.float64)
    out = {}
    for pat in sorted(set(g)):
        tr = g != pat
        if len(np.unique(y[tr])) < 2:
            continue
        te = (g == pat) & (sub.site == "T").to_numpy()
        if te.sum() == 0:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(X[tr], y[tr])
        out.update(dict(zip(sub.barcode.to_numpy()[te],
                            clf.decision_function(X[te]))))
    print(f"  [{tag}] scored {len(out)} held-out primary spots "
          f"from {sub[sub.site=='T'].patient.nunique()} patients")
    return out


results = {"resid_on": RESID_ON, "n_primary_spots": int(len(target)), "tests": []}
for met_sites, tag in [(("HM", "LNM"), "both met sites"),
                       (("HM",), "liver mets only"),
                       (("LNM",), "lymph-node mets only")]:
    sc = resemblance(met_sites, tag)
    t = target[target.barcode.isin(sc)].copy()
    t["met_resemblance"] = t.barcode.map(sc)
    row = {"trained_on": list(met_sites), "n": int(len(t))}
    for col in ["leaving_score", "leaving_score_resid", "score_emt", "score_ecm",
                "score_heldout", "tumor_frac", "caf_frac"]:
        m = np.isfinite(t[col]) & np.isfinite(t.met_resemblance)
        row[col] = float(spearmanr(t[col][m], t.met_resemblance[m]).statistic) \
            if m.sum() > 10 else float("nan")
    # per-patient, to show it is not one slide driving it
    per = [float(spearmanr(gg.leaving_score_resid, gg.met_resemblance).statistic)
           for _, gg in t.groupby("patient") if len(gg) > 20]
    row["resid_per_patient_median"] = float(np.nanmedian(per)) if per else float("nan")
    row["resid_per_patient_pos"] = int(np.sum(np.array(per) > 0))
    row["resid_per_patient_n"] = len(per)
    results["tests"].append(row)
    print(f"    rho(met_resemblance, leaving_raw)   = {row['leaving_score']:+.3f}")
    print(f"    rho(met_resemblance, leaving_resid) = {row['leaving_score_resid']:+.3f}"
          f"   [per-patient median {row['resid_per_patient_median']:+.3f}, "
          f"{row['resid_per_patient_pos']}/{row['resid_per_patient_n']} positive]")
    print(f"    rho(met_resemblance, tumor_frac)    = {row['tumor_frac']:+.3f}"
          f"   rho(.., caf_frac) = {row['caf_frac']:+.3f}")

with open(os.path.join(OUT_DIR, "convergence.json"), "w") as fh:
    json.dump(results, fh, indent=2)

# ------------------------------------------------------------------ verdict
both = results["tests"][0]
hm_only = results["tests"][1]
ln_only = results["tests"][2]
print("\n" + "=" * 74)
print("VERDICT")
print("=" * 74)
print(f"  leaving_resid vs met_resemblance : both={both['leaving_score_resid']:+.3f}  "
      f"HM-trained={hm_only['leaving_score_resid']:+.3f}  "
      f"LNM-trained={ln_only['leaving_score_resid']:+.3f}")
agree = abs(hm_only["leaving_score_resid"] - ln_only["leaving_score_resid"]) < 0.10
strong = abs(both["leaving_score_resid"]) >= 0.10
if strong and both["leaving_score_resid"] > 0:
    print("  -> The leaving programme DOES mark primary tumour that resembles metastasis.")
elif strong:
    print("  -> NEGATIVE: high-leaving primary spots look LESS metastatic. The target is")
    print("     pointing the wrong way and Phase B should not be trained on it as-is.")
else:
    print("  -> Near zero: the leaving programme and the metastatic transcriptional axis")
    print("     are largely INDEPENDENT. The ST target is not a metastasis proxy, so a")
    print("     model trained on it is not predicting metastatic behaviour.")
print(f"  HM- and LNM-trained agree: {agree} -> "
      f"{'resemblance is site-independent' if agree else 'resemblance is site-specific'}")
print(f"\nSaved -> {OUT_DIR}")
