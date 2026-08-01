"""
STAGE 7 -- external replication of the site-specificity finding, in an independent
cohort with THREE metastatic sites.

Why
---
Stage 6 (our cohort, GSE272362) found that the PDAC metastatic program is only
~55 % shared between liver and lymph node, and that what differs is immune.  That
rests on 4 patients who donated two metastatic sites, and on a single site pair.

GSE274557 (Maitra lab, 2025; PMID 40269162) is a much better test bed:
  * 57 Visium sections, 13 patients, ALL treatment-naive
  * primary PDAC (19), liver (16), peritoneal (14), lung (6)
  * 13/13 patients have primary + >=1 metastasis
  * 12/13 have TWO OR MORE DIFFERENT metastatic sites

What it can and cannot test
---------------------------
CAN: the general claim -- is the metastatic program site-dependent, and by how much?
     And it goes further than our cohort could: peritoneum and lung are NOT immune
     organs, so if the site-specific component is immune there too, the finding is
     not an artefact of lymph nodes being lymphoid tissue.
CANNOT: replicate the specific B-cell/lymph-node result -- this cohort has no
     lymph-node samples.

Method mirrors Stage 6 so the numbers are comparable:
  * cell-type fractions are unavailable here (no deconvolution shipped), so
    composition is estimated with MARKER-GENE signature scores computed directly
    from expression. Fewer moving parts than running a deconvolution, and it does
    not depend on a reference dataset.
  * tumour purity proxy = epithelial marker score; the same PURITY SWEEP control is
    applied.
  * unit of analysis is the SECTION (pseudobulk), never the spot.
  * organ-marker scores (hepatocyte / lung / mesothelial) act as positive controls:
    they MUST light up at their own site or the pipeline is broken.

Run:
    python 04_Models/stage7_external_replication.py
"""

import json
import os
import sys
import tarfile

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = os.path.join(ROOT, "dataset", "external", "GSE274557")
RAW = os.path.join(EXT, "raw")
OUT = os.path.join(ROOT, "Outputs", "stage7_external_replication")
os.makedirs(OUT, exist_ok=True)

PURITIES = [0.0, 0.25, 0.50, 0.75]     # quantile of the epithelial score, within section
PRIMARY_Q = 0.50

MARKERS = {
    # tumour / epithelial -- used as the purity proxy
    "Epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "KRT7", "MUC1", "CLDN4",
                   "TFF1", "TFF2", "SOX9", "CEACAM5", "S100P"],
    # immune compartments -- the Stage-6 claim lives here
    "Bcells": ["MS4A1", "CD79A", "CD79B", "CD19", "BANK1", "TNFRSF13C", "IGHM", "CR2"],
    "Tcells": ["CD3D", "CD3E", "CD3G", "CD2", "TRAC", "IL7R", "CD8A", "LCK"],
    "Plasma": ["MZB1", "JCHAIN", "DERL3", "XBP1", "TNFRSF17"],
    "Myeloid": ["CD68", "CD163", "LYZ", "AIF1", "ITGAM", "MRC1", "C1QA", "C1QB"],
    "NK": ["NKG7", "GNLY", "KLRD1", "PRF1", "GZMB"],
    # stroma
    "CAF": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "FAP", "PDGFRB", "POSTN"],
    "myCAF": ["ACTA2", "TAGLN", "MYL9", "TPM2"],
    "Endothelial": ["PECAM1", "VWF", "CDH5", "CLDN5", "PLVAP"],
    # ORGAN markers -- positive controls, one per destination
    "Hepatocyte": ["ALB", "APOA1", "APOA2", "TF", "TTR", "FGA", "FGB", "HP", "SERPINA1"],
    "Lung": ["SFTPC", "SFTPB", "SFTPA1", "NAPSA", "AGER", "SCGB1A1"],
    "Mesothelial": ["MSLN", "UPK3B", "KRT5", "CALB2", "WT1", "LRRN4"],
    # biology of interest
    "EMT": ["VIM", "SNAI1", "SNAI2", "ZEB1", "ZEB2", "CDH2", "FN1", "TWIST1"],
    "Proliferation": ["MKI67", "TOP2A", "PCNA", "CCNB1", "BIRC5"],
    "Hypoxia": ["VEGFA", "CA9", "SLC2A1", "LDHA", "PGK1", "ADM"],
    "Interferon": ["ISG15", "IFI6", "IFIT1", "IFIT3", "MX1", "OAS1", "STAT1"],
}
IMMUNE_SETS = ["Bcells", "Tcells", "Plasma", "Myeloid", "NK"]


def log(m):
    print(m, flush=True)


def load_h5(path):
    with h5py.File(path, "r") as f:
        m = f["matrix"]
        shape = m["shape"][:]                       # (genes, spots)
        X = sp.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=tuple(shape))
        genes = np.array([b.decode() for b in m["features"]["name"][:]])
        bc = np.array([b.decode() for b in m["barcodes"][:]])
    return X.T.tocsr(), genes, bc                    # spots x genes


def module_scores(Xln, genes):
    """Mean log-expression of a marker set minus the global mean, per spot.

    A simple AddModuleScore-style contrast: subtracting each spot's overall mean
    removes depth/among-spot scaling so scores are comparable across sections.
    """
    gidx = {g: i for i, g in enumerate(genes)}
    base = np.asarray(Xln.mean(axis=1)).ravel()
    out = {}
    for name, gs in MARKERS.items():
        idx = [gidx[g] for g in gs if g in gidx]
        if not idx:
            out[name] = np.zeros(Xln.shape[0]); continue
        out[name] = np.asarray(Xln[:, idx].mean(axis=1)).ravel() - base
    return pd.DataFrame(out)


log("=" * 78)
log("STAGE 7 - external replication in GSE274557 (3 metastatic sites, 13 patients)")
log("=" * 78)

ss = pd.read_csv(os.path.join(EXT, "sample_sheet.csv"))
ss = ss[ss.site_short.isin(["T", "HM", "LuM", "PM"])]
log(f"\n[1/5] {len(ss)} Visium sections  " + str(ss.site.value_counts().to_dict()))

files = os.listdir(RAW)
h5map = {}
for f in files:
    if f.endswith("_filtered_feature_bc_matrix.h5"):
        h5map[f.split("_")[0]] = f

rows = []
for _, r in ss.iterrows():
    fn = h5map.get(r.gsm)
    if fn is None:
        log(f"      no h5 for {r.gsm} ({r['sample']})"); continue
    X, genes, bc = load_h5(os.path.join(RAW, fn))
    lib = np.asarray(X.sum(axis=1)).ravel()
    keep = lib >= 500                                   # drop near-empty spots
    X, lib = X[keep], lib[keep]
    Xln = X.multiply(1e4 / lib[:, None]).log1p().tocsr()
    S = module_scores(Xln, genes)
    S["gsm"], S["sample"], S["patient"], S["site"] = r.gsm, r["sample"], r.patient, r.site_short
    if X.shape[0] < 200:
        log(f"      SKIP {r['sample']:12s} {r.site_short:4s} only {X.shape[0]} spots (too few to pseudobulk)")
        continue
    rows.append(S)
    log(f"      {r['sample']:12s} {r.site_short:4s} {X.shape[0]:5d} spots x {X.shape[1]} genes")

SP = pd.concat(rows, ignore_index=True)
SP.to_csv(os.path.join(OUT, "spot_scores.csv.gz"), index=False, compression="gzip")
log(f"\n      total spots scored: {len(SP)}")

# ------------------------------------------------------------ purity sweep
log("\n[2/5] PURITY SWEEP -- restrict to epithelium-rich spots (within-section quantile)")
sweep = []
for q in PURITIES:
    keep = SP.groupby("sample")["Epithelial"].transform(lambda v: v >= v.quantile(q))
    sub = SP[keep]
    pb = sub.groupby(["sample", "patient", "site"], as_index=False).mean(numeric_only=True)
    row = {"q_thresh": q, "n_spots": int(len(sub))}
    for site in ["HM", "LuM", "PM"]:
        a = pb[pb.site == site]; b = pb[pb.site == "T"]
        if len(a) >= 3:
            row[f"{site}_immune"] = float(a[IMMUNE_SETS].sum(1).mean())
    row["T_immune"] = float(pb[pb.site == "T"][IMMUNE_SETS].sum(1).mean())
    row["mean_epi"] = float(pb["Epithelial"].mean())
    sweep.append(row)
SWP = pd.DataFrame(sweep)
SWP.to_csv(os.path.join(OUT, "purity_sweep.csv"), index=False)
log("      quantile  spots   mean-epi   immune(T/HM/LuM/PM)")
for _, r in SWP.iterrows():
    log(f"      >={r.q_thresh:.2f}    {int(r.n_spots):6d}  {r.mean_epi:+.3f}   "
        f"{r.T_immune:+.3f} / {r.get('HM_immune',float('nan')):+.3f} / "
        f"{r.get('LuM_immune',float('nan')):+.3f} / {r.get('PM_immune',float('nan')):+.3f}")

# ------------------------------------------------------------ pseudobulk
log(f"\n[3/5] pseudobulk per SECTION at epithelial quantile >= {PRIMARY_Q}")
keep = SP.groupby("sample")["Epithelial"].transform(lambda v: v >= v.quantile(PRIMARY_Q))
PB = SP[keep].groupby(["sample", "patient", "site"], as_index=False).mean(numeric_only=True)
PB.to_csv(os.path.join(OUT, "pseudobulk_sections.csv"), index=False)
log(f"      {len(PB)} sections  " + str(PB.site.value_counts().to_dict()))

FEAT = [k for k in MARKERS if k != "Epithelial"]

# positive controls
log("\n[4/5] POSITIVE CONTROLS - organ markers must peak at their own site")
for mk, site, nm in [("Hepatocyte", "HM", "liver"), ("Lung", "LuM", "lung"),
                     ("Mesothelial", "PM", "peritoneum")]:
    means = PB.groupby("site")[mk].mean()
    win = means.idxmax()
    log(f"      {mk:12s} highest at {win:4s} (expected {site:4s})  "
        + "  ".join(f"{s}={means.get(s,float('nan')):+.3f}" for s in ["T", "HM", "LuM", "PM"])
        + ("   OK" if win == site else "   <-- UNEXPECTED"))

# ------------------------------------------------------------ site geometry
log("\n[5/5] HOW SITE-SPECIFIC IS THE METASTATIC PROGRAM?")
prim = PB[PB.site == "T"][FEAT].mean()
sd = PB[FEAT].std() + 1e-9
shifts = {}
for site in ["HM", "LuM", "PM"]:
    shifts[site] = ((PB[PB.site == site][FEAT].mean() - prim) / sd).to_numpy()
z = lambda v: v / (np.linalg.norm(v) + 1e-12)
pairs = [("HM", "LuM"), ("HM", "PM"), ("LuM", "PM")]
log("      cosine between the 'becoming a metastasis' shifts of two sites:")
cosines = {}
for a, b in pairs:
    c = float(z(shifts[a]) @ z(shifts[b]))
    cosines[f"{a}_vs_{b}"] = c
    log(f"        {a:4s} vs {b:4s} : {c:+.3f}   -> {100*max(c,0):.0f}% shared")
mean_cos = float(np.mean(list(cosines.values())))
log(f"      mean across site pairs = {mean_cos:+.3f}")
log(f"      (our cohort, liver vs lymph node, Stage 6 = +0.547)")

# per-feature: shared vs site-specific
sh = np.mean([z(shifts[s]) for s in shifts], axis=0)
tab = pd.DataFrame({"feature": FEAT})
for s in shifts:
    tab[f"shift_{s}"] = shifts[s]
tab["mean_shift"] = tab[[f"shift_{s}" for s in shifts]].mean(1)
tab["range_across_sites"] = tab[[f"shift_{s}" for s in shifts]].max(1) - \
                            tab[[f"shift_{s}" for s in shifts]].min(1)
tab["is_immune"] = tab.feature.isin(IMMUNE_SETS)
tab = tab.sort_values("range_across_sites", ascending=False)
tab.to_csv(os.path.join(OUT, "site_shifts.csv"), index=False)

log("\n      features that differ MOST between metastatic sites (site-specific):")
for _, r in tab.head(8).iterrows():
    tag = "  <-- IMMUNE" if r.is_immune else ""
    log(f"        {r.feature:14s} range={r.range_across_sites:5.2f}   "
        f"HM={r.shift_HM:+.2f} LuM={r.shift_LuM:+.2f} PM={r.shift_PM:+.2f}{tag}")
log("\n      features that shift the SAME at every site (shared program):")
for _, r in tab.tail(6).iloc[::-1].iterrows():
    log(f"        {r.feature:14s} range={r.range_across_sites:5.2f}   mean shift={r.mean_shift:+.2f}")

imm = tab[tab.is_immune].range_across_sites.mean()
non = tab[~tab.is_immune].range_across_sites.mean()
log(f"\n      mean between-site range: immune {imm:.2f} vs non-immune {non:.2f}  "
    f"-> immune is {'MORE' if imm > non else 'NOT more'} site-variable")

# paired within-patient test on immune total
log("\n      within-patient test (patients with >=2 different met sites):")
res = []
for p, g in PB[PB.site != "T"].groupby("patient"):
    if g.site.nunique() >= 2:
        v = g.groupby("site")[IMMUNE_SETS].sum(1) if False else \
            g.assign(imm=g[IMMUNE_SETS].sum(1)).groupby("site")["imm"].mean()
        res.append({"patient": p, **v.to_dict()})
PR = pd.DataFrame(res)
PR.to_csv(os.path.join(OUT, "paired_immune.csv"), index=False)
log(f"        {len(PR)} patients with 2+ distinct metastatic sites")
log(PR.to_string(index=False))

json.dump({"cosines": cosines, "mean_cosine": mean_cos,
           "stage6_cosine_our_cohort": 0.547,
           "n_sections": PB.site.value_counts().to_dict(),
           "immune_range": imm, "nonimmune_range": non},
          open(os.path.join(OUT, "metrics.json"), "w"), indent=2, default=float)
log(f"\nSaved -> {OUT}")
