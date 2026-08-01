"""
STAGE 6 -- Is the PDAC metastatic program site-specific, and does liver vs
lymph-node differ mainly in immune content?

READ THIS FIRST (written for a non-biologist)
---------------------------------------------
A "spot" is one 55-micron dot on the Visium slide holding ~1-10 cells; each spot
has a full gene-expression profile.  We have 91,496 of them across 30 tissue
sections from 13 patients.  Three traps decide whether this analysis is real:

TRAP 1 -- PSEUDOREPLICATION.  We have 91k spots but only 17 metastasis sections
  from 13 patients.  Spots inside one section are not independent measurements:
  they are the same tumour, sampled many times.  Testing at spot level would give
  p-values like 1e-300 for nothing.  So every statistical test here is run on
  PSEUDOBULK -- all spots of a section summed into one profile -- and the unit of
  analysis is the SECTION (n=12 liver vs 5 lymph-node), never the spot.

TRAP 2 -- ORGAN TISSUE, NOT TUMOUR.  A liver metastasis sits in liver; a nodal
  metastasis sits in a lymph node, which is an immune organ made of lymphocytes.
  If we compare whole sections we will "discover" that lymph nodes contain immune
  cells, which is a fact about anatomy, not about cancer.  So we restrict to
  TUMOUR-DOMINATED spots (RCTD tumour fraction >= 0.50) and then, crucially, we
  repeat everything at increasing purity thresholds.  If a difference is caused by
  leftover host tissue it must SHRINK as purity rises.  If it is genuine tumour
  biology it should persist.  This purity sweep is thecore control of the study.

TRAP 3 -- PATIENT CONFOUNDING.  Comparing 12 liver sections against 5 nodal ones
  compares different people.  Four patients (PT_6, PT_8, PT_10, PT_12) donated BOTH
  a liver and a nodal metastasis, so for them the comparison is within one person
  and patient identity cancels out.  That paired test is the primary evidence; the
  larger unpaired test is supporting.

Outputs
    Outputs/stage6_site_program/  -- metrics.json, per-test CSVs, and figures
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("COHORT", "full")
from _cohort import ROOT, CELL_TYPES, PATIENT_OF, SITE_OF, ALL_SAMPLES, COHORT

if COHORT != "full":
    sys.exit("requires COHORT=full")

FULL = os.path.join(ROOT, "dataset", "full_cohort")
OUT = os.path.join(ROOT, "Outputs", "stage6_site_program")
os.makedirs(OUT, exist_ok=True)

PURITIES = [0.30, 0.50, 0.70, 0.80]
PRIMARY_PURITY = 0.50

# RCTD channels that are immune
IMMUNE = ["B cells", "C1Q-TAM", "CD4+ cells", "CD8-NK cells", "DCs", "FCN1-TAM",
          "Proliferative T cells", "SPP1-TAM"]
LYMPHOID = ["B cells", "CD4+ cells", "CD8-NK cells", "Proliferative T cells"]

print("=" * 78)
print("STAGE 6 - site-specificity of the metastatic program (liver vs lymph node)")
print("=" * 78)


def log(m):
    print(m, flush=True)


# --------------------------------------------------------------- load per sample
log("\n[1/6] loading RCTD + fges for all 30 sections ...")
rc, fg = {}, {}
for s in ALL_SAMPLES:
    rc[s] = pd.read_csv(os.path.join(FULL, "rctd", f"{s}_rctd_fullfinal.csv"))
    p = os.path.join(FULL, "fges", f"{s}_fges.csv")
    if os.path.exists(p):
        fg[s] = pd.read_csv(p)
SIG = [c for c in next(iter(fg.values())).columns if c != "barcode"]
log(f"      {len(rc)} sections, {len(SIG)} Bagaev signatures")

meta = pd.DataFrame({"sample": ALL_SAMPLES})
meta["patient"] = meta["sample"].map(PATIENT_OF)
meta["site"] = meta["sample"].map(SITE_OF)
MET = meta[meta.site.isin(["HM", "LNM"])].reset_index(drop=True)
PAIRED = sorted(set(MET[MET.site == "HM"].patient) & set(MET[MET.site == "LNM"].patient))
log(f"      metastasis sections: {(MET.site=='HM').sum()} liver, {(MET.site=='LNM').sum()} nodal")
log(f"      patients with BOTH (paired design): {PAIRED}")


def spots_at(sample, purity):
    d = rc[sample]
    return d[d["Tumor Epithelial cells"] >= purity]


# --------------------------------------------------------------- purity sweep
log("\n[2/6] TRAP-2 CONTROL: does the immune difference survive rising tumour purity?")
log("      (if it is host lymph-node tissue it must collapse; if tumour biology it persists)")
sweep = []
for pur in PURITIES:
    rows = []
    for _, r in MET.iterrows():
        d = spots_at(r["sample"], pur)
        if len(d) < 25:
            continue
        row = {"sample": r["sample"], "patient": r.patient, "site": r.site,
               "n_spots": len(d), "tumor_frac": d["Tumor Epithelial cells"].mean(),
               "hepatocyte": d["Hepatocytes"].mean(),
               "immune_total": d[IMMUNE].sum(1).mean(),
               "lymphoid": d[LYMPHOID].sum(1).mean()}
        rows.append(row)
    P = pd.DataFrame(rows)
    if P.empty:
        continue
    hm, ln = P[P.site == "HM"], P[P.site == "LNM"]
    if len(hm) < 2 or len(ln) < 2:
        continue
    u = stats.mannwhitneyu(ln.lymphoid, hm.lymphoid, alternative="greater")
    sweep.append(dict(purity=pur, n_hm=len(hm), n_lnm=len(ln),
                      mean_tumor_hm=hm.tumor_frac.mean(), mean_tumor_lnm=ln.tumor_frac.mean(),
                      hepatocyte_hm=hm.hepatocyte.mean(), hepatocyte_lnm=ln.hepatocyte.mean(),
                      lymphoid_hm=hm.lymphoid.mean(), lymphoid_lnm=ln.lymphoid.mean(),
                      ratio=ln.lymphoid.mean() / max(hm.lymphoid.mean(), 1e-9),
                      p_mwu=float(u.pvalue)))
    P.to_csv(os.path.join(OUT, f"section_summary_purity{int(pur*100)}.csv"), index=False)
sw = pd.DataFrame(sweep)
log("      purity  nHM nLNM  meanTumour(HM/LNM)  hepato(HM)  lymphoid HM->LNM   ratio   p")
for _, r in sw.iterrows():
    log(f"      >={r.purity:.2f}   {int(r.n_hm):2d}  {int(r.n_lnm):2d}   "
        f"{r.mean_tumor_hm:.2f}/{r.mean_tumor_lnm:.2f}        {r.hepatocyte_hm:.3f}     "
        f"{r.lymphoid_hm:.4f} -> {r.lymphoid_lnm:.4f}  {r.ratio:5.2f}x  {r.p_mwu:.4f}")
sw.to_csv(os.path.join(OUT, "purity_sweep.csv"), index=False)

# --------------------------------------------------------------- pseudobulk fges
log(f"\n[3/6] TRAP-1 CONTROL: pseudobulk per section at purity >= {PRIMARY_PURITY}")
rows = []
for _, r in MET.iterrows():
    d = spots_at(r["sample"], PRIMARY_PURITY)
    if len(d) < 25:
        log(f"      skip {r['sample']}: only {len(d)} tumour-dominated spots")
        continue
    f = fg[r["sample"]]
    f = f[f.barcode.isin(d.barcode)]
    row = {"sample": r["sample"], "patient": r.patient, "site": r.site, "n_spots": len(d)}
    row.update({c: f[c].mean() for c in SIG})
    row.update({f"rctd::{c}": d[c].mean() for c in CELL_TYPES})
    rows.append(row)
PB = pd.DataFrame(rows)
PB.to_csv(os.path.join(OUT, "pseudobulk_sections.csv"), index=False)
log(f"      pseudobulk matrix: {PB.shape[0]} sections x {len(SIG)} signatures")

# --------------------------------------------------------------- tests
log("\n[4/6] differential test, unit = SECTION (never spot)")
feat = SIG + [f"rctd::{c}" for c in CELL_TYPES]
res = []
hm, ln = PB[PB.site == "HM"], PB[PB.site == "LNM"]
for c in feat:
    a, b = ln[c].to_numpy(float), hm[c].to_numpy(float)
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    d = (a.mean() - b.mean()) / (np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) + 1e-12)
    # paired within-patient (primary evidence)
    pa, pb = [], []
    for p in PAIRED:
        x = PB[(PB.patient == p) & (PB.site == "LNM")][c]
        y = PB[(PB.patient == p) & (PB.site == "HM")][c]
        if len(x) and len(y):
            pa.append(x.mean()); pb.append(y.mean())
    pw = stats.wilcoxon(pa, pb).pvalue if len(pa) >= 4 else np.nan
    n_same = int(np.sum(np.array(pa) > np.array(pb))) if pa else 0
    res.append(dict(feature=c, mean_HM=b.mean(), mean_LNM=a.mean(),
                    delta=a.mean() - b.mean(), cohens_d=d,
                    p_unpaired=u.pvalue, p_paired=pw,
                    n_paired_LNM_higher=n_same, n_paired=len(pa)))
R = pd.DataFrame(res)
# Benjamini-Hochberg
m = len(R)
R = R.sort_values("p_unpaired")
R["q_unpaired"] = np.minimum.accumulate((R.p_unpaired * m / np.arange(1, m + 1))[::-1])[::-1]
R = R.sort_values("cohens_d", ascending=False)
R.to_csv(os.path.join(OUT, "site_differential.csv"), index=False)

log(f"      features tested: {m}   significant at q<0.10: {(R.q_unpaired<0.10).sum()}")
log("\n      HIGHER IN LYMPH-NODE METASTASIS (top 10 by effect size):")
log(f"      {'feature':42s} {'d':>6s} {'q':>7s}  paired(LNM>HM)")
for _, r in R.head(10).iterrows():
    log(f"      {r.feature:42s} {r.cohens_d:+6.2f} {r.q_unpaired:7.4f}  {r.n_paired_LNM_higher}/{r.n_paired}")
log("\n      HIGHER IN LIVER METASTASIS (top 10):")
for _, r in R.tail(10).iloc[::-1].iterrows():
    log(f"      {r.feature:42s} {r.cohens_d:+6.2f} {r.q_unpaired:7.4f}  {r.n_paired_LNM_higher}/{r.n_paired}")

# --------------------------------------------------------------- how site-specific
log("\n[5/6] HOW site-specific is the metastatic program?")
log("      Compare each metastatic site against PRIMARY tumour, then ask how much")
log("      of what changes is the same at both sites.")
prim_rows = []
for _, r in meta[meta.site == "T"].iterrows():
    d = spots_at(r["sample"], PRIMARY_PURITY)
    if len(d) < 25:
        continue
    f = fg[r["sample"]]; f = f[f.barcode.isin(d.barcode)]
    row = {"sample": r["sample"], "patient": r.patient, "site": "T"}
    row.update({c: f[c].mean() for c in SIG})
    row.update({f"rctd::{c}": d[c].mean() for c in CELL_TYPES})
    prim_rows.append(row)
PT = pd.DataFrame(prim_rows)
dHM = np.array([PB[PB.site == "HM"][c].mean() - PT[c].mean() for c in feat])
dLN = np.array([PB[PB.site == "LNM"][c].mean() - PT[c].mean() for c in feat])
z = lambda v: v / (np.linalg.norm(v) + 1e-12)
cos_sites = float(z(dHM) @ z(dLN))
log(f"      primary sections used: {len(PT)}")
log(f"      cosine( liver-vs-primary shift , nodal-vs-primary shift ) = {cos_sites:+.3f}")
log(f"      -> {100*max(cos_sites,0):.0f}% of the change is shared; the rest is site-specific")

json.dump({"purity_sweep": sw.to_dict("records"),
           "cos_HMshift_LNMshift": cos_sites,
           "paired_patients": PAIRED,
           "n_sections": {"HM": int((PB.site == 'HM').sum()),
                          "LNM": int((PB.site == 'LNM').sum()),
                          "T": int(len(PT))},
           "top_LNM": R.head(12).to_dict("records"),
           "top_HM": R.tail(12).to_dict("records")},
          open(os.path.join(OUT, "metrics.json"), "w"), indent=2, default=float)
PT.to_csv(os.path.join(OUT, "pseudobulk_primary.csv"), index=False)
log(f"\n[6/6] saved tables -> {OUT}")
