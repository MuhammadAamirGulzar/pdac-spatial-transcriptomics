"""
STAGE 9 -- uncertainty and nulls for the site-specificity claim.

Why this exists
---------------
Stage 6 reports cos(HM-shift, LNM-shift) = +0.547 and reads it as "~55% of the
metastatic change is shared".  Stage 7 reports a mean cosine of -0.152 in the
replication cohort.  Both are single point estimates with no interval and no
null, and "55% shared" is not interpretable without knowing what the cosine
looks like when site does NOT matter.

The null is emphatically NOT zero.  Both shifts are measured against the same
primary-tumour centroid, so they share a "becoming a metastasis" component by
construction.  If liver and node deposits ran the *same* program, the cosine
would sit near +1, limited only by sampling noise.  The question is therefore
whether the observed cosine is significantly BELOW what shared-program-plus-
noise would produce -- a lower-tail test, not an upper-tail one.

Two further problems this script addresses:

  * SCALE.  Stage 6 computes the cosine on RAW features.  FGES signature scores
    run to ~5000 while RCTD fractions are ~0.005, so the raw cosine is dominated
    by a handful of large-magnitude signatures and the 15 cell-type features
    contribute almost nothing.  Stage 7 divides by the feature SD and so does
    not have this problem -- which also means +0.547 and -0.152 were never
    computed the same way.  Both conventions are reported here.

  * PSEUDO-REPLICATION.  Stage 6's differential test is an unpaired
    Mann-Whitney over SECTIONS, but PT_2 contributes two HM sections and the
    other patients one.  Sections within a patient are not independent.  This
    script re-runs the differential with the PATIENT as the unit.

Everything here reads only committed CSVs in Outputs/ -- no dataset/, no GPU.

    python 04_Models/stage9_cosine_nulls.py
"""

import itertools
import json
import math
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S6 = os.path.join(ROOT, "Outputs", "stage6_site_program")
S7 = os.path.join(ROOT, "Outputs", "stage7_external_replication")
OUT = os.path.join(ROOT, "Outputs", "stage9_cosine_nulls")
os.makedirs(OUT, exist_ok=True)

RNG = np.random.default_rng(20260829)
N_BOOT = 10000
MAX_EXACT = 60000        # enumerate label assignments exactly below this many

LOG = []


def log(m=""):
    print(m)
    LOG.append(m)


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def cosine_of_shifts(met, prim, feat, sites, sd=None):
    """cos between the two site-vs-primary shift vectors.

    met  : DataFrame of metastatic sections with a 'site' column
    prim : DataFrame of primary sections
    sd   : per-feature scale divisor, or None for stage 6's raw convention
    """
    p = prim[feat].mean().to_numpy(float)
    a = met[met.site == sites[0]][feat].mean().to_numpy(float) - p
    b = met[met.site == sites[1]][feat].mean().to_numpy(float) - p
    if sd is not None:
        a, b = a / sd, b / sd
    return float(unit(a) @ unit(b))


def label_permutation_null(met, prim, feat, sites, sd=None, seed=0):
    """Distribution of the cosine when site labels are meaningless.

    Shuffles the site labels among the metastatic sections while holding the
    group SIZES fixed, so every draw is size-matched to the observed contrast.
    Enumerated exhaustively when the number of assignments is small.
    """
    sub = met[met.site.isin(sites)].reset_index(drop=True)
    n, k = len(sub), int((sub.site == sites[0]).sum())
    n_assign = math.comb(n, k)
    idx = np.arange(n)
    if 0 < n_assign <= MAX_EXACT:
        combos = itertools.combinations(idx, k)
        exact = True
    else:
        rg = np.random.default_rng(seed)
        combos = (rg.permutation(n)[:k] for _ in range(20000))
        exact = False
    out = []
    for c in combos:
        lab = np.full(n, sites[1], dtype=object)
        lab[list(c)] = sites[0]
        s = sub.copy()
        s["site"] = lab
        out.append(cosine_of_shifts(s, prim, feat, sites, sd))
    return np.asarray(out), exact


def split_half_ceiling(met, prim, feat, site, sd=None):
    """Cosine between two disjoint halves of the SAME site -- the noise ceiling.

    If two halves of one site only agree at cosine c, then a cross-site cosine
    of c is indistinguishable from 'same program, measured noisily'.
    """
    sub = met[met.site == site].reset_index(drop=True)
    n = len(sub)
    if n < 4:
        return np.array([])
    k = n // 2
    out = []
    for c in itertools.combinations(range(n), k):
        c = list(c)
        rest = [i for i in range(n) if i not in c]
        p = prim[feat].mean().to_numpy(float)
        a = sub.iloc[c][feat].mean().to_numpy(float) - p
        b = sub.iloc[rest][feat].mean().to_numpy(float) - p
        if sd is not None:
            a, b = a / sd, b / sd
        out.append(float(unit(a) @ unit(b)))
    return np.asarray(out)


def patient_bootstrap(met, prim, feat, sites, sd=None, n=N_BOOT, seed=1):
    """Resample PATIENTS with replacement; recompute the cosine each time."""
    rg = np.random.default_rng(seed)
    pats = sorted(set(met.patient) | set(prim.patient))
    met_by, prim_by = dict(list(met.groupby("patient"))), dict(list(prim.groupby("patient")))
    out = []
    for _ in range(n):
        draw = rg.choice(pats, size=len(pats), replace=True)
        m = pd.concat([met_by[p] for p in draw if p in met_by], ignore_index=True) \
            if any(p in met_by for p in draw) else pd.DataFrame(columns=met.columns)
        t = pd.concat([prim_by[p] for p in draw if p in prim_by], ignore_index=True) \
            if any(p in prim_by for p in draw) else pd.DataFrame(columns=prim.columns)
        if len(t) < 2 or (m.site == sites[0]).sum() < 2 or (m.site == sites[1]).sum() < 2:
            continue
        out.append(cosine_of_shifts(m, t, feat, sites, sd))
    return np.asarray(out)


def bh(p):
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    q = np.empty(m)
    q[o] = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)


# =============================================================== DISCOVERY
log("=" * 78)
log("STAGE 9 -- nulls and uncertainty for the site-specificity claim")
log("=" * 78)

PB = pd.read_csv(os.path.join(S6, "pseudobulk_sections.csv"))
PT = pd.read_csv(os.path.join(S6, "pseudobulk_primary.csv"))
FEAT = [c for c in PB.columns if c not in ("sample", "patient", "site", "n_spots")]
FEAT = [c for c in FEAT if c in PT.columns]

log(f"\n[1/5] discovery cohort: {(PB.site=='HM').sum()} HM + {(PB.site=='LNM').sum()} LNM "
    f"metastatic sections, {len(PT)} primary, {len(FEAT)} features")

SD = pd.concat([PT[FEAT], PB[FEAT]], ignore_index=True).std().to_numpy(float) + 1e-9

log("\n      feature magnitude spread (why scaling matters):")
mag = PB[FEAT].abs().mean().sort_values()
log(f"        smallest: {mag.index[0]:34s} {mag.iloc[0]:12.5f}")
log(f"        largest : {mag.index[-1]:34s} {mag.iloc[-1]:12.5f}")
log(f"        ratio   : {mag.iloc[-1]/max(mag.iloc[0],1e-12):,.0f}x  -> an unscaled cosine is "
    f"dominated by the largest features")

results = {"discovery": {}, "replication": {}}

for tag, sd in [("raw (as published in stage 6)", None), ("SD-scaled (stage 7 convention)", SD)]:
    obs = cosine_of_shifts(PB, PT, FEAT, ("HM", "LNM"), sd)
    null, exact = label_permutation_null(PB, PT, FEAT, ("HM", "LNM"), sd, seed=2)
    boot = patient_bootstrap(PB, PT, FEAT, ("HM", "LNM"), sd, seed=3)
    ceil_hm = split_half_ceiling(PB, PT, FEAT, "HM", sd)
    ceil_ln = split_half_ceiling(PB, PT, FEAT, "LNM", sd)
    p_lower = float((null <= obs).mean())
    lo, hi = (np.percentile(boot, [2.5, 97.5]) if len(boot) else (np.nan, np.nan))

    log(f"\n[2/5] cosine(HM-shift, LNM-shift) -- {tag}")
    log(f"      observed                          {obs:+.3f}")
    log(f"      patient bootstrap 95% CI          [{lo:+.3f}, {hi:+.3f}]   ({len(boot)} usable draws)")
    log(f"      site-label null ({'exact, ' if exact else 'sampled, '}{len(null)} assignments)")
    log(f"        median {np.median(null):+.3f}   2.5-97.5% [{np.percentile(null,2.5):+.3f}, "
        f"{np.percentile(null,97.5):+.3f}]")
    log(f"        P(null <= observed) = {p_lower:.4f}   <- lower-tail test: is the cosine "
        f"SMALLER than if site were irrelevant?")
    crit = float(np.percentile(null, 5))
    log(f"      DETECTION LIMIT: at these group sizes the cosine must fall below {crit:+.3f} "
        f"to reach one-sided p<0.05")
    log("        " + ("-> observed sits INSIDE the null; this cohort cannot resolve it"
                      if obs > crit else "-> observed clears it"))
    if len(ceil_hm):
        log(f"      within-HM  split-half ceiling      median {np.median(ceil_hm):+.3f}  "
            f"[{ceil_hm.min():+.3f}, {ceil_hm.max():+.3f}]  ({len(ceil_hm)} splits)")
    if len(ceil_ln):
        log(f"      within-LNM split-half ceiling      median {np.median(ceil_ln):+.3f}  "
            f"[{ceil_ln.min():+.3f}, {ceil_ln.max():+.3f}]  ({len(ceil_ln)} splits)")

    if sd is None:      # save the raw-convention arrays for the figure script
        np.savez(os.path.join(OUT, "discovery_null_arrays.npz"),
                 null=null, boot=boot, ceil_hm=ceil_hm, ceil_ln=ceil_ln, observed=obs)
    results["discovery"][tag] = dict(
        observed=obs, boot_ci=[float(lo), float(hi)], n_boot=len(boot),
        null_median=float(np.median(null)), null_lo=float(np.percentile(null, 2.5)),
        null_hi=float(np.percentile(null, 97.5)), null_exact=bool(exact), n_null=len(null),
        detection_limit_p05=crit,
        p_lower_tail=p_lower,
        ceiling_HM_median=float(np.median(ceil_hm)) if len(ceil_hm) else None,
        ceiling_LNM_median=float(np.median(ceil_ln)) if len(ceil_ln) else None)

# =============================================== patient-level differential
log("\n[3/5] differential test with the PATIENT as the unit (was: section)")
PBp = PB.groupby(["patient", "site"], as_index=False)[FEAT].mean()
hm_p = PBp[PBp.site == "HM"]
ln_p = PBp[PBp.site == "LNM"]
paired = sorted(set(hm_p.patient) & set(ln_p.patient))
log(f"      HM patients {len(hm_p)}   LNM patients {len(ln_p)}   paired {len(paired)} {paired}")
log(f"      (stage 6 tested {(PB.site=='HM').sum()} vs {(PB.site=='LNM').sum()} SECTIONS -- "
    f"PT_2 contributed 2 HM sections)")

rows = []
for c in FEAT:
    a = ln_p[c].to_numpy(float)
    b = hm_p[c].to_numpy(float)
    d = (a.mean() - b.mean()) / (np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) + 1e-12)
    u = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
    pa = np.array([ln_p[ln_p.patient == p][c].iloc[0] for p in paired])
    pb = np.array([hm_p[hm_p.patient == p][c].iloc[0] for p in paired])
    n_up = int((pa > pb).sum())
    sign_p = stats.binomtest(n_up, len(paired), 0.5).pvalue if len(paired) else np.nan
    rows.append(dict(feature=c, mean_HM=b.mean(), mean_LNM=a.mean(),
                     cohens_d_patient=d, p_unpaired_patient=u,
                     n_paired_LNM_higher=n_up, n_paired=len(paired), p_sign=sign_p))
D = pd.DataFrame(rows)
D["q_unpaired_patient"] = bh(D.p_unpaired_patient)
D = D.sort_values("cohens_d_patient", ascending=False)
D.to_csv(os.path.join(OUT, "site_differential_patient_level.csv"), index=False)

pub = pd.read_csv(os.path.join(S6, "site_differential.csv"))[
    ["feature", "cohens_d", "p_unpaired", "q_unpaired"]].rename(
    columns={"cohens_d": "d_section", "p_unpaired": "p_section", "q_unpaired": "q_section"})
CMP = D.merge(pub, on="feature")
CMP.to_csv(os.path.join(OUT, "differential_section_vs_patient.csv"), index=False)

log(f"      features with q<0.10  --  section-level {(CMP.q_section<0.10).sum()}   "
    f"patient-level {(CMP.q_unpaired_patient<0.10).sum()}")
log(f"\n      {'feature':34s} {'d_sect':>7s} {'q_sect':>7s} {'d_pat':>7s} {'q_pat':>7s}  paired")
for _, r in CMP.head(6).iterrows():
    log(f"      {r.feature:34s} {r.d_section:+7.2f} {r.q_section:7.3f} "
        f"{r.cohens_d_patient:+7.2f} {r.q_unpaired_patient:7.3f}  "
        f"{r.n_paired_LNM_higher}/{r.n_paired}")
for _, r in CMP.tail(3).iloc[::-1].iterrows():
    log(f"      {r.feature:34s} {r.d_section:+7.2f} {r.q_section:7.3f} "
        f"{r.cohens_d_patient:+7.2f} {r.q_unpaired_patient:7.3f}  "
        f"{r.n_paired_LNM_higher}/{r.n_paired}")

bc = CMP[CMP.feature == "rctd::B cells"].iloc[0]
log(f"\n      B cells: section-level d={bc.d_section:+.2f} q={bc.q_section:.3f}  ->  "
    f"patient-level d={bc.cohens_d_patient:+.2f} q={bc.q_unpaired_patient:.3f}, "
    f"sign test p={bc.p_sign:.3f} ({bc.n_paired_LNM_higher}/{bc.n_paired})")

results["discovery"]["differential"] = dict(
    n_hm_patients=int(len(hm_p)), n_lnm_patients=int(len(ln_p)), paired=paired,
    n_sig_section=int((CMP.q_section < 0.10).sum()),
    n_sig_patient=int((CMP.q_unpaired_patient < 0.10).sum()),
    bcells=dict(d_section=float(bc.d_section), q_section=float(bc.q_section),
                d_patient=float(bc.cohens_d_patient),
                q_patient=float(bc.q_unpaired_patient),
                sign_p=float(bc.p_sign),
                n_paired_higher=int(bc.n_paired_LNM_higher), n_paired=int(bc.n_paired)))

# ============================================================ REPLICATION
log("\n[4/5] replication cohort (GSE274557)")
PB7 = pd.read_csv(os.path.join(S7, "pseudobulk_sections.csv"))
# stage 7 excludes "Epithelial": it is the tumour-purity gate variable, not a feature
F7 = [c for c in PB7.columns if c not in ("sample", "patient", "site", "Epithelial")]
P7 = PB7[PB7.site == "T"]
M7 = PB7[PB7.site != "T"]
SD7 = PB7[F7].std().to_numpy(float) + 1e-9
log(f"      {len(P7)} primary + " + ", ".join(
    f"{(M7.site==s).sum()} {s}" for s in ["HM", "LuM", "PM"]) + f"   {len(F7)} features")

rep = {}
log(f"\n      {'pair':14s} {'obs':>7s} {'boot 95% CI':>20s} {'null median':>12s} {'P(null<=obs)':>13s}")
for a, b in [("HM", "LuM"), ("HM", "PM"), ("LuM", "PM")]:
    obs = cosine_of_shifts(M7, P7, F7, (a, b), SD7)
    null, exact = label_permutation_null(M7, P7, F7, (a, b), SD7, seed=4)
    boot = patient_bootstrap(M7, P7, F7, (a, b), SD7, seed=5)
    lo, hi = (np.percentile(boot, [2.5, 97.5]) if len(boot) else (np.nan, np.nan))
    p_low = float((null <= obs).mean())
    log(f"      {a+' vs '+b:14s} {obs:+7.3f}   [{lo:+.3f}, {hi:+.3f}]  {np.median(null):+12.3f} "
        f"{p_low:13.4f}")
    rep[f"{a}_vs_{b}"] = dict(observed=obs, boot_ci=[float(lo), float(hi)], n_boot=len(boot),
                              null_median=float(np.median(null)), null_exact=bool(exact),
                              n_null=len(null), p_lower_tail=p_low)
    np.savez(os.path.join(OUT, f"replication_null_{a}_vs_{b}.npz"), null=null, observed=obs)
results["replication"] = rep

# =================================================================== save
log("\n[5/5] writing outputs")
json.dump(results, open(os.path.join(OUT, "metrics.json"), "w"), indent=2, default=float)
open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8").write("\n".join(LOG) + "\n")
print(f"\nsaved -> {OUT}")
