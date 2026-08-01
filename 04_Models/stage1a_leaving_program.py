"""
STAGE 1A - Intra-PT "leaving program" score (laptop, no GPU).

Defines the PRIMARY metastatic-propensity target locked in REVIEW_PLAN.md:
a confound-free, *within-PT* EMT / partial-EMT / invasion ("leaving") program,
scored per PT spot.  This continuous score becomes the Phase B regression target.

Why this and not mu_HM:  Stage 0 showed the cohort-wide mu_HM - mu_PT axis is a
tissue-of-residence (liver) confound and does not generalize across the 2 HM
patients.  The leaving program is defined entirely inside the 4 PT slides, so it
carries no HM/liver confound.

Method
------
- Per PT sample: CP10k library-size normalize -> log1p.
- AddModuleScore (Seurat-style): mean(signature) - mean(expression-binned control
  genes), so the score is decorrelated from sequencing depth / spot complexity.
- CORE (training) signature = EMT TFs + mesenchymal/invasion markers + ECM panel.
- HELD-OUT validators (VIM, SNAI2, PRRX1, MMP9, MMP14, POSTN, SPARC) are scored
  the SAME way but NEVER used to build the core score -> independent Stage-4 check.
- PCA axis on the core genes (per sample) as an unsupervised cross-check.

Diagnostics / decision gates
----------------------------
- Spatial coherence: hex-neighbour Moran's I per slide (high spots must cluster,
  not scatter randomly).
- Generalization: core score must correlate with the held-out validator score.
- Confound audit: core score must NOT be a CAF/stromal or hepatocyte readout
  (report corr with RCTD myCAF+iCAF and hepatocyte fractions); also report the
  tumour-dominated-spot-only (RCTD Tumor frac >= 0.5) restriction.

Run:
    "C:/Users/datai/anaconda3/envs/tcga/python.exe" stage1a_leaving_program.py

Outputs -> Outputs/stage1a_leaving_program/
    leaving_program_scores.csv  (the Phase B target), metrics.json, summary.txt,
    heatmap_<sample>.png, score_distribution.png, module_vs_pca.png,
    core_vs_heldout.png, confound_bars.png
"""

import os, json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ----------------------------------------------------------------------------- paths
from _cohort import (ROOT, RCTD_DIR, PT_SAMPLES, HEPATOCYTE_IDX, ICAF_IDX,
                     MYCAF_IDX, TUMOR_IDX, RESID_ON, out_dir, banner, boxplot,
                     load_spot_embeddings)

COUNTS_DIR= os.path.join(ROOT, "dataset", "ST", "scVI_counts")
QC_CSV    = os.path.join(ROOT, "Outputs", "Patient-Sample-Information", "spot_qc_mask.csv")
OUT_DIR   = out_dir("stage1a_leaving_program")
banner("STAGE 1A - leaving program")

TUMOR_THRESH   = 0.50

# ----------------------------------------------------------------------------- gene sets
# CORE = training signature for the leaving program (EMT / partial-EMT / invasion + ECM).
EMT_CORE = ["SNAI1", "ZEB1", "ZEB2", "CDH2", "S100A4", "TGFBR1", "TGFB1",
            "MMP1", "LOXL2", "ITGB6", "LAMC2", "SERPINE1"]
ECM_PANEL = ["COL1A1", "COL3A1", "COL5A1", "FN1", "LOX", "TNC", "TIMP1"]
CORE = EMT_CORE + ECM_PANEL
# HELD-OUT validators (Stage 4) - scored but NEVER used to build the core score.
HELDOUT = ["VIM", "SNAI2", "PRRX1", "MMP9", "MMP14", "POSTN", "SPARC"]

RNG = np.random.default_rng(0)
N_BINS = 25          # expression bins for AddModuleScore control matching
N_CTRL = 100         # control genes drawn per signature gene

# ----------------------------------------------------------------------------- helpers
def add_module_score(lognorm, all_genes, sig_genes, gene_avg, bins):
    """Seurat AddModuleScore: mean(signature) - mean(expression-matched controls).
    lognorm: (n_spots, n_genes) log-normalized.  Returns (n_spots,) score."""
    gidx = {g: i for i, g in enumerate(all_genes)}
    sig = [g for g in sig_genes if g in gidx]
    sig_i = [gidx[g] for g in sig]
    # control pool: for each signature gene, sample N_CTRL genes from its expr bin
    ctrl_i = set()
    for g in sig:
        b = bins[gidx[g]]
        pool = np.where(bins == b)[0]
        pool = pool[~np.isin(pool, sig_i)]          # exclude signature genes
        if len(pool) == 0:
            continue
        take = RNG.choice(pool, size=min(N_CTRL, len(pool)), replace=False)
        ctrl_i.update(take.tolist())
    ctrl_i = np.array(sorted(ctrl_i))
    sig_mean = lognorm[:, sig_i].mean(axis=1)
    ctrl_mean = lognorm[:, ctrl_i].mean(axis=1) if len(ctrl_i) else 0.0
    return sig_mean - ctrl_mean, sig


def hex_morans_I(score, rc, n_perm=999):
    """Moran's I on the Visium hex lattice. rc = array of (row,col). Binary
    row-standardized adjacency over the 6 hex neighbours.  Returns (I, p_perm)."""
    pos = {(int(r), int(c)): i for i, (r, c) in enumerate(rc)}
    offs = [(0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    rows, cols = [], []
    for (r, c), i in pos.items():
        for dr, dc in offs:
            j = pos.get((r + dr, c + dc))
            if j is not None:
                rows.append(i); cols.append(j)
    n = len(score)
    if not rows:
        return float("nan"), float("nan")
    rows = np.array(rows); cols = np.array(cols)
    z = score - score.mean()
    denom = (z ** 2).sum()
    W = len(rows)

    def I_of(zv):
        return (n / W) * (zv[rows] * zv[cols]).sum() / denom

    I_obs = I_of(z)
    perm = np.empty(n_perm)
    for k in range(n_perm):
        zp = RNG.permutation(z)
        perm[k] = (n / W) * (zp[rows] * zp[cols]).sum() / (zp ** 2).sum()
    p = (1 + (perm >= I_obs).sum()) / (n_perm + 1)
    return float(I_obs), float(p)


# ----------------------------------------------------------------------------- load QC map (barcode -> patch_stem)
qc = pd.read_csv(QC_CSV)
qc = qc[qc["sample"].isin(PT_SAMPLES)]
bc2stem = {(s, b): st for s, b, st in zip(qc["sample"], qc["barcode"], qc["patch_stem"])}

# ----------------------------------------------------------------------------- per-sample scoring
all_rows = []
results = {"core_genes": CORE, "heldout_genes": HELDOUT, "per_sample": {}}

for sample in PT_SAMPLES:
    path = os.path.join(COUNTS_DIR, sample + ".csv")
    print(f"[{sample}] reading {path} ...")
    mat = pd.read_csv(path, index_col=0)                 # genes x spots
    mat = mat[~mat.index.duplicated(keep="first")]
    genes = mat.index.to_numpy()
    barcodes = mat.columns.to_numpy()
    counts = mat.to_numpy(dtype=np.float64).T            # (n_spots, n_genes)
    del mat

    # CP10k + log1p
    libsize = counts.sum(axis=1, keepdims=True)
    libsize[libsize == 0] = 1.0
    lognorm = np.log1p(counts / libsize * 1e4)

    # expression bins for control matching
    gene_avg = lognorm.mean(axis=0)
    ranks = pd.qcut(pd.Series(gene_avg).rank(method="first"), N_BINS, labels=False).to_numpy()

    score_core, used_core = add_module_score(lognorm, genes, CORE, gene_avg, ranks)
    score_emt,  _         = add_module_score(lognorm, genes, EMT_CORE, gene_avg, ranks)
    score_ecm,  _         = add_module_score(lognorm, genes, ECM_PANEL, gene_avg, ranks)
    score_hout, used_hout = add_module_score(lognorm, genes, HELDOUT, gene_avg, ranks)

    # unsupervised cross-check: PC1 of z-scored core genes, oriented by core module score
    gidx = {g: i for i, g in enumerate(genes)}
    core_i = [gidx[g] for g in used_core]
    Xc = lognorm[:, core_i]
    Xc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-9)
    pc1 = PCA(n_components=1, random_state=0).fit_transform(Xc)[:, 0]
    if np.corrcoef(pc1, score_core)[0, 1] < 0:
        pc1 = -pc1

    # metadata + RCTD fractions.  One consolidated read of the sample's cell
    # embeddings instead of a torch.load per spot (see _cohort.load_spot_embeddings).
    stems, rows_, cols_ = [], [], []
    for b in barcodes:
        st = bc2stem.get((sample, b))
        stems.append(st)
        if st is None:
            rows_.append(-1); cols_.append(-1)
        else:
            parts = st.split("_")
            rows_.append(int(parts[-2])); cols_.append(int(parts[-1]))

    V, has_rctd = load_spot_embeddings(
        RCTD_DIR, sample, [st if st is not None else "" for st in stems])
    has_rctd &= np.array([st is not None for st in stems])
    # Widen to float64 to match the per-spot `float(v[i])` these replaced.  Under
    # NumPy 2's NEP-50 rules a weak np.nan scalar does NOT promote the float32
    # tensor, so without the cast these columns serialise at float32 precision.
    # The CAF sum is taken in float32 first, as `float(v[i] + v[j])` did.
    tumor_f = np.where(has_rctd, V[:, TUMOR_IDX].astype(np.float64), np.nan)
    hep_f   = np.where(has_rctd, V[:, HEPATOCYTE_IDX].astype(np.float64), np.nan)
    caf_f   = np.where(has_rctd, (V[:, ICAF_IDX] + V[:, MYCAF_IDX]).astype(np.float64), np.nan)

    sdf = pd.DataFrame({
        "patch_stem": stems, "sample": sample, "barcode": barcodes,
        "row": rows_, "col": cols_,
        "tumor_frac": tumor_f, "hepatocyte_frac": hep_f, "caf_frac": caf_f,
        "score_core": score_core, "score_emt": score_emt, "score_ecm": score_ecm,
        "score_pca": pc1, "score_heldout": score_hout,
    })
    # within-sample standardized core score = the comparable Phase B target
    sdf["leaving_score"] = (sdf["score_core"] - sdf["score_core"].mean()) / (sdf["score_core"].std() + 1e-9)

    # Residualized target.  RESID_ON="tumor" removes malignant abundance only (the
    # original design, motivated by corr(core, tumor_frac) ~0.5-0.76 measured under
    # OUR RCTD).  Under the paper's RCTD that coupling is only -0.158 while
    # corr(core, CAF) = +0.478, so RESID_ON="tumor_caf" additionally removes stroma
    # -- otherwise the "confound-free" score is largely a desmoplasia detector.
    tf = sdf["tumor_frac"].to_numpy()
    cf = sdf["caf_frac"].to_numpy()
    if RESID_ON == "tumor_caf":
        okf = np.isfinite(tf) & np.isfinite(cf)
        cols = lambda m: [np.ones(m.sum()), tf[m], cf[m]]
    else:
        okf = np.isfinite(tf)
        cols = lambda m: [np.ones(m.sum()), tf[m]]
    resid = np.full(len(sdf), np.nan)
    if okf.sum() > 10:
        A = np.column_stack(cols(okf))
        beta, *_ = np.linalg.lstsq(A, sdf["score_core"].to_numpy()[okf], rcond=None)
        resid[okf] = sdf["score_core"].to_numpy()[okf] - A @ beta
        rstd = np.nanstd(resid)
        resid = (resid - np.nanmean(resid)) / (rstd + 1e-9)
    sdf["leaving_score_resid"] = resid

    # residualize the HELD-OUT score on tumour frac the same way, so the validation
    # compares within-tumour EMT parts (both freed of the shared abundance gradient).
    hresid = np.full(len(sdf), np.nan)
    if okf.sum() > 10:
        bh, *_ = np.linalg.lstsq(A, sdf["score_heldout"].to_numpy()[okf], rcond=None)
        hresid[okf] = sdf["score_heldout"].to_numpy()[okf] - A @ bh
    sdf["heldout_resid"] = hresid
    all_rows.append(sdf)

    # ---- per-sample metrics
    have_rc = sdf["row"] >= 0
    rc = sdf.loc[have_rc, ["row", "col"]].to_numpy()
    I_all, p_all = hex_morans_I(sdf.loc[have_rc, "score_core"].to_numpy(), rc)
    tmask = sdf["tumor_frac"] >= TUMOR_THRESH
    rc_t = sdf.loc[have_rc & tmask, ["row", "col"]].to_numpy()
    if (have_rc & tmask).sum() > 50:
        I_t, p_t = hex_morans_I(sdf.loc[have_rc & tmask, "score_core"].to_numpy(), rc_t)
    else:
        I_t, p_t = float("nan"), float("nan")
    # residualized-target spatial coherence (must survive removing the abundance gradient)
    rmask = have_rc & np.isfinite(sdf["leaving_score_resid"])
    if rmask.sum() > 50:
        I_r, p_r = hex_morans_I(sdf.loc[rmask, "leaving_score_resid"].to_numpy(),
                                sdf.loc[rmask, ["row", "col"]].to_numpy())
    else:
        I_r, p_r = float("nan"), float("nan")

    def corr(a, b, m=None):
        x = sdf[a].to_numpy(); y = sdf[b].to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        if m is not None: ok &= m.to_numpy()
        return float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 10 else float("nan")

    results["per_sample"][sample] = {
        "n_spots": int(len(sdf)),
        "n_tumor_dominated": int(tmask.sum()),
        "morans_I_core": I_all, "morans_p_core": p_all,
        "morans_I_core_tumoronly": I_t, "morans_p_core_tumoronly": p_t,
        "morans_I_resid": I_r, "morans_p_resid": p_r,
        "corr_core_vs_heldout": corr("score_core", "score_heldout"),
        "corr_core_vs_heldout_tumoronly": corr("score_core", "score_heldout", tmask),
        "corr_resid_vs_heldoutresid": corr("leaving_score_resid", "heldout_resid"),
        "corr_resid_vs_tumorfrac": corr("leaving_score_resid", "tumor_frac"),
        "corr_core_vs_pca": corr("score_core", "score_pca"),
        "corr_core_vs_caf": corr("score_core", "caf_frac"),
        "corr_emt_vs_caf": corr("score_emt", "caf_frac"),
        "corr_ecm_vs_caf": corr("score_ecm", "caf_frac"),
        "corr_core_vs_hepatocyte": corr("score_core", "hepatocyte_frac"),
        "corr_core_vs_tumorfrac": corr("score_core", "tumor_frac"),
        "missing_core_genes": [g for g in CORE if g not in genes],
    }
    print(f"  Moran's I(core)={I_all:.3f} p={p_all:.3f} | core~heldout="
          f"{results['per_sample'][sample]['corr_core_vs_heldout']:.3f} | "
          f"core~CAF={results['per_sample'][sample]['corr_core_vs_caf']:.3f}")

scores = pd.concat(all_rows, ignore_index=True)
scores.to_csv(os.path.join(OUT_DIR, "leaving_program_scores.csv"), index=False)

# ----------------------------------------------------------------------------- pooled metrics
def pooled_corr(a, b, mask=None):
    x = scores[a].to_numpy(); y = scores[b].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    if mask is not None: ok &= mask
    return float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 10 else float("nan")

tmask_all = (scores["tumor_frac"] >= TUMOR_THRESH).to_numpy()
results["pooled"] = {
    "n_PT_spots": int(len(scores)),
    "corr_core_vs_heldout": pooled_corr("score_core", "score_heldout"),
    "corr_core_vs_heldout_tumoronly": pooled_corr("score_core", "score_heldout", tmask_all),
    "corr_resid_vs_heldoutresid": pooled_corr("leaving_score_resid", "heldout_resid"),
    "corr_resid_vs_tumorfrac": pooled_corr("leaving_score_resid", "tumor_frac"),
    "corr_core_vs_tumorfrac": pooled_corr("score_core", "tumor_frac"),
    "corr_core_vs_pca": pooled_corr("score_core", "score_pca"),
    "corr_core_vs_caf": pooled_corr("score_core", "caf_frac"),
    "corr_emt_vs_caf": pooled_corr("score_emt", "caf_frac"),
    "corr_ecm_vs_caf": pooled_corr("score_ecm", "caf_frac"),
    "corr_core_vs_hepatocyte": pooled_corr("score_core", "hepatocyte_frac"),
}

# ----------------------------------------------------------------------------- plots
# 1) per-slide spatial heatmaps: raw leaving score vs tumour-residualized target
for sample in PT_SAMPLES:
    s = scores[(scores["sample"] == sample) & (scores["row"] >= 0)]
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    for ax, col, title in [(axs[0], "leaving_score", "leaving score (z)"),
                           (axs[1], "leaving_score_resid", "tumour-residualized (target)")]:
        sc = ax.scatter(s["col"], -s["row"], c=s[col], cmap="magma", s=14)
        ax.set_title(f"{sample}  {title}"); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(sc, ax=ax, fraction=0.046)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, f"heatmap_{sample}.png"), dpi=130); plt.close()

# 2) score distribution per sample
fig, ax = plt.subplots(figsize=(8, 4))
data = [scores.loc[scores["sample"] == s, "score_core"].values for s in PT_SAMPLES]
boxplot(ax, data, PT_SAMPLES, patch_artist=True)
ax.set_ylabel("core leaving-program score"); ax.set_title("Leaving-program score by PT sample")
plt.xticks(rotation=20, ha="right"); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "score_distribution.png"), dpi=130); plt.close()

# 3) module vs PCA
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(scores["score_pca"], scores["score_core"], s=4, alpha=0.3)
ax.set_xlabel("PC1 (core genes)"); ax.set_ylabel("AddModuleScore (core)")
ax.set_title(f"module vs unsupervised  r={results['pooled']['corr_core_vs_pca']:.2f}")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "module_vs_pca.png"), dpi=130); plt.close()

# 4) core vs held-out validators
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(scores["score_heldout"], scores["score_core"], s=4, alpha=0.3, color="#2a9d8f")
ax.set_xlabel("held-out validator score"); ax.set_ylabel("core leaving score")
ax.set_title(f"core vs HELD-OUT  r={results['pooled']['corr_core_vs_heldout']:.2f}")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "core_vs_heldout.png"), dpi=130); plt.close()

# 5) confound bars (per sample corr with CAF / hepatocyte / heldout)
fig, ax = plt.subplots(figsize=(9, 4.5))
labels = PT_SAMPLES
metrics = ["corr_core_vs_heldout", "corr_core_vs_caf", "corr_core_vs_hepatocyte"]
colors = ["#2a9d8f", "#e76f51", "#264653"]
x = np.arange(len(labels)); w = 0.25
for k, (m, c) in enumerate(zip(metrics, colors)):
    vals = [results["per_sample"][s][m] for s in labels]
    ax.bar(x + (k - 1) * w, vals, w, label=m.replace("corr_core_vs_", ""), color=c)
ax.axhline(0, color="k", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylabel("Pearson r with core score"); ax.legend(); ax.set_title("Generalization (heldout) vs confounds (CAF/hepatocyte)")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "confound_bars.png"), dpi=130); plt.close()

# ----------------------------------------------------------------------------- save + verdict
with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
    json.dump(results, f, indent=2)

P = results["pooled"]
ms = [results["per_sample"][s]["morans_I_core"] for s in PT_SAMPLES]
ps = [results["per_sample"][s]["morans_p_core"] for s in PT_SAMPLES]
v = []
v.append("STAGE 1A - INTRA-PT LEAVING-PROGRAM TARGET - VERDICT\n" + "=" * 52)
v.append(f"PT spots scored: {len(scores)}  (4 slides: {', '.join(PT_SAMPLES)})")
v.append(f"Core signature ({len(CORE)} genes): {', '.join(CORE)}")
v.append(f"Held-out validators ({len(HELDOUT)}): {', '.join(HELDOUT)}")
v.append("")
v.append("GATE 1 - spatial coherence (Moran's I of core score per slide):")
for s, i, p in zip(PT_SAMPLES, ms, ps):
    v.append(f"   {s:12s} I={i:.3f}  p={p:.3f}")
v.append("   -> want POSITIVE & significant (high spots cluster at fronts, not random).")
v.append("")
v.append("GATE 2 - generalization to HELD-OUT EMT genes (pooled):")
v.append(f"   corr(core, held-out) = {P['corr_core_vs_heldout']:.3f}  "
         f"(tumour-only {P['corr_core_vs_heldout_tumoronly']:.3f})")
v.append(f"   corr(core, PCA axis) = {P['corr_core_vs_pca']:.3f}  (robustness cross-check)")
v.append("   -> want clearly POSITIVE: core captures the generalizable EMT axis.")
v.append("")
v.append("CONFOUND AUDIT (pooled):")
v.append(f"   corr(core, tumor frac)      = {P['corr_core_vs_tumorfrac']:.3f}  <- abundance coupling")
v.append(f"   corr(core, CAF frac)        = {P['corr_core_vs_caf']:.3f}")
v.append(f"   corr(EMT-only, CAF frac)    = {P['corr_emt_vs_caf']:.3f}")
v.append(f"   corr(ECM-only, CAF frac)    = {P['corr_ecm_vs_caf']:.3f}")
v.append(f"   corr(core, hepatocyte frac) = {P['corr_core_vs_hepatocyte']:.3f}")
v.append("")
v.append("TUMOUR-INTRINSIC TARGET (core residualized on tumor frac, per sample):")
v.append(f"   corr(resid, tumor frac)      = {P['corr_resid_vs_tumorfrac']:.3f}  (~0 by construction)")
v.append(f"   corr(resid, held-out resid)  = {P['corr_resid_vs_heldoutresid']:.3f}  (within-tumour EMT agreement)")
rI = [results['per_sample'][s]['morans_I_resid'] for s in PT_SAMPLES]
v.append("   Moran's I (resid) per slide: " +
         ", ".join(f"{s.split('_')[-1]}={i:.2f}" for s, i in zip(PT_SAMPLES, rI)))
v.append("")
v.append("TARGET saved: Outputs/stage1a_leaving_program/leaving_program_scores.csv")
v.append("   'leaving_score'       = within-sample z of core AddModuleScore (abundance-coupled)")
v.append("   'leaving_score_resid' = tumour-fraction-residualized = RECOMMENDED Phase B target")
txt = "\n".join(v)
open(os.path.join(OUT_DIR, "summary.txt"), "w").write(txt)
print("\n" + txt + f"\n\nSaved -> {OUT_DIR}")
