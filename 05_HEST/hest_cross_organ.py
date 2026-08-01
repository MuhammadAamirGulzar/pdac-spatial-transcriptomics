"""
STAGE 8 -- does an H&E -> expression model transfer ACROSS ORGANS?

The question, and why it follows from our own data
--------------------------------------------------
Stage 6 (our cohort) and Stage 7 (GSE274557) both found that the transcriptional
program of a tumour depends on WHERE it is: liver and lymph-node metastases share
only ~55% of their shift from the primary, and in the treatment-naive replication
cohort the shifts at liver / lung / peritoneum are unrelated to opposite.

If tissue context dominates expression that strongly, then a model that predicts
expression from morphology should be substantially ORGAN-SPECIFIC too: train it on
one organ and it should degrade on another. That is a sharp, falsifiable
prediction, and our own cohort is far too small to test it (4 patients with H&E).

HEST-1k can: 157 human Visium cancer sections across 8 organs, every one shipping
224px H&E tiles at the spot coordinates.

Design
------
  features   UNI2-h (1536-d) per spot -- identical model/preprocessing to the rest
             of this project, so numbers are comparable
  target     CP10k + log1p expression of a COMMON gene panel (genes present in
             every sample, then the most variable among those)
  model      Ridge, fitted on spots, but every SPLIT IS BY SAMPLE -- spots inside
             one section are not independent, so a spot-level split would leak
  metric     per-gene Pearson r on held-out sections, averaged over genes

  WITHIN-ORGAN   leave-one-sample-out inside an organ
  CROSS-ORGAN    train on all samples of organ A, test on all of organ B

A model that has learned general morphology->expression rules scores similarly
both ways.  A model that has learned organ-specific rules collapses off-diagonal.

Run (after hest_embed.py):
    python 05_HEST/hest_cross_organ.py
"""

import json
import os
import sys

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEST = os.path.join(ROOT, "dataset", "external", "HEST")
EMB = os.path.join(HEST, "embeddings")
ST = os.path.join(HEST, "st")
OUT = os.path.join(ROOT, "Outputs", "stage8_hest_cross_organ")
os.makedirs(OUT, exist_ok=True)

N_GENES = 50          # HEST-Benchmark style: a modest, highly-variable panel
MIN_SAMPLES = 4       # an organ needs this many sections to be usable
ALPHA = 1000.0        # ridge strength; 1536-d features, few thousand spots


def log(m):
    print(m, flush=True)


def load_expression(sid):
    """spots x genes raw counts, barcodes, gene symbols.

    HEST is not uniform: X is a CSC/CSR group in most samples but a DENSE dataset
    in others (in which case there is no 'shape' attribute -- the dataset carries
    its own shape). Handle both, or the loader dies partway through the cohort.
    """
    with h5py.File(os.path.join(ST, f"{sid}.h5ad"), "r") as f:
        X = f["X"]
        if isinstance(X, h5py.Dataset):                       # dense
            M = sp.csr_matrix(np.asarray(X))
        else:                                                 # sparse group
            shape = tuple(X.attrs["shape"])
            enc = X.attrs.get("encoding-type", "csr_matrix")
            if isinstance(enc, bytes):
                enc = enc.decode()
            ctor = sp.csc_matrix if enc == "csc_matrix" else sp.csr_matrix
            M = ctor((X["data"][:], X["indices"][:], X["indptr"][:]), shape=shape)
        genes = np.array([g.decode() if isinstance(g, bytes) else str(g)
                          for g in f["var"]["_index"][:]])
        bc = np.array([b.decode() if isinstance(b, bytes) else str(b)
                       for b in f["obs"]["_index"][:]])
    # 10x references repeat gene symbols; keep the first occurrence so the symbol
    # index is unique and pandas get_indexer() works.
    _, first = np.unique(genes, return_index=True)
    keep = np.zeros(len(genes), bool)
    keep[first] = True
    M, genes = M.tocsc()[:, keep], genes[keep]
    return M.tocsr(), bc, genes


def main():
    meta = pd.read_csv(os.path.join(HEST, "HEST_v1_1_0.csv"), low_memory=False)
    idc = "id" if "id" in meta.columns else meta.columns[0]
    # A sample is usable only if BOTH its embedding and its expression file exist.
    # During a download the two arrive independently, so patches can be embedded
    # while st/<id>.h5ad is still missing -- guard against that partial state.
    emb = {f[:-len("_uni2h.npz")] for f in os.listdir(EMB) if f.endswith("_uni2h.npz")}
    stf = {f[:-len(".h5ad")] for f in os.listdir(ST) if f.endswith(".h5ad")}
    have = sorted(emb & stf)
    missing_st = sorted(emb - stf)
    if missing_st:
        log(f"      note: {len(missing_st)} embedded sample(s) have no expression file "
            f"yet (download still running?) -- excluded: {missing_st[:6]}")
    meta = meta[meta[idc].astype(str).isin(have)]
    # Platforms are NOT interchangeable: Xenium/Visium HD use targeted panels of a
    # few hundred genes, so intersecting them with Visium's ~20k collapses the
    # common panel (measured: 98 genes). Keep one platform.
    if "st_technology" in meta.columns:
        n0 = len(meta)
        meta = meta[meta["st_technology"].astype(str) == "Visium"]
        if len(meta) < n0:
            log(f"      dropped {n0-len(meta)} non-Visium section(s) (targeted panels)")
    have = sorted(set(meta[idc].astype(str)))
    log(f"[1/5] {len(have)} usable samples (embedding + expression, Visium)")
    log(meta.organ.value_counts().to_string())

    organs = [o for o, n in meta.organ.value_counts().items() if n >= MIN_SAMPLES]
    meta = meta[meta.organ.isin(organs)]
    log(f"\n      organs with >= {MIN_SAMPLES} sections: {organs}")

    # ---- common gene panel across every sample
    log("\n[2/5] building a common gene panel ...")
    common, per = None, {}
    for sid in meta[idc].astype(str):
        M, bc, genes = load_expression(sid)
        per[sid] = (M, bc, genes)
        common = set(genes) if common is None else (common & set(genes))
    common = np.array(sorted(common))
    log(f"      genes present in all {len(per)} samples: {len(common)}")

    # Choosing the panel by variance ACROSS organs would select organ-identity
    # markers (pancreatic enzymes, KLK3, ...). Those score ~0 on transfer simply
    # because they are not expressed in the other organ at all -- that measures
    # organ identity, not whether morphology->expression rules carry over.
    #
    # Instead take genes that vary WITHIN every organ: compute per-spot variance
    # inside each section, average within organ, then rank by the WEAKEST organ
    # (a min across organs). A gene only survives if it is informative everywhere,
    # so the cross-organ comparison is about transferable rules.
    org_of = {s: meta.loc[meta[idc].astype(str) == s, "organ"].iloc[0] for s in per}
    within_var = {o: [] for o in organs}
    mean_expr = {o: [] for o in organs}
    for sid, (M, bc, genes) in per.items():
        idx = pd.Index(genes).get_indexer(common)
        sub = M[:, idx]
        lib = np.asarray(M.sum(1)).ravel(); lib[lib == 0] = 1
        ln = sp.csr_matrix(sub.multiply(1e4 / lib[:, None])).log1p()
        d = np.asarray(ln.todense())
        within_var[org_of[sid]].append(d.var(0))
        mean_expr[org_of[sid]].append(d.mean(0))
    V = np.vstack([np.mean(within_var[o], 0) for o in organs])     # organs x genes
    E = np.vstack([np.mean(mean_expr[o], 0) for o in organs])
    score = V.min(0)                       # must vary in the WEAKEST organ too
    score[E.min(0) < 0.05] = -np.inf       # and be detectably expressed everywhere
    panel_idx = np.argsort(-score)[:N_GENES]
    panel = common[panel_idx]
    log(f"      panel = {N_GENES} genes variable WITHIN every organ (not organ markers)")
    log(f"      {', '.join(panel[:10])} ...")

    # ---- assemble per-sample matrices
    log("\n[3/5] aligning embeddings to expression by barcode ...")
    data = {}
    for sid, (M, bc, genes) in per.items():
        z = np.load(os.path.join(EMB, f"{sid}_uni2h.npz"), allow_pickle=True)
        Xe, ebc = z["X"], np.array([str(b) for b in z["barcode"]])
        gi = pd.Index(genes).get_indexer(panel)
        sub = M[:, gi]
        lib = np.asarray(M.sum(1)).ravel(); lib[lib == 0] = 1
        Y = np.asarray(sp.csr_matrix(sub.multiply(1e4 / lib[:, None])).log1p().todense())
        pos = pd.Index(bc).get_indexer(ebc)
        keep = pos >= 0
        if keep.sum() < 50:
            log(f"      {sid}: only {keep.sum()} barcodes matched, skipping"); continue
        data[sid] = (Xe[keep].astype(np.float32), Y[pos[keep]].astype(np.float32))
    log(f"      usable samples: {len(data)}")
    org = {s: meta.loc[meta[idc].astype(str) == s, "organ"].iloc[0] for s in data}

    def fit_predict(train_ids, test_ids):
        Xtr = np.vstack([data[s][0] for s in train_ids])
        Ytr = np.vstack([data[s][1] for s in train_ids])
        m = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA)).fit(Xtr, Ytr)
        rs = []
        for s in test_ids:
            Xte, Yte = data[s]
            P = m.predict(Xte)
            r = [np.corrcoef(Yte[:, g], P[:, g])[0, 1]
                 for g in range(Yte.shape[1]) if Yte[:, g].std() > 1e-8]
            rs.append(np.nanmean(r))
        return float(np.nanmean(rs))

    # ---- within-organ, leave-one-sample-out
    log("\n[4/5] WITHIN-ORGAN (leave-one-section-out)")
    within = {}
    for o in organs:
        ids = [s for s in data if org[s] == o]
        if len(ids) < 2:
            continue
        sc = [fit_predict([t for t in ids if t != s], [s]) for s in ids]
        within[o] = float(np.nanmean(sc))
        log(f"      {o:12s} n={len(ids):3d}   mean per-gene r = {within[o]:+.3f}")

    # ---- cross-organ
    log("\n[5/5] CROSS-ORGAN (train on one organ, test on another)")
    mat = pd.DataFrame(index=organs, columns=organs, dtype=float)
    for a in organs:
        tr = [s for s in data if org[s] == a]
        if not tr:
            continue
        for b in organs:
            te = [s for s in data if org[s] == b]
            if not te:
                continue
            mat.loc[a, b] = within[b] if a == b and b in within else \
                (fit_predict(tr, te) if a != b else np.nan)
    log("\n      rows = trained on, cols = tested on (diagonal = within-organ)")
    log(mat.round(3).to_string())

    # ---------------------------------------------------------- BATCH CONTROL
    # HEST samples come from many source studies, and an organ is often covered by
    # a single study. Then "train on organ A, test on organ B" also means "train on
    # study X, test on study Y", and the drop could be pure batch effect -- lab,
    # protocol, sequencing depth, fixation -- with no organ biology in it.
    #
    # The control: WITHIN one organ, train on one study and test on another. If
    # that drops as much as the cross-organ comparison, the effect is batch, not
    # organ, and the cross-organ number means nothing.
    log("\n[6/6] BATCH CONTROL - within-organ, ACROSS study")
    study = {}
    if "dataset_title" in meta.columns:
        study = {s: meta.loc[meta[idc].astype(str) == s, "dataset_title"].iloc[0]
                 for s in data}
    ctrl = []
    for o in organs:
        ids = [s for s in data if org[s] == o]
        studies = {}
        for s in ids:
            studies.setdefault(study.get(s, "?"), []).append(s)
        if len(studies) < 2:
            log(f"      {o:12s} only {len(studies)} study — cannot test")
            continue
        keys = list(studies)
        for a in keys:
            for b in keys:
                if a == b or not studies[a] or not studies[b]:
                    continue
                r = fit_predict(studies[a], studies[b])
                ctrl.append(r)
                log(f"      {o:12s} train[{a[:26]}...] -> test[{b[:26]}...] r={r:+.3f}")
    if ctrl:
        log(f"\n      within-organ ACROSS-study mean : {np.nanmean(ctrl):+.3f}")
    else:
        log("      NO organ has >1 study among the available samples, so organ and")
        log("      study are perfectly confounded. The cross-organ number below")
        log("      CANNOT be attributed to organ biology.")

    diag = np.array([mat.loc[o, o] for o in organs if not np.isnan(mat.loc[o, o])])
    off = mat.values[~np.eye(len(organs), dtype=bool)]
    off = off[~np.isnan(off)]
    if len(off) == 0:
        log(f"\n      within-organ mean : {diag.mean():+.3f}")
        log(f"      only {len(organs)} organ(s) available -- the cross-organ test needs at")
        log("      least 2. Re-run once more organs have downloaded.")
        mat.to_csv(os.path.join(OUT, "cross_organ_matrix.csv"))
        return
    log(f"\n      within-organ mean : {diag.mean():+.3f}")
    log(f"      cross-organ  mean : {off.mean():+.3f}")
    log(f"      drop on transfer  : {diag.mean()-off.mean():+.3f} "
        f"({100*(1-off.mean()/max(diag.mean(),1e-9)):.0f}% of the signal)")
    # The verdict MUST depend on the batch control: a drop is only attributable to
    # organ biology if within-organ/across-study transfer holds up.
    if not ctrl:
        log("\n      VERDICT: UNINTERPRETABLE. Organ and study are perfectly confounded")
        log("      in the available samples, so this drop may be entirely batch effect.")
        log("      Re-run once an organ is covered by more than one study.")
    else:
        cm = float(np.nanmean(ctrl))
        log(f"\n      cross-organ {off.mean():+.3f}  vs  within-organ/across-study {cm:+.3f}")
        if off.mean() < cm - 0.03:
            log("      VERDICT: transfer degrades MORE across organs than across studies of")
            log("      the same organ -> a genuine organ effect beyond batch. This supports")
            log("      the Stage 6-7 prediction that expression is tissue-context specific.")
        elif abs(off.mean() - cm) <= 0.03:
            log("      VERDICT: cross-organ is no worse than cross-study WITHIN an organ ->")
            log("      the drop is BATCH, not organ. This does NOT support the prediction.")
        else:
            log("      VERDICT: cross-study within an organ is worse than cross-organ —")
            log("      batch dominates; treat the organ comparison as unreliable.")

    mat.to_csv(os.path.join(OUT, "cross_organ_matrix.csv"))
    json.dump({"within": within, "panel": panel.tolist(),
               "within_mean": float(diag.mean()), "cross_mean": float(off.mean()),
               "n_samples": len(data),
               "organs": {o: int(sum(1 for s in data if org[s] == o)) for o in organs}},
              open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    log(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
