"""
Gene-level importance for the leaving program.
Answers: (1) which of the 19 panel genes drive the confound-free target, and
(2) data-driven, across ALL ~17.9k genes, which genes most mark high vs low
'leaving' spots -> the top/bottom genes and their biology.

Method: per PT sample, CP10k->log1p normalize; Pearson-correlate every gene with
leaving_score_resid (the confound-free target); average correlation across the 4
slides (equal weight, so no single slide dominates). Output JSON for the figure.

Run: "C:/Users/datai/anaconda3/envs/tcga/python.exe" docs/presentation/_gene_importance.py
"""
import os, json
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__)); PROJ = os.path.dirname(os.path.dirname(ROOT))
COUNTS = os.path.join(PROJ, "dataset", "ST", "scVI_counts")
LEAD = pd.read_csv(os.path.join(PROJ, "Outputs/stage1a_leaving_program/leaving_program_scores.csv"))
PT = ["IU_PDA_T1", "IU_PDA_T3", "IU_PDA_T4", "IU_PDA_T11"]

CORE = ["SNAI1","ZEB1","ZEB2","CDH2","S100A4","TGFBR1","TGFB1","MMP1","LOXL2",
        "ITGB6","LAMC2","SERPINE1","COL1A1","COL3A1","COL5A1","FN1","LOX","TNC","TIMP1"]
HELDOUT = ["VIM","SNAI2","PRRX1","MMP9","MMP14","POSTN","SPARC"]
EPI = ["CDH1","EPCAM","KRT8","KRT18","KRT19","CLDN4","CLDN7","KRT7"]  # epithelial anchors

def corr_cols(X, y):
    """Pearson corr of every column of X (spots x genes) with vector y."""
    Xc = X - X.mean(0); yc = y - y.mean()
    num = Xc.T @ yc
    den = np.sqrt((Xc**2).sum(0) * (yc**2).sum()) + 1e-12
    return num / den

per_sample = {}
for s in PT:
    lead_s = LEAD[LEAD["sample"] == s].set_index("barcode")
    df = pd.read_csv(os.path.join(COUNTS, f"{s}.csv"), index_col=0)
    df.index = [g.strip().strip('"') for g in df.index.astype(str)]
    df = df[~df.index.duplicated(keep="first")]
    common = [b for b in df.columns if b in lead_s.index]
    counts = df[common].to_numpy(dtype=np.float32).T           # spots x genes
    genes = np.array(df.index)
    lib = counts.sum(1, keepdims=True); lib[lib == 0] = 1
    logn = np.log1p(counts / lib * 1e4)                        # CP10k -> log1p
    resid = lead_s.loc[common, "leaving_score_resid"].to_numpy(np.float32)
    raw   = lead_s.loc[common, "leaving_score"].to_numpy(np.float32)
    per_sample[s] = {
        "genes": genes,
        "corr_resid": corr_cols(logn, resid),
        "corr_raw":   corr_cols(logn, raw),
        "n": len(common),
    }
    print(f"{s}: {len(common)} spots x {len(genes)} genes")
    del df, counts, logn

# align genes across samples (intersection)
gsets = [set(per_sample[s]["genes"]) for s in PT]
common_genes = sorted(set.intersection(*gsets))
gi = {g: i for i, g in enumerate(common_genes)}
R = np.zeros((len(common_genes), len(PT)), np.float32)   # resid corr
Rr = np.zeros((len(common_genes), len(PT)), np.float32)  # raw corr
for j, s in enumerate(PT):
    idx = {g: i for i, g in enumerate(per_sample[s]["genes"])}
    sel = np.array([idx[g] for g in common_genes])
    R[:, j]  = per_sample[s]["corr_resid"][sel]
    Rr[:, j] = per_sample[s]["corr_raw"][sel]

mean_resid = np.nanmean(R, 1)
mean_raw   = np.nanmean(Rr, 1)
consistent = (np.sign(R).sum(1))   # +4 => positive in all 4 slides

res = pd.DataFrame({"gene": common_genes, "corr_resid": mean_resid,
                    "corr_raw": mean_raw, "sign_agreement": consistent})
res["in_core"] = res["gene"].isin(CORE)
res["in_heldout"] = res["gene"].isin(HELDOUT)
res["is_epithelial"] = res["gene"].isin(EPI)

# ---- panel gene ranking (impact on target)
core_rank = res[res["in_core"]].sort_values("corr_resid", ascending=False)
print("\n=== 19 CORE genes ranked by corr with confound-free target (resid) ===")
print(core_rank[["gene","corr_resid","corr_raw","sign_agreement"]].to_string(index=False))

# ---- data-driven TOP genes (positive) and BOTTOM genes (negative), all genes
# require consistency across all 4 slides to avoid single-slide artefacts
top = res[(res.sign_agreement >= 3)].sort_values("corr_resid", ascending=False).head(20)
bot = res[(res.sign_agreement <= -3)].sort_values("corr_resid").head(20)
print("\n=== DATA-DRIVEN TOP genes (mark HIGH-leaving spots) ===")
print(top[["gene","corr_resid","in_core","in_heldout","is_epithelial"]].head(12).to_string(index=False))
print("\n=== DATA-DRIVEN BOTTOM genes (mark LOW-leaving spots) ===")
print(bot[["gene","corr_resid","in_core","in_heldout","is_epithelial"]].head(12).to_string(index=False))

# epithelial anchors specifically
epi_rows = res[res.is_epithelial].sort_values("corr_resid")
print("\n=== Epithelial anchor genes (expected NEGATIVE if biology is right) ===")
print(epi_rows[["gene","corr_resid","sign_agreement"]].to_string(index=False))

# ---- how much signal does a REDUCED top-k panel recover vs the full 19?
# Build a simple mean-z score over top-k core genes per sample, correlate with
# the full-panel confound-free target (leaving_score_resid) and held-out validators.
core_order = core_rank["gene"].tolist()  # best -> worst by corr_resid
reduced = {}
for k in [3, 5, 8, 19]:
    picks = core_order[:k]
    per = []
    for s in PT:
        lead_s = LEAD[LEAD["sample"] == s].set_index("barcode")
        gidx = {g: i for i, g in enumerate(per_sample[s]["genes"])}
        # rebuild normalized expression for the picked genes only (cheap: reuse corr? no—need values)
        pass
    reduced[k] = picks
# recompute reduced-panel score properly (needs the normalized matrices again, light pass)
red_corr = {k: [] for k in [3, 5, 8, 19]}
red_heldout = {k: [] for k in [3, 5, 8, 19]}
for s in PT:
    lead_s = LEAD[LEAD["sample"] == s].set_index("barcode")
    df = pd.read_csv(os.path.join(COUNTS, f"{s}.csv"), index_col=0)
    df.index = [g.strip().strip('"') for g in df.index.astype(str)]
    df = df[~df.index.duplicated(keep="first")]
    common = [b for b in df.columns if b in lead_s.index]
    counts = df[common].to_numpy(np.float32).T
    lib = counts.sum(1, keepdims=True); lib[lib == 0] = 1
    logn = np.log1p(counts / lib * 1e4)
    gidx = {g: i for i, g in enumerate(df.index)}
    def zmean(genes):
        cols = [gidx[g] for g in genes if g in gidx]
        Z = logn[:, cols]
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
        return Z.mean(1)
    resid = lead_s.loc[common, "leaving_score_resid"].to_numpy(np.float32)
    hd = lead_s.loc[common, "heldout_resid"].to_numpy(np.float32) if "heldout_resid" in lead_s else None
    for k in [3, 5, 8, 19]:
        sc = zmean(reduced[k])
        red_corr[k].append(np.corrcoef(sc, resid)[0, 1])
        if hd is not None:
            red_heldout[k].append(np.corrcoef(sc, hd)[0, 1])
    del df, counts, logn
print("\n=== Reduced top-k panel vs FULL confound-free target (avg of 4 tumours) ===")
for k in [3, 5, 8, 19]:
    hh = np.nanmean(red_heldout[k]) if red_heldout[k] else float("nan")
    print(f"top-{k:2d} genes ({', '.join(reduced[k])}): corr vs 19-gene target = {np.nanmean(red_corr[k]):.3f} | vs held-out EMT = {hh:.3f}")

out = {
    "reduced_panels": {str(k): {"genes": reduced[k],
                                "corr_vs_full_target": float(np.nanmean(red_corr[k])),
                                "corr_vs_heldout": float(np.nanmean(red_heldout[k])) if red_heldout[k] else None}
                       for k in [3, 5, 8, 19]},
    "core_ranked": core_rank[["gene","corr_resid","corr_raw","sign_agreement"]].to_dict("records"),
    "top_genes": top[["gene","corr_resid","in_core","in_heldout"]].head(15).to_dict("records"),
    "bottom_genes": bot[["gene","corr_resid","is_epithelial"]].head(15).to_dict("records"),
    "epithelial": epi_rows[["gene","corr_resid","sign_agreement"]].to_dict("records"),
}
op = os.path.join(PROJ, "Outputs/stage1a_leaving_program/gene_importance.json")
json.dump(out, open(op, "w"), indent=2, default=float)
res.to_csv(os.path.join(PROJ, "Outputs/stage1a_leaving_program/gene_importance_all.csv"), index=False)
print("\nsaved ->", op)
