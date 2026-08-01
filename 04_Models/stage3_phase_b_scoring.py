"""
STAGE 3 - Phase B scoring with confound control (laptop, no GPU needed).

Goal (REVIEW_PLAN.md Stage 3):
  Produce a per-PT-spot, VISION-ONLY metastatic-propensity score (proves ST is not
  needed at inference, per Dr. Ashiq) by predicting the Stage-1A "leaving program"
  target from a histology-FM representation -- then audit it against the tissue /
  abundance confounds and check PT11->HM11 convergence.

Why this script also runs the bridge-vs-frozen-FM ABLATION first
---------------------------------------------------------------
Stage 2 (`stage2_results` memory) found the tri-modal InfoNCE bridge does NOT
generalize across patients: held-out vision->leaving was rho~0.28 on the RAW
(abundance-driven) target and at the NOISE FLOOR (rho~0.08, neg R2) on the
confound-free RESID target. The locked next step before committing a Phase B head
was the *decisive* ablation:

    Does a plain Ridge/MLP on the FROZEN UNI2-h embedding (1536d, NO bridge)
    do as well as the 128d bridge vision embedding?

If frozen-direct ~= bridge, the contrastive bridge isn't earning its keep and Phase B
should be built directly on the frozen FM (still ST-only-at-train: ST defines the
target, never used at inference). This script measures that head-to-head, then trains
the winning predictor as the Phase B scoring head.

Targets (Stage 1A, locked):
  - leaving_score        (RAW)   = abundance-aware upper bound
  - leaving_score_resid  (RESID) = tumour-fraction-residualized = CONFOUND-FREE HEADLINE

Method
------
- 4-fold LOSO over PT samples (T1, T3, T4, T11). Train on 3, predict held-out one.
  Out-of-fold (OOF) predictions cover all 13,578 PT spots with no patient leakage.
- Predictors compared, each x {raw, resid} target:
      bridge128   : Stage-2 UNI2-h Variant-A vision_emb (128d)
      frozenFM    : raw UNI2-h embedding (1536d)
  RidgeCV (standardized features) is the primary probe; a small MLP is run on the
  winner for a non-linear check.
- The winning predictor's OOF prediction IS the Phase B score (raw & resid heads).
- Spatial smoothing: k=6 Visium hex-neighbour mean of the predicted score, per slide;
  both smoothed and raw predictions are reported.
- Confound audit (mandatory): correlate the final predicted score with RCTD
  tumor_frac and hepatocyte_frac. The confound-free (resid) score must NOT be
  explained by these.
- Convergence (secondary, qualitative): on PT11, correlate predicted score with the
  Stage-1B HM11-resemblance axis (expected weak, ~0.17, per Stage 1B).

Run:
    "C:/Users/datai/anaconda3/envs/tcga/python.exe" stage3_phase_b_scoring.py

Outputs -> Outputs/stage3_phase_b/
    phase_b_scores.csv, metrics.json, summary.txt,
    ablation_bars.png, pred_vs_target.png, confound_bars.png,
    heatmap_<sample>.png (predicted score, smoothed), convergence_pt11.png
"""

import os, json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# ----------------------------------------------------------------------------- paths
from _cohort import ROOT, PT_SAMPLES, OUT_TAG, out_dir, banner

BRIDGE_PT  = os.path.join(ROOT, "Outputs", "stage2", "UNI2h", "stage2_UNI2h_outputs",
                          "all_spots_embeddings.pt")
FM_DIR     = os.path.join(ROOT, "dataset", "Feature Extraction Embeddings", "UNI2-h")
LEAVING_CSV= os.path.join(ROOT, "Outputs", "stage1a_leaving_program" + OUT_TAG,
                          "leaving_program_scores.csv")
PT11_CSV   = os.path.join(ROOT, "Outputs", "stage1b_pt11_anchor" + OUT_TAG,
                          "pt11_hm11_resemblance.csv")
OUT_DIR    = out_dir("stage3_phase_b")
banner("STAGE 3 - Phase B scoring")

FM_SUFFIX  = "_uni2h.pt"
SEED       = 42
np.random.seed(SEED)

# ----------------------------------------------------------------------------- load target
print("[1/6] Loading Stage-1A leaving target ...")
tgt = pd.read_csv(LEAVING_CSV)
tgt = tgt.set_index("patch_stem")
print(f"      {len(tgt)} PT spots, samples={tgt['sample'].value_counts().to_dict()}")

# ----------------------------------------------------------------------------- load bridge vision emb (128d, Variant A)
print("[2/6] Loading bridge (128d) + frozen FM (1536d) embeddings ...")
b = torch.load(BRIDGE_PT, map_location="cpu", weights_only=False)
bridge_ids = b["ids"]
bridge_vis = b["vision_emb"].numpy()
bridge_map = {sid: i for i, sid in enumerate(bridge_ids)}

# load frozen UNI2-h per PT sample, build patch_stem -> 1536d
fm_vec = {}
for s in PT_SAMPLES:
    d = torch.load(os.path.join(FM_DIR, s + FM_SUFFIX), map_location="cpu",
                   weights_only=False)
    emb = d["embeddings"].numpy()
    names = d["patch_names"]
    for n, v in zip(names, emb):
        fm_vec[n] = v

# align all three to the common PT spot set (target order)
ids = [sid for sid in tgt.index if sid in bridge_map and sid in fm_vec]
print(f"      aligned spots (target & bridge & FM): {len(ids)} / {len(tgt)}")
tgt = tgt.loc[ids]

X_bridge = np.stack([bridge_vis[bridge_map[sid]] for sid in ids]).astype(np.float32)
X_fm     = np.stack([fm_vec[sid] for sid in ids]).astype(np.float32)
samples  = tgt["sample"].values
y_raw    = tgt["leaving_score"].values.astype(np.float32)
y_resid  = tgt["leaving_score_resid"].values.astype(np.float32)
print(f"      X_bridge {X_bridge.shape}  X_fm {X_fm.shape}")

# ----------------------------------------------------------------------------- LOSO probe
ALPHAS = np.logspace(-1, 4, 12)

def loso_ridge(X, y, samples):
    """Leave-one-sample-out RidgeCV; returns OOF predictions + per-fold/pooled rho."""
    oof = np.full(len(y), np.nan, dtype=np.float64)
    per_fold = {}
    for s in PT_SAMPLES:
        te = samples == s
        tr = ~te
        sc = StandardScaler().fit(X[tr])
        model = RidgeCV(alphas=ALPHAS).fit(sc.transform(X[tr]), y[tr])
        oof[te] = model.predict(sc.transform(X[te]))
        rho = spearmanr(y[te], oof[te]).correlation
        per_fold[s] = float(rho)
    pooled = float(spearmanr(y, oof).correlation)
    return oof, per_fold, pooled

def loso_mlp(X, y, samples):
    oof = np.full(len(y), np.nan, dtype=np.float64)
    per_fold = {}
    for s in PT_SAMPLES:
        te = samples == s
        tr = ~te
        sc = StandardScaler().fit(X[tr])
        model = MLPRegressor(hidden_layer_sizes=(128, 32), alpha=1e-2,
                             max_iter=400, early_stopping=True, random_state=SEED)
        model.fit(sc.transform(X[tr]), y[tr])
        oof[te] = model.predict(sc.transform(X[te]))
        per_fold[s] = float(spearmanr(y[te], oof[te]).correlation)
    pooled = float(spearmanr(y, oof).correlation)
    return oof, per_fold, pooled

print("[3/6] Ablation: LOSO Ridge  {bridge128, frozenFM} x {raw, resid} ...")
ablation = {}
oof_store = {}
for name, X in [("bridge128", X_bridge), ("frozenFM", X_fm)]:
    for tname, y in [("raw", y_raw), ("resid", y_resid)]:
        oof, per_fold, pooled = loso_ridge(X, y, samples)
        key = f"{name}_{tname}"
        ablation[key] = {"pooled_rho": pooled, "per_fold_rho": per_fold}
        oof_store[key] = oof
        print(f"      ridge {key:18s} pooled rho = {pooled:+.3f}  "
              f"folds={ {k: round(v,2) for k,v in per_fold.items()} }")

# MLP non-linear check on frozen FM (the higher-dim representation)
print("[3b]  MLP non-linear check on frozenFM ...")
for tname, y in [("raw", y_raw), ("resid", y_resid)]:
    oof, per_fold, pooled = loso_mlp(X_fm, y, samples)
    key = f"frozenFM_mlp_{tname}"
    ablation[key] = {"pooled_rho": pooled, "per_fold_rho": per_fold}
    oof_store[key] = oof
    print(f"      mlp   {key:18s} pooled rho = {pooled:+.3f}")

# ----------------------------------------------------------------------------- pick predictor
# Decision rule: the Phase B headline is the CONFOUND-FREE resid score. Choose the
# predictor with the best held-out resid rho; report raw alongside as the upper bound.
cands = {k: v["pooled_rho"] for k, v in ablation.items() if k.endswith("_resid")
         and "mlp" not in k}
best_resid_key = max(cands, key=cands.get)
best_predictor = best_resid_key.replace("_resid", "")   # 'bridge128' or 'frozenFM'
print(f"[4/6] Best resid predictor = {best_predictor} "
      f"(resid rho={ablation[best_resid_key]['pooled_rho']:+.3f})")

pred_raw   = oof_store[f"{best_predictor}_raw"]
pred_resid = oof_store[f"{best_predictor}_resid"]

# also keep the other predictor's resid for the report
other = "frozenFM" if best_predictor == "bridge128" else "bridge128"
bridge_minus_frozen_resid = (ablation["bridge128_resid"]["pooled_rho"]
                             - ablation["frozenFM_resid"]["pooled_rho"])

# ----------------------------------------------------------------------------- spatial smoothing (k=6 hex)
print("[5/6] k=6 hex-neighbour spatial smoothing ...")
# Visium array (row,col) hex neighbours: (r, c+-2) and (r+-1, c+-1)
HEX = [(0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)]

def hex_smooth(values, rows, cols, samples):
    out = values.copy().astype(np.float64)
    for s in PT_SAMPLES:
        idx = np.where(samples == s)[0]
        pos = {(int(rows[i]), int(cols[i])): i for i in idx}
        for i in idx:
            r, c = int(rows[i]), int(cols[i])
            acc = [values[i]]
            for dr, dc in HEX:
                j = pos.get((r + dr, c + dc))
                if j is not None:
                    acc.append(values[j])
            out[i] = np.mean(acc)
    return out

rows = tgt["row"].values
cols = tgt["col"].values
pred_raw_sm   = hex_smooth(pred_raw,   rows, cols, samples)
pred_resid_sm = hex_smooth(pred_resid, rows, cols, samples)

# ----------------------------------------------------------------------------- confound audit
print("[6/6] Confound audit + convergence + figures ...")
tumor = tgt["tumor_frac"].values
hep   = tgt["hepatocyte_frac"].values
caf   = tgt["caf_frac"].values

def sp(a, b):
    return float(spearmanr(a, b).correlation)

confound = {
    "pred_raw_vs_tumor_frac"      : sp(pred_raw,   tumor),
    "pred_raw_vs_hepatocyte_frac" : sp(pred_raw,   hep),
    "pred_raw_vs_caf_frac"        : sp(pred_raw,   caf),
    "pred_resid_vs_tumor_frac"    : sp(pred_resid, tumor),
    "pred_resid_vs_hepatocyte_frac": sp(pred_resid, hep),
    "pred_resid_vs_caf_frac"      : sp(pred_resid, caf),
    "pred_raw_vs_pred_resid"      : sp(pred_raw,   pred_resid),
}
for k, v in confound.items():
    print(f"      {k:32s} {v:+.3f}")

# ----------------------------------------------------------------------------- PT11 convergence (secondary)
conv = {}
try:
    pt11 = pd.read_csv(PT11_CSV).set_index("patch_stem")
    common = [sid for sid in pt11.index if sid in set(ids)]
    pos = {sid: i for i, sid in enumerate(ids)}
    if len(common) > 20:
        idx = np.array([pos[sid] for sid in common])
        res_pt11 = pt11.loc[common, "hm11_resemblance"].values
        conv = {
            "n_pt11": len(common),
            "pred_resid_vs_hm11_resemblance": sp(pred_resid[idx], res_pt11),
            "pred_raw_vs_hm11_resemblance"  : sp(pred_raw[idx],   res_pt11),
        }
        print(f"      PT11 convergence (n={len(common)}): "
              f"resid->HM11 {conv['pred_resid_vs_hm11_resemblance']:+.3f}, "
              f"raw->HM11 {conv['pred_raw_vs_hm11_resemblance']:+.3f}")
except Exception as e:
    conv = {"error": str(e)}
    print("      convergence skipped:", e)

# ----------------------------------------------------------------------------- save scores csv
out = tgt.reset_index()[["patch_stem", "sample", "row", "col",
                         "tumor_frac", "hepatocyte_frac", "caf_frac",
                         "leaving_score", "leaving_score_resid"]].copy()
out["pred_raw"]        = pred_raw
out["pred_resid"]      = pred_resid
out["pred_raw_smooth"] = pred_raw_sm
out["pred_resid_smooth"] = pred_resid_sm
out["predictor"]       = best_predictor
out.to_csv(os.path.join(OUT_DIR, "phase_b_scores.csv"), index=False)

# ----------------------------------------------------------------------------- metrics json
metrics = {
    "n_spots": len(ids),
    "best_predictor": best_predictor,
    "ablation_ridge": ablation,
    "bridge_minus_frozen_resid_rho": bridge_minus_frozen_resid,
    "confound_audit": confound,
    "convergence_pt11": conv,
    "smoothing": {
        "pred_raw_rho_target_after_smooth":   sp(pred_raw_sm,   y_raw),
        "pred_resid_rho_target_after_smooth": sp(pred_resid_sm, y_resid),
        "pred_raw_rho_target_raw":   sp(pred_raw,   y_raw),
        "pred_resid_rho_target_raw": sp(pred_resid, y_resid),
    },
}
with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# ----------------------------------------------------------------------------- figures
# (a) ablation bars
fig, ax = plt.subplots(figsize=(8, 4.5))
keys = ["bridge128_raw", "frozenFM_raw", "bridge128_resid", "frozenFM_resid"]
vals = [ablation[k]["pooled_rho"] for k in keys]
colors = ["#4C78A8", "#F58518", "#4C78A8", "#F58518"]
ax.bar(range(len(keys)), vals, color=colors)
ax.set_xticks(range(len(keys)))
ax.set_xticklabels(keys, rotation=20, ha="right")
ax.axhline(0, color="k", lw=0.7)
ax.axhline(0.08, color="grey", ls="--", lw=0.8, label="~noise floor (Stage 2)")
ax.set_ylabel("held-out (LOSO) Spearman rho")
ax.set_title("Stage 3 ablation: bridge-128 vs frozen-FM-1536  ->  leaving score")
ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "ablation_bars.png"), dpi=130); plt.close()

# (b) pred vs target scatter (best predictor)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, p, y, t in [(axs[0], pred_raw, y_raw, "RAW"),
                    (axs[1], pred_resid, y_resid, "RESID (confound-free)")]:
    ax.scatter(y, p, s=3, alpha=0.2)
    ax.set_xlabel(f"Stage-1A target ({t})"); ax.set_ylabel("vision-only prediction")
    ax.set_title(f"{t}  rho={sp(p,y):+.3f}")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "pred_vs_target.png"), dpi=130); plt.close()

# (c) confound bars
fig, ax = plt.subplots(figsize=(8, 4.5))
ck = list(confound.keys()); cv = [confound[k] for k in ck]
ax.barh(range(len(ck)), cv, color=["#E45756" if abs(v) > 0.3 else "#72B7B2" for v in cv])
ax.set_yticks(range(len(ck))); ax.set_yticklabels(ck, fontsize=8)
ax.axvline(0, color="k", lw=0.7)
ax.set_xlabel("Spearman rho"); ax.set_title("Stage 3 confound audit")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "confound_bars.png"), dpi=130); plt.close()

# (d) spatial heatmaps (smoothed resid prediction)
for s in PT_SAMPLES:
    m = samples == s
    fig, ax = plt.subplots(figsize=(5, 5))
    sca = ax.scatter(cols[m], -rows[m], c=pred_resid_sm[m], cmap="magma", s=12)
    plt.colorbar(sca, ax=ax, shrink=0.8)
    ax.set_title(f"{s}: vision-only predicted leaving (resid, k6-smoothed)")
    ax.set_aspect("equal"); ax.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, f"heatmap_{s}.png"), dpi=120); plt.close()

# (e) convergence scatter
if "pred_resid_vs_hm11_resemblance" in conv:
    idx = np.array([{sid: i for i, sid in enumerate(ids)}[sid] for sid in common])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(pred_resid[idx], res_pt11, s=6, alpha=0.4)
    ax.set_xlabel("PT11 vision-only predicted leaving (resid)")
    ax.set_ylabel("HM11 resemblance (Stage 1B)")
    ax.set_title(f"PT11->HM11 convergence  rho={conv['pred_resid_vs_hm11_resemblance']:+.3f}")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "convergence_pt11.png"), dpi=130); plt.close()

# ----------------------------------------------------------------------------- summary.txt
lines = []
lines.append("STAGE 3 - PHASE B SCORING (vision-only) + CONFOUND AUDIT")
lines.append("=" * 60)
lines.append(f"Aligned PT spots: {len(ids)}")
lines.append("")
lines.append("ABLATION (held-out LOSO Spearman rho, Ridge):")
for k in keys:
    lines.append(f"  {k:18s} {ablation[k]['pooled_rho']:+.3f}")
lines.append(f"  frozenFM_mlp_raw   {ablation['frozenFM_mlp_raw']['pooled_rho']:+.3f}")
lines.append(f"  frozenFM_mlp_resid {ablation['frozenFM_mlp_resid']['pooled_rho']:+.3f}")
lines.append(f"  bridge - frozen (resid) = {bridge_minus_frozen_resid:+.3f}")
lines.append(f"  -> best resid predictor: {best_predictor}")
lines.append("")
lines.append("CONFOUND AUDIT (Spearman rho):")
for k, v in confound.items():
    lines.append(f"  {k:34s} {v:+.3f}")
lines.append("")
lines.append("CONVERGENCE (PT11 -> HM11, secondary):")
for k, v in conv.items():
    lines.append(f"  {k}: {v}")
lines.append("")
lines.append("READ: RESID is the confound-free headline; RAW is the abundance-aware")
lines.append("upper bound. If both predictors give resid rho ~ noise floor (<~0.10),")
lines.append("the confound-free EMT signal is not recoverable from H&E on held-out")
lines.append("patients; only the abundance-driven RAW score (rho~0.3) is. If")
lines.append("frozenFM >= bridge128, the contrastive bridge is not earning its keep")
lines.append("-> pivot Phase B to direct frozen-FM supervised regression.")
with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
    f.write("\n".join(lines))

print("\n".join(lines))
print(f"\nDONE -> {OUT_DIR}")
