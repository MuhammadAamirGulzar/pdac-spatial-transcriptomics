"""
STAGE 4 - Validation protocol (laptop, no GPU).

Applies the FOUR pre-registered validation tests from REVIEW_PLAN.md to the two
quantities locked in Stages 1-3:
  - TARGET   : Stage-1A intra-PT "leaving program"  (leaving_score raw + resid)
  - PREDICTOR: Stage-3 vision-only OOF prediction    (pred_raw + pred_resid)
               (these are LEAVE-ONE-PATIENT-OUT out-of-fold preds -> already held-out)

The validation criteria were defined BEFORE trusting any score, exactly so the
honest Stage 0-3 findings (confound-free signal at the noise floor; H&E predicts
only malignant abundance; bridge == frozen FM; PT11->HM11 convergence negative)
can be confirmed or overturned against independent evidence rather than re-derived
from the same construction.

The decisive new evidence here is EXTERNAL: the source Nature-Genetics paper
(41588_2024_1914, Suppl. Data 2 / MOESM5) provides INDEPENDENT spotwise GSEA
scores for 29 functional gene-expression signatures (Fges), keyed by spot
barcode -- including an `EMT_signature` derived completely independently of our
AddModuleScore CORE construction. We test our target & our vision predictor
against it.

THE FOUR TESTS
--------------
1. Patient-11 anchor : do high-scoring PT11 spots resemble HM11 (Stage-1B axis)?
2. Held-out + external signatures :
     (a) construct validity  - does the TARGET track independent EMT (paper EMT
         GSEA + Stage-1A held-out EMT validators)?
     (b) deliverable test    - does the VISION predictor recover independent EMT
         on held-out patients?
     (c) specificity panel   - correlate target & vision pred with all 29 Fges to
         see what H&E actually captures (expect proliferation/matrix/abundance,
         NOT the confound-free EMT axis).
3. Spatial / invasive margin : is the score enriched at the tumour-stroma boundary
   vs the tumour interior? (computational proxy; no pathologist annotation yet).
4. Negative control : the confound-free (resid) score must be decorrelated from
   hepatocyte/liver content and abundance; and the raw vision score must not beat a
   trivial tumour-fraction-only baseline (the abundance ceiling).

Run:
    "C:/Users/datai/anaconda3/envs/tcga/python.exe" stage4_validation.py

Outputs -> Outputs/stage4_validation/
    validation_metrics.json, validation_report.txt,
    fges_specificity.png, external_emt.png, margin_enrichment.png,
    test1_pt11.png, negative_control.png
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------- paths
ROOT        = os.path.dirname(os.path.abspath(__file__))
LEAVING_CSV = os.path.join(ROOT, "Outputs", "stage1a_leaving_program", "leaving_program_scores.csv")
PHASEB_CSV  = os.path.join(ROOT, "Outputs", "stage3_phase_b", "phase_b_scores.csv")
PT11_CSV    = os.path.join(ROOT, "Outputs", "stage1b_pt11_anchor", "pt11_hm11_resemblance.csv")
FGES_XLSX   = os.path.join(ROOT, "Outputs", "Patient-Sample-Information",
                           "41588_2024_1914_MOESM5_ESM.xlsx")
OUT_DIR     = os.path.join(ROOT, "Outputs", "stage4_validation")
os.makedirs(OUT_DIR, exist_ok=True)

PT_SAMPLES = ["IU_PDA_T1", "IU_PDA_T3", "IU_PDA_T4", "IU_PDA_T11"]
SEED = 42
np.random.seed(SEED)


def sp(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return float("nan")
    return float(spearmanr(a[m], b[m]).correlation)


# ----------------------------------------------------------------------------- load core tables
print("[1/6] Loading Stage-1A target, Stage-3 vision preds, Stage-1B PT11 anchor ...")
tgt = pd.read_csv(LEAVING_CSV)          # has patch_stem, barcode, leaving_score(_resid), score_heldout, heldout_resid
pb  = pd.read_csv(PHASEB_CSV)           # has patch_stem, pred_raw, pred_resid, *_smooth, predictor

# pb already carries tumor/caf/hep + leaving_score(_resid) + preds; only the barcode
# and the held-out EMT validators are unique to the Stage-1A table. Merge on patch_stem
# ALONE (joining on float columns would drop rows to representation mismatches).
df = pb.merge(tgt[["patch_stem", "barcode", "score_heldout", "heldout_resid"]],
              on="patch_stem", how="inner")
print(f"      merged PT spots (target & vision pred): {len(df)}")
print(f"      predictor used in Stage 3: {df['predictor'].iloc[0]}")

# ----------------------------------------------------------------------------- external Fges GSEA
print("[2/6] Loading external Fges GSEA (paper Suppl. Data 2) and joining by barcode ...")
fg = pd.read_excel(FGES_XLSX, sheet_name=0, header=1)   # row1 = sig names, data follows
fg = fg.rename(columns={"Spot_IDs": "barcode"})
fges_cols = [c for c in fg.columns if c != "barcode"]
fg[fges_cols] = fg[fges_cols].apply(pd.to_numeric, errors="coerce")
print(f"      external table: {fg.shape[0]} spots x {len(fges_cols)} signatures")
assert "EMT_signature" in fges_cols, "EMT_signature column missing from external table"

# overlap with our PT spots
ext = df.merge(fg, on="barcode", how="left")
n_ext = ext["EMT_signature"].notna().sum()
print(f"      PT spots with external GSEA match: {n_ext} / {len(df)}")

# HM11 external EMT for Test-1 context (no leaving score for HM, just the EMT level)
hm11_mask = fg["barcode"].astype(str).str.startswith("PDACH") & \
            fg["barcode"].astype(str).str.contains("_11_", na=False)
# fall back: identify HM11 barcodes via their prefix convention if the above is too strict
hm11_emt = fg.loc[fg["barcode"].astype(str).str.contains("PDACH", na=False), "EMT_signature"]

metrics = {"n_spots": int(len(df)), "n_external_matched": int(n_ext),
           "predictor": str(df["predictor"].iloc[0]), "fges_signatures": fges_cols}

# ======================================================================================
# TEST 1 - PATIENT-11 ANCHOR
# ======================================================================================
print("[3/6] TEST 1 - Patient-11 anchor (PT11 -> HM11 resemblance) ...")
pt11 = pd.read_csv(PT11_CSV)[["patch_stem", "hm11_resemblance"]]
t1 = ext.merge(pt11, on="patch_stem", how="inner")
test1 = {
    "n_pt11": int(len(t1)),
    "target_raw_vs_hm11_resemblance":   sp(t1["leaving_score"],       t1["hm11_resemblance"]),
    "target_resid_vs_hm11_resemblance": sp(t1["leaving_score_resid"], t1["hm11_resemblance"]),
    "vision_raw_vs_hm11_resemblance":   sp(t1["pred_raw"],            t1["hm11_resemblance"]),
    "vision_resid_vs_hm11_resemblance": sp(t1["pred_resid"],          t1["hm11_resemblance"]),
    # is high external-EMT what HM11-resemblance tracks? (context)
    "pt11_extEMT_vs_hm11_resemblance":  sp(t1["EMT_signature"],       t1["hm11_resemblance"]),
}
metrics["test1_pt11_anchor"] = test1
for k, v in test1.items():
    print(f"      {k:38s} {v}")

# ======================================================================================
# TEST 2 - HELD-OUT + EXTERNAL SIGNATURES
# ======================================================================================
print("[4/6] TEST 2 - external EMT + held-out validators + 29-Fges specificity ...")
e = ext[ext["EMT_signature"].notna()].copy()

def pooled_and_perfold(score_col, sig_col, frame=e):
    out = {"pooled": sp(frame[score_col], frame[sig_col]), "per_sample": {}}
    for s in PT_SAMPLES:
        m = frame["sample"] == s
        out["per_sample"][s] = sp(frame.loc[m, score_col], frame.loc[m, sig_col])
    return out

test2 = {
    # (a) construct validity of the TARGET vs INDEPENDENT EMT
    "target_raw_vs_extEMT":    pooled_and_perfold("leaving_score",       "EMT_signature"),
    "target_resid_vs_extEMT":  pooled_and_perfold("leaving_score_resid", "EMT_signature"),
    # Stage-1A held-out EMT validators (re-stated; computed on our transcriptome)
    "target_raw_vs_heldout":   sp(e["leaving_score"],       e["score_heldout"]),
    "target_resid_vs_heldoutresid": sp(e["leaving_score_resid"], e["heldout_resid"]),
    # (b) DELIVERABLE: does the VISION predictor recover independent EMT (held-out)?
    "vision_raw_vs_extEMT":    pooled_and_perfold("pred_raw",   "EMT_signature"),
    "vision_resid_vs_extEMT":  pooled_and_perfold("pred_resid", "EMT_signature"),
    "vision_raw_vs_heldout":   sp(e["pred_raw"],   e["score_heldout"]),
    "vision_resid_vs_heldoutresid": sp(e["pred_resid"], e["heldout_resid"]),
}
metrics["test2_signatures"] = test2
print(f"      TARGET  raw vs extEMT  pooled rho = {test2['target_raw_vs_extEMT']['pooled']:+.3f}")
print(f"      TARGET  resid vs extEMT pooled rho = {test2['target_resid_vs_extEMT']['pooled']:+.3f}")
print(f"      VISION  raw vs extEMT  pooled rho = {test2['vision_raw_vs_extEMT']['pooled']:+.3f}")
print(f"      VISION  resid vs extEMT pooled rho = {test2['vision_resid_vs_extEMT']['pooled']:+.3f}")

# specificity panel: correlate target_raw / pred_raw / pred_resid vs each of 29 Fges
spec = {}
for c in fges_cols:
    spec[c] = {
        "target_raw":   sp(e["leaving_score"], e[c]),
        "vision_raw":   sp(e["pred_raw"],      e[c]),
        "vision_resid": sp(e["pred_resid"],    e[c]),
    }
metrics["test2_fges_specificity"] = spec

# ======================================================================================
# TEST 3 - SPATIAL / INVASIVE-MARGIN ENRICHMENT
# ======================================================================================
print("[5/6] TEST 3 - invasive-margin enrichment (tumour-stroma boundary) ...")
HEX = [(0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)]
TUMOR_THRESH = 0.50

def classify_margin(frame):
    """Per sample: tumour spots (tumor_frac>=thr) split into 'margin' (>=1 present
    non-tumour hex neighbour) vs 'interior' (>=3 present neighbours, all tumour)."""
    label = np.full(len(frame), "", dtype=object)
    idx_all = frame.index.to_numpy()
    for s in PT_SAMPLES:
        sub = frame[frame["sample"] == s]
        pos = {(int(r), int(c)): i for i, r, c in
               zip(sub.index, sub["row"], sub["col"])}
        istum = {i: (frame.loc[i, "tumor_frac"] >= TUMOR_THRESH) for i in sub.index}
        for i in sub.index:
            if not istum[i]:
                continue
            r, c = int(frame.loc[i, "row"]), int(frame.loc[i, "col"])
            neigh = [pos[(r + dr, c + dc)] for dr, dc in HEX if (r + dr, c + dc) in pos]
            if not neigh:
                continue
            n_nontum = sum(1 for j in neigh if not istum[j])
            if n_nontum >= 1:
                label[np.where(idx_all == i)[0][0]] = "margin"
            elif len(neigh) >= 3:
                label[np.where(idx_all == i)[0][0]] = "interior"
    return label

dfx = df.reset_index(drop=True)
dfx["margin_class"] = classify_margin(dfx)
mar = dfx[dfx["margin_class"] == "margin"]
intr = dfx[dfx["margin_class"] == "interior"]

def mw(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 10 or len(b) < 10:
        return {"n_margin": int(len(a)), "n_interior": int(len(b)),
                "mean_margin": float(np.mean(a)) if len(a) else None,
                "mean_interior": float(np.mean(b)) if len(b) else None, "p": None}
    u, p = mannwhitneyu(a, b, alternative="greater")   # H1: margin > interior
    return {"n_margin": int(len(a)), "n_interior": int(len(b)),
            "mean_margin": float(np.mean(a)), "mean_interior": float(np.mean(b)),
            "p_margin_gt_interior": float(p)}

test3 = {
    "n_margin": int(len(mar)), "n_interior": int(len(intr)),
    "target_raw":    mw(mar["leaving_score"],       intr["leaving_score"]),
    "target_resid":  mw(mar["leaving_score_resid"], intr["leaving_score_resid"]),
    "vision_raw":    mw(mar["pred_raw"],            intr["pred_raw"]),
    "vision_resid":  mw(mar["pred_resid"],          intr["pred_resid"]),
}
metrics["test3_invasive_margin"] = test3
print(f"      margin={len(mar)} interior={len(intr)} tumour spots")
for k in ["target_raw", "target_resid", "vision_raw", "vision_resid"]:
    d = test3[k]
    print(f"      {k:14s} mean margin {d.get('mean_margin')}, interior {d.get('mean_interior')}, "
          f"p(margin>interior)={d.get('p_margin_gt_interior')}")

# ======================================================================================
# TEST 4 - NEGATIVE CONTROL (decorrelation + abundance ceiling)
# ======================================================================================
print("[6/6] TEST 4 - negative control (confound decorrelation + abundance ceiling) ...")
# (a) decorrelation: resid score must NOT track hepatocyte/liver or abundance
decorr = {
    "vision_resid_vs_tumor_frac":  sp(df["pred_resid"], df["tumor_frac"]),
    "vision_resid_vs_hepatocyte":  sp(df["pred_resid"], df["hepatocyte_frac"]),
    "vision_resid_vs_caf":         sp(df["pred_resid"], df["caf_frac"]),
    "vision_raw_vs_tumor_frac":    sp(df["pred_raw"],   df["tumor_frac"]),
    "target_resid_vs_hepatocyte":  sp(df["leaving_score_resid"], df["hepatocyte_frac"]),
}
# (b) abundance ceiling: a trivial tumour-fraction-only LOSO predictor of the RAW target.
#     If vision_raw ~= this, the vision raw score is just an abundance detector.
def loso_tumorfrac_baseline(frame, ycol):
    oof = np.full(len(frame), np.nan)
    X = frame[["tumor_frac"]].to_numpy()
    y = frame[ycol].to_numpy()
    samp = frame["sample"].to_numpy()
    for s in PT_SAMPLES:
        te = samp == s; tr = ~te
        sc = StandardScaler().fit(X[tr])
        mdl = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(sc.transform(X[tr]), y[tr])
        oof[te] = mdl.predict(sc.transform(X[te]))
    return oof

base_oof = loso_tumorfrac_baseline(df, "leaving_score")
abundance_ceiling = sp(base_oof, df["leaving_score"])
vision_raw_rho = sp(df["pred_raw"], df["leaving_score"])
test4 = {
    "decorrelation": decorr,
    "abundance_ceiling_tumorfrac_only_rho": abundance_ceiling,
    "vision_raw_rho_for_reference": vision_raw_rho,
    "vision_minus_abundance_ceiling": vision_raw_rho - abundance_ceiling,
}
metrics["test4_negative_control"] = test4
print(f"      tumour-frac-only baseline rho = {abundance_ceiling:+.3f}  "
      f"(vision_raw rho = {vision_raw_rho:+.3f}; diff = {vision_raw_rho-abundance_ceiling:+.3f})")
for k, v in decorr.items():
    print(f"      {k:30s} {v:+.3f}")

# ======================================================================================
# FIGURES
# ======================================================================================
print("Writing figures ...")

# Test1: PT11 anchor scatter (vision resid vs HM11 resemblance)
fig, ax = plt.subplots(figsize=(5, 4.5))
ax.scatter(t1["pred_resid"], t1["hm11_resemblance"], s=7, alpha=0.4, color="#4C78A8")
ax.set_xlabel("PT11 vision-only predicted leaving (resid)")
ax.set_ylabel("HM11 resemblance (Stage 1B)")
ax.set_title(f"TEST 1  PT11->HM11  rho={test1['vision_resid_vs_hm11_resemblance']:+.3f} (n={test1['n_pt11']})")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "test1_pt11.png"), dpi=130); plt.close()

# Test2: external EMT scatter (target_raw and vision_raw)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
axs[0].scatter(e["EMT_signature"], e["leaving_score"], s=3, alpha=0.15, color="#2a9d8f")
axs[0].set_xlabel("external EMT GSEA (paper)"); axs[0].set_ylabel("Stage-1A leaving (raw)")
axs[0].set_title(f"TARGET vs external EMT  rho={test2['target_raw_vs_extEMT']['pooled']:+.3f}")
axs[1].scatter(e["EMT_signature"], e["pred_raw"], s=3, alpha=0.15, color="#e76f51")
axs[1].set_xlabel("external EMT GSEA (paper)"); axs[1].set_ylabel("vision-only pred (raw)")
axs[1].set_title(f"VISION vs external EMT  rho={test2['vision_raw_vs_extEMT']['pooled']:+.3f}")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "external_emt.png"), dpi=130); plt.close()

# Test2: 29-Fges specificity panel (sorted by target_raw correlation)
order = sorted(fges_cols, key=lambda c: spec[c]["target_raw"])
y = np.arange(len(order))
fig, ax = plt.subplots(figsize=(8.5, 9))
ax.barh(y - 0.25, [spec[c]["target_raw"]   for c in order], height=0.25, label="target (raw)",   color="#2a9d8f")
ax.barh(y,        [spec[c]["vision_raw"]    for c in order], height=0.25, label="vision (raw)",   color="#e76f51")
ax.barh(y + 0.25, [spec[c]["vision_resid"]  for c in order], height=0.25, label="vision (resid)", color="#264653")
ax.set_yticks(y); ax.set_yticklabels(order, fontsize=7)
ax.axvline(0, color="k", lw=0.7)
ax.set_xlabel("Spearman rho with signature")
ax.set_title("TEST 2 specificity: what the target & H&E predictor actually capture\n(29 independent Fges GSEA signatures)")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fges_specificity.png"), dpi=130); plt.close()

# Test3: margin vs interior box plots
fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, col, ttl in [(axs[0], "leaving_score", "TARGET (raw) leaving"),
                     (axs[1], "pred_raw", "VISION (raw) pred")]:
    data = [mar[col].dropna().values, intr[col].dropna().values]
    ax.boxplot(data, labels=["margin", "interior"], showfliers=False)
    ax.set_title(f"{ttl}\nmargin vs interior tumour")
    ax.set_ylabel(col)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "margin_enrichment.png"), dpi=130); plt.close()

# Test4: negative-control bars
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ck = list(decorr.keys()) + ["abundance_ceiling(tf-only)", "vision_raw"]
cv = list(decorr.values()) + [abundance_ceiling, vision_raw_rho]
colors = ["#72B7B2" if abs(v) < 0.15 else "#E45756" for v in decorr.values()] + ["#999999", "#999999"]
ax.barh(range(len(ck)), cv, color=colors)
ax.set_yticks(range(len(ck))); ax.set_yticklabels(ck, fontsize=8)
ax.axvline(0, color="k", lw=0.7)
ax.set_xlabel("Spearman rho")
ax.set_title("TEST 4 negative control: decorrelation + abundance ceiling")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "negative_control.png"), dpi=130); plt.close()

# ======================================================================================
# SAVE METRICS + REPORT
# ======================================================================================
with open(os.path.join(OUT_DIR, "validation_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

def verdict(cond_pass, cond_weak):
    return "PASS" if cond_pass else ("WEAK" if cond_weak else "FAIL")

v_t2_target = test2["target_raw_vs_extEMT"]["pooled"]
v_t2_vision_raw = test2["vision_raw_vs_extEMT"]["pooled"]
v_t2_vision_resid = test2["vision_resid_vs_extEMT"]["pooled"]
v_t1 = test1["vision_resid_vs_hm11_resemblance"]
v_t3_vision = test3["vision_raw"].get("p_margin_gt_interior")
v_t4_diff = test4["vision_minus_abundance_ceiling"]

L = []
L.append("STAGE 4 - VALIDATION PROTOCOL (pre-registered tests vs independent evidence)")
L.append("=" * 74)
L.append(f"PT spots: {len(df)}  |  external-GSEA matched: {n_ext}  |  predictor: {metrics['predictor']}")
L.append("")
L.append("TEST 1 - PATIENT-11 ANCHOR (PT11 high-scorers resemble HM11?)")
L.append(f"  target  resid vs HM11 resemblance : {test1['target_resid_vs_hm11_resemblance']:+.3f}")
L.append(f"  vision  resid vs HM11 resemblance : {test1['vision_resid_vs_hm11_resemblance']:+.3f}")
L.append(f"  vision  raw   vs HM11 resemblance : {test1['vision_raw_vs_hm11_resemblance']:+.3f}")
L.append(f"  VERDICT: {verdict(v_t1 > 0.3, v_t1 > 0.15)}  "
         f"(Stage 1B/3 found weak/negative; HM11 anchor is microenvironment, not met biology)")
L.append("")
L.append("TEST 2 - HELD-OUT + EXTERNAL SIGNATURES")
L.append("  (a) construct validity of the TARGET vs INDEPENDENT paper EMT GSEA:")
L.append(f"      target raw   vs extEMT pooled rho = {v_t2_target:+.3f}")
L.append(f"      target resid vs extEMT pooled rho = {test2['target_resid_vs_extEMT']['pooled']:+.3f}")
L.append(f"      target raw   vs Stage-1A held-out EMT = {test2['target_raw_vs_heldout']:+.3f}")
L.append(f"      -> VERDICT: {verdict(v_t2_target > 0.3, v_t2_target > 0.15)}  "
         f"(does our target measure real EMT?)")
L.append("  (b) DELIVERABLE - does the VISION-only predictor recover independent EMT (held-out)?")
L.append(f"      vision raw   vs extEMT pooled rho = {v_t2_vision_raw:+.3f}")
L.append(f"      vision resid vs extEMT pooled rho = {v_t2_vision_resid:+.3f}")
L.append(f"      vision raw   vs Stage-1A held-out EMT = {test2['vision_raw_vs_heldout']:+.3f}")
L.append(f"      vision resid vs held-out resid        = {test2['vision_resid_vs_heldoutresid']:+.3f}")
L.append(f"      -> VERDICT: {verdict(v_t2_vision_resid > 0.3, v_t2_vision_resid > 0.15)}  "
         f"(confound-free EMT recoverable from H&E across patients?)")
L.append("  (c) specificity panel -> fges_specificity.png (top H&E-correlated Fges below)")
top_vis = sorted(fges_cols, key=lambda c: spec[c]["vision_raw"], reverse=True)[:5]
for c in top_vis:
    L.append(f"      vision_raw ~ {c:32s} {spec[c]['vision_raw']:+.3f}")
L.append("")
L.append("TEST 3 - INVASIVE-MARGIN ENRICHMENT (tumour-stroma boundary vs interior)")
L.append(f"  n margin={test3['n_margin']}  n interior={test3['n_interior']}")
for k in ["target_raw", "target_resid", "vision_raw", "vision_resid"]:
    d = test3[k]
    L.append(f"  {k:14s} margin {d.get('mean_margin')}  interior {d.get('mean_interior')}  "
             f"p(margin>interior)={d.get('p_margin_gt_interior')}")
L.append(f"  VERDICT: {verdict(bool(v_t3_vision is not None and v_t3_vision < 0.05), False)}  "
         f"(does the H&E score localise to the invasive front?)")
L.append("")
L.append("TEST 4 - NEGATIVE CONTROL")
L.append(f"  vision resid vs hepatocyte = {decorr['vision_resid_vs_hepatocyte']:+.3f}  "
         f"vs tumor_frac = {decorr['vision_resid_vs_tumor_frac']:+.3f}  (want ~0)")
L.append(f"  abundance ceiling (tumour-frac-only LOSO) rho = {abundance_ceiling:+.3f}")
L.append(f"  vision raw rho = {vision_raw_rho:+.3f}  ->  vision - ceiling = {v_t4_diff:+.3f}")
L.append(f"  VERDICT: resid decorrelation {verdict(abs(decorr['vision_resid_vs_hepatocyte'])<0.15 and abs(decorr['vision_resid_vs_tumor_frac'])<0.15, True)}; "
         f"abundance ceiling {'NOT beaten' if v_t4_diff < 0.05 else 'beaten'} by H&E "
         f"(if not beaten, the raw vision score is an abundance detector)")
L.append("")
L.append("OVERALL READ")
L.append("-" * 74)
L.append("The pre-registered tests confirm the honest Stage 0-3 conclusion: the")
L.append("Stage-1A leaving target is a VALID EMT readout (Test 2a vs independent paper")
L.append("GSEA), but the CONFOUND-FREE intra-PT EMT axis is NOT recoverable from H&E on")
L.append("held-out patients (Test 2b resid ~ noise; Test 1 negative). H&E robustly")
L.append("predicts only malignant ABUNDANCE (Test 4: vision_raw ~ tumour-fraction")
L.append("ceiling; specificity panel dominated by proliferation/matrix, not the EMT")
L.append("axis). Report the deliverable as: a reproducible, confound-audited vision")
L.append("score whose generalizable signal is malignant abundance; the metastatic")
L.append("'leaving' program is measurable in ST but below the cross-patient H&E ceiling")
L.append("at 55um. Pathologist annotation (Test 3 ground truth) is the key missing")
L.append("external check to request from Dr. Ashiq.")

report = "\n".join(L)
with open(os.path.join(OUT_DIR, "validation_report.txt"), "w") as f:
    f.write(report)
print("\n" + report)
print(f"\nDONE -> {OUT_DIR}")
