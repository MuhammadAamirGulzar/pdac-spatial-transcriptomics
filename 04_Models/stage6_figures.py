"""
Figures for Stage 6 -- site-specificity of the PDAC metastatic program.

Every panel is built to be read by someone who is not a pathologist: each carries
a plain-language subtitle stating what the panel shows and what it means, and all
series are direct-labelled so identity never depends on colour alone.

Run after stage6_site_specific_program.py:
    COHORT=full python 04_Models/stage6_figures.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("COHORT", "full")
from _cohort import ROOT, CELL_TYPES, PATIENT_OF, SITE_OF, ALL_SAMPLES

OUT = os.path.join(ROOT, "Outputs", "stage6_site_program")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

# validated categorical palette (light surface #fcfcfb): all checks PASS
C_LNM = "#2a78d6"   # slot 1 blue  - lymph-node metastasis
C_HM = "#eb6834"    # slot 2 orange - liver metastasis
C_T = "#1baf7a"     # slot 3 aqua   - primary tumour
C_NP = "#8a8a85"    # neutral       - normal pancreas
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#9a9a94"
SURF = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": "#d5d5d0", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.labelcolor": INK2, "font.size": 9.5,
    "grid.color": "#e8e8e3", "grid.linewidth": 0.7,
})


def titled(ax, title, sub):
    """Title above subtitle, both left-aligned over the axes.

    Written with explicit axes-fraction offsets rather than ax.set_title(): the
    subtitles here run 2-3 lines and set_title() does not reserve room for them,
    so the two texts overprinted each other.
    """
    n = sub.count("\n") + 1
    y_sub = 1.03
    ax.text(0, y_sub, sub, transform=ax.transAxes, fontsize=8.6, color=INK2,
            va="bottom", ha="left", linespacing=1.45)
    ax.text(0, y_sub + 0.052 * n + 0.035, title, transform=ax.transAxes,
            fontsize=11.5, fontweight="bold", color=INK, va="bottom", ha="left")


PB = pd.read_csv(os.path.join(OUT, "pseudobulk_sections.csv"))
PT = pd.read_csv(os.path.join(OUT, "pseudobulk_primary.csv"))
SW = pd.read_csv(os.path.join(OUT, "purity_sweep.csv"))
R = pd.read_csv(os.path.join(OUT, "site_differential.csv"))
M = json.load(open(os.path.join(OUT, "metrics.json")))
PAIRED = M["paired_patients"]

# =====================================================================  FIG 1
fig, ax = plt.subplots(figsize=(8.4, 4.6))
sites = ["T", "HM", "LNM", "NP"]
site_lab = {"T": "Primary\n(pancreas)", "HM": "Liver\nmetastasis",
            "LNM": "Lymph-node\nmetastasis", "NP": "Normal\npancreas"}
cols = {"T": C_T, "HM": C_HM, "LNM": C_LNM, "NP": C_NP}
pats = sorted({PATIENT_OF[s] for s in ALL_SAMPLES}, key=lambda p: int(p.split("_")[1]))
for xi, p in enumerate(pats):
    for yi, st in enumerate(sites):
        n = sum(1 for s in ALL_SAMPLES if PATIENT_OF[s] == p and SITE_OF[s] == st)
        if n:
            ax.scatter([xi], [yi], s=210, c=cols[st], edgecolors=SURF, linewidths=2, zorder=3)
            if n > 1:
                ax.text(xi, yi, str(n), ha="center", va="center", color="white",
                        fontsize=8, fontweight="bold", zorder=4)
for xi, p in enumerate(pats):
    if p in PAIRED:
        ax.axvspan(xi - 0.42, xi + 0.42, color="#2a78d6", alpha=0.09, zorder=0)
ax.set_xticks(range(len(pats)))
ax.set_xticklabels([p.replace("PT_", "P") for p in pats])
ax.set_yticks(range(len(sites)))
ax.set_yticklabels([site_lab[s] for s in sites], fontsize=8.6)
ax.set_xlabel("patient")
ax.set_ylim(-0.6, len(sites) - 0.4); ax.set_xlim(-0.7, len(pats) - 0.3)
ax.grid(axis="y", zorder=0)
titled(ax, "Fig 1  What tissue we have, per patient",
       "One dot = one tissue section. Shaded patients (P6, P8, P10, P12) gave BOTH a liver and a\n"
       "lymph-node metastasis - those four allow a within-person comparison, where patient identity cancels out.")
ax.text(0.995, 0.03, "30 sections · 13 patients", transform=ax.transAxes,
        ha="right", fontsize=8, color=MUTED)
fig.tight_layout(rect=[0,0,1,0.84]); fig.savefig(os.path.join(FIG, "fig1_cohort_design.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# =====================================================================  FIG 2
fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.9), gridspec_kw={"width_ratios": [1.2, 1, 1]})
ax = axes[0]
ax.plot(SW.purity, SW.lymphoid_lnm, "-o", color=C_LNM, lw=2, ms=8,
        markeredgecolor=SURF, markeredgewidth=1.5, zorder=3)
ax.plot(SW.purity, SW.lymphoid_hm, "-o", color=C_HM, lw=2, ms=8,
        markeredgecolor=SURF, markeredgewidth=1.5, zorder=3)
ax.text(SW.purity.iloc[-1] + 0.012, SW.lymphoid_lnm.iloc[-1], "lymph-node met",
        color=C_LNM, fontsize=9, fontweight="bold", va="center")
ax.text(SW.purity.iloc[-1] + 0.012, SW.lymphoid_hm.iloc[-1], "liver met",
        color=C_HM, fontsize=9, fontweight="bold", va="center")
ax.set_xlabel("minimum tumour purity of a spot (RCTD tumour fraction)")
ax.set_ylabel("lymphoid cell fraction")
ax.set_xlim(0.26, 0.94); ax.grid(axis="y")
titled(ax, "Fig 2a  The control that decides the finding",
       "Lymph nodes are made of immune cells, so any whole-section comparison would be trivially\n"
       "'positive'. Here we keep only increasingly tumour-pure spots. The gap does NOT close.")
# One measure per axis: a ratio and a fraction cannot share a y-scale (the
# hepatocyte bars were invisible next to a 2.4x ratio), so they get their own panels.
ax = axes[1]
ax.bar(np.arange(len(SW)), SW.ratio, width=0.55, color=C_LNM, zorder=3)
ax.axhline(1.0, color=INK2, lw=1.4, ls="--", zorder=4)
ax.text(len(SW) - 0.45, 1.06, "1.0 = no difference", fontsize=8.2, color=INK2, ha="right")
for i, r in enumerate(SW.itertuples()):
    ax.text(i, r.ratio + 0.06, f"{r.ratio:.1f}x", ha="center", fontsize=9,
            color=INK, fontweight="bold")
ax.set_xticks(range(len(SW)))
ax.set_xticklabels([f"≥{p:.2f}" for p in SW.purity])
ax.set_xlabel("minimum tumour purity")
ax.set_ylabel("lymphoid fraction ratio  (nodal ÷ liver)")
ax.set_ylim(0, max(SW.ratio) * 1.25); ax.grid(axis="y")
titled(ax, "Fig 2b  The effect does not shrink",
       "Nodal mets hold ~2x the lymphoid content of liver mets, and that holds even in the\n"
       "purest tumour regions. A contamination artefact would decay towards the dashed line.")

ax = axes[2]
ax.bar(np.arange(len(SW)), SW.hepatocyte_hm, width=0.55, color=C_HM, zorder=3)
for i, r in enumerate(SW.itertuples()):
    ax.text(i, r.hepatocyte_hm + 0.004, f"{r.hepatocyte_hm:.3f}", ha="center",
            fontsize=9, color=INK, fontweight="bold")
ax.set_xticks(range(len(SW)))
ax.set_xticklabels([f"≥{p:.2f}" for p in SW.purity])
ax.set_xlabel("minimum tumour purity")
ax.set_ylabel("hepatocyte fraction in liver mets")
ax.set_ylim(0, max(SW.hepatocyte_hm) * 1.3); ax.grid(axis="y")
titled(ax, "Fig 2c  Proof the filter works",
       "Liver-cell contamination falls 8x (0.133 → 0.017) across the same thresholds. The filter\n"
       "is removing host tissue exactly as intended — which is why Fig 2b is trustworthy.")
fig.tight_layout(rect=[0,0,1,0.86]); fig.savefig(os.path.join(FIG, "fig2_purity_control.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# =====================================================================  FIG 3
key = "rctd::B cells"
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), gridspec_kw={"width_ratios": [1, 1.25]})
ax = axes[0]
for p in PAIRED:
    h = PB[(PB.patient == p) & (PB.site == "HM")][key].mean()
    l = PB[(PB.patient == p) & (PB.site == "LNM")][key].mean()
    ax.plot([0, 1], [h, l], "-", color=MUTED, lw=1.4, zorder=2)
    ax.scatter([0], [h], s=110, color=C_HM, edgecolors=SURF, linewidths=1.6, zorder=3)
    ax.scatter([1], [l], s=110, color=C_LNM, edgecolors=SURF, linewidths=1.6, zorder=3)
    ax.text(1.05, l, p.replace("PT_", "P"), fontsize=8.5, color=INK2, va="center")
ax.set_xticks([0, 1]); ax.set_xticklabels(["liver met", "lymph-node met"])
ax.set_xlim(-0.35, 1.42)
ax.set_ylabel("B-cell fraction (tumour-rich spots)")
ax.grid(axis="y")
titled(ax, "Fig 3a  Same patient, both sites",
       "Each line is one patient measured twice. All 4 rise from liver to lymph node -\n"
       "the direction never reverses, which is what makes this credible with only 4 people.")
ax = axes[1]
d = {"HM": PB[PB.site == "HM"][key].to_numpy(), "LNM": PB[PB.site == "LNM"][key].to_numpy(),
     "T": PT[key].to_numpy()}
for i, (k, c, lab) in enumerate([("T", C_T, "primary"), ("HM", C_HM, "liver met"),
                                 ("LNM", C_LNM, "lymph-node met")]):
    v = d[k]
    ax.scatter(np.random.default_rng(0).normal(i, 0.055, len(v)), v, s=64, color=c,
               edgecolors=SURF, linewidths=1.5, zorder=3)
    ax.plot([i - 0.22, i + 0.22], [np.median(v)] * 2, color=INK, lw=2, zorder=4)
    ax.text(i, ax.get_ylim()[1], "", ha="center")
    ax.text(i, -0.008, lab, ha="center", fontsize=9, color=INK2, transform=ax.get_xaxis_transform())
ax.set_xticks([]); ax.set_ylabel("B-cell fraction")
ax.grid(axis="y")
row = R[R.feature == key].iloc[0]
titled(ax, "Fig 3b  All sections, not just the paired ones",
       f"Black bar = median. Effect size d = {row.cohens_d:+.2f}, FDR q = {row.q_unpaired:.3f} — the only\n"
       "immune feature that survives multiple-testing correction across 42 tested features.")
fig.tight_layout(rect=[0,0,1,0.85]); fig.savefig(os.path.join(FIG, "fig3_bcells_paired.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# =====================================================================  FIG 4
top = pd.concat([R.head(11), R.tail(11)]).drop_duplicates("feature")
top = top.sort_values("cohens_d")
fig, ax = plt.subplots(figsize=(9.2, 7.4))
colors = [C_LNM if v > 0 else C_HM for v in top.cohens_d]
ax.barh(np.arange(len(top)), top.cohens_d, color=colors, height=0.72, zorder=3)
ax.set_yticks(np.arange(len(top)))
ax.set_yticklabels([f.replace("rctd::", "cell: ") for f in top.feature], fontsize=8.6)
for i, r in enumerate(top.itertuples()):
    if r.q_unpaired < 0.10:
        ax.text(r.cohens_d + (0.09 if r.cohens_d > 0 else -0.09), i,
                f"q={r.q_unpaired:.3f} *", va="center",
                ha="left" if r.cohens_d > 0 else "right",
                fontsize=8.4, fontweight="bold", color=INK)
ax.axvline(0, color=INK2, lw=1)
ax.set_xlabel("effect size (Cohen's d)      ← higher in liver met      higher in lymph-node met →")
ax.grid(axis="x")
titled(ax, "Fig 4  What actually differs between the two metastatic sites",
       "42 features tested (27 expression signatures + 15 cell types), one value per section.\n"
       "Only 2 survive FDR correction (marked *): B cells up in nodal mets, hepatocytes up in liver mets —\n"
       "the latter is a positive control that proves the pipeline detects a difference it must detect.")
fig.tight_layout(rect=[0,0,1,0.88]); fig.savefig(os.path.join(FIG, "fig4_effect_sizes.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# =====================================================================  FIG 5
feat = [c for c in R.feature]
dHM = np.array([PB[PB.site == "HM"][c].mean() - PT[c].mean() for c in feat])
dLN = np.array([PB[PB.site == "LNM"][c].mean() - PT[c].mean() for c in feat])
sd = np.array([np.std(np.concatenate([PB[c].to_numpy(), PT[c].to_numpy()])) + 1e-9 for c in feat])
x, y = dHM / sd, dLN / sd
fig, ax = plt.subplots(figsize=(7.4, 6.6))
ax.axhline(0, color="#d5d5d0", lw=1); ax.axvline(0, color="#d5d5d0", lw=1)
lim = max(np.abs(x).max(), np.abs(y).max()) * 1.18
ax.plot([-lim, lim], [-lim, lim], "--", color=MUTED, lw=1.2, zorder=1)
ax.text(lim * 0.62, lim * 0.72, "perfectly shared\n(same change at both sites)",
        color=MUTED, fontsize=8.2, ha="center")
ax.scatter(x, y, s=52, color=C_T, edgecolors=SURF, linewidths=1.3, zorder=3)
for i, f in enumerate(feat):
    if abs(x[i]) > 0.8 or abs(y[i]) > 0.8:
        ax.annotate(f.replace("rctd::", ""), (x[i], y[i]), fontsize=7.6, color=INK2,
                    xytext=(4, 3), textcoords="offset points")
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel("change in LIVER met vs primary  (SD units)")
ax.set_ylabel("change in LYMPH-NODE met vs primary  (SD units)")
ax.grid(alpha=0.5)
titled(ax, "Fig 5  How much of 'becoming a metastasis' is the same at both sites?",
       f"Each dot is one feature. If metastasis were one universal program every dot would sit on the\n"
       f"dashed line. Observed alignment (cosine) = {M['cos_HMshift_LNMshift']:+.3f} → only "
       f"~{100*M['cos_HMshift_LNMshift']:.0f}% shared; the rest is site-specific.")
fig.tight_layout(rect=[0,0,1,0.86]); fig.savefig(os.path.join(FIG, "fig5_shared_vs_sitespecific.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

print("figures written to", FIG)
for f in sorted(os.listdir(FIG)):
    print("   ", f)
