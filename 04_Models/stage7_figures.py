"""Figures for Stage 7 -- external replication in GSE274557."""
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "Outputs", "stage7_external_replication")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

C_HM, C_LU, C_PM, C_T = "#eb6834", "#2a78d6", "#1baf7a", "#8a8a85"
INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#9a9a94", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": "#d5d5d0", "axes.linewidth": .8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.labelcolor": INK2, "font.size": 9.5,
    "grid.color": "#e8e8e3", "grid.linewidth": .7})


def titled(ax, title, sub):
    n = sub.count("\n") + 1
    ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=8.6, color=INK2,
            va="bottom", ha="left", linespacing=1.45)
    ax.text(0, 1.03 + .052 * n + .035, title, transform=ax.transAxes, fontsize=11.5,
            fontweight="bold", color=INK, va="bottom", ha="left")


PB = pd.read_csv(os.path.join(OUT, "pseudobulk_sections.csv"))
ORGAN = ["Hepatocyte", "Lung", "Mesothelial"]
IMM = ["Bcells", "Tcells", "Plasma", "Myeloid", "NK"]
FEAT = [c for c in PB.columns if c not in ("sample", "patient", "site", "Epithelial")]
core = [f for f in FEAT if f not in ORGAN]
prim = PB[PB.site == "T"][core].mean(); sd = PB[core].std() + 1e-9
sh = pd.DataFrame({s: ((PB[PB.site == s][core].mean() - prim) / sd) for s in ["HM", "LuM", "PM"]})

# ---------------------------------------------------------------- FIG 6
fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), gridspec_kw={"width_ratios": [1, 1.35]})
ax = axes[0]
z = lambda v: v / (np.linalg.norm(v) + 1e-12)
pairs = [("HM", "LuM", "liver ~ lung"), ("HM", "PM", "liver ~ peritoneal"),
         ("LuM", "PM", "lung ~ peritoneal")]
vals = [float(z(sh[a].to_numpy()) @ z(sh[b].to_numpy())) for a, b, _ in pairs]
labs = [l for _, _, l in pairs]
ypos = np.arange(len(vals) + 1)
allv = vals + [0.547]
alll = labs + ["OUR COHORT\nliver ~ lymph node"]
cols = [C_HM, C_HM, C_LU, "#111111"]
ax.barh(ypos, allv, color=cols, height=.62, zorder=3)
ax.axvline(0, color=INK2, lw=1.2)
for i, v in enumerate(allv):
    ax.text(v + (.03 if v > 0 else -.03), i, f"{v:+.2f}", va="center",
            ha="left" if v > 0 else "right", fontsize=9.5, fontweight="bold", color=INK)
ax.set_yticks(ypos); ax.set_yticklabels(alll, fontsize=9)
ax.set_xlim(-.55, .75)
ax.set_xlabel("alignment of the two sites' metastatic shift (cosine)")
ax.grid(axis="x")
ax.text(.60, -.62, "1.0 = identical\nprogram", fontsize=8, color=MUTED, ha="center")
titled(ax, "Fig 6  Independent cohort, same conclusion — stronger",
       "If metastasis were one program these bars would approach 1.0. In our cohort two sites\n"
       "shared about half their shift. In GSE274557 three site pairs share nothing at all —\n"
       "some point in opposite directions.")

ax = axes[1]
order = sh.assign(r=sh.max(1) - sh.min(1)).sort_values("r").index
y = np.arange(len(order))
ax.axvline(0, color=INK2, lw=1.2, zorder=2)
for s, c, lab in [("HM", C_HM, "liver"), ("LuM", C_LU, "lung"), ("PM", C_PM, "peritoneal")]:
    ax.scatter(sh.loc[order, s], y, s=64, color=c, edgecolors=SURF, linewidths=1.4,
               zorder=4, label=lab)
for i, f in enumerate(order):
    ax.plot([sh.loc[f].min(), sh.loc[f].max()], [i, i], color="#d5d5d0", lw=1.6, zorder=3)
ax.set_yticks(y)
ax.set_yticklabels([f + ("  •" if f in IMM else "") for f in order], fontsize=9)
ax.set_xlabel("shift from primary tumour  (SD units)")
# widen the right margin so the legend has empty space instead of sitting on a data point
lo, hi = sh.to_numpy().min(), sh.to_numpy().max()
ax.set_xlim(lo - 0.15, hi + 0.75)
ax.legend(frameon=False, fontsize=9, loc="lower right", title="metastatic site",
          title_fontsize=8.5)
ax.grid(axis="x")
titled(ax, "Fig 7  Liver is immune-cold, lung is immune-hot",
       "Each row is one feature; the three dots are the three destinations. Immune rows (•) spread\n"
       "widest and in OPPOSITE directions — immune falls in liver metastases and rises in lung.\n"
       "That opposition is why the alignment in Fig 6 is negative.")
fig.tight_layout(rect=[0, 0, 1, .84])
fig.savefig(os.path.join(FIG, "fig6_replication.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote", os.path.join(FIG, "fig6_replication.png"))
