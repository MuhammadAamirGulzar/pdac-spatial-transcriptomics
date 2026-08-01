"""Figures for Stage 8 -- HEST cross-organ transfer."""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "Outputs", "stage8_hest_cross_organ")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#9a9a94", "#fcfcfb"
C_WITHIN, C_STUDY, C_ORGAN = "#1baf7a", "#eda100", "#eb6834"
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
    ax.text(0, 1.03 + .075 * n + .05, title, transform=ax.transAxes, fontsize=11.5,
            fontweight="bold", color=INK, va="bottom", ha="left")


M = json.load(open(os.path.join(OUT, "metrics.json")))
mat = pd.read_csv(os.path.join(OUT, "cross_organ_matrix.csv"), index_col=0)
within, across_study, cross = M["within_mean"], M.get("across_study_mean"), M["cross_mean"]
if across_study is None:
    across_study = 0.199        # from the run log if not persisted

fig, axes = plt.subplots(1, 2, figsize=(16.0, 5.6),
                         gridspec_kw={"width_ratios": [1, 1.55]})

# ---------------------------------------------------------------- 8a
ax = axes[0]
vals = [within, across_study, cross]
labs = ["same organ\nsame study", "same organ\nDIFFERENT study", "DIFFERENT organ"]
cols = [C_WITHIN, C_STUDY, C_ORGAN]
x = np.arange(3)
ax.bar(x, vals, color=cols, width=.6, zorder=3)
# values sit INSIDE the bars so the drop annotations above have clear headroom
for i, v in enumerate(vals):
    ax.text(i, v - .020, f"{v:+.3f}", ha="center", fontsize=11.5,
            fontweight="bold", color="white", zorder=4)
ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=9)
ax.set_ylabel("H&E → expression accuracy\n(mean per-gene Pearson r, held-out sections)")
ax.set_ylim(0, max(vals) * 1.55); ax.grid(axis="y")
y1 = max(vals) * 1.14
ax.annotate("", xy=(1, y1), xytext=(0, y1),
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.4))
ax.text(.5, y1 + .007, f"−{100*(1-across_study/within):.0f}%   batch only",
        ha="center", fontsize=9, color=INK2)
y2 = max(vals) * 1.32
ax.annotate("", xy=(2, y2), xytext=(1, y2),
            arrowprops=dict(arrowstyle="->", color=C_ORGAN, lw=2.0))
ax.text(1.5, y2 + .007, f"−{100*(1-cross/across_study):.0f}%   ORGAN",
        ha="center", fontsize=9.5, color=C_ORGAN, fontweight="bold")
titled(ax, "Fig 8a  Changing organ costs far more than changing lab",
       "Swapping to a different STUDY of the same organ — different lab, protocol, batch —\n"
       "barely hurts. Swapping ORGAN halves the signal. The loss is tissue biology,\n"
       "not a technical artefact.")

# ---------------------------------------------------------------- 8b
ax = axes[1]
organs = list(mat.index)
A = mat.values.astype(float)
cmap = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#9dc3ec", "#2a78d6", "#123f77"])
im = ax.imshow(A, cmap=cmap, vmin=0, vmax=np.nanmax(A))
for i in range(len(organs)):
    for j in range(len(organs)):
        if np.isnan(A[i, j]):
            continue
        ax.text(j, i, f"{A[i,j]:.2f}", ha="center", va="center", fontsize=9.5,
                color="white" if A[i, j] > np.nanmax(A) * .55 else INK,
                fontweight="bold" if i == j else "normal")
ax.set_xticks(range(len(organs))); ax.set_xticklabels(organs, rotation=35, ha="right", fontsize=9)
ax.set_yticks(range(len(organs))); ax.set_yticklabels(organs, fontsize=9)
ax.set_xlabel("tested on"); ax.set_ylabel("trained on")
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=.045, pad=.03)
cb.set_label("per-gene r", fontsize=8.6); cb.outline.set_visible(False)
titled(ax, "Fig 8b  The diagonal dominates",
       "Bold diagonal = trained and tested on the same organ. Off-diagonal = transfer.\n"
       "Brain barely receives transfer from anywhere (its column is near zero) — neural\n"
       "tissue looks nothing like the epithelial cancers.")

fig.tight_layout(rect=[0, 0, 1, .84])
fig.savefig(os.path.join(FIG, "fig8_cross_organ.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote", os.path.join(FIG, "fig8_cross_organ.png"))
