"""
Figure for stage 9: the permutation nulls behind the site-specificity claim.

Panel A  discovery cohort. The null is centred well above zero because both site
         shifts are measured from the same primary centroid. The observed value
         sits inside it, and the within-lymph-node split-half ceiling sits on top
         of the observed value, which is the clearest way to see that this cohort
         cannot resolve the contrast.
Panel B  replication cohort. All three site pairs fall far below their nulls.

    python 04_Models/stage9_figures.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "Outputs", "stage9_cosine_nulls")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

INK, NULLC, OBS, CEIL = "#222222", "#9aa0b4", "#b3392f", "#2c6e63"

M = json.load(open(os.path.join(OUT, "metrics.json")))
D = M["discovery"]["raw (as published in stage 6)"]

fig = plt.figure(figsize=(11.6, 4.5))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.22], wspace=0.24)

# ---------------------------------------------------------------- panel A
ax = fig.add_subplot(gs[0, 0])
z = np.load(os.path.join(OUT, "discovery_null_arrays.npz"))
null, obs = z["null"], float(z["observed"])

ax.hist(null, bins=44, color=NULLC, edgecolor="white", linewidth=.4, alpha=.95,
        label=f"site-label null ({len(null)} exact assignments)")
ax.axvline(obs, color=OBS, lw=2.4, label=f"observed  {obs:+.3f}")
ax.axvline(D["detection_limit_p05"], color=INK, lw=1.3, ls="--",
           label=f"detection limit  {D['detection_limit_p05']:+.3f}")
ax.axvline(D["ceiling_LNM_median"], color=CEIL, lw=1.8, ls=":",
           label=f"within-LNM ceiling  {D['ceiling_LNM_median']:+.3f}")

ax.set_xlabel("alignment between the two site shifts (cosine)")
ax.set_ylabel("number of label assignments")
ax.set_title("A   Discovery cohort, GSE272362\n9 liver vs 5 lymph node sections\n"
             f"P(null $\\leq$ observed) = {D['p_lower_tail']:.3f}, not significant",
             fontsize=10.5, loc="left", color=INK)
ax.legend(fontsize=8, frameon=False, loc="upper left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# ---------------------------------------------------------------- panel B
ax2 = fig.add_subplot(gs[0, 1])
pairs = [("HM_vs_LuM", "liver vs lung"),
         ("HM_vs_PM", "liver vs peritoneum"),
         ("LuM_vs_PM", "lung vs peritoneum")]
rows = []
for key, lab in pairs:
    p = os.path.join(OUT, f"replication_null_{key}.npz")
    if not os.path.exists(p):
        continue
    zz = np.load(p)
    rows.append((lab, zz["null"], float(zz["observed"]), M["replication"][key]["p_lower_tail"]))

for i, (lab, nl, ob, pv) in enumerate(rows):
    y = len(rows) - 1 - i
    lo, hi = np.percentile(nl, [2.5, 97.5])
    ax2.plot([lo, hi], [y, y], color=NULLC, lw=7, solid_capstyle="butt",
             label="null, 2.5 to 97.5%" if i == 0 else None)
    ax2.plot([np.median(nl)], [y], marker="|", color="#5b6178", ms=13, mew=2,
             label="null median" if i == 0 else None)
    ax2.plot([ob], [y], marker="o", color=OBS, ms=9, zorder=3,
             label="observed" if i == 0 else None)
    ptxt = "p < 0.0001" if pv < 1e-4 else f"p = {pv:.4f}"
    ax2.annotate(f"{ob:+.3f}   {ptxt}", (ob, y), textcoords="offset points",
                 xytext=(0, 13), ha="center", fontsize=8.6, color=OBS)

ax2.axvline(0, color="#cfd2dc", lw=1, zorder=0)
ax2.set_yticks(range(len(rows)))
ax2.set_yticklabels([r[0] for r in rows][::-1], fontsize=9.5)
ax2.set_xlabel("alignment between the two site shifts (cosine)")
ax2.set_xlim(-0.55, 1.0)
ax2.set_ylim(-0.6, len(rows) - 0.15)
ax2.set_title("B   Replication cohort, GSE274557, treatment naive\n"
              "every pairing falls below its null", fontsize=10.5, loc="left", color=INK)
ax2.legend(fontsize=8, frameon=False, loc="lower right")
for s in ("top", "right", "left"):
    ax2.spines[s].set_visible(False)
ax2.tick_params(axis="y", length=0)

fig.savefig(os.path.join(FIG, "fig9_nulls.png"), dpi=200, bbox_inches="tight",
            facecolor="white")
print("wrote", os.path.join(FIG, "fig9_nulls.png"))
