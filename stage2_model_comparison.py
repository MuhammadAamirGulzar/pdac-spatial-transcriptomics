"""
Stage 2 — 4-backbone comparison + radar plots.
Reads the per-model stage2_decision_summary.json files and produces:
  - model_comparison.csv         (raw metrics, Variant A & B)
  - radar_variantA.png           (4 models on 6 higher=better axes, Variant A)
  - bars_raw_resid_AB.png        (decision-relevant: raw & resid leaving rho, A vs B)
"""
import json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("Outputs/stage2")
FILES = {
    "CONCH V1":    ROOT / "conch-v1/stage2_outputs/stage2_decision_summary.json",
    "CONCH V1.5":  ROOT / "conch-v1.5/stage2_CONCH-V1.5_outputs/stage2_decision_summary.json",
    "UNI2-h":      ROOT / "UNI2h/stage2_UNI2h_outputs/stage2_decision_summary.json",
    "H-Optimus-1": ROOT / "H-Optimus-1/stage2_outputs/stage2_decision_summary.json",
}
OUT = ROOT / "_model_comparison"
OUT.mkdir(exist_ok=True)

data = {m: json.load(open(f)) for m, f in FILES.items()}
models = list(FILES)

# ── flat table ───────────────────────────────────────────────────────────────
rows = []
for m in models:
    for v in ["A", "B"]:
        d = data[m][v]
        rows.append({
            "model": m, "variant": v,
            "raw_rho":   d["probe_rho_raw_mean"],
            "resid_rho": d["probe_rho_resid_mean"],
            "resid_r2":  d["probe_r2_resid_mean"],
            "vg_R1_excl": d["vg_R1_excl_mean"],
            "mean_val":   d["mean_val"],
            "mean_val_pt": d["mean_val_pt"],
            "mean_val_hm": d["mean_val_hm"],
        })
df = pd.DataFrame(rows)
df.to_csv(OUT / "model_comparison.csv", index=False)

print("="*92)
print("STAGE 2 — 4-BACKBONE COMPARISON  (held-out vision->leaving probe is the decision metric)")
print("="*92)
print("InfoNCE chance loss = ln(512) = 6.238  (val at/near this = no cross-patient alignment)\n")
show = df.copy()
for c in ["raw_rho", "resid_rho", "resid_r2", "mean_val", "mean_val_pt", "mean_val_hm"]:
    show[c] = show[c].map(lambda x: f"{x:+.3f}")
show["vg_R1_excl"] = show["vg_R1_excl"].map(lambda x: f"{x:.4f}")
print(show.to_string(index=False))

# Variant A leaders
A = df[df.variant == "A"].set_index("model")
print("\n-- Variant A leaders --")
print(f"  raw leaving rho  (signal that exists) : {A['raw_rho'].idxmax()}  ({A['raw_rho'].max():.3f})")
print(f"  resid leaving rho (confound-free)     : {A['resid_rho'].idxmax()}  ({A['resid_rho'].max():.3f})  [all ~noise]")
print(f"  best alignment (lowest val loss)      : {A['mean_val'].idxmin()}  ({A['mean_val'].min():.3f})")
print(f"  best v->g R@1 (neighbour-excluded)    : {A['vg_R1_excl'].idxmax()}  ({A['vg_R1_excl'].max():.4f})  [near chance]")

# ── radar (Variant A) ────────────────────────────────────────────────────────
# axes, all transformed so higher = better, then min-max scaled across models to [0.08,1]
axes_spec = [
    ("Raw leaving\nrho", lambda d: d["probe_rho_raw_mean"]),
    ("Resid leaving\nrho", lambda d: d["probe_rho_resid_mean"]),
    ("Resid R^2\n(higher)", lambda d: d["probe_r2_resid_mean"]),
    ("v->g R@1\n(excl)", lambda d: d["vg_R1_excl_mean"]),
    ("Alignment\n(-val)", lambda d: -d["mean_val"]),
    ("PT alignment\n(-val_pt)", lambda d: -d["mean_val_pt"]),
]
labels = [a[0] for a in axes_spec]
raw = np.array([[a[1](data[m]["A"]) for a in axes_spec] for m in models])  # [models, axes]
mn, mx = raw.min(0), raw.max(0)
norm = 0.08 + 0.92 * (raw - mn) / (mx - mn + 1e-12)

ang = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
ang += ang[:1]
COL = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
fig = plt.figure(figsize=(8.5, 8.5))
ax = plt.subplot(111, polar=True)
for i, m in enumerate(models):
    vals = norm[i].tolist(); vals += vals[:1]
    ax.plot(ang, vals, color=COL[i], lw=2, label=m)
    ax.fill(ang, vals, color=COL[i], alpha=0.08)
ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels, fontsize=10)
ax.set_yticks([]); ax.set_ylim(0, 1.05)
ax.set_title("Stage 2 — Vision backbone comparison (Variant A)\n"
             "axes min-max scaled across models; higher = better", fontsize=12, pad=24)
ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.10), fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "radar_variantA.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ── decision-relevant bars: raw & resid leaving rho, A vs B ───────────────────
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
x = np.arange(len(models)); w = 0.38
for ax, key, title in [(axs[0], "raw_rho", "Vision->RAW leaving rho (held-out PT)\n= signal that EXISTS (abundance-driven)"),
                       (axs[1], "resid_rho", "Vision->RESID leaving rho (held-out PT)\n= confound-free EMT (at noise floor)")]:
    a = [df[(df.model==m)&(df.variant=="A")][key].values[0] for m in models]
    b = [df[(df.model==m)&(df.variant=="B")][key].values[0] for m in models]
    ax.bar(x-w/2, a, w, label="A (baseline)", color="#4C72B0")
    ax.bar(x+w/2, b, w, label="B (met-aware)", color="#C44E52")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=15, fontsize=9)
    ax.set_title(title, fontsize=10); ax.axhline(0, color="k", lw=0.6); ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)
axs[1].set_ylim(axs[0].get_ylim())  # same scale -> visually shows resid << raw
fig.suptitle("Stage 2 decision metric — held-out vision->leaving Spearman rho", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "bars_raw_resid_AB.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved -> {OUT}/  (model_comparison.csv, radar_variantA.png, bars_raw_resid_AB.png)")
