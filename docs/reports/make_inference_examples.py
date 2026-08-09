"""
Worked inference example: predict gene expression from H&E alone on a section the
model has never seen, and show what that looks like spot by spot.

Panels
  a  real H&E tiles from the held-out section, with the expression actually
     measured underneath each one
  b  the same section mapped twice: measured expression, then predicted from
     image only
  c  predicted against measured, one point per spot
  d  which genes the image predicts well and which it does not

Everything here comes from a section held out of training, so nothing is fitted
to the tissue being shown.
"""

import json
import os

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HEST = os.path.join(ROOT, "dataset", "external", "HEST")
DEST = os.path.join(ROOT, "docs", "reports", "figures")
os.makedirs(DEST, exist_ok=True)

INK, INK2, MUTED, SURF = "#12161d", "#4a5361", "#8b93a1", "#ffffff"
SEQ = LinearSegmentedColormap.from_list("seq", ["#eef3f9", "#9dc3ec", "#2a78d6", "#12335e"])
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF, "font.size": 9,
                     "axes.edgecolor": "#d5d5d0", "text.color": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})

ORGAN = "Bowel"      # most sections, so the model has something to learn from
N_GENES = 50


def load_expr(sid):
    with h5py.File(os.path.join(HEST, "st", f"{sid}.h5ad"), "r") as f:
        X = f["X"]
        if isinstance(X, h5py.Dataset):
            M = sp.csr_matrix(np.asarray(X))
        else:
            shp = tuple(X.attrs["shape"])
            enc = X.attrs.get("encoding-type", "csr_matrix")
            enc = enc.decode() if isinstance(enc, bytes) else enc
            ctor = sp.csc_matrix if enc == "csc_matrix" else sp.csr_matrix
            M = ctor((X["data"][:], X["indices"][:], X["indptr"][:]), shape=shp)
        genes = np.array([g.decode() if isinstance(g, bytes) else str(g)
                          for g in f["var"]["_index"][:]])
        bc = np.array([b.decode() if isinstance(b, bytes) else str(b)
                       for b in f["obs"]["_index"][:]])
    _, first = np.unique(genes, return_index=True)
    k = np.zeros(len(genes), bool); k[first] = True
    return M.tocsc()[:, k].tocsr(), bc, genes[k]


def lognorm(M, idx):
    lib = np.asarray(M.sum(1)).ravel(); lib[lib == 0] = 1
    return np.asarray(sp.csr_matrix(M[:, idx].multiply(1e4 / lib[:, None])).log1p().todense())


meta = pd.read_csv(os.path.join(HEST, "HEST_v1_1_0.csv"), low_memory=False)
idc = "id" if "id" in meta.columns else meta.columns[0]
emb = {f[:-len("_uni2h.npz")] for f in os.listdir(os.path.join(HEST, "embeddings"))}
sel = meta[(meta.organ == ORGAN) & (meta.st_technology == "Visium")
           & (meta[idc].astype(str).isin(emb))]
ids = sel[idc].astype(str).tolist()

data, genesets = {}, None
for sid in ids:
    try:
        M, bc, genes = load_expr(sid)
    except Exception:
        continue
    if len(genes) < 5000:
        continue
    data[sid] = (M, bc, genes)
    genesets = set(genes) if genesets is None else (genesets & set(genes))
common = np.array(sorted(genesets))

# panel = genes that actually vary in this organ
acc = []
for sid, (M, bc, genes) in data.items():
    gi = pd.Index(genes).get_indexer(common)
    acc.append(lognorm(M, gi).var(0))
panel = common[np.argsort(-np.mean(acc, 0))[:N_GENES]]

X, Y, S = {}, {}, {}
for sid, (M, bc, genes) in data.items():
    z = np.load(os.path.join(HEST, "embeddings", f"{sid}_uni2h.npz"), allow_pickle=True)
    ebc = np.array([str(b) for b in z["barcode"]])
    pos = pd.Index(bc).get_indexer(ebc)
    keep = pos >= 0
    gi = pd.Index(genes).get_indexer(panel)
    X[sid] = z["X"][keep].astype(np.float32)
    Y[sid] = lognorm(M, gi)[pos[keep]].astype(np.float32)
    with h5py.File(os.path.join(HEST, "patches", f"{sid}.h5"), "r") as f:
        S[sid] = f["coords"][:][keep]

held = max(data, key=lambda s: X[s].shape[0])
train = [s for s in data if s != held]
print(f"organ={ORGAN}  train on {len(train)} sections, held out {held} "
      f"({X[held].shape[0]} spots)")

model = make_pipeline(StandardScaler(), Ridge(alpha=1000.0))
model.fit(np.vstack([X[s] for s in train]), np.vstack([Y[s] for s in train]))
P = model.predict(X[held])
Yh = Y[held]
r = np.array([np.corrcoef(Yh[:, g], P[:, g])[0, 1] if Yh[:, g].std() > 1e-8 else np.nan
              for g in range(len(panel))])
best = int(np.nanargmax(r))
gene = panel[best]
print(f"best-predicted gene: {gene}  r={r[best]:.3f}   median r={np.nanmedian(r):.3f}")

with h5py.File(os.path.join(HEST, "patches", f"{held}.h5"), "r") as f:
    imgs = f["img"][:]
    bc_p = np.array([b[0].decode() if isinstance(b[0], bytes) else str(b[0])
                     for b in f["barcode"][:]])
Mh, bch, gh = data[held]
pos = pd.Index(bch).get_indexer(bc_p)
imgs = imgs[pos >= 0]

fig = plt.figure(figsize=(13.4, 9.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.25], hspace=.34, wspace=.26,
                      left=.05, right=.97, top=.90, bottom=.06)

# ---------------------------------------------------------------- a
axa = fig.add_subplot(gs[0, :])
axa.axis("off")
order = np.argsort(Yh[:, best])
picks = [order[2], order[len(order) // 3], order[2 * len(order) // 3], order[-3]]
for k, sp_i in enumerate(picks):
    sub = axa.inset_axes([0.045 + k * 0.245, 0.18, 0.185, 0.74])
    sub.imshow(imgs[sp_i]); sub.set_xticks([]); sub.set_yticks([])
    for s_ in sub.spines.values():
        s_.set_edgecolor("#d5d5d0")
    sub.set_title(f"measured {gene}\n{Yh[sp_i, best]:.2f}   predicted {P[sp_i, best]:.2f}",
                  fontsize=8.6, color=INK2, pad=6)
axa.text(0, 1.06, f"a   Four real H&E tiles from the held-out section, ordered by how much "
                  f"{gene} they contain",
         transform=axa.transAxes, fontsize=11.5, fontweight="bold", va="bottom")
axa.text(0, 0.02, "Each tile is 224 x 224 pixels and covers one spot. The model sees only the "
                  "image and returns a number; the measured value is shown beside it.",
         transform=axa.transAxes, fontsize=8.6, color=INK2)

# ---------------------------------------------------------------- b
xy = S[held].astype(float)
for k, (vals, name) in enumerate([(Yh[:, best], "measured"), (P[:, best], "predicted from H&E")]):
    ax = fig.add_subplot(gs[1, k])
    v = (vals - np.percentile(vals, 2)) / (np.percentile(vals, 98) - np.percentile(vals, 2) + 1e-9)
    ax.scatter(xy[:, 0], -xy[:, 1], c=np.clip(v, 0, 1), cmap=SEQ, s=6, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_title(f"{'b' if k == 0 else ''}   {gene}, {name}" if k == 0 else f"{gene}, {name}",
                 fontsize=11.5 if k == 0 else 10, fontweight="bold" if k == 0 else "normal",
                 loc="left", pad=8)

# ---------------------------------------------------------------- c
ax = fig.add_subplot(gs[1, 2])
ax.scatter(Yh[:, best], P[:, best], s=7, color="#2a78d6", alpha=.35, linewidths=0)
ax.set_xlabel(f"measured {gene} (log normalised)")
ax.set_ylabel("predicted from H&E")
ax.grid(alpha=.4)
ax.set_title(f"c   one point per spot, r = {r[best]:.2f}", fontsize=11.5,
             fontweight="bold", loc="left", pad=8)
lo = min(Yh[:, best].min(), P[:, best].min()); hi = max(Yh[:, best].max(), P[:, best].max())
ax.plot([lo, hi], [lo, hi], ls="--", color=MUTED, lw=1)

fig.suptitle(f"Predicting expression from H&E on a section the model never saw "
             f"({ORGAN.lower()}, {X[held].shape[0]} spots)",
             fontsize=13, fontweight="bold", x=.05, ha="left", y=.975)
fig.savefig(os.path.join(DEST, "fig_inference_example.png"), dpi=190, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- d
fig, ax = plt.subplots(figsize=(9.0, 6.2))
o = np.argsort(-np.nan_to_num(r, nan=-9))
top, bot = o[:12], o[-8:]
sel_i = np.concatenate([top, bot])
y = np.arange(len(sel_i))
cols = ["#1baf7a"] * len(top) + ["#eb6834"] * len(bot)
ax.barh(y, r[sel_i], color=cols, height=.72, zorder=3)
ax.set_yticks(y); ax.set_yticklabels(panel[sel_i], fontsize=9)
ax.invert_yaxis(); ax.axvline(0, color=INK2, lw=1)
ax.set_xlabel("correlation between predicted and measured, held-out section")
ax.grid(axis="x", alpha=.4)
for s_ in ["top", "right"]:
    ax.spines[s_].set_visible(False)
ax.set_title("d   The image predicts some genes well and others not at all",
             fontsize=12, fontweight="bold", loc="left", pad=10)
ax.text(0, 1.02, "Green: best predicted. Orange: worst. Genes tied to visible structure are "
                 "easier than genes that are not.",
        transform=ax.transAxes, fontsize=8.8, color=INK2, va="bottom")
fig.tight_layout()
fig.savefig(os.path.join(DEST, "fig_inference_genes.png"), dpi=190, bbox_inches="tight")
plt.close(fig)

json.dump({"organ": ORGAN, "held_out": held, "n_train_sections": len(train),
           "n_spots": int(X[held].shape[0]), "best_gene": str(gene),
           "best_r": float(r[best]), "median_r": float(np.nanmedian(r)),
           "top_genes": {str(panel[i]): float(r[i]) for i in top},
           "worst_genes": {str(panel[i]): float(r[i]) for i in bot}},
          open(os.path.join(DEST, "inference_example.json"), "w"), indent=2)
print("wrote inference figures ->", DEST)
