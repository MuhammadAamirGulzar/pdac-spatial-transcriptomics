"""
scVI latent embeddings for the FULL 30-sample cohort.

Why this exists (and is not the notebook)
-----------------------------------------
`scvi-latent-embeddings.ipynb` trains on the 6 original slides and writes one
`.pt` per spot, named by *patch_stem* -- which it obtains by iterating the H&E
PNG patches.  The 24 new slides have no patches (the full-resolution WSIs are not
on this machine), so patch_stem does not exist for them and that output layout is
unavailable.

This script therefore keys the latents by **barcode**, the identifier every
`dataset/full_cohort/` artefact already shares (`rctd/`, `fges/`, `coords/`), and
writes one CSV per sample in the same shape as those.  Joining to patch_stem
remains possible later via `coords/` once patches exist.

Hyperparameters are copied verbatim from the notebook so the two runs are
comparable; the only intended difference is the training cohort (6 -> 30 slides,
which necessarily yields a different latent space -- a cohort-level model must be
retrained, not reused).

Input   dataset/full_cohort/scVI_counts/<sample>.csv   (already QC-filtered)
Output  dataset/full_cohort/gene_scvi/<sample>_scvi_latent.csv   barcode + 50 dims
        dataset/full_cohort/gene_scvi/_scvi_model/
        dataset/full_cohort/gene_scvi/training_history.csv

Run:
    C:/Users/datainsight/anaconda3/envs/stvi/python.exe \
        03_Embedding_Extraction/Gene/scvi_full_cohort.py
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
COUNTS_DIR = ROOT / "dataset" / "full_cohort" / "scVI_counts"
OUT_DIR = ROOT / "dataset" / "full_cohort" / "gene_scvi"

# ---- scVI hyperparameters: identical to scvi-latent-embeddings.ipynb ----
N_LATENT = 50          # must match the gene branch input dim in Phase A
N_HIDDEN = 128
N_LAYERS = 2
N_EPOCHS = 400
BATCH_SIZE = 256
SEED = 42
N_TOP_GENES = 3000

OUT_DIR.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_counts(path):
    """genes x barcodes CSV -> (barcodes, genes, CSR spots x genes float32).

    The matrix is ~90% zeros, so it goes to sparse immediately: dense float32 for
    the whole cohort would be 87k x 17.9k = 6.2 GB, and scVI only ever needs the
    sparse form.  Reading straight to float32 also halves the parse-side memory.
    """
    df = pd.read_csv(path, index_col=0, engine="c")
    genes = df.index.to_numpy()
    barcodes = df.columns.to_numpy()
    X = sp.csr_matrix(df.to_numpy(dtype=np.float32).T)   # spots x genes
    del df
    return barcodes, genes, X


def main():
    import anndata as ad
    import scanpy as sc
    import scvi
    import torch

    log(f"scvi-tools {scvi.__version__} | torch {torch.__version__} | "
        f"CUDA {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"  GPU: {torch.cuda.get_device_name(0)}")

    csvs = sorted(COUNTS_DIR.glob("*.csv"))
    csvs = [p for p in csvs if not p.stem.endswith("_qc_metrics")]
    if not csvs:
        sys.exit(f"no count CSVs in {COUNTS_DIR} -- run 04_Models/split_full_cohort.R first")
    log(f"found {len(csvs)} samples in {COUNTS_DIR}")

    adatas, ref_genes = [], None
    for p in csvs:
        sample = p.stem
        t0 = time.time()
        barcodes, genes, X = read_counts(p)
        if ref_genes is None:
            ref_genes = genes
        elif not np.array_equal(ref_genes, genes):
            sys.exit(f"FATAL: {sample} gene order differs from {csvs[0].stem}")
        a = ad.AnnData(X=X)
        a.obs_names = barcodes
        a.var_names = genes
        a.obs["sample"] = sample
        adatas.append(a)
        log(f"  {sample:14s} {X.shape[0]:5d} spots x {X.shape[1]} genes  "
            f"({time.time()-t0:.1f}s, {100*X.nnz/(X.shape[0]*X.shape[1]):.1f}% nonzero)")

    adata = ad.concat(adatas, join="inner")
    del adatas
    adata.obs_names_make_unique()
    n_unique = len(set(adata.obs_names))
    if n_unique != adata.n_obs:
        sys.exit(f"FATAL: {adata.n_obs - n_unique} duplicate barcodes across samples")
    adata.layers["counts"] = adata.X.copy()
    log(f"combined: {adata.n_obs} spots x {adata.n_vars} genes across "
        f"{adata.obs['sample'].nunique()} samples")

    # ---- HVG selection (normalise only to select; scVI needs raw counts back)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=N_TOP_GENES, subset=False,
                                flavor="seurat_v3", batch_key="sample", layer="counts")
    log(f"HVGs selected: {int(adata.var['highly_variable'].sum())}")
    adata.X = adata.layers["counts"].copy()

    # ---- train
    scvi.settings.seed = SEED
    ah = adata[:, adata.var["highly_variable"]].copy()
    log(f"training on {ah.n_obs} spots x {ah.n_vars} HVGs")
    scvi.model.SCVI.setup_anndata(ah, layer="counts", batch_key="sample")
    model = scvi.model.SCVI(ah, n_latent=N_LATENT, n_hidden=N_HIDDEN, n_layers=N_LAYERS)
    t0 = time.time()
    model.train(max_epochs=N_EPOCHS, batch_size=BATCH_SIZE, early_stopping=True,
                early_stopping_patience=20, plan_kwargs={"lr": 1e-3})
    log(f"training done in {(time.time()-t0)/60:.1f} min "
        f"({len(model.history['elbo_train'])} epochs), "
        f"final ELBO {model.history['elbo_train'].iloc[-1].item():.2f}")

    # ---- latents, written per sample keyed by barcode
    latent = model.get_latent_representation(ah)
    log(f"latent: {latent.shape}")
    cols = [f"scvi_{i}" for i in range(latent.shape[1])]
    lat = pd.DataFrame(latent, columns=cols)
    lat.insert(0, "barcode", ah.obs_names.to_numpy())
    lat["sample"] = ah.obs["sample"].to_numpy()

    for sample, g in lat.groupby("sample", sort=True):
        out = OUT_DIR / f"{sample}_scvi_latent.csv"
        g.drop(columns="sample").to_csv(out, index=False)
        log(f"  wrote {out.name}  {len(g)} spots")

    pd.concat([model.history["elbo_train"], model.history.get("elbo_validation")],
              axis=1).to_csv(OUT_DIR / "training_history.csv")
    model.save(str(OUT_DIR / "_scvi_model"), overwrite=True)
    log(f"DONE -> {OUT_DIR}")


if __name__ == "__main__":
    main()
