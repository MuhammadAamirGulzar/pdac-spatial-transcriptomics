"""
Build the CORRECTED cell-modality embeddings from the paper's own RCTD output.

Why this exists
---------------
Our Kaggle RCTD run (03_Embedding_Extraction/Cell/rctd-embedding.ipynb) deconvolved
against the reference's `celltype_new1` column (15 collapsed types).  The published
Nature Genetics pipeline instead ran RCTD at 25-type resolution (`celltypes`) and
then AGGREGATED the weights down to the same 15 labels -- that aggregate is shipped
inside every sample .rds as the `rctd_fullfinal` assay.

The two are not interchangeable.  Measured agreement (paper vs ours), per spot:

    Hepatocytes / B cells / DCs / Endothelial ...   r = 0.53 .. 0.999   (agree)
    iCAF, myCAF, PVL, Normal Epi, Proliferative T   r ~ 0                (disagree)
    Tumor Epithelial cells                          r = -0.12 / -0.77 / -0.70

Tumour fraction -- the confound variable used throughout Stage 0/1a/3/4 -- is
ANTI-correlated with the published estimate on primary tumours.  We therefore
switch the cell modality to `rctd_fullfinal`.

Input   dataset/Cell Embedding Extraction/RCTD_paper/rctd_fullfinal_paper_6samples.csv
        Outputs/Patient-Sample-Information/spot_qc_mask.csv   (patch_stem <-> barcode)
Output  dataset/Cell Embedding Extraction/RCTD_paper/<sample>/<patch_stem>.pt
        one float32 tensor of shape (15,) per spot -- same layout/dtype as RCTD/

Run:
    "C:/Users/datai/anaconda3/python.exe" build_rctd_paper_embeddings.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CELL_ROOT = os.path.join(ROOT, "dataset", "Cell Embedding Extraction")
SRC_CSV = os.path.join(CELL_ROOT, "RCTD_paper", "rctd_fullfinal_paper_6samples.csv")
QC_CSV = os.path.join(ROOT, "Outputs", "Patient-Sample-Information", "spot_qc_mask.csv")
OUT_DIR = os.path.join(CELL_ROOT, "RCTD_paper")
OLD_DIR = os.path.join(CELL_ROOT, "RCTD")

# alphabetical -- must match the existing RCTD/ ordering exactly so that
# HEPATOCYTE_IDX=7 / TUMOR_IDX=14 stay valid in the stage scripts
CELL_TYPES = [
    "B cells", "C1Q-TAM", "CD4+ cells", "CD8-NK cells", "DCs",
    "Endothelial cells", "FCN1-TAM", "Hepatocytes", "iCAF", "myCAF",
    "Normal Epithelial cells", "Proliferative T cells", "PVL", "SPP1-TAM",
    "Tumor Epithelial cells",
]
assert CELL_TYPES == sorted(CELL_TYPES, key=str.lower), "expected case-insensitive alphabetical"

print("[1/4] loading ...")
df = pd.read_csv(SRC_CSV)
qc = pd.read_csv(QC_CSV)
print(f"      rctd_fullfinal rows : {len(df)}")
print(f"      spot_qc_mask rows   : {len(qc)}")

missing_cols = [c for c in CELL_TYPES if c not in df.columns]
if missing_cols:
    sys.exit(f"FATAL: missing cell-type columns in source CSV: {missing_cols}")

print("[2/4] joining patch_stem via (sample, barcode) ...")
merged = qc.merge(
    df[["sample", "barcode"] + CELL_TYPES],
    on=["sample", "barcode"],
    how="left",
    validate="one_to_one",
)
unmatched = merged[CELL_TYPES[0]].isna().sum()
print(f"      matched {len(merged) - unmatched} / {len(merged)}  (unmatched={unmatched})")
if unmatched:
    sys.exit(f"FATAL: {unmatched} spots in spot_qc_mask have no rctd_fullfinal row")

W = merged[CELL_TYPES].to_numpy(dtype=np.float32)
rowsum = W.sum(axis=1)
print(f"      proportion row-sums: min={rowsum.min():.6f} max={rowsum.max():.6f} "
      f"mean={rowsum.mean():.6f}")
if not np.allclose(rowsum, 1.0, atol=1e-3):
    print("      WARNING: some rows do not sum to 1 -- check normalize_weights upstream")

print("[3/4] writing .pt tensors ...")
n = 0
for sample, sub in merged.groupby("sample", sort=True):
    sdir = os.path.join(OUT_DIR, sample)
    os.makedirs(sdir, exist_ok=True)
    vals = sub[CELL_TYPES].to_numpy(dtype=np.float32)
    for stem, vec in zip(sub["patch_stem"].to_numpy(), vals):
        torch.save(torch.from_numpy(np.ascontiguousarray(vec)),
                   os.path.join(sdir, f"{stem}.pt"))
        n += 1
    print(f"      {sample:14s} {len(sub):5d} spots")
print(f"      wrote {n} tensors -> {OUT_DIR}")

print("[4/4] verifying against the old RCTD/ layout ...")
for sample in sorted(merged["sample"].unique()):
    new_files = set(os.listdir(os.path.join(OUT_DIR, sample)))
    old_path = os.path.join(OLD_DIR, sample)
    if not os.path.isdir(old_path):
        print(f"      {sample}: no old dir to compare")
        continue
    old_files = set(os.listdir(old_path))
    same = new_files == old_files
    print(f"      {sample:14s} filenames identical to old RCTD/: {same} "
          f"({len(new_files)} vs {len(old_files)})")
    if not same:
        only_new = sorted(new_files - old_files)[:3]
        only_old = sorted(old_files - new_files)[:3]
        print(f"        only_new={only_new} only_old={only_old}")

# spot-check one tensor round-trip
probe_sample = "IU_PDA_HM11"
probe = sorted(os.listdir(os.path.join(OUT_DIR, probe_sample)))[0]
t = torch.load(os.path.join(OUT_DIR, probe_sample, probe), weights_only=False)
print(f"\n      probe {probe}: shape={tuple(t.shape)} dtype={t.dtype} sum={float(t.sum()):.6f}")
print(f"      tumour frac (idx 14) = {float(t[14]):.4f} | hepatocyte (idx 7) = {float(t[7]):.4f}")
print("\nDONE. Point CELL_DIR / RCTD_DIR at 'RCTD_paper' to use the corrected modality.")
