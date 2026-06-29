"""
build_qc_mask.py  —  run once to build Outputs/Patient-Sample-Information/spot_qc_mask.csv

Output columns:
  patch_stem   IU_PDA_HM11_patch-000001_50_102     (unique patch identifier)
  sample       IU_PDA_HM11
  barcode      PDACH_10_AAACAAGTATCTCCCA-1
  nFeature     6031                                (nFeature_Spatial — genes detected)
  nCount       15638                               (nCount_Spatial — total UMI counts)

Usage (local):
    python build_qc_mask.py

Usage from another notebook/script:
    import subprocess; subprocess.run(["python", "build_qc_mask.py"])

The mask is consumed by phase_a_clip_training.ipynb Cell 4:
  pairs are filtered by cfg.MIN_GENES and cfg.MIN_COUNTS at training time.
"""

import os
import pandas as pd
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────
PROJECT    = Path(__file__).parent
COORDS_CSV = PROJECT / "Outputs/Patient-Sample-Information/spot_spatial_coordinates.csv"
QC_DIR     = PROJECT / "dataset/ST/scVI_counts"
PNG_ROOT   = PROJECT / "dataset/.png patches/.png patches"
OUT_CSV    = PROJECT / "Outputs/Patient-Sample-Information/spot_qc_mask.csv"

SAMPLES = [
    "IU_PDA_HM11", "IU_PDA_HM13",
    "IU_PDA_T1",   "IU_PDA_T3",
    "IU_PDA_T4",   "IU_PDA_T11",
]

# ── Load coordinates: (sample, row, col) → barcode ──────────────────────────
print("Loading spot_spatial_coordinates.csv ...")
coords = pd.read_csv(COORDS_CSV)
coords["rc_key"] = (
    coords["image"].astype(str) + "_" +
    coords["row"].astype(str)   + "_" +
    coords["col"].astype(str)
)
rc_to_barcode = dict(zip(coords["rc_key"], coords["spot_barcode"]))
print(f"  {len(rc_to_barcode)} coordinate entries loaded")

# ── Load QC metrics: barcode → (nFeature, nCount) ───────────────────────────
print("Loading QC metrics ...")
qc_frames = []
for sample in SAMPLES:
    csv_path = QC_DIR / f"{sample}_qc_metrics.csv"
    if not csv_path.exists():
        print(f"  MISSING: {csv_path}")
        continue
    df = pd.read_csv(csv_path)
    df["sample"] = sample
    qc_frames.append(df)
    print(f"  {sample}: {len(df)} spots")

qc_all = pd.concat(qc_frames, ignore_index=True)
barcode_to_qc = {
    row["barcode"]: (int(row["nFeature"]), int(row["nCount"]))
    for _, row in qc_all.iterrows()
}
print(f"  {len(barcode_to_qc)} barcodes with QC metrics")

# ── Iterate PNG patches, join everything ────────────────────────────────────
print("\nBuilding spot_qc_mask.csv ...")
rows = []
missing_coord = 0
missing_qc    = 0

for sample in SAMPLES:
    png_dir = PNG_ROOT / sample
    if not png_dir.is_dir():
        print(f"  {sample}: PNG directory not found — {png_dir}")
        continue

    patches = sorted(png_dir.glob("*.png"))
    for png_path in patches:
        stem  = png_path.stem
        parts = stem.split("_")
        try:
            row, col = int(parts[-2]), int(parts[-1])
        except (IndexError, ValueError):
            missing_coord += 1
            continue

        barcode = rc_to_barcode.get(f"{sample}_{row}_{col}")
        if barcode is None:
            missing_coord += 1
            continue

        qc = barcode_to_qc.get(barcode)
        if qc is None:
            # Spot has a patch but no QC metrics (e.g. off-tissue in RCTD dataset)
            missing_qc += 1
            n_feature, n_count = 0, 0
        else:
            n_feature, n_count = qc

        rows.append({
            "patch_stem": stem,
            "sample":     sample,
            "barcode":    barcode,
            "nFeature":   n_feature,
            "nCount":     n_count,
        })

mask_df = pd.DataFrame(rows)
mask_df.to_csv(OUT_CSV, index=False)

# ── Summary ─────────────────────────────────────────────────────────────────
print(f"\nTotal patches processed : {len(rows)}")
print(f"Missing coordinates     : {missing_coord}")
print(f"Missing QC metrics      : {missing_qc}")
print(f"\nSpots passing thresholds:")
for min_g, min_c in [(0, 0), (200, 400), (300, 600), (500, 1000)]:
    mask = (mask_df["nFeature"] >= min_g) & (mask_df["nCount"] >= min_c)
    n = mask.sum()
    pct = 100 * n / len(mask_df)
    print(f"  nFeature>={min_g:4d} & nCount>={min_c:5d} : {n:6,} spots  ({pct:.1f}%)")

print(f"\nPer-sample counts at current threshold (200/400):")
m = (mask_df["nFeature"] >= 200) & (mask_df["nCount"] >= 400)
print(mask_df[m].groupby("sample").size().to_string())

print(f"\nSaved: {OUT_CSV}")
