"""
Which of the 30 cohort samples have a full-resolution WSI on disk, and where.

The source WSIs come from the Masood Lab SharePoint
(`.../Primary_PDAC Vs. Mets/0_Lab_images/spatial_images`, IU-authenticated -- not
reachable programmatically, download via a logged-in browser).

Two kinds of file live in the source directory:

  * `image_IU_PDA_<sample>-<ImageName>.tif` -- ALREADY CROPPED to one sample.
    ~21k x 22k, which matches the Visium full-resolution coordinate space, so these
    are drop-in for `create_patches.py`.

  * `<slide_id>.tiff` (e.g. `328.tiff`, `V326.tiff`, `1__117.tiff`) -- a WHOLE
    Visium slide, ~100k x 190k pyramidal Philips TIFF holding up to FOUR capture
    areas. Each area is a different sample, identified by `AreaCode` (A1/B1/C1/D1).
    These must be cropped per area before patching (the lab's note suggests QuPath).

`Tiff files metadata.xlsx` is the authoritative sample <-> (slide_id, area) map.

Usage:
    python 01_Patch_Extraction/wsi_inventory.py [--wsi-dir DIR] [--out CSV]
"""

import argparse
import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WSI = r"D:\Aamir Gulzar\KSA_project3\old_project_data\ST_source_WSI_data"
DEFAULT_META = os.path.join(DEFAULT_WSI, "Tiff files metadata.xlsx")
DEFAULT_OUT = os.path.join(ROOT, "Outputs", "Patient-Sample-Information", "wsi_inventory.csv")


def load_mapping(xlsx):
    """Rows 8..37 of Sheet1 are the sample -> (slide id, area) table."""
    raw = pd.read_excel(xlsx, header=None)
    hdr = raw.index[raw[0].astype(str).str.strip() == "Patient"]
    if len(hdr) == 0:
        sys.exit(f"could not find the header row in {xlsx}")
    m = raw.iloc[hdr[0] + 1:].copy()
    m.columns = ["patient", "image_name", "sample", "slide_id", "area", "origin"]
    m = m[m["sample"].astype(str).str.startswith("IU_PDA_")]
    m["slide_id"] = (m["slide_id"].astype(str)
                     .str.replace(r"\.0$", "", regex=True).str.strip())
    m["sample"] = m["sample"].astype(str).str.strip()
    return m.reset_index(drop=True)


def index_files(wsi_dir):
    """filename -> (kind, key). kind is 'cropped' or 'slide'."""
    out = []
    for f in sorted(os.listdir(wsi_dir)):
        if not f.lower().endswith((".tif", ".tiff")):
            continue
        stem = os.path.splitext(f)[0]
        if stem.startswith("image_IU_PDA_"):
            out.append((f, "cropped", stem.split("-")[0].replace("image_", "")))
        else:
            # numbers in the stem are candidate slide ids: 328, V326, 1__117 -> 117
            ids = {str(int(t)) for t in re.findall(r"\d+", stem)}
            out.append((f, "slide", ids, stem))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi-dir", default=DEFAULT_WSI)
    ap.add_argument("--meta", default=DEFAULT_META)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()

    if not os.path.isdir(a.wsi_dir):
        sys.exit(f"WSI directory not found: {a.wsi_dir}")
    m = load_mapping(a.meta)
    files = index_files(a.wsi_dir)

    cropped, slides = {}, {}
    for rec in files:
        if rec[1] == "cropped":
            cropped.setdefault(rec[2], []).append(rec[0])
        else:
            _, _, ids, stem = rec
            slides[rec[0]] = (ids, stem)

    rows = []
    for _, r in m.iterrows():
        crop = sorted(cropped.get(r["sample"], []))
        whole = sorted(f for f, (ids, stem) in slides.items()
                       if r["slide_id"] in ids
                       or str(r["image_name"]).replace("-", "_").lower() == stem.lower())
        rows.append({
            "sample": r["sample"], "patient": f"PT_{r['patient']}", "origin": r["origin"],
            "slide_id": r["slide_id"], "area": r["area"], "image_name": r["image_name"],
            "cropped_tif": crop[0] if crop else "",
            "whole_slide_tif": whole[0] if whole else "",
            # ready = usable by create_patches.py with no cropping step
            "status": "ready" if crop else ("needs_crop" if whole else "MISSING"),
        })
    inv = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    inv.to_csv(a.out, index=False)

    print(inv[["sample", "origin", "slide_id", "area", "status",
               "cropped_tif", "whole_slide_tif"]].to_string(index=False))
    print()
    print(inv["status"].value_counts().to_string())
    site = inv["sample"].str.extract(r"IU_PDA_(LNM|NP|HM|T)")[0]
    print("\nby site (available = ready + needs_crop):")
    for s in ["T", "HM", "LNM", "NP"]:
        g = inv[site == s]
        ok = (g.status != "MISSING").sum()
        miss = sorted(g[g.status == "MISSING"]["sample"].tolist())
        print(f"  {s:4s} {ok}/{len(g)}" + (f"   missing: {miss}" if miss else "   COMPLETE"))
    missing_slides = sorted({r.slide_id for _, r in inv.iterrows() if r.status == "MISSING"})
    if missing_slides:
        print(f"\nTO FINISH THE COHORT, download these slide IDs: {missing_slides}")
        for sid in missing_slides:
            g = inv[(inv.slide_id == sid)]
            print(f"  slide {sid}: " + ", ".join(f"{r['sample']}({r['area']})"
                                                 for _, r in g.iterrows()))
    print(f"\nSaved -> {a.out}")


if __name__ == "__main__":
    main()
