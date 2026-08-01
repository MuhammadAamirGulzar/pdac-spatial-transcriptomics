"""
Identify WHICH SAMPLE occupies each capture area of a multi-area Visium slide.

The problem
-----------
Only 22 WSI files were shared, and three slide IDs the cohort needs (352, 308, 57)
are absent under those names -- while two files (327, 343) are not referenced by
the mapping table at all. Several whole-slide TIFFs hold four capture areas, some
of which are tissue sections that exist nowhere else. Labelling them by eye needs a
pathologist.

We do not need one, because every sample already has a reference image of exactly
its own capture area: the Visium hires PNG in `dataset/WSI images/image_<sample>.png`
is what spaceranger produced for that area. So "which sample is this area?" becomes
a measurable 1-of-30 match rather than a judgement call.

Method
------
1. Detect the capture areas on a slide by clustering tissue pixels into 4 groups
   along x (areas sit in one row).
2. Normalise every image -- both detected areas and the 30 references -- by
   cropping to its TISSUE BOUNDING BOX and resizing to a fixed square. This
   cancels the difference between "capture area with margin" and "tissue extent".
3. Score each (area, sample) pair by the best normalised cross-correlation over
   the 8 dihedral orientations, on both the tissue mask and the greyscale texture.
4. Assign samples to areas with a global optimum (Hungarian), so one sample cannot
   be claimed by two areas.

The method is VALIDATED on the slides whose mapping is already known: if it
recovers those assignments it can be trusted on the unknown ones. Every result is
written with its score and margin over the runner-up so a domain expert can review
the ranked candidates rather than start from scratch.

Outputs
    Outputs/Patient-Sample-Information/capture_area_matches.csv
    Outputs/Patient-Sample-Information/capture_area_review/<slide>_<area>.png
        side-by-side of the detected area and its best-matching reference

Run:
    python 01_Patch_Extraction/match_capture_areas.py            # all slides
    python 01_Patch_Extraction/match_capture_areas.py 327 343    # specific slides
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from scipy.optimize import linear_sum_assignment
import scipy.ndimage as ndi

Image.MAX_IMAGE_PIXELS = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WSI_DIR = os.environ.get(
    "WSI_DIR", r"D:\Aamir Gulzar\KSA_project3\old_project_data\ST_source_WSI_data")
HIRES_DIR = os.path.join(ROOT, "dataset", "WSI images")
OUT_CSV = os.path.join(ROOT, "Outputs", "Patient-Sample-Information",
                       "capture_area_matches.csv")
REVIEW_DIR = os.path.join(ROOT, "Outputs", "Patient-Sample-Information",
                          "capture_area_review")
N_AREAS = 4
NORM = 192          # normalised thumbnail edge


# --------------------------------------------------------------------- imaging
def tissue_mask(rgb):
    """H&E tissue: red clearly above green, not near-black, not paper-white."""
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    lum = (r + g + b) / 3.0
    m = (r - g > 12) & (r > 90) & (lum > 60) & (lum < 245)
    m = ndi.binary_opening(m, np.ones((3, 3)))
    m = ndi.binary_closing(m, np.ones((7, 7)))
    return m


def normalise(rgb, mask=None):
    """Crop to the tissue bbox and resize to NORM x NORM. Returns (gray, mask)."""
    m = tissue_mask(rgb) if mask is None else mask
    if m.sum() < 50:
        return None, None
    ys, xs = np.where(m)
    sub = rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    sm = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    g = np.asarray(Image.fromarray(sub).convert("L").resize((NORM, NORM), Image.BILINEAR),
                   np.float32)
    mm = np.asarray(Image.fromarray(sm.astype(np.uint8) * 255).resize((NORM, NORM),
                                                                      Image.BILINEAR),
                    np.float32) / 255.0
    # invert grey so tissue is high, then zero-mean for correlation
    g = 255.0 - g
    g = (g - g.mean()) / (g.std() + 1e-6)
    return g, mm


def dihedral(a):
    for k in range(4):
        r = np.rot90(a, k)
        yield r
        yield np.fliplr(r)


def score_pair(gA, mA, gB, mB):
    """Best correlation across the 8 orientations; texture and shape combined."""
    best = -1.0
    zb_g = (gB - gB.mean()) / (gB.std() + 1e-6)
    zb_m = (mB - mB.mean()) / (mB.std() + 1e-6)
    n = gB.size
    for rg, rm in zip(dihedral(gA), dihedral(mA)):
        zg = (rg - rg.mean()) / (rg.std() + 1e-6)
        zm = (rm - rm.mean()) / (rm.std() + 1e-6)
        s = 0.5 * float((zg * zb_g).sum() / n) + 0.5 * float((zm * zb_m).sum() / n)
        best = max(best, s)
    return best


# --------------------------------------------------------------------- slides
def slide_level(path, max_side=3000):
    with tifffile.TiffFile(path) as t:
        s = t.series[0]
        full = s.levels[0].shape[:2]
        i = next(k for k, l in enumerate(s.levels) if max(l.shape[:2]) <= max_side)
        return s.levels[i].asarray(), full, full[1] / s.levels[i].shape[1]


def detect_areas(rgb, n=N_AREAS):
    """Return the n capture-area bboxes, left->right.

    The capture areas sit in one row separated by clear white gaps, so a column
    profile of tissue splits them.  Two things must be handled or the split is
    wrong: the slide LABEL on the far left is inked and would otherwise be taken
    for an area (it was -- k-means on x put a cluster at cols 3-193), and tissue
    occasionally bridges two neighbouring areas, merging their runs.
    """
    m = tissue_mask(rgb)
    H, W = m.shape
    # the label occupies the far left of a Visium slide; never a capture area
    m = m.copy()
    m[:, :int(0.14 * W)] = False
    if m.sum() < 500:
        return []

    colp = m.sum(0).astype(float)
    thr = max(3.0, 0.06 * colp.max())
    on = colp > thr
    # close 1-column dropouts inside an area
    on = ndi.binary_closing(on, np.ones(9))
    lab, k = ndi.label(on)
    runs = [(np.where(lab == i)[0].min(), np.where(lab == i)[0].max())
            for i in range(1, k + 1)]
    runs = [r for r in runs if r[1] - r[0] > 0.02 * W]
    runs.sort()

    # split any run wide enough to contain several areas
    if runs:
        widths = [r[1] - r[0] for r in runs]
        unit = min(widths) if len(widths) > 1 else (runs[0][1] - runs[0][0]) / n
        split = []
        for c0, c1 in runs:
            k_sub = max(1, int(round((c1 - c0) / max(unit, 1e-6))))
            if k_sub <= 1:
                split.append((c0, c1))
            else:
                edges = np.linspace(c0, c1, k_sub + 1).astype(int)
                split.extend(zip(edges[:-1], edges[1:]))
        runs = split

    # keep the n widest, then restore left->right order
    runs = sorted(sorted(runs, key=lambda r: r[1] - r[0], reverse=True)[:n])

    boxes = []
    for c0, c1 in runs:
        sub = m[:, c0:c1 + 1]
        ys = np.where(sub.any(1))[0]
        xs = np.where(sub.any(0))[0]
        if len(ys) < 10 or len(xs) < 10:
            continue
        boxes.append((ys.min(), ys.max(), c0 + xs.min(), c0 + xs.max()))
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slides", nargs="*", help="slide ids, e.g. 327 343 (default: all)")
    a = ap.parse_args()
    os.makedirs(REVIEW_DIR, exist_ok=True)

    # ---- references: one hires PNG per cohort sample
    refs = {}
    for f in sorted(os.listdir(HIRES_DIR)):
        if not f.lower().endswith(".png"):
            continue
        s = os.path.splitext(f)[0].replace("image_", "")
        g, m = normalise(np.asarray(Image.open(os.path.join(HIRES_DIR, f)).convert("RGB")))
        if g is not None:
            refs[s] = (g, m)
    print(f"references loaded: {len(refs)} samples")

    # ---- expected mapping, for validation
    meta = os.path.join(WSI_DIR, "source files mapping.xlsx")
    if not os.path.exists(meta):
        meta = os.path.join(WSI_DIR, "Tiff files metadata.xlsx")
    raw = pd.read_excel(meta, header=None)
    hdr = raw.index[raw[0].astype(str).str.strip() == "Patient"][0]
    mp = raw.iloc[hdr + 1:].copy()
    mp.columns = ["patient", "image_name", "sample", "slide_id", "area", "origin"]
    mp = mp[mp["sample"].astype(str).str.startswith("IU_PDA_")]
    mp["slide_id"] = (mp["slide_id"].astype(str)
                      .str.replace(r"\.0$", "", regex=True).str.strip())
    expected = {sid: dict(zip(g["area"], g["sample"])) for sid, g in mp.groupby("slide_id")}

    files = {}
    for f in sorted(os.listdir(WSI_DIR)):
        if not f.lower().endswith((".tif", ".tiff")):
            continue
        stem = os.path.splitext(f)[0]
        if re.fullmatch(r"V?\d+", stem):
            files[str(int(re.sub(r"\D", "", stem)))] = f
    todo = a.slides or sorted(files, key=int)
    print(f"whole-slide files: {sorted(files, key=int)}")

    rows = []
    for sid in todo:
        if sid not in files:
            print(f"  slide {sid}: no file"); continue
        path = os.path.join(WSI_DIR, files[sid])
        rgb, full, scale = slide_level(path)
        boxes = detect_areas(rgb)
        exp = expected.get(sid, {})
        print(f"\n=== slide {sid} ({files[sid]}) full={full} areas={len(boxes)} "
              f"expected={exp if exp else 'UNKNOWN — not in mapping'} ===")
        if not boxes:
            continue

        areaimgs, S = [], np.zeros((len(boxes), len(refs)))
        names = list(refs)
        for i, (r0, r1, c0, c1) in enumerate(boxes):
            crop = rgb[r0:r1 + 1, c0:c1 + 1]
            g, m = normalise(crop)
            areaimgs.append(crop)
            if g is None:
                continue
            for j, s in enumerate(names):
                S[i, j] = score_pair(g, m, *refs[s])

        ri, ci = linear_sum_assignment(-S)
        assign = dict(zip(ri, ci))
        for i in range(len(boxes)):
            j = assign.get(i)
            order = np.argsort(-S[i])
            top = [(names[k], S[i, k]) for k in order[:3]]
            best, best_s = names[j], S[i, j]
            margin = best_s - (top[1][1] if top[0][0] == best else top[0][1])
            # Capture areas are numbered RIGHT-TO-LEFT on these scans: A1 is the
            # RIGHTMOST section.  Confirmed visually on slide 118, where the
            # left-to-right order is D1, C1(NP11), B1(HM11), A1(T11).  Assuming
            # left-to-right silently mislabels every area.
            area_code = f"{chr(ord('A') + (len(boxes) - 1 - i))}1"
            exp_s = exp.get(area_code)
            ok = "" if exp_s is None else (" MATCH" if exp_s == best else f" != expected {exp_s}")
            print(f"  area{i} ({area_code}): best={best:14s} score={best_s:+.3f} "
                  f"margin={margin:+.3f}{ok}")
            print(f"          top3: " + ", ".join(f"{n}({s:+.3f})" for n, s in top))
            rows.append(dict(slide=sid, file=files[sid], area_index=i,
                             area_code_guess=area_code, assigned_sample=best,
                             score=round(float(best_s), 4), margin=round(float(margin), 4),
                             expected_sample=exp_s or "",
                             agrees=("" if exp_s is None else str(exp_s == best)),
                             top3="; ".join(f"{n}:{s:.3f}" for n, s in top),
                             bbox_level=f"{boxes[i]}", level_scale=round(scale, 2)))
            # review image: detected area beside its assigned reference
            try:
                A = Image.fromarray(areaimgs[i]); A.thumbnail((520, 520))
                B = Image.open(os.path.join(HIRES_DIR, f"image_{best}.png")).convert("RGB")
                B.thumbnail((520, 520))
                sheet = Image.new("RGB", (A.width + B.width + 24,
                                          max(A.height, B.height) + 8), "white")
                sheet.paste(A, (4, 4)); sheet.paste(B, (A.width + 20, 4))
                sheet.save(os.path.join(REVIEW_DIR, f"slide{sid}_area{i}_{best}.png"))
            except Exception as e:
                print(f"    (review image failed: {e})")

    if rows:
        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
        df.to_csv(OUT_CSV, index=False)
        known = df[df.agrees != ""]
        if len(known):
            acc = (known.agrees == "True").mean()
            print(f"\nVALIDATION on slides with a known mapping: "
                  f"{(known.agrees=='True').sum()}/{len(known)} correct ({acc:.0%})")
        print(f"Saved -> {OUT_CSV}\nReview images -> {REVIEW_DIR}")


if __name__ == "__main__":
    main()
