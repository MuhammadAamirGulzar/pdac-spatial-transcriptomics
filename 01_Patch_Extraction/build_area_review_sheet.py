"""
Build human-review sheets that identify the capture areas on unmapped slides.

Why a review sheet and not automatic labels
-------------------------------------------
Automated matching of a capture area to its sample was attempted several ways
(tissue-silhouette cross-correlation at 192/384 px, mask-weighted variants,
aspect-preserving normalisation, and the Visium spot pattern) and NONE was
reliable: on slides whose mapping is already known it scored 3/21, with winning
margins of ~0.01 -- indistinguishable from noise.  Assigning labels on that basis
would silently corrupt every downstream result, which is the exact failure mode
this project has already suffered twice.

Visual identification, by contrast, is easy and certain: printed side by side, a
section's tears, holes and fragment outline identify it immediately.

The task is also far smaller than it looks.  21 of 30 samples are already
accounted for, so any unidentified area can only be one of the **9 remaining**
samples.  This script therefore renders each unknown capture area next to all 9
candidates, so a reviewer confirms a 1-of-9 visual match rather than starting
from scratch.

The correlation ranking is printed as a HINT only and is explicitly labelled
unverified -- do not treat it as an answer.

Outputs
    Outputs/Patient-Sample-Information/area_review/<slide>_area<i>.png
    Outputs/Patient-Sample-Information/area_review/README.txt

Run:
    python 01_Patch_Extraction/build_area_review_sheet.py 327 343
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from match_capture_areas import (slide_level, detect_areas, normalise, score_pair,
                                 HIRES_DIR, WSI_DIR, ROOT)

Image.MAX_IMAGE_PIXELS = None
OUT = os.path.join(ROOT, "Outputs", "Patient-Sample-Information", "area_review")
TILE = 330


def label(img, text, h=22):
    out = Image.new("RGB", (img.width, img.height + h), "white")
    out.paste(img, (0, h))
    ImageDraw.Draw(out).text((3, 5), text, fill=(0, 0, 0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slides", nargs="*", default=["327", "343"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    inv = pd.read_csv(os.path.join(ROOT, "Outputs", "Patient-Sample-Information",
                                   "wsi_inventory.csv"))
    cands = inv[inv.status == "MISSING"]["sample"].tolist()
    print(f"candidate samples (the only ones unaccounted for): {cands}")

    refs = {}
    for s in cands:
        p = os.path.join(HIRES_DIR, f"image_{s}.png")
        if os.path.exists(p):
            refs[s] = np.asarray(Image.open(p).convert("RGB"))
    print(f"loaded {len(refs)} candidate reference images")

    lines = [
        "CAPTURE-AREA IDENTIFICATION - REVIEW SHEETS",
        "=" * 60,
        "",
        "Each PNG shows one capture area from an unmapped slide (LEFT, boxed) beside",
        "all candidate samples. Pick the candidate whose tissue outline, tears and",
        "holes match the left image.",
        "",
        f"Candidates are the {len(cands)} samples not yet accounted for anywhere:",
        "  " + ", ".join(cands),
        "",
        "The 'hint' percentage is an automated correlation score. It was validated",
        "against slides with a known mapping and scored only 3/21 - it is NOT",
        "reliable. Use it to order your attention, never as the answer.",
        "",
        "Expected content of the missing slides (from source files mapping.xlsx):",
        "  slide 352 -> HM8(A1), T8(B1), HM10(C1), T10(D1)   [4 areas]",
        "  slide 308 -> T12(A1), HM12(B1), HM5(D1)           [3 areas]",
        "  slide 57  -> HM6(C1), T6(D1)                      [2 areas]",
        "",
        "So a 4-tissue unmapped slide is most likely slide 352.",
        "",
        "NOTE: area order on these scans is NOT reliably left-to-right; on slide 118",
        "A1 was the RIGHTMOST area. Identify by image content, not position.",
        "",
    ]

    for sid in a.slides:
        f = None
        for cand in (f"{sid}.tiff", f"{sid}.tif", f"V{sid}.tiff"):
            if os.path.exists(os.path.join(WSI_DIR, cand)):
                f = cand
                break
        if f is None:
            print(f"slide {sid}: file not found")
            continue
        rgb, full, scale = slide_level(os.path.join(WSI_DIR, f))
        boxes = detect_areas(rgb)
        print(f"\nslide {sid} ({f}): {len(boxes)} capture areas, full={full}")
        lines.append(f"slide {sid} ({f}): {len(boxes)} areas detected")

        for i, (r0, r1, c0, c1) in enumerate(boxes):
            crop = rgb[r0:r1 + 1, c0:c1 + 1]
            g, m = normalise(crop)
            scores = {}
            if g is not None:
                for s, ref in refs.items():
                    rg, rm = normalise(ref)
                    if rg is not None:
                        scores[s] = score_pair(g, m, rg, rm)
            ranked = sorted(scores, key=scores.get, reverse=True)

            A = Image.fromarray(crop); A.thumbnail((TILE * 2, TILE * 2))
            A = label(A, f"slide {sid}  AREA {i}  (x{c0}-{c1})  <-- IDENTIFY THIS")
            tiles = [A]
            for s in ranked:
                B = Image.fromarray(refs[s]); B.thumbnail((TILE, TILE))
                tiles.append(label(B, f"{s.replace('IU_PDA_','')}  hint {scores[s]*100:+.0f}%"))
            wsum = sum(t.width + 8 for t in tiles) + 8
            hmax = max(t.height for t in tiles) + 8
            sheet = Image.new("RGB", (wsum, hmax), "white")
            x = 8
            for t in tiles:
                sheet.paste(t, (x, 4)); x += t.width + 8
            ImageDraw.Draw(sheet).rectangle([6, 2, A.width + 10, A.height + 6],
                                            outline=(200, 0, 0), width=3)
            p = os.path.join(OUT, f"slide{sid}_area{i}.png")
            sheet.save(p)
            top = ", ".join(f"{s.replace('IU_PDA_','')}({scores[s]:+.2f})" for s in ranked[:3])
            print(f"  area{i}: hint order -> {top}")
            lines.append(f"  area{i}: hint -> {top}")

    with open(os.path.join(OUT, "README.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nReview sheets -> {OUT}")


if __name__ == "__main__":
    main()
