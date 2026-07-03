"""
Tissue-thumbnail renderer for presentation figures.
Assembles per-spot 224x224 H&E patches onto the Visium hex grid so every result
map can be shown next to the REAL tissue it came from, in the same orientation.

Patch files: dataset/.png patches/.png patches/{sample}/{sample}_patch-NNNNNN_{row}_{col}.png
Visium geometry: x_phys = col * (t/2), y_phys = row * (t*0.866)  (hex offset honoured).
Renders are cached as PNG so downstream figures reuse them.
"""
import os, glob, re
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(ROOT))
PATCH_DIR = os.path.join(PROJ, "dataset", ".png patches", ".png patches")
CACHE = os.path.join(ROOT, "_tissue_cache")
os.makedirs(CACHE, exist_ok=True)

_SUFFIX = re.compile(r"_(\d+)_(\d+)$")   # ..._{row}_{col}

def _row_col(stem):
    m = _SUFFIX.search(stem)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)

def render_he(sample, t=14, force=False):
    """Return (img_uint8 HxWx3, extent) for a sample. extent = [xmin,xmax,ymax,ymin]
    in the SAME physical coords used to plot score maps, so panels align."""
    cache_png = os.path.join(CACHE, f"{sample}_he.png")
    cache_ext = os.path.join(CACHE, f"{sample}_he_extent.npy")
    if os.path.exists(cache_png) and os.path.exists(cache_ext) and not force:
        return np.array(Image.open(cache_png)), np.load(cache_ext)

    files = glob.glob(os.path.join(PATCH_DIR, sample, "*.png"))
    coords = []
    for f in files:
        r, c = _row_col(os.path.splitext(os.path.basename(f))[0])
        if r is not None:
            coords.append((r, c, f))
    if not coords:
        raise RuntimeError(f"no patches for {sample}")
    rows = [c[0] for c in coords]; cols = [c[1] for c in coords]
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)

    def X(c): return int(round((c - c0) * (t / 2)))
    def Y(r): return int(round((r - r0) * (t * 0.866)))
    W = X(c1) + t + 1
    H = Y(r1) + t + 1
    canvas = np.full((H, W, 3), 255, np.uint8)
    for r, c, f in coords:
        try:
            p = np.asarray(Image.open(f).convert("RGB").resize((t, t), Image.BILINEAR))
        except Exception:
            continue
        y, x = Y(r), X(c)
        canvas[y:y + t, x:x + t] = p
    # extent in physical coords (x=col*t/2 shifted, y=row*t*.866) for alignment with maps
    extent = [c0 * (t / 2), c1 * (t / 2) + t, r1 * (t * 0.866) + t, r0 * (t * 0.866)]
    Image.fromarray(canvas).save(cache_png)
    np.save(cache_ext, np.array(extent, float))
    return canvas, np.array(extent, float)

# --- REAL whole-slide image (WSI) support -------------------------------------
# Two sources per sample:
#  * CLEAN downsampled H&E: dataset/WSI images/image_{sample}.png (~2000px, clearest
#    tissue, ALL 6 samples incl. T3) -> used for overview panels (render_wsi).
#  * FULL-RES overlay: dataset/patch overlay/{sample}/{sample}_overlay.tiff (~22k px,
#    has faint Visium fiducial frame) -> used for sharp zoom crops (crop_wsi).
# Spot pixel coords (imagerow/imagecol) live in spot_spatial_coordinates.csv in the
# FULL-RES space; multiply by the returned `scale` to plot on either render.
OVERLAY_DIR = os.path.join(PROJ, "dataset", "patch overlay")
WSI_DIR = os.path.join(PROJ, "dataset", "WSI images")
META_DIR = os.path.join(PROJ, "dataset", ".png patches", ".png patches")
COORD_CSV = os.path.join(PROJ, "Outputs", "Patient-Sample-Information", "spot_spatial_coordinates.csv")
HAS_WSI = {"IU_PDA_T1", "IU_PDA_T3", "IU_PDA_T4", "IU_PDA_T11", "IU_PDA_HM11", "IU_PDA_HM13"}

def has_wsi(sample):
    return os.path.exists(os.path.join(WSI_DIR, f"image_{sample}.png"))

def wsi_full_dims(sample):
    """Full-res WSI (W,H) in pixels from the patch metadata (the coord space)."""
    import json
    m = json.load(open(os.path.join(META_DIR, sample, "patches_metadata.json")))["image_info"]
    w, h = m["wsi_dimensions"].split(" x ")
    return int(w), int(h)

def render_wsi(sample, max_dim=2000, force=False):
    """Return (img_uint8 HxWx3, scale) for the CLEAN whole-slide H&E PNG. `scale` maps
    full-res pixel coords -> this image's pixels; plot a spot at
    (imagecol*scale, imagerow*scale) with default imshow origin."""
    Image.MAX_IMAGE_PIXELS = None
    p = os.path.join(WSI_DIR, f"image_{sample}.png")
    if not os.path.exists(p):
        raise FileNotFoundError(f"no clean WSI png for {sample}")
    im = Image.open(p).convert("RGB")
    if max(im.size) > max_dim:
        r = max_dim / max(im.size)
        im = im.resize((int(im.size[0]*r), int(im.size[1]*r)), Image.BILINEAR)
    fw, fh = wsi_full_dims(sample)
    scale = im.size[0] / fw            # px-per-fullres-px (uniform; aspect preserved)
    return np.asarray(im), float(scale)

def crop_wsi(sample, x0, y0, x1, y1, out_px=900):
    """Crop the WSI in the box [x0,x1]x[y0,y1] (given in FULL-RES pixel coords),
    resized so its long side == out_px. Uses the full-res overlay for sharpness;
    falls back to the clean PNG (e.g. T3, no overlay). Returns (img_uint8, box) where
    box=(x0,y0,x1,y1,s); plot a spot inside at ((imagecol-x0)*s, (imagerow-y0)*s)."""
    Image.MAX_IMAGE_PIXELS = None
    x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(x1), int(y1)
    ov = os.path.join(OVERLAY_DIR, sample, f"{sample}_overlay.tiff")
    if os.path.exists(ov):
        im = Image.open(ov).convert("RGB")
        crop = im.crop((x0, y0, x1, y1))
    else:                                  # fallback: clean PNG, rescale box to png space
        im = Image.open(os.path.join(WSI_DIR, f"image_{sample}.png")).convert("RGB")
        fw, fh = wsi_full_dims(sample); r = im.size[0] / fw
        crop = im.crop((int(x0*r), int(y0*r), int(x1*r), int(y1*r)))
    s = out_px / max(x1 - x0, y1 - y0)
    crop = crop.resize((max(1, int((x1 - x0) * s)), max(1, int((y1 - y0) * s))), Image.BILINEAR)
    return np.asarray(crop), (x0, y0, x1, y1, s)

def wsi_spot_xy(sample, barcodes):
    """Display-independent full-res pixel coords (x=imagecol, y=imagerow) for given
    barcodes on {sample}'s WSI. Multiply by the render_wsi scale to plot. Returns
    dict barcode -> (x_px, y_px)."""
    import pandas as pd
    c = pd.read_csv(COORD_CSV)
    c = c[c["image"] == sample].set_index("spot_barcode")
    out = {}
    for b in barcodes:
        if b in c.index:
            out[b] = (float(c.loc[b, "imagecol"]), float(c.loc[b, "imagerow"]))
    return out

def phys_xy(row, col, t=14):
    """Map (row,col) arrays to the ABSOLUTE physical x,y the H&E canvas extent uses,
    so score scatters overlay/align with render_he() exactly (any sample origin)."""
    x = np.asarray(col) * (t / 2) + t / 2
    y = np.asarray(row) * (t * 0.866) + t / 2
    return x, y

def sample_origin(sample):
    """Min (row,col) for a sample's patches — needed so score maps share H&E origin."""
    files = glob.glob(os.path.join(PATCH_DIR, sample, "*.png"))
    rc = [_row_col(os.path.splitext(os.path.basename(f))[0]) for f in files]
    rc = [x for x in rc if x[0] is not None]
    return min(r for r, c in rc), min(c for r, c in rc)

if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    samples = ["IU_PDA_T1", "IU_PDA_T3", "IU_PDA_T4", "IU_PDA_T11", "IU_PDA_HM11", "IU_PDA_HM13"]
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    for ax, s in zip(axs.ravel(), samples):
        img, ext = render_he(s)
        ax.imshow(img, extent=ext, aspect="equal")
        tag = "Primary tumour" if "_T" in s else "Liver metastasis"
        ax.set_title(f"{s.replace('IU_PDA_','')}  ({tag})", fontsize=12, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        print("rendered", s, img.shape)
    fig.suptitle("H&E tissue (reconstructed from Visium spot patches)", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(ROOT, "fig_tissue_gallery_TEST.png")
    plt.savefig(out, dpi=110); print("saved", out)
