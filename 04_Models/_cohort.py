"""
Shared cohort definitions for the stage scripts.

Two things live here because getting either wrong silently corrupts every
downstream metric:

1. PATIENT_OF -- sample -> patient.  IU_PDA_HM11 and IU_PDA_T11 are the matched
   primary/metastasis pair from the SAME patient (PT_11).  Leave-one-SAMPLE-out
   therefore leaks: holding out T11 leaves that patient's own HM11 in training.
   Cross-patient claims must use leave-one-PATIENT-out.

2. RCTD_VARIANT -- which cell-modality deconvolution to read.
     "RCTD_paper" (default) = the published `rctd_fullfinal` assay, produced by
                              running RCTD at 25-type resolution then aggregating
                              to 15.  Peer-reviewed, reproduces the Nat Genet figs.
     "RCTD"                 = our Kaggle re-run against `celltype_new1` (15 types
                              directly).  Tumour fraction is ANTI-correlated with
                              the published values (r = -0.12 / -0.77 / -0.70 on
                              HM11 / T11 / T4), so results built on it differ.

   Override with the RCTD_VARIANT environment variable to reproduce old results:
       set RCTD_VARIANT=RCTD  &&  python stage0_confound_diagnostic.py

   Outputs are written to a variant-tagged directory so nothing is overwritten.
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------------- cohort
# COHORT="six" (default) = the original 6 Visium slides, the only ones for which
#                          patches / vision / gene embeddings currently exist.
# COHORT="full"          = all 30 slides of GSE272362, produced by
#                          04_Models/split_full_cohort.R into dataset/full_cohort/.
#                          Selecting it before the downstream artefacts have been
#                          rebuilt will fail loudly on missing embeddings, which is
#                          the intent -- it must never silently mix cohorts.
COHORT = os.environ.get("COHORT", "six")
if COHORT not in ("six", "full"):
    raise ValueError(f"COHORT must be 'six' or 'full', got {COHORT!r}")

FULL_COHORT_DIR = os.path.join(ROOT, "dataset", "full_cohort")

_SIX_PATIENT_OF = {
    "IU_PDA_HM11": "PT_11",
    "IU_PDA_T11":  "PT_11",   # <- same patient as HM11
    "IU_PDA_HM13": "PT_13",
    "IU_PDA_T1":   "PT_1",
    "IU_PDA_T3":   "PT_3",
    "IU_PDA_T4":   "PT_4",
}


def _read_patient_map():
    """sample -> patient from the 30-sample split's patient_map.csv."""
    import csv
    path = os.path.join(FULL_COHORT_DIR, "patient_map.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"COHORT=full needs {path}. Build it first:\n"
            f"  Rscript 04_Models/split_full_cohort.R "
            f"dataset/_zenodo_full/PDAC_Updated.rds dataset/full_cohort")
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["sample"]: r["patient"] for r in csv.DictReader(fh)}


PATIENT_OF = _SIX_PATIENT_OF if COHORT == "six" else _read_patient_map()


def _site_of(sample):
    """T = primary pancreas, HM = liver met, LNM = lymph-node met, NP = normal."""
    tag = sample.replace("IU_PDA_", "")
    for pre in ("LNM", "HM", "NP", "T"):
        if tag.startswith(pre):
            return pre
    return "?"


def _natkey(sample):
    """Natural order: T1 < T3 < T4 < T11, not the lexicographic T1 < T11 < T3.

    Sample ORDER IS LOAD-BEARING -- stage1a concatenates per-sample frames in
    PT_SAMPLES order, so it fixes the row order of leaving_program_scores.csv.
    This reproduces the original hand-written lists exactly.
    """
    import re
    tag = sample.replace("IU_PDA_", "")
    # Split into alternating text/number runs so every embedded number compares
    # numerically -- this also orders PT_1 < PT_2 < ... < PT_10 < PT_13.
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", tag))


SITE_OF     = {s: _site_of(s) for s in PATIENT_OF}
_by_site    = lambda site: sorted((s for s in PATIENT_OF if SITE_OF[s] == site), key=_natkey)
PT_SAMPLES  = _by_site("T")
HM_SAMPLES  = _by_site("HM")
LNM_SAMPLES = _by_site("LNM")
NP_SAMPLES  = _by_site("NP")
ALL_SAMPLES = PT_SAMPLES + HM_SAMPLES + LNM_SAMPLES + NP_SAMPLES
PATIENTS    = sorted(set(PATIENT_OF.values()), key=_natkey)


def patients_of(samples):
    """Map an array/list of sample ids to their patient ids."""
    import numpy as np
    return np.array([PATIENT_OF[s] for s in samples])


# ------------------------------------------------------------- RCTD variant
RCTD_VARIANT = os.environ.get("RCTD_VARIANT", "RCTD_paper")
if RCTD_VARIANT not in ("RCTD", "RCTD_paper"):
    raise ValueError(f"RCTD_VARIANT must be 'RCTD' or 'RCTD_paper', got {RCTD_VARIANT!r}")

RCTD_DIR = os.path.join(ROOT, "dataset", "Cell Embedding Extraction", RCTD_VARIANT)
# Both variants get a suffix so the ORIGINAL untagged output dirs -- produced by the
# pre-fix scripts (old RCTD + leave-one-sample-out) -- are never overwritten.
OUT_TAG = {"RCTD": "_rctdold", "RCTD_paper": "_rctdpaper"}[RCTD_VARIANT]

# ------------------------------------------------------- residualisation design
# What stage1a regresses out of score_core to build `leaving_score_resid`.
#   "tumor"     (default) = tumour fraction only -- the original design
#   "tumor_caf"           = tumour fraction AND CAF fraction (iCAF+myCAF)
#
# Under the paper's RCTD the dominant confound is no longer abundance but stroma:
# corr(core, CAF) = +0.478 (ECM-only +0.598), and the resulting vision score still
# carries corr(vision_resid, caf_frac) = +0.357 with its top fges hit being
# Cancer-associated-fibroblasts (+0.190) ahead of EMT (+0.077).  Removing tumour
# fraction alone therefore does NOT yield a confound-free target.
RESID_ON = os.environ.get("RESID_ON", "tumor")
if RESID_ON not in ("tumor", "tumor_caf"):
    raise ValueError(f"RESID_ON must be 'tumor' or 'tumor_caf', got {RESID_ON!r}")
if RESID_ON == "tumor_caf":
    OUT_TAG += "_residcaf"

# RCTD 15-dim alphabetical order -- identical in both variants
CELL_TYPES = [
    "B cells", "C1Q-TAM", "CD4+ cells", "CD8-NK cells", "DCs",
    "Endothelial cells", "FCN1-TAM", "Hepatocytes", "iCAF", "myCAF",
    "Normal Epithelial cells", "Proliferative T cells", "PVL", "SPP1-TAM",
    "Tumor Epithelial cells",
]
HEPATOCYTE_IDX = 7
ICAF_IDX = 8
MYCAF_IDX = 9
TUMOR_IDX = 14


def out_dir(name):
    """Variant-tagged output directory under Outputs/, created on demand."""
    d = os.path.join(ROOT, "Outputs", name + OUT_TAG)
    os.makedirs(d, exist_ok=True)
    return d


# --------------------------------------------------------------- embedding cache
# Both embedding modalities are stored as ONE ~1 KB .pt file per spot
# (RCTD_paper/<sample>/<patch_stem>.pt, scvi_latent_pt_embeddings/<sample>/...).
# At 20k spots that is already ~39k open+unpickle round-trips per stage run; the
# 30-sample cohort takes it past 400k, where the per-file syscall overhead dominates
# wall clock entirely.
#
# load_spot_embeddings() stacks a sample's tensors once into a single .npz and
# reuses it thereafter.  The returned values are the same float32 numbers the .pt
# files hold -- this is purely a storage-layout change, never a numeric one.
_EMB_CACHE_VERSION = 1


def _emb_cache_path(emb_dir, sample):
    return os.path.join(emb_dir, f"{sample}.__stackcache_v{_EMB_CACHE_VERSION}.npz")


def _build_emb_cache(emb_dir, sample, path):
    """Read every .pt under emb_dir/sample once and persist them as one array."""
    import numpy as np
    import torch

    sdir = os.path.join(emb_dir, sample)
    if not os.path.isdir(sdir):
        return None, None
    stems = sorted(f[:-3] for f in os.listdir(sdir) if f.endswith(".pt"))
    if not stems:
        return None, None
    vecs = [torch.load(os.path.join(sdir, s + ".pt"),
                       map_location="cpu", weights_only=False).numpy()
            for s in stems]
    X = np.vstack(vecs).astype(np.float32)
    try:
        np.savez(path, stems=np.array(stems), X=X,
                 version=np.array([_EMB_CACHE_VERSION]))
    except OSError:
        pass  # read-only location -- fall back to rebuilding each run
    return np.array(stems), X


def load_spot_embeddings(emb_dir, sample, stems):
    """Stack per-spot embeddings for `sample` into an (len(stems), d) float32 array.

    Returns (X, found) where `found` is a boolean mask marking which requested
    stems had a tensor; rows of X for missing stems are zero-filled, so callers
    must consult `found` exactly as they previously consulted os.path.exists().
    """
    import numpy as np

    stems = list(stems)
    path = _emb_cache_path(emb_dir, sample)
    cached_stems = cached_X = None
    if os.path.exists(path):
        try:
            z = np.load(path, allow_pickle=False)
            if int(z["version"][0]) == _EMB_CACHE_VERSION:
                cached_stems, cached_X = z["stems"], z["X"]
        except Exception:
            cached_stems = cached_X = None
    if cached_X is None:
        cached_stems, cached_X = _build_emb_cache(emb_dir, sample, path)
    if cached_X is None:
        return np.zeros((len(stems), 0), np.float32), np.zeros(len(stems), bool)

    pos = {s: i for i, s in enumerate(cached_stems.tolist())}
    take = np.array([pos.get(s, -1) for s in stems], dtype=np.int64)
    found = take >= 0
    X = np.zeros((len(stems), cached_X.shape[1]), np.float32)
    X[found] = cached_X[take[found]]
    return X, found


def boxplot(ax, data, labels, **kw):
    """matplotlib 3.9 renamed Axes.boxplot's `labels` kwarg to `tick_labels`.
    conda base ships 3.7 and py3.13 ships 3.10+, so pick at runtime."""
    import matplotlib
    ver = tuple(int(x) for x in matplotlib.__version__.split(".")[:2])
    key = "tick_labels" if ver >= (3, 9) else "labels"
    return ax.boxplot(data, **{key: labels}, **kw)


def banner(stage):
    print(f"{'=' * 70}\n{stage}   [RCTD_VARIANT={RCTD_VARIANT}]\n"
          f"  cell dir : {RCTD_DIR}\n  out tag  : {OUT_TAG or '(none)'}\n{'=' * 70}")
