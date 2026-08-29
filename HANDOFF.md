# Engineering log — PDAC ST + Foundation Model project

Append-only. Newest state first; earlier sections are kept as the record of how each conclusion was
reached, **not** as live instructions. Started 2026-07-28 as a machine-handoff document.

---

## WHERE THE PROJECT STANDS — 2026-08-29

**Stages 0 through 8 are complete and committed.** Every analysis in the pipeline table in
[README.md](README.md) has run and its outputs are in `Outputs/`. The 30-sample split (§4, written
as "the immediate next step") finished on 2026-07-29 and everything downstream of it is done.

**Repository.** `master` is the single line of work, at the site-specificity result. The
`full-cohort-and-site-specificity` branch was fast-forwarded into it and deleted; the six-sample
history remains reachable at `f2293fc`.

**The finding, as it stands after stage 9 (§13) — this wording matters.**

- **Site-specificity is established in GSE274557, not in GSE272362.** All three site pairs in the
  replication cohort fall far below their permutation nulls (`p = 0.0005 / <0.0001 / 0.012`). The
  discovery cohort's `+0.547` does **not** clear its own null (`p = 0.144`) — see §13. Any write-up
  that leads with "liver and node metastases share only ~55 %" is overclaiming.
- **The B-cell difference between liver and node deposits is solid** — `d = +2.03` at section level,
  `+2.01` with the patient as the unit, 4/4 paired patients, and it holds across a 30→80 % purity
  sweep while hepatocyte contamination falls eightfold.
- **The derived prediction holds**: H&E→expression models do not cross organs (HEST-1k, within-organ
  `0.235`, same-organ-different-lab `0.199`, cross-organ `0.125` — the organ effect beats the batch
  effect).

So the defensible shape of the claim is: *deposits at different sites differ in specific,
identifiable features — B cells above all — and in the treatment-naive replication cohort they
differ globally too; the discovery cohort is too small to resolve the global contrast either way.*

**Written up.** `docs/reports/build_research_paper.py` generates the full manuscript;
`docs/presentation/build_deck_site.py` the clinician deck.

### What is NOT done — the work between here and a submission

Ordered by how much it affects acceptance. Everything in the first group is recomputation over CSVs
already in `Outputs/` — no GPU, no new data, no re-running the cohort split.

1. ~~**No uncertainty on any cosine.**~~ **DONE 2026-08-29 — see §13.** Outcome was not the expected
   one: the discovery cohort's `+0.547` does not clear its own null. The manuscript's framing has to
   change, not just gain error bars.
2. **The B-cell claim still depends on one deconvolution.** The *statistical* half of this is
   **done** — the patient-level re-test in §13 leaves it intact (`d = +2.01`, `q = 0.065`, 4/4).
   What remains is the orthogonal readout: recompute a B-cell marker score directly from counts
   (`MS4A1`, `CD79A`, `CD19`, `IGHM`) per section so the claim stops depending on RCTD at all.
   **Needs `dataset/full_cohort/`, so it runs on the D: machine, not this one.**
3. **Treatment exposure is uncontrolled in the discovery cohort.** The replication cohort is
   treatment-naive; GSE272362 is not characterised on this axis. Clinical fields exist for only 6 of
   13 patients (see "Not attempted: clinical validation" below) — tabulate what there is and test
   within it.
4. **Stages 5–7 are pseudobulk, so nothing yet needs the data to be *spatial*.** The strongest
   available answer: show node-met B cells are spatially *organised* (TLS-like aggregates) where
   liver-met B cells are diffuse — Moran's I per section plus a neighbourhood-enrichment count.
5. **The replication cohort has no lymph nodes**, so the specific B-cell claim is untested
   elsewhere rather than confirmed. Either find an LN-containing cohort or demote B cells from
   headline to worked example.
6. **No power statement behind the negatives**, which leaves them open to "absence of evidence".

### Deliberately not being pursued

- **Expanding the H&E arm.** `Outputs/Patient-Sample-Information/wsi_inventory.csv`: of 30 sections,
  **11 ready, 10 needs_crop, 9 MISSING** (slides 352, 308, 57 — worth requesting from the lab, it
  costs nothing and has long latency). The automated capture-area matcher validated at **3/21** and
  is not usable, so the 10 `needs_crop` sections need a human to identify capture areas from the
  review sheets before QuPath. What that buys is narrow: **primaries go 4 → 6** (adding T2 and T9),
  against a target already shown to be unrecoverable (`rho = -0.043` for anything
  metastasis-derived, and the two candidate targets correlate at only `0.101`). HEST's 156 sections
  serve the imaging-generalisation claim better than two more PDAC primaries.
- **Further RCTD variants.** The `rctdold` / `rctdpaper` / `rctdpaper_residcaf` sweep has run and
  the conclusion is stable across all three. That question is closed.

---

## STATUS UPDATE — 2026-07-29, migrated and running

The migration is done. **§4's "immediate next step" (the 30-sample split) is COMPLETE.**
Sections 2 and 3 are superseded by what follows.

**Machine.** 128 GB RAM, i9-13900K (24C/32T), RTX 4090 24 GB. The ≥32 GB RAM blocker is gone
and the GPU stages are now viable locally. Project root: `D:\Aamir Gulzar\KSA_project3\ST_Project`.

**Environment.** `stproj` conda env, **pure-pip** scientific stack:
numpy 2.4.6 / pandas 2.3.3 / scipy / scikit-learn 1.9.0 / matplotlib / seaborn / umap-learn /
openpyxl / **torch 2.13.0+cu126 (CUDA works on the 4090)**.
```bash
C:/Users/datainsight/anaconda3/envs/stproj/python.exe
```
Two traps worth remembering:
- Do **not** install numpy/scipy/sklearn from conda alongside pip's torch. Conda's MKL ships a
  second `libiomp5md.dll` and `sklearn.decomposition.PCA` then aborts the process silently
  (exit 127, no traceback). The pure-pip stack has one OpenMP runtime and is fine.
- pandas is pinned `<3`. conda resolves to pandas 3.x, whose copy-on-write and string-dtype
  changes can silently move numbers that every validated result depends on.
- R 4.5.2 / Seurat 5.3.1 / SeuratObject 5.2.0 / Matrix 1.7.4 / **data.table** (now required).

**`PDAC_Updated.rds` was NOT transferred** — only a stale `dl_progress.log`. Re-downloaded from
Zenodo (~1 h at 3 MB/s); md5 `d0f0b12e0fb013f3def1a62d0f925cbf` verified.

### The 30-sample cohort is split — and it is richer than §4 assumed

`dataset/full_cohort/` — 91,496 spots, 87,055 QC-passing, 17,893 genes, all 30 images matched.
Runtime **5.5 min** (not the ~45 min estimated). Artefacts: `ST/` (30 slim .rds, 1.5 GB),
`scVI_counts/`, `rctd/`, `fges/`, `coords/`, `cohort_metadata.csv`, `patient_map.csv`,
`split_summary.csv`. Total 4.5 GB.

**Validation: all six original samples regenerate BYTE-IDENTICAL (md5) to `dataset/ST/scVI_counts/`,
and the RCTD matches the existing `RCTD_paper` embeddings to 2.98e-08 (= float32 eps).**

The cohort is **not** just "more PT and HM". It is **30 samples / 13 patients / 4 site types**:

| site | n | spots | meaning |
|---|---|---|---|
| T   | 10 | 35,458 | primary pancreas |
| HM  | 12 | 28,520 | liver metastasis |
| **LNM** | **5** | 17,698 | **lymph-node metastasis** |
| **NP**  | **3** | 9,820  | **normal pancreas** |

- **9 of 13 patients have a primary AND ≥1 metastasis** (was 1: PT_11). Matched-pair analyses
  are now properly powered.
- `PT_10` has **all four sites** (T10 / HM10 / LNM10 / NP10).
- **LNM is the scientifically important addition.** The entire Stage-0 confound story is
  hepatocyte contamination in liver mets. Lymph-node mets have no hepatocytes, so LNM is a
  metastatic site *without* the liver confound — the natural control for "is this metastatic
  biology or liver tissue?". NP gives a true normal baseline.
- Folds go 5 → **13 patients**.

### FIRST FULL-COHORT RESULT — Stage 0, and it changes the confound story

`04_Models/stage0_full_cohort.py` → `Outputs/stage0_full_cohort/`. Gene modality from
`03_Embedding_Extraction/Gene/scvi_full_cohort.py` (scVI on all 30 slides: 87,055 spots × 3,000 HVGs,
374 epochs, 16.8 min on the 4090, ELBO 994.20, latents keyed by **barcode** since the new slides have
no patches). Leave-one-patient-out, pooled OOF balanced accuracy, tumour-dominated spots only
(Tumor Epithelial ≥ 0.50).

| contrast (tumour-only) | RCTD composition | RCTD − hepatocyte − tumour | **scVI transcriptome** |
|---|---|---|---|
| HM vs primary | 0.822 | 0.776 | **0.623** |
| LNM vs primary | 0.585 | 0.564 | **0.613** |
| HM vs LNM | 0.673 | 0.558 | 0.615 |
| **NULL** (arbitrary primary/primary patient splits, 5 reps) | 0.469 | — | **0.430** |

**Read the two modalities separately — they say opposite things.**

1. **Composition is mostly organ, not tumour.** HM-vs-primary scores 0.822 on cell composition and
   still 0.776 after deleting *both* the hepatocyte and tumour channels, yet the same test on LNM
   collapses to 0.585 (null 0.469). A "metastatic direction" read off cell composition is therefore
   largely **liver-organ identity** — precisely the confound §1 feared, now measured. Note this is
   *not* fixed by dropping hepatocytes: the rest of the liver microenvironment carries it.
2. **The transcriptome separates metastasis from primary at both sites — but not along the same
   axis.** HM 0.623 and LNM 0.613 are nearly identical, both ≈ +0.19 over the null floor (+0.12
   against a 0.5 chance reference), so each metastatic site is equally *separable* from primary.
   **Equal separability is not a shared direction, and it turned out not to be one:** the two
   trained directions in scVI space have cosine similarity **0.466**, and projected onto the same
   primary spots their scores correlate at only **ρ = +0.32**. HM-vs-LNM is itself separable at
   0.615. So there is a partially shared metastatic component plus substantial site-specific
   structure — not one portable "metastatic axis".

**Implication for Phase B:** build the target from the **transcriptome**, not from cell composition —
composition-derived separation is substantially organ context. But see §8: the current leaving-programme
target does not track the transcriptional metastatic axis either.

Caveats stated plainly: the null floor sits *below* 0.5 (arbitrary labels generalise worse than
chance across held-out patients), so margins measured against it are generous — the conclusion holds
against a 0.5 reference too, just smaller. LNM has 5 patients vs 10 primaries.

### Fixes made during migration (all verified, see §7)

Four were real bugs, not cleanups — most importantly `split_full_cohort.R` was **not QC-filtering
the counts CSVs**, which would have fed scVI a materially different training set.

### What is NOT done yet — and the one hard blocker

Downstream artefacts still cover only the original 6 samples.

> **WSI STATUS — mostly RESOLVED (2026-07-29).** 21 of 30 samples have a full-resolution WSI at
> `D:\Aamir Gulzar\KSA_project3\old_project_data\ST_source_WSI_data` (27.4 GB, 22 TIFFs).
> Run `python 01_Patch_Extraction/wsi_inventory.py` → `Outputs/Patient-Sample-Information/wsi_inventory.csv`.
>
> | status | n | meaning |
> |---|---|---|
> | `ready` | 11 | `image_IU_PDA_<sample>-<ImageName>.tif`, already cropped to one sample, ~21k × 22k — matches the Visium full-res coordinate space, drop-in for `create_patches.py` |
> | `needs_crop` | 10 | inside a whole-slide `<id>.tiff` (~100k × 190k, 12-page pyramidal **Philips** TIFF) holding up to 4 capture areas; crop by `AreaCode` first (lab suggests QuPath) |
> | `MISSING` | 9 | slides **352, 308, 57** not downloaded |
>
> By site: **LNM 5/5 and NP 3/3 COMPLETE**; primaries **6/10** (was 4 — adds T2 and T9); HM 7/12.
>
> **RESOLVED 2026-07-30 — the two unmapped files are NOT the missing slides.**
> The source folder holds 22 files: 9 whole-slide TIFFs (117, 118, 119, 120, 326, **327**, 328,
> **343**, 377) plus per-sample crops. 327 and 343 appear in no mapping row, so the hypothesis was
> that they might be 352/308/57 under other names. They are not:
> * **Every slide's serial number is physically printed on it.** Reading the label (crop the left
>   ~10% of a low pyramid level and `rotate(90, expand=True)` — verified on 118, which reads
>   "12M07-118") confirms the FILENAME IS THE SERIAL: 117→V12M07-117, 119→V12M07-119,
>   326→V12A25-326, 377→V12A25-377.
> * **343 reads "V12A24-3…"** — batch **12A24**, which appears nowhere in this cohort (ours are
>   V12A25, V12M07, V11N29, V11D13, V12M15). By the same convention 327 = V12A25-**327**; the cohort
>   uses V12A25-326/-328/-377 but never -327.
> * Visually comparing 327's four capture areas against the four slide-352 samples
>   (HM8/T8/HM10/T10) shows no correspondence.
>
> **Conclusion: 327 and 343 are slides from other experiments that happen to sit in the same folder.
> Slides 352, 308 and 57 were never downloaded and must be fetched from the source.**
>
> **Note the two groups of missing samples are different problems:**
> * **T2 and T9 are NOT missing** — they live on slides 377 and 326, which we HAVE. They need a
>   reliable per-area crop, not a download.
> * The other **9 samples** (HM5, HM6, T6, HM8, T8, HM10, T10, HM12, T12) need slides 352/308/57.
>
> **To finish the cohort, download exactly three slides** from the Masood Lab SharePoint
> (`.../Primary_PDAC Vs. Mets/0_Lab_images/spatial_images`):
> * slide **352** → HM8 (A1), T8 (B1), HM10 (C1), T10 (D1)
> * slide **308** → T12 (A1), HM12 (B1), HM5 (D1)
> * slide **57**  → HM6 (C1), T6 (D1)
>
> That SharePoint is IU-authenticated and returns **HTTP 403** to any non-browser client — it cannot
> be fetched programmatically; download via a logged-in browser.
>
> `Tiff files metadata.xlsx` in the same folder is the authoritative sample ↔ (slide id, area) map.
> `dataset/WSI images/*.png` is **not** a substitute for any of this — those are Visium
> `tissue_hires_image.png` at 2000 × 1910, ~10× too small (a 224 px native patch becomes ~22 px).

### 8c. Vision modality extended — 11 samples now have patches + UNI2-h (was 6)

Done 2026-07-29 from the WSIs described above:
- `01_Patch_Extraction/create_patches.py` is now **inventory-driven and machine-independent**
  (`Config.WORKSPACE_DIR` resolved from the repo; sample→TIF from `wsi_inventory.csv`; accepts
  `SAMPLE…` or `all` on the command line). It only ever accepts `status == ready` samples.
- New patches: **LNM6 3745, LNM7 3186, LNM10 4147, NP10 2966, NP11 3859** — all with **0 failures**,
  and every count matches the ST spot count for that sample exactly.
- `03_Embedding_Extraction/Vision/extract_uni2h_local.py` runs UNI2-h on the 4090 (~120 patch/s,
  ~30 s per sample). It is a faithful port of the Kaggle notebook — same model args, same
  Resize(256)/CenterCrop(224)/ImageNet-normalise, same skip of the 8 register tokens, same output
  schema — with a multi-worker DataLoader so PNG decode does not starve the GPU.

> **Verified against the Kaggle output**: re-extracting `IU_PDA_T4` locally gives
> **cosine similarity 1.000000** (min and mean) vs the existing file, mean abs diff 3.9e-05
> (3e-04 relative, i.e. fp16 non-determinism), and identical `patch_names`. New and old embeddings
> are interchangeable.

Two traps fixed along the way: `os.system` mangles the space-containing repo path on Windows (use
`subprocess`), and the script's `✓`/`✗` console characters raised `UnicodeEncodeError` under cp1252
**inside a broad `except`**, which silently downgraded it to grid-based extraction and produced
0 patches from 11,009 attempts. Non-ASCII removed.

**This does NOT yet fix the target-comparison power problem.** Both newly-available primaries
(T2, T9) are `needs_crop`, so primaries with patches are still **4** (T1, T3, T4, T11). What the new
samples do unlock is H&E coverage of 3 lymph-node mets and 2 normal pancreas — sites that had no
vision modality at all before.

What *can* proceed now on all 30 slides:
1. **Cell modality** — no re-run needed, read `dataset/full_cohort/rctd/` (already validated).
2. **scVI** on the expanded counts — needs `scvi-tools`; install it in a SEPARATE env, its jax
   dependency tree can disturb the validated `stproj`.
3. **Stage 0's confound diagnostics** and the newly-unblocked tumour-only HM-vs-PT test, plus the
   LNM control — these need only cell + gene modalities.

Blocked on the WSIs: patch extraction, vision embeddings (UNI2-h and H-optimus-1 weights *are*
already in the local HF cache; CONCH is not), Phase A retrain, stages 3/4.

---

## 0. How to carry the conversation context across

Three mechanisms, in order of usefulness:

### (a) The memory directory — do this one (68 KB)

Claude Code keeps per-project memory as plain markdown on disk:

```
C:\Users\datai\.claude\projects\c--Users-datai-Downloads-ST-Project\memory\
    MEMORY.md                        <- index, loaded into context every session
    project_pdac_overview.md
    pdac_review_decisions.md
    pdac_data_structures.md
    pdac_rctd_provenance.md          <- the RCTD bug + fix
    pdac_full_cohort_source.md       <- where the other 24 samples live
    stage_results_corrected_rctd.md  <- current results, supersedes stage3/4
    stage2_results.md / stage3_results.md / stage4_results.md
    presentation_figures.md
```

The folder name is the project's absolute path with `:` and `\` replaced by `-`
(`c:\Users\datai\Downloads\ST_Project` → `c--Users-datai-Downloads-ST-Project`).

**Easiest transfer:** on the new machine, `cd` into the project and start Claude Code once so it
creates `~/.claude/projects/<encoded-path>/`, then copy the `memory/` folder into it. If the project
sits at a different path there, the encoded folder name differs — let Claude Code create it rather
than guessing.

### (b) This document

Committed to the repo, so it travels with `git`. Self-contained: a fresh model with no memory can
pick up from here.

### (c) The raw transcript (optional, reference only)

```
C:\Users\datai\.claude\projects\c--Users-datai-Downloads-ST-Project\81cc1ff9-*.jsonl   (62 MB)
```

Full-fidelity JSONL of the session. Too large to load as context and not designed to be re-imported —
copy it only if you want a searchable record. **(a) + (b) is what actually matters.**

---

## 1. Where the project stands

**Goal.** Predict which primary-tumour (PT) spots are "metastatic in behaviour" from H&E alone.
Spatial transcriptomics is used only to *supervise* during training; at inference only H&E is needed.

**Data.** 6 Visium samples (4 PT + 2 HM), 20,395 spots. These are a **20% subset of GSE272362**
(Khaliq et al., Nat Genet 2024, PMID 39294496 — Dr. Ashiq Masood's own lab). The full 30-sample
cohort is downloaded but not yet split (see §4).

### The big finding this session: the cell modality was wrong

Every sample `.rds` already contains the paper's own deconvolution as the `rctd_fullfinal` assay,
with **exactly our 15 cell types in our exact order**. Our Kaggle RCTD re-run disagrees with it:

| cell type | HM11 | T11 | T4 |
|---|---|---|---|
| Hepatocytes / B cells / DCs / Endothelial | 0.85–0.999 | | |
| iCAF / myCAF / PVL / Normal Epi | ~0 | ~0 | ~0 |
| **Tumor Epithelial cells** | **−0.12** | **−0.77** | **−0.70** |

**Cause (confirmed).** The reference `scRNA-seq_Data_post_qc.rds` has four annotation columns:
`celltypes`/`seurat_clusters` (25 types), `celltype_new1` (15), `celltype_new` (9). The paper ran
RCTD at **25-type resolution then aggregated to 15**; `03_Embedding_Extraction/Cell/rctd-embedding.ipynb`
ran directly against the collapsed 15. Verified: aggregating the 25 → 15 (incl. `CAF-S4 → PVL`)
reproduces `rctd_fullfinal` at mean abs diff 0.006, Tumor Epi r = 0.9992.

**Decision: use the paper's `rctd_fullfinal`.** Peer-reviewed, better practice, zero compute, and it
exists for all 30 samples.

### Corrected results (three configurations)

| | ours / tumour | paper / tumour | paper / tumour+CAF |
|---|---|---|---|
| Stage3 frozenFM_resid ρ | +0.081 | +0.250 | **+0.121** |
| **vision_resid vs external EMT** | +0.006 | +0.077 | **+0.002** |
| vision_resid vs held-out resid | +0.063 | +0.206 | +0.111 |
| resid confound: CAF | −0.014 | +0.357 | +0.027 |
| **target_raw vs external EMT** | +0.205 | +0.205 | **+0.205** |
| abundance ceiling | +0.478 | +0.113 | +0.113 |
| vision_raw − ceiling | −0.190 | +0.156 | **+0.177** |

**Honest read.** The intermediate +0.250 was about half desmoplasia. Removing CAF too kills the CAF
confound (+0.357 → +0.027) and drops ρ to +0.121; against the *independent external* EMT signature
`vision_resid` is **+0.002** and EMT ranks **#16 of 29** signatures.

What robustly survives all three configurations:
1. `target_raw vs external EMT = +0.205` — the ST leaving program **is** a valid EMT readout.
2. `vision_raw − abundance ceiling = +0.177` — H&E carries real signal beyond malignant abundance.
3. That signal is Neutrophil (+0.113), Tumour-proliferation (+0.109), Matrix-remodeling (+0.098) —
   **not** EMT.
4. frozenFM ≥ bridge128 everywhere — the contrastive bridge adds nothing. Phase B should pivot to
   direct frozen-FM supervised regression.

### Two structural problems with the 6-sample cohort

- **Patient leakage.** `IU_PDA_HM11` and `IU_PDA_T11` are the matched pair from patient `PT_11`, so
  6 samples = **5 patients**. Folds must be leave-one-*patient*-out. Fixed in `_cohort.py`;
  Phase A's `n_folds: int = 6` in `phase_a_clip_training.ipynb` is **still wrong**.
- **The tumour-only HM-vs-PT test is impossible.** Under the paper's RCTD every HM tumour-dominated
  spot comes from HM11 (876 spots); HM13's mean tumour fraction is 0.022 → zero spots. No fold can
  hold out an HM tumour spot. Stage 0 correctly returns `nan`. The old RCTD masked this by reporting
  accuracy 0.482 with TPR_HM = 0.000. **Only the cohort expansion fixes this.**

---

## 2. Environment setup on the new machine

The old `tcga` conda env **no longer exists**; script docstrings referencing it are stale.

**Python** — needs a **numpy 2.x** stack. The Kaggle-written `.pt` files fail to unpickle under
numpy 1.x with `ModuleNotFoundError: No module named 'numpy._core'`.

```bash
pip install "numpy>=2" pandas scipy scikit-learn matplotlib seaborn
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU is enough for stages 0-4
```
Verified working combination: numpy 2.2.6, torch 2.13.0+cpu, sklearn 1.9.0, pandas 2.3.1.

Two API breaks are already patched in the repo, but note them if you touch the code:
- torch ≥2.6 defaults `torch.load(..., weights_only=True)` → pass `weights_only=False`.
- matplotlib ≥3.9 renamed `boxplot(labels=)` → `tick_labels=`; use the `boxplot()` shim in `_cohort.py`.

**R** — needed only for the cohort split. R 4.4+, `Seurat` (tested 5.5.1), `SeuratObject`, `Matrix`.
`spacexr` is **not** required — we no longer run RCTD ourselves.

**RAM** — this is the binding constraint. `PDAC_Updated.rds` needs **≥32 GB** (48 GB comfortable);
`readRDS` is atomic so there is no partial-read workaround. The old laptop had 7.7 GB, which is why
the split never ran there.

---

## 3. What to copy

| Item | Size | Notes |
|---|---|---|
| `dataset/_zenodo_full/PDAC_Updated.rds` | **11.64 GB** | md5 `d0f0b12e0fb013f3def1a62d0f925cbf` — verify after transfer |
| `dataset/ST/*.rds` (6 files) | 2.3 GB | can be regenerated from the above |
| `dataset/scRNA-seq_Data_post_qc.rds` | 1.2 GB | RCTD reference |
| `dataset/ST/scVI_counts/` | ~1 GB | counts + qc_metrics |
| `dataset/WSI images/` | 30 PNGs | all 30 samples already present |
| `dataset/Feature Extraction Embeddings/` | — | CONCH V1/V1.5, UNI2-h, H-Optimus-1, ResNet18 |
| `dataset/Cell Embedding Extraction/RCTD_paper/` | — | **the corrected cell modality** |
| `Outputs/` | — | includes `*_rctdold` / `*_rctdpaper` / `*_rctdpaper_residcaf` |
| `~/.claude/.../memory/` | 68 KB | see §0 |
| **Whole repo excluding `.git`** | **25.07 GB** | `dataset/` 24.13 GB + `Outputs/` 0.81 GB |

Minimum viable transfer if bandwidth is tight: skip `dataset/ST/*.rds` (2.3 GB) and
`dataset/ST/scVI_counts/` (~1 GB) — `split_full_cohort.R` regenerates both from `PDAC_Updated.rds`,
byte-identically.

`PDAC_Updated.rds` is re-downloadable if the transfer is painful:
```bash
curl -L -C - -o PDAC_Updated.rds \
  "https://zenodo.org/records/10712047/files/PDAC_Updated.rds?download=1"
```
Zenodo throttles to ~20 MB/min (several hours).

> `keys.txt` at the repo root holds the Kaggle API token. It is in `.gitignore`. Move it out of band —
> never commit it.

---

## 4. ~~The immediate next step~~ — DONE 2026-07-29

> **Superseded.** The split ran on 2026-07-29 and everything below it in this section is history.
> Results are in the 2026-07-29 status update above; the cohort lives in `dataset/full_cohort/`.

Split the 30-sample object. **This is the highest-value action available** — it takes PT 4 → 10,
**HM 2 → 12**, spots 20,395 → 91,496, and folds 5 → ~13 patients. Every Phase B conclusion currently
rests on 2 HM slides, only one of which is informative.

```bash
Rscript 04_Models/split_full_cohort.R  dataset/_zenodo_full/PDAC_Updated.rds  dataset/full_cohort
```

~90 s/sample (~45 min total). Validated end-to-end on a single sample: regenerated counts came back
**byte-identical** to `dataset/ST/scVI_counts/` (max abs diff 0, 17,893 genes) and the median
genes/counts matched the existing qc_metrics exactly.

It writes per sample: counts CSV (genes × barcodes), `rctd_fullfinal` (15 cols), `fges` (27 Bagaev
signatures), coords, slim `.rds` (57 MB vs 480 MB), plus `cohort_metadata.csv` and `patient_map.csv`.
It drops SCT/integrated/scale.data on load, roughly halving peak RAM.

Then, in order:
1. **Patch H&E** at the new spot coordinates — `01_Patch_Extraction/`. Watch the T11 transpose quirk
   documented in `pdac_data_structures` memory; re-check per new sample.
2. **Vision embeddings** — `03_Embedding_Extraction/Vision/` (GPU).
3. **scVI** on the expanded cohort — `02_Gene_Export/` + `03_Embedding_Extraction/Gene/` (GPU).
   Cell modality needs **no** rerun: take `rctd_fullfinal` straight from the split.
4. **Phase A** — fix `n_folds` to leave-one-**patient**-out first.
5. **Stages 0/1a/1b/3/4** — they now take `RCTD_VARIANT` and `RESID_ON` env vars.

---

## 5. Code added/changed this session

New:
- `04_Models/_cohort.py` — `PATIENT_OF`, `RCTD_VARIANT` (`RCTD` | `RCTD_paper`), `RESID_ON`
  (`tumor` | `tumor_caf`), output tagging, matplotlib `boxplot()` shim.
- `04_Models/build_rctd_paper_embeddings.py` — builds the 20,395 corrected `.pt` tensors.
- `04_Models/split_full_cohort.R` — the 30-sample split.
- `dataset/Cell Embedding Extraction/RCTD_paper/` — corrected cell modality + `clinical_metadata_6samples.csv`.

Modified — `stage0_confound_diagnostic.py`, `stage1a_leaving_program.py`,
`stage1b_pt11_hm11_anchor.py`, `stage3_phase_b_scoring.py`, `stage4_validation.py`:
- `ROOT` was `dirname(__file__)`, which stopped resolving when the repo restructure moved these into
  `04_Models/`. All paths were broken; now resolved via `_cohort.py`.
- Stage 0 switched to leave-one-patient-out with pooled OOF balanced accuracy.
- Stage 4's `OVERALL READ` was **hardcoded prose** asserting the old conclusion; it contradicted its
  own computed numbers once the RCTD was fixed. Now fully derived from the metrics, and it prints a
  residual-confound ranking plus EMT's rank among the 29 Fges signatures.

Run any stage as:
```bash
RCTD_VARIANT=RCTD_paper RESID_ON=tumor_caf python 04_Models/stage3_phase_b_scoring.py
```
Outputs are tagged (`_rctdold`, `_rctdpaper`, `_rctdpaper_residcaf`) so nothing overwrites the
originals.

---

## 7. Migration fixes — 2026-07-29

Every change below was verified against the pre-existing outputs; none moves a validated number.

### Correctness bugs

1. **`split_full_cohort.R` did not QC-filter the counts CSVs.** `02_Gene_Export/scvi-gene-export.ipynb`
   subsets to `nFeature ≥ 200 & nCount ≥ 400` *before* writing, so the production
   `IU_PDA_T4.csv` has 3,587 barcode columns for a 3,621-spot sample. The split wrote all spots.
   On the full cohort the gap is large — HM6 keeps 1,035/1,666 (37.9 % dropped), HM9 1,369/1,908,
   HM13 1,387/2,182 — i.e. scVI would have been trained on a much larger, lower-quality set.
   Also restored the missing `pass_qc` column. Counts files are now byte-identical to production.
2. **Phase A folds leaked patients.** `HM11` and `T11` are the same patient, so old fold 1 held out
   HM11 while training on T11, and fold 6 did the reverse. `FOLD_DEFS` is now derived from
   `PATIENT_OF` → leave-one-**patient**-out, 0 leaked folds, and it scales to 13 patients.
   `summarise()` no longer assumes `vals[:2]` are the HM folds. **The existing
   `Outputs/stage2/*` checkpoints were trained under the leaky scheme.**
3. **`build_qc_mask.py` had the broken-`ROOT` bug** (`Path(__file__).parent` → `04_Models/`), plus a
   hardcoded 6-sample list. Now resolves via `_cohort.ROOT` and reads `ALL_SAMPLES`.
4. **`build_stage2_notebook.py` wrote to `phase_a_stage2.ipynb`** via a CWD-relative path, but the
   restructure renamed the notebook to `stage2a_trimodal_training.ipynb` — it had been emitting an
   orphan file. Now writes next to itself, under the real name.
5. **`VisiumV1` `misc` slot.** The published object predates the `misc` slot SeuratObject 5.x
   declares, so `subset()` dies with *"slots in class definition but not in object: misc"*. The
   split backfills it on load.
6. **`write_spot_matrix` assay/subdir.** The RCTD assay is `rctd_fullfinal` but its output dir is
   `rctd`; these are now separate arguments.

### Performance (all output-identical)

| change | before | after | check |
|---|---|---|---|
| counts CSV write (`write.csv` → `data.table::fwrite`, int storage) | 44.2 s/sample | **1.3 s** (35×) | md5 identical |
| embedding load (per-spot `.pt` → per-sample `.npz` cache) | 7.45 s | **0.02 s** (372×) | bit-exact on 39,254 tensors |
| `img_for()` sample→image resolution | O(samples × images × spots) | precomputed map | 30/30 matched |
| slim `.rds` write | gzip level 6 | level 1 | ~50 MB/sample |

`_cohort.load_spot_embeddings(dir, sample, stems)` builds `<dir>/<sample>.__stackcache_v1.npz`
on first use and returns `(X, found)`; `found` replaces the old `os.path.exists()` check.

> **NumPy 2 trap.** Under NEP-50 a weak `np.nan` scalar does **not** promote a float32 array, so
> `np.where(mask, V[:, i], np.nan)` silently yields float32 and serialises fewer digits. The cast to
> float64 in stage1a/1b is load-bearing — without it the RCTD fraction columns drift by ~3e-08.

### Cohort selection

`_cohort.py` now takes `COHORT=six` (default, the validated 6 slides) or `COHORT=full`
(30 slides via `dataset/full_cohort/patient_map.csv`), and exposes `SITE_OF`, `PT/HM/LNM/NP_SAMPLES`,
`PATIENTS`. Sample order is a **natural** sort (T1 < T3 < T4 < T11), which reproduces the original
hardcoded lists exactly — order is load-bearing because stage1a concatenates per-sample frames in
`PT_SAMPLES` order.

### Regression check

Stages 0/1a/1b/3/4 re-run under `RCTD_paper` reproduce the §1 table:
`frozenFM_resid +0.121`, `resid CAF +0.027`, `target_raw vs extEMT +0.205`, ceiling `+0.113`,
`vision_raw − ceiling +0.177`, EMT #16/29. `leaving_program_scores.csv` diffs to **exactly 0.0**
against the pre-migration file. Stage-0 `metrics.json` moves ≤1.8e-03 in logistic-regression
accuracies only — BLAS/sklearn version drift, not the cache (proven bit-exact separately).

---

## 8. Stage 1A full cohort — the convergence test, and it is NEGATIVE

`04_Models/stage1a_full_cohort.py` → `Outputs/stage1a_full_cohort_residcaf/`.

**Part 1 — the target rebuilds cleanly on 10 primaries (34,069 spots, 10 patients).**
Scoring is identical to `stage1a_leaving_program.py`; on the overlapping slides it reproduces the
6-sample numbers exactly (T1: Moran's I 0.389, core~heldout +0.263, core~CAF +0.463). Across all 10:
Moran's I 0.28–0.67 (every slide p = 0.001) and core~heldout +0.20 to +0.62 — so the leaving
programme is spatially coherent and generalises to held-out EMT genes everywhere. The CAF coupling
also persists cohort-wide (core~CAF +0.17 to +0.72, median ≈ +0.56).

**Part 2 — does the leaving programme mark primary tumour that resembles metastasis?**
Train metastasis-vs-primary on tumour-dominated spots (scVI latents), leave-one-patient-out, then
score each held-out patient's PRIMARY spots and correlate with the target.

| trained on | ρ(resemblance, leaving_raw) | ρ(resemblance, leaving_resid) | per-patient median |
|---|---|---|---|
| both met sites | +0.002 | **+0.066** | +0.106 (7/10 positive) |
| liver mets only | +0.083 | **+0.124** | +0.148 (9/10 positive) |
| lymph-node mets only | −0.064 | **+0.001** | +0.074 (6/10 positive) |

**The ST leaving-programme target is largely INDEPENDENT of the transcriptional metastatic axis.**
Training an H&E model to predict it is therefore not training it to predict metastatic behaviour.
This is review risk #3 ("met ≠ primer") — previously an assumption, now measured on 13 patients.

What weak convergence exists is **liver-specific** (+0.124 HM-trained vs +0.001 LNM-trained), which
matches the cosine-0.466 finding above: the two metastatic directions are only partially shared.

This does **not** invalidate the target as a biological measurement — it is spatially coherent and
validated against held-out EMT genes and (from §1) an independent paper GSEA at +0.205. It says the
target measures *EMT/desmoplasia within the primary*, which is not the same thing as *metastatic
propensity*.

### 8b. Which target should Phase B use? — `stage3_target_comparison.py`

The obvious fix is to re-target Phase B on `met_resemblance` (metastasis-derived by construction).
Tested directly: same frozen UNI2-h features, same leave-one-patient-out ridge, same spots — only the
label changes. Restricted to the 4 primaries that have H&E patches (T1, T3, T4, T11).

| target | pooled held-out ρ | per-patient |
|---|---|---|
| leaving programme (raw) | — | |
| leaving programme (residualised) | **+0.116** | +0.14 / −0.07 / +0.02 / −0.02 |
| **met_resemblance** (metastasis-derived) | **−0.043** | — |
| correlation between the two targets | +0.101 | |

*(+0.116 independently reproduces Stage 3's +0.121 on a different code path, so the harness is sound.)*

**This is the crux of the project, and it is a squeeze:**
- the target H&E *can* predict (+0.116) is **not** a metastasis proxy (§8);
- the target that *is* metastasis-derived is **not predictable from H&E** (−0.043).

So on the data available today, "predict metastatic behaviour from H&E alone" is not supported.
What H&E demonstrably reads is EMT/desmoplasia in the primary — a real and defensible finding, but a
different claim from the one the project has been making.

**Before treating this as final, note the two ways it could be a power problem, not a true negative:**
1. Only **4 patients** carry vision. LOPO over 4 folds is extremely thin, and the per-patient spread
   (+0.14 to −0.07) is consistent with noise. The other 6 primaries have no patches (§ WSI blocker).
2. `met_resemblance` is a noisy label — the classifier producing it runs at ~0.62 balanced accuracy,
   and label noise attenuates any achievable correlation.

Both are resolved by the same thing: **the full-resolution WSIs.** With patches for all 10 primaries
this becomes a 10-fold test on a much larger spot set. Until then treat −0.043 as "no evidence of
signal", not "proof of no signal".

Remaining option worth trying now, needing no WSIs: test either target against the clinical fields in
`dataset/full_cohort/cohort_metadata.csv` (AJCC stage, grade, lymphovascular/perineural invasion,
nodes positive, survival start date) — the project has never had external clinical validation.

---

## 9. Stage 5 — decomposing the metastatic axis (`stage5_shared_met_axis.py`)

Follows directly from §1's finding that the HM and LNM metastatic directions are only partly shared.
Fits met-vs-primary separately per site on tumour-dominated spots in scVI space, splits into a
SHARED direction (w_HM + w_LNM) and a SITE-SPECIFIC one (w_HM − w_LNM), scores all 10,219 primary
spots out-of-fold, then characterises both against the 27 Bagaev signatures.

**Geometry.** cos(w_HM, w_LNM) = **+0.466**; the shared axis aligns **+0.856** with each site's own
direction. So a substantial site-independent component genuinely exists — it is not all site noise.

**The clearest biological result is the SITE-SPECIFIC axis, and it is an immune axis:**
Th2 +0.190, Effector-cells +0.189, Treg +0.188, Th1 +0.181, Checkpoint-molecules +0.159.
That is exactly what it should be — **lymph nodes are lymphoid tissue**, so what most distinguishes a
nodal from a hepatic metastasis is immune content. This is a useful internal validity check: the
decomposition recovers a known biological difference without being told about it.

**The shared axis is weak and not robustly characterised.** Top hits are ECM +0.103,
Matrix-remodeling +0.075, CAF +0.072, with immune signatures negative (~−0.12). But every |ρ| ≤ 0.13,
and the fges "ECM" signature (+0.103) disagrees with Stage-1a's 7-gene `score_ecm` (+0.000) — two
measures of the same thing pointing different ways. Treat the shared axis as *real but not yet
interpretable*; do not build a story on it.

**H&E does not predict any of these axes** (4 primaries, leave-one-patient-out):
shared −0.111, site-specific −0.002, HM-axis −0.049, LNM-axis −0.155. Combined with §8b's
met_resemblance −0.043, that is **five** metastasis-derived targets, none recoverable from H&E,
against +0.116 for the leaving programme. The consistency matters more than any single number,
though all of it rests on only 4 patients.

**Relation to the Stage-1a target.** Pooled correlations of the leaving programme with the shared
axis are near zero (+0.036 raw, +0.078 resid) but per-patient medians are higher (+0.126, 6–8 of 9
patients positive) — a between- vs within-patient variance effect. Weak either way.

### Not attempted: clinical validation

`cohort_metadata.csv` populates the clinical fields for only **6 of 13 patients**, every one of them
pM1, with a surgery date but no follow-up or event — so there is no survival endpoint and no useful
stage variation. Running associations on that would be fitting noise. It needs the full clinical
table from the lab before it is worth doing.

---

## 10. Stage 6 — the site-specific program (`stage6_site_specific_program.py`)

Where §9 worked in scVI space on spots, this works on **purity-gated pseudobulk per section**, so
the unit of analysis is a section and the features are interpretable (15 RCTD fractions + 27 Bagaev
signatures = 42).

**The core control is the purity sweep**, not the p-value. If a difference between liver and node
mets is leftover host tissue it must shrink as the tumour threshold rises. It does not:

| purity ≥ | n HM | n LNM | hepatocyte (HM) | lymphoid HM → LNM ratio | p |
|---|---|---|---|---|---|
| 0.30 | 10 | 5 | 0.133 | 2.44 | 0.020 |
| 0.50 | 9  | 5 | 0.085 | 2.25 | 0.014 |
| 0.70 | 6  | 3 | 0.028 | 1.96 | 0.083 |
| 0.80 | 5  | 3 | 0.017 | 2.30 | 0.125 |

Hepatocyte contamination falls 8× while the lymphoid ratio holds near 2. The rising p-values are
n collapsing (10→5 sections), not the effect fading.

**Geometry.** cos(HM-shift, LNM-shift) = **+0.547** on the pseudobulk — about half the departure
from the primary is shared, half is destination-specific.

**The differential.** Only two of 42 features survive FDR: `rctd::Hepatocytes` (`d = −1.38`,
`q = 0.042` — the trivial positive control, and a useful sanity check that the test works) and
**`rctd::B cells`** (`0.0044` in HM vs `0.0594` in LNM, `d = +2.03`, `q = 0.042`, higher in LNM in
**4/4** paired patients: PT_6, PT_8, PT_10, PT_12). The paired Wilcoxon is `p = 0.125`, which is the
**floor** for n=4 — it cannot reach significance and its value is the direction consistency, not the
number. Nodal deposits also trend higher on Protumor-cytokines, PVL and DCs (`d ≈ 1.0–1.15`,
q not significant); liver deposits trend higher on FCN1-TAM, Treg and Th2.

> Read honestly, this is an effect-size-and-direction result on 5 LNM sections, not a
> well-powered significance result. Item 2 of the "What is NOT done" list above is what turns it
> into something defensible.

---

## 11. Stage 7 — external replication (`stage7_external_replication.py`)

GSE274557 (Maitra lab), **13 treatment-naive patients**, four site types — 13 primary, 15 liver,
14 peritoneal, 6 lung sections entering the pseudobulk. Treatment-naive matters: it removes the
differential-chemotherapy explanation that is still open in the discovery cohort.

Same construction as stage 6 — per-site shift away from the matched primary, then the cosine
between shifts:

| pair | cosine |
|---|---|
| liver vs lung | **−0.205** |
| lung vs peritoneal | **−0.247** |
| liver vs peritoneal | −0.004 |
| **mean** | **−0.152** |

So in an independent, treatment-naive cohort the site shifts are **unrelated to opposed** —
*stronger* than the discovery cohort's `+0.547`, not weaker. Liver deposits come out immune-poor,
lung deposits immune-rich.

**One caveat to carry.** The immune feature range across sites (`1.164`) is barely above the
non-immune range (`1.148`), so "the difference is *specifically* immune" is well supported in the
discovery cohort and only weakly supported here. The text should say the shifts differ, and that
immune features are the clearest example, rather than claiming immune specificity in both.

**This cohort has no lymph nodes**, so it replicates the general claim and cannot touch the B-cell
one.

---

## 12. Stage 8 — cross-organ transfer on HEST-1k (`05_HEST/hest_cross_organ.py`)

A **prediction derived from stages 6 and 7 before it was run**, not a finding fitted to them: if
tissue context governs expression that strongly, an H&E→expression model should be organ-specific.
Our own cohort (4 patients with usable H&E) cannot test it; HEST-1k can.

**Design.** UNI2-h 1536-d per spot — the same backbone and preprocessing as the rest of the project,
so the numbers are comparable. Target is CP10k + log1p over a 50-gene common highly-variable panel.
Ridge (`alpha = 1000`), metric per-gene Pearson r on held-out sections. **Every split is by sample**,
never by spot — spots within a section are not independent and a spot-level split would leak.
156 sections, 7 organs with ≥4 sections (Bowel 61, Prostate 34, Kidney 24, Brain 22, Breast 7,
Lymph node 4, Pancreas 4).

| condition | mean r | cost |
|---|---|---|
| within organ (leave-one-sample-out) | **0.235** | — |
| same organ, **different study** | **0.199** | −15 % (batch) |
| **across organs** | **0.125** | −37 % further (organ) |

**The batch control is the result.** Without it, the cross-organ drop is unattributable — a
different organ is usually also a different laboratory. With it, the organ effect is clearly the
larger of the two. This nearly went the other way: before three sections using a targeted gene panel
were excluded, the same code concluded the loss was batch rather than organ. Commits `71b9a26` and
`7eaabb0` are that sequence.

**Two limits to state in any write-up.** Only Bowel, Prostate, Brain and Breast had ≥2 studies, so
the batch control covers 4 of 7 organs; and absolute r is modest throughout — these numbers compare
conditions against each other and are not a claim that expression is accurately recoverable from an
image.

---

## 13. Stage 9 — nulls and uncertainty (`stage9_cosine_nulls.py`), 2026-08-29

Runs off committed CSVs only — no `dataset/`, no GPU, ~2 min on the laptop. **Both existing
pipelines are reproduced exactly first** (stage 6 `+0.547`; stage 7 `−0.205 / −0.004 / −0.247`), so
the nulls apply to the same quantities the manuscript reports.

### The null is not zero, and the test runs the other way

Both site-shifts are measured against the **same** primary centroid, so they share a "becoming a
metastasis" component by construction. If liver and node deposits ran the *same* program the cosine
would sit near +1, limited only by sampling noise. So the question is whether the observed cosine is
significantly **below** what shared-program-plus-noise produces — a **lower-tail** test. Shuffling
site labels among the 14 metastatic sections at fixed group sizes (9/5, all 2002 assignments
enumerated exactly) gives that null.

### Discovery cohort — the headline does not survive

| | raw (as published) | SD-scaled (stage 7 convention) |
|---|---|---|
| observed | **+0.547** | +0.445 |
| patient bootstrap 95 % CI | [+0.019, +0.762] | [−0.031, +0.606] |
| site-label null, median | +0.756 | +0.651 |
| **P(null ≤ observed)** | **0.144** | 0.133 |
| detection limit (one-sided p<0.05) | +0.396 | +0.329 |
| within-HM split-half ceiling | +0.768 (126 splits) | +0.642 |
| within-LNM split-half ceiling | **+0.572** (10 splits) | +0.562 |

Two ways of seeing the same thing. The permutation p is `0.144` — the observed cosine sits inside
the null. And the **within-LNM split-half ceiling (+0.572) is essentially the observed cross-site
value (+0.547)**: two halves of the lymph-node group agree with each other no better than the node
group agrees with the liver group. At 9 HM and 5 LNM sections the cosine would have to fall below
`+0.396` to be detectable, so this cohort cannot resolve the global contrast in either direction.

> **This is not a negative result about the biology — it is a power result about the cohort.**
> "~55 % shared" should not appear as a claim. What can be said is that the discovery cohort is
> uninformative about the *global* geometry, while remaining informative about *specific features*.

### Scale: the two cohorts were never computed the same way

Stage 6 takes the cosine on **raw** features; FGES scores reach ~5,200 while RCTD fractions are
~0.00006 — a spread of **8×10⁷**, so the raw cosine is set by a handful of large signatures and the
15 cell-type features contribute almost nothing. Stage 7 divides by the feature SD. `+0.547` and
`−0.152` were therefore never comparable. Both conventions are now reported; the conclusion is the
same under either, so nothing downstream turns on the choice — but the manuscript must pick one.

### Replication cohort — this is where the evidence actually lives

| pair | observed | null median | P(null ≤ obs) |
|---|---|---|---|
| HM vs LuM | −0.205 | +0.630 | **0.0005** |
| HM vs PM | −0.004 | +0.717 | **<0.0001** |
| LuM vs PM | −0.247 | +0.177 | **0.0123** |

All three clear their nulls decisively. Treatment-naive, three sites, 48 sections. **The site-
specificity claim should be led by GSE274557 and supported by GSE272362, not the other way round** —
which inverts the current discovery/replication framing of the manuscript.

### Patient-level differential — the B-cell result survives

Stage 6 tested 9 vs 5 **sections**, but PT_2 contributes two HM sections. Aggregating to
patient×site means (8 HM patients vs 5 LNM, 4 paired) changes almost nothing:

| feature | d (section) | q (section) | d (patient) | q (patient) | paired |
|---|---|---|---|---|---|
| `rctd::B cells` | +2.03 | 0.042 | **+2.01** | **0.065** | 4/4 |
| `rctd::Hepatocytes` | −1.38 | 0.042 | −1.50 | 0.065 | 0/4 |

Still exactly two features at q<0.10, still the same two, and the hepatocyte positive control still
behaves. The q drifts above 0.05 purely from losing one unit of n. Exact sign test on the 4 paired
patients is `p = 0.125` — the floor at n=4, so direction consistency (4/4) is the evidence, not the
p-value. Full table: `Outputs/stage9_cosine_nulls/differential_section_vs_patient.csv`.

---

## 6. Open questions worth carrying forward

- Does the +0.121 EMT signal recover with 10 PT + 12 HM slides, or stay at noise?
- Pathologist annotation is still the missing external ground truth — Test 3 uses an RCTD-derived
  margin proxy, not a real annotation. Worth requesting from Dr. Ashiq.
- Phase A overfits fast (best epochs 4–14 on CONCH V1); `Pv [768, 384]` was sized for UNI2-h (1536d).
- `GSE277783` (Nature 2025, Maitra) is **not** usable for training — CosMx (~1,000-gene panel, IF not
  H&E) + bulk. External validation only.
- TCGA-PAAD via UCSC Xena is bulk — validation only. GDC does hold 466 open-access PAAD slide images,
  which would allow an H&E-only survival check.

New, opened by the cohort split (2026-07-29):
- **Does the H&E signal survive in LNM?** Lymph-node mets carry no hepatocytes. If the
  metastatic-direction axis holds for LNM as well as HM, it is not a liver-tissue artifact — this is
  the cleanest confound test available and the 6-sample cohort could not run it.
- **NP as a true negative.** 3 normal-pancreas slides (PT_2, PT_10, PT_11) give a real baseline for
  "leaving programme" scores; previously there was none.
- **The tumour-only HM-vs-PT test is now RUNNABLE — confirmed, not speculation.** On 6 samples it
  returned `nan` because all 876 tumour-dominated HM spots came from HM11, so no fold could hold one
  out. Measured on the split (Tumor Epithelial ≥ 0.50):

  | site | slides | tumour-dominated spots | patients contributing | mean hepatocyte |
  |---|---|---|---|---|
  | T   | 10 | 10,443 | 10 | 0.001 |
  | HM  | 12 | 9,665  | **9** | 0.322 |
  | LNM | 5  | 7,488  | 5  | **0.0003** |
  | NP  | 3  | 57     | 3  | 0.013 |

  **9 HM patients** now contribute tumour-dominated spots (was 1). §1's "Only the cohort expansion
  fixes this" is discharged. Note HM9 and HM13 still contribute zero — HM13 was the informative-free
  slide that made the old test degenerate.
- **LNM is a near-perfect confound control**: 7,488 tumour-dominated spots across all 5 patients at
  mean hepatocyte fraction **0.0003**, vs 0.322 for HM. Any HM-vs-PT axis that also separates
  LNM-vs-PT cannot be explained by liver tissue.
- `PT_10` has all four sites, so a full within-patient T→LNM→HM→NP series is available for one patient.
