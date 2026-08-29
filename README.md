# PDAC Spatial Transcriptomics — Site-Specific Metastatic Programs

Paired **H&E histology + spatial transcriptomics (10x Visium)** across three cohorts, asking whether
"metastasis" in pancreatic cancer is one transcriptional state or several.

It is not. **Where a tumour spreads changes what it becomes**, and that finding is what the
repository is now organised around.

> Plain-language walkthrough: [PROJECT_EXPLAINED.md](PROJECT_EXPLAINED.md).
> Engineering log — environment, every fix, every decision: **[HANDOFF.md](HANDOFF.md)**.

---

## Headline result

**Deposits at different sites differ in specific, identifiable features — B cells above all.** In
the discovery cohort, B-cell content is one of only two features surviving FDR correction between
liver and lymph-node deposits (`d = +2.03` by section, `+2.01` with the patient as the unit, same
direction in 4/4 paired patients), and it holds as tumour purity rises from 30 % to 80 % while
hepatocyte contamination falls eightfold.

**Globally, the site-specificity is established in GSE274557** — 13 treatment-naive patients,
liver/lung/peritoneal deposits — where all three site pairs diverge far below their permutation
nulls (`p = 0.0005`, `<0.0001`, `0.012`; observed cosines `-0.205`, `-0.004`, `-0.247`).

The discovery cohort's own global cosine (`+0.547`) does **not** clear its null (`p = 0.144`): at
9 liver and 5 node sections it would have to fall below `+0.396` to be detectable, and two halves of
the node group agree with each other at `+0.572` — no better than node agrees with liver. That is a
power limit, not evidence of sameness, and it is why the claim is stated per-feature for the
discovery cohort and globally for the replication cohort. See stage 9 and HANDOFF §13.

A prediction derived from those two results, then tested on **HEST-1k** (156 sections, 7 organs):
an H&E-to-expression model should not cross organ boundaries. It doesn't. Crossing laboratories
within one organ costs 15 % of performance (`0.235 -> 0.199`); crossing organs costs a further
37 % (`-> 0.125`). **The organ effect is larger than the batch effect.**

The original aim — recovering metastatic behaviour from primary-tumour H&E — returned a chain of
negatives that now read as consequences rather than failures. See
[Negative results](#negative-results-and-why-they-matter) below.

---

## Cohorts

| Role | Dataset | Content |
|---|---|---|
| **Discovery** | GSE272362 (Khaliq *et al.*, Nat Genet 2024) | 30 sections / 13 patients / 91,496 spots (87,055 QC-passing) — 10 primary, 12 liver-met, 5 node-met, 3 normal pancreas |
| **Replication** | GSE274557 (Maitra lab 2025, PMID 40269162) | 55 sections / 13 treatment-naive patients; 48 enter the purity-gated pseudobulk — primary, liver, peritoneal, lung |
| **Generalisation** | HEST-1k (Jaume *et al.*, NeurIPS 2024) | 156 human Visium cancer sections across 7 organs |

Discovery-cohort scale is selected with `COHORT=full`; `COHORT=six` keeps the original validated
6-slide subset that stages 0–4 were developed against.

---

## Repository structure

```
ST_Project/
├── 01_Patch_Extraction/        # H&E patches at Visium spot coordinates
│   ├── create_patches.py       #   224x224 patches; inventory-driven
│   ├── wsi_inventory.py        #   which of the 30 samples has a usable WSI -> wsi_inventory.csv
│   ├── match_capture_areas.py  #   4-area slide identification (NOT reliable — validated 3/21)
│   └── build_area_review_sheet.py  # human-review sheets for unlabelled capture areas
├── 02_Gene_Export/             # QC-filtered count matrices
├── 03_Embedding_Extraction/    # Per-spot embeddings
│   ├── Cell/                   #   RCTD deconvolution (15-d) — use the paper's `rctd_fullfinal`
│   ├── Gene/                   #   scVI latents (50-d); scvi_full_cohort.py runs all 30 sections
│   └── Vision/                 #   UNI2-h / CONCH / H-Optimus-1; extract_uni2h_local.py (GPU)
├── 04_Models/                  # stages 0-7 — see the pipeline table below
├── 05_HEST/                    # stage 8: HEST-1k subset download, UNI2-h embedding, cross-organ test
├── dataset/                    # not in git — see HANDOFF for what lives here
│   ├── full_cohort/            #   30-section split (counts, rctd, fges, coords, scVI)
│   └── external/               #   GSE274557 replication + HEST
├── Outputs/                    # results, tagged by cohort/variant so nothing overwrites
└── docs/                       # figures, reports, manuscript builder
```

Stage scripts are driven by environment variables — `COHORT` (`six`|`full`),
`RCTD_VARIANT` (`RCTD_paper`|`RCTD`), `RESID_ON` (`tumor`|`tumor_caf`) — and write to a
correspondingly tagged output directory, so re-running a variant never clobbers another one.

---

## Pipeline

| Stage | Script | What it does |
|-------|--------|-------------|
| **0** | `stage0_confound_diagnostic.py` · `stage0_full_cohort.py` | Liver-vs-pancreas confound; site contrasts with a patient-shuffle null |
| **1a** | `stage1a_leaving_program.py` · `stage1a_full_cohort.py` | Intra-primary EMT "leaving program"; transfer from each metastatic site |
| **1b** | `stage1b_pt11_hm11_anchor.py` | PT11 to HM11 convergence check (null, reported honestly) |
| **2** | `stage2a_trimodal_training.ipynb` · `stage2b_model_comparison.py` | Tri-modal InfoNCE bridge; 4 histology foundation models compared |
| **3** | `stage3_phase_b_scoring.py` · `stage3_target_comparison.py` | LOSO RidgeCV head; bridge vs frozen-FM ablation; which target is recoverable |
| **4** | `stage4_validation.py` | 6 pre-registered validation tests; external GSEA from the source paper |
| **5** | `stage5_shared_met_axis.py` | Decomposes the metastatic axis into shared and site-specific parts |
| **6** | `stage6_site_specific_program.py` · `stage6_figures.py` | **The result** — purity-swept liver vs node program |
| **7** | `stage7_external_replication.py` · `stage7_figures.py` | GSE274557 replication across three sites |
| **8** | `05_HEST/hest_cross_organ.py` · `05_HEST/stage8_figures.py` | Cross-organ H&E-to-expression transfer, batch-controlled |
| **9** | `stage9_cosine_nulls.py` | Permutation nulls, patient bootstrap CIs and a patient-level differential for stages 6-7 |

Every number in `docs/reports/` is regenerable from these scripts; outputs land in the matching
`Outputs/stage*/` directory.

---

## Negative results, and why they matter

These were the original project. They are kept because they are what motivated the positive finding.

- **Metastasis-vs-primary is largely the organ.** Cell composition separates liver mets from
  primaries at `0.883` balanced accuracy, but `0.822` on tumour-dominated spots and `0.776` once
  hepatocyte *and* tumour fraction are removed. For node mets the same contrast is `0.585`. A
  patient-shuffle null sits at `0.469`, so the signal is real — it is simply site-dependent.
- **The transferable "leaving program" is liver-specific.** Trained on liver mets it transfers into
  primaries at `rho = 0.124`, positive in **9 of 10 patients**. Trained on node mets: `rho = 0.001`,
  6 of 10. Same method, same primaries, opposite outcome by destination.
- **H&E recovers the intra-primary program, not the metastatic one.** Across 4 patients /
  2,679 spots: leaving programme `rho = 0.229` raw, `0.116` residualised, metastasis-derived
  resemblance `-0.043`. The two targets correlate at only `0.101` — they were never the same thing.
- **The contrastive bridge adds nothing** over the frozen foundation model, at any of the four
  vision backbones tested.

---

## Environment

Python — a **pure-pip** scientific stack (the `stproj` env on the current machine):

```
numpy>=2   pandas<3   scipy   scikit-learn   matplotlib   seaborn   umap-learn   openpyxl
torch      # CUDA build for stages 2 and 8; CPU is sufficient for stages 0-1 and 3-7
```

Do **not** mix conda's numpy/scipy/scikit-learn with pip's torch: the two OpenMP runtimes make
`sklearn.decomposition.PCA` abort the process silently at exit 127, with no traceback. `pandas` is
pinned `<3` because pandas 3's copy-on-write and string-dtype changes can silently move numbers
that validated results depend on.

R 4.5+ with `Seurat`, `SeuratObject`, `Matrix` and `data.table` is needed only for
`04_Models/split_full_cohort.R`. `spacexr` is **not** required — the paper's own `rctd_fullfinal`
assay is used rather than a local RCTD re-run (HANDOFF explains why this matters).

---

## Data

Raw data is not in git. The `dataset/` layout, sizes and provenance are documented in
[HANDOFF.md](HANDOFF.md); the six-sample subset also lives on Kaggle as
[wanianaeem/zenodo-pt-and-hm-dataset](https://www.kaggle.com/datasets/wanianaeem/zenodo-pt-and-hm-dataset).

[REVIEW_PLAN.md](REVIEW_PLAN.md) is the historical execution log for stages 0–4 — kept as a record
of how the negatives were arrived at, not as a live plan.
