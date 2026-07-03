# PDAC Spatial Transcriptomics — Multimodal Metastatic Biomarker Pipeline

A 4-stage research pipeline that uses paired **H&E histology + spatial transcriptomics (10x Visium)** to discover and score an intra-tumour EMT "leaving program" in pancreatic cancer (PDAC), with the goal of predicting liver-metastasis propensity from H&E alone at inference.

**Cohort:** 6 PDAC samples — 4 primary tumours (T1, T3, T4, T11) + 2 hepatic metastases (HM11, HM13) — 18,859 QC-passed spots.

> For a full plain-language walkthrough of the project, findings, and what they mean, see [PROJECT_EXPLAINED.md](PROJECT_EXPLAINED.md).

---

## Repository Structure

```
ST_Project/
├── 01_Patch_Extraction/        # Extract 224×224 H&E patches from Visium WSIs
├── 02_Gene_Export/             # Export scVI-normalised count matrices
├── 03_Embedding_Extraction/    # Per-spot embedding notebooks
│   ├── Cell/                   #   RCTD cell-type deconvolution (15-d)
│   ├── Gene/                   #   scVI / Harmony / Scanorama gene latents (50-d)
│   └── Vision/                 #   Foundation model embeddings: UNI2-h, CONCH v1/v1.5, H-Optimus-1, ResNet18
└── 04_Models/                  # Analysis pipeline (run in stage order)
    ├── build_qc_mask.py        #   Build spot QC mask
    ├── stage0_confound_diagnostic.py
    ├── stage1a_leaving_program.py
    ├── stage1b_pt11_hm11_anchor.py
    ├── stage2a_trimodal_training.ipynb
    ├── stage2b_model_comparison.py
    ├── stage3_phase_b_scoring.py
    └── stage4_validation.py
```

```
Outputs/
├── Patient-Sample-Information/ # Cohort metadata + source-paper supplementary tables
├── stage0_confound/            # Confound diagnostic results
├── stage1a_leaving_program/    # EMT leaving-program scores + figures
├── stage1b_pt11_anchor/        # PT11→HM11 resemblance analysis
├── stage3_phase_b/             # Phase B scoring results
├── stage4_validation/          # Validation test results
└── presentation_figures/       # Publication-ready figures (fig1–fig7 + slides/)
```

```
docs/
├── Data_Requirements.pdf
├── Research_Scope.pdf
├── presentation/               # Scripts that generate all figures and the Word report
└── reports/                    # Project reports and meeting PDFs
```

---

## Pipeline Overview

| Stage | Script | What it does |
|-------|--------|-------------|
| **0** | `stage0_confound_diagnostic.py` | Proves liver-vs-pancreas confound; demotes cohort Δ direction |
| **1a** | `stage1a_leaving_program.py` | Defines intra-PT EMT "leaving program" score (19 genes, 13,578 spots) |
| **1b** | `stage1b_pt11_hm11_anchor.py` | PT11→HM11 convergence check (weak/null, reported honestly) |
| **2** | `stage2a_trimodal_training.ipynb` | Tri-modal InfoNCE contrastive bridge (Phase A) — Kaggle GPU |
| **2** | `stage2b_model_comparison.py` | Compares foundation models on held-out prediction |
| **3** | `stage3_phase_b_scoring.py` | LOSO RidgeCV scoring head; decisive bridge vs frozen-FM ablation |
| **4** | `stage4_validation.py` | 6 pre-registered validation tests; external GSEA from source paper |

---

## Data

All raw data (Visium H5 files, H&E patches, pre-extracted embeddings) lives on Kaggle:
**[wanianaeem/zenodo-pt-and-hm-dataset](https://www.kaggle.com/datasets/wanianaeem/zenodo-pt-and-hm-dataset)**

Clone this repo and attach the Kaggle dataset to reproduce any stage.

---

## Key Result

> The intra-primary EMT "leaving program" is a real, externally-validated transcriptomic signal; but from a routine H&E slide the model currently recovers only **how much tumour is present**, not the confound-free metastatic state, on unseen patients. The multi-modal bridge adds nothing over the frozen foundation model.

See [REVIEW_PLAN.md](REVIEW_PLAN.md) for the full technical execution log with stage-by-stage decisions.

---

## Environment

```bash
conda activate tcga   # Python 3.12, PyTorch, scvi-tools, scanpy, spatialdata
```
