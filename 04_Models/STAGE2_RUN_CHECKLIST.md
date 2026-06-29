# Stage 2 — Run Checklist (`phase_a_stage2.ipynb`)

REVIEW_PLAN.md Stage 2. The notebook auto-detects Kaggle vs local — same code path on both.

## 0. One-time Kaggle dataset prep (REQUIRED for Variant B)
The Stage-1A target is the only new dependency vs the original notebook.

1. Open `kaggle.com/datasets/wanianaeem/zenodo-pt-and-hm-dataset` → **New Version**.
2. Drag in **`Outputs/stage1a_leaving_program/leaving_program_scores.csv`** (rename is fine,
   but keep it as `leaving_program_scores.csv` at the dataset root).
3. Confirm these are already in the dataset (from the original Phase-A run):
   - `Feature Extraction Embeddings/CONCH V1/` (vision .pt bundles)
   - `scvi_latent_pt_embeddings/` (gene .pt, per spot)
   - `Cell Embedding Extraction/RCTD/` (cell .pt, per spot)
   - `spot_qc_mask.csv`
4. Save the new dataset version.

If `leaving_program_scores.csv` is missing, the notebook **auto-skips Variant B** and runs A only
(it prints a warning). So always add it.

## 1. Run
- **Kaggle:** attach the dataset + a **GPU** accelerator, then *Save & Run All*. No path edits needed.
- **Local:** open in the `tcga` env (`C:\Users\datai\anaconda3\envs\tcga\python.exe`). Paths auto-resolve.

The notebook runs **both variants in one pass** (`VARIANTS_TO_RUN = ["A","B"]` in the config cell):
- Variant A = baseline tri-modal InfoNCE, `proj_dim` 256→**128**, batch-balanced sampling.
- Variant B = A **+** auxiliary head regressing the Stage-1A `leaving_score_resid` from the
  vision projection (`aux_weight=0.3`, PT spots only, masked).

## 2. What it does differently from the original Phase-A notebook
| Change | Why (REVIEW_PLAN) |
|---|---|
| `proj_dim` 256 → 128 | folds 1–2 stopped ep ~4–14 = capacity red flag |
| Batch-balanced sampler (`WeightedRandomSampler`) | kills slide-identity shortcut in InfoNCE |
| Retrieval reported **neighbour-excluded** (`R=2` Chebyshev, same-sample) | spatial autocorrelation inflates R@1 |
| Variant B aux head | whole-transcriptome alignment may wash out the metastasis subspace |
| **Decision gate = held-out vision→leaving ridge probe** | the real A-vs-B comparison (not val loss) |

## 3. The decision gate
For each **PT-holdout fold (3–6)**, a Ridge probe is trained on the *training* patients'
vision projections → leaving score, then predicts the **held-out PT patient** (pure
patient-level generalisation). Reported as Spearman ρ + R² for **both** `leave_resid`
(headline, confound-free) and `leave_raw` (abundance-aware upper bound).

**Winner = higher mean `probe_rho_resid` across folds 3–6.** Printed in the
`STAGE 2 DECISION SUMMARY` block and saved to `stage2_decision_summary.json`.
HM folds (1–2) have no leaving labels → probe is `None` there (expected).

## 4. Outputs (`Outputs/stage2/` local, `/kaggle/working/stage2/` on Kaggle)
- `{A,B}_fold{1..6}.pt` — per-fold checkpoints
- `stage2_decision_summary.json` — A-vs-B metrics + winner
- `stage2_training_curves.png` — LOSO curves, both variants
- `final_model_{winner}.pt` — winner retrained on all 6 samples
- `all_spots_embeddings.pt` — **Phase B input** (vision/gene/cell 128-d projections + rows/cols/samples)
- `config.json`

## 5. After it finishes — the gate decision
- Note the winner from the decision summary.
- **Scale the winner to UNI2-h (1536d):** set `cfg.VISION_DIR` to the `UNI2-h` folder
  (Kaggle: `.../Feature Extraction Embeddings/UNI2-h`; local: same under `dataset/`) and re-run.
  Everything else is dim-agnostic (`Pv` infers `vision_dim`).
- Then proceed to **Stage 3** (Phase B scoring + confound audit) using `all_spots_embeddings.pt`.

## Notes / caveats baked in
- Within-tumour EMT (`leave_resid`) is a **modest** signal at 55 µm (Stage 1A) → expect a
  **low probe ceiling** by construction. `leave_raw` is reported alongside to bound it; do not
  read a low resid ρ as failure on its own — compare A vs B *relative* to each other.
- Aux head trains on the z-scored headline target (`leaving_score_resid`); switch to raw via
  `cfg.aux_target` if you want the abundance-aware variant.
