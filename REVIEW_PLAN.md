# PDAC ST + FM — Methodological Review & Staged Execution Plan

> Self-contained plan written after a counter-expert review of the project.
> Intended to be executed **one stage at a time**, reading top-to-bottom.
> Read this top-to-bottom before starting; each stage has a decision gate.

---

## 0. Context recap

**Goal:** Identify which primary-tumour (PT) spots are *liver-metastasis-prone*, using
Histopathology FM image embeddings augmented by Spatial Transcriptomics (ST). Hard
constraint (Dr. Ashiq): **ST only at training time; inference uses H&E alone.** The
visual projector must implicitly encode transcriptomic biology.

**Data:** 6 samples. PT: T1, T3, T4, T11. HM (hepatic met): HM11, HM13.
Only **patient 11 is matched** (PT11 + HM11). 18,859 QC-passed spots.
Modalities per spot: CONCH/UNI2-h vision embedding, scVI 50d gene latent, RCTD 15d cell proportions.

**Existing pipeline:** Phase A = tri-modal InfoNCE alignment (vision↔gene↔cell → 256d).
Phase B = metastatic scoring via direction vector Δ = μ_HM − μ_PT.

---

## 1. The three problems this plan exists to fix

1. **Liver-vs-pancreas confound.** μ_HM − μ_PT is dominated by *tissue of residence*
   (HM cells sit in liver), not metastatic potential. The vision score risks being a
   "looks like liver" detector.
2. **No per-spot metastasis ground truth.** No PT spot is labelled as having
   metastasised, so the Phase B [0,1] score is currently unfalsifiable. Validation
   protocol must be defined *before* generating scores.
3. **Established-met ≠ prone-primary phenotype.** HM transcriptome is the post-MET,
   liver-adapted *endpoint*; the "prone PT spot" is the *primer*. Different biology.

**Target decision (locked):** Primary endpoint = **intra-PT "leaving program"**
(EMT / invasive-front program defined *within* PT, confound-free).
Secondary = **μ_HM convergence** (does high-scoring PT resemble HM, esp. PT11→HM11?).
The headline result is the *agreement* of these two independently-derived axes.

---

## STAGE 0 — Confound diagnostic in gene space (do this FIRST, laptop, no GPU)

**Why first:** costs nothing and tells you whether the whole μ_HM approach is salvageable.

**Inputs:** scVI 50d latents (18,859 .pt), RCTD 15d proportions, QC mask, sample labels.

**Do:**
- Can RCTD cell composition *alone* separate HM vs PT? (logistic reg / simple classifier,
  patient-level cross-val). If yes → tissue confound is real and must be controlled.
- Re-test *within high-malignant-fraction spots only* (tumour-epithelial-dominated).
  Does the HM/PT separation survive when comparing tumour-to-tumour?
- How much of Δ = μ_HM − μ_PT (in scVI space) is explained by hepatocyte/liver-cell content?
- Negative control: train a tissue-of-origin classifier; note its accuracy as the
  "confound ceiling" any later vision score must beat / be decorrelated from.

**Visualisations:** UMAP of scVI latent coloured by (a) sample, (b) PT/HM, (c) RCTD
malignant fraction, (d) RCTD hepatocyte fraction. Confusion matrices. Δ-vs-composition
correlation bars.

**Decision gate:** If HM/PT separation *vanishes* within malignant-dominated spots →
μ_HM approach needs the within-tumour restriction (carry to Stage 3). If it stays huge
regardless → demote μ_HM to corroboration-only and lean entirely on intra-PT (Stage 1).

**>>> STAGE 0 RESULT (2026-06-28) — RAN, see `Outputs/stage0_confound/`:**
- Hepatocyte confound real but removable: corr(μ_HM dir, hepatocyte) 0.40 all → **0.07
  tumour-only**. Tumour-only restriction works.
- **No patient-generalizable tumour-intrinsic HM/PT signature**: tumour-only scVI LOSO
  predicts held-out HM patients at 0% (N=2 HM don't resemble each other). The generalizable
  HM/PT separator is microenvironment composition (RCTD 0.81), i.e. tissue confound.
- **DECISION: μ_HM → patient-11 matched anchor ONLY. No cohort-wide Δ_meta direction.**
  Intra-PT leaving program (Stage 1A) is the sole primary endpoint.
- **Pending follow-up before locking:** repeat tumour-only HM/PT test on a non-batch-corrected
  representation (marker-gene expression / harmony-PCA) — confirm signal is absent vs erased
  by scVI batch correction. Then proceed to Stage 1A.

---

## Gene-set audit result (resolved — no blocker)

- Count matrices are the **full transcriptome (~17,894 genes)**, not just the 62-gene panel.
- EMT/invasion coverage is strong: master EMT-TFs **SNAI1 + ZEB1** present (only TWIST1 of
  the big-three missing), plus CDH2, TGFBR1, MMP1, LOXL2, ITGB1, ITGB6, S100A4, KRT7.
  Full-transcriptome backups present: VIM, SNAI2, PRRX1, SPARC, POSTN, MMP9, MMP14, COL1A1,
  TGFB1; epithelial pole CDH1, EPCAM.
- The 6 "missing" markers (TWIST1, CDKN2A, IGF2BP3, RACGAP1, RARRES3, MALAT1) are
  **absent from the matrix entirely** → CANNOT be held-out validators. Instead hold out an
  independent *present* EMT subset (VIM, SNAI2, PRRX1, MMP9/14, POSTN, SPARC) + score
  external published PDAC-liver-met signatures over the gene overlap.
- Hepatocyte genes **ALB, APOA1, HP, TTR present** → use as a gene-level liver-confound
  score in Stage 0 (complements RCTD).

## STAGE 1 — Define the metastatic-propensity targets (laptop, no GPU)

**1A. Primary: intra-PT leaving program.**
- Score each PT spot for an EMT / partial-EMT / invasion program. Training signature core:
  SNAI1, ZEB1, CDH2, TGFBR1, MMP1, LOXL2, ITGB6, S100A4 (+ ECM panel). Hold out
  VIM, SNAI2, PRRX1, MMP9, MMP14, POSTN, SPARC as independent validators (Stage 4).
- Options: signature scoring (AddModuleScore-style) or unsupervised axis (NMF/PCA on EMT
  genes). Keep it simple and interpretable first.
- Output: continuous per-PT-spot "leaving-program" score → this is the regression target.

**1B. Secondary: patient-11 matched anchor ONLY** (revised by Stage 0 — cohort-wide μ_HM
direction is not viable; tumour-intrinsic HM/PT signal does not generalize across the 2 HM
patients).
- Compute PT11→HM11 resemblance within patient 11, tumour-dominated spots only.
- Use purely as a qualitative corroboration check in Stage 4 (do high intra-PT-leaving-score
  PT11 spots resemble HM11?), NOT as a training direction.

**>>> STAGE 1B RESULT (2026-06-29) — RAN, see `Outputs/stage1b_pt11_anchor/`
(`stage1b_pt11_hm11_anchor.py`):**
- Patient-11 ONLY, tumour-dominated (Tumor frac >= 0.5): PT11 = 976 spots, **HM11 = only 77
  spots** (HM11 is mostly liver/stroma → fragile anchor centroid). Non-batch-corrected log-norm,
  2000 HVGs, deliberately NOT the 1A EMT gene set (kept independent). scVI NOT used (would
  batch-correct away the PT→HM axis).
- **CONVERGENCE IS WEAK / mixed (the headline did NOT hold up):** corr(1A leaving raw, HM11
  resemblance) = **-0.07** (none); corr(1A leaving *residualized*, resemblance) = **+0.17**
  (weak, but N=976 so significant). Liver-gene exclusion changes nothing (axis corr 1.000).
- **Axis is STILL microenvironment, even tumour-dominated within one patient** (confirms Stage 0):
  PT11-like drivers = desmoplastic CAF/collagen (COL1A2, COL6A1/2, COL11A1, FBLN1, MMP14, SDC1);
  HM11-like = ALB (liver) + immunoglobulins IGHA1/IGKC/IGHG1 (plasma/B) + MT2A/ENG/HOPX.
  resemblance anti-correlates with tumor frac (-0.50) and hepatocyte (-0.26). Moran's I 0.84
  (p=0.001) → highly spatially structured = regional microenvironment, not metastasis biology.
- **DECISION:** 1B does NOT provide strong corroboration → lean ENTIRELY on the 1A intra-PT
  leaving program as primary. Report 1B honestly as a weak/null convergence check (≤0.17),
  which itself supports the Stage-0 μ_HM demotion. Keep as Stage-4 qualitative material only;
  do NOT promote to a headline "two-axes-agree" claim. **Stage 1 COMPLETE → proceed to Stage 2.**

**Visualisations:** spatial heatmaps of the leaving-program score on each PT slide;
distribution of scores per sample; overlap/correlation between 1A and 1B targets.

**Decision gate:** Confirm 1A score is spatially coherent (high spots cluster at
margins/fronts, not random) and correlates with held-out signatures. If not, revisit gene set.

**>>> STAGE 1A RESULT (2026-06-28) — RAN, see `Outputs/stage1a_leaving_program/`
(`stage1a_leaving_program.py`):**
- 13,578 PT spots scored across T1/T3/T4/T11. Core sig = 19 genes (12 EMT + 7 ECM);
  held-out validators VIM/SNAI2/PRRX1/MMP9/MMP14/POSTN/SPARC scored but never used to build it.
- Method: CP10k→log1p, Seurat-style AddModuleScore (expression-binned control subtraction),
  per-sample. Unsupervised PC1 cross-check r=0.77.
- **GATE 1 (spatial coherence) PASS:** hex Moran's I = T1 0.39 / T3 0.61 / T4 0.37 / T11 0.37,
  all p=0.001. Survives residualization (0.16–0.27).
- **GATE 2 (held-out generalization) PASS:** corr(core, held-out) = 0.46 pooled, positive on
  all 4 slides.
- **CONFOUND FOUND:** core score is 0.61-correlated with RCTD tumor fraction — ~37% is
  malignant *abundance* (inherent Visium composition effect), not EMT. Residualizing on tumor
  frac per sample → corr 0.00 with abundance; within-tumour EMT agreement (resid core vs resid
  held-out) = 0.12 — weak but positive & spatially coherent. corr(core, CAF) and (core,
  hepatocyte) both negative → not a stromal/liver readout.
- **DECISION (LOCKED):** Phase B target = **BOTH `leaving_score` (raw) and
  `leaving_score_resid` (tumour-fraction-residualized)**; residualized is the confound-free
  **headline**, raw is reported as the abundance-aware upper bound / ablation. Target CSV =
  `Outputs/stage1a_leaving_program/leaving_program_scores.csv`.
- **Implication for Stage 2/3:** within-tumour EMT is a modest signal at 55 µm; vision→resid
  prediction will have a low ceiling by construction — report raw alongside to bound it.

---

## STAGE 2 — Phase A training on Kaggle (GPU)

Keep the existing tri-modal InfoNCE, but run **two variants** to test Problem (whole-
transcriptome alignment washing out the metastasis subspace):

- **Variant A (baseline):** current pipeline, CONCH V1, 6-fold LOSO. Shrink `proj_dim`
  256→128 to cut overfitting (folds 1–2 stopped at ep ~4–14 = capacity red flag).
- **Variant B (metastasis-aware):** add an auxiliary head that regresses the Stage-1A
  leaving-program score from the vision projection (small weight, e.g. 0.2–0.5). This
  *pushes* the projector to preserve the metastasis subspace.

**Leakage controls to add before training:**
- Batch-balanced InfoNCE sampling across the 6 slides (avoid slide-identity shortcut).
- Retrieval metrics: exclude same-spot *spatial neighbours* from the candidate pool,
  or report at region level — otherwise R@1 is inflated by spatial autocorrelation.

**Outputs:** per-fold val loss / best epoch / retrieval (R@1/5/10, MRR),
`all_spots_embeddings.pt`, training curves. Compare A vs B on how well the vision
embedding alone predicts the Stage-1A target on held-out patients.

**Decision gate:** Pick the variant whose held-out vision→leaving-program prediction is
best. Then scale the winner to UNI2-h (1536d).

---

## STAGE 3 — Phase B scoring with confound control (GPU/laptop)

- Train the scoring head to predict the **Stage-1A leaving-program target** from the
  Phase-A vision embedding (vision only — proves ST not needed at inference).
- Apply the within-tumour / composition-residualised μ_HM axis as the **secondary** score.
- **Convergence figure (headline):** correlate per-PT-spot primary score vs μ_HM-resemblance;
  show PT11 high-scorers resemble HM11.
- SpatialGAT k=6 hex smoothing as you planned, but report *both* smoothed and raw.

**Confound audit (mandatory):** correlate the final vision score with the Stage-0
tissue-confound classifier output and with RCTD hepatocyte/liver content. The score must
*not* be explained by these. Report this explicitly.

**>>> STAGE 3 RESULT (2026-06-29) — RAN, see `Outputs/stage3_phase_b/`
(`stage3_phase_b_scoring.py`):**
- Vision-only Phase B head = LOSO RidgeCV (4-fold leave-one-PT-sample-out) predicting the
  Stage-1A target from the FM representation; OOF preds cover all 13,578 PT spots, no patient
  leakage. k=6 Visium hex smoothing applied (both smoothed + raw reported).
- **DECISIVE ABLATION (bridge vs frozen FM) — the contrastive bridge adds NOTHING:**
  held-out leaving rho — bridge-128 raw 0.289 / frozen-FM-1536 raw 0.270; bridge resid 0.086 /
  frozen resid 0.081 (**bridge − frozen = +0.005**, i.e. tied). FM→MLP is *worse* (raw 0.218,
  overfits). → Phase A is not earning its keep; Phase B can be built directly on the frozen FM
  (still ST-only-at-train: ST defines the target, never used at inference → satisfies Dr. Ashiq).
- **CONFOUND-FREE (resid) IS AT THE NOISE FLOOR (~0.08), confirming Stage 2:** the within-tumour
  EMT "leaving" program is NOT recoverable from H&E on held-out patients. The only thing H&E
  predicts is **raw / abundance** (rho ~0.29).
- **CONFOUND AUDIT — the punchline (you get confound-free OR predictive, not both):**
  pred_raw vs tumor_frac = **+0.51** (the raw vision score is largely a malignant-abundance
  detector); pred_resid vs tumor_frac **−0.02**, vs hepatocyte **−0.00**, vs CAF **−0.01**
  (resid is genuinely confound-free by construction — but also near-empty of signal).
- **PT11→HM11 convergence is NEGATIVE** (pred_resid −0.21, pred_raw −0.29): high-predicted-leaving
  PT11 spots *anti*-resemble HM11. Consistent artifact — raw pred ∝ tumor_frac and HM11 resemblance
  anti-corr tumor_frac (−0.50, Stage 1B). Kills the μ_HM "two-axes-agree" headline for good.
- **DECISION:** Drop the tri-modal contrastive bridge from the inference path. Phase B = direct
  frozen-FM supervised regression to the Stage-1A target. **Report honestly:** (a) a vision-only
  H&E predictor of the *confound-free* intra-PT leaving program does NOT generalize across patients
  (noise floor); (b) H&E robustly predicts only malignant abundance (raw rho ~0.3); (c) the bridge
  ≈ frozen FM. Carry both raw (abundance-aware) + resid (confound-free upper bound) scores into
  Stage 4. **Stage 3 COMPLETE → proceed to Stage 4 (validation/figures), tempering claims accordingly.**

---

## STAGE 4 — Validation protocol (define BEFORE trusting any score)

1. **Patient-11 anchor:** do high-scoring PT11 spots resemble HM11 transcriptomically and
   in held-out met signatures?
2. **Held-out signatures:** score correlates with TWIST1/EMT/PDAC-liver-met signatures
   that were kept out of training.
3. **Spatial / pathology:** high-score spots localise to invasive margins / perineural /
   perivascular niches. Get even a small pathologist annotation from Dr. Ashiq.
4. **Negative control:** score should beat and be decorrelated from the tissue-of-origin
   classifier (Stage 0).

**>>> STAGE 4 RESULT (2026-06-29) — RAN, see `Outputs/stage4_validation/`
(`stage4_validation.py`):**
- All 4 pre-registered tests applied to the locked Stage-1A target (leaving raw+resid) and
  Stage-3 vision-only OOF preds (pred raw+resid). **Decisive new EXTERNAL evidence:** the
  source paper's Suppl. Data 2 (MOESM5) supplies INDEPENDENT spotwise GSEA for 29 Fges
  signatures incl. `EMT_signature`, keyed by barcode — joined to **all 13,578 PT spots (100%)**.
- **TEST 2 (the crux) — target is valid EMT, but H&E can't recover it:**
  - (a) construct validity PASS-ish: target_raw vs *independent* paper EMT GSEA pooled **+0.205**
    (resid +0.205); vs Stage-1A held-out EMT +0.414. Our leaving target IS measuring real EMT.
  - (b) deliverable FAIL: vision_raw vs ext EMT **+0.104**, vision_resid vs ext EMT **+0.006**
    (noise). Confound-free EMT is NOT recoverable from H&E across patients. Confirms Stage 2/3.
  - (c) specificity: top vision_raw Fges hits are immune/proliferation-adjacent (Effector/Th1/Th2/
    NK ~0.19–0.21), not a specific EMT readout → H&E tracks region/abundance, not the EMT axis.
- **TEST 1 (PT11 anchor) FAIL:** vision_resid vs HM11 resemblance **−0.217**, raw −0.290; target_resid
  +0.186 only. HM11 anchor is microenvironment, not met biology (consistent w/ Stage 1B/3).
- **TEST 3 (invasive margin) FAIL the hypothesis:** leaving/vision are HIGHER in tumour *interior*
  than at the tumour-stroma margin (raw p(margin>interior)=1.0; resid flat, p~0.86–0.999) — i.e.
  abundance-driven, not invasive-front-localised. (No pathologist annotation yet — request from
  Dr. Ashiq remains the key missing ground-truth check.)
- **TEST 4 (negative control):** resid decorrelation PASS (vs hepatocyte −0.004, vs tumor_frac −0.017).
  **Abundance ceiling NOT beaten:** a trivial tumour-fraction-only LOSO predictor scores rho **+0.478**
  vs vision_raw +0.288 (vision − ceiling = **−0.190**) → the raw vision score is *below* a pure
  abundance detector. Punchline locked.
- **DECISION / final framing:** deliverable = a reproducible, confound-audited vision score whose
  only generalizable signal is malignant abundance; the intra-PT EMT "leaving" program is real in ST
  but below the cross-patient H&E ceiling at 55 µm. **Stage 4 COMPLETE — pipeline validation done;
  remaining external check = pathologist annotation of high-score spots from Dr. Ashiq.**

---

## Execution order

```
Stage 0  (laptop)  → confound diagnostic        ← START HERE, cheap insurance
Stage 1  (laptop)  → define targets (1A primary, 1B secondary)
Stage 2  (Kaggle)  → Phase A variants A & B, leakage controls
Stage 3  (mixed)   → Phase B scoring + confound audit
Stage 4  (mixed)   → validation + figures
```

Do **not** start Phase A (Stage 2) before Stages 0–1 are green. Phase A will *look*
successful regardless (loss converges, R@1 decent) — that's the trap. Stages 0–1 are what
tell you whether you're measuring metastasis or tissue.
