# Methodology-Figure Prompts — PDAC ST-Guided Histopathology Project

Copy-paste prompts for generating the project's methodology diagram with **web-based Claude
(claude.ai)**.

**How to use**
1. Open a new chat at claude.ai.
2. Attach your 3 reference figures (THREADS, MixTIME, the two-stage pretraining figure) as
   style references.
3. Paste ONE of the prompts below (inside the fenced ```` ```text ```` blocks).
4. Ask Claude to render it as an **HTML artifact with inline SVG** so you can preview it live,
   then iterate ("make Phase B blue", "swap that icon") and export to PNG/PDF.

Three variants are provided:
- **Variant 1 — Clean end-to-end method** (architecture only; closest to THREADS/MixTIME look).
- **Variant 2 — Method + rigor / validation** (adds the confound-audit, LOSO, residualization,
  external GSEA validation that make the study rigorous; includes the honest findings).
- **Variant 3 — Complete single-page** (everything above merged into one poster-style figure).

All numbers below come from the actual pipeline — edit freely if a value changes.

---

## Variant 1 — Clean end-to-end method

```text
You are an expert scientific-figure designer (think Nature Methods / BioRender quality).
Produce a SINGLE self-contained HTML file containing one inline <svg> that renders a clean,
horizontal, end-to-end METHODOLOGY diagram for the computational-pathology project described
below. I have attached 3 reference figures (THREADS, MixTIME, a two-stage pretraining figure) —
match that visual language: rounded-rectangle modules, color-coded modality lanes, clear
left-to-right flow arrows, small flat icons, a legend, and lettered panels (a, b, c, d, e).

OUTPUT REQUIREMENTS
- One HTML file, inline SVG, no external assets/CDNs. Canvas ~1700 x 980, white background.
- Vector only (crisp at any zoom). Use a clean sans-serif (system-ui / Inter / Helvetica).
- Color-code by MODALITY and keep it consistent everywhere:
    Histology/H&E = pink/magenta (#E8A6C8 fill, #B05A8A stroke)
    Transcriptomics/genes = green (#9FD4A3 fill, #3F8F4E stroke)
    Cell composition = blue (#A9C7E8 fill, #3F6FA0 stroke)
    Fusion / shared space = lavender/purple (#C9B8E8 fill, #6A4FA0 stroke)
    Scoring/output = warm orange (#F4C28A fill, #C77B2C stroke)
- Add a small legend with a "frozen ❄" (snowflake) vs "trainable 🔥" (flame) key, like the
  references. Mark frozen vs trainable modules with these glyphs.
- Lettered section headers (a–e) in bold, on tinted band backgrounds like the references.
- Keep it uncluttered: this is a clean architecture figure, NOT a results figure.

TITLE
"Spatial-Transcriptomics–Guided Histopathology Model for Metastatic-Propensity Scoring in PDAC"
Subtitle: "Multimodal training from H&E + spatial transcriptomics; histology-only at inference."

DIAGRAM CONTENT (left → right):

(a) COHORT & INPUT  [leftmost column]
- Icon: pancreas + liver. Label: "PDAC cohort — 6 patient samples".
- Two groups: "Primary Tumor (PT): T1, T3, T4, T11" and "Hepatic Metastasis (HM): HM11, HM13".
  Small note: "Patient 11 = matched PT/HM pair".
- Each sample provides PAIRED data per tissue spot:
  a whole-slide H&E image + 10x Visium spatial transcriptomics (hexagonal spot grid).
- Show a small WSI thumbnail with a Visium hex-grid overlay; an arrow "per spot" feeding panel b.

(b) PER-SPOT TRI-MODAL FEATURE EXTRACTION  [three stacked lanes]
- Lane 1 — Histology (pink): "224×224 H&E patch per spot" → "Frozen pathology foundation model
  (UNI2-h, 1536-d; also evaluated: CONCH v1.5, H-Optimus-1)" ❄ → "vision embedding".
- Lane 2 — Transcriptomics (green): "Full-transcriptome counts (~17,900 genes)" → "scVI
  variational autoencoder" 🔥 → "50-d gene latent".
- Lane 3 — Cell composition (blue): "RCTD deconvolution" → "15-d cell-type proportions
  (Tumor Epithelial, Hepatocytes, CAFs, TAMs, T/B cells…)".
- All three lanes converge into panel c.

(c) PHASE A — TriModalBridge: contrastive alignment  [center, lavender band]
- Three small MLP projectors 🔥 (one per modality) → "Shared L2-normalized embedding space
  (256-d)".
- Loss box: "Tri-modal InfoNCE contrastive loss" with three weighted links:
  vision↔gene (w=2.0), vision↔cell (w=2.0), gene↔cell (w=1.0).
- Small caption under it: "6-fold leave-one-sample-out cross-validation".
- Visual: three colored embedding clusters pulled together in a circle (like a CLIP/alignment
  motif).

(d) PHASE B — Metastatic-propensity scoring  [orange band]
- Box 1: "Target from spatial transcriptomics (training-only): intra-tumor 'leaving program'
  — EMT / invasion signature scored over 19 driver genes".
- Flow: "Aligned embedding (128-d)" → "Alignment head (supervised contrastive)" 🔥 →
  "Scoring MLP (regression)" 🔥 → "Spatial graph smoothing (SpatialGAT, k=6 Visium neighbors)".
- Output: "Per-spot metastatic-propensity score [0,1]" → small heatmap-over-H&E thumbnail.

(e) INFERENCE / DEPLOYMENT  [rightmost, set off in its own panel]
- Input: "New patient — H&E whole-slide image ONLY (no spatial transcriptomics needed)".
- Flow: H&E → Frozen FM ❄ → trained projector + scoring head 🔥(now fixed) → per-spot score
  → spatial risk heatmap on the slide.
- Banner tagline: "Spatial transcriptomics guides training; H&E alone deploys."

GLOBAL ARROWS
- Solid arrows for forward data flow; a thin dashed feedback arrow on the contrastive-loss box.
- A faint horizontal "Training (a→d)" vs "Inference (e)" divider so the two regimes read clearly.

Make spacing generous, fonts legible at presentation scale, and the whole thing publication-clean.
After rendering, briefly list which colors/icons map to which modality so I can request tweaks.
```

---

## Variant 2 — Method + rigor / validation controls

```text
You are an expert scientific-figure designer (Nature Methods / BioRender quality). Produce a
SINGLE self-contained HTML file with one inline <svg> rendering a STAGED methodology +
validation diagram for the computational-pathology study below. Style it like the attached
two-stage pretraining reference: horizontal STAGE BANDS stacked top-to-bottom, each band a
tinted background with a bold stage title, rounded-rectangle modules, color-coded modality
lanes, flow arrows, small flat icons, and a legend. This version must communicate not just the
architecture but the methodological RIGOR (confound audit, cross-validation, residualization,
external validation) and the HONEST FINDINGS.

OUTPUT REQUIREMENTS
- One HTML file, inline SVG, no external assets. Canvas ~1700 x 1500, white background.
- Vector only; clean sans-serif. Consistent modality colors:
    Histology/H&E = pink (#E8A6C8 / #B05A8A) · Genes = green (#9FD4A3 / #3F8F4E)
    Cells = blue (#A9C7E8 / #3F6FA0) · Fusion/shared = lavender (#C9B8E8 / #6A4FA0)
    Scoring/output = orange (#F4C28A / #C77B2C)
- Status accents: PASS = green check, FAIL = red cross, WEAK/NULL = grey dash. Use these to tag
  the validation gates honestly (do not hide the negatives).
- Legend: modality colors + "frozen ❄ / trainable 🔥" + PASS/FAIL/WEAK key.
- Bold stage titles on tinted bands (Stage 0 … Stage 4). Keep readable at presentation scale.

TITLE
"A confound-audited, ST-guided histopathology pipeline for PDAC metastatic-propensity scoring"
Subtitle: "Spatial transcriptomics defines and validates the target during training; the
deployed predictor uses H&E only."

STAGE BANDS (top → bottom):

STAGE 0 — Data & confound diagnostic  (grey band)
- Cohort: 6 PDAC samples — Primary Tumor (T1,T3,T4,T11) + Hepatic Metastasis (HM11,HM13);
  patient 11 = matched pair. Per spot: paired H&E patch + 10x Visium transcriptomics.
- QC: keep spots with nFeature ≥ 200 AND nCount ≥ 400 → 18,859 spots.
- Confound test box: "Liver-vs-pancreas confound". Hepatocyte fraction HM 0.14 vs PT 0.002;
  corr(HM-direction, hepatocyte) = 0.40 over all spots → 0.07 within tumor-only spots
  (RCTD Tumor-Epithelial restriction removes liver admixture).
- Key finding (red-flag callout): "No patient-generalizable tumor-intrinsic HM-vs-PT signature
  (tumor-only scVI LOSO predicts held-out HM at 0%)". DECISION: demote cohort Δ = μ_HM − μ_PT;
  use intra-PT target instead.

STAGE 1 — Target definition from ST  (green band)
- Inputs: scVI gene latent (50-d) from full-transcriptome counts (~17,900 genes);
  RCTD cell proportions (15-d).
- Target: "Intra-tumor 'leaving program' (EMT / invasion)". 13,578 PT spots; CP10k→log1p;
  Seurat AddModuleScore over 19 core genes (12 EMT TFs/invasion + 7 ECM).
- GATE 1 — spatial coherence: hex Moran's I 0.37–0.61, p=0.001  [PASS].
- GATE 2 — held-out generalization: corr(core, held-out validator genes) = 0.46  [PASS].
  (Validators VIM/SNAI2/PRRX1/MMP9/MMP14/POSTN/SPARC scored but never used to build the score.)
- Confound control: core corr 0.61 with RCTD tumor fraction → RESIDUALIZE on tumor fraction →
  two targets: leaving_score (raw, abundance-aware) + leaving_score_resid (confound-free,
  headline).

STAGE 2 — Phase A: tri-modal contrastive bridge  (lavender band)
- Three frozen/encoded modalities → MLP projectors 🔥 → shared L2-normalized space (128-d).
  Tri-modal InfoNCE: vision↔gene 2.0, vision↔cell 2.0, gene↔cell 1.0.
- Backbones evaluated: UNI2-h (selected), CONCH v1.5, CONCH v1, H-Optimus-1.
- 6-fold leave-one-sample-out CV. Leakage control: "neighbor-excluded retrieval" (exclude
  spatial neighbors from candidate pool).
- Honest finding callout: "Bridge does NOT generalize across patients — val InfoNCE ≈ chance
  (ln 512 = 6.24); neighbor-excluded R@1 ≈ 0.002. Earlier high R@1 was spatial-autocorrelation
  inflation."  [WEAK/NULL]

STAGE 3 — Phase B: scoring + decisive ablation  (orange band)
- 4-fold LOSO RidgeCV: predict leaving target from embeddings → per-spot score → SpatialGAT
  (k=6 Visium hex) smoothing → score [0,1] + heatmap on H&E.
- ABLATION callout: "Contrastive bridge ≈ frozen FM (raw ρ 0.289 vs 0.270; resid 0.086 vs
  0.081) → drop the bridge; deploy DIRECT frozen-FM regression (still ST-only-at-train)."
- Confound audit: pred_raw vs tumor_fraction = +0.51 (raw score = malignant-abundance detector);
  pred_resid decorrelated from tumor fraction / hepatocyte / CAF.

STAGE 4 — External validation & deployment  (blue band)
- External validator: independent spotwise GSEA of 29 Fges signatures (incl. EMT_signature)
  from the source paper, joined to all 13,578 PT spots at 100%.
- TEST 2 (construct validity): target vs independent EMT GSEA = +0.205  [PASS — target is real
  EMT]; but vision_resid vs external EMT ≈ 0.006  [FAIL — confound-free EMT not recoverable
  from H&E on unseen patients].
- TEST 4 (negative control): residual score decorrelated from confounders [PASS]; abundance
  ceiling NOT beaten (tumor-fraction-only LOSO ρ +0.478 > vision_raw +0.288).
- Inference panel: "New patient H&E only → frozen FM ❄ → fixed scoring head → spatial risk
  heatmap. No spatial transcriptomics needed."
- Bottom banner (honest headline): "The intra-tumor EMT 'leaving program' is a genuine,
  externally-validated ST signal, but from H&E alone the model recovers only malignant
  abundance on unseen patients at 55 µm resolution."

Use solid arrows for data flow, dashed for losses/feedback. Make the PASS/FAIL/WEAK tags
prominent so the figure reads as a rigorous, self-critical pipeline. After rendering, list the
color/icon/status legend so I can request edits.
```

---

## Variant 3 — Complete single-page (architecture + rigor + findings)

```text
You are an expert scientific-figure designer (Nature Methods / BioRender quality). Produce a
SINGLE self-contained HTML file with one inline <svg> rendering a COMPLETE one-page methodology
POSTER for the study below. Combine (1) the end-to-end multimodal architecture and (2) the
staged rigor/validation narrative into ONE coherent figure. Match the attached references
(THREADS = clean horizontal multimodal lanes; MixTIME = multi-expert fusion + leaderboard;
two-stage pretraining figure = tinted stage bands). This is a dense but well-organized poster.

OUTPUT REQUIREMENTS
- One HTML file, inline SVG, no external assets. Large canvas ~1800 x 1700, white background,
  generous margins. Vector only; clean sans-serif; legible at A3/poster scale.
- Consistent modality colors throughout:
    Histology/H&E = pink (#E8A6C8 / #B05A8A) · Genes = green (#9FD4A3 / #3F8F4E)
    Cells = blue (#A9C7E8 / #3F6FA0) · Fusion/shared = lavender (#C9B8E8 / #6A4FA0)
    Scoring/output = orange (#F4C28A / #C77B2C)
- One master legend (top-right): modality colors + frozen ❄ / trainable 🔥 + PASS/FAIL/WEAK.
- Lettered panels with bold titles on tinted bands. Solid arrows = data flow, dashed = loss.
- A faint vertical or horizontal divider separating "TRAINING (multimodal)" from "INFERENCE
  (H&E only)".

LAYOUT (two zones):

=== TOP ZONE: ARCHITECTURE (horizontal, left → right) ===
(a) Cohort & input: 6 PDAC samples — PT (T1,T3,T4,T11) + HM (HM11,HM13), patient 11 matched.
    Per spot = paired H&E patch + 10x Visium transcriptomics (hex grid). QC → 18,859 spots.
(b) Tri-modal per-spot feature extraction — three lanes:
    • Histology: 224×224 H&E patch → frozen pathology FM (UNI2-h 1536-d; CONCH v1.5, H-Optimus-1
      also tested) ❄ → vision embedding.
    • Transcriptomics: ~17,900-gene counts → scVI VAE 🔥 → 50-d latent.
    • Cell composition: RCTD deconvolution → 15-d cell-type proportions.
(c) Phase A — TriModalBridge: 3 MLP projectors 🔥 → shared L2-normalized space (128–256-d);
    tri-modal InfoNCE (vision↔gene 2.0, vision↔cell 2.0, gene↔cell 1.0); 6-fold LOSO.
(d) Phase B — scoring: aligned embedding → alignment head (SupCon) 🔥 → scoring MLP 🔥 →
    SpatialGAT (k=6) smoothing → per-spot metastatic-propensity score [0,1] + H&E heatmap.
(e) Inference: NEW patient H&E only → frozen FM ❄ → fixed scoring head → spatial risk heatmap.
    No ST needed. Tagline: "ST guides training; H&E alone deploys."

=== BOTTOM ZONE: RIGOR & VALIDATION (staged bands, left → right or as a row of cards) ===
Stage 0 — Confound diagnostic: hepatocyte frac HM 0.14 vs PT 0.002; corr w/ HM-direction 0.40
    all-spots → 0.07 tumor-only. "No patient-generalizable tumor-intrinsic HM signature
    (tumor-only LOSO predicts held-out HM at 0%)" → demote cohort μ_HV − μ_PT; use intra-PT
    target.  [red-flag]
Stage 1 — Target = intra-tumor 'leaving program' (EMT/invasion, 19 genes, 13,578 PT spots).
    GATE 1 Moran's I 0.37–0.61 p=0.001 [PASS]; GATE 2 held-out validator corr 0.46 [PASS];
    confound corr 0.61 w/ tumor fraction → residualize → raw + confound-free (resid) targets.
Stage 2 — Phase A honest result: bridge ≈ chance cross-patient (val InfoNCE ≈ ln512 = 6.24;
    neighbor-excluded R@1 ≈ 0.002 after leakage control). [WEAK/NULL]
Stage 3 — Ablation: bridge ≈ frozen FM (raw ρ 0.289 vs 0.270) → drop bridge, use direct
    frozen-FM regression. Confound audit: pred_raw vs tumor-frac +0.51 (abundance detector);
    pred_resid decorrelated.
Stage 4 — External validation: independent 29-Fges spotwise GSEA (incl. EMT) joined 100%.
    target vs ext EMT +0.205 [PASS, target is real EMT]; vision_resid vs ext EMT ≈ 0.006 [FAIL];
    abundance ceiling not beaten (tumor-frac LOSO ρ +0.478 > vision_raw +0.288).

BOTTOM BANNER (honest headline, full width):
"The intra-tumor EMT 'leaving program' is a genuine, externally-validated spatial-transcriptomic
signal — but from H&E alone the model recovers only malignant abundance on unseen patients at
55 µm. The bridge adds nothing over the frozen foundation model; the deliverable is a
reproducible, confound-audited H&E score with an explicit abundance ceiling."

Keep the two zones visually distinct but connected (e.g., the target in Stage 1 feeds Phase B in
the top zone — draw a light linking arrow). Prioritize clarity over density; if it gets crowded,
shrink module text rather than dropping panels. After rendering, output the legend mapping so I
can request tweaks.
```

---

## Tips for iterating in web Claude
- If the artifact is too cramped, ask: "increase the canvas height and add 40px padding between
  panels."
- To match the references more closely: "add small flat icons (microscope, DNA helix, cell
  cluster, brain/network) to each module like the THREADS figure."
- To export: ask for "a download button that exports the SVG as PNG at 3x resolution," or open
  the artifact and screenshot/print-to-PDF.
- Keep the ❄ (frozen) vs 🔥 (trainable) glyphs accurate: the **pathology foundation model is
  frozen**; scVI, the bridge projectors, and the scoring heads are **trained**.
- Core message to preserve in every variant: **spatial transcriptomics guides training; the
  deployed predictor uses H&E only.**
