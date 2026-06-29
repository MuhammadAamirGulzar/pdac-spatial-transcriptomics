# Presentation figure set — PDAC Spatial-Transcriptomics + Foundation-Model project

Audience: clinical / biology domain experts (non-technical on ML).
Every spatial result is shown **next to its real H&E tissue** in the same orientation.
Regenerate with: `"C:/Users/datai/anaconda3/envs/tcga/python.exe" presentation/make_figures.py`
(tissue thumbnails are cached in `presentation/_tissue_cache/`).

The seven figures tell one honest story, in order. Suggested talking points below each.

---

## fig1_cohort_overview.png — *The question*
6 tissues (4 primary pancreatic tumours, 2 liver metastases; 18,859 spots), each spot
carrying H&E + gene expression + cell composition.
**Say:** "We want to flag metastasis-prone tumour regions from a routine H&E slide alone.
Spatial transcriptomics is the teacher during training; at inference, only H&E. Note only
patient 11 has *both* a primary (T11) and its matched liver met (HM11)."

## fig2_stage0_confound.png — *Why we don't just compare primary vs metastasis*
(a) Liver-met spots are ~15% liver cells (hepatocytes) vs ~0% in primary — a tissue-of-origin
confound. (b) With only 2 met patients, one cannot predict the other (0% recovered) while
primaries generalise (93–100%). (c) The naive "metastasis axis" is mostly liver content, and
restricting to tumour-rich spots removes it (0.40 → 0.07).
**Say:** "Comparing met-tissue to primary-tissue would measure *liver*, not *metastasis*. So we
instead define an **intra-primary 'leaving program'** — the EMT/invasion state inside the
primary tumour itself."

## fig3_stage1a_leaving_maps.png — *The leaving program exists and is spatially real*
Each primary tumour: H&E (left) vs leaving-program score (right). High-score regions form
coherent zones (Moran's I 0.37–0.61, all p<0.001), consistent with invasive fronts.
**Say:** "This is a real, organised biological pattern in the tissue — not noise. It is built
from 19 canonical EMT/invasion genes."

## fig4_stage1a_program_biology.png — *What the program is, and that it's valid*
(a) The 19 driver genes grouped (EMT transcription factors, mesenchymal markers, TGF-β,
proteases, ECM). (b) Computed independently, our target matches the *source paper's own* EMT
(+0.21), CAF (+0.31) and matrix (+0.29) signatures — and not unrelated programs.
**Say:** "These are textbook dissemination genes (SNAI1, ZEB1, MMP1…), and an outside dataset
confirms our score is genuine EMT biology, not an artefact."

## fig5_stages23_ceiling.png — *What H&E can and cannot recover (on unseen patients)*
(a) H&E predicts the raw score moderately (ρ +0.29); (b) once the tumour-amount confound is
removed, the confound-free EMT signal is at the noise floor (ρ +0.09). (c) A trivial
"how-much-tumour" model (+0.48) **beats** the H&E model. (d) The transcriptomic bridge does
not beat the foundation model alone.
**Say:** "Honestly: from H&E the model reliably reads tumour *abundance*, not the fine
metastatic state — and injecting transcriptomics didn't add real signal here."

## fig6_stage3_spatial_prediction.png — *The ceiling, seen on the tissue*
For 2 tumours: H&E | actual leaving score | predicted-from-H&E. The prediction captures the
broad high/low zonation but misses finer structure.
**Say:** "Visually, H&E recovers the coarse pattern (driven by tumour density) but smooths
away the detail — the picture behind the previous figure's numbers."

## fig7_stage4_scorecard.png — *Verdict & what's next*
Six pre-registered tests with status. Take-home: the leaving program is a genuine, validated
signal; today's H&E model reads only tumour abundance on unseen patients. Gap-closers:
pathologist annotation of invasive fronts, higher-resolution platform (Visium HD / Xenium),
more matched primary–metastasis patients.
**Say:** "Rigorous, confound-audited result. The biology is real; the H&E read-out is
abundance-limited at current resolution — and here is exactly what would push it further."

---

### Honest one-line summary for the talk
> The intra-primary EMT "leaving program" is a real, externally-validated transcriptomic
> signal; from a routine H&E slide our model currently recovers only how much tumour is
> present, not the confound-free metastatic state, on unseen patients.
