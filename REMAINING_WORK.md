# Remaining work

Written 2026-08-29. This is a work order, not a summary. It is meant to be executed one task at a
time on the workstation that holds the full dataset, by a person or an agent with no memory of how
the project got here. Read section 0 first, then do the tasks in the order given.

Companion documents: [README.md](README.md) for what the project is, [HANDOFF.md](HANDOFF.md) for
the engineering record and every decision behind the current results.

---

## 0. Before starting

### Where things stand

Stages 0 through 9 are complete and committed. The site-specificity manuscript is drafted and
current as of stage 9 (`docs/reports/build_research_paper.py` builds it). Nothing in this document
changes an existing result. Every task here either adds a missing control, fills a gap the
manuscript currently marks as awaiting, or belongs to a second paper.

Two papers are planned. Keep their material separate. Mixing them is the main way this goes wrong.

| | Paper A | Paper B |
|---|---|---|
| Subject | Site-specific metastatic programs in PDAC | The limits of predicting spatial transcriptomics from H&E |
| Repository | this one | this one plus `ecomap-spatial-intelligence-pipeline` |
| Status | drafted, three gaps | not started, experiments defined below |
| Tasks | A1 to A4 | B1 to B6 |

### Machine and environment

Project root on the workstation: `D:\Aamir Gulzar\KSA_project3\ST_Project`

Python is the `stproj` conda environment, a **pure pip** scientific stack. Do not install
numpy, scipy or scikit-learn from conda alongside pip's torch: two OpenMP runtimes make
`sklearn.decomposition.PCA` abort the process silently at exit 127 with no traceback. `pandas` is
pinned below 3, because pandas 3's copy-on-write and string dtype changes can move numbers that
validated results depend on.

```
numpy>=2   pandas<3   scipy   scikit-learn   matplotlib   seaborn   umap-learn   openpyxl
torch      # CUDA build; CPU is enough for everything in this document except B6
```

### Data that must be present

| Path | Needed by |
|---|---|
| `dataset/full_cohort/scVI_counts/` | A1, A2 |
| `dataset/full_cohort/rctd/` | A1, A2, B1 |
| `dataset/full_cohort/coords/` | A2 |
| `dataset/full_cohort/cohort_metadata.csv` | A3 |
| `dataset/Feature Extraction Embeddings/` (UNI2-h) | B6 |
| EcoMap dataset from Google Drive | B1 to B5 |

### Runtimes worth knowing

`04_Models/stage9_cosine_nulls.py` takes roughly 20 minutes on a laptop. The cost is the patient
bootstrap, which rebuilds a DataFrame with `pd.concat` on each of 10,000 draws for each of four
contrasts. If it needs to be re-run often, pre-split the sections into NumPy arrays indexed by
patient and drop the DataFrame from the inner loop; the result is unchanged and it becomes seconds.
Nothing else in this document is slower than a few minutes on CPU except B6, which wants the GPU.

### Rules for every task

1. Write results to a new `Outputs/<task>/` directory. Never overwrite an existing one.
2. Every script takes the patient, not the section and not the spot, as the unit of analysis
   wherever a test is being run. This project has been bitten by both.
3. Report effect sizes and intervals alongside p values. With four to five sections per group the
   p value is often at its floor and carries no information.
4. When a script finishes, record the numbers in `HANDOFF.md` under a new numbered section, in the
   same style as sections 10 to 13. That file is the project's memory.
5. Commit with a message that says what changed and why. **Do not add Co-Authored-By trailers or
   any other tool attribution.**

---

## Paper A. Finishing the site-specificity manuscript

Three gaps. A1 and A2 are the ones a reviewer will raise; A3 may not be solvable with the data that
exists.

### A1. B cell marker score computed directly from counts

**Why.** The central claim of the discovery cohort is that B cell content is higher in lymph node
deposits than in liver deposits from the same patient (`d = +2.01` with the patient as the unit,
`q = 0.065`, higher in 4 of 4 paired patients). That number comes from `rctd::B cells`, a
deconvolution output. The deconvolution is the paper's own `rctd_fullfinal` assay and is sound, but
the claim currently stands on one derived quantity. An orthogonal readout from raw expression
removes that dependency entirely.

**Method.** For each of the 30 sections, take the spots passing the same tumour purity gate stage 6
used (`Tumor Epithelial cells >= 0.50`), compute CP10k and log1p normalised expression, then a mean
z-score across the B cell markers `MS4A1`, `CD79A`, `CD19`, `IGHM`, `CD79B`, `BANK1`. Aggregate to
one value per section, then to one value per patient and site. Repeat the stage 6 tests: Cohen's d,
Mann Whitney at patient level with Benjamini Hochberg across however many features are tested, and
the exact sign test on the four paired patients.

Add two negative controls from the same counts, scored identically: a T cell panel (`CD3D`, `CD3E`,
`CD2`, `TRAC`) and a myeloid panel (`CD68`, `CD14`, `LYZ`, `AIF1`). Neither separated liver from
node deposits in the RCTD analysis. If the marker version reproduces that pattern too, the agreement
between the two readouts is much stronger evidence than the B cell result alone.

**Inputs.** `dataset/full_cohort/scVI_counts/`, `dataset/full_cohort/rctd/`,
`Outputs/stage6_site_program/pseudobulk_sections.csv` for the section and patient index.

**Output.** `Outputs/stage10_bcell_markers/` containing `marker_scores_sections.csv`,
`marker_scores_patients.csv`, `metrics.json`, `summary.txt`.

**Acceptance.** The direction agrees with RCTD in at least 3 of the 4 paired patients and the effect
size has the same sign. If it does not, that is a real finding and must be reported, not buried:
it would mean the RCTD B cell signal is a deconvolution artefact and the manuscript's main claim
has to be withdrawn. Either outcome is publishable; only silence is not.

**Feeds.** Manuscript section 4.2, and a new supplementary table.

### A2. Spatial organisation of B cells

**Why.** This is the gap that most weakens the paper. Stages 5, 6 and 7 all average spots into
section means, so a reviewer can fairly ask why spatial transcriptomics was needed rather than bulk
RNA sequencing of the same deposits. Nothing in the manuscript currently answers that. The answer is
available in the data: if node B cells sit in organised aggregates resembling tertiary lymphoid
structures while liver B cells are sparse and diffuse, that is a spatial fact no bulk assay could
produce.

**Method.** Per section, on all QC-passing spots rather than only tumour-dominated ones, because
lymphoid aggregates sit at the deposit margin:

1. Moran's I on the B cell fraction over the Visium hexagonal neighbourhood (k = 6), with a
   permutation p value from at least 999 shuffles. `hex_morans_I` in
   `04_Models/stage1b_pt11_hm11_anchor.py` already implements this correctly; reuse it rather than
   writing a second version.
2. A hot spot count: contiguous runs of spots above the section's 90th percentile of B cell
   fraction, with the size of the largest run recorded.
3. Compare Moran's I and the hot spot statistics between liver and node deposits at patient level,
   the same way A1 does.

**Inputs.** `dataset/full_cohort/rctd/`, `dataset/full_cohort/coords/`.

**Output.** `Outputs/stage11_bcell_spatial/` with `morans_by_section.csv`, `hotspots.csv`,
`metrics.json`, and one figure per paired patient showing the B cell fraction map for the liver and
node section side by side.

**Acceptance.** Node sections show higher Moran's I than liver sections in most paired patients.
Report the value whichever way it comes out. A null result here is worth stating plainly and simply
means the immune difference is compositional rather than architectural.

**Feeds.** A new manuscript section 4.2.1 and a new figure. This is the highest value task in the
document.

### A3. Clinical and treatment table

**Why.** The discovery cohort includes patients who received neoadjuvant chemotherapy and the
manuscript does not characterise this. A liver versus node immune difference is exactly what
differential treatment exposure could produce. The replication cohort being treatment naive is the
current defence, and it is a good one, but it is indirect.

**Method.** Read `dataset/full_cohort/cohort_metadata.csv` and tabulate, per patient: treatment
status, stage, surgery date, and any interval to collection. HANDOFF records that these fields are
populated for only 6 of 13 patients, all at pM1, with no follow-up and no survival endpoint. If that
is still the case, **do not attempt an association analysis on 6 patients**. Produce the table,
state the coverage, and check only whether the paired patients driving the B cell result differ
systematically in treatment from the rest.

**Blocked on.** The full clinical table has been requested from the lab. If it arrives, revisit.

**Output.** `Outputs/stage12_clinical/clinical_table.csv` and a paragraph for manuscript section 3.1.

**Acceptance.** A table exists and the manuscript states the coverage honestly. This task most
likely ends as a documented limitation rather than an analysis, and that is an acceptable outcome.

### A4. Regenerate the manuscript

After A1 to A3, run:

```bash
python docs/reports/build_research_paper.py
```

It reads every number from `Outputs/*/metrics.json`, so the text updates itself. Replace the
placeholders marked `[AWAITING]` in sections 4.2.1 and 3.1 with the real findings, and delete the
`awaiting` note boxes once the corresponding task is done.

---

## Paper B. The limits of H&E to omics prediction

The second repository, `ecomap-spatial-intelligence-pipeline`, trains a multimodal teacher on
morphology, gene expression and cell composition, then distils it into a student that sees only
morphology. It reports about 88 percent teacher and 79 percent student accuracy on a five class
ecotype task.

Those numbers cannot be used as they stand, for four reasons found by reading the code:

1. **Folds are not grouped by patient.** Both teacher and student use
   `StratifiedKFold(shuffle=True)` over spots. `groups=` appears nowhere in the repository.
   `patient_id` is loaded only for plotting. Adjacent Visium spots therefore appear in training and
   test at once.
2. **The ensemble teacher saw every spot.** `teacher_logits_all` is computed over the whole dataset
   from a teacher weight-averaged across all five folds, then indexed by `val_idx`. The student's
   validation soft targets come from a model that trained on those exact spots.
3. **The labels are derived from an input modality.** Ecotypes are ISCHIA plus RCTD clusters, and
   RCTD composition is fed to the teacher.
4. **Distillation is never isolated.** The loss is `0.7 * CE(hard) + 0.2 * KL(soft) + 0.1 * feature`.
   At that weighting the student is mostly a plain supervised classifier, and no run sets the soft
   term to zero.

This is the paper. Each fault maps onto a control the H&E to omics literature routinely omits, and
one pipeline exhibits three of them at once. Run it correctly, report the difference, and the
difference is the contribution.

> The EcoMap cohort labelled `GEO 5-Patient` appears to be the same six slides and five patients as
> this project's original cohort: the fused dimension of 1177 matches UNI 1024 plus scVI 128 plus
> RCTD 25 exactly, and T11 and HM11 belong to the same patient. **Confirm this by intersecting the
> barcode lists before writing anything that depends on it.** This is the first thing to check when
> the Drive data is mounted.

### B1. Ecotype accuracy against tumour fraction

**Do this one first.** It is cheap, needs no retraining, and it decides what Paper B argues.

**Why.** This project's established finding is that H&E recovers how much tumour is present and
little beyond it. Ecotypes are largely composition classes. If the student's per-spot correctness
is predicted by tumour fraction, then EcoMap's 79 percent and this project's abundance ceiling are
the same result in two formulations, and Paper B has a single unified claim.

**Method.** Take the existing `predictions_all_spots.csv` from an EcoMap student run. Join to
`rctd::Tumor Epithelial cells` per barcode. Then: correctness rate by tumour fraction decile; a
logistic regression of correctness on tumour fraction with the area under the curve; and per class
mean tumour fraction, to see whether the classes the student gets right are simply the ones with
distinctive purity.

**Output.** `Outputs/paperB_b1_abundance/` with `accuracy_by_decile.csv`, `metrics.json`, one figure.

**Decides.** If tumour fraction explains most of the variance in correctness, Paper B's thesis is
"one ceiling, found by four architectures". If it does not, the thesis has to be reconsidered before
spending effort on B2 to B5.

### B2. The headline experiment: patient-grouped folds

**Method.** Change `StratifiedKFold` to `StratifiedGroupKFold` with `groups=patient_id`, in both
`pipeline/train_mlp.py` and `pipeline/train_student_model_unified.py`. Rebuild the ensemble teacher
using only folds that excluded the held-out patient, so fix fault 2 at the same time. Run teacher
and student both ways, same seed, same architecture, same epochs.

**Output.** `Outputs/paperB_b2_grouped_cv/` with a table of accuracy under spot-random and
patient-grouped folds, for teacher and student.

**This is the paper's headline number.** Report the pair, not just the corrected value. The size of
the gap is the finding.

### B3. Floor and ceiling baselines

Five classes does not mean chance is 20 percent unless the classes are balanced. Report, under
patient-grouped folds: the majority class rate; a composition-only classifier using RCTD features
alone; and a tumour-fraction-only classifier. The last two bracket how much of any accuracy is
composition rather than morphology.

**Output.** `Outputs/paperB_b3_baselines/`.

### B4. Teacher without the cell modality

Retrain the teacher on morphology and gene expression only, with patient-grouped folds. The gap
against the full teacher quantifies how much of its accuracy came from being fed the modality its
labels were derived from.

**Output.** `Outputs/paperB_b4_circularity/`.

### B5. Student without distillation

Set the soft loss weight to zero and retrain the student on hard labels alone, patient-grouped.
If accuracy is unchanged, distillation contributes nothing and the repository's central claim does
not hold. Report either way.

**Output.** `Outputs/paperB_b5_distillation/`.

### B6. Teacher and student on this project's cohort

**Why.** This project now has UNI2-h embeddings for 11 sections across 7 patients: T1, T3, T4, T11,
HM11, HM13, LNM6, LNM7, LNM10, NP10, NP11. That is better resourced than EcoMap's 6 sections and 5
patients, and it includes lymph node and normal pancreas sections, so there is real site diversity
rather than only tumour against liver. Seven patients makes leave-one-patient-out viable.

**Two design rules.** Do not use ISCHIA plus RCTD ecotypes as the target here, or the circularity of
fault 3 is imported along with the method. Use either the leaving programme score from
`Outputs/stage1a_full_cohort_residcaf/leaving_program_scores.csv`, which is continuous and already
externally validated, or ecotypes derived from expression clustering alone with no cell input. And
compare against the frozen foundation model baseline from stage 3, not against chance, since stage 3
already established that the contrastive bridge does not beat it.

**Expected outcome.** The same abundance ceiling. Distillation is a genuinely different transfer
mechanism from the contrastive bridge, so this is a fair new test, but the value is in adding a
fourth architecture that fails the same way, not in expecting it to succeed. Frame it that way from
the start.

**Output.** `Outputs/paperB_b6_pdac_distillation/`.

---

## Things not to do

- **Do not put EcoMap's 88 or 79 percent into a talk, poster or thesis without the patient-grouped
  figure beside it.** Once an uncorrected number is in circulation it is hard to withdraw.
- **Do not merge the two repositories.** They share a cohort, not a codebase, and this repository's
  stage numbering is referenced throughout the manuscript.
- **Do not run further RCTD variants.** The `rctdold`, `rctdpaper` and `rctdpaper_residcaf` sweep is
  complete and the conclusion is stable across all three. That question is closed.
- **Do not expand the H&E arm by cropping the remaining whole-slide images.** Of 30 sections, 11
  have a ready cropped WSI, 10 need manual QuPath work and 9 are missing. The automatic capture-area
  matcher validated at 3 of 21 and is unusable. The manual effort buys two more primaries, T2 and
  T9, against a target already measured at `rho = -0.043`. HEST's 156 sections serve the imaging
  generalisation claim better.
- **Do not re-run stage 9 expecting a different answer.** The discovery cohort's inability to
  resolve the global alignment is a property of having 9 liver and 5 node sections, not of the
  method.

## Waiting on other people

| Item | Asked of | Blocks |
|---|---|---|
| Full clinical table with treatment status and follow-up | the lab | A3, and the treatment confound in Paper A |
| Whole-slide images for slides 352, 308 and 57 | the lab | nothing currently; would revive the H&E arm if the plan changes |
| Confirmation that EcoMap's GEO cohort is this project's five patients | check the barcodes directly | the framing of Paper B |
