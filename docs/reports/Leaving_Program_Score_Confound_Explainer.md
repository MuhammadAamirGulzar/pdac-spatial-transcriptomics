# Leaving Program, Leaving Score & Confounds — Explainer

*A plain-language walkthrough of the Stage 1A target, grounded in the real
IU_PDA primary-tumour samples. Use this to answer "what is the leaving program /
score" and "what do you mean by confound" in the meeting.*

Source of every number below: `Outputs/stage1a_leaving_program/` (`summary.txt`,
`metrics.json`, `leaving_program_scores.csv`, `heatmap_*.png`).

---

## 0. The one-page mental model

| Term | What it is | Analogy |
|------|-----------|---------|
| **Leaving *program*** | The *biology* — a fixed list of 19 genes that tumour cells switch on when they start to invade / prepare to metastasise (EMT + ECM remodelling). | The *concept* of "fitness". |
| **Leaving *score*** | The *measurement* — one number per ST spot saying how strongly that spot expresses the program. | The *number* on a fitness test. |
| **Confound** | A hidden third factor that fakes the signal, so the score measures the wrong thing until you correct for it. | Scoring high on the fitness test *just because you were heavier on the scale*, not fitter. |

The whole point of Stage 1A is to go from the raw score (contaminated by
confounds) to a **confound-free target** (`leaving_score_resid`) that the H&E /
foundation model is then asked to predict in Phase B.

---

## 1. The leaving *program* — the biology

Metastasis is not one gene; it's a **coordinated program** a carcinoma cell runs
to detach from the primary tumour, remodel its surroundings, and move. We encode
that program as 19 genes in two biological modules
(`stage1a_leaving_program.py:68-72`):

### Module 1 — EMT core (the cell changes its identity)
Epithelial-to-mesenchymal transition: an epithelial (glued-in-place) tumour cell
loosens its junctions and takes on a migratory, mesenchymal character.

| Gene | Biological role in "leaving" |
|------|------------------------------|
| `SNAI1`, `ZEB1`, `ZEB2` | Master EMT transcription factors — they *repress E-cadherin* and flip the cell to a migratory state. |
| `CDH2` (N-cadherin) | The "cadherin switch" — replaces epithelial E-cadherin; hallmark of a cell that has left its epithelial nest. |
| `S100A4` | Motility / metastasis marker (a.k.a. metastasin). |
| `TGFB1`, `TGFBR1` | TGF-β signalling — the upstream driver that *induces* EMT. |
| `MMP1`, `LOXL2`, `ITGB6`, `LAMC2`, `SERPINE1` | Invasion machinery — degrade/cross-link matrix, engage integrins, cut a path out. |

### Module 2 — ECM panel (the cell rebuilds its surroundings)
Extracellular-matrix remodelling — the desmoplastic, fibrotic front that PDAC
invades through.

`COL1A1`, `COL3A1`, `COL5A1`, `FN1`, `LOX`, `TNC`, `TIMP1` — collagens,
fibronectin, tenascin, cross-linkers. High where the tumour is actively
remodelling the stroma at an invasive edge.

### Why *this* signature and not the metastasis samples themselves
The obvious idea — "compare the liver-metastasis (HM) slides against the
primary-tumour (PT) slides and call the difference metastatic signal" — is
**broken by a confound** (see §4). So the program is defined **entirely inside
the 4 PT slides**: it asks *"which tumour cells, still in the pancreas, are
already running the leaving program?"* No liver tissue ever enters the
definition.

### Held-out biology (used only to validate, never to build the score)
7 independent EMT/invasion genes — `VIM`, `SNAI2`, `PRRX1`, `MMP9`, `MMP14`,
`POSTN`, `SPARC` — are scored the same way but kept out of the core. If the core
is real biology and not an artifact, it should *predict* these. It does
(pooled correlation **+0.46**, §3).

---

## 2. The leaving *score* — turning biology into a number

For every one of the **13,578 PT spots**, we compute one number. Method
(`stage1a_leaving_program.py`):

1. **Normalise** each spot's counts (CP10k → log1p) so a spot doesn't look
   "high" just because it was sequenced more deeply.
2. **AddModuleScore** (Seurat-style): `mean(program genes) − mean(matched
   control genes)`. Subtracting expression-binned control genes removes the
   generic "this spot is transcriptionally busy" baseline, so the score reflects
   *program-specific* expression.
3. **z-score within each slide** → `leaving_score`. Positive = this spot runs the
   leaving program more than the typical spot on its own slide.

### Does the score capture real, organised biology? Two gates it passed:

**Gate 1 — spatial coherence (Moran's I).** If the score were noise, high spots
would scatter randomly. Instead they **cluster** — high exactly where you'd
expect biologically, at invasive fronts. Every slide is positive and
significant:

| Sample | Moran's I (core) | p |
|--------|------------------|-----|
| IU_PDA_T1 | 0.389 | 0.001 |
| IU_PDA_T3 | **0.613** | 0.001 |
| IU_PDA_T4 | 0.374 | 0.001 |
| IU_PDA_T11 | 0.369 | 0.001 |

See `heatmap_IU_PDA_T3.png` — the high-score region is a coherent front, not
salt-and-pepper.

**Gate 2 — generalisation.** The core score predicts the 7 held-out EMT genes it
never saw: **corr = +0.46** pooled (and the unsupervised PCA axis on the same
genes agrees at **+0.78**). So the score is measuring a genuine, reproducible EMT
axis, not 19 arbitrary genes.

---

## 3. The confounds — what fakes the signal

A **confound** is a third variable driving an apparent relationship, so your
measurement is really measuring something else. Two of them matter here.

### Confound A — liver / tissue-of-residence (the fatal one, Stage 0)
The original `mu_HM − mu_PT` axis mostly separated **liver vs pancreas tissue**,
not metastatic vs non-metastatic tumour. Metastasis samples sit in liver
(hepatocytes, liver stroma); primary sits in pancreas. Tell-tale sign it was an
artifact: it **did not generalise across the 2 HM patients** — real biology
would; a tissue-of-origin artifact wouldn't.
→ **Fix:** define the program *only inside PT slides*. This confound becomes
structurally impossible.

### Confound B — tumour abundance (the one the residual fixes)
A spot with **more tumour cells** scores higher on the leaving program **just
because there is more tumour in the spot** — not because those cells are more
invasive. The raw score is therefore partly an *abundance* readout.

We can see it directly. Correlation of raw `leaving_score` with RCTD tumour
fraction:

| Sample | corr(leaving_score, tumor_frac) |
|--------|-------------------------------|
| IU_PDA_T1 | 0.519 |
| IU_PDA_T3 | 0.761 |
| IU_PDA_T4 | 0.582 |
| IU_PDA_T11 | 0.597 |
| **Pooled** | **0.609** |

**0.61** is a lot of the signal. If we stopped here, a model could "predict the
leaving score" just by counting tumour — and learn nothing about invasion.

**Sanity checks that it is NOT a stromal/CAF or liver readout instead:**
- corr(core, CAF fraction) = **−0.32** (negative — it's not just fibroblasts)
- corr(core, hepatocyte fraction) = **−0.23** (negative — not hepatocyte spillover)

So the only confound left to remove is tumour abundance itself.

### The fix — residualisation
Per slide, regress the score on tumour fraction and keep the **leftover**:

```
leaving_score_resid = leaving_score − (part explained by tumour fraction)
```

By construction corr(resid, tumor_frac) ≈ **0.000**. The residual means
*"leaving-program expression beyond what the amount of tumour alone predicts"* —
i.e. **tumour-intrinsic invasiveness**. This is the **recommended Phase B
target**. It still holds organised biology: Moran's I stays positive
(T1=0.19, T3=0.27, T4=0.17, T11=0.16), and it still tracks the held-out EMT genes
after they too are residualised.

---

## 4. Worked example — three real spots from IU_PDA_T3

These are actual barcodes in `leaving_program_scores.csv`. They show why raw
score and residual can disagree — and why the residual is the honest target.
(In T3, raw score correlates with tumour fraction at **0.76**; after
residualising, **0.00**.)

| Spot (barcode) | tumour frac | raw `leaving_score` (z) | `leaving_score_resid` | Biological reading |
|----------------|:-----------:|:-----------------------:|:---------------------:|--------------------|
| **A** `…CAGATAATGGGCGGGT-1` (r66,c102) | 0.40 | **+2.58** | **+3.68** | Moderate tumour, yet program *far* above what that tumour amount predicts → genuinely invasive front. High on **both** — a true positive. |
| **B** `…CCACGGCAGGTGTAGG-1` (r71,c31) | 0.13 | +1.81 | **+3.95** | Only 13% tumour, so raw score is merely "above average" — but *for so little tumour* the program is extreme. Residual **surfaces** it: a few cells strongly running the leaving program. This is the spot the residual **rescues**. |
| **C** `…TCACTCAGCGCATTAG-1` (r0,c44) | 0.83 | +0.29 | **−2.18** | Dense tumour (83%), raw score looks "fine" — but for *that much* tumour the program is far **too low**. Residual correctly flags it **negative**: a bulky, non-invasive tumour core. The raw score was flattered by abundance. |

**Read A vs C together — this is the whole argument on one line:** spot C has
*twice* the tumour of spot A but is *not* invasive; spot A has less tumour but is
running the leaving program hard. The raw score can't tell them apart cleanly
(C even looks slightly positive). The **residual** separates them decisively
(A **+3.68** vs C **−2.18**). That separation is the biomarker we actually want.

---

## 5. Per-sample summary (the four PT slides at a glance)

| Sample | PT spots | tumour-dominated | Moran's I core | corr(core, held-out) | corr(core, tumor_frac) |
|--------|:-------:|:-----:|:-----:|:-----:|:-----:|
| IU_PDA_T1 | 3073 | 2143 | 0.389 | 0.263 | 0.519 |
| IU_PDA_T3 | 4241 | 1214 | 0.613 | 0.610 | 0.761 |
| IU_PDA_T4 | 3587 | 381 | 0.374 | 0.420 | 0.582 |
| IU_PDA_T11 | 2677 | 976 | 0.369 | 0.403 | 0.597 |

T3 is the strongest example slide (highest coherence *and* validator agreement);
T4 has the fewest tumour-dominated spots, which is worth flagging when it comes up.

---

## 6. Why this matters downstream (one sentence to pre-empt the next question)

In Phase B, the H&E / foundation model recovers the **raw** leaving score (the
*abundance* part, ρ≈0.3) but **not** the residual target (it sits at the noise
floor). Translation: **the images can see *how much* tumour is in a region, but
not *whether that tumour is invasive*** — because invasiveness is exactly the
signal we isolated by removing the abundance confound. That gap is the central
finding, and it only means something *because* the target was made confound-free
first.

---

## 7. Where the 19 genes come from (provenance & references)

The panel is **not** a machine-learned set — it's a *curated, canonical* EMT /
invasion signature assembled from three well-established sources, then validated
against the data (§3, §8). This is deliberate: with only 4 primary tumours, a
signature *derived* from this data would overfit. Anchoring on textbook biology
makes the target defensible and comparable to outside work.

**The three biological sources behind the panel:**

1. **Core EMT transcription factors** — `SNAI1`, `ZEB1`, `ZEB2`, `CDH2` (the
   "cadherin switch"). These are the consensus master regulators of EMT from the
   canonical reviews:
   - Kalluri & Weinberg (2009), *J Clin Invest* — "The basics of EMT."
   - Nieto, Huang, Jackson & Thiery (2016), *Cell* — "EMT: 2016" (consensus TFs).
   - Lambert, Pattabiraman & Weinberg (2017), *Cell* — EMT in metastasis.

2. **ECM / protease / secreted invasion genes** — `COL1A1`, `COL3A1`, `COL5A1`,
   `FN1`, `LOX`, `LOXL2`, `TNC`, `LAMC2`, `TIMP1`, `MMP1`, `SERPINE1`, `S100A4`,
   `TGFB1`. The large majority of these are members of the **MSigDB Hallmark
   Epithelial-Mesenchymal-Transition** gene set (Liberzon et al. 2015, *Cell
   Systems*) — the standard reference EMT set — which is heavily weighted toward
   exactly these secreted-matrix and remodelling genes. `ITGB6` (integrin
   αvβ6) is a well-documented PDAC invasion/TGF-β-activating marker.

3. **PDAC-specific dissemination & desmoplasia biology** — the choice to weight
   ECM remodelling so heavily reflects pancreatic cancer's defining desmoplastic,
   invasion-through-stroma phenotype:
   - Rhim et al. (2012), *Cell* — EMT and circulating dissemination *precede*
     overt PDAC.
   - Öhlund et al. (2017), *J Exp Med* — myCAF/iCAF and the desmoplastic ECM
     (motivates the CAF-fraction confound check in §3).

**The 7 held-out validators** (`VIM`, `SNAI2`, `PRRX1`, `MMP9`, `MMP14`,
`POSTN`, `SPARC`) are equally canonical EMT/invasion genes drawn from the same
literature — deliberately kept *out* of the core so they can independently test
it (they agree at **+0.46**, §3). Note 6 intended markers (`TWIST1`, `CDKN2A`,
`IGF2BP3`, `RACGAP1`, `RARRES3`, `MALAT1`) were **absent from the count matrix**
and could not be included.

> One-line answer for the meeting: *"The 19 genes are the standard textbook EMT
> program — the master transcription factors from the Kalluri/Weinberg and Nieto
> EMT reviews, plus the MSigDB Hallmark-EMT matrix/protease genes, tuned toward
> PDAC's desmoplastic invasion biology. We curated from literature rather than
> fitting the signature to 4 patients, then confirmed the choice against the data
> and an independent paper."*

---

## 8. Which genes actually carry the signal — and "top 5 vs 19"

We measured each gene's individual link to the **confound-free target**
(`leaving_score_resid`), averaged over all 4 tumours (see
`Outputs/stage1a_leaving_program/gene_importance.json`,
`docs/presentation/figB_leaving_gene_importance.png`).

### 8a. Not all 19 genes contribute equally
| Rank | Gene | corr with confound-free target | Note |
|-----:|------|:------------------------------:|------|
| 1 | `SERPINE1` | **+0.39** | pro-invasive (TGF-β target) |
| 2 | `S100A4` | +0.30 | motility / metastasin |
| 3 | `COL5A1` | +0.28 | ECM |
| 4 | `LAMC2` | +0.25 | invasion (laminin-γ2) |
| 5 | `FN1` | +0.25 | ECM |
| … | | | |
| 17 | `CDH2` | +0.12 | EMT-TF |
| 18 | `SNAI1` | +0.09 | EMT-TF |
| 19 | `MMP1` | +0.07 | protease |

**Key, slightly counter-intuitive finding:** the signal is carried by the
**secreted ECM / protease / TGF-β genes**, *not* by the famous EMT transcription
factors (`SNAI1`, `ZEB1`, `CDH2` sit at the bottom). This is expected at **55 µm
Visium resolution**: transcription factors are low-copy and noisy, whereas
secreted matrix genes are abundant and spatially coherent. So the leaving score
is really reading the *consequences* of EMT (matrix remodelling, invasion) more
than the upstream switch itself.

### 8b. Which genes are "abundance" vs "tumour-intrinsic" (the residual re-weights them)
Comparing each gene's correlation with the **raw** score vs the **residual**:
- `COL1A1` (raw **+0.62** → resid +0.20) and `FN1` (**+0.59** → +0.25) collapse
  after residualising — they mostly mark **how much tumour** is present.
- `TNC` (+0.10 → **+0.21**), `LAMC2` (+0.17 → **+0.25**), `ITGB6`, `TGFB1` go the
  *other* way — more linked to the confound-free target than the raw. These are
  the **genuinely tumour-intrinsic invasion** genes the residual promotes.

This is the gene-level picture of exactly what residualisation does.

### 8c. "What if we picked the top 4–5 genes?" — the impact, quantified
We built reduced scores from just the top-k genes and correlated them with the
full 19-gene confound-free target (avg of 4 tumours; a simple 19-gene score
tops out at 0.663 vs the AddModuleScore target, so treat that as 100%):

| Panel | Genes | corr vs full target | % of full signal | corr vs held-out EMT |
|-------|-------|:------------------:|:----------------:|:--------------------:|
| **Top 3** | SERPINE1, S100A4, COL5A1 | 0.517 | ~78% | 0.058 |
| **Top 5** | + LAMC2, FN1 | 0.549 | ~83% | 0.074 |
| **Top 8** | + TIMP1, LOXL2, TGFB1 | 0.610 | ~92% | 0.087 |
| Full 19 | (all) | 0.663 | 100% | 0.086 |

**What this means:**
- **A 5-gene panel recovers ~83% of the signal.** So yes — the target is
  compressible; most of the information lives in a handful of secreted markers,
  led by `SERPINE1` and `S100A4`. Good for a cheap readout (e.g. a small IHC /
  targeted panel).
- **But fewer genes generalise slightly worse.** Correlation with the
  *independent* held-out EMT genes *drops* (0.087 at 8 genes → 0.058 at 3). The
  full panel is the more robust, less noise-prone definition — which is why we
  keep all 19 as the target and use the top genes only for *interpretation*.

### 8d. The "bottom" genes — an honest negative
Ranking *all ~17,900 genes*: the strongest **positive** hits are the panel genes
themselves (independent confirmation the panel is the real signal). The
**negative** end, though, is weak (all around −0.05) with no coherent program,
and the canonical **epithelial** anchors — `CDH1` (E-cadherin), `EPCAM`, keratins
— sit at **≈ 0**, not negative. Interpretation to state plainly: at this
resolution the leaving score is an **added invasion/ECM program on top of
tumour**, *not* a clean epithelial-off / mesenchymal-on switch. We report this
rather than overselling a "cadherin switch" the data doesn't show.

---

### TL;DR for a slide
> **Leaving program** = the EMT + ECM gene signature a tumour cell runs to invade
> and leave the primary. **Leaving score** = one number per spot for how strongly
> it runs that program. **Confound** = a factor that fakes the score — here,
> *tumour abundance* (more tumour ⇒ higher score regardless of invasiveness) and,
> earlier, *liver tissue*. We remove abundance by residualising against tumour
> fraction, giving `leaving_score_resid`: tumour-*intrinsic* invasiveness, the
> real target.
