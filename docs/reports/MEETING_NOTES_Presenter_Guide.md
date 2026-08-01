# Presenter's Guide & Meeting Notes
### Spatial Transcriptomics + Foundation Models — PDAC Liver-Metastasis Project
*A plain-language script for presenting to clinicians and pathologists. You do not need a machine-learning or bioinformatics background to deliver this. Everything below is written to be said out loud, with the jargon translated in-line and a full glossary at the end.*

---

## 0 · The one paragraph to memorise

> We wanted to look at a **routine pancreatic-cancer slide** (the pink-and-purple H&E a pathologist reads every day) and flag the tumour regions that are biologically **"getting ready to spread"** to the liver — without needing an expensive molecular test at diagnosis. To teach the model, we used a richer research technology called **spatial transcriptomics**, which reads gene activity at thousands of tiny spots across the tissue. **What we found:** the "getting ready to leave" biology is **real and confirmed by an independent dataset**; we can see it in the genes and the cells. But from the plain slide alone, on a patient the model has never seen, it currently reads reliably **how much tumour is present**, not yet the finer "is it preparing to leave" state. That is an honest, rigorously-checked result — and we know exactly what would push it further.

If you remember nothing else, say that. Everything else is support.

---

## 1 · How to frame the talk (say this at the top)

- **This is a discovery talk, not a product demo.** We found a genuine biological signal and we are being honest about how far a routine slide can currently read it.
- **The project is deliberately built to catch itself cheating.** Several times an easy "success" would have been a mirage; each time we ran the check that exposed it. That rigor is the deliverable as much as any number.
- **What we most want from this room:** your pathologist's eye. The single most valuable missing piece is a few slides with the **invasive fronts marked by an expert**, so we can test our score against your read, not just against gene data.

---

## 2 · The clinical problem (1 slide, ~60 seconds)

Pancreatic cancer (PDAC) is so deadly largely because it **spreads early — usually to the liver — often before the patient feels sick.** If we could look at the original pancreas tumour and point to the regions "primed to leave," that could eventually help with prognosis and treatment timing.

The catch: the clue to "primed to leave" lives in the tumour's **gene activity**, which you can't see on a normal H&E slide. Measuring gene activity everywhere is expensive and not routine. **The opportunity:** if a model learns the link between *what tissue looks like* and *what its genes are doing*, then in the clinic you'd only need the cheap H&E.

**The teaching analogy (use this — it lands well):** picture training a medical student with a senior expert beside them. During training the student sees both the slide *and* the expert's molecular read-out. The goal is that afterwards the **student makes the call from the slide alone** — the expert (the gene data) leaves the room at exam time. Here, the gene data is the teacher (training only) and the image model is the student (used forever after).

---

## 3 · The data (1 slide)

We have **6 tissue samples** from PDAC patients — **18,859 tiny tissue locations ("spots")**:

| Type | Samples | What it is |
|------|---------|------------|
| **Primary tumour (PT)** | T1, T3, T4, T11 | The original tumour in the pancreas |
| **Liver metastasis (HM)** | HM11, HM13 | Tumour that already spread to the liver |

Only **patient 11** has *both* the primary (T11) and its own liver metastasis (HM11) — remember this, it matters later.

**Every spot carries three matched readings** (this is the heart of the whole project):
1. **An H&E image patch** — a 224×224-pixel picture of the tissue there (what a pathologist sees).
2. **Gene expression** — which of ~17,900 genes are switched on, and how strongly (the real lab measurement).
3. **Cell make-up** — the mix of cell types in that spot (tumour %, fibroblast %, immune %, liver % …), estimated from the genes.

So each spot = one picture + one gene read-out + one cell recipe, all of the same place. **Lining up those three views is the entire task.**

---

## 4 · NEW — Reading the biology directly off the slide *(gene atlas figure `figG_atlas_T11`)*

**What the audience sees:** the real T11 slide, then the same slide seven times, each dot a spot coloured by one gene's activity.

**Say:** "Before any modelling, look what the raw gene data already shows. **EPCAM and KRT8** — the tumour/epithelial genes — light up the malignant glands. **Collagen (COL1A1) and SPARC** fill the surrounding scar tissue. And the invasion genes — **SERPINE1, S100A4, POSTN** — pick out gradients toward the tumour edges. The biology partitions the tissue in front of your eyes; it's not an abstraction."

**Why it matters:** this establishes trust in the data *before* any model. A pathologist can sanity-check every panel against the morphology.

---

## 5 · NEW — The liver confound, proven in the raw genes *(figure `figG_liver_genes_HM11`)*

**What the audience sees:** the HM11 (liver metastasis) slide, then four hepatocyte (liver-cell) genes — **ALB, APOA1, HP, TTR** — mapped on it.

**Say:** "Here's the one trap that could have fooled the whole project. A liver metastasis sits *inside the liver*, so some spots are genuinely liver cells. Watch: all four classic **liver genes light up in exactly the same regions** — that's real liver tissue mixed into the sample. If we'd naively trained a model to tell 'metastasis vs primary,' it would mostly have learned **'does this look like liver?'** — location, not cancer biology. Four independent genes agreeing is airtight proof the contamination is real, so we designed around it."

**The fix (state it plainly):** we do **not** compare metastasis-vs-primary across patients. Instead we define our target **inside the primary tumours**, where there's no liver to confuse us.

---

## 6 · The "leaving program" — our target *(figures `fig3`, `fig4`, `figB`)*

**What it is:** when a cancer cell prepares to detach and travel, it goes through a known shift called **EMT (epithelial-to-mesenchymal transition)** — it stops being a tidy, stuck-in-place cell and becomes a mobile, invasive one. There are textbook genes for this.

We built a **"leaving-program score"** per primary-tumour spot from **19 canonical EMT/invasion genes** — essentially a checklist: the more "departure-prep" genes switched on, the higher the score.

**Two honesty checks (both passed):**
- **It's spatially organised**, not random — high-score spots form coherent zones, like invasive fronts (a clumpiness statistic, *Moran's I*, confirms it).
- **An outside dataset agrees** — the source paper's own independently-computed EMT signature lines up with our score (+0.21), and unrelated programs don't. An outside ruler says we're measuring genuine EMT.

**Which genes carry it (figure `figB`):** interestingly, the confound-free signal is driven by **secreted matrix and protease genes** (SERPINE1, S100A4, collagens, laminin), not the classic switch genes (SNAI1/ZEB1) — those are too faint to read reliably at this spot size. That's biologically sensible for tissue-scale spots.

---

## 7 · NEW — Anatomy of a spot *(flagship figure `figH_anatomy_of_a_spot`)*

**This is the "not a black box" slide. Slow down here.**

**What the audience sees:** two spots from the *same* T11 tumour, walked down the full chain — whole slide → zoomed tissue → the exact H&E patch → the cell make-up → the gene read-out.

**Say:** "Two spots, both about equally **tumour-rich** — 67% and 85% tumour — so 'how much cancer' is controlled for. But look at the genes. The **invasive spot switches ON** the invasion program — SERPINE1, S100A4, and the mesenchymal gene VIM — and turns **down** the epithelial genes EPCAM and KRT8. The **bulky spot** does the opposite: it keeps its epithelial identity and stays quiet on invasion. Same amount of tumour, opposite behaviour — and you can check every step by eye, from the tissue to the patch to the cells to the genes. **That** is what our score measures: not density, but the switch into an invasive state."

**Why it's the centrepiece:** it makes the abstract score physically real and verifiable for a pathologist. It literally is the WSI → spot → cells → genes chain in one view.

---

## 8 · NEW — The tumour ecosystem *(figure `figC_cell_ecosystem`)*

**What the audience sees:** (A) the cell make-up of each of the 6 tissues, (B) how each cell type relates to the leaving score, (C) the tumour-cell map next to the raw score on a real slide.

**Say:** "This panel is where the whole story clicks into place. Look at panel B. The **raw** leaving score tracks one thing above all — the **fraction of malignant tumour cells** (+0.53); everything else, the fibroblasts and immune cells, is negative simply because that's where the tumour *isn't*. Now watch the grey bars: once we remove 'how much tumour is here,' the **confound-free** score has **no cell-type signature at all** — every bar collapses to about zero. And panel C shows it on the real slide: the tumour-cell map and the raw-score map are essentially the same picture."

**Why it matters (this is the punchline of the honesty story):** the part of our score that a slide can read is, at heart, **tumour content**. The finer invasion state, once you subtract abundance, doesn't correspond to any single cell population you can point at — which is exactly what you'd expect when each spot (~55 µm) blends several cells together. This *reinforces* the whole thesis rather than contradicting it.

*(Panel A also visually re-proves the confound: the two metastases carry a grey "liver / hepatocyte" band; the primaries have none.)*

---

## 9 · What a plain slide can — and cannot — do *(figures `fig5`, `figP_T4`, `figP_T1`)*

**This is the honest ceiling. Deliver it calmly and confidently — it's a strength, not an apology.**

We tested the real clinical way: **leave-one-patient-out** — train on 3 primary tumours, predict the 4th the model has *never seen*, repeat for all four. Findings:

1. From H&E on a new patient, the model predicts the **raw** score **moderately** (correlation +0.29). Not nothing.
2. But once we remove "how much tumour is here," the **confound-free** invasion signal drops to the **noise floor** (~+0.09). The fine state isn't readable from the slide yet on unseen patients.
3. Humbling check: a trivial "just measure how much tumour" model scores **+0.48 — and beats our fancy model.** So the model's real signal *is* tumour amount.
4. The elaborate **multi-modal bridge added essentially nothing** over a standard off-the-shelf image model (0.289 vs 0.270 — tied).

**On the tissue (`figP_T4`, `figP_T1`):** the prediction captures the **broad** high/low zones (because those track tumour density) but **smooths away fine detail** — exactly what the numbers said. The green "agree" and red "disagree" zoom boxes make it checkable spot-by-spot.

**One-liner for this section:** *"Today the slide reliably senses how much tumour is present — not yet the finer metastatic state — on new patients. Better to know that precisely than to over-claim."*

---

## 10 · The honest scorecard *(figure `fig7`)*

Six checks, written down **before** we trusted any number (so we couldn't move the goalposts):

| Check | Result |
|---|---|
| Is the target real EMT? | ✅ **Yes** (matches independent paper, +0.21) |
| Is it spatially organised? | ✅ **Yes** (coherent zones) |
| Can H&E read the confound-free state on new patients? | ✘ Not yet (noise floor) |
| Does it beat a trivial "tumour amount" model? | ✘ No (amount model wins) |
| Did the multi-modal bridge help? | ✘ No (tied with plain image model) |
| Did the primary→metastasis link hold in patient 11? | ✘ No (that axis is microenvironment) |

**The take:** the **biology is real and validated**; the **slide read-out is amount-limited at this resolution.** Clean, confound-checked, no over-claiming.

---

## 11 · What would push it further (the roadmap — end on this)

1. **Pathologist annotation of invasive fronts** — *the single most valuable next step, and it's an ask for this room.* It gives us expert ground truth to test the score against.
2. **Higher-resolution platforms** (Visium HD / Xenium) — a spot today (~55 µm) blends several cells together and blurs the fine signal; single-cell resolution would sharpen it.
3. **More matched primary + metastasis patients** — only one patient here has both; more would let the primary→metastasis link be *learned* rather than guessed.

**Deployment note (be honest if asked):** the true "run it on a brand-new H&E-only slide" step needs the **full-resolution scanned slides** and a GPU pass of the foundation model — a scoped engineering step, not done here. What we *did* show (leave-one-patient-out) is the faithful stand-in: predicting a patient the model never saw.

---

## 12 · Anticipated questions (prep)

**Q: Isn't a negative result disappointing?**
A: The *biology* is a positive, validated finding — we can see the leaving program in genes and cells. What's honestly bounded is how much a *routine slide* can read it today. Knowing that precisely is what lets us invest in the right next step instead of over-promising a tool that isn't ready.

**Q: Why only 6 samples? Isn't that too few?**
A: Yes, and we say so. Spatial transcriptomics is expensive, and matched primary+metastasis pairs are rare — we have only one. The pipeline is built so that adding samples strengthens it directly. Small N is exactly why we leaned so hard on confound checks and external validation rather than raw performance numbers.

**Q: Could the model just be detecting necrosis / staining artefacts?**
A: We filtered on quality (spots had 0% mitochondrial reads, so no dying-cell distortion), and the score is anchored to specific, named EMT genes that an outside dataset independently confirms — not to generic "busy-looking" tissue.

**Q: You removed the "tumour amount" signal — isn't tumour amount itself clinically useful?**
A: It can be, and the raw score (which includes it) is the more *predictable* one. We separate the two deliberately so we don't fool ourselves into calling an abundance detector a "metastasis detector." Both are reported.

**Q: What's a "spot," exactly?**
A: One tiny tissue location about 55 microns across — roughly a handful of cells — where the Visium technology measured gene activity. Every dot in our maps is one spot.

**Q: What do you actually need from us?**
A: A few primary-tumour slides with the **invasive fronts outlined** by a pathologist. That single piece of expert ground truth would let us test the score against clinical reality and is the fastest way to move the project forward.

**Q: Is any of this ready for clinical use?**
A: No — this is research. It's a rigorous foundation and an honest map of what's real versus what's not yet readable, not a diagnostic tool.

---

## 13 · Glossary (say the plain version)

| Term | Plain meaning |
|------|---------------|
| **H&E** | The standard pink/purple stained tissue slide pathologists read. |
| **Spatial transcriptomics (Visium)** | Technology that measures gene activity at thousands of tiny spots across a tissue slide, keeping each spot's location. |
| **Spot** | One tiny tissue location (~55 µm, a few cells) with its own image + gene read-out + cell mix. |
| **Foundation model (UNI2-h, CONCH…)** | A large image network pre-trained on millions of pathology images; turns a patch into a numeric "fingerprint." |
| **Embedding** | A list of numbers summarising an image or gene profile so a computer can compare them. |
| **Gene expression / counts** | How strongly each gene is switched on in a spot — the real lab measurement. |
| **EMT / "leaving program"** | The shift a cancer cell makes when preparing to detach, invade, and spread. |
| **Hepatocyte** | A liver cell. Its genes (ALB, TTR…) mark real liver tissue — the confound in the metastases. |
| **CAF (cancer-associated fibroblast)** | Scar-forming stromal cells that partner with invading tumour. |
| **RCTD** | A method that estimates the **mix of cell types** in each spot from its genes. |
| **scVI** | A method that compresses ~17,900 noisy gene measurements into a clean 50-number summary. |
| **Confound** | A sneaky alternative explanation (e.g. "looks like liver") masquerading as the thing you care about. |
| **Residualize / confound-free** | Mathematically subtract a confound (here, "how much tumour") so what's left is the clean signal. |
| **Leave-one-patient-out (LOSO)** | Honest testing: train on some patients, test on one never seen — the real measure of generalisation. |
| **Moran's I** | A "clumpiness" score — high means high-value spots cluster together (real geography, not noise). |
| **Correlation (ρ)** | How strongly two things move together: 0 = unrelated, ~1 = lockstep. +0.29 = moderate, +0.09 ≈ nothing. |
| **Noise floor** | The "signal" you'd get from pure randomness; being *at* it means nothing real is left. |

---

*Figures referenced live in `Outputs/presentation_figures/` (full size) and `.../slides/` (slide-formatted). The built deck is `docs/presentation/Meeting - Results and Findings (PDAC ST + Foundation Model).pptx`. Full technical log: `REVIEW_PLAN.md`; plain narrative: `PROJECT_EXPLAINED.md`.*
