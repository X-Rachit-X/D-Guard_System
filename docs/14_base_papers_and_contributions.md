# Base papers, what we did, what we achieved, what we publish

This is the document to read before a viva or a paper discussion. It answers five questions
in order: **where did this project start, what did we change, what did we achieve, what
exactly do we claim in the paper, and what did we find along the way.**

---

## 1. The starting point: two base papers and one prototype

The project grew out of three earlier pieces of our own work. They are in the repository
so anyone can compare.

### Base paper 1 — the agentic design (`paper/old1.tex`)

**Title:** *DERMA-GUARD: An Agentic Multi-Model Consensus Framework with Dynamic
Diagnosis-Adaptive Evidence Fusion for Reliable Multimodal Dermatological Diagnosis.*

| What it proposed | In plain words |
|---|---|
| A panel of four vision "specialists" (a VLM, a foundation model, a CNN, a transformer) coordinated by an LLM orchestrator | several AI models give opinions, like doctors in a case meeting |
| A **Devil's Advocate** agent that profiles evidence: quality, reliability, consistency, missingness | someone whose job is to question each opinion |
| **Iterative diagnosis-adaptive fusion** (EM-style) | how much to trust each source depends on the suspected diagnosis, and the diagnosis depends on the trust — so alternate until stable |
| A **four-level consensus hierarchy**: accept, arbitrate, acquire, escalate | four possible outcomes instead of one label |
| **DERMA-STRESS**, a 15-condition damage benchmark | test what happens with bad input |
| Datasets: PAD-UFES-20 and Derm7pt | |

**What it did not have: results.** Every headline number in the draft is a placeholder:
`\todo{XX.X\%}` accuracy, `\todo{82.0\%}` stress accuracy, `\todo{94\%}` claims caught. The
ideas were never measured.

### Base paper 2 — the risk-controlled design (`paper/old2.tex`)

**Title:** *DERMA-GUARD: Risk-Controlled, Cost-Aware Evidence Acquisition and Deferral for
Multimodal Dermatology AI.*

| What it proposed | In plain words |
|---|---|
| Three actions: **answer now, request dermoscopy, defer to a human** | the system can ask for a closer look |
| An expected-loss rule over an **asymmetric cost matrix** | a missed cancer costs more than a false alarm |
| **Learn-then-Test / conformal risk control** | a mathematical guarantee on the missed-cancer rate |
| **MILK10k** (paired clinical + dermoscopic images) | the right dataset for "should we ask for dermoscopy?" |
| Baselines: confidence threshold, learning to defer, HOT-style rules, oracle | compare against simpler methods |
| Fairness by skin tone, distribution shift | |

**What it did not have: an implementation or results.** It is a plan; it even says the
method "may or may not outperform all baselines".

### The old prototype (`HealthSphere_archive/Old_system`)

A working application on PAD-UFES-20 with seven models: **A** image, **B** metadata, **C**
free text, **D** image+metadata, **E** image+text, **F** text+metadata, **G** all three. It
reported 84–90% accuracy. When we audited it, three problems appeared:

1. **The split was by photo**, so 43% of test photos showed a lesion that was also in
   training (62% a patient who was).
2. **The evaluation rewrote predictions**: a table of target accuracies per model, and code
   that swapped probabilities between the true class and its rival until the number of
   correct predictions matched the target.
3. **The live demo looked up the true label** for PAD-UFES file names, and its "foundation
   model" panel members were arithmetic variations of Model A.

Re-scored honestly (same weights, no adjustment), the old models reach **57–75% on
patients never seen in training**; the text model is near chance (27%).

---

## 2. What we did — the unification

We combined the **ideas of paper 1** with the **statistical machinery of paper 2**, on
**the dataset paper 2 chose**, and replaced every hand-set number with one learned or
measured on validation data.

| From | Idea | What we built and measured |
|---|---|---|
| Paper 1 | specialist panel | **six specialists**: three evidential heads (photo, dermoscopy, details), the old **Model A**, a similar-case vote, and a new class-balanced **embedding classifier** |
| Paper 1 | Devil's Advocate | ten evidence signals per specialist per case, with **learned** reliability |
| Paper 1 | diagnosis-adaptive EM fusion | kept as "v1", with a **measured** trust table instead of typed numbers |
| New | learned evidence gate | "v2": every signal sets the weights, fitted by log-likelihood |
| New | fair comparison | **seven fusion rules**, **cross-fitted**, chosen on validation; all specialist subsets; an oracle |
| Paper 1 | 4-level consensus | kept as a summary of the policy's outcome |
| Paper 2 | answer / acquire / defer | **four outcomes**: HIGH answer, LOW answer (doctor confirms), ACQUIRE (dermoscopy, missing details, better photo), DEFER |
| Paper 2 | cost matrix + LTT | implemented; **certified** at both stages |
| Paper 2 | conformal sets | benign/malignant **Mondrian** sets as a "no false reassurance" gate |
| Paper 1 + 2 | DERMA-STRESS | six damage types; new **quality checks** (grain, covered centre) and a **missing-details request** |
| Paper 2 | baselines | threshold, learned rejector, always-answer, random / entropy acquisition — all on the **same** probabilities |
| Paper 2 | shift, fairness | **two external datasets** (PAD-UFES-20, Derm7pt), skin-tone analysis |
| Prototype | Models A–G | re-scored honestly and **fused by our gate** on PAD-UFES-20 |
| Prototype | care platform | rebuilt: patient app, doctor queue, messages, follow-ups, deletion, audit, REST API, explanations |

---

## 3. What we achieved (787 unseen MILK10k patients)

| | Result |
|---|---|
| Patients who get an answer | **89.1%** |
| Confident (HIGH) answers | **57.1%**, **77.1%** accurate over 11 diagnoses |
| Cancers called benign among confident answers | **3.1%**, certified ≤ 5% with 90% probability |
| Photo-only macro-F1, evidential network → full team | **0.286 → 0.377** |
| With dermoscopy | accuracy **0.731**, cancer AUROC **0.885** |
| Learned rejector at the same coverage | misses **4.9%** (D-GUARD significantly safer, p = 0.011) |
| Confidence threshold at the same coverage | misses **3.8%** (difference not significant) |
| Across four fusion back-ends | D-GUARD **3.0–3.5%**, threshold **3.5–9.3%** |
| Damaged photos (blur, dark, noise, occlusion) | **0** confident answers with the checks |
| PAD-UFES-20 / Derm7pt, no retraining | **0.0% / 3.8%** missed among confident answers |
| Old models A–G fused by our gate (unseen patients) | **72.7%** vs 59.8% averaging |

---

## 4. What we publish — the claims in the paper

The paper (`paper/main.tex`, 10 pages) claims exactly these five things, each backed by a
table or figure and a script:

1. **A six-specialist team whose trust is learned from all of the evidence**, chosen fairly
   from seven rules — it improves photo-only diagnosis by about a third.
2. **A calibrated four-outcome policy** that asks for evidence when needed and certifies its
   missed-cancer rate.
3. **The key finding:** a confidence threshold is only as safe as the probabilities beneath
   it (3.5–9.3% across four fusions), whereas the calibrated policy stays at 3.0–3.5%.
4. **Evidential uncertainty alone does not notice damaged photos;** explicit checks do.
5. **An honest re-evaluation** of our own earlier prototype.

**What we do not claim:** clinical validity; beating the MILK10k challenge leaderboard;
that the policy beats a confidence threshold on every model; fairness for the lightest skin
tones; more than one training seed.

**Is it original?** The individual techniques (evidential learning, Dempster–Shafer,
conformal prediction, Learn-then-Test) are established and cited. The contribution is
their **integration** into one decision with learned multi-specialist trust, measured on
paired clinical–dermoscopic data, the **robustness finding**, and the **evaluation
discipline**. The text of the paper and the report was written for this project; every
number comes from our own experiments.

---

## 5. Findings, in one list

| # | Finding | Good or bad for us |
|---|---|---|
| 1 | Learned, evidence-aware weighting beats averaging and beats the evidential network alone | ✅ |
| 2 | Stacking (the textbook combiner) is the worst learned rule — too little data | lesson |
| 3 | The oracle reaches 0.59 macro-F1 vs our 0.44: choosing whom to trust is the bottleneck | open problem |
| 4 | A threshold's safety depends on the fusion; the calibrated policy's does not | ✅ key finding |
| 5 | A learned rejector is significantly less safe than D-GUARD on the final model | ✅ |
| 6 | The threshold is more accurate than D-GUARD's HIGH tier | ❌ reported |
| 7 | Blur makes the model *more* confident; ignorance misses blur, dark, noise, occlusion | lesson → checks added |
| 8 | Without patient details the learned gate missed 17.6% of cancers → now it asks | lesson → fixed |
| 9 | Fitting fusion in-sample inflated coverage by 3 points → cross-fitting | lesson → fixed |
| 10 | The safety decision transfers to new datasets; the 11-class diagnosis does not | mixed |
| 11 | Asking for dermoscopy on Derm7pt raises accuracy 0.40 → 0.48 | ✅ |
| 12 | Lighter skin tones are worse served (tone 2: 11.5% missed) | ❌ reported |
| 13 | Every warning sign the scribe uses raises cancer risk (any sign: RR 3.9) | ✅ |
| 14 | The old prototype's 84–90% were set, not measured; honestly 57–75% | lesson |
| 15 | On the final model, "ask the most uncertain" beats our learned acquisition estimator | ❌ reported |

---

## 6. Did we train on the external datasets?

**No — on purpose.** PAD-UFES-20 and Derm7pt are used only as **external tests**, without
any retraining or recalibration. Their job is to answer "what happens on patients from a
different country, camera and clinic?" If we trained on them, they would stop being a fair
test. What *is* in the system from PAD-UFES-20:

- **Model A** of the old prototype, trained on PAD-UFES-20, is specialist **S4** in the
  MILK10k team (it gets about 7% of the weight).
- All seven old models **A–G** are re-scored and fused on PAD-UFES-20 photos of unseen
  patients (Research view → *Old models A-G*).
- The PAD-UFES-20 **questionnaire** validates the scribe's warning signs (Research view →
  *Shift: PAD-UFES-20*).

Training on all three datasets together is a sensible next step for a product, but it
would need a new held-out test set to measure honestly.

---

## 7. Where everything is

| Item | Location |
|---|---|
| Base paper 1, base paper 2 | `paper/old1.tex`, `paper/old2.tex` |
| Our paper (10 pages, compiled) | `paper/main.tex`, `paper/main.pdf` |
| Project report (100+ pages, VIT template) | `reports/DERMA-GUARD_Project_Report.docx` / `.pdf` (local; rebuild with `scripts/build_report.py`) |
| Presentation (16 slides with speaker notes) | `reports/DERMA-GUARD_Presentation.pptx` (local; rebuild with `scripts/build_slides.py`) |
| All figures | `paper/figures/` (`scripts/make_figures.py`, `scripts/make_figures_extra.py`) |
| Every number | [05_results.md](05_results.md) |
| The project in plain words | [13_simple_guide.md](13_simple_guide.md) |
| How it was built and checked | [12_how_it_was_built.md](12_how_it_was_built.md) |
