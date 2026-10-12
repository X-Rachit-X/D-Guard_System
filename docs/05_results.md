# Results

Every number here is produced by a script (see [12_how_it_was_built.md](12_how_it_was_built.md)
Part 6) and every figure by `scripts/make_figures.py` (in [../paper/figures/](../paper/figures/)).
Dataset: MILK10k, 11 classes. Test split: 787 held-out lesions, scored once. Everything else
— fusion rules and which one wins, reliabilities, calibration, gates, thresholds — was fitted
on the 786 validation lesions, with fusion rules **cross-fitted** (fit on 4/5, predict the
other 1/5). Operating point: missed-cancer cost 5, minimal-deferral mode, all gates on.
Team: **six specialists**, fused by the **learned gate** (Devil's Advocate v2, product
pooling), picked on validation.

---

## 1. Headline

| What patients get | Share | Accuracy (11 classes) | Missed cancers |
|---|---|---|---|
| ✅ **Answer — HIGH confidence** | **57.1%** | **77.1%** | **3.1%** |
| ⚠️ Answer — LOW confidence (doctor confirms) | 32.0% | 49.6% · **top-3: 85.7%** | 21.0% |
| 🧑‍⚕️ Deferred — evidence contradicts itself | 10.9% | — | — |
| **Answered in total** | **89.1%** | 67.2% | |
| 📷 Dermoscopy requested on the way | 29.5% | | |

At the same 57.1% coverage, on the same probabilities: **learned rejector 4.9% missed**
(significantly worse, p = 0.011), **confidence threshold 3.8%** (n.s., p = 0.18; but more
accurate, 81.5%).

**The key finding:** across four fusion back-ends D-GUARD's miss rate stays at 3.0–3.5%;
a confidence threshold ranges from 3.5% to 9.3% (§5).

---

## 2. The six specialists (test, with dermoscopy)

| Specialist | Acc. | Macro-F1 | Bal. acc. | Cancer AUROC |
|---|---|---|---|---|
| S1 photo head (evidential) | 0.595 | 0.264 | 0.266 | 0.787 |
| S2 dermoscopy head (evidential) | 0.705 | 0.400 | 0.411 | 0.854 |
| S3 details head (evidential) | 0.600 | 0.246 | 0.252 | 0.749 |
| S4 the old project's Model A | 0.219 | 0.077 | 0.096 | 0.553 |
| S5 similar-case vote (15 nearest training lesions) | 0.719 | 0.431 | 0.402 | 0.866 |
| **S6 embedding classifier** (class-balanced LR, C chosen on val) | 0.700 | 0.440 | **0.447** | **0.875** |

S6 is new: it draws one boundary through all training embeddings with class weights, so
rare diagnoses are not drowned out. Adding it to the team raised photo-only macro-F1 of the
selected rule from 0.332 to 0.377.

---

## 3. Fusion — seven rules, cross-fitted, picked on validation

*Val* = out-of-fold validation macro-F1 (used to choose); the rest is test.
Figure: `fig_fusion_rules`.

| Rule | Val | Acc. | **Macro-F1** | Bal. acc. | AUROC |
|---|---|---|---|---|---|
| **Photo + details (stage A)** | | | | | |
| Dempster–Shafer (heads only) | 0.274 | 0.644 | 0.286 | 0.280 | 0.799 |
| Simple average | 0.344 | 0.637 | 0.343 | 0.338 | 0.824 |
| Original formula Q·R·U | 0.380 | 0.673 | 0.381 | 0.369 | 0.836 |
| Best head per case | 0.319 | 0.672 | 0.337 | 0.323 | 0.788 |
| Iterative EM (v1) | 0.374 | 0.670 | **0.383** | **0.375** | 0.832 |
| Stacking | 0.272 | 0.644 | 0.308 | 0.296 | 0.822 |
| Learned gate, mixture | 0.396 | **0.677** | 0.371 | 0.355 | 0.833 |
| **Learned gate, product ← picked** | **0.412** | 0.657 | 0.377 | 0.360 | **0.838** |
| *Oracle best head (uses the label)* | — | 0.799 | 0.542 | | |
| **With dermoscopy (stage B)** | | | | | |
| Dempster–Shafer (heads only) | 0.361 | 0.690 | 0.376 | 0.378 | 0.856 |
| Simple average | 0.457 | 0.714 | 0.422 | 0.400 | 0.870 |
| Original formula Q·R·U | 0.496 | 0.719 | **0.460** | **0.446** | 0.872 |
| Best head per case | 0.509 | 0.727 | 0.449 | 0.427 | 0.816 |
| Iterative EM (v1) | 0.492 | 0.717 | 0.459 | 0.444 | 0.862 |
| Stacking | 0.413 | 0.720 | 0.372 | 0.361 | 0.853 |
| Learned gate, mixture | 0.485 | **0.731** | 0.452 | 0.431 | 0.878 |
| **Learned gate, product ← picked** | **0.517** | **0.731** | 0.438 | 0.418 | **0.885** |
| *Oracle best head* | — | 0.840 | 0.594 | | |

**Read:**
- Every evidence-aware rule beats the evidential network alone; photo-only macro-F1
  0.286 → 0.377 (+32%). With dermoscopy the picked rule has the best accuracy and AUROC.
- On test, the iterative EM and the original formula have a slightly higher macro-F1 than
  the picked gate. We keep the pre-declared choice (macro-F1 on 11 classes with several
  tiny classes is noisy).
- **Stacking**, the textbook learned combiner, is the worst learned rule — too many free
  inputs for 786 validation lesions.
- **Subset study** (31 / 63 subsets): the full team is within 0.007 (stage A) and 0.033
  (stage B) of the best subset on validation; S6 is in every top subset.
- **Oracle**: 0.594 macro-F1 — choosing the right specialist per case is the bottleneck.
- **Weights** (figure `fig_weights`): details head 0.41 / 0.35, S6 0.22 / 0.21, dermoscopy
  0.18 when present, old Model A 0.07.
- **Calibrated confidence**: mean shown 0.757 vs actual 0.731 (stage B); 0.660 vs 0.657
  (stage A). Figure `fig_calibration`.

---

## 4. Devil's Advocate evidence profile

Per specialist and case: quality Q, learned reliability R, uncertainty U, confidence,
margin, agreement, track record for the suspected diagnosis, consistency C, missingness M,
OOD distance. In v2 all of them set the weights through the learned gate
(`src/fusion/gating.py`). The gate listens most to confidence, margin, agreement,
consistency and OOD distance (figure `fig_gate_importance`). OOD flags 1.0% of test lesions.

---

## 5. Decision engine vs baselines (cost 5)

| Policy | Answered | Dermoscopy | Acc. | Bal. acc. | **Missed cancers** | False alarms | Mean cost |
|---|---|---|---|---|---|---|---|
| B1 always answer, photo only | 100% | 0% | 0.657 | 0.360 | 7.2% | 47.6% | **0.759** |
| B2 always take dermoscopy | 100% | 100% | 0.731 | 0.418 | 6.4% | 38.0% | 1.124 |
| B3 confidence threshold, same coverage | 57.1% | 0% | 0.815 | 0.413 | 3.8% | 39.3% | 0.881 |
| B3 + dermoscopy for everyone | 57.1% | 100% | **0.893** | **0.520** | **2.4%** | 32.9% | 1.293 |
| B4 learned rejector (L2D), same coverage | 57.1% | 0% | 0.764 | — | 4.9% | — | — |
| B5 dermoscopy for random patients | 100% | 29.5% | 0.682 | 0.361 | 6.5% | 44.7% | 0.842 |
| B5 dermoscopy for most uncertain | 100% | 29.5% | 0.700 | 0.386 | 5.6% | 45.2% | 0.790 |
| **D-GUARD (HIGH tier)** | **57.1%** | **29.5%** | 0.771 | 0.305 | **3.1%** | 50.0% | 1.047 |

- **Learn-then-Test certified** at both stages (λ̂ = 0.218, 0.276; α = 5%, δ = 10%).
- **Against us:** the threshold is more accurate; dermoscopy-for-all + threshold is better
  on everything but costs a dermoscopy per patient; always answering is cheapest; HIGH
  answers raise more false alarms (50%).

### The key finding — abstention borrows its safety from the fusion

Same policy, four fusion back-ends, all cross-fitted (figure `fig_backend_robustness`):

| Back-end | HIGH | HIGH acc. | **D-GUARD missed** | **Threshold missed** | AE policy (entropy) |
|---|---|---|---|---|---|
| Iterative EM, 5 specialists | 57.2% | 0.762 | 3.3% | **9.3%** | +0.30 (−0.04) |
| Learned gate, 5 specialists | 54.4% | 0.769 | 3.5% | 3.5% | +0.20 (+0.20) |
| Iterative EM, 6 specialists | 59.2% | 0.777 | 3.0% | 4.7% | +0.19 (+0.02) |
| **Learned gate, 6 specialists (final)** | **57.1%** | **0.771** | **3.1%** | **3.8%** | +0.07 (+0.26) |

D-GUARD stays at 3.0–3.5%. The threshold ranges 3.5–9.3%: **it is exactly as safe as the
ranking it is given**, and you cannot know which kind of ranking you have without test
data. Against us: on the final team, "ask the most uncertain for dermoscopy" beats the
learned acquisition estimator (AE +0.26 vs +0.07), and softmax ranks errors better than
ignorance (AUGRC 0.0757 vs 0.0820).

---

## 6. Cost of a missed cancer

| Miss cost | Answered | HIGH | HIGH accuracy | Missed (HIGH) | Threshold, same coverage |
|---|---|---|---|---|---|
| 2 | 88.8% | 61.6% | 0.763 | 4.1% | 4.1% |
| 3 | 88.9% | 59.1% | 0.768 | 3.9% | 4.3% |
| **5** | **89.1%** | **57.1%** | **0.771** | **3.1%** | **3.8%** |
| 10 | 89.2% | 53.4% | 0.786 | 1.9% | 3.8% |

---

## 7. Robustness — DERMA-STRESS (300 test lesions, photo + details)

"Model alone" = without the pipeline's quality and missing-details checks. Figure `fig_stress`.

| Condition | ignorance u | HIGH (model) | Missed (model) | HIGH (full system) | Missed (full) | Photo flagged |
|---|---|---|---|---|---|---|
| clean | 0.679 | 46% | 4.3% | 43% | 4.7% | 8% |
| **blur** | **0.605 ↓** | **50% ↑** | 2.0% | **0%** | — | 100% |
| **dark** | 0.723 | 22% | **12.3%** | **0%** | — | 100% |
| **noise** | 0.824 | 16% | 6.1% | **0%** | — | 100% (grainy) |
| JPEG | 0.741 | 27% | 5.0% | 22% | 1.5% | 24% |
| **occlusion** | 0.634 | 16% | **14.3%** | **0%** | — | 100% (covered) |
| **no details** | 0.754 | 48% | 5.6% | **0%** (details requested) | — | 8% |

Ignorance alone catches none of the dangerous cases (blur even makes the model *more*
confident). The pipeline's checks catch all of them. Cost: 8% of clean photos flagged.

---

## 7a. Statistics

| Quantity | Value | 95% CI |
|---|---|---|
| Answered | 89.1% | 86.8–91.2% |
| HIGH share | 57.1% | 53.5–60.6% |
| HIGH accuracy | 77.1% | 73.0–80.8% |
| Missed cancers, HIGH | **3.1%** | 1.6–4.8% |
| Missed, threshold | 3.8% | 2.1–5.6% |
| Missed, learned rejector | 4.9% | 3.0–7.0% |
| Team macro-F1 (stage B) | 0.438 | 0.360–0.503 |

| Paired test | Difference | 95% CI | one-sided p |
|---|---|---|---|
| Missed: learned rejector − D-GUARD | +1.8 pts | +0.2 to +3.5 | **0.011** |
| Missed: threshold − D-GUARD | +0.7 pts | −0.7 to +2.0 | 0.18 (n.s.) |
| Macro-F1: team − Dempster–Shafer (stage B) | +0.062 | −0.006 to +0.121 | **0.038** |
| Missed: gate − EM (6 specialists) | +0.1 pts | −0.5 to +0.7 | n.s. |
| HIGH share: gate − EM | −2.2 pts | −3.8 to −0.4 | significant |

### Fairness — by skin tone

| Skin tone | n | HIGH share | HIGH accuracy | Missed (HIGH) |
|---|---|---|---|---|
| 1 | 14 | 14% | 50% | 0% — too few to judge |
| 2 | 67 | 39% | 81% | **11.5%** |
| 3 | 489 | 64% | 77% | 2.2% |
| 4 | 167 | 53% | 76% | 4.5% |
| 5 | 50 | 42% | 76% | 0% |

Lighter tones are under-represented and worse served. Reported, not fixed.

### Net benefit ("refer as possibly malignant")

| Threshold | Refer all | P(malignant) ≥ t | D-GUARD rule |
|---|---|---|---|
| 0.1 | 0.706 | 0.698 | 0.694 |
| 0.3 | 0.622 | 0.626 | **0.628** |
| 0.5 | 0.471 | **0.548** | 0.507 |

---

## 7b. External tests, no retraining (figure `fig_shift`)

| | MILK10k | PAD-UFES-20 | Derm7pt |
|---|---|---|---|
| Lesions / photos | 787 | 2,298 photos | 1,003 |
| Dermoscopy | yes | no | when requested |
| Malignant share | 72% | 79% (MILK definition) | 29% |
| Mean ignorance, photo stage | 0.676 | 0.748 | 0.797 |
| Photos failing quality | 8% | 28% | 29% |
| OOD flags | 1% | 6% | 2% |
| Dermoscopy requested | 30% | 28% (cannot be given) | 39% |
| Accuracy, all cases | 0.731 | 0.367 | 0.479 (photo only 0.402) |
| Cancer AUROC | 0.885 | 0.840 | 0.738 |
| HIGH share | 57.1% | 11.4% | 13.0% |
| HIGH accuracy | 0.771 | 0.517 | 0.708 |
| **Missed cancers, HIGH** | 3.1% | **0.0%** | **3.8%** |
| Threshold, same coverage: missed / acc. | 3.8% / 0.815 | 0.0% / 0.574 | 5.4% / 0.846 |

The system **notices** the shift (confident answers 57% → 11–13%) and the safety target
holds on both. On Derm7pt, asking for dermoscopy lifts accuracy 0.40 → 0.48. The 11-class
diagnosis does **not** transfer (melanoma recall 31% on Derm7pt; the threshold is more
accurate). Safe, not accurate, on a new population.

---

## 7c. The old project's seven models, re-scored (figure `fig_legacy_models`)

`scripts/legacy_pad_models.py` loads the saved A–G weights and cached features unchanged.

| Model | Inputs | Reported | Same test photos | Lesion unseen (197) | **Patient unseen (132)** |
|---|---|---|---|---|---|
| A | image | 85.5% | 61.4% | 55.3% | 56.8% |
| B | metadata | 84.1% | 75.9% | 68.5% | **75.0%** |
| C | free text | 83.5% | 23.5% | 25.4% | 27.3% |
| D | image + metadata | 88.4% | 63.5% | 57.4% | 58.3% |
| E | image + text | 88.1% | 61.4% | 55.8% | 57.6% |
| F | text + metadata | 87.2% | 72.2% | 67.0% | 72.7% |
| G | all three | 89.9% | 62.0% | 55.8% | 57.6% |
| **A–G fused by D-GUARD's learned gate** (5-fold CV) | | | | | **72.7%** (average: 59.8%) |

The old evaluation adjusted predictions toward hard-coded target accuracies, and its split
was by photo ([what_did_not_work.md](what_did_not_work.md) §40). Honest numbers: 57–75%.

## 7d. Do the scribe's warning signs mean anything? (figure `fig_symptoms`)

PAD-UFES-20 questionnaire, 1,641 lesions, cancer = BCC/SCC/MEL:

| Sign | Cancer if yes | if no | Relative risk (95% CI) |
|---|---|---|---|
| bleeding | 85.0% | 32.0% | 2.66 (2.43–2.91) |
| rapid growth | 52.7% | 17.5% | 3.00 (2.51–3.59) |
| change | 61.1% | 30.8% | 1.98 (1.67–2.35) |
| itching | 52.8% | 32.1% | 1.65 (1.45–1.87) |
| pain | 84.8% | 37.2% | 2.28 (2.09–2.48) |
| raised (added to the scribe after this) | 60.1% | 21.1% | 2.84 (2.43–3.34) |
| **any sign** | **48.7%** | **12.4%** | **3.91 (2.77–5.52)** |

---

## 8. Other measured facts

| | |
|---|---|
| Dermoscopy improves cancer AUROC of the evidential heads | 0.799 → 0.856 |
| ...and lowers ignorance | 0.676 → 0.439 |
| Local model = Kaggle model | test macro-F1 0.376 vs 0.379 |
| Post-hoc logit adjustment | helped val, hurt test — dropped |
| Rare classes (BEN_OTH, INF, MAL_OTH, VASC) | almost never predicted |

## 9. Seeds

One training seed (42) so far; each extra Kaggle seed also redraws the split.
`python scripts/run_seeds.py runs/<seed7> runs/<seed2026>` → mean ± sd of every headline number.

## 10. Limits

See [limitations.md](limitations.md) and [what_did_not_work.md](what_did_not_work.md).
