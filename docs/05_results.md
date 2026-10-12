# Results

Every number is produced by a script in the repository (§11). Dataset: MILK10k, 11 classes.
Test split: 787 held-out lesions, scored once. Everything else — fusion rules and which one
wins, learned reliabilities, calibration, gates, thresholds — was fitted on the 786
validation lesions, with fusion rules **cross-fitted** (fit on 4/5, apply to the other 1/5)
so the safety calibration never sees in-sample probabilities.
Operating point: missed-cancer cost 5, minimal-deferral mode, quality / OOD /
missing-details gates on. Fusion: the learned gate (Devil's Advocate v2), picked on
validation.

---

## 1. Headline

| What patients get | Share | Accuracy (11 classes) | Missed cancers |
|---|---|---|---|
| ✅ **Answer — HIGH confidence** | **54.4%** | **76.9%** | **3.5%** |
| 📷 Asked for dermoscopy first | 26.3% | (then answered) | |
| ⚠️ Answer — LOW confidence (doctor confirms) | 35.1% | 54.3% · **top-3: 81.5%** | 13.0% |
| 🧑‍⚕️ Deferred — evidence contradicts itself | 10.5% | — | — |
| **Answered in total** | **89.5%** | | |

**Against simple abstention on the same probabilities, same 54.4% coverage:** confidence
threshold 3.5% missed (81.1% accurate), learned rejector 3.3% (77.8%). **Equally safe** —
see §5 for why this is the most important finding.

**Fusion:** the learned Devil's Advocate wins on validation; photo-only macro-F1
**0.286 → 0.332** over evidential fusion.

---

## 2. The specialist team

| Specialist | What it is | Acc. | Macro-F1 | Bal. acc. | Cancer AUROC |
|---|---|---|---|---|---|
| S1 clinical-photo head | evidential, ResNet50 fine-tuned on MILK10k | 0.595 | 0.264 | 0.266 | 0.787 |
| S2 dermoscopy head | evidential, same backbone | 0.705 | 0.400 | 0.411 | 0.854 |
| S3 patient-details head | evidential, residual MLP | 0.600 | 0.246 | 0.252 | 0.749 |
| S4 **PAD-UFES Model A** | **the original project's image model**, never saw MILK10k | 0.219 | 0.077 | 0.096 | 0.553 |
| S5 similar-case vote | 15 nearest **training** lesions | **0.719** | **0.431** | 0.402 | **0.866** |

(test split, with dermoscopy available)

- The **original Model A transfers poorly** (21.9%): trained on phone photos from another
  dataset. Every learned rule gives it 0.03–0.05 weight without being told.
- Models **B–G** of the old system cannot run on MILK10k — they need PAD-UFES's clinical
  fields or text generated from them ([12_how_it_was_built.md](12_how_it_was_built.md) Part 4).
- The **similar-case specialist is the strongest single source.**

---

## 3. Fusion — seven rules, cross-fitted, picked on validation

`scripts/run_team.py`, `src/fusion/methods.py`. *Val* = out-of-fold macro-F1 on validation
(used to choose); everything else is test.

| Rule | Val | Acc. | **Macro-F1** | Bal. acc. | Cancer AUROC |
|---|---|---|---|---|---|
| **With dermoscopy (stage B)** | | | | | |
| Dempster–Shafer (heads only) | 0.361 | 0.690 | 0.376 | 0.378 | 0.856 |
| Simple average | 0.383 | 0.661 | 0.317 | 0.311 | 0.845 |
| Original formula Q·R·U (learned R) | 0.420 | 0.710 | 0.440 | 0.408 | 0.855 |
| Best head per case (highest R) | 0.397 | 0.717 | 0.409 | 0.388 | 0.825 |
| Iterative diagnosis-adaptive EM (v1) | 0.397 | 0.718 | **0.453** | **0.423** | 0.848 |
| Stacking (logistic regression) | 0.382 | 0.703 | 0.337 | 0.332 | 0.850 |
| **Learned gate, mixture (v2) ← picked** | **0.471** | **0.719** | 0.418 | 0.393 | 0.861 |
| Learned gate, product | 0.400 | 0.709 | 0.401 | 0.379 | **0.870** |
| *Oracle best head (uses the label)* | — | 0.806 | 0.533 | | |
| **Photo + details (stage A)** | | | | | |
| Dempster–Shafer (heads only) | 0.274 | 0.644 | 0.286 | 0.280 | 0.799 |
| Simple average | 0.221 | 0.564 | 0.241 | 0.240 | 0.779 |
| Original formula Q·R·U | 0.316 | 0.648 | 0.324 | 0.307 | **0.821** |
| Best head per case | 0.303 | 0.659 | 0.293 | 0.284 | 0.782 |
| Iterative EM (v1) | 0.302 | 0.640 | 0.292 | 0.281 | 0.778 |
| Stacking | 0.275 | 0.611 | 0.240 | 0.245 | 0.802 |
| **Learned gate, mixture (v2) ← picked** | **0.340** | **0.661** | **0.332** | **0.314** | 0.820 |
| Learned gate, product | 0.320 | 0.656 | 0.326 | 0.310 | 0.818 |
| *Oracle best head* | — | 0.738 | 0.408 | | |

**Read:**
- The learned gate is best on validation in both stages and best on test without
  dermoscopy. **With dermoscopy, v1 scores higher on test** (0.453 vs 0.418) — a validation
  winner is not always the test winner when rare classes drive macro-F1. We keep the
  pre-declared choice and report both.
- A **plain average** is worst: the gain is in *how* specialists are weighted.
- The **original formula**, once its reliability is learned, is a strong rule. In its old
  form, C and S were equal for every specialist and cancelled out of the weights.
- **Subset study** (all 15 / 31 combinations): no subset beats the full team by more than
  0.015 on validation — every specialist is either useful or ignored safely.
- The **oracle** ceiling (0.533) shows how much better per-case *selection* could be.
- **Where the weight goes (v2, test):** similar cases 0.77, dermoscopy 0.11, photo 0.09,
  Model A 0.03, details ≈0 (stage B). **What the gate listens to most:** track record for
  the suspected diagnosis, OOD distance, quality, learned reliability.
- **Calibrated confidence:** mean shown 0.742 vs actual accuracy 0.719 (stage B);
  0.668 vs 0.661 (stage A).

### The diagnosis-adaptive prior, measured

| Diagnosis | n | photo | dermoscopy | details |
|---|---|---|---|---|
| BCC | 398 | 0.82 | **0.93** | 0.92 |
| NV | 133 | **0.84** | 0.79 | 0.66 |
| SCCKA | 70 | 0.36 | **0.74** | 0.40 |
| BKL | 70 | 0.17 | **0.31** | 0.00 |
| MEL | 60 | **0.52** | 0.40 | 0.43 |
| AKIEC | 32 | **0.06** | 0.03 | 0.00 |

(Melanoma: 0.52 measured vs 0.96 typed into the old system.)

---

## 4. Devil's Advocate evidence profile

Per specialist and case: quality **Q**, learned reliability **R**, uncertainty **U**,
track record for the suspected diagnosis, confidence, margin, agreement, consistency
**C**, missingness **M**, and **OOD** distance (Mahalanobis, Ledoit–Wolf, validation p99).
In v2 all of them set the weights through a learned gate (`src/fusion/gating.py`).
OOD flags **1.0%** of test lesions.

---

## 5. Decision engine vs baselines (cost 5, learned-gate probabilities)

| Policy | Answered (HIGH) | Dermoscopy | Acc. | Bal. acc. | **Missed cancers** | False alarms | Mean cost |
|---|---|---|---|---|---|---|---|
| B1 always answer, photo only | 100% | 0% | 0.661 | 0.314 | 6.1% | 51.4% | **0.719** |
| B2 always take dermoscopy | 100% | 100% | 0.719 | 0.393 | 7.8% | 38.5% | 1.193 |
| B3 confidence threshold, same coverage | 54.4% | 0% | 0.811 | 0.342 | 3.5% | 45.1% | 0.910 |
| B3 + dermoscopy for everyone | 54.4% | 100% | **0.893** | **0.436** | **2.3%** | 33.8% | 1.321 |
| B5 dermoscopy for random patients | 100% | 26.3% | 0.667 | 0.311 | 6.6% | 51.4% | 0.865 |
| B5 dermoscopy for most uncertain | 100% | 26.3% | 0.704 | 0.360 | 6.0% | 45.7% | 0.787 |
| **D-GUARD (HIGH tier)** | **54.4%** | **26.3%** | 0.769 | 0.284 | **3.5%** | 53.0% | 1.074 |

- **Learn-then-Test certified** at both stages (λ̂ = 0.212, 0.365; α = 5%, δ = 10%).
- **Against us:** with these probabilities the confidence threshold is as safe and more
  accurate; always answering is cheapest; HIGH answers raise more false alarms (53%).

### The key finding — abstention borrows its safety from the fusion

The same policy on the two best fusion back-ends (both cross-fitted):

| Back-end | HIGH | HIGH acc. | **D-GUARD missed** | **Threshold missed (same coverage)** | Acquisition efficiency (vs most-uncertain) | AUGRC 1−u / softmax |
|---|---|---|---|---|---|---|
| Iterative EM (v1) | 57.2% | 0.762 | **3.3%** | **9.3%** | **+0.30** (−0.04) | **0.0874** / 0.0918 |
| Learned gate (v2) | 54.4% | 0.769 | **3.5%** | **3.5%** | +0.20 (+0.20) | 0.0852 / **0.0791** |

D-GUARD's miss rate hardly moves. The threshold's swings from 9.3% to 3.5%: **a confidence
threshold is exactly as safe as the ranking it is given.** The calibrated policy gives a
stable, certified miss rate whatever it is given; when the fusion is good, the simple
baseline catches up. The same holds for choosing whom to ask for dermoscopy. This is
reported as the paper's central negative result.

*(Earlier versions reported 3.0% vs 9.3% at 60.2% coverage. That used v1 fitted in-sample
on the calibration lesions; cross-fitting lowered HIGH coverage to 57.2% — see
[what_did_not_work.md](what_did_not_work.md) §34.)*

---

## 6. Cost of a missed cancer (learned gate)

| Miss cost | Answered | HIGH | HIGH accuracy | Missed (HIGH) | Threshold, same coverage |
|---|---|---|---|---|---|
| 2 | 89.1% | 59.2% | 0.758 | 4.3% | 3.9% |
| 3 | 89.1% | 57.1% | 0.768 | 4.0% | 3.8% |
| **5** | **89.5%** | **54.4%** | **0.769** | **3.5%** | **3.5%** |
| 10 | 89.5% | 50.2% | 0.787 | 2.3% | 3.3% |

---

## 7. Robustness — DERMA-STRESS (300 test lesions, stage A)

"Model" = without the pipeline's quality and missing-details gates; "Gated" = full system.

| Condition | ignorance u | HIGH (model) | Missed (model) | HIGH (gated) | Missed (gated) | Dermoscopy asked | Photo flagged |
|---|---|---|---|---|---|---|---|
| clean | 0.679 | 46% | 4.3% | 44% | 4.6% | 28% | 8% |
| **blur** | **0.605 ↓** | **52% ↑** | 1.9% | **0%** | — | 15% | **100%** |
| **dark** | 0.723 | 23% | **11.4%** | **0%** | — | 32% | **100%** |
| **noise** | 0.824 ↑ | 17% | 6.0% | **0%** | — | 48% | **100%** (grainy) |
| JPEG | 0.741 ↑ | 26% | 3.8% | 22% | **0.0%** | 31% | 24% |
| **occlusion** | 0.634 | 23% | 8.8% | **0%** | — | 16% | **100%** (covered) |
| **no details** | 0.754 ↑ | 45% | **17.6%** | **0%** — details requested | — | 32% | 8% |

**Findings**
1. **Ignorance alone catches none of the dangerous cases.** Blur even makes the model
   *more* confident.
2. **The pipeline's checks catch all of them:** blur, darkness, grain and a covered centre
   are flagged on 100% of damaged photos; a photo without details triggers a request.
   No damaged input gets a HIGH answer.
3. **The learned gate is brittle without details** (17.6% missed): it relies on the
   similar-case vote, whose embedding includes the details. Asking for them is necessary.
4. **Cost:** 8% of clean photos are flagged (was 6% before the grain check).

---

## 7a. Statistics — confidence intervals and significance

Bootstrap, 2000 resamples of the 787 test lesions (`scripts/analyze_milk.py`):

| Quantity | Value | 95% CI |
|---|---|---|
| Answered | 89.5% | 87.3–91.6% |
| HIGH-confidence share | 54.4% | 50.8–57.8% |
| HIGH accuracy | 76.9% | 72.7–80.8% |
| Missed cancers, HIGH | **3.5%** | 1.8–5.4% |
| Missed cancers, threshold (same coverage) | 3.5% | 1.9–5.3% |
| Missed cancers, learned rejector | 3.3% | 1.7–5.0% |
| Learned-gate macro-F1 (stage B) | 0.418 | 0.343–0.475 |

| Paired test | Difference | 95% CI | one-sided p |
|---|---|---|---|
| Missed: threshold − D-GUARD | 0.0 pts | −1.4 to +1.4 | 0.53 (n.s.) |
| Missed: learned rejector − D-GUARD | −0.2 pts | −1.7 to +1.2 | 0.63 (n.s.) |
| Macro-F1: learned gate − DS (stage B) | +0.042 | −0.033 to +0.106 | 0.17 (n.s.) |
| v2 − v1: missed | +0.2 pts | −0.4 to +0.8 | n.s. |
| v2 − v1: HIGH share | −2.8 pts | −5.0 to −0.6 | significant |

### Fairness — by skin tone

| Skin tone | n | HIGH share | HIGH accuracy | Missed (HIGH) |
|---|---|---|---|---|
| 1 | 14 | 14% | 50% | 0% — **too few to judge** |
| 2 | 67 | 34% | 83% | **8.7%** |
| 3 | 489 | 61% | 78% | 3.0% |
| 4 | 167 | 51% | 73% | 4.7% |
| 5 | 50 | 36% | 72% | 0.0% |

Lighter tones (1–2) fare worse and are under-represented. Reported, not fixed.

### Decision curve — net benefit of "refer as possibly malignant"

| Threshold | Refer everyone | Refer if P(malignant) ≥ t | D-GUARD rule |
|---|---|---|---|
| 0.1 | 0.706 | 0.697 | 0.693 |
| 0.3 | 0.622 | 0.626 | 0.625 |
| 0.5 | 0.471 | **0.555** | 0.502 |

---

## 7b. Distribution shift — PAD-UFES-20, no refitting

The full MILK10k-calibrated system on all 2,298 PAD-UFES-20 smartphone photos (Brazil,
6 classes mapped to MILK10k, age/sex/site only, no dermoscopy). The original PAD-UFES
Model A is switched off — it was trained on these images. `scripts/run_shift_pad.py`.

| | MILK10k test | PAD-UFES-20 |
|---|---|---|
| Mean ignorance | 0.676 | **0.748** |
| OOD-flagged / severe photo quality | 1% / 8% | **6% / 28%** |
| HIGH-confidence answers | 54% | **14%** |
| Dermoscopy requested (none exists) | 26% | 30% |
| Deferred | 10% | **20%** |
| Cancer-vs-benign AUROC | 0.799 | **0.807** |
| Accuracy, all cases | 0.644 | 0.382 |
| Accuracy, HIGH tier | 0.769 | 0.491 (confidence threshold: 0.657) |
| Missed cancers, HIGH tier | 3.5% | **0.0%** |

**Read:** the system *notices* the shift — every warning signal rises and confident answers
drop from 54% to 14% — and no cancer is missed among them. But the 11-class diagnosis does
not transfer: HIGH answers are 49% right, below a plain confidence threshold (66%). Safe,
not accurate, under shift.


## 7c. External test with dermoscopy — Derm7pt, no refitting

Derm7pt (Kawahara et al., 2019): 1,003 lesions from an Italian atlas, each with a clinical
photo **and** a dermoscopy image, so the whole loop runs as in the app — photo + details
first, dermoscopy only if the system asks. Diagnoses map onto 6 MILK10k classes (NV 575,
MEL 252, BKL 85, BCC 42, VASC 29, DF 20); sex and site only, no age.
`scripts/run_shift_derm7pt.py`.

| | MILK10k test | Derm7pt — D-GUARD flow | Derm7pt — always with dermoscopy |
|---|---|---|---|
| Cancer share | 72% | 29% | 29% |
| Dermoscopy requested | 26% | **55%** | 100% |
| Accuracy, all cases | 0.644 (photo) | 0.427 (photo only: 0.295) | 0.472 |
| Cancer-vs-benign AUROC | 0.856 | 0.730 | 0.769 |
| HIGH-confidence answers | 54% | **11%** | 17% |
| Deferred | 10% | 14% | 4% |
| Accuracy, HIGH tier | 0.769 | 0.566 (threshold: 0.745) | 0.655 |
| Missed cancers, HIGH tier | 3.5% | **3.8%** (threshold: 3.8%) | 3.5% |

**Read:**
- **Asking for dermoscopy works under shift:** the photo stage asked for it for 55% of
  lesions, and accuracy rose from 0.30 to 0.43.
- **Safety holds** (3.8% missed among HIGH answers, inside the 5% target) because the
  system answers confidently only 11% of the time — ignorance is high (0.80) and 29% of
  photos fail the quality check (old atlas scans).
- **The diagnosis does not transfer:** melanoma recall 33%, DF and VASC almost never right.
  A confidence threshold at the same coverage is more accurate. Same picture as PAD-UFES:
  safe, not accurate, on a new population.


---

## 8. Other measured facts

| | |
|---|---|
| Dermoscopy improves cancer-vs-benign AUROC | 0.799 → 0.856 (evidential heads) |
| ...and lowers ignorance | 0.676 → 0.439 |
| Local model = Kaggle model | test macro-F1 0.376 vs 0.379 |
| Heads-level H1 (all three inputs ≥ best pair) | **not supported**: 0.379 vs 0.403 |
| Post-hoc logit adjustment | helped val, **hurt test** — dropped |
| Rare classes (BEN_OTH, INF, MAL_OTH) | never predicted |

### Do the scribe's warning signs mean anything? (PAD-UFES-20 questionnaire)

MILK10k has no symptoms, so the scribe's rule was checked on PAD-UFES-20, where patients
answered *bleed / grew / changed / itch / hurt / raised* and every lesion has a diagnosis
(1,641 lesions; cancer = BCC, SCC or melanoma; `scripts/analyze_pad_symptoms.py`):

| Warning sign | Cancer if yes | Cancer if no | Relative risk (95% CI) |
|---|---|---|---|
| bleeding | 85.0% | 32.0% | 2.66 (2.43–2.91) |
| rapid growth | 52.7% | 17.5% | 3.00 (2.51–3.59) |
| change | 61.1% | 30.8% | 1.98 (1.67–2.35) |
| itching | 52.8% | 32.1% | 1.65 (1.45–1.87) |
| pain | 84.8% | 37.2% | 2.28 (2.09–2.48) |
| raised (added to the scribe after this) | 60.1% | 21.1% | 2.84 (2.43–3.34) |
| **any scribe warning sign** | **48.7%** | **12.4%** | **3.91 (2.77–5.52)** |

Every sign raises the cancer rate, so raising *urgency* on them is justified. Caveats:
questionnaire answers, not free text; a Brazilian referral population; and with
MILK10k's wider cancer definition (adding actinic keratosis) growth and "raised" no longer
point the same way. The signs change urgency only — never the diagnosis.

---

## 9. Seeds

One training seed so far (42). Each Kaggle seed run also redraws the lesion split, so
extra seeds measure training *and* split variation. When the runs are downloaded:
`python scripts/run_seeds.py runs/<seed7> runs/<seed2026>` → `outputs/seeds/summary.json`
(mean ± sd of every headline number).

---

## 10. Limits

One training seed; ImageNet ResNet50; validation fits every fusion and decision
parameter; a referral population (74% malignant); no centre column. See
[limitations.md](limitations.md) and [what_did_not_work.md](what_did_not_work.md).

---

## 11. Reproduce

See [12_how_it_was_built.md](12_how_it_was_built.md) Part 6 for the full command list.
