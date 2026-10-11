# Results

Every number is produced by a script in the repository (§10). Dataset: MILK10k, 11 classes.
Test split: 787 held-out lesions, scored once. Everything else — fusion weights, learned
reliabilities, calibration, gates, thresholds — was fitted on the 786 validation lesions.
Operating point: missed-cancer cost 5, minimal-deferral mode, quality and OOD gates on.

---

## 1. Headline

| What patients get | Share | Accuracy (11 classes) | Missed cancers |
|---|---|---|---|
| ✅ **Answer — HIGH confidence** | **60.2%** | **76.2%** | **3.0%** |
| 📷 Asked for dermoscopy first | 27.3% | (then answered) | |
| ⚠️ Answer — LOW confidence (doctor confirms) | 29.4% | 51.5% · **top-3: 81.8%** | 14.7% |
| 🧑‍⚕️ Deferred — evidence contradicts itself | 10.4% | — | — |
| **Answered in total** | **89.6%** | | |

**Against the standard approach at the same 60.2% coverage:** missed cancers **3.0% vs 9.3%**
(3.1× fewer), accuracy 76.2% vs 72.4%, mean cost 1.013 vs 1.037.

**Fusion:** the old design's iterative diagnosis-adaptive fusion over the full specialist
team raises 11-class macro-F1 from **0.376 → 0.453** (+20% relative) over evidential
fusion alone.

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

- The **original Model A transfers poorly** to MILK10k (21.9%): it was trained on
  smartphone photos from a different dataset. The evidence manager learns this on its
  own and gives it a weight of **0.02–0.09**.
- The **similar-case specialist is the strongest single source.**

---

## 3. Fusion — the old design, rebuilt with learned parts

| Method | Acc. | **Macro-F1** | Bal. acc. | Cancer AUROC |
|---|---|---|---|---|
| **With dermoscopy (stage B)** | | | | |
| Dempster–Shafer (heads only) | 0.690 | 0.376 | 0.378 | **0.856** |
| Simple average (whole team) | 0.661 | 0.317 | 0.311 | 0.845 |
| Adaptive fusion (heads only) | 0.704 | 0.394 | 0.381 | 0.844 |
| **Adaptive fusion (whole team)** | **0.718** | **0.453** | **0.423** | 0.848 |
| **Photo + details (stage A)** | | | | |
| Dempster–Shafer (heads only) | 0.644 | 0.286 | 0.280 | **0.799** |
| Simple average (whole team) | 0.564 | 0.241 | 0.240 | 0.779 |
| Adaptive fusion (heads only) | **0.645** | 0.269 | 0.269 | 0.794 |
| **Adaptive fusion (whole team)** | 0.640 | **0.292** | **0.281** | 0.778 |

**Read:**
- A *simple average* of the same specialists is **worse** than Dempster–Shafer — the gain
  comes from evidence-aware weighting, not from having more models.
- The gain is large with dermoscopy (+0.077 macro-F1) and small without it.
- Cancer-vs-benign AUROC is slightly lower than Dempster–Shafer (0.848 vs 0.856).
- Fitted on validation: stage B `a=4, b=2, g=0.5, ρ=0.55, T=0.55`; mean weights on test —
  dermoscopy 0.29, similar cases 0.30, details 0.22, photo 0.17, Model A 0.02.
- Converges fast: 38% of cases need no refinement, 62% need one step.

### The diagnosis-adaptive prior, measured

The old system typed in "how much to trust each source for each diagnosis". It is now
measured on validation (how often each head alone names the true class):

| Diagnosis | n | photo | dermoscopy | details |
|---|---|---|---|---|
| BCC | 398 | 0.82 | **0.93** | 0.92 |
| NV | 133 | **0.84** | 0.79 | 0.66 |
| SCCKA | 70 | 0.36 | **0.74** | 0.40 |
| BKL | 70 | 0.17 | **0.31** | 0.00 |
| MEL | 60 | **0.52** | 0.40 | 0.43 |
| AKIEC | 32 | **0.06** | 0.03 | 0.00 |
| VASC | 5 | 0.20 | **1.00** | 0.40 |

(Melanoma: 0.52 measured vs 0.96 assumed by the old system.)

### Calibrated confidence

The raw fused probability of the top class is spread over 11 classes and always *looks*
low. The app shows the **isotonic-calibrated probability that the stated diagnosis is
correct** instead: on test, mean shown 0.770 vs actual accuracy 0.718 with dermoscopy;
0.658 vs 0.640 without.

---

## 4. Devil's Advocate evidence profile

Per specialist and case: quality **Q**, learned reliability **R** (logistic regression on
validation), uncertainty **U**, consistency **C** (Jensen–Shannon), missingness **M**, and
**OOD** (Mahalanobis to nearest class, Ledoit–Wolf covariance, validation p99). On test,
OOD flags **1.0%** of lesions; the OOD gate downgraded 2 HIGH answers and the quality gate
27.

---

## 5. Decision engine vs baselines (cost 5, team probabilities)

| Policy | Answered (HIGH) | Dermoscopy | Acc. | Bal. acc. | **Missed cancers** | False alarms | Mean cost |
|---|---|---|---|---|---|---|---|
| B1 always answer, photo only | 100% | 0% | 0.640 | 0.281 | 8.1% | 51.4% | **0.821** |
| B2 always take dermoscopy | 100% | 100% | 0.718 | 0.423 | 6.7% | 41.3% | 1.161 |
| B3 confidence threshold, same coverage | 60.2% | 0% | 0.724 | 0.308 | 9.3% | 38.1% | 1.037 |
| B3 + dermoscopy for everyone | 60.2% | 100% | **0.861** | **0.455** | 3.8% | 29.3% | 1.309 |
| B5 dermoscopy for random patients | 100% | 27.3% | 0.656 | 0.312 | 7.8% | 51.0% | 0.926 |
| B5 dermoscopy for most uncertain | 100% | 27.3% | 0.680 | 0.358 | 8.8% | 44.2% | 0.924 |
| **D-GUARD (HIGH tier)** | **60.2%** | **27.3%** | 0.762 | 0.312 | **3.0%** | 53.1% | 1.013 |

- **3.1× fewer missed cancers** than a confidence threshold at the same coverage, at higher
  accuracy and lower cost.
- **Acquisition efficiency +0.33** (most-uncertain-first: −0.03).
- **Learn-then-Test certified** at both stages (λ̂ = 0.193, 0.254).
- **Ignorance now ranks errors better than softmax** (AUGRC 0.0874 vs 0.0918).
- **Against us:** always answering is cheapest on mean cost; HIGH answers raise more false
  alarms (53.1%) because the 5:2 cost matrix makes "possibly cancer" the safer statement.

### Ablation — what the team adds

| Policy driven by | HIGH share | HIGH accuracy | Missed (HIGH) | Baseline at same coverage |
|---|---|---|---|---|
| Dempster–Shafer of the 3 heads | 35.3% | 83.8% | 1.8% | 4.7% |
| **Adaptive fusion of the 5 specialists** | **60.2%** | 76.2% | 3.0% | 9.3% |

The team nearly **doubles** the share of confident answers while keeping missed cancers at
a third of the matched baseline.

---

## 6. Cost of a missed cancer

| Miss cost | Answered | HIGH | HIGH accuracy | Missed (HIGH) | Baseline, same coverage |
|---|---|---|---|---|---|
| 2 | 89.6% | 65.1% | 0.754 | 3.9% | 8.6% |
| 3 | 89.7% | 63.8% | 0.753 | 4.0% | 8.8% |
| **5** | **89.6%** | **60.2%** | **0.762** | **3.0%** | **9.3%** |
| 10 | 90.1% | 55.0% | 0.774 | 2.3% | 9.5% |

---

## 7. Robustness — DERMA-STRESS (300 test lesions, stage A, full system)

| Condition | ignorance u | HIGH (no gate) | Missed (no gate) | HIGH (gate) | Missed (gate) | Dermoscopy asked | Quality-flagged |
|---|---|---|---|---|---|---|---|
| clean | 0.679 | 49% | 4.1% | 46% | 4.3% | 28% | 6% |
| **blur** | **0.605 ↓** | **53% ↑** | 1.9% | **0%** | — | 14% | **100%** |
| **dark** | 0.723 | 29% | **13.8%** | **0%** | — | 28% | **100%** |
| noise | **0.821 ↑** | 19% | 5.3% | 18% | 5.5% | **51%** | 2% |
| JPEG | 0.741 ↑ | 29% | 4.6% | 25% | **1.4%** | 42% | 18% |
| **occlusion** | 0.634 | 29% | 9.3% | 14% | **9.3%** | 9% | 51% |
| no details | 0.754 ↑ | 49% | 6.1% | 47% | **6.4%** | 42% | 6% |

**Findings**
1. **Blur fools the evidential model** — ignorance falls and confident answers rise. The
   quality gate removes them all; same for dark photos (which missed 13.8% ungated).
2. **Noise, JPEG and missing details raise ignorance** and the system asks for dermoscopy
   far more (28% → 42–51%).
3. **Remaining failures:** occluded lesions (9.3% missed in the remaining HIGH answers) and
   missing details (6.4% vs 4.3%). Neither signal fully registers these — reported as open.

---

## 8. Other measured facts

| | |
|---|---|
| Dermoscopy improves cancer-vs-benign AUROC | 0.799 → 0.856 (evidential heads) |
| ...and lowers ignorance | 0.676 → 0.439 |
| Local model = Kaggle model | test macro-F1 0.376 vs 0.379 |
| Heads-level H1 (all three inputs ≥ best pair) | **not supported**: 0.379 vs 0.403 |
| Post-hoc logit adjustment | helped val (0.36→0.44), **hurt test** (0.376→0.359) — dropped |
| Rare classes (BEN_OTH, INF, MAL_OTH) | never predicted |

---

## 9. Limits

One training seed; ImageNet ResNet50; validation fits every fusion and decision
parameter; a referral population (74% malignant); no centre column. See
[limitations.md](limitations.md) and [what_did_not_work.md](what_did_not_work.md).

---

## 10. Reproduce

```bash
python scripts/milk_native_infer.py          # evidential heads on val/test   (~2.5 min)
python scripts/build_specialists.py          # all 5 specialists, embeddings  (~15 min)
python scripts/run_team.py                   # Devil's Advocate + adaptive fusion (~1 min)
for c in 2 3 5 10; do
  python scripts/run_eade_milk.py --native outputs/milk_team --missed-cost $c \
         --out outputs/eade_team/cost_$c
done
python scripts/build_milk_index.py           # similar-case index for the app (~3 min)
python scripts/run_stress_milk.py --out outputs/stress_milk/report_team.json   # (~10 min)
```
