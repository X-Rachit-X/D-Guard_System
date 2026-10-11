# Results

Every number here is produced by a script in the repository (§9). Dataset: MILK10k.
Model: ResNet50 evidential model trained on Kaggle (seed 42). Test split: 787 held-out
lesions, scored once. Final configuration: missed-cancer cost 5, minimal-deferral mode,
quality gate on.

---

## 1. Headline

| What patients get | Share | Accuracy (11 classes) | Missed cancers |
|---|---|---|---|
| ✅ **Answer — HIGH confidence** | **35.3%** | **83.8%** | **1.8%** |
| 📷 Asked for dermoscopy first | 6.4% | (then answered) | |
| ⚠️ Answer — LOW confidence (doctor confirms) | 55.0% | 51.7% · **top-3: 82.2%** | 6.2% |
| 🧑‍⚕️ Deferred — inputs contradict | 9.7% | — | — |
| **Answered in total** | **90.3%** | | |

**Against the standard approach at the same coverage (35.3%):** missed cancers **1.8% vs
4.7%**, balanced accuracy **0.355 vs 0.300**, accuracy 83.8% vs 84.2%, mean cost 1.105 vs
1.109.

---

## 2. The trained model

| Test metric | Value |
|---|---|
| Accuracy | 0.693 (BCC alone is 48.8% of test lesions) |
| Macro-F1 / balanced accuracy | 0.379 / 0.379 |
| Macro AUROC | 0.824 |
| Malignant-vs-benign AUROC | 0.799 photo + details → **0.856 with dermoscopy** |

Runs locally and matches Kaggle (macro-F1 0.376 vs 0.379; the gap is GPU mixed precision).

| Class | n | F1 | | Class | n | F1 |
|---|---|---|---|---|---|---|
| BCC | 384 | 0.84 | | SCCKA | 69 | 0.57 |
| VASC | 8 | 0.88 | | MEL | 68 | 0.54 |
| NV | 91 | 0.72 | | AKIEC | 57 | 0.24 |
| BKL | 90 | 0.21 | | DF | 9 | 0.18 |
| BEN_OTH / INF / MAL_OTH | 4 / 6 / 1 | 0.00 | | | | |

Strong on common classes; the three rarest are never predicted.

---

## 3. Dermoscopy: does it help, and does the model know?

| | macro-F1 | accuracy | mean ignorance `u` |
|---|---|---|---|
| photo + details | 0.286 | 0.644 | 0.676 |
| **+ dermoscopy** | **0.376** | **0.690** | **0.439** |

It helps, and the model registers it: ignorance drops by a third. That is what lets the
system ask for dermoscopy **selectively**:

| Who gets dermoscopy (same 6.4% rate) | Acquisition efficiency (0 = random, 1 = perfect) |
|---|---|
| **D-GUARD's learned estimator** | **+0.27** (cost 10: **+0.40**) |
| Most uncertain patients first | +0.01 (cost 10: −0.06) |

---

## 4. Decision engine vs baselines (cost 5)

| Policy | Answered (HIGH) | Dermoscopy | Acc. | Bal. acc. | **Missed cancers** | False alarms | Mean cost |
|---|---|---|---|---|---|---|---|
| B1 always answer, photo only | 100% | 0% | 0.644 | 0.280 | 5.2% | 52.4% | **0.703** |
| B2 always take dermoscopy | 100% | 100% | 0.690 | 0.378 | 6.5% | 44.7% | 1.187 |
| B3 confidence threshold, same coverage | 35.3% | 0% | 0.842 | 0.300 | 4.7% | 24.1% | 1.109 |
| B3 + dermoscopy for everyone | 35.3% | 100% | **0.935** | **0.587** | 2.2% | 22.5% | 1.535 |
| B5 dermoscopy for random patients | 100% | 6.4% | 0.651 | 0.286 | 5.2% | 51.4% | 0.726 |
| B5 dermoscopy for most uncertain | 100% | 6.4% | 0.654 | 0.285 | 5.6% | 49.5% | 0.732 |
| **D-GUARD (HIGH tier)** | **35.3%** | **6.4%** | 0.838 | **0.355** | **1.8%** | 34.8% | 1.105 |

Read: at the same coverage, D-GUARD misses well under half as many cancers as a confidence
threshold, at slightly lower cost, while using dermoscopy for 6% of patients. Dermoscopy for
everyone is more accurate but 39% more expensive and needs a dermatoscope at every visit.

**Also:**
- **Learn-then-Test certified** a threshold at both stages (λ̂ = 0.354, 0.313).
- **Native ignorance ranks errors as well as softmax** — AUGRC 0.0900 vs 0.0893.
- **Against us:** on *mean* cost, always answering (0.703) is cheapest — every case needing a
  doctor is charged 1.5. D-GUARD's value is the safety of its confident answers.

---

## 5. The cost of a missed cancer

| Miss cost | Answered | HIGH | Missed (HIGH) | Baseline, same coverage | Answers that say "cancer" |
|---|---|---|---|---|---|
| 2 | 89.3% | 38.1% | 4.3% | 5.0% | 81% |
| 3 | 90.2% | 37.5% | 4.4% | 5.1% | 83% |
| **5** | **90.3%** | **35.3%** | **1.8%** | **4.7%** | 87% |
| 10 | 90.5% | 31.3% | 0.0% | 2.4% | 100% |

At cost 10 no cancer is missed — because every confident answer *is* a cancer diagnosis
(MILK10k is 73.6% malignant). Cost 5 is the balanced point and what the system uses.

---

## 6. Robustness — DERMA-STRESS (300 test lesions, stage A)

Each condition degrades the patient's photo or withholds details; nothing is retrained.

| Condition | ignorance `u` | HIGH (model only) | Missed in HIGH (model only) | HIGH (with quality gate) | Missed in HIGH (gate) | Answered | Quality-flagged |
|---|---|---|---|---|---|---|---|
| clean | 0.679 | 35% | 2.9% | 33% | 3.0% | 85% | 6% |
| **blur** | **0.605** ↓ | **46%** ↑ | 0.7% | **0%** | — | 79% | **100%** |
| **dark** | 0.723 | 11% | **12.5%** | **0%** | — | 70% | **100%** |
| noise | **0.821** ↑ | 6% | 0.0% | 6% | 0.0% | 89% | 2% |
| JPEG q=8 | 0.741 ↑ | 17% | 4.0% | 14% | **0.0%** | 80% | 18% |
| 35% occluded | 0.634 | 7% | 4.5% | 5% | **0.0%** | 46% | 51% |
| details missing | **0.754** ↑ | 33% | 1.0% | 32% | 1.0% | 78% | 6% |

**Findings**

1. **Ignorance catches some degradations, not all.** Noise, JPEG and missing details raise
   `u`, and the system answers confidently less often. **Blur lowers `u`** — the model grows
   *more* confident on blurred photos (HIGH 35% → 46%).
2. **The quality agent catches what the model misses.** It flags 100% of blurred and dark
   photos; with the gate, those produce **no** HIGH answers, and **no condition raises the
   HIGH-tier miss rate above the clean level**.
3. **Missing details → more dermoscopy requests** (7% → 22%): the system asks for evidence
   when it has less.
4. **Occlusion → more deferrals** (8% → 46%): a covered lesion makes the inputs disagree.

(The clean miss rate here, 3.0%, is on a 300-lesion subset; the full test set gives 1.8%.)

---

## 7. Which input to trust, per diagnosis (measured)

The old system typed this table by hand. Measured on validation — how often each input
alone names the true diagnosis:

| Diagnosis | n | photo | dermoscopy | details |
|---|---|---|---|---|
| BCC | 398 | 0.82 | **0.93** | 0.92 |
| NV | 133 | **0.84** | 0.79 | 0.66 |
| SCCKA | 70 | 0.36 | **0.74** | 0.40 |
| BKL | 70 | 0.17 | **0.31** | 0.00 |
| MEL | 60 | **0.52** | 0.40 | 0.43 |
| AKIEC | 32 | **0.06** | 0.03 | 0.00 |
| VASC | 5 | 0.20 | **1.00** | 0.40 |

Melanoma: 0.52 measured vs 0.96 assumed by the old system. Shown to the doctor for the
system's diagnosis; never used to change a probability.

---

## 8. Fusion ablation (H1)

| Inputs | macro-F1 |
|---|---|
| photo / dermoscopy / details alone | 0.268 / 0.396 / 0.245 |
| photo + details | 0.290 |
| photo + dermoscopy | 0.379 |
| **dermoscopy + details** | **0.403** |
| all three | 0.379 |

**H1 is not supported**: all three inputs score 0.024 below the best pair. Evidence-aware
fusion limits but does not remove the drag of the weaker clinical photo.

---

## 9. Reproduce

```bash
python scripts/milk_native_infer.py                                       # ~2.5 min
python scripts/run_eade_milk.py --missed-cost 5 --out outputs/eade_milk/cost_5
python scripts/run_eade_milk.py --missed-cost 10 --out outputs/eade_milk/cost_10
python scripts/build_milk_index.py                                        # ~3 min
python scripts/run_stress_milk.py                                         # ~3 min
```

Outputs: `outputs/milk_native/summary.json`, `outputs/eade_milk/cost_*/report.json`,
`outputs/stress_milk/report.json`, `runs/milk10k_resnet50_seed42/metrics.json`.

## 10. Limits of these numbers

One training seed; ImageNet ResNet50; validation used both for epoch selection and
calibration; a referral population (74% malignant); no centre column, so no centre-shift
test. See [limitations.md](limitations.md) and [what_did_not_work.md](what_did_not_work.md).
