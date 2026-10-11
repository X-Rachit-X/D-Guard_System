# Results — Native Model on MILK10k

Every number here comes from a file in the repository and can be regenerated with the
commands in §8. Updated 2026-10-11. Single training seed (42) — see §7.

---

## 1. What was trained

| | |
|---|---|
| Data | MILK10k, 5,240 lesions, 11 classes, each lesion with a clinical photo **and** a dermoscopy image |
| Split | lesion-level, seeded: train 3,667 · validation 786 · test 787 |
| Model | ResNet50 (ImageNet) shared by both image types, separate **evidential head** per source (clinical, dermoscopy, metadata), Dempster–Shafer fusion |
| Training | 25 epochs on a Kaggle GPU, backbone frozen for the first 3, modality dropout 0.25, best epoch 23 by validation macro-F1 |
| Artifacts | `runs/milk10k_resnet50_seed42/` (`metrics.json`, `predictions_test.csv`, `manifest.json`; weights kept out of git) |

**The model running locally is the model that was trained on Kaggle.**
`scripts/milk_native_infer.py` rebuilds the identical split, loads the weights strictly and
reproduces test macro-F1 **0.3758** against Kaggle's **0.3794** (the gap is GPU mixed
precision vs CPU float32).

---

## 2. Classifier quality

| Metric (test, 787 lesions) | Value |
|---|---|
| Accuracy | 0.693 (BCC alone is 48.8% of test lesions) |
| Balanced accuracy | 0.379 |
| Macro-F1 | 0.379 |
| Macro AUROC (one-vs-rest) | 0.824 |
| **Malignant-vs-benign AUROC** | **0.799 photo + details → 0.856 with dermoscopy** |
| Adaptive ECE | 0.156 |

Per class:

| Class | n | F1 | | Class | n | F1 |
|---|---|---|---|---|---|---|
| BCC | 384 | **0.84** | | SCCKA | 69 | 0.57 |
| VASC | 8 | 0.88 | | MEL | 68 | 0.54 |
| NV | 91 | 0.72 | | AKIEC | 57 | 0.24 |
| BKL | 90 | 0.21 | | DF | 9 | 0.18 |
| BEN_OTH | 4 | 0.00 | | INF | 6 | 0.00 |
| MAL_OTH | 1 | 0.00 | | | | |

**Honest reading.** Good on the common classes, weak on the rare ones — BEN_OTH, INF and
MAL_OTH are never predicted. This is a ResNet50 trained for 25 epochs on 3,667 lesions; a
dermatology foundation encoder (DermLIP/PanDerm) is the expected fix and is the next
training task.

---

## 3. Does dermoscopy help? (the premise of the project)

Every MILK10k lesion has both images, so this is measured, not simulated:

| | macro-F1 | accuracy | mean ignorance `u` |
|---|---|---|---|
| photo + details | 0.286 | 0.644 | 0.676 |
| **+ dermoscopy** | **0.376** | **0.690** | **0.439** |

Two things matter here. Dermoscopy improves diagnosis **and** the model *knows* it has
more evidence: ignorance falls from 0.676 to 0.439. That second fact is what lets the
system decide **when** to ask for dermoscopy instead of asking for it every time.

---

## 4. H1 — does using every source beat the best pair?

From the trained model's availability ablation (`metrics.json`):

| Sources | macro-F1 |
|---|---|
| clinical only | 0.268 |
| metadata only | 0.245 |
| dermoscopy only | 0.396 |
| clinical + metadata | 0.290 |
| clinical + dermoscopy | 0.379 |
| **dermoscopy + metadata** | **0.403** ← best |
| all three | 0.379 |

**H1 fails, narrowly.** All three sources score 0.024 below the best pair. Evidence-aware
fusion reduced the damage a weaker source does (on PAD-UFES-20, naive concatenation lost
0.020 to text), but it did **not** remove it: the clinical photo still pulls the result
down. Reported in [what_did_not_work.md](what_did_not_work.md) §17.

---

## 5. The decision engine (the actual contribution)

Two-stage protocol, calibrated on the 786 validation lesions, scored once on the 787 test
lesions (`scripts/run_eade_milk.py`):

```
Stage A   photo + details          -> PROCEED / ACQUIRE dermoscopy / DEFER
Stage B   + the real dermoscopy    -> PROCEED / DEFER        (ACQUIRE cases only)
```

Configuration: state the most likely diagnosis (`label_rule=argmax`); a conformal set over
**benign / malignant** blocks false reassurance (`ambiguity_gate=reassurance`, LAC score,
class-conditional, α = 0.10); Learn-then-Test threshold on `1 − u` (α = 0.05, δ = 0.10);
conflict threshold = validation p90.

### 5.1 Main result (missed-cancer cost = 5, the balanced operating point)

| Policy | answered | dermoscopy | accuracy | **missed cancers** | false alarms | mean cost |
|---|---|---|---|---|---|---|
| B1 always answer, photo only | 100% | 0% | 0.644 | 5.2% | 52.4% | **0.703** |
| B2 always take dermoscopy | 100% | 100% | 0.690 | 6.5% | 44.7% | 1.187 |
| B3 max-softmax abstention, same coverage | 36.8% | 0% | 0.841 | 5.2% | 23.6% | 1.098 |
| B3 same, but dermoscopy for everyone | 36.8% | 100% | **0.931** | 2.1% | 26.2% | 1.517 |
| B5 dermoscopy for random cases, same rate | 100% | 6.4% | 0.651 | 5.2% | 51.4% | 0.726 |
| B5 dermoscopy for highest-entropy cases | 100% | 6.4% | 0.654 | 5.6% | 49.5% | 0.732 |
| **D-GUARD** | **36.8%** | **6.4%** | **0.838** | **1.7%** | 32.7% | 1.086 |

Missed cancers = malignant lesions called benign, among answered cases. False alarms =
benign lesions called malignant, among answered benign lesions.

**What this shows**

1. **3× fewer missed cancers than the standard baseline at the same coverage**
   (1.7% vs 5.2%), at **lower** mean cost (1.086 vs 1.098), with similar accuracy.
2. **Dermoscopy for 6.4% of patients, not 100%.** Asking everyone for dermoscopy (row 4)
   is more accurate but costs 40% more and needs a dermatoscope at every visit.
3. **The acquisition estimator picks the right patients.** Acquisition efficiency (1 =
   perfect, 0 = random): **+0.27** for D-GUARD's estimator vs **+0.01** for "most
   uncertain first" at cost 5; **+0.40** vs **−0.06** at cost 10.
4. **Native ignorance ranks as well as softmax** — AUGRC 0.0900 for `1 − u` vs 0.0893 for
   max-softmax. In bridge mode over the legacy models `u` clearly lost (0.163 vs 0.128);
   training real evidential heads closed that gap.
5. **The risk guarantee is now certifiable.** Learn-then-Test returned a threshold at both
   stages (λ̂ = 0.354 and 0.313). On PAD-UFES-20 with 432 calibration cases it could not.

**What it does not show — reported against us**

- **On mean cost alone, "always answer" wins** (0.703 vs 1.086). Each deferral costs 1.5
  and D-GUARD defers 63%. A deferral system is worth its price only if safety among the
  answered cases matters more than throughput — which is the premise, but it is a premise.
- **It answers alone only 37% of the time.** The rest get a tentative opinion and go to a
  doctor. That rate is limited by the classifier, not the policy: with a better encoder the
  same calibration would let more cases through.
- One training seed.

### 5.2 The cost of a missed cancer is a choice — here is what it changes

| missed-cancer cost | answered | missed cancers | max-softmax, same coverage | says "cancer" in |
|---|---|---|---|---|
| 2 | 40% | 4.1% | 5.1% | 80% of answers |
| 3 | 39% | 4.2% | 4.9% | 82% |
| **5** | **37%** | **1.7%** | **5.2%** | **86%** |
| 10 (pre-registered) | 32% | **0.0%** | 3.5% | **100%** |

At cost 10 the system never misses a cancer — but only because every answer it gives *is*
a cancer diagnosis. MILK10k is 73.6% malignant, so a system that only ever confirms cancers
can look perfect on this metric while never reassuring anyone. **Cost 5 is the balanced
point** and is what the demo uses. Choosing between them is a clinical judgement, not a
statistical one.

### 5.3 Two design errors this experiment caught

The first version of the gate **deferred 100% of test cases**. Two independent causes, both
fixed and kept as options so the comparison is reproducible:

| Problem | Effect | Fix |
|---|---|---|
| Non-randomised APS score on 2 classes | any lesion whose true class ranks second scores exactly 1.0 → quantile collapses to 1.0 → every set holds both outcomes | LAC score `1 − p(true)` (Sadinle et al., 2019) |
| Gate blocked **both** directions | "likely cancer, see a dermatologist" was blocked as hard as "probably harmless" | block only false reassurance (`ambiguity_gate=reassurance`) |

And one before that: stating the **lowest-cost** label (Chow's rule) with a 10:2 cost ratio
on a 73.6%-malignant population called **98.5%** of lesions malignant. The stated label is
now the most likely diagnosis; the cost matrix decides only *whether* it may be stated.

---

## 6. Which source to trust for which diagnosis

The old system used a hand-typed matrix of how much to trust each source per diagnosis
(e.g. images 0.96 for melanoma). Here it is **measured** on validation — how often each
source, on its own, names the true diagnosis:

| True class | n | clinical photo | dermoscopy | details |
|---|---|---|---|---|
| BCC | 398 | 0.82 | **0.93** | 0.92 |
| NV | 133 | **0.84** | 0.79 | 0.66 |
| SCCKA | 70 | 0.36 | **0.74** | 0.40 |
| BKL | 70 | 0.17 | **0.31** | 0.00 |
| MEL | 60 | **0.52** | 0.40 | 0.43 |
| AKIEC | 32 | **0.06** | 0.03 | 0.00 |
| VASC | 5 | 0.20 | **1.00** | 0.40 |

Melanoma: 0.52 measured vs 0.96 assumed. The doctor view shows the row for the system's
tentative diagnosis. It is **information for the clinician** and never changes a
probability.

---

## 7. Limits of these results

- **One seed.** Seeds 7 and 2026 are required by the protocol before any number is final.
- **ResNet50 / ImageNet.** The image encoder is the bottleneck; DermLIP is expected to
  raise every number in §2.
- **Validation used twice.** It picked the best epoch *and* calibrates the policy. A
  separate calibration split is the cleaner design (see `docs/prereg.yaml`).
- **Referral population.** 73.6% of test lesions are malignant; nothing here transfers to
  screening without recalibration.
- **No centre column**, so no centre-shift result.

---

## 8. Reproduce

```bash
python scripts/milk_native_infer.py                         # ~2.5 min on CPU
python scripts/run_eade_milk.py --missed-cost 5  --out outputs/eade_milk/cost_5
python scripts/run_eade_milk.py --missed-cost 10 --out outputs/eade_milk/cost_10
python scripts/demo_care.py                                  # the end-to-end story
streamlit run app/dguard_app.py                              # the app
```
