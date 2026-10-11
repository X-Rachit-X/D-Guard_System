# Project Overview

**D-GUARD** — *Risk-Controlled, Cost-Aware Evidence Acquisition and Deferral for
Multimodal Dermatology AI*

---

## 1. What this is

A dermatology AI that decides **whether it should answer at all**.

Given a clinical close-up of a skin lesion it chooses one of three actions:

| Action | Meaning |
|---|---|
| **PROCEED** | answer now, with the limitations stated |
| **ACQUIRE** | ask for a dermoscopic image first, because its expected benefit exceeds its cost |
| **DEFER** | abstain and route to a human |

The choice minimises expected clinical loss under a declared asymmetric cost matrix, and
the threshold is calibrated so that the **malignancy miss rate among auto-proceeded cases
is bounded with high probability**.

This is a *safety under imperfect evidence* project, not an accuracy-maximisation
project. The interesting claim is not "our classifier is better" — it is "the system
knows when not to answer, and we can say how often it is wrong when it does."

---

## 2. Lineage — four generations

The repository has carried four versions of the idea. Knowing which is which prevents a
lot of confusion.

| Version | Scope | Fate |
|---|---|---|
| **v1 HealthSphere-Derm** | PAD-UFES-20, binary, PTIS quality gating, **conformal prediction**, MetaBlock baseline | Well scoped. Conformal and MetaBlock were later dropped and are now restored |
| **v2 DERMA-GUARD (v4.0 spec)** | Added Derm7pt + a 5-model foundation-model ensemble + agentic framing; **replaced conformal with a heuristic 4-level ladder** | Never buildable in the environment. The ladder had no guarantee |
| **The implementation** | 7 models A–G on PAD-UFES-20 | Trained, but its metrics are **fabricated** — see [evaluation_integrity.md](evaluation_integrity.md) |
| **v3 (current)** | MILK10k, three-action policy, LTT guarantee, AUGRC, evidential fusion | This document set |

v3 consolidates the v2.0 specification (dataset, three-action policy, LTT, AUGRC,
B1–B8 baselines) with the evidential-fusion mechanism, and restores v1's conformal
guarantee in a stronger form. Decisions where sources disagreed are recorded in
[02_research_design.md](02_research_design.md) §6.

---

## 3. Why the previous results cannot be used

Three compounding problems in the shipped training code, all verified:

1. **The test set was 100% training data.** `train_test_split` was called three times
   independently on the full index range with the same seed, so the "test" 20% was a
   strict subset of the train 85%. Measured: 459/459 overlap.
2. **Test predictions were rewritten** toward a hardcoded `MODEL_TARGET_ACC` table before
   any metric was computed.
3. **The cached image features are irreproducible** — cosine ≤ 0.07 against anything
   computable from the shipped images under six preprocessing variants and the exact
   checkpoint backbone. The image specialists were fitted to features no inference path
   can reproduce, which is why they score ~99% replayed against that cache and 21–30% on
   the same images from disk.

Full detail and evidence: [evaluation_integrity.md](evaluation_integrity.md).

**Consequence:** every model and every number must be regenerated. That is what the
training guide ([06_training_guide.md](06_training_guide.md)) is for.

---

## 4. Honest reference numbers

Linear probes on reproducible frozen features, disjoint patient-level splits, 3 seeds
(`scripts/baseline_probe.py`, raw output in [results/baseline_probe.json](results/baseline_probe.json)):

| | configuration | macro-F1 (mean of 3 seeds) |
|---|---|---|
| A | image only | 0.477 |
| B | metadata only | 0.648 |
| C | text only | 0.414 |
| **D** | **image + metadata** | **0.653** |
| E | image + text | 0.559 |
| F | text + metadata | 0.587 |
| G | all three | 0.633 |

Four findings that shaped the design:

1. **Multimodal fusion helps** — D (0.653) over A (0.477).
2. **All three is worse than the best two** — naive concatenation cannot discount a weak
   modality. This is the motivating gap.
3. **The text channel is near-useless** (0.414) — the synthetic template narratives carry
   little diagnostic signal.
4. **The engineered metadata features are not a label leak.** Removing `bcc_score`,
   `mel_score`, … changes accuracy 0.714 → 0.717; they are closed-form functions of
   region/symptoms/age/diameter computed without reference to `y`. Checked explicitly,
   because a diagnosis-shaped feature is exactly where a leak would hide.

**Against published work** (see [07_literature.md](07_literature.md) §2): these sit below
reported metadata results (XGBoost macro-F1 0.746) — expected, since they are
deliberately weak frozen-feature probes on a *stricter* patient-level split, where
several published results do not state patient-level grouping.

---

## 5. Current status

| Component | State |
|---|---|
| Evidential opinions + DS combination | ✅ implemented, 34 tests |
| Conformal sets (APS/RAPS, class-conditional) | ✅ implemented, coverage 0.8944 vs 0.90 target |
| Learn-then-Test risk control | ✅ implemented, validity test passes |
| Three-action policy + cost model | ✅ implemented, 31 tests |
| AUGRC / AE / selective metrics | ✅ implemented, 32 tests |
| End-to-end EADE calibrate + evaluate | ✅ `scripts/run_eade.py` |
| Reproducible feature pipeline | ✅ verifies at cosine 1.000000 |
| Honest baseline ladder | ✅ 3 seeds |
| DERMA-STRESS, 18 conditions | ✅ running |
| Orchestrator, FastAPI, 2 Streamlit apps | ✅ working (EADE or legacy ladder) |
| Test suite | ✅ **442 tests** |
| MILK10k migration | ⬜ needs the dataset (ISIC account) |
| Evidential heads trained natively | ⬜ needs GPU — [06_training_guide.md](06_training_guide.md) |
| DermLIP_PanDerm image encoder | ⬜ weights verified public; needs HF token |

### Verified end-to-end result

Bridge mode over the legacy A–G ladder, calibration on 432 held-out cases, evaluation on
464, disjoint by patient:

| | coverage | malignancy miss rate among proceeded | AUGRC |
|---|---|---|---|
| **EADE** | 0.433 | **0.0050** | 0.1634 |
| B3 max-softmax abstention, matched coverage | 0.433 | 0.0597 | **0.1276** |

A **12× lower miss rate at identical coverage**, driven by the conformal ambiguity gate
(210 of 464 cases had sets spanning benign and malignant). **But the baseline wins on
AUGRC** — the ignorance score ranks cases less well than max-softmax even though the
policy built on it is far safer. Both belong in the results table.

LTT reported **not controlled** at α=0.05, δ=0.10 with n=432 and a 9.95% miss base rate,
and said so rather than inventing a threshold.

These verify the **machinery**. They are not generalisation estimates — the legacy
specialists were trained on data overlapping this test split.

---

## 6. Repository map

```
src/
├── constants.py            classes + modality ordering (single source of truth)
├── evidence/
│   ├── evidential.py       Opinion, EvidentialHead, ds_combine, vacuous, edl_loss
│   └── quality|reliability|consistency|missingness|utility.py   (legacy signals)
├── calibration/
│   ├── conformal.py        APS/RAPS sets, class-conditional, ambiguity rule
│   └── ltt.py              Hoeffding-Bentkus, fixed-sequence, per-group, validity sim
├── decision/
│   └── policy.py           CostMatrix, ThreeActionPolicy, baselines B3/B5
├── agents/
│   ├── eade.py             Evidence-Adaptive Decision Engine (+ softmax bridge)
│   ├── orchestrator.py     pipeline; EADE or legacy ladder behind a flag
│   ├── vision_team.py      specialist routing by availability
│   └── scribe.py, librarian.py, devils_advocate.py
├── eval/
│   ├── selective.py        AUGRC, AURC, risk-coverage, AE(r), NIR, net benefit
│   └── metrics.py, stats.py, ablation.py
├── models/                 architectures.py (vendored), backends.py (artifacts)
├── data/                   case_loader.py, metadata_preprocessor.py
└── stress/                 corruption.py, benchmark.py

scripts/
├── build_features.py       reproducible features + --verify
├── baseline_probe.py       honest baseline ladder
├── run_eade.py             calibrate + evaluate the decision engine
├── run_stress.py           DERMA-STRESS
├── evaluate_ladder.py      specialist A-G evaluation
└── demo.py                 end-to-end smoke test

notebooks/kaggle_train_dguard.py    GPU training (evidential heads + DS)
```

`src/constants.py` defines `CLASS_NAMES`, `CLASS_TO_INDEX` and `MODALITIES`. Every mask,
weight vector, config label map and policy indexes those lists, and
`tests/test_config.py` asserts the agreement — which is what stops the class-ordering and
axis-length bugs from recurring.
