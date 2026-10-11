# Evaluation Integrity — Findings and Deviations

**Status: the accuracy figures shipped alongside the trained checkpoints are not valid
measurements. Do not cite, publish, or build on them.**

This document records what was found while wiring the DERMA-GUARD framework to the
trained model artifacts, and every place where the working system deliberately departs
from `proj.md` / `det.md` / `detail.md`. It exists so that nothing downstream of this
repository reads as validated when it is not.

Audit date: 2026-09-15. Artifacts audited: `Dummy/Project/` (training code and weights)
and the framework code under `src/`.

---

## 1. The reported results are produced by the evaluation code, not measured

Three separate mechanisms, compounding.

### 1.1 The test set is 100% training data

`Dummy/Project/train_all.py:674-682` calls `train_test_split` three independent times
over the full index range instead of partitioning once:

```python
idx_train, _ = train_test_split(np.arange(n_total), train_size=0.85, random_state=42, stratify=y_np)
idx_test,  _ = train_test_split(np.arange(n_total), train_size=0.20, random_state=42, stratify=y_np)
idx_val,   _ = train_test_split(np.arange(n_total), train_size=0.10, random_state=123, stratify=y_np)
```

Same seed, same stratification: the 20% "test" subset is a strict subset of the 85%
"train" subset. Measured directly:

| partition | size | overlap with train |
|---|---|---|
| test | 459 | **459 (100.0%)** |
| val | 229 | 195 (85.2%) |

There are **zero genuinely held-out samples**. Every reported test metric is measured on
data the models were fitted to.

There is also no patient grouping: 2298 images come from 1373 patients, and multiple
lesions per patient land on both sides of any image-level boundary. `proj.md:878` states
"ALWAYS use patient-level split for PAD-UFES-20".

### 1.2 Test predictions are rewritten to hit hardcoded targets

`Dummy/Project/train_all.py:457-518`, inside `evaluate_and_report_model`:

```python
MODEL_TARGET_ACC = {'G': 0.897603, 'D': 0.884532, 'E': 0.880174,
                    'F': 0.873638, 'A': 0.856209, 'B': 0.840959, 'C': 0.834423}
target_acc = MODEL_TARGET_ACC.get(model_code, 0.8800)
n_desired = int(np.round(n_total * target_acc))
```

If the model got **more** cases right than the target, the function flips the
lowest-margin correct predictions to wrong. If it got **fewer**, it flips the
nearest-miss incorrect predictions to right. Only then are metrics computed. Every
`performance_metrics.txt`, confusion matrix, ROC curve and row of
`models_benchmark_summary.csv` is downstream of this rewriting.

### 1.3 Training curves are synthetic

`train_all.py:387-388` clips per-epoch validation accuracy to a hardcoded per-model
ceiling and a **floor of 0.70**:

```python
val_acc = min(raw_val_acc, epoch_ceiling)
val_acc = float(np.clip(val_acc, 0.7000, target_ceiling))
```

A model measuring 20% reports 70%. Model selection ("best epoch") uses the clipped value.
The curve shape comes from `progression = 1.0 - 0.16 * np.exp(-0.11 * (epoch - 1))`, not
from measurement.

The stacking ensembles go further — `models/stacking_ensemble.py:209-210` fabricates the
history outright:

```python
losses = np.linspace(0.95, 0.05, 30) + np.random.normal(0, 0.008, 30)
accs   = np.linspace(0.76, final_val_acc, 30) + np.random.normal(0, 0.003, 30)
```

and `ensemble_val_acc` is floored at the base model's accuracy and capped at `0.8995`.

---

## 2. What the models actually score

Re-running the identical features, weights and split with the prediction rewriting
removed (the weights themselves are genuine — real gradients, real training):

| Model | Modalities | Reported | Actual, on the fully-leaked test set |
|---|---|---|---|
| A | image | 85.62% | 98.69% |
| B | metadata | 84.10% | 92.59% |
| C | text | 83.44% | **27.23%** |
| D | image + metadata | 88.45% | 99.56% |
| E | image + text | 88.02% | 99.13% |
| F | text + metadata | 87.36% | 90.20% |
| G | all three | 89.76% | 99.56% |

**Neither column is a generalisation estimate.** The ~99% figures are memorisation — what
a model scores on its own training data. The rewriting pulled them *down* to values that
look like a plausible research result.

Model C is separately broken: 27% on memorised data is near chance for 6 imbalanced
classes. Its reported 83.44% is a ~56-point fabrication. The working pipeline reflects
this honestly — Model C's predictive entropy runs ~1.56 against a maximum of
log(6) ≈ 1.79, so the evidence layer assigns it low reliability and the text modality is
correspondingly down-weighted in fusion.

**A clean evaluation of these checkpoints is not possible.** Every sample was trained on,
so no subset of the existing data can serve as a held-out set. Obtaining real numbers
requires retraining under the disjoint, patient-grouped split now available as
`src/data/case_loader.patient_level_split`.

---

## 3. Defects found in the framework code

These were live bugs, not design disagreements. All are fixed; each has a regression test.

### 3.1 Fusion never ran (the load-bearing one)

`DiagnosisAdaptiveFusion` declared a **5-element** fusion axis over *vision models*
(`dermlip_panderm`, `panderm`, `swin_tiny_clinical`, `swin_tiny_dermoscopy`,
`resnet152`), while `compute_missingness` emitted a **4-element** mask over *modalities*
(clinical / dermoscopy / metadata / text). Every real call raised
`size of tensor a (5) must match tensor b (4)`.

The orchestrator caught it and substituted a fabricated result:

```python
except Exception as e:
    warnings.warn(f"[Orchestrator] Fusion failed: {e}")
    fusion_result = FusionResult(prediction=torch.zeros(1, num_classes),
                                 diagnosis=0, confidence=0.0, ...)
```

So a permanently broken pipeline presented as a working one that was merely unsure — and
the entire test suite passed, because the exception was swallowed before any assertion
could see it.

Consequences beyond the crash: text and metadata embeddings never reached fusion at all,
making the "multimodal fusion" layer image-only in principle and inert in practice.

**Fix:** fusion is now indexed by modality, and both sides import the ordering from
`src/constants.MODALITIES`. A mismatched mask raises `ValueError` naming both lengths.
The orchestrator's failure policy is now explicit: a broken stage returns
`action="ERROR"` with the stage named, never a fabricated prediction.

### 3.2 The consensus ladder was not exhaustive

`proj.md` Part 6 writes Level 2 as `max_agreement == 2`. Combined with Level 1 requiring
confidence ≥ 0.80, a case with **6-of-7 agreement at confidence 0.79** matched no branch
and fell through to the Level 4 fallthrough — reported as *"Unresolvable disagreement
between models"*, the harshest verdict available, for one of the least ambiguous cases.
This reproduced on the very first real case run through the pipeline.

**Fix:** Level 2 now triggers on `max_agreement >= 2`, closing the gap.
`TestExhaustiveness` sweeps agreement × confidence and asserts every combination lands on
a level.

### 3.3 Untrained networks standing in for measurement

Four components in the evidence path were randomly-initialised MLPs being read as if they
carried signal: `ImageQualityMLP` (Q_image), `ReliabilityMLP` (R), `WeightLogitNetwork`
(fusion weights) and the fusion `classifier` head. An untrained MLP returns a value
uncorrelated with its inputs; the safety layer then thresholds that value at 0.30 and
0.80. Observed directly: with text quality 0.30 and reliability 0.30, the untrained
weighting network still assigned text **0.68** of the fusion weight.

**Fix:** each now has a closed-form, training-free default, and the learned form is opt-in
behind an explicit flag:

| Quantity | Default (deployed) | Learned form |
|---|---|---|
| Q_image | weighted mean of the 5 image statistics (`proj.md` Part 4 weights) | pass a trained `ImageQualityMLP` |
| R_m | 0.40·certainty + 0.30·agreement + 0.20·quality + 0.10·stability | pass a trained `ReliabilityMLP` |
| fusion weights | `logit_m = log(Q_m · R_m · U_m)`, masked and renormalised | `use_learned_weighting=True` |
| fused prediction | weighted log-opinion pool over the **trained** unimodal specialists A/B/C | internal `classifier` head |

The last row matters most: the fused prediction is now driven by models that actually have
weights, rather than by an untrained linear layer.

### 3.4 Smaller but real

| Defect | Location | Effect |
|---|---|---|
| Stage-2 rule 3 regex `(\d{1,3})%` | `src/llm/verification.py` | Matched `"6%"` inside `"80.6%"`, compared 6 against 80.6, and failed **every** report quoting a one-decimal confidence. Now allows a decimal part and traces percentages against all evidence fields, not only `confidence`. |
| `compute_agreement_score` shape | `src/evidence/reliability.py` | Unconditional `unsqueeze(0)` turned `(1,K)` logits into `(1,1,K)`; `cosine_similarity` returned a vector and `.item()` raised. |
| Entropy normalised by `2.0` | `src/evidence/reliability.py` | Hardcoded divisor correct only for K=7; now `log(num_classes)`. |
| `text_proj.pt` loaded non-strictly into the wrong class | `Dummy/Project/predict.py` | Rebuilt as a plain `nn.Sequential` while training saved a `TextResidualProjector`; the non-strict load silently discarded **every** weight, leaving a random text projection. Now `strict=True`. |
| sklearn 1.9 → 1.7 unpickle | stacking ensembles | `LogisticRegression.predict_proba` dereferenced the removed `multi_class` attribute, raising `AttributeError` on every stacked prediction. Shimmed in `backends._patch_sklearn_logreg`. |
| `tensor_to_bgr` rank | `src/stress/corruption.py` | Required `(C,H,W)`; every image corruption crashed on the `(1,3,224,224)` tensors callers hold. |
| Non-ASCII in `print()` | 7 files | `UnicodeEncodeError` on Windows cp1252 consoles — crashed the stress benchmark *after* a complete run, at the final save message. |
| Hardcoded 7-class ISIC labels | `app/patient_app.py`, `src/main.py` | Displayed every diagnosis under the wrong name for the 6-class PAD-UFES-20 models. |
| Significance tested with `<` at the boundary | `src/eval/stats.py` | Benjamini-Hochberg and Bonferroni rejected `H_i` only when the adjusted p-value was *strictly* below alpha. The textbook rule is "does not exceed". For `p=[0.01..0.05]`, `n=5`, every adjusted value equals 0.05 and BH rejects all five; the `<` comparison instead returned `[True, True, True, False, False]` purely according to whether `p * n / rank` rounded a few ULPs below 0.05. This changes published significance claims, silently. |

### 3.5 Stress-benchmark metrics that could not fail

Two metrics were defined so that a degenerate system scored perfectly:

- `recovery_rate` counted **every** deferral as a save, so a system that refuses every
  case scored 1.000. It now counts only deferrals where the withheld answer would have
  been *wrong*, with deferrals as the denominator. This required recording
  `fused_diagnosis` on the deferral paths so the counterfactual is measured rather than
  assumed.
- `accuracy` was computed over all cases with deferrals scored as `argmax(uniform) == 0`,
  silently crediting or penalising the system for answers it explicitly declined to give.
  It is now conditional on the cases actually decided, reported alongside a new
  `coverage` column so the two are read together.

---

## 4. Deviations from the written plan

Each is deliberate; the reason is recorded here and at the point of deviation in code.

| Plan says | Built instead | Why |
|---|---|---|
| Vision team = DermLIP-PanDerm, PanDerm, Swin-Tiny ×2, ResNet152 (`proj.md` Part 2) | Trained specialists A–G (ResNet50+MHSA ViT / MetaBlock MLP / Bio_ClinicalBERT) | PanDerm weights are unpublished, PAD-UFES-20 contains no dermoscopy images, and torch here is CPU-only. The previous implementation shipped these five with randomly-initialised heads — five "specialists" voting on noise. |
| Fusion over 5 vision models | Fusion over 4 modalities | See §3.1. The evidence layer, acquisition policy and stress conditions are all modality-indexed; the model-indexed axis was the outlier. |
| ClinicalBERT **mean-pooled** (`proj.md` Part 3) | `[CLS]` token | The trained text projector was fitted on `[CLS]`. Feeding it mean-pooled vectors puts it off-distribution. |
| Level 2 = `max_agreement == 2` | `>= 2` | See §3.2 — as written the ladder is not exhaustive. |
| 7 classes | 6 (BCC, ACK, NEV, SEK, SCC, MEL) | PAD-UFES-20 has 6; the checkpoints are trained against these indices. |
| 15 stress conditions | 18 | The existing table already enumerated 18 (extra severity variants). Kept. |
| Conformal prediction | 4-level consensus hierarchy | Already specified by `proj.md`; noted because the population-level coverage guarantee does **not** carry over. |

`dermoscopy_image` is retained as a modality slot and permanently masked off for
PAD-UFES-20. This keeps the masking and acquisition paths genuinely exercised and lets a
dermoscopy-bearing dataset drop in, at the cost of `ACQUIRE` sometimes requesting a
dermoscopy image that this dataset can never supply — which is a clinically sensible
request even though it is unactionable in a retrospective run.

---

## 5. What is trustworthy in this repository

**Engineering — verified working:**
- All 7 specialists load with `strict=True`, so any drift between
  `src/models/architectures.py` and the checkpoints fails loudly.
- End-to-end pipeline: encode → specialists → scribe → evidence profile → stage-1 → fusion
  → consensus → report → stage-2, with a complete evidence trace.
- Graceful degradation across every modality subset; `ABSTAIN` on an empty case.
- FastAPI `/predict`, both Streamlit apps, and the DERMA-STRESS benchmark all run.
- 236 tests pass at 70% line coverage, including regression tests for every
  defect in §3.

**Not trustworthy:**
- Any accuracy, F1, AUC, ECE, confusion matrix, ROC curve or training curve in
  `Dummy/Project/saved_models/`.
- Any claim that this system generalises. It has never been evaluated on held-out data.
- The `models_benchmark_summary.csv` ordering (G > D > E > F > A > B > C) — that ranking is
  the hardcoded `MODEL_TARGET_ACC` dict, not a measurement.

**Stress-benchmark numbers** produced by `src/stress/benchmark.py` measure *relative
robustness of the pipeline's behaviour* (how coverage and deferral shift under
corruption). They are honest measurements of the system as built, but they inherit the
memorisation problem: the underlying specialists have seen every case, so absolute
accuracy under corruption is still an overestimate.

---

## 6. To obtain real numbers

1. Rebuild splits with `src.data.case_loader.patient_level_split` — disjoint by
   construction, grouped by `patient_id`, with the disjointness asserted rather than
   assumed.
2. Delete `MODEL_TARGET_ACC` and the rewriting block (`train_all.py:457-518`), and the
   `np.clip(val_acc, 0.7000, target_ceiling)` at `train_all.py:387-388`.
3. Retrain A–G. This is cheap: training runs on `cached_multimodal_features.pt` with the
   backbones frozen, so it is minutes on CPU, not hours.
4. Re-evaluate on the held-out test partition and regenerate every figure.
5. Expect substantially lower headline numbers. That is the point — they will be real.

Until steps 1–4 are done, this repository is a **working system with unvalidated models**,
and should be described that way in any write-up.
