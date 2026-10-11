# Implementation Status

What is built, what is verified, and how to reproduce it. Updated 2026-10-11.

---

## 1. Built and verified

| Component | Module | Tests | Verification |
|---|---|---|---|
| Evidential opinions, DS combination | `src/evidence/evidential.py` | 34 | vacuous = identity element; conflict ⊥ ignorance; fold order-independent |
| Conformal sets (APS/RAPS, Mondrian) | `src/calibration/conformal.py` | 20 | empirical coverage **0.8944** vs 0.90 target on 464 cases |
| Learn-then-Test risk control | `src/calibration/ltt.py` | 14 | **validity simulation passes** — violation rate ≤ δ over repeated calibration draws |
| Three-action policy + cost model | `src/decision/policy.py` | 31 | conflict→DEFER vs ignorance→ACQUIRE routing; gates only increase caution |
| Selective metrics (AUGRC, AE, ECE) | `src/eval/selective.py` | 32 | AUGRC ∈ [0, ½] exactly; AE=1 for oracle, ≈0 for random, <0 for inverted |
| EADE engine (+ softmax bridge) | `src/agents/eade.py` | — | end-to-end via `scripts/run_eade.py` |
| Reproducible features | `scripts/build_features.py` | — | `--verify` passes at **cosine 1.000000** |
| Honest baselines | `scripts/baseline_probe.py` | — | 3 seeds, disjoint patient splits |
| Patient/lesion-level splits | `src/data/case_loader.py` | 15 | disjointness **asserted at runtime** |
| **MILK10k loader** | `src/data/milk10k.py` | **38** | runs on the real 10,480-image release; 5,240 lesions, 11 classes, stratified lesion-disjoint splits, deterministic across processes |
| DERMA-STRESS, 18 conditions | `src/stress/` | 35 | all conditions generate; metrics fixed |
| Statistics (bootstrap, McNemar, BH) | `src/eval/stats.py` | 23 | hand-computed values; BH boundary rule fixed |
| Config ↔ code consistency | — | 20 | class ordering and axis length pinned |
| Orchestrator, API, 2 Streamlit apps | `src/`, `app/` | 21 | both decision paths run |
| **RAG similar-case retrieval** | `src/retrieval/rag_index.py`, `scripts/build_rag_index.py` | **39** | 1,403 training cases indexed; **no calibration/test patient present**, asserted by `--verify`; spec mismatch raises instead of searching the wrong space |
| **7-agent pipeline** | `src/agents/` | 21 | specialists, scribe, librarian, devil's advocate, EADE, report, stage-2 verifier — all reached on a real case |

**Total: 442 tests passing.**

```bash
pytest tests -q                        # 442 passed, 3 skipped
pytest tests -q -m "not integration"   # skip the ones needing real weights
```

---

## 2. Verified end-to-end result

`scripts/run_eade.py`, bridge mode over the legacy A–G ladder. Disjoint patient split:
train 1402 / calibration 432 / test 464.

| | coverage | malignancy miss rate among proceeded | AUGRC |
|---|---|---|---|
| **EADE** | 0.433 | **0.0050** | 0.1634 |
| B3 max-softmax abstention, matched coverage | 0.433 | 0.0597 | **0.1276** |

**12× lower miss rate at identical coverage.** Driven by the conformal ambiguity gate:
210 of 464 cases had a prediction set spanning benign and malignant classes, which
max-softmax confidence does not detect.

Action split: PROCEED 201 (43.3%) · ACQUIRE 132 (28.4%) · DEFER 131 (28.2%).
Deferral reasons: `AMBIGUOUS_SET` 210 · `CONFLICT` 41 · `COST` 12.

**Counter-evidence, reported:** B3 is better on AUGRC. The ignorance score ranks cases
less well than max-softmax even though the policy built on it is far safer. Both numbers
belong in the results table.

**LTT reported NOT CONTROLLED** at α=0.05, δ=0.10 with n=432 and a 9.95% miss base
rate — and said so rather than inventing a threshold. The conformal and conflict gates
achieved the low miss rate on their own.

### What this does and does not show

✅ The machinery works: evidential fold, conformal gating, cost-aware routing,
distinct deferral reasons, honest failure reporting.

❌ **Not a generalisation estimate.** The legacy specialists were trained on data
overlapping this test split ([evaluation_integrity.md](evaluation_integrity.md)). Native
numbers need the GPU run.

---

## 2a. MILK10k is downloaded, audited and wired in

The dataset arrived on 2026-10-11 (368 MB, extracted at `./milk10k`, gitignored). It is no
longer a planned dependency. Everything below is measured, not quoted:

```bash
python scripts/milk10k_audit.py                 # structure, splits, power, feasibility
python scripts/milk10k_audit.py --hash          # SHA-256 every image (slow)
pytest tests/test_milk10k.py -q                 # 38 passed
```

| | |
|---|---|
| lesions / images | 5,240 / 10,480 — each lesion exactly one clinical close-up + one dermoscopic |
| classes | **11** (AKIEC, BCC, BEN_OTH, BKL, DF, INF, MAL_OTH, MEL, NV, SCCKA, VASC) |
| malignant | **71.7%**, BCC alone 48.1% — a referral, not screening, population |
| histopathology | 95.7% (the published "~96%" checks out) |
| duplicates | **none**; all 10,480 files byte-distinct by SHA-256 |
| splits | train 3,143 / calibration 1,048 / test 1,049, lesion-disjoint, stratified; prevalence held at 0.717 / 0.718 / 0.716 |
| metadata | age, sex, site, 6-level skin tone, **7 MONET concept scores** → 26-d encoded |

**Three things this changed.**

1. **α=0.05 is now attainable.** With n=1,048 calibration lesions the Hoeffding slack is
   0.0331, so an empirical miss rate below 0.0169 certifies the target. At PAD-UFES-20's
   n=432 it was infeasible, and LTT correctly reported `controlled=False`.
2. **No centre-shift experiment.** The release publishes no centre column, so a protocol
   we had written three times over cannot run. See
   [what_did_not_work.md](what_did_not_work.md) §10.
3. **5 of 11 classes get no class-conditional guarantee** (BEN_OTH, DF, INF, MAL_OTH,
   VASC — all under ⌈1/α⌉=20 calibration points), and neither does skin tone 0 (1 point).
   The implementation withholds thresholds rather than fabricating them.

The Kaggle trainer reads the verified schema directly — the previous
`milk_detect_columns` pattern-matcher is gone — and was smoke-tested end to end on CPU
against the real images, emitting `predictions_test.csv` with `ignorance_u`, `conflict_C`
and 11 class probabilities, which is exactly what the decision engine consumes.

---

## 3. Not yet done

| Item | Blocker | Where |
|---|---|---|
| **MILK10k GPU training** | **needs a GPU — this is the one remaining blocker** | [06](06_training_guide.md) §3 |
| Native evidential heads | needs GPU (same run) | [06](06_training_guide.md) §3 |
| DermLIP_PanDerm encoder | needs HF token; licence check | [06](06_training_guide.md) §4 |
| Acquisition-value estimator `h_m` | needs the trained heads (data is in hand) | [04](04_methods.md) §4 |
| Baselines B4, B6–B9 | after native training | [05](05_experimental_protocol.md) §3 |
| Group-conditional fairness results | needs the trained heads (skin tone is in hand) | [04](04_methods.md) §5 |
| Guarantee-violation under shift | needs both datasets trained | [05](05_experimental_protocol.md) §7 |
| Rebuild the RAG index on MILK10k | after training (current index is PAD-UFES-20) | [10](10_retrieval.md) §7 |
| Persist the train-fitted metadata preprocessor | would make the stronger 2930-d index runtime-queryable (agreement 0.875 vs 0.500) | [10](10_retrieval.md) §2 |

---

## 4. Reproduce from a clean clone

```bash
pip install -e ".[dev]"
# Windows/Anaconda: KMP_DUPLICATE_LIB_OK=TRUE, or torch and MKL both load OpenMP and abort

# 1. features (~4.5 min CPU), then prove they regenerate exactly
python scripts/build_features.py --fit-metadata-on-train
python scripts/build_features.py --verify data/processed/features_v2.pt   # cosine 1.000000

# 2. honest baselines, 3 seeds
python scripts/baseline_probe.py --seeds 42 7 2026

# 3. calibrate + evaluate the decision engine
python scripts/run_eade.py --alpha 0.05 --delta 0.10

# 4. robustness
python scripts/run_stress.py --limit 50

# 5. pipeline smoke test
python scripts/demo.py

# 6. tests
pytest tests -q
```

Serving: `python src/main.py` (FastAPI, `:8000/docs`) ·
`streamlit run app/patient_app.py` · `streamlit run app/clinician_console.py`.

---

## 5. Environment

| | |
|---|---|
| Python | 3.13.9 |
| torch | 2.13.0+**cpu** — no CUDA locally, hence the Kaggle split |
| torchvision | 0.28.0 (`ResNet50_Weights.DEFAULT` = **IMAGENET1K_V2**; pinned explicitly, because `DEFAULT` silently moved and that is part of why the old cache is irreproducible) |
| transformers | 5.14.1 |
| scikit-learn | 1.7.2 — the legacy `.joblib` ensembles were pickled under 1.9.0; `backends._patch_sklearn_logreg` shims the removed `multi_class` attribute |
| seeds | 42, 7, 2026 across `random`, `numpy`, `torch` |

---

## 6. Reproducibility guards

1. Feature **manifest**: library versions, backbone weight enum, transform spec,
   SHA-256 over the image bytes consumed, and `img_ids` **stored in the artifact** so row
   order is never inferred.
2. `--verify` re-extracts a random sample and fails below cosine 0.999.
3. Split disjointness **asserted**, not assumed.
4. `tests/test_config.py` pins config ↔ `src/constants.py` agreement (class ordering,
   modality axis length, retrieval dim, image preprocessing).
5. Per-case predictions saved, so any metric is recomputable without retraining.
6. Checkpoints load with `strict=True`, so architecture drift fails loudly instead of
   silently degrading to random initialisation.

Guard 2 belongs in CI — it is the specific check that would have caught the failure which
invalidated the previous generation of results.

---

## 7. Known defects fixed along the way

Each has a regression test. Full evidence in
[evaluation_integrity.md](evaluation_integrity.md).

| Defect | Effect |
|---|---|
| Fusion axis 5 (models) vs evidence mask 4 (modalities) | fusion raised on **every** case; orchestrator swallowed it and substituted `diagnosis=0, confidence=0.0` — the entire test suite passed while fusion never ran |
| Consensus ladder not exhaustive | 6/7 agreement at confidence 0.79 matched no level and fell through to "unresolvable disagreement" |
| Four untrained MLPs thresholded as signal | with text quality 0.30 and reliability 0.30, the untrained weighting net still gave text **0.68** of the fusion weight |
| `recovery_rate` divided by all cases | a system refusing everything scored 1.000 |
| Benjamini–Hochberg used `<` not `≤` | significance flipped on floating-point noise |
| AUGRC right-endpoint Riemann sum | returned 0.5025, breaking the documented [0, ½] bound |
| Label map in `preprocessing.py` | SCC↔ACK and NEV↔MEL transposed vs the trained heads — four of six classes mislabelled, silently |
| Stage-2 regex `(\d{1,3})%` | matched "6%" inside "80.6%", failing every report quoting a one-decimal confidence |
| `text_proj.pt` loaded non-strictly into the wrong class | discarded **every** weight, leaving a random text projection |
| `tensor_to_bgr` required `(C,H,W)` | every image corruption crashed on batched tensors |
| Non-ASCII in `print()` | `UnicodeEncodeError` on Windows cp1252 — killed the stress run *after* a complete pass |
| Hardcoded 7-class ISIC labels in the UI | every diagnosis displayed under the wrong name |
