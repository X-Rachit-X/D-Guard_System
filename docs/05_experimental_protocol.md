# Experimental Protocol

Fixed in advance so results are confirmatory rather than selected. Deviations get
recorded in [what_did_not_work.md](what_did_not_work.md) with their reason.

Pre-registration values: [prereg.yaml](prereg.yaml), committed before the test split is
touched.

---

## 1. Datasets

### Primary — MILK10k (ISIC)

Verified 2026-10-11 from the ISIC DOI page:

| Property | Value |
|---|---|
| Size | **10,480 images / 5,240 lesions** — every lesion has a paired clinical close-up **and** dermatoscopic image |
| Ground truth | ~96% histopathology-confirmed; `method of ground truth` field present |
| Licence | **CC-BY-NC** — non-commercial, credit the MILK study team |
| Download | 344.7 MB complete bundle (metadata CSV 2.0 MB) |
| DOI | training `10.34970/648456` · benchmark `10.34970/262082` |
| Centres | Vienna, Ankara, UQ, Memorial Sloan Kettering, Skopje (5) — **described by the study but NOT identifiable in the release**: `attribution` is the single constant value "MILK study team" for all 10,480 rows |
| Metadata | age (5-yr bins), sex, anatomic site, **skin tone (6 levels, 0–5)**, diagnosis (ISIC-Dx + 11-category), MONET concept probabilities (7 term groups) |
| Benchmark | 479 lesions, labels hidden, challenge submission platform |

**Why this dataset decides the project.** Every lesion has the dermoscopic image, so
"should I acquire dermoscopy?" is a **real experiment with an oracle upper bound** rather
than a simulation. Skin tone makes the fairness analysis nearly free.

**Correction, measured 2026-10-11 on the extracted release.** An earlier version of this
protocol planned a centre-held-out shift experiment on the strength of the 5 centres. That
is **not possible**: no centre column is published. The shift experiment runs on
PAD-UFES-20 only, and `tests/test_milk10k.py` pins the finding so the claim cannot
silently return.

**Verified composition** (`python scripts/milk10k_audit.py`):

| | |
|---|---|
| lesions / images | 5,240 / 10,480 — every lesion has exactly one clinical close-up and one dermoscopic image |
| classes | **11** (AKIEC, BCC, BEN_OTH, BKL, DF, INF, MAL_OTH, MEL, NV, SCCKA, VASC) |
| malignant prevalence | **71.7%** (3,757 lesions); BCC alone is **48.1%** |
| histopathology-confirmed | 95.7% |
| rarest class | MAL_OTH, **9 lesions in total** |
| skin tone | 6 ordinal levels; level 3 is 60.6% of images, level 0 has 6 lesions |
| duplicate images | **none** — all 10,480 files are byte-distinct by SHA-256 |

**The prevalence is the headline caveat.** At 71.7% malignant this is a tertiary-referral
population, not a screening one. Every cost-matrix conclusion and every risk target
calibrated here inherits that base rate, and none of it transfers to a screening setting
without re-calibration.

Access requires an ISIC account (the bundle URL is authenticated).

### Shift — PAD-UFES-20

Smartphone clinical images, no dermoscopy. Used **only** for clinical-branch distribution
shift, with an explicit label mapping. Already prepared locally with a verified
reproducible feature pipeline and honest baselines
([01_project_overview.md](01_project_overview.md) §4).

### Optional

Derm7pt (second paired dataset, replication if time allows); HAM10000 / ISIC 2019
(dermoscopy-only, pretraining or shift only).

---

## 2. Splits and leakage rules

| Item | Decision |
|---|---|
| Split unit | **Lesion** for MILK10k (no patient IDs published); **patient** for PAD-UFES-20 |
| Proportions | 60 train / 15 validation / **10 calibration** / 15 test, stratified |
| Seeds | 42, 7, 2026 — all results as mean ± std |
| Dedup | via the provided ISIC IDs; residual leakage risk stated in limitations |
| Calibration | carved out separately; conformal and LTT use it **only** |
| Acquisition targets | **out-of-fold** (5-fold) — in-fold targets are optimistic |
| Preprocessor | fitted on **train only** |
| Test split | touched **once**, after `prereg.yaml` is committed |

Three rules that exist because each was previously broken:

1. Partition **once**; never call a splitter repeatedly on the full index range.
2. **No metric is post-processed.** Any equivalent of `MODEL_TARGET_ACC` is misconduct.
3. **Nothing is clipped to a target.** Measured values are reported as measured.

---

## 3. Baselines

| ID | Baseline | Actions |
|---|---|---|
| B1 | clinical-image-only classifier | proceed always |
| B2 | fused clinical + dermoscopy (always acquire) | proceed always |
| B3 | confidence-threshold abstention, LTT-calibrated to the same α | proceed / defer |
| B4 | learning to defer (Mozannar & Sontag surrogate) | proceed / defer |
| B5 | **HOT-style**: acquire if entropy > τ | proceed / acquire |
| B6 | greedy information-gain acquisition + abstention | all three |
| B7 | random acquisition at matched rate | lower bound |
| B8 | oracle acquisition at matched rate | upper bound |
| B9 | **MetaBlock** (Pacheco & Krohling) image+metadata | proceed always |
| **Ours** | EADE: evidential + DS + conformal + LTT + cost | all three |

B5 and B9 are non-negotiable. B5 is the closest published precedent (HOT AI already
recommends dermoscopy when the clinical image is insufficient); B9 is the reference
fusion method published by the PAD-UFES-20 authors on this exact problem family.

B3/B5 are implemented in `src/decision/policy.py`.

---

## 4. Ablations

**Mechanism** (each row removes one thing from the full EADE):

| Variant | Question |
|---|---|
| full | — |
| − DS (evidential heads + arithmetic mean) | is the combination rule doing the work, or just the heads? |
| − evidential heads (DS on softmax = bridge mode) | is Dirichlet evidence necessary? |
| − conflict gate | does `C` contribute beyond `u`? |
| − conformal ambiguity gate | how much of the safety gain is the mixed-set rule? |
| − LTT (raw expected loss) | does the guarantee cost coverage? |
| − acquisition estimator (entropy trigger) | is the learned estimator better than HOT-style? |
| uncalibrated vs temperature-scaled | does calibration before thresholding matter? |
| single pass vs 3-seed ensemble | is EDL's one-pass uncertainty sufficient? |
| cost sweep `ℓ_miss ∈ {5,10,20}`, `c_acq ∈ {0.1,0.5,1.0}` | **mandatory** — conclusions depend on these |

**Availability.** All 15 non-empty subsets of 4 modalities × each method. Report the full
table. A method whose advantage appears only with everything present has not supported
the robustness claim.

**Degradation.** DERMA-STRESS, 18 conditions, replayed from cached corruptions so every
method sees identical inputs.

---

## 5. Metrics

**Primary endpoint (pre-registered):** **coverage at guaranteed risk** — the fraction
auto-proceeded while the malignancy miss rate among proceeded is ≤ α, at α = 0.05,
δ = 0.10.

**Selective prediction:** AUGRC (primary multi-threshold); AURC for comparability only;
risk–coverage curves; selective risk at 70/80/90% coverage; realised `R(λ̂)` vs nominal α;
**guarantee violation** per stress family.

**Classification:** balanced accuracy, macro-F1, **per-class** precision/recall/F1
(always, never macro alone), confusion matrix, binary malignancy AUROC with DeLong CIs.

**Calibration:** adaptive (equal-mass) ECE, class-wise ECE, Brier with
reliability/resolution/uncertainty decomposition.

**Conformal:** empirical coverage, mean set size, mixed benign/malignant set rate,
per-skin-tone coverage.

**Acquisition:** acquisition rate, **AE(r)** against oracle and random, loss before/after,
unnecessary-request rate, NIR with magnitude, MSES.

**Clinical utility:** decision-curve net benefit; expected cost per case.

**Operational:** job success rate, recovery, end-to-end latency, traceability.

Coverage and accuracy are always reported **as a pair**.

---

## 6. Statistics

| Purpose | Method |
|---|---|
| Intervals | paired lesion-level bootstrap, ≥ 2000 resamples, 95% percentile CI |
| Paired model comparison | McNemar; DeLong for AUROC |
| Effect size | Cohen's `d`; absolute difference with CI |
| Multiplicity | Holm (or Benjamini–Hochberg) across the comparison family; report raw **and** adjusted p |
| Seeds | mean ± std over 3 splits; never a single-seed headline |

Implemented and unit-tested in `src/eval/stats.py` (23 tests), including a fix to the
Benjamini–Hochberg boundary rule that had made significance flip on floating-point noise.

**Pre-committed decision rule for H1:** accept if the EADE's malignancy miss rate at
matched coverage is lower than B3's and B5's with Holm-adjusted `p < 0.05` on at least
two of three seeds. Anything weaker is reported as "did not replicate".

---

## 7. DERMA-STRESS families

| Family | Conditions |
|---|---|
| A — image degradation | Gaussian blur, noise, brightness, JPEG, occlusion (3 severities) |
| B — missing evidence | drop dermoscopy / metadata / text / combinations |
| C — contradictory context | permuted site, shifted age bin, conflicting symptom text |
| D — distribution shift | PAD-UFES-20 clinical branch. **Centre shift is NOT available** — the MILK10k release publishes no centre column (verified 2026-10-11) |
| E — leave-class-out OOD | remove a class from training; does the policy defer on it? |
| F — combined failures | degraded image + missing metadata; missing dermoscopy + contradiction; shift + degradation |
| G — operational | inference timeout, worker restart, corrupt upload, LLM unavailable |

Per condition: coverage, accuracy **conditional on coverage**, AUGRC, **guarantee
violation** (`R̂_test − α`), conformal set size, acquisition rate, deferral reasons,
latency.

`recovery_rate` is the fraction of **deferrals** whose withheld answer would have been
wrong — denominator is deferrals, not all cases, because otherwise a system that refuses
everything scores a perfect 1.000.

---

## 8. Twelve-week plan

| Week | Work | Milestone |
|---|---|---|
| 1 | MILK10k licence + download; EDA; label grouping; lesion-level splits | — |
| 2 | Clinical-only baseline B1; reproducible features; `--verify` passes | **M1 vertical slice** |
| 3 | Dermoscopy + fused models with modality dropout; B2 | — |
| 4 | Evidential heads (native mode); temperature scaling; calibration metrics | **M2 trustworthy baseline** |
| 5 | DS fusion; conformal sets; B3 | — |
| 6 | Acquisition estimator (out-of-fold); B5–B8; **B9 MetaBlock** | **M3 fair comparison** |
| 7 | Expected-loss policy + LTT layer; group-conditional calibration | **prereg frozen** |
| 8 | B4 learning to defer; mechanism check H4 (error vs `u_m`) | — |
| 9 | DERMA-STRESS A–G; guarantee violation | **M4 policy value (validation only)** |
| 10 | **single test-split run**; fairness tables | — |
| 11 | Ablations, 3 seeds, bootstrap + Holm; figures 300 DPI | **M5 integrated system** |
| 12 | Write-up; `what_did_not_work.md`; demo | submission |

**Critical path:** weeks 4–7. If the EADE does not beat B3/B5 on validation, the paper
pivots to DERMA-STRESS plus the guarantee-violation analysis — a legitimate contribution
requiring no change to weeks 8–12.

---

## 9. Threats to validity

| Threat | Mitigation | Residual |
|---|---|---|
| Legacy weights' metrics invalid | Retrain from scratch; audit documented | Resolved by retraining |
| Synthetic text ≠ patient text | Label everywhere; report text-ablated results | **High** |
| MILK10k has no patient IDs | Lesion-level split + ISIC-ID dedup; stated | Medium |
| Rare classes (MAL_OTH has **9 lesions total**, 5 after splitting) | Binary malignancy for the guarantee; no rare-class claims. 5 of 11 classes cannot receive a class-conditional guarantee at α=0.05 and the implementation withholds one | **High** |
| **71.7% malignant prevalence** — a referral, not screening, population | Report the base rate beside every risk number; no screening claim | **High** |
| Guarantee fails under shift | Expected; measured and reported as a finding | By design |
| Cost matrix seen as arbitrary | Pre-registered + mandatory sweep | Medium |
| Acquisition estimator overfits | Out-of-fold targets; oracle/random bounds | Low |
| PanDerm pretraining may include MILK10k | **Check the paper's data list before claiming a clean probe** | Open |
| Selection on test | Test touched once | Low |
| Feature drift | manifest + `--verify` at cosine > 0.999, in CI | Low |

---

## 10. Reproducibility requirements

1. `scripts/build_features.py` then `--verify` passes at cosine > 0.999
2. Seeds 42/7/2026 fixed across `random`, `numpy`, `torch`
3. Manifest (library versions, backbone weight enum, transform spec, image-set SHA-256)
   stored beside every result
4. Config hash + git commit recorded with every prediction dump
5. **Per-case predictions saved**, not only aggregates, so any metric can be recomputed
   without retraining
6. `--verify` in CI, so a silent preprocessing change fails the build
7. `prereg.yaml` committed before the test split is touched

Point 6 is the specific guard against the failure that invalidated the previous
generation.
