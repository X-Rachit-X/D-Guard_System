# Experimental Protocol and Semester Plan

Fixed in advance so that results are confirmatory rather than selected. Any deviation
gets recorded in `docs/what_did_not_work.md` with its reason.

---

## 1. The central hypothesis

From the honest baseline ([01_system_analysis.md](01_system_analysis.md) §4):

> Image + metadata (macro-F1 **0.653**) outperforms all three modalities concatenated
> (**0.633**), because naive fusion cannot discount the near-useless text channel
> (0.414).

This gives a sharp, falsifiable primary hypothesis:

> **H1.** Evidence-aware fusion with per-modality Dirichlet evidence and Dempster–Shafer
> combination recovers or exceeds the best *hand-selected* modality subset (D) **without
> being told which modality to distrust**, while naive fusion (G) does not.

H1 is a statement about *automatic* robustness, which is a stronger and more interesting
claim than a raw accuracy delta. Secondary hypotheses:

> **H2.** Per-modality ignorance `u_m` is informative: cases with high `u_m` have
> measurably higher error for modality `m`. (Falsifiable: if the reliability diagram is
> flat, the evidence layer is decorative and H1's mechanism is not what we claim.)
>
> **H3.** Conformal risk control attains its nominal selective-risk target
> (`R(λ̂) ≤ α`) on test, where the current 4-level heuristic ladder has no guarantee and
> will be shown to miss the target at some operating points.
>
> **H4.** Under DERMA-STRESS corruption, the evidence-aware system degrades more
> gracefully: it sheds *coverage* rather than *accuracy among accepted cases*.
>
> **H5.** DS conflict mass `C` separates genuine cross-modal contradiction from shared
> ignorance, which the current `1 − mean JS` consistency score provably cannot.

H2 and H5 are the mechanism checks. If H1 holds but H2 fails, the gain came from
somewhere other than the stated mechanism and must be reported as such.

---

## 2. Data protocol — frozen

| Item | Decision |
|---|---|
| Dataset | PAD-UFES-20 only (2298 clinical images, 1373 patients, 6 classes) |
| Task | 6-class primary. Binary malignant/benign as a secondary view (comparability with v1) |
| Split | **Patient-level disjoint**, 70/15/15 by `patient_id` — `case_loader.patient_level_split`, disjointness asserted at runtime |
| Seeds | 42, 7, 2026 (three independent splits). All results reported as mean ± std across seeds |
| Features | `scripts/build_features.py`, manifest-tracked, `--verify` must pass at cosine > 0.999 |
| Metadata preprocessor | Fitted on **train split only** (`--fit-metadata-on-train`). Note this yields 114 dims, not 116, since two one-hot levels are absent from train — expected and correct |
| Calibration split | Carved from train, **never** from test. Conformal calibration consumes it |
| Text | Synthetic (template-generated). Must be labelled as such in every result |

Three rules that exist because they were previously broken:

1. **The test split is touched once**, at the end, for the final table. All selection
   happens on validation.
2. **No metric is post-processed.** The historical `MODEL_TARGET_ACC` rewriting is
   deleted; any future equivalent is research misconduct.
3. **Nothing is clipped to a target.** Measured numbers are reported as measured.

---

## 3. Model ladder

| # | Model | Purpose |
|---|---|---|
| L0 | Majority class | Trivial floor |
| L1 | Image only (linear probe / fine-tuned CNN) | Unimodal |
| L2 | Metadata only | Unimodal — the surprisingly strong one |
| L3 | Text only | Unimodal — expected weak |
| L4 | Early fusion (concat → MLP) | Naive multimodal |
| L5 | **MetaBlock** (image+metadata) | **Literature baseline on this exact dataset** |
| L6 | Gated fusion (existing `AdaptiveGatedFusion`) | Learned-gate multimodal |
| L7 | Geometric late fusion, uniform weights | Pooling baseline |
| **L8** | **Evidential + DS combination** | **Proposed** |
| **L9** | **L8 + EM diagnosis-adaptive refinement** | **Proposed, full** |

L5 is non-negotiable: MetaBlock was published by the PAD-UFES-20 authors on this dataset
and a reviewer will ask for it.

---

## 4. Ablations

**Component ablation on L9** (each row removes one thing):

| Variant | Tests |
|---|---|
| L9 full | — |
| − EM refinement (DS only) | Does the iterative step earn its complexity? |
| − DS (evidential heads + arithmetic mean) | Is the combination rule doing the work, or just the evidential heads? |
| − evidential heads (softmax + DS on softmax) | Is Dirichlet evidence necessary? |
| − conflict term | Does `C` contribute beyond `u`? |
| hard vs **soft** diagnosis prior | Confirms the monotonicity fix ([04 §4.3](04_mathematical_foundations.md)) matters |
| uniform `w` | Is adaptive weighting needed at all? |
| − modality dropout in training | Does absence-aware training matter? |

**Availability ablation (the important one).** All 7 non-empty modality subsets × each
model. Report the full 7×N table. A method whose advantage appears only with all three
modalities present has not supported H1.

**Degradation ablation.** DERMA-STRESS, 18 conditions × each model, replayed from the
cached corruptions so every model sees identical inputs.

---

## 5. Metrics

**Classification** — per-class precision/recall/F1 always reported, never macro alone
(the project rule, and necessary with MEL at 52 cases): accuracy, balanced accuracy,
macro-F1, weighted-F1, per-class AUROC and AUPRC, confusion matrix.

**Calibration** — adaptive ECE (equal-mass bins; state bin count), Brier with its
reliability/resolution/uncertainty decomposition, reliability diagrams per modality.

**Selective prediction** — the primary family for this project:

- risk–coverage curve, **AURC** and **E-AURC**
- coverage at `risk ≤ α` for `α ∈ {0.05, 0.10, 0.20}`
- realised `R(λ̂)` on test vs nominal `α` (this is the H3 test)
- deferral-reason breakdown: conflict / ignorance / cost

**Robustness** — per DERMA-STRESS condition: coverage, accuracy *conditional on
coverage*, ECE, Brier, and `recovery_rate` (fraction of deferrals whose withheld answer
would have been wrong). Coverage and accuracy are always reported as a pair; neither
alone is interpretable for an abstaining system.

**Efficiency** — parameters, inference latency per case on CPU, peak memory.

---

## 6. Statistics

| Purpose | Method |
|---|---|
| Interval estimates | Bootstrap, 1000 resamples, **case-level** (not modality-variant-level), 95% percentile CI |
| Paired model comparison | McNemar on identical test cases |
| Effect size | Cohen's `d`; absolute difference with CI |
| Multiple comparisons | Benjamini–Hochberg across the comparison family; report raw and adjusted p |
| Across seeds | mean ± std over the three splits; never a single-seed headline |

Already implemented and unit-tested in `src/eval/stats.py` (`tests/test_stats.py`,
23 tests), including a fix to the BH boundary rule that had made significance flip on
floating-point noise.

**Pre-committed decision rule for H1:** accept H1 if L9's macro-F1 ≥ D's within one
standard deviation across seeds **and** L9 ≥ G with BH-adjusted `p < 0.05` by McNemar on
at least two of three seeds. Anything weaker is reported as "did not replicate".

---

## 7. Semester plan

Fourteen weeks, assuming one person part-time. Training runs on Kaggle
([08_kaggle_training.md](08_kaggle_training.md)); analysis runs locally on CPU.

| Week | Work | Deliverable |
|---|---|---|
| 1 | Data audit; confirm reproducible features; confirm splits disjoint | `features_v2.pt` + manifest, audit report |
| 2 | Honest baselines L0–L3; per-class metrics; calibration | Baseline table (**done** — §4 of system analysis) |
| 3 | L4, L7 naive fusion; **L5 MetaBlock** | Literature baseline matched or explained |
| 4 | L6 gated fusion; availability ablation over 7 subsets | 7×N availability table |
| 5–6 | **L8**: evidential heads + DS combination; modality dropout training | L8 beats/matches D without being told text is weak (H1) |
| 7 | H2 mechanism check: reliability diagrams of error vs `u_m` | H2 confirmed or H1's mechanism retracted |
| 8 | **L9**: EM refinement; convergence curves of `J` | Monotonicity verified empirically |
| 9 | Conformal risk control; class-conditional calibration | H3: realised risk vs nominal `α` |
| 10 | Cost-sensitive deferral; cost-sensitivity curves | Deferral policy with declared cost matrix |
| 11 | DERMA-STRESS across all models | H4: degradation profiles |
| 12 | Full ablations, 3 seeds, bootstrap + McNemar + BH | Ablation table with CIs |
| 13 | Error analysis; retrieval channel (C7); figures | Figures at 300 DPI; qualitative case studies |
| 14 | Write-up; `what_did_not_work.md`; reproducibility statement | Paper draft + reproducible repo |

**Milestones.** W2: honest baselines exist. W6: H1 answered. W9: guarantee demonstrated.
W12: all numbers final. W14: draft complete.

**Critical path:** W5–6 (L8). If evidential fusion does not beat naive fusion, the
project pivots to a negative-result paper — *"principled evidence management does not
beat a tuned static baseline on PAD-UFES-20, and here is the benchmark that shows it"* —
which is still a legitimate contribution and requires no change to W9–14.

---

## 8. Threats to validity

| Threat | Mitigation | Residual |
|---|---|---|
| Synthetic text is not patient text | Label it as such everywhere; report text-ablated results | **High.** Any claim about patient-authored text is a claim about a simulation |
| Single dataset, single centre | Report as a limitation; design for a second dataset | **High.** No external validation |
| MEL n=52 → ~8–14 per test split | Report per-class CIs; group malignant classes for class-conditional conformal | **High.** Melanoma-specific conclusions are not supportable |
| Selection on test | Test touched once, at the end | Low |
| Multiple comparisons | BH correction across the family | Low |
| Seed luck | Three seeds, mean ± std | Medium |
| Feature-extraction drift | Manifest + `--verify` at cosine > 0.999 | Low — this is the previously-fatal failure, now guarded |
| Metadata features encoding diagnosis priors | Checked: removing them changes accuracy 0.714 → 0.717, and they use no labels | Low, but state it |

---

## 9. Reproducibility requirements

Every result must be regenerable from a clean clone:

1. `scripts/build_features.py --fit-metadata-on-train`, then `--verify` passes
2. seeds 42/7/2026 fixed across `random`, `numpy`, `torch`
3. feature manifest (library versions, backbone weight enum, transform spec, image-set
   SHA-256) stored beside every result
4. config hash + git commit recorded with every prediction dump
5. per-case predictions saved, not just aggregates, so any metric can be recomputed
   without retraining
6. `--verify` in CI, so a silent change to preprocessing fails the build

Point 6 is the specific guard against the failure that invalidated the previous
generation of results.
