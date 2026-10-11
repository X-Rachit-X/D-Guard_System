# D-GUARD

**Risk-Controlled, Cost-Aware Evidence Acquisition and Deferral for Multimodal Dermatology AI**

> ⚠️ **Research prototype. Not clinically validated. Do not use for medical diagnosis.**
>
> 🔴 Metrics shipped with the *original* checkpoints are **invalid** — the training code
> evaluated on a test set that was 100% training data and rewrote predictions toward
> hardcoded targets. Read **[docs/evaluation_integrity.md](docs/evaluation_integrity.md)**
> before citing any number produced before 2026-10-11.

---

## The question

A dermatology AI does not fail because its classifier is a few points short. It fails
because it answers confidently when it should not — on a blurred photograph, an empty
symptom form, or a case where the image and the history contradict each other.

> Given a clinical close-up of a skin lesion, should the system **answer now**, **ask for
> a dermoscopic image first**, or **defer to a human** — and can it make that choice with
> a statistical guarantee on how often it misses a malignancy?

| Action | When |
|---|---|
| **PROCEED** | expected loss of answering is lowest, and all safety gates pass |
| **ACQUIRE** | the expected benefit of dermoscopy exceeds its cost |
| **DEFER** | conflict, ignorance, or deferral is simply cheaper |

---

## Verified result

`scripts/run_eade.py` — disjoint patient split, calibration on 432 held-out cases,
evaluation on 464:

| | coverage | **malignancy miss rate among proceeded** | AUGRC |
|---|---|---|---|
| **D-GUARD** | 0.433 | **0.0050** | 0.1634 |
| max-softmax abstention, *matched coverage* | 0.433 | 0.0597 | **0.1276** |

A **12× lower miss rate at identical coverage**, driven by the conformal ambiguity gate:
210 of 464 cases had a prediction set spanning benign and malignant classes, which
max-softmax confidence cannot detect.

**Reported against us:** the baseline wins on AUGRC. The ignorance score ranks cases less
well than max-softmax even though the policy built on it is far safer. Both belong in the
results table.

Conformal coverage landed at **0.8944** against a 0.90 target. Learn-then-Test reported
**not controlled** at α=0.05 with n=432 and a 9.95% miss base rate — and said so rather
than inventing a threshold.

These verify the **machinery**, not generalisation: the legacy specialists were trained on
data overlapping this test split.

---

## Approach

Every hand-set constant replaced by a derivation or a calibration procedure.

| Component | Before | Now |
|---|---|---|
| Per-modality quality | `0.4·certainty + 0.3·agreement + 0.2·quality + 0.1·stability` | Dirichlet ignorance `u = K/S` from evidential heads |
| Combining modalities | mask + renormalise | **Dempster–Shafer**; absent modality = vacuous opinion = **identity element** |
| Conflict detection | `1 − mean pairwise JS`, threshold 0.3 | DS **conflict mass** `C`, calibrated — separates *conflict* from *shared ignorance*, which JS provably cannot |
| Abstention | 4-level ladder, 0.80/0.50/0.30 | expected loss + **Learn-then-Test** `λ̂`, `P(R ≤ α) ≥ 1−δ` |
| Error costs | symmetric thresholds | explicit, reviewable **cost matrix** + Chow's rule |
| Primary metric | accuracy, then AURC | **AUGRC** (Traub et al., NeurIPS 2024 — AURC is unsuitable for threshold aggregation) |
| Epistemic uncertainty | 3-seed deep ensemble | single-pass EDL (3× cheaper) |

The distinction that does the work:

| situation | JS divergence | Dempster–Shafer | correct response |
|---|---|---|---|
| both modalities confident, **different** classes | large | `C` high, `u` low | **defer to a human** |
| both near-uniform — nobody knows | small → reads *consistent* | `C` low, `u` high | **acquire more evidence** |

Derivations: **[docs/04_methods.md](docs/04_methods.md)**.

---

## Architecture

```
clinical ─┐
dermo    ─┤─► shared backbone ─► EvidentialHead ─► Dir(α)   (separate head each)
metadata ─┤─► residual MLP ─────► EvidentialHead ─► Dir(α)
text     ─┘─► BERT [CLS] ───────► EvidentialHead ─► Dir(α)
                        │
        [ABSENT → vacuous (b=0, u=1) = DS identity element]
                        │
          Dempster–Shafer fold → belief b, ignorance u, conflict C
                        │
   ┌────────────────────┼────────────────────┐
conformal set     expected loss          LTT λ̂
(APS/RAPS,        under cost matrix      P(R≤α)≥1−δ
 class-conditional)
   └────────────────────┼────────────────────┘
                        │
     PROCEED(ŷ)   ACQUIRE(dermoscopy)   DEFER(CONFLICT|AMBIGUOUS_SET|RISK_BOUND|COST)
```

---

## Status

| | |
|---|---|
| Evidential + DS, conformal, LTT, three-action policy, AUGRC/AE | ✅ implemented, **442 tests** |
| End-to-end calibrate + evaluate | ✅ `scripts/run_eade.py` |
| Reproducible features | ✅ verifies at cosine 1.000000 |
| Honest baselines, 3 seeds | ✅ `scripts/baseline_probe.py` |
| DERMA-STRESS, 18 conditions | ✅ running |
| Orchestrator, FastAPI, 2 Streamlit apps | ✅ EADE or legacy ladder behind a flag |
| **RAG similar-case retrieval** | ✅ built, wired in, leak-guarded — [docs/10_retrieval.md](docs/10_retrieval.md) |
| **7-agent pipeline** | ✅ specialists · scribe · librarian · devil's advocate · EADE · report · verifier |
| MILK10k migration · native evidential heads · DermLIP encoder | ⬜ **[docs/06_training_guide.md](docs/06_training_guide.md)** |
| Original checkpoint metrics | ❌ invalid — [docs/evaluation_integrity.md](docs/evaluation_integrity.md) |

### Honest baselines (frozen features, patient-level splits, 3 seeds)

| config | macro-F1 | | config | macro-F1 |
|---|---|---|---|---|
| image only | 0.477 | | **image + metadata** | **0.653** |
| metadata only | 0.648 | | image + text | 0.559 |
| text only | 0.414 | | text + metadata | 0.587 |
| | | | all three | 0.633 |

**All three is worse than the best two** — naive concatenation cannot discount the
near-useless text channel. That gap is the motivation for the whole design.

---

## Retrieval, and why it cannot touch the decision

Similar-case retrieval is **explanation only**. That is a requirement, not caution:
conformal prediction and Learn-then-Test both assume exchangeability, and a corpus that
varies at inference time forfeits the risk bound. So the index is built from the
**training split only**, the excluded patients are written to an audit file, and
`--verify` asserts no calibration or test case can be retrieved.

The leak this blocks would be invisible otherwise — a test-split neighbour carries its
true label into the **prose**, where no metric would ever show it.

```bash
python scripts/build_rag_index.py --verify
# 1403 training cases indexed (895 calibration/test cases EXCLUDED)
# no calibration/test patient indexed : ok
```

Honest cost: the runtime-queryable index drops metadata, because the preprocessor emits
116 columns at runtime but 114 fitted on train only. That lowers neighbour agreement from
**0.875 to 0.500** — metadata carries much of the retrieval signal. Recorded rather than
hidden; [docs/10_retrieval.md](docs/10_retrieval.md) has the fix.

---

## Quickstart

```bash
pip install -e ".[dev]"
# Windows/Anaconda: set KMP_DUPLICATE_LIB_OK=TRUE, or torch and MKL both load OpenMP and abort
```

```bash
python scripts/milk10k_audit.py                                           # audit the primary dataset
python scripts/build_rag_index.py --verify                                # retrieval index + leak check
python scripts/build_features.py --fit-metadata-on-train
python scripts/build_features.py --verify data/processed/features_v2.pt   # cosine 1.000000
python scripts/baseline_probe.py --seeds 42 7 2026
python scripts/run_eade.py --alpha 0.05 --delta 0.10
python scripts/run_stress.py --limit 50
python scripts/demo.py
pytest tests -q                                                           # 442 passed, 3 skipped
```

MILK10k is expected at `./milk10k` (gitignored; CC-BY-NC, so never committed or
republished). Without it, `milk10k_audit.py` and 4 integration tests skip; everything
else runs.

Serving: `python src/main.py` (docs at `:8000/docs`) ·
`streamlit run app/patient_app.py` · `streamlit run app/clinician_console.py`

GPU training: **[docs/06_training_guide.md](docs/06_training_guide.md)** with
[`notebooks/kaggle_train_dguard.py`](notebooks/kaggle_train_dguard.py) (supports MILK10k
and PAD-UFES-20; smoke-tested on CPU).

---

## Datasets

| | Role | Notes |
|---|---|---|
| **MILK10k** (ISIC, `10.34970/648456`) | **primary** | 5,240 lesions × paired clinical + dermoscopic. Every lesion has both modalities, so "acquire dermoscopy" is a real experiment with an oracle bound |
| **PAD-UFES-20** (`10.1016/j.dib.2020.106221`) | shift set | 2298 smartphone clinical images, no dermoscopy. Already prepared locally |

**MILK10k as measured, not as quoted** (`python scripts/milk10k_audit.py`, 2026-10-11):

| | |
|---|---|
| lesions / images | 5,240 / 10,480 — each lesion exactly one clinical + one dermoscopic |
| classes | **11**: AKIEC, BCC, BEN_OTH, BKL, DF, INF, MAL_OTH, MEL, NV, SCCKA, VASC |
| malignant | **71.7%** (BCC alone 48.1%) — a **referral**, not screening, population |
| histopathology | 95.7% · duplicates: **none**, all 10,480 files byte-distinct |
| rarest class | MAL_OTH, **9 lesions** · 5 of 11 classes get no class-conditional guarantee |
| skin tone | 6 ordinal levels; tone 3 = 60.6% of images, tone 0 = 6 lesions |
| centres | **not identifiable** — `attribution` is one constant value, so no centre-shift experiment is possible (this corrects an earlier claim of ours) |

Two consequences worth stating up front. At 71.7% malignant a trivial always-malignant
policy has a 0% miss rate, so **miss rate is only meaningful quoted beside coverage and
the base rate**. And with n=1048 calibration lesions the Hoeffding slack is 0.0331, so
α=0.05 **is** attainable here — it was not at PAD-UFES-20's n=432.

---

## Documentation

Start with **[docs/00_index.md](docs/00_index.md)**.

[Integrity audit](docs/evaluation_integrity.md) · [Overview](docs/01_project_overview.md) ·
[Research design](docs/02_research_design.md) · [Architecture](docs/03_architecture.md) ·
[Methods](docs/04_methods.md) · [Protocol](docs/05_experimental_protocol.md) ·
[Training](docs/06_training_guide.md) · [Literature](docs/07_literature.md) ·
[Status](docs/08_implementation_status.md) ·
**[What you need to do](docs/09_what_you_need_to_do.md)** ·
[Retrieval & agents](docs/10_retrieval.md) · [prereg.yaml](docs/prereg.yaml)

---

## Honest positioning

**Not novel:** evidential deep learning, Dempster–Shafer combination, conformal
prediction, Learn-then-Test, selective prediction, learning to defer, active modality
acquisition, multimodal fusion, recommending dermoscopy (**HOT AI**, Mehta et al., *JEADV*
2025, already does this).

**Defensible:** their integration into one risk-controlled three-action policy; the
application to a paired-modality dermatology dataset where acquisition is *measurable*;
and the combined-failure stress protocol with guarantee-violation measurement.

**Prohibited:** "first", "state of the art", "clinically safer", "guaranteed safe". The
guarantee is statistical, conditional on exchangeability, and **fails under shift** —
measuring where it fails is a result, not a disclaimer.

**If the main hypothesis fails** against the abstention and HOT-style baselines, the paper
centres on DERMA-STRESS and the guarantee-violation analysis.
See [what_did_not_work.md](docs/what_did_not_work.md).

### Limitations

1. **Synthetic text** (PAD-UFES-20 narratives are template-generated) — largest
   external-validity threat. MILK10k ships none.
2. **MILK10k has no patient IDs** — splits are lesion-level with ISIC-ID dedup; residual
   leakage risk stated.
3. **Rare classes** — MILK10k MAL_OTH has **9 lesions in total**; 5 of 11 classes fall
   below the ⌈1/α⌉=20 calibration points a class-conditional guarantee needs, and the
   implementation withholds one rather than fabricating it. The guarantee is on binary
   malignancy, and no rare-class claims are made.
6. **No centre-shift experiment** — the MILK10k release publishes no centre column, so
   shift is measured across datasets only. Corrected from an earlier claim of ours.
4. **Retrospective only**; no clinical-utility claim.
5. **Fairness** is reported per skin tone, but groups below `⌈1/α⌉` calibration points get
   **no** per-group guarantee rather than a fabricated one.

Citations verified 2026-10-11 against primary sources; status marks per entry in
[docs/07_literature.md](docs/07_literature.md), including corrections to earlier claims
of ours that were wrong.

**Licence/data:** MILK10k is CC-BY-NC (credit the MILK study team); PAD-UFES-20 is
CC BY 4.0. No patient data is committed to this repository.
