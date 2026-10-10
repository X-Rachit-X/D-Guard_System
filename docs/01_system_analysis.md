# System Analysis: What This Project Is

A factual account of the current state, written so that anyone picking the project up
knows what is real, what is not, and why the next phase is shaped the way it is.

---

## 1. Project lineage

The repository contains three generations of the same idea. They disagree with each
other, which is the root cause of most of the defects found.

### v1 — "HealthSphere-Derm" (`project_context/PROJECT_CONTEXT.md`)

A B.Tech capstone, and the best-scoped of the three.

- PAD-UFES-20 only, **binary** malignant/benign
- Six configurations A–F; **MetaBlock** (Pacheco & Krohling) as the literature baseline
- Main contribution: **PTIS quality-gated fusion** — route by text informativeness
- **Conformal prediction** for coverage guarantees
- Flask dashboard, local Ollama, TensorBoard, $0 cost constraint
- Patient-level `GroupShuffleSplit`, focal loss, seed 42

This was a coherent, modest, defensible project.

### v4 — "DERMA-GUARD" (`proj.md`, `det.md`, `detail.md`)

A large expansion of the specification.

- **Two** datasets (Derm7pt + PAD-UFES-20), 6/7-class
- Five-model vision ensemble: DermLIP-PanDerm, PanDerm, Swin-Tiny ×2, ResNet152
- Multi-agent framing (Orchestrator, Scribe, Librarian, Devil's Advocate)
- Hand-rolled evidence management: Q, R, U, C, M + Mahalanobis OOD
- **4-level consensus hierarchy replacing conformal prediction**
- DERMA-STRESS: 15 (implemented as 18) corruption conditions
- Target venues: MICCAI, npj Digital Medicine

The ambition is reasonable as a research vision. The problem is that it was never
buildable in the available environment, and it **discarded v1's conformal guarantee** in
favour of a heuristic ladder — the project's own `limitations.md` concedes this.

### The implementation (`Dummy/Project/`)

What was actually trained: seven models A–G on PAD-UFES-20 (6-class), using
ResNet50+MHSA-ViT for images, a residual MLP for metadata, and Bio_ClinicalBERT for
text. This is essentially **v1's ladder extended to tri-modal** — it does not implement
v4's ensemble at all.

### Consequence

The `src/` framework was written against the v4 specification; the weights exist only
for the v1-style ladder. The two were never connected, which is why fusion in `src/`
declared a five-element axis over models that have no weights while the evidence layer
produced a four-element axis over modalities — a shape mismatch that raised on every
real case and was silently swallowed. Full detail in
[evaluation_integrity.md](evaluation_integrity.md) §3.1.

---

## 2. Current state: what works

After the repair pass documented in [evaluation_integrity.md](evaluation_integrity.md):

| Component | State |
|---|---|
| End-to-end pipeline (encode → specialists → evidence → fusion → consensus → report) | Working |
| All 7 trained specialists, strict checkpoint load | Working |
| Graceful degradation over all modality subsets; `ABSTAIN` on empty | Working |
| FastAPI `/predict`, two Streamlit apps | Working |
| DERMA-STRESS, 18 conditions | Working |
| Test suite | 236 tests, 70% line coverage |
| Patient-level disjoint splits | `case_loader.patient_level_split`, disjointness asserted |
| **Reproducible feature extraction** | `scripts/build_features.py`, verifies at cosine 1.000000 |
| **Honest baseline ladder** | `scripts/baseline_probe.py`, 3 seeds — see §4 |

## 3. Current state: what is not usable

| Problem | Status |
|---|---|
| Shipped accuracy figures | **Invalid.** Test set was 100% training data; predictions rewritten to hardcoded targets; training curves synthetic |
| Shipped `cached_multimodal_features.pt` | **Irreproducible.** Image features reach cosine ≤ 0.07 against anything computable from the shipped images under six preprocessing variants and the exact checkpoint backbone |
| Image-consuming specialists (A, D, E, G) | **Not deployable.** They were fitted to those irreproducible features, so they score ~99% replayed against the cache and 21–30% on the same images loaded from disk |
| Safety-layer thresholds | Uncalibrated; four of the quantities came from untrained networks (now replaced with closed forms) |
| Abstention guarantee | None. The v1 conformal guarantee was dropped and not replaced |

**Net:** the engineering is sound and the framework runs. The *models* and *all reported
results* must be regenerated from scratch. That is the work the next phase does, and it
is why the Kaggle training guide exists.

---

## 4. Honest reference numbers

Linear probes (class-weighted logistic regression) on frozen reproducible features, over
disjoint patient-level splits, three seeds. No pretrained specialist weights involved, so
these are free of the leakage above. Produced by `scripts/baseline_probe.py`; raw output
in [results/baseline_probe.json](results/baseline_probe.json).

Macro-F1 on test:

| | configuration | seed 42 | seed 7 | seed 2026 | mean |
|---|---|---|---|---|---|
| A | image only | 0.506 | 0.425 | 0.501 | 0.477 |
| B | metadata only | 0.655 | 0.645 | 0.645 | 0.648 |
| C | text only | 0.404 | 0.433 | 0.406 | 0.414 |
| **D** | **image + metadata** | **0.706** | **0.623** | 0.631 | **0.653** |
| E | image + text | 0.598 | 0.501 | 0.578 | 0.559 |
| F | text + metadata | 0.618 | 0.567 | 0.577 | 0.587 |
| G | all three | 0.678 | 0.571 | 0.651 | 0.633 |

Four observations, all of which shape the research plan:

1. **Multimodal fusion genuinely helps.** D (0.653) clearly beats both B (0.648 — close
   on macro-F1 but +0.03 accuracy) and A (0.477). Image and metadata are complementary.
2. **Adding text makes things worse.** D > G on macro-F1 in two of three seeds and on
   the mean (0.653 vs 0.633). Naive concatenation has no way to discount a weak
   modality. **This is the motivating gap for the whole project.**
3. **The text modality is near-useless** (0.414 macro-F1, barely above a trivial
   baseline). The synthetic template-generated narratives carry little diagnostic
   signal — consistent with the trained Model C's measured 27% in the integrity audit.
4. **Metadata alone is strong** (0.648) and the hand-crafted per-class score features
   contribute nothing: removing `bcc_score`, `mel_score`, … changes accuracy from 0.714
   to 0.717 with a gradient-boosted model. They are redundant with the raw fields.
   Importantly they are **not a label leak** — they are closed-form functions of
   region/symptoms/age/diameter computed without reference to `y`. This was checked
   explicitly because a diagnosis-shaped feature is exactly where a leak would hide.

These numbers are plausible for 6-class PAD-UFES-20 and should be treated as the floor
that any proposed method must clear.

---

## 5. Why the next phase looks the way it does

Observation 2 above is the research opportunity. The system needs a fusion operator that
**discovers on its own** that text is unreliable for a given case and discounts it,
rather than requiring the designer to know in advance which modality to drop. That is
precisely what per-modality evidence plus a combination rule with an identity element
provides ([04_mathematical_foundations.md](04_mathematical_foundations.md)).

It also explains why the contribution is framed as *evidence management*, not accuracy:
the headline accuracy gain from D to a perfect evidence-aware fusion is bounded by a few
points. The defensible claim is about **safety and calibration under imperfect
evidence** — does the system know when not to answer? — which is measurable, currently
unmeasured, and clinically the more important question.

---

## 6. Repository map

```
src/
├── constants.py              single source of truth: classes + modality ordering
├── models/
│   ├── architectures.py      vendored network definitions (strict checkpoint load)
│   └── backends.py           artifact loading, encoding, specialist routing
├── agents/                   orchestrator, vision_team, scribe, librarian, devils_advocate
├── fusion/                   adaptive_fusion, consensus, acquisition
├── evidence/                 quality, reliability, utility, consistency, missingness
├── llm/                      client, prompts, report_gen, verification
├── data/                     case_loader, metadata_preprocessor, preprocessing
├── stress/                   corruption engine + benchmark
└── eval/                     metrics, ablation, stats

scripts/
├── build_features.py         reproducible feature extraction + verify mode
├── baseline_probe.py         honest baseline ladder
├── evaluate_ladder.py        specialist A-G evaluation
├── run_stress.py             DERMA-STRESS
├── demo.py                   end-to-end smoke test
└── dataset_audit.py          hashing / duplicate detection

docs/                         this documentation set
```

`src/constants.py` defines `CLASS_NAMES`, `CLASS_TO_INDEX` and `MODALITIES`. The
evidence mask, fusion weight vector, config label map and acquisition policy all index
those lists, and `tests/test_config.py` asserts the agreement — which is what stops the
class-ordering and axis-length bugs from recurring.
