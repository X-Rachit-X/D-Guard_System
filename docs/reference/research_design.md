# Research Design

**Working title:** *Risk-Controlled, Cost-Aware Evidence Acquisition and Deferral for
Multimodal Dermatology AI*

Consolidates the v2 specification (three-action policy, MILK10k, LTT, AUGRC) with the
evidential-fusion mechanism. Where the two disagreed, the reasoning is recorded in §6.

---

## 1. Research question

> Given a clinical close-up photograph of a skin lesion, should the system **answer
> now**, **ask for a dermoscopic image first**, or **defer to a human** — and can it make
> that choice with a statistical guarantee on how often it misses a malignancy?

One sharp question, three actions, one guaranteed quantity. This replaces the earlier
framing ("can evidence-aware fusion discover which modality to trust"), which was a
statement about accuracy rather than about safety.

---

## 2. What the literature already settles

Each ingredient has prior art. None of it can be claimed as new.

| Established | By | Consequence for us |
|---|---|---|
| Recommending dermoscopy when the clinical image is insufficient | **HOT AI** (Mehta et al., *JEADV* 39(12):2121–2133, 2025) | "AI that asks for dermoscopy" is **not** novel. Must be implemented as baseline B5 |
| Conformal prediction in dermatology | Fayyad et al.; Lu et al.; Zoravar et al. | CP in skin lesions is not novel. We use it to **drive actions**, not to report uncertainty |
| Distribution-free risk control | Learn then Test (Angelopoulos, Bates, Candès, Jordan & Lei, *Ann. Appl. Stat.* 2025) | We do not invent risk-controlling calibration |
| Selective-classification evaluation | Traub et al., NeurIPS 2024 (AUGRC) | Use AUGRC; do not propose a new metric |
| Learned deferral | Mozannar & Sontag, ICML 2020 | Must be baseline B4 |
| Cost-aware modality acquisition | CAMA; MICCAI 2022 RL; ACO | Compare against oracle/greedy/random, not only fixed rules |
| Evidential fusion with Dempster–Shafer | **TMC** (Zongbo Han et al., ICLR 2021); EDL (Sensoy et al., NeurIPS 2018) | The fusion **operator** is not ours |
| Multimodal dermatology fusion | MetaBlock; DermFormer; CosCatNet; JI-ADF | Fusion is a component, not a contribution |

### The gap

> No work we have found combines **(a)** an explicit clinical cost model, **(b)** a
> finite-sample risk guarantee, **(c)** a three-action proceed/acquire/defer policy, and
> **(d)** combined-failure stress testing with measured guarantee breakdown — on a public
> paired-modality dermatology dataset.

Each column is covered by someone. The claim is the **row**. If full-text extraction
shows HOT AI already fills the cost and guarantee columns, the claim narrows to
DERMA-STRESS and the guarantee-violation analysis (§7).

---

## 3. Contributions

### A — Risk-controlled, cost-aware decision engine (EADE)

Expected-loss minimisation over PROCEED / ACQUIRE / DEFER under a declared asymmetric
cost matrix, with thresholds calibrated by Learn then Test so that

```
P( malignancy miss rate on proceeded cases <= alpha )  >=  1 - delta
```

distribution-free, under exchangeability. Formal statement in
[04_methods.md](../04_methods.md) §4–5. Implemented in `src/decision/policy.py` and
`src/calibration/ltt.py`.

### B — Evidential evidence layer (the mechanism)

Per-modality Dirichlet evidence combined by Dempster–Shafer, giving three things the
policy needs and the previous hand-tuned signals could not supply:

1. **Absence is free.** An absent modality is the vacuous opinion `(b=0, u=1)`, the
   **identity element** of DS combination. No mask, no renormalisation, no
   zero-division guard. Verified by test.
2. **Conflict is separated from ignorance.** DS yields conflict mass `C` *and*
   ignorance `u`. Jensen–Shannon divergence between posteriors — the v2 spec's
   consistency signal — provably cannot distinguish *two confident modalities
   disagreeing* from *two uninformative ones*, because both can give similar JS. The
   three-action policy needs that distinction: **conflict → defer to a human**,
   **ignorance → acquire more evidence**. Verified by
   `test_conflict_and_ignorance_are_independent_signals`.
3. **Epistemic uncertainty in one forward pass.** No ensemble. The v2 spec planned a
   3-seed deep ensemble while listing limited GPU as a risk; this is a 3× saving on the
   critical path.

Implemented in `src/evidence/evidential.py`.

### C — DERMA-STRESS

Versioned corruption protocol (18 conditions across availability / technical / semantic
/ compound), evaluated on **coverage, accuracy conditional on coverage, and guarantee
violation** — not accuracy alone. Already implemented and running
(`src/stress/`, `scripts/run_stress.py`).

### D — Acquisition-value analysis

Learned estimator of the benefit of acquiring dermoscopy, scored by **acquisition
efficiency** against oracle and random bounds at matched rates:

```
AE(r) = (G_policy - G_random) / (G_oracle - G_random)
```

`AE = 1` is oracle-level, `0` is no better than random, **negative is worse than
random** — a real possible outcome that the implementation reports rather than clips.
Implemented in `src/eval/selective.py`.

---

## 4. Hypotheses

Fixed before the final test run; recorded in [prereg.yaml](prereg.yaml).

> **H1 (risk control).** At a matched coverage, the EADE achieves a lower malignancy miss
> rate than confidence-threshold abstention calibrated to the same target.
>
> **H2 (acquisition efficiency).** At a matched acquisition rate, the EADE recovers a
> larger fraction of the oracle benefit than random, entropy-triggered (HOT-style) and
> greedy information-gain acquisition.
>
> **H3 (guarantee behaviour).** Under DERMA-STRESS, the realised risk degrades
> *measurably* rather than silently, and the violation is smaller than the baselines'.
>
> **H4 (mechanism).** Per-modality ignorance `u_m` is informative: cases with high `u_m`
> show measurably higher error for modality `m`. *Falsifiable — a flat reliability
> diagram means the evidence layer is decorative and H1's stated mechanism is wrong.*
>
> **H5 (conflict ≠ ignorance).** DS conflict mass separates genuine cross-modal
> contradiction from shared ignorance, where JS divergence cannot.

### Preliminary evidence for H1

Bridge mode over the legacy A–G ladder, disjoint patient-level splits, calibration on
432 held-out cases, evaluation on 464 (`scripts/run_eade.py`):

| | coverage | malignancy miss rate among proceeded |
|---|---|---|
| **EADE** | 0.433 | **0.0050** |
| B3 (max-softmax abstention), matched coverage | 0.433 | 0.0597 |

A 12× lower miss rate at identical coverage. Driven by the conformal ambiguity gate: 210
of 464 cases had a prediction set spanning benign and malignant classes, which
max-softmax confidence does not detect.

**Counter-evidence, reported:** on AUGRC the baseline is *better* (0.1276 vs 0.1634).
The ignorance score ranks cases less well than max-softmax even though the policy built
on it is far safer. Both numbers belong in the results table.

These are **machinery-verification** numbers, not generalisation estimates: the legacy
specialists were trained on data overlapping this test split
([evaluation_integrity.md](evaluation_integrity.md)).

---

## 5. Gap matrix

✓ present · ✗ absent · ? not determinable from the abstract (must come from full text).

| Work | Domain | Paired clinical+dermoscopy | Defer | Acquire | Explicit cost | Risk guarantee | Combined-failure stress | Subgroup |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| HOT AI | Derm | ✓ | ✓ (OOD alert) | ✓ | ? | ? | ? | ? |
| Fayyad et al. | Derm | ✗ | ✓ (sets) | ✗ | ✗ | ✓ (coverage) | ? | ? |
| Lu et al. | Derm | ✗ | ✓ (sets) | ✗ | ✗ | ✓ (group) | ✗ | ✓ |
| Zoravar et al. | Derm | ✗ | ✓ (sets) | ✗ | ✗ | ✓ | ? (shift) | ? |
| Roy et al. (HOD) | Derm | ✗ | ✓ | ✗ | ✓ (eval only) | ✗ | ? | ✓ |
| TMC (Han et al.) | General | n/a | ✗ | ✗ | ✗ | ✗ | ✓ (noise) | ✗ |
| DermFormer | Derm | ✓ | ? | ✗ | ✗ | ✗ | ✓ (corruption) | ? |
| JI-ADF | Derm (MILK10k) | ✓ | ? | ✗ | ✗ | ✗ | ? | ? |
| CAMA | General | n/a | ✗ | ✓ | ✓ (budget) | ✗ | ✗ | ✗ |
| Learning to defer | General | n/a | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ |
| LTT / CRC | General | n/a | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ |
| **Ours** | **Derm (MILK10k)** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

Provisional — cells drawn from abstracts must be confirmed by full-text extraction before
any novelty claim.

---

## 6. Decisions where the two plans disagreed

| Question | Chosen | Reason |
|---|---|---|
| Primary dataset | **MILK10k** | Every lesion has both modalities, so "acquire dermoscopy" is a real experiment with an oracle bound. PAD-UFES-20 has no dermoscopy at all, which made the earlier plan's `dermoscopy_image` slot permanently masked — designing around a dataset that cannot test the main question |
| Primary metric | **AUGRC** | Traub et al. show AURC is built on selective risk, which "is not suitable for aggregation over rejection thresholds", over-weights high-confidence failures, and changes rankings on 5 of 6 datasets. The earlier AURC/E-AURC recommendation was wrong |
| Decision rule | **Three-action expected-loss** | Clinically crisper than "evidence-aware fusion", and maps onto an action a clinician takes |
| Uncertainty source | **EDL + DS** (not a 3-seed ensemble) | One forward pass; and DS separates conflict from ignorance, which JS divergence cannot |
| Consistency signal | **DS conflict mass**, replacing JS divergence | §3-B-2 |
| Missing modalities | **Vacuous opinion** (DS identity), replacing modality dropout + masking | Correct marginalisation rather than imputation |
| Iterative diagnosis-adaptive fusion | **Dropped** | Nice derivation attached to a question nobody asked; the v2 plan dropped it and that was right |
| Image encoder | **DermLIP_PanDerm**, linear probe | Weights are public (verified); frozen ImageNet reaches only 0.477 macro-F1. Licence is CC-BY-NC-**ND**, so prefer probing over fine-tuning |
| PAD-UFES-20 | **Shift dataset** | Keeps the existing reproducible pipeline and honest baselines useful |
| Platform scope | **Trimmed** | The research core is ~20% of the v2 engineering plan. Frontend pages 1, 9, 10 cut pre-emptively; evaluation quality before features |

---

## 7. Honest novelty position

**Not novel:** evidential deep learning, Dempster–Shafer combination, conformal
prediction, Learn then Test, selective prediction, learning to defer, active modality
acquisition, multimodal fusion, recommending dermoscopy.

**Defensible:** their integration into a single risk-controlled three-action policy; the
application to a paired-modality dermatology dataset where acquisition is measurable; the
combined-failure stress protocol with guarantee-violation measurement.

**Permitted wording:** *"We propose and evaluate a risk-controlled, cost-aware
proceed/acquire/defer policy for multimodal dermatology, and measure its behaviour under
controlled evidence failures."*

**Prohibited:** "first", "state of the art", "clinically safer", "guaranteed safe",
"proven novel". The guarantee is statistical, conditional on exchangeability, and **fails
under shift** — measuring where it fails is a result, not a disclaimer.

**If H1 fails** against B3/B5, the paper centres on DERMA-STRESS and the
guarantee-violation analysis. That is a legitimate contribution and requires no change to
the rest of the protocol.
