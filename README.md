# D-GUARD

**Evidence-Aware Multimodal Fusion with Risk-Controlled Deferral for Skin Lesion Classification**

Dataset: **PAD-UFES-20** — 2298 smartphone clinical images, 1373 patients, 6 classes
(BCC, ACK, NEV, SEK, SCC, MEL).

> ⚠️ **Research prototype. Not clinically validated. Do not use for medical diagnosis.**
>
> 🔴 **The accuracy figures shipped with the original trained checkpoints are invalid.**
> The training code evaluated on a test set that was 100% training data and then rewrote
> predictions to hit hardcoded targets. Read
> **[docs/evaluation_integrity.md](docs/evaluation_integrity.md)** before citing any
> number produced before 2026-10-11.

---

## The question this project asks

A dermatology AI system does not fail because its classifier is a few points short. It
fails because it answers confidently when it should not — on a blurred photograph, an
empty symptom form, or a case where the image and the history contradict each other.

Most multimodal work optimises accuracy under *complete* evidence. This project asks a
different question:

> **Can a system measure how trustworthy each piece of evidence is, for this patient, and
> decline to answer when the evidence does not support an answer — with a statistical
> guarantee on how often it is wrong when it does answer?**

### The finding that motivates everything

Honest baselines on disjoint patient-level splits (three seeds, linear probes on frozen
features — [reproduce with `scripts/baseline_probe.py`](scripts/baseline_probe.py)):

| | configuration | macro-F1 (mean of 3 seeds) |
|---|---|---|
| A | image only | 0.477 |
| B | metadata only | 0.648 |
| C | text only | 0.414 |
| **D** | **image + metadata** | **0.653** |
| E | image + text | 0.559 |
| F | text + metadata | 0.587 |
| G | **all three** | 0.633 |

**Using all three modalities is worse than using the best two.** The text channel is
near-useless (0.414), and naive concatenation has no mechanism to discount it. A human
designer can spot this and drop text — but a deployed system must work it out per case,
because on *some* cases the narrative is the only informative modality.

That is the gap this project targets: **fusion that discovers which evidence to trust,
rather than being told.**

---

## Approach

Replace every hand-tuned quantity with one that has a derivation and a calibration
procedure.

| Component | Current | Proposed | Gain |
|---|---|---|---|
| Per-modality quality | `0.4·certainty + 0.3·agreement + 0.2·quality + 0.1·stability` | Dirichlet ignorance `u_m = K/S_m` from evidential heads | Fitted, not invented |
| Combining modalities | mask + renormalise | Dempster–Shafer; absent modality = vacuous opinion = **identity** | Correct marginalisation, not approximation |
| Conflict detection | `1 − mean pairwise JS`, threshold 0.3 | DS conflict mass `C`, calibrated | Separates *conflict* from *shared ignorance* — JS cannot |
| Adaptive re-weighting | 3 iterations, `max\|Δw\| < 0.05` | Coordinate ascent on one objective | Provable monotone convergence |
| Abstention | 4-level ladder, thresholds 0.80/0.50/0.30 | Conformal risk control, class-conditional | `P(risk ≤ α) ≥ 1−δ`, distribution-free |
| Error costs | symmetric thresholds | Explicit cost matrix + Chow rule | A missed melanoma ≠ a false alarm |
| Headline metric | accuracy | risk–coverage curve, AURC, E-AURC | Interpretable for a system that abstains |

Full derivations: **[docs/04_mathematical_foundations.md](docs/04_mathematical_foundations.md)**.
Why DS over tensor fusion, cross-attention, MoPoE and the rest:
**[docs/05_fusion_techniques.md](docs/05_fusion_techniques.md)**.

### Why Dempster–Shafer specifically

| Requirement from this dataset | How DS satisfies it |
|---|---|
| Modalities routinely absent | Vacuous opinion is the **identity element** — absence handled by the algebra |
| 2298 cases, MEL n=52 | One linear head per modality. Cheapest option that meets the other requirements |
| Quality varies per case | `u_m` is per-case and learned |
| Modalities genuinely contradict | `C` (conflict) is **separate from** `u` (ignorance) |
| Deferral needs calibrated uncertainty | `u` and `C` feed the decision rule directly |

It is the only operator in the design space that satisfies all five — with fewer
parameters than the gated fusion already in the codebase.

---

## Architecture

```
          image ──► encoder (fine-tuned) ──► EvidentialHead ──► Dir(α_img) ─┐
       metadata ──► MLP (train-fit)      ──► EvidentialHead ──► Dir(α_meta)─┤
           text ──► BERT [CLS]           ──► EvidentialHead ──► Dir(α_text)─┤
     [absent]   ─────────────────────────────────────────────► vacuous ─────┤
                                                                            │
                               Dempster–Shafer fold (⊕, associative) ◄──────┘
                                            │
                               belief b · ignorance u · conflict C
                                            │
                       EM refinement (diagnosis-adaptive, monotone)
                                            │
                       Risk-controlled deferral (Learn-then-Test + Chow)
                                            │
                  ┌─────────────────────────┼──────────────────────┐
             predict ŷ              defer: CONFLICT          defer: IGNORANCE
                                  (→ human review)        (→ acquire evidence)
                                                           defer: COST
```

Current vs target, and the incremental migration path:
**[docs/03_architecture.md](docs/03_architecture.md)**.

---

## Status

| Area | State |
|---|---|
| End-to-end pipeline (encode → specialists → evidence → fusion → consensus → report) | ✅ working |
| FastAPI `/predict`, two Streamlit apps | ✅ working |
| DERMA-STRESS robustness benchmark, 18 conditions | ✅ working |
| Test suite | ✅ 236 tests, 70% coverage |
| Patient-level disjoint splits | ✅ asserted at runtime |
| **Reproducible feature extraction** | ✅ manifest + `--verify`, cosine 1.000000 |
| **Honest baseline ladder** | ✅ 3 seeds — table above |
| Fine-tuned image encoder | ⬜ needs GPU → [Kaggle guide](docs/08_kaggle_training.md) |
| Evidential heads + DS fusion | ⬜ designed, Kaggle script written |
| Conformal risk control | ⬜ designed; runs on returned predictions |
| **Original checkpoint metrics** | ❌ **invalid** — see [evaluation_integrity.md](docs/evaluation_integrity.md) |

---

## Quickstart

```bash
pip install -e ".[dev]"
# Windows/Anaconda: set KMP_DUPLICATE_LIB_OK=TRUE, or torch and MKL both load OpenMP and abort
```

```bash
# 1. reproducible features (~4.5 min CPU), then verify they regenerate exactly
python scripts/build_features.py --fit-metadata-on-train
python scripts/build_features.py --verify data/processed/features_v2.pt

# 2. honest baselines
python scripts/baseline_probe.py --seeds 42 7 2026

# 3. end-to-end demo of the running pipeline
python scripts/demo.py

# 4. robustness benchmark
python scripts/run_stress.py --limit 50

# 5. tests
pytest tests -q
```

Serving:

```bash
python src/main.py                                    # FastAPI, docs at :8000/docs
streamlit run app/patient_app.py --server.port 8501
streamlit run app/clinician_console.py --server.port 8502
```

GPU training runs on Kaggle: **[docs/08_kaggle_training.md](docs/08_kaggle_training.md)**
with [`notebooks/kaggle_train_dguard.py`](notebooks/kaggle_train_dguard.py)
(smoke-tested end-to-end on CPU).

---

## Documentation

| Doc | Contents |
|---|---|
| [00_index.md](docs/00_index.md) | Reading order |
| [01_system_analysis.md](docs/01_system_analysis.md) | What exists, project lineage, honest reference numbers |
| [02_research_gaps_and_contributions.md](docs/02_research_gaps_and_contributions.md) | Literature position, the gap, C1–C7, novelty assessment |
| [03_architecture.md](docs/03_architecture.md) | Current + target architecture, migration path |
| [04_mathematical_foundations.md](docs/04_mathematical_foundations.md) | Every quantity, derived |
| [05_fusion_techniques.md](docs/05_fusion_techniques.md) | Fusion design space and the justified choice |
| [06_retrieval_design.md](docs/06_retrieval_design.md) | kNN + literature retrieval, with the evidence firewall |
| [07_experimental_protocol.md](docs/07_experimental_protocol.md) | Hypotheses, ablations, statistics, 14-week plan |
| [08_kaggle_training.md](docs/08_kaggle_training.md) | GPU training and what to send back |
| [09_literature_and_benchmarks.md](docs/09_literature_and_benchmarks.md) | **Verified citations, comparable published numbers, two corrections** |
| [evaluation_integrity.md](docs/evaluation_integrity.md) | **Audit findings — read before trusting any old number** |
| [limitations.md](docs/limitations.md) · [what_did_not_work.md](docs/what_did_not_work.md) | Honest negative results |

---

## Honest positioning

**Not novel:** evidential deep learning, Dempster–Shafer combination, conformal risk
control, multimodal fusion, selective prediction. All established.

**Defensible as a contribution:** their integration into a single evidence-management
layer; application to patient-authored, routinely-incomplete dermatology evidence; a
systematic corruption benchmark; and the empirical question of whether principled
evidence management actually buys safety over a well-tuned static baseline.

**If the answer is no** — if a tuned image+metadata model matches the evidential
system — that is a publishable negative result, and
[what_did_not_work.md](docs/what_did_not_work.md) commits the project to reporting it.

### Known limitations

1. **Synthetic text.** Narratives are template-generated, not patient-written. Any claim
   about patient-authored text is a claim about a simulation. Largest external-validity
   threat.
2. **One dataset, one centre.** No cross-site evidence.
3. **MEL n=52** → ~8–14 per test split. Melanoma-specific conclusions are not
   supportable, and class-conditional guarantees are not attainable for MEL at α=0.1.
4. **No dermoscopy.** The diagnosis-adaptive premise cannot be fully tested.
5. **Retrospective only.** No clinical-utility claim.
6. **Fairness not addressed.** Fitzpatrick distribution is a known confounder, out of
   scope here.

---

## Where we stand against the literature

Key citations were **verified against primary sources on 2026-10-11** —
[docs/09_literature_and_benchmarks.md](docs/09_literature_and_benchmarks.md), which also
records two claims of mine that were **wrong**.

| | Ours (patient-level, 3 seeds) | Published |
|---|---|---|
| metadata only | macro-F1 0.648 | XGBoost **0.746** ([TRACE](https://arxiv.org/pdf/2411.08701)) |
| image + metadata | macro-F1 0.653 | CNN+MetaBlock BACC 0.770 |
| 6-class | macro-F1 0.653 | DiffMIC BACC 0.646 |

Ours sit below published metadata results, as expected: they are deliberately weak
*frozen-feature linear probes*, on a **stricter patient-level split** (several published
results do not state patient-level grouping, and this dataset has 1641 lesions from 1373
patients — an image-level split leaks a patient's other lesions). There is clear headroom.

**Biggest available win.** The image pathway is encoder-limited (0.477 with frozen
ImageNet). **DermLIP_PanDerm** — the dermatology foundation model from
[PanDerm, *Nature Medicine* 2025](https://arxiv.org/abs/2410.15038), self-supervised on
>2M skin images — has **public weights** at
[`redlessone/DermLIP_PanDerm-base-w-PubMed-256`](https://huggingface.co/redlessone/DermLIP_PanDerm-base-w-PubMed-256),
which is the exact repo id `proj.md` specified. The original code never loaded it because
**the repo is gated**. Supported in the Kaggle script via
`IMAGE_BACKBONE="dermlip_panderm"`.

Caveats: gated (needs HF token), non-commercial licence labelled inconsistently
(CC-BY-NC vs CC-BY-NC-**ND** — "ND" arguably forbids fine-tuning, so prefer linear
probing), and **check whether PAD-UFES-20 is in PanDerm's pretraining corpus** before
claiming a clean result.

Verified: PAD-UFES-20 (Pacheco et al., *Data in Brief* 32:106221, 2020) · MetaBlock
(Pacheco & Krohling, *IEEE JBHI* 25(9):3554–3563, 2021) · Evidential Deep Learning
(Sensoy, Kaplan & Kandemir, NeurIPS 2018) · Trusted Multi-View Classification (Zongbo Han
et al., ICLR 2021) · Learn then Test (Angelopoulos, Bates, Candès, Jordan & Lei, *Ann.
Appl. Stat.* 2025) · PanDerm (Yan et al., *Nature Medicine* 2025).
Still unverified (low-risk): Dempster, Shafer, Jøsang, Chow, and the fusion-survey
citations.

**Licence / data:** PAD-UFES-20 is CC BY 4.0. No patient data is committed to this
repository.
