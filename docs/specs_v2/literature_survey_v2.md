# DERMA-GUARD — Literature Survey and Base-Paper Dossier (v2.0)

**Project:** DERMA-GUARD
**Working title:** *Risk-Controlled, Cost-Aware Evidence Acquisition and Deferral for Multimodal Dermatology AI*
**Document version:** 2.0 — aligned with Project Specification v2.0
**Supersedes:** Literature Survey v1.0 (fusion-centred)
**Status:** Research-planning document. Hypotheses are fixed; novelty and effectiveness are not yet demonstrated.

---

## Contents

0. What changed from v1.0
1. How to use this document
2. Verification legend
3. DERMA-GUARD in one page
4. Base papers (re-selected) and how we use them
5. Literature by category
   - A. Dermatology triage, deferral and acquisition (closest task)
   - B. Selective prediction, calibration and risk control (our method)
   - C. Learning to defer (alternative formulation)
   - D. Active feature / modality acquisition (our acquisition question)
   - E. Multimodal dermatology fusion (backbone comparators)
   - F. Uncertainty, OOD and shift in dermatology
   - G. Robustness, fairness, data quality and label uncertainty
   - H. Datasets and clinical motivation
   - I. Language-model and text-related dermatology work (context only)
6. Gap matrix
7. What the literature already establishes
8. Research gaps we target
9. Experiment matrix (mapped to v2 baselines)
10. Dataset strategy
11. Implementation decisions informed by the literature
12. Permitted and prohibited claims
13. Reading order and extraction template
14. Reference list

---

## 0. What changed from v1.0

| # | Change | Reason |
|---|--------|--------|
| 1 | **Base papers re-selected.** HOT AI is now the primary base. Learn-then-Test / conformal risk control, AUGRC and active modality acquisition join as methodological bases. DermFormer and CosCatNet move to "backbone comparators". | Our contribution is the decision policy *on top of* a model, not a new fusion architecture. |
| 2 | **Three new categories added:** selective prediction & risk control (B), learning to defer (C), active feature/modality acquisition (D). | These are our closest *methodological* competitors and were missing. |
| 3 | **Gap matrix added** (Section 6). | One table shows exactly where our niche lies. |
| 4 | **MILK10k, the JAAD 2026 reader study and JI-ADF added.** Derm7pt demoted to secondary. | Matches Specification v2.0 dataset decision. |
| 5 | **Verification status added to every entry.** | Several 2026 entries had not been checked; one could not be located. |
| 6 | **Specific corrections:** HOT AI journal details and its differently-titled arXiv preprint; QA-MoE evaluated on non-dermatology tasks and has a same-name ACL 2026 paper; UAGF is a single-modality multi-backbone method; DermFusion could not be located. | Accuracy. |
| 7 | **Duplicates removed** (DDI listed twice; synthesized clinical notes listed twice). | Correct paper count. |
| 8 | **Experiment matrix remapped** to baselines B1–B8 and v2 metrics (coverage at guaranteed risk, AUGRC, acquisition efficiency). | Consistency with the specification. |

---

## 1. How to use this document

This is a project-oriented survey, not a systematic review and not a guarantee of novelty.

For each paper we record:
- **Problem** it addresses.
- **Method** at a high level.
- **Relevance** to DERMA-GUARD.
- **Use** — how it shapes our implementation or experiments.
- **Rules out** — what we can no longer claim as new.
- **Status** — verification level (Section 2).

**Evidence rule:** an abstract or publisher summary is not enough to claim that a method is *absent* from a paper. Before any novelty claim, read the full paper, supplement and code. A cell in the gap matrix (Section 6) marked from an abstract alone is provisional until the full-text extraction is done.

**Result labelling rule:** every number we report is tagged `paper-reported`, `our-reimplementation`, or `our-method`.

**Freshness:** includes work surfaced through October 2026. Re-check publication status, final issue details, code availability and licences before the formal paper.

---

## 2. Verification legend

| Mark | Meaning |
|------|---------|
| ✅ | Existence, venue and the key claim used here were checked during this revision. |
| 📄 | Carried over from the v1.0 team survey with a DOI/link; **not independently checked in this revision.** Open the DOI before citing. |
| ⚠️ | Could not be located during this revision. Do not cite until a team member confirms it. |
| 📚 | Long-established reference; bibliographic details are standard. Still confirm before formal citation. |

---

## 3. DERMA-GUARD in one page

### 3.1 Research question

> Can a cost-aware, risk-controlled proceed/acquire/defer policy achieve a better reliability–coverage–cost trade-off than confidence-threshold abstention, learning-to-defer and HOT-style "recommend dermoscopy when uncertain" rules — on clean data and under controlled evidence failures?

### 3.2 Contributions

- **A — Risk-controlled decision engine (EADE).** Expected-loss rule over `PROCEED` / `ACQUIRE` / `DEFER` with an asymmetric clinical cost matrix; thresholds calibrated by Learn-then-Test so that the malignancy miss rate on proceeded cases is bounded with high probability (under exchangeability).
- **B — DERMA-STRESS.** Reproducible stress protocol (degradation, missing modality, contradictory metadata, shift, leave-class-out OOD, combined and operational failures), including measurement of **guarantee violation** under shift.
- **C — Acquisition-value analysis.** Learned estimate of the benefit of acquiring dermoscopy, evaluated against oracle and random acquisition at matched rates (acquisition efficiency).

### 3.3 What is not claimed as new

Multimodal fusion, uncertainty estimation, conformal prediction, OOD detection, abstention, learning to defer, active feature acquisition, recommending dermoscopy, RAG and audit trails all have prior art. The research gamble is the **combination** — cost model + statistical guarantee + three-action policy + combined-failure evaluation — in dermatology, and whether it measurably helps.

---

## 4. Base papers (re-selected)

We use a **set** of complementary base papers. Each covers one part of the project.

| Role | Base paper | Status |
|------|-----------|--------|
| Closest task (dermatology acquire/defer) | **HOT AI** — Mehta et al., JEADV 2025 | ✅ |
| Core method: risk-controlled thresholds | **Learn then Test** — Angelopoulos et al. | 📚 |
| Core method: conformal risk control | **Conformal Risk Control** — Angelopoulos et al. | 📚 |
| Evaluation metric | **AUGRC** — Traub et al., NeurIPS 2024 | ✅ |
| Acquisition formulation | **CAMA** — Rheude et al. (cohort-based active modality acquisition) | ✅ |
| Dataset and clinical motivation | **MILK10k** (ISIC) + **Müller et al., JAAD 2026** | ✅ |
| Backbone/fusion comparator | **DermFormer** and **Zuo et al. / CosCatNet** | 📄 |

### 4.1 HOT AI — primary base (closest task)

**Citation:** Mehta D., Primiero C., Betz-Stablein B., Nguyen T.D., Gal Y., Bowling A., Haskett M., Sashindranath M., Bonnington P., Mar V., Soyer H.P., Ge Z. *Multi-task AI models in dermatology: Overcoming critical clinical translation challenges for enhanced skin lesion diagnosis.* Journal of the European Academy of Dermatology and Venereology, 39(12), 2121–2133, 2025. DOI 10.1111/jdv.20551. ✅
**Preprint:** *Revamping AI Models in Dermatology: Overcoming Critical Challenges for Enhanced Skin Lesion Diagnosis*, arXiv 2311.01009 (different title — cite the journal version, mention the preprint once).

**Method (from abstract):** From a clinical image, produces (1) a hierarchical prediction, (2) an out-of-distribution alert, and (3) a recommendation for dermoscopy when the clinical image alone is insufficient; when followed, clinical and dermoscopic images are fused for the final diagnosis.

**Use:**
- Implement a **HOT-style baseline (B5)**: request dermoscopy when an uncertainty score exceeds a validation-tuned threshold.
- Compare triggers, action space and evaluation endpoints.
- Full-text extraction must answer: Is there an explicit cost? A statistical guarantee? A defer action separate from the OOD alert? Combined-failure tests? Which dataset (appears to be an in-house cutaneous lesion dataset)?

**Rules out:** "AI that recommends dermoscopy" and "hierarchical triage with OOD alerts" as new ideas.

### 4.2 Learn then Test & Conformal Risk Control — core method bases

**Citations:** Angelopoulos, Bates, Candès, Jordan, Lei — *Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control* (arXiv 2110.01052). Angelopoulos, Bates, Fisch, Lei, Schuster — *Conformal Risk Control* (arXiv 2208.02814). 📚

**Method:** Choose model/decision thresholds on a held-out calibration set so that a user-chosen risk is controlled with finite-sample, distribution-free guarantees (valid under exchangeability).

**Use:** Calibrate the EADE PROCEED threshold so that malignancy miss rate on auto-proceeded cases ≤ α with probability ≥ 1 − δ. Fixed-sequence testing over a threshold grid.

**Rules out:** inventing risk-controlling calibration. Our contribution is applying it to a three-action, cost-aware dermatology policy and **measuring where it breaks under shift**.

### 4.3 AUGRC — evaluation base

**Citation:** Traub J., Bungert T.J., Lüth C.T., Baumgartner M., Maier-Hein K.H., Maier-Hein L., Jäger P.F. *Overcoming Common Flaws in the Evaluation of Selective Classification Systems.* NeurIPS 2024 (Spotlight). arXiv 2407.01032. ✅

**Method:** Defines five requirements for multi-threshold selective-classification metrics, shows existing metrics (including AURC) fail some, and proposes AUGRC, interpretable as the average risk of undetected failures. Reports that metric rankings change on most of their benchmark datasets.

**Use:** AUGRC is our primary multi-threshold metric; AURC reported only for comparability.

**Rules out:** proposing a new general-purpose selective-classification metric.

### 4.4 CAMA — acquisition base

**Citation:** Rheude T. et al. *Cohort-Based Active Modality Acquisition.* arXiv 2505.16791 (v3 December 2025). ✅

**Method:** Formalises test-time selection of which samples should receive an additional modality under limited resources; introduces upper-bound heuristics as performance ceilings.

**Use:** Motivates our oracle upper bound and matched-rate comparisons; possible cohort-budget variant of our acquisition experiment.

**Rules out:** test-time modality acquisition as a new problem.

### 4.5 MILK10k + JAAD reader study — data and motivation base

**MILK10k (ISIC):** 10,480 images from 5,240 lesions, each with a paired clinical close-up and dermoscopic image; ~95.7% histopathology-confirmed; metadata include age bins, sex, site, a 6-level skin-tone field (deliberately not Fitzpatrick), and MONET concept probabilities; 11 broad diagnostic categories; no patient identifiers listed. A 479-lesion hidden-label benchmark is run as an ISIC challenge. DOI 10.34970/648456. ✅

**Müller C. et al., JAAD 2026** (DOI 10.1016/j.jaad.2026.03.069): online reader study (283 participants, 1,567 paired images from MILK10k) quantifying standalone and complementary value of dermoscopy and clinical close-ups. Dermoscopy alone had higher sensitivity for malignancy; clinical close-ups had higher specificity; each added independent value. ✅

**Use:** primary dataset; introduction motivation for the acquisition question.

### 4.6 Backbone comparators — DermFormer and Zuo et al. / CosCatNet

Moved from "primary base" to "backbone comparator": they are fusion models whose outputs our policy can sit on top of. Details in Section 5.E.

---

## 5. Literature by category

### A. Dermatology triage, deferral and acquisition (closest task)

#### A1. HOT AI — see §4.1. ✅

#### A2. Vision Transformer-Based Uncertainty Quantification for Triaging Skin Lesions: a Probabilistic Framework for Automated Biopsy Recommendation 📄
*Frontiers in Bioengineering and Biotechnology*, 2026. DOI 10.3389/fbioe.2026.1859844
- **Problem/method:** uncertainty quantification for triage and probabilistic biopsy recommendation.
- **Use:** compare triage logic and endpoints; candidate inspiration for B3/B6 baselines.
- **Rules out:** uncertainty-based triage as new.

#### A3. Does Your Dermatology Classifier Know What It Doesn't Know? (Roy et al., hierarchical outlier detection) 📚
- **Method:** frames rare-condition detection as OOD with multiple abstention classes; subgroup analysis; introduces a clinical cost metric.
- **Use:** precedent for cost-weighted evaluation of abstention in dermatology; leave-class-out design (DERMA-STRESS family E).
- **Rules out:** "first cost-aware evaluation of abstention in dermatology".

#### A4. DermFusion 2026 ⚠️
Listed in v1.0 as *Scientific Reports* 2026, DOI 10.1038/s41598-026-72311-6, with uncertainty-threshold referral. **Could not be located during this revision.** A team member must open the DOI; if it resolves, move it here as a high-priority competitor; if not, delete it.

---

### B. Selective prediction, calibration and risk control (our method)

#### B1. Geifman & El-Yaniv — Selective Classification for Deep Neural Networks (NeurIPS 2017) 📚
- **Use:** risk–coverage framework; selective risk at fixed coverage.

#### B2. Guo et al. — On Calibration of Modern Neural Networks (ICML 2017) 📚
- **Use:** temperature scaling; motivation for calibration before thresholding.

#### B3. Angelopoulos & Bates — A Gentle Introduction to Conformal Prediction (arXiv 2107.07511) 📚
- **Use:** teaching reference for the team and the report.

#### B4. Angelopoulos et al. — RAPS: Uncertainty Sets for Image Classifiers using Conformal Prediction (arXiv 2009.14193) 📚
- **Use:** prediction-set construction; "mixed benign/malignant set → do not proceed" rule.

#### B5. Learn then Test — see §4.2. 📚
#### B6. Conformal Risk Control — see §4.2. 📚
#### B7. AUGRC (Traub et al., NeurIPS 2024) — see §4.3. ✅

#### B8. Fayyad et al. — Empirical Validation of Conformal Prediction for Trustworthy Skin Lesions Classification ✅
arXiv 2312.07460; journal version listed in v1.0 as *Computer Methods and Programs in Biomedicine* 2024, DOI 10.1016/j.cmpb.2024.108231 (📄 for the journal details).
- **Method:** compares conformal prediction, Monte Carlo dropout and evidential deep learning on skin-lesion and blood-cell datasets; finds conformal prediction most robust.
- **Use:** precedent for CP in dermatology; our difference is using CP/LTT to **drive actions**, not only to quantify uncertainty.
- **Rules out:** "first conformal prediction for skin lesions".

#### B9. Lu et al. — Fair Conformal Predictors for Applications in Medical Imaging (arXiv 2109.04392) ✅
- **Method:** group-conditional conformal prediction evaluated on dermatology data with skin type as the group attribute; rule-in/rule-out use cases.
- **Use:** basis for our per-skin-tone calibration.
- **Rules out:** group-conditional conformal by skin type as new.

#### B10. Zoravar et al. — Domain Adaptive Skin Lesion Classification via Conformal Ensemble of Vision Transformers (arXiv 2505.15997) ✅
- **Method:** ensemble of ViTs trained across dermatology datasets with conformal calibration under domain shift.
- **Use:** comparison point for conformal behaviour under shift (DERMA-STRESS family D).

#### B11. An Empirical Analysis of Calibration and Selective Prediction in Multimodal Clinical Condition Classification (CHIL 2026, PMLR v333) 📄
- **Use:** class-dependent miscalibration affects selective decisions → report class-wise calibration; baseline thresholds must be calibrated, not raw.

---

### C. Learning to defer

#### C1. Mozannar & Sontag — Consistent Estimators for Learning to Defer to an Expert (ICML 2020) 📚
- **Method:** trains classifier and deferral rule jointly with a consistent surrogate loss.
- **Use:** baseline **B4**.
- **Rules out:** learned deferral as new.

*(Action item: search for learning-to-defer papers applied to dermatology during the novelty audit; none are recorded here yet.)*

---

### D. Active feature / modality acquisition

#### D1. CAMA — see §4.4. ✅

#### D2. Valancius et al. — Acquisition Conditioned Oracle for Nongreedy Active Feature Acquisition (arXiv 2302.13960) ✅
- **Method:** nonparametric oracle-based approach to AFA; discusses weaknesses of RL, generative-surrogate and greedy approaches.
- **Use:** motivates oracle and greedy baselines (B6, B8).

#### D3. Shim, Hwang & Yang — Joint Active Feature Acquisition and Classification with Variable-Size Set Encoding (NeurIPS 2018) ✅
- **Method:** RL agent decides at test time to stop-and-predict or acquire a feature, cost-sensitively.
- **Use:** related work; RL acquisition is listed as future work for us.

#### D4. Reinforcement Learning for Active Modality Selection during Diagnosis (MICCAI 2022) ✅
- **Method:** RL policy recommends the next imaging modality, optimising accuracy minus acquisition cost (cardiac imaging).
- **Use:** shows modality-level cost-aware acquisition in medical imaging predates us.

#### D5. SAGEAgent — Cost-Aware Modality Acquisition in Multimodal Survival Prediction (arXiv 2607.09521) ✅
- **Method:** training-free LLM agent for sequential modality acquisition respecting clinical ordering; burden scored via multi-criteria analysis.
- **Use:** related work; shows the area is active in 2026. Note name collision with SAGE (F1).

#### D6. SDM-Q — Cost-Aware Staged Decision-Making for Multi-Omics Classification (arXiv 2605.31014) ✅
- **Use:** related work for staged, cost-aware acquisition with terminal reward.

---

### E. Multimodal dermatology fusion (backbone comparators)

#### E1. DermFormer — Nested Multi-Modal Vision Transformers for Robust Skin Cancer Detection 📄
*Pattern Analysis and Applications*, 2025. DOI 10.1007/s10044-025-01572-0
- **Method:** nested multimodal transformer over clinical, dermoscopic and metadata inputs; robustness tests under corruption.
- **Use:** corruption-protocol reference; possible fusion backbone.
- **Rules out:** multimodal fusion and corruption testing in dermatology as new.

#### E2. Zuo et al. / CosCatNet — Multi-Stage Multi-Modal Learning with Adaptive Multimodal Fusion 📄
*Artificial Intelligence in Medicine*, 2025. DOI 10.1016/j.artmed.2025.103091. Code: github.com/Zuo-Lihan/CosCatNet-Adaptive_Fusion_Algorithm
- **Method:** intermediate fusion of two image modalities, cosine similarity, uncertainty-based late fusion with metadata.
- **Use:** adaptive-fusion comparator; check whether code runs on MILK10k.
- **Rules out:** uncertainty-based adaptive fusion as new.

#### E3. JI-ADF — Joint-Individual Learning with Adaptive Decision Fusion (arXiv 2604.27343) ✅
- **Method:** multimodal fusion trained and evaluated on MILK10k paired images, following the official hidden test.
- **Use:** fusion reference **on our exact dataset**; label any numbers as paper-reported.

#### E4. SkinM2Former (WACV 2025) 📄
- **Use:** multi-label multimodal formulation; related architecture.

#### E5. MHDF-Net — Mixture-of-Experts Hierarchical Dynamic Multimodal Fusion (Neurocomputing, DOI 10.1016/j.neucom.2025.132585) 📄
- **Use:** dynamic gating comparator (PAD-UFES-20, Derm7pt per v1.0 summary).

#### E6. Cross-Attention Fusion and Collaborative Edge Computing (CMIG 2025, DOI 10.1016/j.compmedimag.2025.102588) 📄
- **Use:** latency/compute considerations.

#### E7. UAGF — Uncertainty-Guided Adaptive Gated Fusion of Multiple Backbones ✅
*Expert Systems with Applications*, 2026, vol. 332, article 133657.
- **Method:** gated fusion of several image backbones using temperature-scaled uncertainty features (entropy, max probability, top-2 margin); evaluated on ISIC2018 and ISIC2019.
- **Correction vs v1.0:** this is **single-modality** (dermoscopy) multi-backbone fusion, not multimodal. Lower priority.
- **Use:** uncertainty-feature design reference.

#### E8. QA-MoE — Quality-Aware and Stable Multimodal Mixture-of-Experts (IJCAI 2026) ✅
Sun L., Sheng V.S. DOI 10.24963/ijcai.2026/766. Code: github.com/Illidon/QA-MoE
- **Method:** decouples reliability (evidential quality scoring) from routing; filters noisy modalities before sparse MoE fusion.
- **Correction vs v1.0:** evaluated on **ADNI** (Alzheimer's staging) and **MIMIC-IV** (length of stay) — not dermatology.
- **Name collision:** a different paper, *QA-MoE: Towards a Continuous Reliability Spectrum…* (Zhu et al., ACL 2026), is about multimodal sentiment analysis. Always cite ours by DOI.
- **Use:** conceptual quality-aware routing comparator; only reimplement if time allows.

#### E9. Missing-Modality Enabled Multi-Modal Fusion Architecture for Medical Data (JBI 2025, DOI 10.1016/j.jbi.2025.104796) 📄
- **Use:** missingness baseline concept; distinguish "robust to missing" from "decides to acquire".

#### E10. Kawahara et al. — Seven-Point Checklist and Skin Lesion Classification using Multitask Multimodal Neural Nets (IEEE JBHI 2019) 📚
- **Use:** origin of Derm7pt; historical context.

---

### F. Uncertainty, OOD and shift in dermatology

#### F1. SAGE — Multi-Criterion Uncertainty Estimation Improves Skin Cancer Distribution Shift Detection and Malignancy Prediction 📄
*npj Digital Medicine*, 2026. nature.com/articles/s41746-026-03293-y
- **Use:** OOD/shift signal design; keep OOD metrics separate from classification metrics.

#### F2. Closing the AI Generalisation Gap by Adjusting for Dermatology Condition Distribution Differences (eBioMedicine 2025, DOI 10.1016/j.ebiom.2025.105766) 📄
- **Use:** synthetic corruption ≠ real shift; document acquisition-setting differences.

#### F3. Energy-based OOD detection (Liu et al., NeurIPS 2020) and Mahalanobis OOD (Lee et al., NeurIPS 2018) 📚
- **Use:** OOD score implementations.

#### F4. Deep Ensembles (Lakshminarayanan et al., NeurIPS 2017) 📚
- **Use:** epistemic uncertainty (mutual information) for B6 and EADE features.

#### F5. Review of uncertainty quantification in medical image analysis (Medical Image Analysis 2024, DOI 10.1016/j.media.2024.103223) 📄
- **Use:** justify UQ choices.

---

### G. Robustness, fairness, data quality and label uncertainty

#### G1. Scoping Review of Robustness Concepts for ML in Healthcare (npj Digit. Med. 2025, DOI 10.1038/s41746-024-01420-1) 📄
- **Use:** taxonomy to define DERMA-STRESS dimensions.

#### G2. Robustness and Readiness of Large Frontier Models in Health AI (Nature Medicine 2026) 📄
- **Use:** broad stress-testing context.

#### G3. Investigating the Quality of DermaMNIST and Fitzpatrick17k (Scientific Data 2025) 📄
- **Use:** duplicate/leakage checks are mandatory.

#### G4. DDI — Disparities in Dermatology AI Performance on a Diverse, Curated Clinical Image Set (Science Advances 2022, DOI 10.1126/sciadv.abq6147) 📄
- **Use:** subgroup reporting; possible external clinical-image test if licence permits.

#### G5. Evaluating Medical AI Systems in Dermatology Under Uncertain Ground Truth (Medical Image Analysis 2025, DOI 10.1016/j.media.2025.103556) 📄
- **Use:** label-uncertainty limitations; MILK10k is mostly histopathology-confirmed, which partially mitigates this.

#### G6. Vickers & Elkin — Decision Curve Analysis (Medical Decision Making 2006) 📚
- **Use:** net-benefit endpoint.

---

### H. Datasets and clinical motivation

| Dataset | Modalities | Role | Status |
|---------|-----------|------|--------|
| **MILK10k** | Paired clinical + dermoscopic, metadata, skin tone, MONET concepts | **Primary** | ✅ |
| MILK10k Benchmark | Same, hidden labels (479 lesions) | Optional external leaderboard | ✅ |
| Derm7pt (Kawahara 2019) | Clinical + dermoscopic + checklist | Secondary replication | 📚 |
| PAD-UFES-20 (Pacheco 2020, DOI 10.1016/j.dib.2020.106221) | Smartphone clinical + metadata, no dermoscopy | Clinical-branch shift test | 📚 |
| HAM10000 (Tschandl 2018, DOI 10.1038/sdata.2018.161) | Dermoscopy only | Optional pretraining/shift | 📚 |
| DDI (Science Advances 2022) | Clinical, diverse skin tones | Optional fairness/shift test | 📄 |

**Clinical motivation:** Müller et al., JAAD 2026 (§4.5). ✅
**Concept annotations:** MONET (Kim et al., 2024) — used by MILK10k for concept probabilities. 📚 (confirm citation details)

---

### I. Language-model and text-related dermatology work (context only)

These inform the small RAG/explanation module and text-contradiction stress tests. None is a baseline for the decision policy.

| Paper | Use | Status |
|-------|-----|--------|
| PanDerm — multimodal vision foundation model for dermatology (Nature Medicine 2025, DOI 10.1038/s41591-025-03747-y) | Candidate backbone after licence/compute check | 📄 |
| DermEVAL (WACV 2026) | Fluent text ≠ correct diagnosis; summary evaluation | 📄 |
| Clinical freetext + dermatoscopic images (Communications Medicine 2026, DOI 10.1038/s43856-026-01456-2) | Text-leakage awareness | 📄 |
| Synthesized Clinical Notes from Unimodal Dermatology Datasets (npj Digit. Med. 2026, DOI 10.1038/s41746-026-03040-3) | Synthetic text must be labelled; not our contribution | 📄 |
| MEL-IA — interoperable multimodal system (J. Med. Syst. 2026, DOI 10.1007/s10916-026-02424-y) | Workflow/integration context | 📄 |
| SkinGPT-4 (Nature Communications 2024) | Multimodal LLM context | 📄 (verify DOI) |
| Esteva et al., Nature 2017 (DOI 10.1038/nature21056) | Historical context | 📚 |

---

## 6. Gap matrix

**How to read:** ✓ = present, ✗ = absent, ? = not determinable from abstract (must be filled from full text). This matrix is **provisional** — cells come from abstracts and must be confirmed during full-text extraction (Section 13).

| Work | Domain | Paired clinical + dermoscopy | Defer / abstain | Acquire extra modality | Explicit acquisition cost | Statistical risk guarantee | Combined-failure stress tests | Subgroup (skin tone) |
|------|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| HOT AI | Derm | ✓ | ✓ (OOD alert) | ✓ (dermoscopy) | ? | ? | ? | ? |
| Fayyad et al. (CP) | Derm | ✗ | ✓ (set-based) | ✗ | ✗ | ✓ (coverage) | ? | ? |
| Lu et al. (fair CP) | Derm | ✗ | ✓ (set-based) | ✗ | ✗ | ✓ (group coverage) | ✗ | ✓ |
| Zoravar et al. (CE-ViTs) | Derm | ✗ | ✓ (set-based) | ✗ | ✗ | ✓ (coverage) | ? (domain shift) | ? |
| Roy et al. (HOD) | Derm | ✗ | ✓ | ✗ | ✓ (cost metric for eval) | ✗ | ? | ✓ |
| DermFormer | Derm | ✓ | ? | ✗ | ✗ | ✗ | ✓ (corruption) | ? |
| Zuo / CosCatNet | Derm | ✓ | ✗ | ✗ | ✗ | ✗ | ? | ? |
| JI-ADF | Derm (MILK10k) | ✓ | ? | ✗ | ✗ | ✗ | ? | ? |
| QA-MoE | Non-derm | n/a | ✗ | ✗ (filters) | ✗ | ✗ | ✓ (noise/missing) | ✗ |
| CAMA | General | n/a | ✗ | ✓ | ✓ (budget) | ✗ | ✗ | ✗ |
| MICCAI 2022 RL selection | Cardiac | n/a | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ |
| Learning to defer | General | n/a | ✓ | ✗ | ✓ (defer cost) | ✗ | ✗ | ✗ |
| Learn then Test / CRC | General | n/a | ✓ (via thresholds) | ✗ | ✗ | ✓ | ✗ | ✗ |
| **DERMA-GUARD (proposed)** | **Derm (MILK10k)** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

**Reading the matrix:** each individual column is covered by someone. Our candidate niche is the **row** — all columns together, in dermatology, on a public paired dataset. If full-text extraction shows HOT AI (or another paper) already fills the cost and guarantee columns, we narrow the claim to DERMA-STRESS and guarantee-violation analysis.

---

## 7. What the literature already establishes

| Area | Evidence | Consequence |
|------|----------|-------------|
| Recommending dermoscopy | HOT AI | Acquisition itself is not new; our claim must be the cost model + guarantee + evaluation. |
| Conformal prediction in dermatology | Fayyad; Lu; Zoravar | CP is not new; using LTT/CRC to control a *clinical* risk in a three-action policy is our use. |
| Selective classification evaluation | Geifman; Traub (AUGRC) | Use AUGRC; do not invent a new general metric. |
| Learned deferral | Mozannar & Sontag | Must be a baseline. |
| Cost-aware modality acquisition | CAMA; MICCAI 2022; SAGEAgent; SDM-Q | Must compare against greedy/oracle/random, not only fixed rules. |
| Multimodal dermatology fusion | DermFormer; CosCatNet; JI-ADF; MHDF-Net | Fusion is a component, not a contribution. |
| Quality-aware routing | QA-MoE (non-derm) | Quality signals are inputs, not novelty. |
| Corruption robustness | DermFormer; robustness reviews | Stress tests must add combined failures + guarantee violation. |
| Distribution shift | SAGE; eBioMedicine 2025 | Synthetic corruption ≠ real shift; report separately. |
| Skin-tone disparities | DDI; Lu et al. | Subgroup reporting is mandatory. |
| Label uncertainty | MedIA 2025 | State label limitations. |
| Data quality issues | Scientific Data 2025 | Dedup and leakage checks are part of the method. |

---

## 8. Research gaps we target

These are candidate gaps stated narrowly enough to test. They are not proof that nobody has studied them.

### Gap 1 — Risk-controlled three-action policy in dermatology
**Question:** Can LTT-calibrated proceed/acquire/defer achieve higher coverage at a guaranteed malignancy miss rate than calibrated confidence-threshold abstention (B3), learning to defer (B4) and HOT-style acquisition (B5)?
**Evidence needed:** coverage at guaranteed risk; AUGRC; paired bootstrap CIs.
**Weakens claim:** no gain over B3/B5.

### Gap 2 — Cost-aware acquisition with measured efficiency
**Question:** Does a learned acquisition-value estimator recover more of the oracle benefit than random, entropy-triggered and greedy information-gain acquisition at matched rates?
**Evidence needed:** acquisition efficiency AE(r) curves; cost sensitivity sweep.
**Weakens claim:** AE near zero, or gains vanish under realistic costs.

### Gap 3 — Guarantee behaviour under combined failures
**Question:** How large is the guarantee violation under each DERMA-STRESS family, and does the policy degrade more gracefully than baselines?
**Evidence needed:** violation (test risk − α) per stress condition; deferral response to OOD.
**Weakens claim:** violations as large as baselines' with no compensating deferral.

### Supporting engineering objective
Full traceability of every assessment to evidence snapshot, model, calibration, cost-matrix and policy versions. Strength, not novelty.

---

## 9. Experiment matrix (mapped to v2)

**Baselines:** B1 clinical-only · B2 always-acquire fused · B3 calibrated confidence abstention · B4 learning to defer · B5 HOT-style entropy-triggered acquisition · B6 greedy information-gain + abstention · B7 random acquisition (matched rate) · B8 oracle acquisition (matched rate) · **EADE (ours)**

| Exp | Condition | Methods | Primary measurements | Question |
|-----|-----------|---------|---------------------|----------|
| E1 | Clean MILK10k | B1, B2 | Balanced acc., macro-F1, malignancy AUROC, Brier, NLL, adaptive ECE | Are backbones competent and calibrated? |
| E2 | Clean | B3, B4, EADE | **Coverage at guaranteed risk**, AUGRC, selective risk @ 70/80/90% | Does the policy beat abstention/deferral? |
| E3 | Clean | B5, B6, B7, B8, EADE | **Acquisition efficiency AE(r)**, acquisition rate, loss reduction | Is acquisition well targeted? |
| E4 | Image degradation (family A) | B3–B6, EADE | AUGRC, guarantee violation, action shift | Does the policy respond to bad images? |
| E5 | Missing evidence (B) | B3–B6, EADE | Risk, coverage, NIR | Behaviour when evidence is absent |
| E6 | Contradictory metadata (C) | B3, B6, EADE (± consistency signal) | Contradiction detection, deferral rate | Does consistency help? |
| E7 | Distribution shift (D): PAD-UFES-20 clinical branch; centre shift if available | B3, B5, EADE | Guarantee violation, OOD AUROC, coverage | Where does the guarantee break? |
| E8 | Leave-class-out OOD (E) | B3, B5, EADE | Deferral rate on unseen classes, confident-error rate | Does it defer on unknowns? |
| E9 | Combined failures (F) | All | All of the above | Behaviour under interacting failures |
| E10 | Ablations | EADE variants | Δ in primary endpoints | Which components matter? |
| E11 | Cost sensitivity | EADE, B5 | Primary endpoints vs ℓ_miss and c_acq | Robustness to cost assumptions |
| E12 | Fairness | B3, EADE (± group-conditional) | Per-skin-tone coverage and miss rate | Does the guarantee hold per group? |
| E13 | Operational (G) | Platform | Recovery, latency, traceability | Does the system fail safely? |

**Controls:** lesion-level splits; same cases for all methods; thresholds on validation/calibration only; out-of-fold acquisition targets; paired bootstrap (≥2,000); McNemar / DeLong where relevant; Holm correction; 3 seeds where feasible; pre-registration committed before the test run.

---

## 10. Dataset strategy

### Primary: MILK10k
- Verify licence and redistribution terms before download.
- Split by lesion (no patient IDs); deduplicate via provided ISIC IDs; state residual leakage risk.
- Lesion-level stratified split: 60 / 15 / 10 / 15 (train / validation / calibration / test).
- Report coarse grouping (rare classes merged) and binary malignancy; guarantee is on binary malignancy.
- Optional hidden-test submission to MILK10k Benchmark (check rules).

### Secondary
- **Derm7pt** — replication on a second paired dataset if time allows.
- **PAD-UFES-20** — clinical-branch shift only, with explicit label mapping.
- **HAM10000 / DDI** — optional, only for compatible experiments.

### Data-integrity checklist
- Duplicate detection (hash + near-duplicate).
- Class-imbalance report.
- Label provenance (histopathology vs other).
- Synthetic text labelled and diagnosis-free.
- External data never used for tuning.

---

## 11. Implementation decisions informed by the literature

| Decision | Informed by |
|----------|-------------|
| Temperature scaling before any thresholding | Guo 2017; CHIL 2026 |
| 3-seed deep ensemble for epistemic uncertainty | Lakshminarayanan 2017 |
| RAPS/APS prediction sets; mixed-set rule | Angelopoulos 2020 |
| LTT fixed-sequence calibration of PROCEED threshold | LTT; CRC |
| Group-conditional calibration by skin tone | Lu et al. |
| AUGRC as primary multi-threshold metric | Traub 2024 |
| Oracle/random/greedy acquisition baselines | CAMA; ACO; Shim 2018 |
| Learning-to-defer baseline | Mozannar & Sontag 2020 |
| HOT-style acquisition baseline | HOT AI |
| Energy/Mahalanobis OOD | Liu 2020; Lee 2018 |
| Leave-class-out OOD and cost-weighted evaluation | Roy et al. (HOD) |
| Combined + corruption stress families | DermFormer; robustness scoping review |
| Separate synthetic vs real shift | eBioMedicine 2025; SAGE |
| Net benefit endpoint | Vickers & Elkin 2006 |
| RAG kept out of decision path | DermEVAL (fluency ≠ correctness) |

---

## 12. Permitted and prohibited claims

### Safe before experiments
- "We propose a risk-controlled, cost-aware proceed/acquire/defer policy."
- "We calibrate the policy with Learn-then-Test to bound malignancy miss rate on auto-proceeded cases under exchangeability."
- "We measure how this guarantee degrades under controlled evidence failures."
- "DERMA-STRESS is a project-proposed evaluation protocol."

### Avoid unless proven
- "First AI to request dermoscopy." (HOT AI exists.)
- "First conformal method for skin lesions." (Fayyad, Lu, Zoravar exist.)
- "Guaranteed safe." (Guarantee is statistical, conditional on exchangeability, and fails under shift.)
- "State of the art", "clinically safer", "reduces patient harm", "proven novel".
- "Clinically validated metric" for AE, MSES or NIR.

---

## 13. Reading order and extraction template

### Round 1 — closest overlap (Weeks 1–2)
1. HOT AI (journal + preprint)
2. Learn then Test
3. Conformal Risk Control
4. AUGRC (Traub 2024)
5. CAMA
6. Fayyad et al. (CP in dermatology)
7. Lu et al. (fair CP)
8. Mozannar & Sontag (learning to defer)
9. JI-ADF (MILK10k)
10. Müller et al. JAAD 2026
11. **Confirm or delete DermFusion** ⚠️

### Round 2 — comparators and stress design
12. DermFormer
13. Zuo / CosCatNet
14. Roy et al. (HOD)
15. SAGE
16. Zoravar et al.
17. QA-MoE
18. Biopsy-recommendation UQ (Frontiers 2026)
19. CHIL 2026 calibration/selective prediction
20. Robustness scoping review
21. eBioMedicine 2025 shift
22. DDI; dataset-quality paper; uncertain-ground-truth paper

### Round 3 — context
23. MHDF-Net, SkinM2Former, CMIG 2025, UAGF, missing-modality JBI
24. PanDerm, DermEVAL, freetext + dermoscopy, synthesized notes, MEL-IA, SkinGPT-4
25. Active acquisition: ACO, Shim 2018, MICCAI 2022, SAGEAgent, SDM-Q

### Novelty-audit searches (record every hit)
*dermoscopy acquisition*, *selective prediction dermatology*, *conformal deferral dermatology*, *learning to defer dermatology*, *active modality acquisition skin*, *risk control medical imaging triage*, plus forward citations of HOT AI.

### Extraction template (one per paper, stored in `docs/extractions/<slug>.md`)

```
Citation / DOI / venue / date:
Verification status: ✅ / 📄 / ⚠️  — checked by: ____ on ____
Task and labels:
Dataset(s), size, modalities:
Split and leakage controls:
Backbone / fusion:
Uncertainty / calibration / OOD method:
Action space (predict / abstain / defer / acquire):
Explicit costs? (Y/N, how):
Statistical guarantee? (Y/N, which):
Stress / shift protocol:
Subgroup analysis:
Metrics and key reported numbers (paper-reported):
Code / weights / licence:
Author-stated limitations:
Our inferred limitations (marked as inference):
Gap-matrix cells to update:
DERMA-GUARD experiment it informs:
Can we reproduce it fairly? (Y/N, why):
```

---

## 14. Reference list

### Verified during this revision (✅)
1. Mehta D. et al. Multi-task AI models in dermatology: Overcoming critical clinical translation challenges for enhanced skin lesion diagnosis. *JEADV* 39(12):2121–2133, 2025. https://doi.org/10.1111/jdv.20551 — Preprint: https://arxiv.org/abs/2311.01009
2. Traub J. et al. Overcoming Common Flaws in the Evaluation of Selective Classification Systems. NeurIPS 2024. https://arxiv.org/abs/2407.01032
3. Rheude T. et al. Cohort-Based Active Modality Acquisition. https://arxiv.org/abs/2505.16791
4. MILK10k dataset, ISIC Archive. https://doi.org/10.34970/648456 · https://api.isic-archive.com/collections/425/
5. MILK10k Benchmark. https://challenge.isic-archive.com/landing/milk10k/
6. Müller C. et al. Comparative and complementary diagnostic value of dermatoscopy and clinical close-up photography in skin cancer diagnosis: a study from the MILK10k dataset. *JAAD* 2026. https://doi.org/10.1016/j.jaad.2026.03.069
7. JI-ADF: Joint-Individual Learning with Adaptive Decision Fusion for Multimodal Skin Lesion Classification. https://arxiv.org/abs/2604.27343
8. Fayyad J. et al. Empirical Validation of Conformal Prediction for Trustworthy Skin Lesions Classification. https://arxiv.org/abs/2312.07460
9. Lu C. et al. Fair Conformal Predictors for Applications in Medical Imaging. https://arxiv.org/abs/2109.04392
10. Zoravar M. et al. Domain Adaptive Skin Lesion Classification via Conformal Ensemble of Vision Transformers. https://arxiv.org/abs/2505.15997
11. Valancius M. et al. Acquisition Conditioned Oracle for Nongreedy Active Feature Acquisition. https://arxiv.org/abs/2302.13960
12. Shim H., Hwang S.J., Yang E. Joint Active Feature Acquisition and Classification with Variable-Size Set Encoding. NeurIPS 2018.
13. Reinforcement Learning for Active Modality Selection during Diagnosis. MICCAI 2022. https://doi.org/10.1007/978-3-031-16431-6_56
14. SAGEAgent: A Self-Evolving Agent for Cost-Aware Modality Acquisition in Multimodal Survival Prediction. https://arxiv.org/abs/2607.09521
15. SDM-Q: Cost-Aware Staged Decision-Making for Multi-Omics Classification with Deep Q-Learning. https://arxiv.org/abs/2605.31014
16. Sun L., Sheng V.S. QA-MoE: Quality-Aware and Stable Multimodal Mixture-of-Experts for Robust Clinical Prediction in Noisy and Missing-Modal Settings. IJCAI 2026. https://doi.org/10.24963/ijcai.2026/766
17. Enhancing Skin Lesion Classification via Uncertainty-Guided Adaptive Gated Fusion of Multiple Backbones (UAGF). *Expert Systems with Applications* 332:133657, 2026.

### Established references (📚 — confirm details before formal citation)
18. Angelopoulos A.N. et al. Learn then Test. https://arxiv.org/abs/2110.01052
19. Angelopoulos A.N. et al. Conformal Risk Control. https://arxiv.org/abs/2208.02814
20. Angelopoulos A.N. et al. Uncertainty Sets for Image Classifiers using Conformal Prediction (RAPS). https://arxiv.org/abs/2009.14193
21. Angelopoulos A.N., Bates S. A Gentle Introduction to Conformal Prediction. https://arxiv.org/abs/2107.07511
22. Geifman Y., El-Yaniv R. Selective Classification for Deep Neural Networks. NeurIPS 2017. https://arxiv.org/abs/1705.08500
23. Guo C. et al. On Calibration of Modern Neural Networks. ICML 2017. https://arxiv.org/abs/1706.04599
24. Lakshminarayanan B. et al. Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. NeurIPS 2017. https://arxiv.org/abs/1612.01474
25. Mozannar H., Sontag D. Consistent Estimators for Learning to Defer to an Expert. ICML 2020. https://arxiv.org/abs/2006.01862
26. Liu W. et al. Energy-based Out-of-distribution Detection. NeurIPS 2020. https://arxiv.org/abs/2010.03759
27. Lee K. et al. A Simple Unified Framework for Detecting OOD Samples and Adversarial Attacks. NeurIPS 2018. https://arxiv.org/abs/1807.03888
28. Roy A.G. et al. Does Your Dermatology Classifier Know What It Doesn't Know? Detecting the Long-Tail of Unseen Conditions. Medical Image Analysis 2022.
29. Vickers A.J., Elkin E.B. Decision Curve Analysis. *Medical Decision Making* 2006.
30. Kawahara J. et al. Seven-Point Checklist and Skin Lesion Classification Using Multitask Multimodal Neural Nets. *IEEE JBHI* 2019. https://doi.org/10.1109/JBHI.2018.2824327
31. Pacheco A.G.C. et al. PAD-UFES-20. *Data in Brief* 2020. https://doi.org/10.1016/j.dib.2020.106221
32. Tschandl P. et al. The HAM10000 Dataset. *Scientific Data* 2018. https://doi.org/10.1038/sdata.2018.161
33. Esteva A. et al. Dermatologist-level classification of skin cancer with deep neural networks. *Nature* 2017. https://doi.org/10.1038/nature21056
34. Kim C. et al. MONET (concept annotation for dermatology images), 2024 — confirm exact title/venue.

### Carried from v1.0 survey (📄 — open the DOI before citing)
35. DermFormer. *Pattern Anal. Appl.* 2025. https://doi.org/10.1007/s10044-025-01572-0
36. Zuo et al. / CosCatNet. *Artif. Intell. Med.* 2025. https://doi.org/10.1016/j.artmed.2025.103091
37. SkinM2Former. WACV 2025.
38. MHDF-Net. *Neurocomputing*. https://doi.org/10.1016/j.neucom.2025.132585
39. Cross-attention fusion + edge computing. *CMIG* 2025. https://doi.org/10.1016/j.compmedimag.2025.102588
40. Missing-modality enabled fusion. *JBI* 2025. https://doi.org/10.1016/j.jbi.2025.104796
41. SAGE. *npj Digit. Med.* 2026. https://www.nature.com/articles/s41746-026-03293-y
42. ViT UQ for triaging skin lesions / biopsy recommendation. *Front. Bioeng. Biotechnol.* 2026. https://doi.org/10.3389/fbioe.2026.1859844
43. Calibration and selective prediction in multimodal clinical classification. CHIL 2026. https://proceedings.mlr.press/v333/lopez26a.html
44. Conformal prediction for skin lesions (journal version). *CMPB* 2024. https://doi.org/10.1016/j.cmpb.2024.108231
45. Uncertain ground truth in dermatology. *Med. Image Anal.* 2025. https://doi.org/10.1016/j.media.2025.103556
46. Robustness concepts scoping review. *npj Digit. Med.* 2025. https://doi.org/10.1038/s41746-024-01420-1
47. Frontier-model robustness in health AI. *Nature Medicine* 2026. https://www.nature.com/articles/s41591-026-04501-8
48. Dermatology generalisation gap. *eBioMedicine* 2025. https://doi.org/10.1016/j.ebiom.2025.105766
49. DermaMNIST / Fitzpatrick17k quality. *Scientific Data* 2025. https://www.nature.com/articles/s41597-025-04382-5
50. DDI. *Science Advances* 2022. https://doi.org/10.1126/sciadv.abq6147
51. UQ review in medical imaging. *Med. Image Anal.* 2024. https://doi.org/10.1016/j.media.2024.103223
52. PanDerm. *Nature Medicine* 2025. https://doi.org/10.1038/s41591-025-03747-y
53. DermEVAL. WACV 2026.
54. Clinical freetext + dermatoscopic images. *Commun. Med.* 2026. https://doi.org/10.1038/s43856-026-01456-2
55. Synthesized clinical notes. *npj Digit. Med.* 2026. https://doi.org/10.1038/s41746-026-03040-3
56. MEL-IA. *J. Med. Syst.* 2026. https://doi.org/10.1007/s10916-026-02424-y
57. SkinGPT-4. *Nature Communications* 2024 (verify DOI).
58. AI in dermatology: approved applications review. *Int. J. Dermatol.* 2025. https://doi.org/10.1111/ijd.17847
59. AI-driven dermatological diagnostics systematic review, 2026 (verify metadata).

### Could not be located (⚠️)
60. DermFusion. *Scientific Reports* 2026, listed DOI 10.1038/s41598-026-72311-6 — **confirm or remove.**

---

## 15. Final position

> **A dermatology AI should not only predict — it should decide whether its evidence is sufficient, whether dermoscopy is worth requesting, and when to defer, and it should be able to state how often that decision goes wrong.**

The individual ingredients all exist in the literature. DERMA-GUARD's bet is that combining a clinical cost model, a statistical risk guarantee, a three-action policy and a combined-failure stress protocol on a public paired dataset produces a measurable, reproducible improvement — and an honest map of where the guarantee fails.

**Immediate next step:** complete Round 1 extractions, confirm or remove DermFusion, fill the provisional cells of the gap matrix, then freeze the baseline protocol.
