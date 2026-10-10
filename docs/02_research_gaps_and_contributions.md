# Research Gaps and Proposed Contributions

> Companion to [01_system_analysis.md](01_system_analysis.md) (what exists today) and
> [04_mathematical_foundations.md](04_mathematical_foundations.md) (the formal machinery).
>
> **Citation note.** Author/venue attributions below are from memory and were not
> verified against the published papers while writing this document. Every citation
> must be checked against the actual paper before it appears in a submission. Claims
> marked **[verify]** are ones I am least certain about.

---

## 1. Where the field actually is

Four literatures touch this problem and mostly do not talk to each other.

**(a) Multimodal skin-lesion classification.** PAD-UFES-20 (Pacheco et al., *Data in
Brief*, 2020) established the clinical-image + metadata benchmark, and the same group's
**MetaBlock** (Pacheco & Krohling, *IEEE JBHI*, 2021) remains the reference method for
conditioning image features on metadata. MetaNet and various concatenation/attention
variants followed. These works optimise **accuracy under complete evidence**. They
assume every modality is present and usable, and report a single population-level
number.

**(b) Uncertainty quantification.** Deep ensembles (Lakshminarayanan et al., 2017),
MC-dropout, temperature scaling (Guo et al., 2017) and evidential deep learning
(**EDL**, Sensoy et al., NeurIPS 2018) give per-prediction uncertainty. EDL is
particularly relevant: a Dirichlet output head yields belief masses plus an explicit
*"I don't know"* mass from a single forward pass, with no ensembling cost.

**(c) Trusted multi-view learning.** **Trusted Multi-View Classification** (Han et al.,
ICLR 2021; extended in TPAMI as ETMC) combines per-view Dirichlet evidence with the
**Dempster–Shafer** rule. This is the closest existing work to what DERMA-GUARD
hand-rolls: it produces per-view uncertainty, a principled combination operator, and a
conflict term — all with derivations. It has been applied to generic multi-view
benchmarks, **not** to dermatology with patient-authored text and missing modalities.

**(d) Selective prediction and conformal methods.** Chow's rule (1970) gives the
optimal reject threshold under known costs. Conformal prediction (Vovk, Gammerman &
Shafer) gives distribution-free coverage; **conformal risk control** and
**Learn-then-Test** (Angelopoulos, Bates et al.) extend this to bounding arbitrary
risks — including the error rate *among accepted cases*, which is exactly the clinical
quantity of interest. Mondrian/class-conditional conformal prediction gives per-class
guarantees.

### The gap

> No existing work provides **per-case, per-modality evidence quality** with a
> **principled combination rule that handles missing and conflicting modalities**, and a
> **statistically guaranteed abstention policy**, evaluated under **systematic evidence
> corruption**, on **patient-authored multimodal dermatology data**.

Each ingredient exists. The combination, and specifically the application to
patient-generated evidence where imperfection is the norm rather than the exception,
does not. That is a defensible semester-scale research position — it is an
*integration-and-rigour* contribution, not a claim to have invented evidential fusion.

---

## 2. What the current project got right, and what it got wrong

The existing system is an attempt at exactly this gap. Its instincts are good; its
execution is not rigorous.

| Intent | Current implementation | Why it fails as research |
|---|---|---|
| Per-modality quality `Q` | Weighted mean of 5 image statistics; PTIS heuristic for text | Hand-tuned weights, no validation that `Q` predicts anything |
| Per-modality reliability `R` | `0.4·certainty + 0.3·agreement + 0.2·quality + 0.1·stability` | Coefficients invented; no derivation, no fitting, no evaluation |
| Modality utility `U` | Leave-one-out loss delta, needs labels | Unusable at inference; falls back to the constant 0.5 |
| Fusion weights | `softmax(log(Q·R·U))`, masked and renormalised | A reasonable heuristic with no optimality claim |
| Missing modalities | Zero the mask, renormalise the rest | Ad hoc. Renormalisation is not the same as marginalising out an absent variable |
| Conflict `C` | `1 − mean pairwise JS divergence`, threshold 0.3 | Threshold picked by hand; JS between *predictions* conflates disagreement with uncertainty |
| Abstention | 4-level ladder with thresholds 0.80/0.50/0.30 | **No guarantee of any kind.** The project's own `limitations.md` concedes the conformal coverage guarantee was lost when the ladder replaced it |
| Safety evaluation | DERMA-STRESS, 18 corruption conditions | **This part is genuinely good** and worth keeping |

Two things are worth stressing. First, every quantity above is a **number that gets
thresholded**, and not one of those numbers has a calibration argument. Second, four of
them were produced by **randomly-initialised neural networks** that were never trained
(see [evaluation_integrity.md](evaluation_integrity.md) §3.3) — so the safety layer was
thresholding noise.

The proposal below replaces each hand-tuned quantity with one that has a derivation and
a calibration procedure.

---

## 3. Proposed contributions

### C1 — Evidential per-modality heads (replaces Q, R, U)

Each modality encoder emits **non-negative evidence** `e_m ∈ R₊^K` through a softplus
head, interpreted as a Dirichlet posterior `α_m = e_m + 1` over class probabilities.
This yields, in one forward pass and *per case*:

- belief mass per class, `b_{m,k} = e_{m,k} / S_m`
- an explicit ignorance mass, `u_m = K / S_m`, with `Σ_k b_{m,k} + u_m = 1`

`u_m` **is** the per-modality, per-case reliability the project currently approximates
with invented coefficients — but it is learned from data under a proper loss and is
directly interpretable. Full derivation in
[04_mathematical_foundations.md](04_mathematical_foundations.md) §2.

*Why this matters here:* a blurred photograph and a three-word symptom description
should both produce low evidence and high `u_m`. The current system asks a hand-written
formula to notice that; EDL learns it.

### C2 — Dempster–Shafer combination (replaces masking and renormalisation)

Combine per-modality opinions with the DS rule. Three properties fall out that the
current system engineers by hand or gets wrong:

1. **Missing modalities are free.** An absent modality is the *vacuous* opinion
   (`b = 0, u = 1`), which is the identity element of DS combination. Combining with it
   changes nothing. No mask, no renormalisation, no zero-division guard — and unlike
   renormalisation, this is the mathematically correct way to say "no information".
2. **Conflict is measured, not estimated.** The DS conflict mass
   `C = Σ_{i≠j} b_{m,i} b_{m',j}` appears in the normaliser. Cross-modal conflict
   becomes a by-product of the combination rule rather than a bolted-on JS divergence
   with a hand-picked threshold.
3. **Uncertainty composes.** Combined ignorance is `u = u_m u_{m'} / (1 − C)`, so
   agreement between confident modalities reduces uncertainty while conflict raises it.

### C3 — Diagnosis-adaptive fusion with a convergence argument

The project's central algorithmic idea — re-weighting modalities based on the
*suspected* diagnosis, because melanoma assessment leans on visual pattern while
inflammatory conditions lean on history — is good and worth keeping. What it lacks is
an objective function.

Reformulated as coordinate ascent on a single variational objective (derivation in
§3 of the maths doc), the iteration becomes:

- **E-step:** `q(y) ∝ ∏_m p(y | x_m)^{w_m}` — a weighted log-opinion pool
- **M-step:** `w ∝ exp(λ · E_q[log p(y | x_m)])` — weight each modality by how well it
  explains the current posterior

This monotonically increases a free energy, so it **provably converges** — replacing the
current "run 3 iterations and hope", and giving a principled stopping rule instead of a
hand-set `convergence_threshold = 0.05`.

### C4 — Risk-controlled selective prediction (restores the lost guarantee)

Replace the three hand-set thresholds with a **single calibrated** one. Using conformal
risk control, choose `λ` on a held-out calibration split such that

> `P( selective risk on accepted cases ≤ α ) ≥ 1 − δ`

distribution-free, with no assumption about the model being correct. Add
**class-conditional (Mondrian)** calibration so that melanoma — 52 of 2298 cases — gets
its own coverage guarantee rather than being averaged away by the two majority classes.

This directly repairs the limitation the project documents and never fixed, and it is
*stronger* than what v1 of this project had, because the guarantee is now on the
clinically meaningful quantity (error among accepted) and holds per class.

### C5 — Cost-sensitive deferral

Derive the deferral threshold from an explicit cost matrix via a generalised Chow rule,
instead of the symmetric confidence thresholds used now. Missing a melanoma and raising
a false alarm on a nevus are not equal errors, and the decision rule should say so in
its parameters rather than in a comment.

### C6 — DERMA-STRESS, kept and repaired

The 18-condition corruption benchmark is the existing work's strongest asset and should
be retained as a contribution. Two metric definitions need the fixes already applied
(see [evaluation_integrity.md](evaluation_integrity.md) §3.5): `recovery_rate` must
divide by deferrals rather than by all cases, and accuracy must be reported *conditional
on coverage*. A robustness metric that a model refusing every case can max out is not
measuring robustness.

### C7 — Retrieval as an evidence channel

Nearest-neighbour retrieval over per-modality embeddings, with the agreement rate among
neighbours used as an **additional conformity score** feeding C4 — not as an extra vote.
Design and the reasons for the strict label/explanation separation are in
[06_retrieval_design.md](06_retrieval_design.md).

---

## 4. Scope discipline

This project has drifted twice: a scoped capstone (v1: PAD-UFES-20, binary, PTIS
gating, conformal) became a specification spanning two datasets and a five-model
foundation-model ensemble (v4), of which the ensemble was never buildable in the
available environment.

The proposal above deliberately **narrows** scope:

| Keep | Drop | Reason |
|---|---|---|
| PAD-UFES-20, 6-class | Derm7pt | One dataset done rigorously beats two done partially. Derm7pt also needs dermoscopy, which PAD-UFES lacks, so the two never shared a pipeline |
| A–G specialist ladder as **baselines** | DermLIP-PanDerm, PanDerm, Swin-Tiny ×2, ResNet152 | PanDerm weights are unpublished **[verify]**; the environment is CPU-only; these shipped with random heads and voted on noise |
| MetaBlock as the literature baseline | — | It is the reference method on this exact dataset and must be beaten or matched honestly |
| DERMA-STRESS | — | Genuine contribution |
| Patient-level splits, focal loss, TensorBoard, local Ollama, $0 cost | — | All sound choices from v1 |
| LLM report generation as **engineering** | LLM orchestration as a **scientific claim** | Useful product surface; not a research result, and attributing diagnostic credit to it would be unsupportable |

The contribution is then: *a calibrated, evidence-aware fusion and deferral system for
imperfect patient-authored multimodal dermatology data, with distribution-free safety
guarantees and systematic robustness evaluation.* One dataset, one task, defensible.

---

## 5. Limitations this programme fixes, and what remains

**Fixed by the above:**

| Old limitation | How it is resolved |
|---|---|
| No abstention guarantee (`limitations.md`) | C4 — conformal risk control, class-conditional |
| Hand-tuned Q/R/U coefficients | C1 — learned Dirichlet evidence under a proper loss |
| Masking/renormalisation for missing modalities | C2 — vacuous opinion is the DS identity |
| Conflict threshold picked by hand | C2 — conflict mass from the combination rule; threshold calibrated |
| Fusion iteration with no convergence argument | C3 — monotone coordinate ascent |
| Untrained MLPs thresholded as if meaningful | C1 — all quantities trained or derived |
| Irreproducible cached features | `scripts/build_features.py` — manifest + verify mode, cosine 1.000000 |
| Overlapping train/test splits | `case_loader.patient_level_split` — disjointness asserted |
| Metrics rewritten to hardcoded targets | Removed; honest evaluation in the Kaggle pipeline |

**Remaining, and must be stated in any write-up:**

1. **Synthetic patient text.** The narratives are template-generated from structured
   fields, not written by patients. Any claim about patient-authored text is a claim
   about a simulation of it. This is the single largest external-validity threat.
2. **One dataset, one centre.** PAD-UFES-20 is from a single Brazilian programme. No
   cross-site generalisation evidence.
3. **Severe class imbalance.** MEL has 52 cases; a 15% test split leaves ~8. Per-class
   conclusions about melanoma will have very wide intervals, and that must be reported
   rather than smoothed over.
4. **No dermoscopy.** The diagnosis-adaptive premise (melanoma leans on dermoscopic
   pattern) cannot be fully tested without dermoscopic images.
5. **Retrospective only.** No prospective or clinical validation; no claim of clinical
   utility is supportable.
6. **Fairness not addressed.** Fitzpatrick skin-type distribution is a known confounder
   and is out of scope here — which should be said explicitly rather than omitted.

---

## 6. Honest novelty assessment

For a semester project this is a strong position, provided it is framed accurately:

- **Not novel:** evidential deep learning, DS combination, conformal risk control,
  multimodal fusion, selective prediction. All established.
- **Novel enough to defend:** their *integration* into a single evidence-management
  layer; the application to patient-authored, routinely-incomplete dermatology evidence;
  the systematic corruption benchmark; and the empirical question of whether
  evidence-aware fusion actually buys safety over a well-tuned static baseline.
- **The honest framing:** *"We ask whether principled evidence management measurably
  improves safety under imperfect evidence, and we build the benchmark needed to answer
  that."* If the answer turns out to be "no, a tuned static baseline matches it", that
  is a publishable negative result — and `what_did_not_work.md` already commits the
  project to reporting it.

What would push this past semester scale: a second dataset for external validation, real
patient-authored text, and a reader study comparing the deferral policy against
clinicians. Those are the natural follow-ons, not prerequisites.
