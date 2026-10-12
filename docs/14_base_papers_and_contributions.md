# Base papers, what we did, what we achieved, what we publish — and how to present it

Read this before the viva. It answers, in order: **which papers is this project based on,
where did our own work start, what did we change, what did we achieve, what exactly does the
paper claim, what did we find, and how should it be presented.**

---

## 1. The two base papers

Out of the papers in the literature dossier (`docs/reference/literature_survey_v2.md`), two
are the bases of this project — one for the **task**, one for the **method**. Both are in the
paper's references.

### Base paper 1 — HOT AI (the task)

> D. Mehta, C. Primiero, B. Betz-Stablein, … Z. Ge. *Multi-task AI models in dermatology:
> Overcoming critical clinical translation challenges for enhanced skin lesion diagnosis.*
> Journal of the European Academy of Dermatology and Venereology 39(12):2121–2133, 2025.
> doi:10.1111/jdv.20551

**What it does:** from a clinical photo it gives (1) a hierarchical diagnosis, (2) an
"this is unusual" (out-of-distribution) alert, and (3) a recommendation to take a
**dermoscopic image** when the photo alone is not enough; then it fuses both images.

**Why it is our base:** it is the closest task to ours — an AI that knows it should ask for a
closer look. It means "AI that asks for dermoscopy" is **not** our novelty.

**What we add on top (and can prove):**

| HOT AI | DERMA-GUARD |
|---|---|
| recommends dermoscopy when uncertain | asks for dermoscopy **and** missing details **and** a better photo |
| OOD alert | separate **DEFER** when the evidence contradicts itself, plus a **LOW** tier |
| no stated cost model (from the abstract) | explicit costs: missed cancer 5, false alarm 2, dermoscopy 0.5, review 1.5 |
| no stated guarantee | **certified** missed-cancer rate (Learn-then-Test) |
| in-house data | **public** paired data (MILK10k) + 2 external datasets + stress tests |

**How we compared:** HOT AI's model is not public, so we built a **HOT-style baseline**: ask
the most uncertain patients for dermoscopy (same rate as ours), then answer.
Result: that rule picks *whom to ask* better than our learned estimator (acquisition
efficiency +0.26 vs +0.07) — **we report this** — but answering everyone afterwards misses
**5.6%** of cancers, while DERMA-GUARD's confident answers miss **3.1%**.

### Base paper 2 — Learn-then-Test (the method)

> A. N. Angelopoulos, S. Bates, E. J. Candès, M. I. Jordan, L. Lei. *Learn then Test:
> Calibrating predictive algorithms to achieve risk control.* The Annals of Applied
> Statistics, 2025.

**What it does:** picks a threshold on held-out data so that a chosen error rate stays below
a target with high probability (a written guarantee), as long as future data look like the
calibration data.

**How we use it:** unchanged, on the error that matters most — *cancers called benign among
confident answers* — with target 5% and confidence 90%.
Result: **certified at both stages** (λ̂ = 0.218 without dermoscopy, 0.276 with). On two
other datasets, where the theory does not promise anything, the target still held
(PAD-UFES-20 **0.0%**, Derm7pt **3.8%**).

### Other papers we compare against (not base papers)

| Paper / method | Used as | Result |
|---|---|---|
| Learning to defer (Mozannar & Sontag, ICML 2020) | a learned "know when you are wrong" rejector | misses **4.9%** vs our **3.1%** (p = 0.011) |
| Selective classification (Geifman & El-Yaniv) | confidence threshold | **3.8%** vs 3.1% (not significant) — but **3.5–9.3%** across four fusions, ours **3.0–3.5%** |
| AUGRC (Traub et al., NeurIPS 2024) | evaluation metric | softmax ranks errors better than ignorance (0.0757 vs 0.0820) — reported |
| CAMA (Rheude et al.) | random / oracle acquisition bounds | dermoscopy asked for 29.5% |
| MILK10k + reader study (Müller et al., JAAD 2026) | dataset; "each modality adds value" | dermoscopy AUROC 0.854 vs photo 0.787; fused **0.885** |
| JI-ADF (MILK10k hidden test) | fusion on the same dataset | not comparable (we did not use the hidden test) |

---

## 2. Where our own work started (not base papers — our origin)

Before this version, our team wrote two drafts and built one prototype. They are **our own
earlier work**, kept in the repository so the progress is visible:

| Our earlier work | Idea we kept | What was missing |
|---|---|---|
| Draft 1 — agentic panel (`paper/old1.tex`) | specialist team, Devil's Advocate, diagnosis-adaptive EM fusion, 4-level consensus | **no results** — every number is a `\todo{XX.X%}` placeholder |
| Draft 2 — risk-controlled plan (`paper/old2.tex`) | answer / acquire / defer, cost matrix, Learn-then-Test, MILK10k, baselines | **no implementation** — a plan |
| Prototype (`HealthSphere_archive/Old_system`) | Models A–G on PAD-UFES-20, a care app | evaluation problems (below) |

**Prototype audit (say this calmly in the viva):** its split was by photo (43% of test photos
showed a lesion already in training), and its evaluation code adjusted predictions toward
hard-coded target accuracies. Re-scored honestly with the same weights, its models reach
**57–75%** on patients never seen in training (reported: 84–90%). Fused by our new gate,
the seven old models reach **72.7%** vs 59.8% for averaging them.

---

## 3. What we did

| From | What we built and measured |
|---|---|
| HOT AI (base 1) | asking for evidence + a HOT-style baseline |
| Learn-then-Test (base 2) | the certified miss-rate bound |
| Our draft 1 | **six specialists**, Devil's Advocate with 10 signals, EM fusion (v1) with a **measured** trust table, and a **learned gate** (v2); **seven fusion rules compared**, cross-fitted, every subset of specialists, oracle ceiling |
| Our draft 2 | **four outcomes** (HIGH, LOW, ACQUIRE, DEFER), cost matrix, conformal "no false reassurance" gate, baselines on the same probabilities |
| New | quality checks (blur, light, grain, covered centre), missing-details request, explainer, 2 external datasets, honest prototype re-evaluation, care platform with deletion and audit log |

---

## 4. What we achieved (787 unseen MILK10k patients)

| | Result |
|---|---|
| Patients who get an answer | **89.1%** |
| Confident (HIGH) answers | **57.1%**, **77.1%** accurate over 11 diagnoses |
| Cancers called benign among confident answers | **3.1%**, certified ≤ 5% with 90% probability |
| Photo-only macro-F1 (network alone → full team) | **0.286 → 0.377** |
| With dermoscopy | accuracy **0.731**, cancer AUROC **0.885** |
| vs learned rejector / confidence threshold | **4.9%** (p = 0.011) / **3.8%** (n.s.) missed |
| Across four fusion back-ends | ours **3.0–3.5%**, threshold **3.5–9.3%** |
| Damaged photos | **0** confident answers with the checks |
| PAD-UFES-20 / Derm7pt, no retraining | **0.0% / 3.8%** missed among confident answers |

---

## 5. What the paper claims (and what it does not)

**Claims** (each backed by a table/figure and a script):
1. Learning whom to trust from all of the evidence improves photo-only diagnosis by a third.
2. A calibrated four-outcome policy that asks for evidence and certifies its miss rate.
3. **Key finding:** a confidence threshold is only as safe as the probabilities beneath it;
   the calibrated policy stays at 3.0–3.5% on every fusion.
4. Evidential uncertainty alone misses damaged photos; explicit checks do not.
5. An honest re-evaluation of our own earlier prototype.

**Not claimed:** clinical validity; first AI to ask for dermoscopy (HOT AI exists);
first conformal method for skin (Fayyad, Lu exist); beating the MILK10k leaderboard;
beating a threshold on every model; fairness for lighter skin tones; multiple seeds.

---

## 6. All findings

| # | Finding | For / against us |
|---|---|---|
| 1 | Evidence-aware weighting beats averaging and the evidential network alone | ✅ |
| 2 | Stacking (textbook combiner) is the worst learned rule — too little data | lesson |
| 3 | Oracle 0.59 vs our 0.44 macro-F1: choosing whom to trust is the bottleneck | open |
| 4 | Threshold safety depends on the fusion; ours does not | ✅ key |
| 5 | Learned rejector significantly less safe | ✅ |
| 6 | Threshold is more accurate than our HIGH tier | ❌ reported |
| 7 | HOT-style "ask the most uncertain" beats our acquisition estimator | ❌ reported |
| 8 | Blur makes the model *more* confident; checks needed | lesson → fixed |
| 9 | Without details the gate missed 17.6% → now it asks | lesson → fixed |
| 10 | In-sample fitting inflated coverage 3 points → cross-fitting | lesson → fixed |
| 11 | Safety transfers to new datasets; 11-class diagnosis does not | mixed |
| 12 | Asking for dermoscopy on Derm7pt: accuracy 0.40 → 0.48 | ✅ |
| 13 | Skin tone 2: 11.5% missed | ❌ reported |
| 14 | Every scribe warning sign raises cancer risk (any sign: RR 3.9) | ✅ |
| 15 | Prototype's 84–90% were not real; honestly 57–75% | lesson |

---

## 7. Did we train on the external datasets?

**No — on purpose.** PAD-UFES-20 and Derm7pt are only **tests**, so they show what happens
on a new country, camera and clinic. Training on them would remove that honest check. From
PAD-UFES-20 the system does use the old **Model A** (specialist S4), the seven old models in
the legacy study, and the questionnaire that validates the warning signs.

---

## 8. How to present it (10–12 minutes, slides in `reports/DERMA-GUARD_Presentation.pptx`)

1. **Problem (1 min):** "Skin-cancer AI always answers; blur even makes it *more* confident."
2. **Idea (1 min):** four outcomes, like a careful dermatologist.
3. **Base papers (1 min):** "HOT AI showed an AI can ask for dermoscopy; Learn-then-Test gives
   a written guarantee. We combine them, add costs, a defer action and stress tests."
4. **Data (30 s):** MILK10k, split by lesion, test scored once.
5. **System (2 min):** six specialists → Devil's Advocate → fusion chosen on validation →
   gatekeeper → outcome.
6. **Results (2 min):** 89% answered, 57% confident, 3.1% missed; rejector 4.9%.
7. **Key finding (1 min):** threshold 3.5–9.3% vs ours 3.0–3.5%.
8. **Robustness and shift (1 min):** checks catch damaged photos; safety holds on two other
   datasets, fine diagnosis does not.
9. **Honesty (30 s):** our old prototype re-evaluated.
10. **Demo (2 min):** Patient view (dermoscopy unticked → request), Compare systems, Stress Lab
    with blur.
11. **Limits (30 s):** say them yourself — one seed, skin tone, not clinically validated.

**Likely questions:** see [07_panel_guide.md](07_panel_guide.md) §8, including *"Isn't this
just HOT AI?"* (Q-base) and *"What is your base paper?"*.

---

## 9. Where everything is

| Item | Location |
|---|---|
| Literature dossier (all candidate papers) | `docs/reference/literature_survey_v2.md` |
| Our earlier drafts | `paper/old1.tex`, `paper/old2.tex` |
| The paper (10 pages, 26 references, compiled) | `paper/main.tex`, `paper/main.pdf` |
| Project report (90 pages, VIT template) | `reports/DERMA-GUARD_Project_Report.docx` / `.pdf` (local; `scripts/build_report.py`) |
| Slides with speaker notes | `reports/DERMA-GUARD_Presentation.pptx` (local; `scripts/build_slides.py`) |
| Every number | [05_results.md](05_results.md) |
| The project in plain words | [13_simple_guide.md](13_simple_guide.md) |
