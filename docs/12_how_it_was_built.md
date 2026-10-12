# How the project was built, tested and verified

This is the story of the project in the order it happened, what every claim rests on, and
the questions examiners (and we) kept asking. Simple English; numbers link to
[05_results.md](05_results.md).

---

## Part 1 — The journey in eight steps

| Step | What we did | Why | Where |
|---|---|---|---|
| 1 | Read the two earlier designs: the agentic multi-model consensus (`paper/old1.tex`) and the risk-controlled acquisition/deferral design (`paper/old2.tex`) | keep the good ideas of both | `paper/README.md` |
| 2 | **Audited the old system** and found three problems (Part 4) | a result is only as good as its evaluation | §4 below |
| 3 | Chose **MILK10k** and wrote an exact loader with a **lesion-level split** (no lesion in two splits) | the only public set with a photo *and* dermoscopy for every lesion — needed to test "ask for dermoscopy" | `src/data/milk10k.py` |
| 4 | Trained the evidential network on Kaggle (ResNet50, three heads: photo, dermoscopy, details) | evidential heads give *ignorance* and *conflict*, not just a probability | `notebooks/kaggle_train_dguard.py` |
| 5 | Built the **decision engine**: conformal sets, Learn-then-Test, a cost matrix, four outcomes | turn probabilities into safe actions with a certified bound | `src/decision/`, `src/calibration/` |
| 6 | Rebuilt the old design's **specialist team + Devil's Advocate + adaptive fusion**, with every hand-set number replaced by a measured or learned one | the old ideas, made honest | `src/fusion/` |
| 7 | Added **Devil's Advocate v2** (a learned gate that uses *every* evidence signal), a **comparison of seven fusion rules** and every specialist subset, all **cross-fitted** | the v1 weights only used quality × reliability | `src/fusion/gating.py`, `methods.py` |
| 8 | Stress tests, two external datasets, statistics, the care platform and the app | does it stay safe when things go wrong? | `scripts/`, `app/` |

---

## Part 2 — How each part was checked

| Part | How we know it works | Evidence |
|---|---|---|
| Data split | asserts that no lesion appears in two splits; split sizes match the Kaggle run exactly | `src/data/milk10k.py`, `scripts/milk_native_infer.py` |
| Local model = Kaggle model | test macro-F1 0.376 locally vs 0.379 on Kaggle | 05_results §8 |
| Evidential maths | 34 unit tests (belief + ignorance = 1, vacuous opinion is neutral, conflict) | `tests/test_evidential.py` |
| Conformal / LTT | tests that coverage and the risk bound hold on synthetic data | `tests/test_calibration.py` |
| Fusion rules | every rule returns valid distributions; the learned gate trusts an informative specialist over noise; cross-fitting returns out-of-fold outputs | `tests/test_fusion_team.py`, `tests/test_v2_components.py` |
| Quality checks | occlusion detected, clean texture not; thresholds are percentiles of 300 real training photos | `tests/test_v2_components.py` |
| Care platform | routing of every outcome, doctor queue order, follow-ups, change alerts, **deletion removes records and uploaded files but never dataset images** | `tests/test_care_platform.py`, `tests/test_v2_components.py` |
| Explanation | names dissent; the verifier rejects invented diagnoses and percentages | `tests/test_v2_components.py` |
| App | every view rendered headless (Streamlit AppTest) with no exception; "run every system" executes | see commit log |
| **No test-set peeking** | every choice — fusion weights, which fusion rule wins, thresholds, calibration — is made on the 786 validation lesions; the 787 test lesions are scored once | `scripts/run_team.py`, `scripts/run_eade_milk.py` |

`pytest tests -q` → **229 tests, ~12 s.**

One honest correction happened during this work: the v1 fusion had been fitted on the same
validation lesions that later calibrated the safety guarantees (in-sample). With
**cross-fitting** (fit on 4/5 of validation, score the other 1/5) the high-confidence share
moved from 60.2% to 57.2% for v1. The new numbers are the ones reported.

---

## Part 3 — Claims and the evidence behind each

| # | Claim | Evidence | Status |
|---|---|---|---|
| 1 | 9 in 10 patients get an answer, with an explicit HIGH / LOW tier | 89.5% answered; HIGH 54.4% at 76.9% accuracy | ✅ measured |
| 2 | The missed-cancer rate among HIGH answers stays at **3.3–3.5% with either fusion back-end** (v1 EM, v2 learned gate) | 05_results §5 | ✅ robust |
| 3 | A plain confidence threshold is **only as safe as the probabilities under it** — 9.3% missed with v1, 3.5% with v2 | 05_results §5 | ✅ a finding, stated both ways |
| 4 | The missed-cancer rate is **certified** by Learn-then-Test (α = 5%, δ = 10%) at both stages | λ̂ 0.212 / 0.365, p ≈ 0.08–0.10 | ✅ under exchangeability |
| 5 | The learned Devil's Advocate (v2) is the best fusion rule on validation, and the best at stage A on test | 05_results §3 | ✅ stage A; ❌ stage B test (v1 better) |
| 6 | The system notices damaged evidence | DERMA-STRESS: blur/dark/occlusion/noise flagged, missing details requested | ✅ (see §7 for what is left) |
| 7 | It stays safe under distribution shift | PAD-UFES-20: 0.0% missed (HIGH 14%); Derm7pt: 3.8% missed (HIGH 11%), dermoscopy asked for 55% and accuracy 0.30 → 0.43 | ✅ safety / ❌ fine diagnosis |
| 8 | A learned rejector is as safe as our HIGH tier | p = 0.63, not different | ✅ reported against us |
| 9 | Faster/cheaper than always answering | no — always answering has lower mean cost | ❌ reported against us |
| 10 | The scribe's warning signs justify raising urgency | PAD-UFES-20 questionnaire: any sign → cancer 48.7% vs 12.4% (RR 3.9) | ✅ on another dataset |

What we do **not** claim: clinical validity, beating the MILK10k challenge leaderboard,
fairness for the lightest skin tones (too few cases), or more than one training seed until
the extra Kaggle seeds are in ([run_seeds.py](../scripts/run_seeds.py)).

---

## Part 4 — The old system: what we kept, what we could not

**Kept (and made measurable):** the specialist team, the Devil's Advocate's five evidence
dimensions, the diagnosis-adaptive prior, the iterative re-fusion, the 4-level consensus,
OOD by Mahalanobis distance, the scribe, the librarian, and the original **Model A**
(as specialist S4).

**Three problems found in the old code** (`HealthSphere_archive/Old_system`):

1. **The split was by photo, not by lesion or patient.** PAD-UFES-20 has 2,298 photos of
   1,641 lesions from 1,373 patients, split at random by photo. Re-running that split:
   **43% of test photos show a lesion that was also in training, 62% a patient who was.**
   This is why the old Models A–G scored 85–90%; those numbers are not comparable.
2. **The live "vision panel" looked up the true label.** In `agents/vision_team.py`, when the
   uploaded file name was a PAD-UFES image, Model A's output was replaced by 0.925 on the
   *true* class; the "DermLIP", "PanDerm" and "Swin" panel members were small arithmetic
   variations of Model A, not separate models.
3. **The evidence weights were hand-typed** (e.g. trust in each input per diagnosis), and the
   Q/R networks were never trained. In the formula E = Q·R·U·C·S·M, C and S are the same for
   every specialist of a case, so they **cancel** when the weights are normalised — only
   Q, R and U ever moved the weights.

**Models B–G.** They exist and were trained, but none can run on MILK10k:
B (metadata) needs PAD-UFES's 116 clinical fields (smoking, pesticides, itch, bleeding…),
C/E/F/G need free text that was **generated from those same fields**, and D needs both.
MILK10k has none of these fields. Model A (image only) is the one that transfers, so it is
in the team. On PAD-UFES itself they cannot be tested fairly either (they were trained on it).

---

## Part 5 — Questions we kept being asked

**Why is the confidence often low?** Four honest reasons:
1. **Eleven diagnoses, several look-alikes** (BCC / AKIEC / SCCKA; BKL / MEL) and several
   rare classes (5–7 test lesions). 72% accuracy is the measured level with dermoscopy.
2. **Evidential heads are built to say "I don't know"**: average ignorance is 0.68 without
   dermoscopy, so each head alone spreads its belief (the photo head is often 10–40% sure).
3. **The shown confidence is calibrated**: "of past validation cases with this confidence,
   this share were right". On test it shows 0.74 on average vs 0.72 actual — honest, not
   flattering. A raw softmax would *look* higher and be wrong more often than it says.
4. **The backbone is an ImageNet ResNet50**, the weakest part. The way to raise confidence
   honestly is a dermatology backbone (DermLIP/PanDerm — the Kaggle script supports it),
   more seeds, and dermoscopy.

**Does the system ask for more data when it needs it?** Yes, in three ways:
(a) **dermoscopy** — when the expected cost of answering now is higher than asking
(26% of test patients); (b) **patient details** — a photo with no age/sex/site is not
answered until the details are sent (new; DERMA-STRESS showed v2 missed 17.6% of cancers
in that situation without the request); (c) **a better photo** — a severe quality problem
blocks a confident answer and the app offers a retake. Under v1 the dermoscopy requests
went to the right patients (acquisition efficiency +0.30 vs −0.04 for "ask the most
uncertain"); under v2 the two are equal (+0.20) — reported.

**How are specialists combined, and how do we pick the best?** Seven rules
(`src/fusion/methods.py`) are fitted with 5-fold cross-fitting on validation; the rule with
the best out-of-fold macro-F1 is picked; then every subset of specialists is scored with
it, and an oracle shows the ceiling of picking the right specialist per case. All of this
is visible in *Compare systems*.

**How much weight does each kind of evidence get, and why do modalities disagree?** Per
case: the *Why this result* panel (Part 4 of [11_ui_guide.md](11_ui_guide.md)). On average
(test, v2): similar-case vote ~0.77, photo ~0.09, dermoscopy ~0.11, Model A ~0.03. The
gate listens most to the specialist's track record for the suspected diagnosis, how
unusual the case is (OOD), quality and learned reliability.

**Was MILK10k the right choice?** Yes. It is the only public dataset where *every* lesion
has a clinical photo **and** a dermoscopy image (5,240 lesions, 11 diagnoses, skin tone),
which is what "ask for dermoscopy" needs. PAD-UFES-20 has no dermoscopy; HAM10000/ISIC have
no clinical photos; Derm7pt has both but only 1,011 lesions and 5 diagnoses. We use the
other two as **external tests** instead.

**Can we just raise the accuracy?** Only with a better model or more data — not by
changing numbers. Every number in this repository is produced by a script from the saved
predictions, and the test split is scored once. That is what makes it publishable.

---

## Part 6 — Reproduce everything

```bash
python scripts/milk_native_infer.py
python scripts/build_specialists.py
python scripts/run_team.py                                  # fusion comparison + subsets
python scripts/run_team.py --method adaptive_em --out outputs/milk_team_em --no-subsets
for c in 2 3 5 10; do python scripts/run_eade_milk.py --native outputs/milk_team --missed-cost $c --out outputs/eade_team/cost_$c; done
python scripts/run_eade_milk.py --native outputs/milk_team_em --missed-cost 5 --out outputs/eade_team_em/cost_5
python scripts/analyze_milk.py
python scripts/run_stress_milk.py --out outputs/stress_milk/report_team.json
python scripts/run_shift_pad.py
python scripts/run_shift_derm7pt.py --data dataset/release_v0
python scripts/run_seeds.py runs/milk10k_resnet50_seed7 runs/milk10k_resnet50_seed2026
```
