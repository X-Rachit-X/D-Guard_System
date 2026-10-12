# Documentation

**D-GUARD** — Risk-Controlled, Cost-Aware Evidence Acquisition and Deferral for Multimodal
Dermatology AI. Final version, 2026-10-11. Dataset: **MILK10k**.

---

## Read in this order

| # | Doc | What you get |
|---|---|---|
| 01 | [01_overview.md](01_overview.md) | what it is, which dataset (MILK10k), old → new, is it trained |
| 02 | [02_learning_path.md](02_learning_path.md) | every concept from zero, with diagrams |
| 03 | [03_system.md](03_system.md) | architecture, agents, platform, API, one patient traced |
| 04 | [04_methods.md](04_methods.md) | the formulas |
| 05 | [05_results.md](05_results.md) | all results |
| 06 | [06_training.md](06_training.md) | Kaggle training (optional runs) |
| 07 | [07_panel_guide.md](07_panel_guide.md) | slides, demo script, Q&A |
| 08 | [08_study_plan.md](08_study_plan.md) | 10-day learning plan + glossary |
| 09 | [09_todo.md](09_todo.md) | what is left — nothing required |
| 10 | [10_project_structure.md](10_project_structure.md) | folders, what was moved out |
| 11 | [11_ui_guide.md](11_ui_guide.md) | **the app, view by view** — what every button does |
| 14 | [14_base_papers_and_contributions.md](14_base_papers_and_contributions.md) | **base papers → what we did → what we achieved → what we publish; all findings** |
| 13 | [13_simple_guide.md](13_simple_guide.md) | **the whole project in plain words, 15 minutes — start here** |
| 12 | [12_how_it_was_built.md](12_how_it_was_built.md) | **how it was built and checked**; every claim → its evidence; FAQ (why is confidence low? why MILK10k? models B–G?) |

Also: [what_did_not_work.md](what_did_not_work.md) · [limitations.md](limitations.md) ·
the paper in [../paper/main.tex](../paper/main.tex).

## Reference (history and background)

| | |
|---|---|
| [reference/evaluation_integrity.md](reference/evaluation_integrity.md) | why the original prototype's numbers were invalid |
| [reference/literature.md](reference/literature.md) | verified citations |
| [reference/research_design.md](reference/research_design.md) | hypotheses H1–H5, novelty |
| [reference/experimental_protocol.md](reference/experimental_protocol.md) | protocol |
| [reference/prereg.yaml](reference/prereg.yaml) | pre-registration |
| [reference/specs_v2/](reference/specs_v2/) | the v2 specification and literature survey |
| [reference/pad_ufes_prototype_results/](reference/pad_ufes_prototype_results/) | earlier prototype baselines (PAD-UFES-20) |

---

## Quick answers

**Which dataset?** MILK10k, only. PAD-UFES-20 was the earlier prototype's dataset.

**Multiclass?** Yes — 11 diagnoses, 5 of them malignant.

**Trained?** Yes. Nothing else is required to run it.

**How often does it answer?** 9 in 10 patients: 57% HIGH confidence, 32% LOW confidence
(doctor confirms), 11% deferred because the evidence contradicts itself.

**How good?** HIGH-confidence answers: 77% accurate on 11 classes, 3.1% missed cancers
(learned rejector 4.9%, confidence threshold 3.8%). Across four fusions D-GUARD stays at
3.0–3.5% while the threshold ranges 3.5–9.3%. Fusion: photo-only macro-F1 0.286 → 0.377.

**Is fusion dynamic?** Yes — the old project's iterative diagnosis-adaptive fusion runs over
a five-specialist team (including the original PAD-UFES Model A); each specialist's weight
is set per patient by its learned reliability, quality and the suspected diagnosis, and is
shown for every case.
