# Documentation Index

**D-GUARD** — *Risk-Controlled, Cost-Aware Evidence Acquisition and Deferral for
Multimodal Dermatology AI*

A skin-lesion AI that decides, per patient, whether to **answer**, **ask for a dermoscopy
photo**, or **hand the case to a doctor** — wrapped in a care platform where patients,
doctors and follow-ups meet. Updated 2026-10-11.

---

## If you have 10 minutes

1. [README](../README.md) — what it is and the headline result
2. [14_results.md](14_results.md) — the numbers
3. Run `python scripts/demo_care.py`

## If you are new to the project

Follow **[15_study_plan.md](15_study_plan.md)** — a 10-day plan that tells you what to read,
run and open each day. Concepts are taught in [11_learning_path.md](11_learning_path.md).

## If you are presenting it

**[13_panel_guide.md](13_panel_guide.md)** — slide plan, demo script, and answers to the
questions a panel will ask.

---

## All documents, in reading order

| # | Doc | What it covers |
|---|---|---|
| 0 | **[evaluation_integrity.md](evaluation_integrity.md)** | Why no number from before 2026-10-11 can be trusted, with evidence |
| 1 | [01_project_overview.md](01_project_overview.md) | The project and its history |
| 11 | **[11_learning_path.md](11_learning_path.md)** | Every concept from zero, with diagrams, in plain English |
| 12 | [12_system_walkthrough.md](12_system_walkthrough.md) | One case traced end to end through every stage |
| 2 | [02_research_design.md](02_research_design.md) | Research question, contributions, hypotheses H1–H5, what is and is not novel |
| 3 | [03_architecture.md](03_architecture.md) | Layers, components, what was kept from the old system |
| 4 | [04_methods.md](04_methods.md) | Every formula, derived |
| 5 | [05_experimental_protocol.md](05_experimental_protocol.md) | Datasets, splits, baselines, metrics, statistics |
| 6 | [06_training_guide.md](06_training_guide.md) | Kaggle training, step by step |
| 14 | **[14_results.md](14_results.md)** | **Results on MILK10k with the trained model** |
| 7 | [07_literature.md](07_literature.md) | Verified citations and comparable numbers |
| 10 | [10_retrieval.md](10_retrieval.md) | RAG and the agent pipeline |
| 8 | [08_implementation_status.md](08_implementation_status.md) | What is built and verified |
| 16 | [16_project_structure.md](16_project_structure.md) | Folders, what is in git, what the cleanup removed |
| 9 | **[09_what_you_need_to_do.md](09_what_you_need_to_do.md)** | **What is trained, what is not, and every remaining task** |
| 13 | [13_panel_guide.md](13_panel_guide.md) | Presenting and defending the project |
| 15 | [15_study_plan.md](15_study_plan.md) | 10-day learning plan + glossary |

## Supporting

| Doc | Contents |
|---|---|
| [what_did_not_work.md](what_did_not_work.md) | 23 negative results and corrections — read this |
| [limitations.md](limitations.md) | Known limitations |
| [prereg.yaml](prereg.yaml) | Pre-registration |
| [results/](results/) | Raw baseline output (PAD-UFES-20) |
| [specs_v2/](specs_v2/) | The v2 specification, literature survey and planning notes this project consolidates |
| [../paper/main.tex](../paper/main.tex) | The IEEE paper (Overleaf-ready) |

---

## Quick answers

**What is it?** A dermatology decision-support system with three actions — PROCEED,
ACQUIRE (ask for dermoscopy), DEFER (to a doctor) — chosen per patient from calibrated
evidence, plus a patient/doctor platform with instructions, questions and follow-ups.

**Is it trained?** Yes. An evidential model was trained on MILK10k on Kaggle (seed 42) and
runs locally, matching the Kaggle result. Two more seeds and a stronger image encoder are
still to run — [09](09_what_you_need_to_do.md).

**Does it work?** At the same coverage as standard confidence-based abstention, it misses
**1.7%** of cancers instead of **5.2%**, at lower cost, and asks for dermoscopy for only
**6%** of patients. It answers alone for about a third of patients; the rest get a
tentative opinion and go to a doctor. [14](14_results.md).

**Is fusion dynamic?** Yes — each source's weight is set per case by how much evidence it
produced, shown in the app for every case. The old system's diagnosis-dependent trust is
now measured from data instead of typed by hand. [14](14_results.md) §6.

**When it defers, does it say anything?** Yes. Every DEFER carries a tentative diagnosis,
a differential, a cancer probability and an urgency flag, labelled unconfirmed, with a
strong instruction to see a doctor.

**Can I trust the old results?** No — [evaluation_integrity.md](evaluation_integrity.md).

**What is novel?** The combination into one calibrated three-action rule and the
measured value of acquiring dermoscopy on paired data — not the individual techniques.
[02](02_research_design.md).
