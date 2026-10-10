# Documentation Index

## Read in this order

| # | Doc | Why |
|---|---|---|
| 0 | **[evaluation_integrity.md](evaluation_integrity.md)** | **Start here.** Audit findings. Explains why no pre-2026-10-11 number can be trusted, and what was wrong with the code that produced them |
| 1 | [01_system_analysis.md](01_system_analysis.md) | What exists, the three-generation lineage, and the honest baseline numbers |
| 2 | [02_research_gaps_and_contributions.md](02_research_gaps_and_contributions.md) | Where the field is, the gap, contributions C1–C7, honest novelty assessment |
| 3 | [03_architecture.md](03_architecture.md) | Current architecture, target architecture, incremental migration |
| 4 | [04_mathematical_foundations.md](04_mathematical_foundations.md) | Every quantity derived — evidential heads, DS combination, EM, conformal, Chow |
| 5 | [05_fusion_techniques.md](05_fusion_techniques.md) | The fusion design space and why DS wins for *this* dataset |
| 6 | [06_retrieval_design.md](06_retrieval_design.md) | kNN retrieval as a conformity score; literature retrieval behind an evidence firewall |
| 7 | [07_experimental_protocol.md](07_experimental_protocol.md) | Hypotheses H1–H5, ablations, statistics, 14-week plan |
| 8 | [08_kaggle_training.md](08_kaggle_training.md) | GPU training, what to send back, expected results |

## Supporting

| Doc | Contents |
|---|---|
| [limitations.md](limitations.md) | Known limitations |
| [what_did_not_work.md](what_did_not_work.md) | Negative results, including the measured failure of the text modality |
| [repro_statement.md](repro_statement.md) | Seeds, hashes, hardware |
| [running_guide.md](running_guide.md) · [benchmarks_and_execution_guide.md](benchmarks_and_execution_guide.md) | Operational notes |
| [results/baseline_probe.json](results/baseline_probe.json) | Raw honest-baseline output, 3 seeds |

## Original specifications (historical)

`proj.md`, `det.md`, `detail.md` in the repository root are the v4 "DERMA-GUARD"
specification, and `project_context/PROJECT_CONTEXT.md` is the earlier v1
"HealthSphere-Derm" capstone context. They are kept for provenance. **Where they conflict
with the docs above, the docs above win** — the specifications describe models and
datasets that were never built here, and the lineage is explained in
[01_system_analysis.md](01_system_analysis.md) §1.

---

## Quick answers

**What is this project?** Evidence-aware multimodal classification of skin lesions on
PAD-UFES-20, with calibrated deferral. Not an accuracy-maximisation project — a *safety
under imperfect evidence* project.

**What is the core finding so far?** All three modalities concatenated (macro-F1 0.633)
is **worse** than image+metadata alone (0.653), because the text channel is near-useless
(0.414) and naive fusion cannot discount it.

**What is the core proposal?** Per-modality Dirichlet evidence + Dempster–Shafer
combination, so the system discovers which modality to distrust per case; plus conformal
risk control so the abstention policy carries a distribution-free guarantee.

**Can I trust the old results?** No. See doc 0.

**What needs a GPU?** Fine-tuning the image encoder — the single biggest expected gain,
since frozen ImageNet features reach only 0.477 macro-F1. See doc 8.

**What is genuinely novel?** The integration and the benchmark, not the individual
techniques. Doc 2 §6 is explicit about this.
