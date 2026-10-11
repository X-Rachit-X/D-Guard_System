# Paper

IEEE conference paper, `IEEEtran` class.

## Compile on Overleaf

1. New Project → **Upload Project** → upload `main.tex` and `refs.bib` (or a zip of this folder).
2. Menu → Compiler: **pdfLaTeX**. Main document: `main.tex`.
3. Recompile. Overleaf runs BibTeX automatically.

`IEEEtran`, `tikz`, `booktabs`, `siunitx` are all in Overleaf's default TeX Live — no
extra packages to upload.

## Compile locally

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Before submitting

- [ ] Fill in the author block (names, department, institution, email).
- [ ] Optional: run seeds 7 and 2026, then report mean ± spread (`docs/06_training.md` §2).
- [ ] If the DermLIP run improves the model, replace the numbers and the Limitations line.
- [ ] Open each reference in `refs.bib` marked "check" and confirm the exact title.
- [ ] Choose the headline cost (5 or 10) and make the abstract match.

## Where each number comes from

| Paper element | Source |
|---|---|
| Specialists (Table II), fusion (Table III) | `outputs/milk_team/team_report.json` — `scripts/run_team.py` |
| Reliability per diagnosis (Table IV) | `outputs/eade_team/cost_5/report.json` → `modality_reliability_by_class` |
| Policy, tiers (Tables V–VI) | `outputs/eade_team/cost_5/report.json` (`baselines`, `tiers`) |
| Cost sweep (Table VII) | `outputs/eade_team/cost_{2,3,5,10}/report.json` |
| Stress (Table VIII) | `outputs/stress_milk/report_team.json` |
| Heads-only ablation | `outputs/eade_milk/cost_5/report.json` |
| Dermoscopy value (ignorance, AUROC) | `outputs/milk_native/summary.json` |

Previous versions: `old1.tex` (agentic consensus), `old2.tex` (risk-controlled design).
Regenerate everything: `docs/05_results.md` §10.
