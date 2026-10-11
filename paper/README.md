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
- [ ] Re-run seeds 7 and 2026, then update Tables II–V with mean ± spread
      (`docs/09_what_you_need_to_do.md` ①).
- [ ] If the DermLIP run improves the model, replace the numbers and the Limitations line.
- [ ] Open each reference in `refs.bib` marked "check" and confirm the exact title.
- [ ] Choose the headline cost (5 or 10) and make the abstract match.

## Where each number comes from

| Paper element | Source |
|---|---|
| Table I (classifier, dermoscopy value) | `outputs/milk_native/summary.json`, `runs/milk10k_resnet50_seed42/metrics.json` |
| Table II (ablation) | `runs/milk10k_resnet50_seed42/metrics.json` → `availability_ablation` |
| Table III (policy) | `outputs/eade_milk/cost_5/report.json` |
| Table IV (cost sweep) | `outputs/eade_milk/sweep_*/report.json` |
| Reliability values | `outputs/eade_milk/cost_5/report.json` → `modality_reliability_by_class` |
| AE, AUGRC, LTT λ̂ | `outputs/eade_milk/cost_5/report.json`, `cost_10/report.json` |
