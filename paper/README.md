# Paper

IEEE conference paper (`IEEEtran`), about 8 pages with 7 figures and 4 tables.

## Compile on Overleaf

1. New Project → **Upload Project** → upload this whole folder as a zip
   (`main.tex`, `refs.bib` and the `figures/` folder).
2. Menu → Compiler: **pdfLaTeX**. Main document: `main.tex`.
3. Recompile. Overleaf runs BibTeX automatically.

Locally: `pdflatex main && bibtex main && pdflatex main && pdflatex main`.

## Figures

All figures are drawn from saved results by `python scripts/make_figures.py`, which writes
a vector PDF (used by LaTeX) and a PNG (used by the Word report) for each.

| Figure in the paper | File | What it shows |
|---|---|---|
| Fig. 1 | `fig_architecture` | the whole pipeline |
| Fig. 2 | `fig_fusion_rules` | all seven fusion rules, validation and test |
| Fig. 3 | `fig_backend_robustness` | the key finding: policy vs threshold on four fusion back-ends |
| Fig. 4 | `fig_stress` | DERMA-STRESS, model alone vs with checks |
| Fig. 5 | `fig_legacy_models` | the earlier prototype's models, reported vs re-scored |

Also available for a longer version or slides: `fig_risk_coverage` (risk–coverage curves),
`fig_calibration` (reliability diagram), `fig_confusion`, `fig_weights` (mean fusion weights),
`fig_gate_importance` (what the gate listens to), `fig_outcomes_cost` (cost sweep),
`fig_shift` (external tests), `fig_symptoms` (warning signs).

## Where each number comes from

| Paper element | Script | Output |
|---|---|---|
| Specialists, fusion table, subsets, weights | `scripts/run_team.py` | `outputs/milk_team/team_report.json` |
| Decision policy, baselines, cost sweep | `scripts/run_eade_milk.py` | `outputs/eade_team/cost_*/report.json` |
| Four back-ends (Fig. 3) | `run_team.py --method adaptive_em`, `--no-s6` variants | `outputs/eade_team*_noS6`, `outputs/eade_team_em` |
| CIs, paired tests, learned rejector, skin tone | `scripts/analyze_milk.py` | `outputs/analysis/analysis.json` |
| DERMA-STRESS | `scripts/run_stress_milk.py` | `outputs/stress_milk/report_team.json` |
| External tests | `scripts/run_shift_pad.py`, `scripts/run_shift_derm7pt.py` | `outputs/shift_*/report.json` |
| Earlier prototype | `scripts/legacy_pad_models.py` | `outputs/legacy_pad/report.json` |
| Warning signs | `scripts/analyze_pad_symptoms.py` | `outputs/analysis/pad_symptoms.json` |

## Before submitting

- [ ] Confirm the author block (names, emails, order).
- [ ] If more training seeds are run (`scripts/run_seeds.py`), add mean ± sd and update the
      Limitations paragraph.
- [ ] The paper describes the earlier prototype's evaluation problems (Section V-E). Agree on
      the wording with your guide before submission.

`old1.tex` and `old2.tex` are the two earlier drafts this paper unifies.
