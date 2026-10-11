# What Is Left To Do

Updated 2026-10-11.

## The short answer

**Nothing is required for the system to work.** It is trained, calibrated, and running —
app, API and demo — on the MILK10k model. No more downloads, no more training needed for
the demo or the panel.

Everything below is **optional** and only makes the paper's numbers final or stronger.

---

## 1. Status

| | Status |
|---|---|
| MILK10k data (downloaded, audited) | ✅ |
| Model trained on Kaggle (ResNet50, seed 42), verified locally | ✅ |
| Calibration bundle, acquisition estimator | ✅ `outputs/eade_milk/cost_5/` |
| Similar-case index (3,667 training lesions) | ✅ `outputs/milk_rag/` |
| Quality thresholds (fitted on MILK10k photos and dermoscopy) | ✅ |
| Agents: quality, librarian, report + verifier | ✅ |
| Specialist team (incl. original PAD-UFES Model A), Devil's Advocate, iterative adaptive fusion | ✅ `scripts/run_team.py` |
| Confidence tiers (answers 90% of patients; 60% confidently) | ✅ |
| Care platform, web app, REST API, terminal demo | ✅ |
| Experiments: decision engine, baselines, cost sweep, stress test | ✅ |
| Docs, panel guide, study plan, paper | ✅ |
| Old prototype code and data | moved to `../HealthSphere_archive/` |

---

## 2. Optional — for the paper

| # | Task | Why | Effort |
|---|---|---|---|
| 1 | Seeds 7 and 2026 on Kaggle | error bars; reviewers expect ≥3 seeds | ~2 GPU hours — [06_training.md](06_training.md) §2, cell 2 |
| 2 | DermLIP encoder on Kaggle | biggest expected accuracy gain → more HIGH-confidence answers | ~1–2 GPU hours — §2, cell 3 |
| 3 | Fill in authors in `paper/main.tex` | title block | 5 min (your `authors.md` is empty) |
| 4 | Choose the headline cost (5 or 10) | 5 = balanced (used everywhere); 10 = zero missed cancers but only ever says "cancer" | your call |

## 3. Optional — nice to have

| Task | How |
|---|---|
| LLM-written report prose | `ollama serve` then `ollama pull qwen2.5:7b`; the report agent detects it automatically and verifies every rewrite |
| Free disk space | delete `../HealthSphere_archive/` once you no longer need the old prototype |
| Patient chat / mobile app | build on `src/api.py` — the whole workflow is already exposed |

---

## 4. Before the panel

1. `streamlit run app/dguard_app.py` once, 30 minutes before (first load ~10 s).
2. Keep `python scripts/demo_care.py` ready in a terminal as the backup.
3. Read [07_panel_guide.md](07_panel_guide.md) §6–8.
