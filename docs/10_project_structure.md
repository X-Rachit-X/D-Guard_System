# Project Structure (final)

Only the final system is in the project folder. The earlier PAD-UFES prototype was moved —
not deleted — to `../HealthSphere_archive/`.

---

## 1. Top level

```
HealthSphere/
├── README.md               start here
├── app/dguard_app.py       the web app: patient, doctor, system views
├── src/                    the library (everything imports from here)
├── scripts/                6 command-line tools
├── notebooks/              kaggle_train_dguard.py — training (GPU)
├── runs/                   trained runs: metrics + predictions in git, weights not
├── tests/                  217 tests, ~6 seconds
├── docs/                   documentation
├── paper/                  IEEE paper (LaTeX, Overleaf-ready)
├── pyproject.toml, requirements.txt, Dockerfile
│
├── milk10k/                NOT in git — the dataset (CC-BY-NC)
├── outputs/                NOT in git — calibration bundle, retrieval index, experiment reports
└── data/care/              NOT in git — the app's database and uploads (created at runtime)
```

## 2. `src/`

```
src/
├── constants.py          11 MILK10k classes, malignant set, MONET columns
├── data/milk10k.py       exact loader, lesion-level splits
├── evidence/evidential.py   opinions, Dempster–Shafer fusion (reference implementation)
├── calibration/
│   ├── binary.py         benign/malignant conformal sets (Mondrian LAC) — used by the gate
│   ├── ltt.py            Learn-then-Test risk control
│   └── conformal.py      11-class APS/RAPS (kept for comparison)
├── decision/
│   ├── policy.py         four outcomes, tiers, gates, cost matrix
│   └── explain.py        patient and doctor wording
├── native/engine.py      the trained model, one case at a time
├── agents/
│   ├── quality.py        image-quality agent
│   ├── librarian.py      similar confirmed cases (RAG)
│   ├── report.py         report + claim verifier
│   └── pipeline.py       quality → engine → librarian → report
├── retrieval/rag_index.py   FAISS cosine index with spec checking
├── care/
│   ├── service.py        the workflow
│   └── store.py          SQLite + audit log
├── llm/client.py         optional local LLM (Ollama)
├── eval/selective.py     AUGRC, acquisition efficiency, calibration metrics
└── api.py                REST API
```

## 3. `scripts/` — in the order you would run them

| Script | Does | Time (CPU) |
|---|---|---|
| `milk10k_audit.py` | audit the dataset | seconds |
| `milk_native_infer.py` | run the trained model on val + test, verify against Kaggle | ~2.5 min |
| `run_eade_milk.py` | the main experiment: decision engine vs 6 baselines; writes the calibration bundle | ~20 s |
| `build_milk_index.py` | similar-case index over training lesions | ~3 min |
| `run_stress_milk.py` | robustness to degraded photos / missing details | ~3 min |
| `demo_care.py` | four patients end to end | ~30 s |

## 4. `docs/`

```
docs/
├── 00_index.md              map of the docs
├── 01_overview.md           what it is, which dataset, old → new, is it trained
├── 02_learning_path.md      every concept from zero
├── 03_system.md             architecture, agents, platform, API, one case traced
├── 04_methods.md            formulas
├── 05_results.md            all results
├── 06_training.md           Kaggle training
├── 07_panel_guide.md        presenting and defending
├── 08_study_plan.md         10-day plan + glossary
├── 09_todo.md               what is left (nothing required)
├── 10_project_structure.md  this file
├── limitations.md
├── what_did_not_work.md
└── reference/               history and background: integrity audit, literature,
                             research design, protocol, pre-registration, v2 specs,
                             PAD-UFES prototype results
```

## 5. What was moved out, and why

| Moved to `../HealthSphere_archive/` | Was | Why it is not in the final project |
|---|---|---|
| `Dummy/` | legacy A–G model weights (PAD-UFES) | the final system uses the MILK10k model |
| `data/` | PAD-UFES-20 images and features | the final system uses MILK10k only |
| `Old_system/` | the previous project (React/Express/MongoDB) | its ideas were ported (see [01_overview.md](01_overview.md) §3) |
| `legacy_code/` | 7-model pipeline, old orchestrator, fusion ladder, old evidence scores, old LLM report chain, 14 old test files, configs | replaced by `src/native`, `src/agents`, `src/decision` |
| `milk10k.zip` | the original download | `milk10k/` is extracted and verified |
| `docs_superseded/` | earlier versions of docs now merged into the final set | — |
| `outputs_legacy/` | old experiment outputs | — |

Nothing was deleted. Delete the archive when you no longer need the history.

## 6. In git / not in git

| In git | Not in git |
|---|---|
| `src/`, `scripts/`, `app/`, `tests/`, `notebooks/`, `docs/`, `paper/` | `milk10k/` (licence: never republish) |
| `runs/*/metrics.json`, `manifest.json`, `history.json`, `predictions_test.csv` | `runs/*/checkpoints/` (weights) |
| | `outputs/`, `data/`, `.env` |

## 7. Where to start reading code

1. `src/care/service.py` — top-of-file diagram = the user-facing system
2. `src/agents/pipeline.py` — the agents around the engine
3. `src/native/engine.py` — one case through the model
4. `src/decision/policy.py` — how the outcome is chosen
5. `src/evidence/evidential.py` — the maths it rests on
