# Project Structure

What every folder is for, what is in git and what is not, and where to start reading.
Updated 2026-10-11 after the cleanup.

---

## 1. Top level

```
HealthSphere/
├── README.md              start here
├── app/                   the web app (Streamlit): patient, doctor, system views
├── src/                   all library code (imported by everything else)
├── scripts/               command-line tools you run
├── notebooks/             the Kaggle training script (GPU)
├── runs/                  trained-model results (metrics + predictions in git; weights not)
├── tests/                 473 tests
├── docs/                  documentation (this folder)
├── paper/                 the IEEE paper (LaTeX, Overleaf-ready)
├── configs/               YAML configuration for the legacy pipeline
├── pyproject.toml         package + dependencies
├── requirements.txt
├── Dockerfile
│
├── data/                  NOT in git - PAD-UFES-20 images, features, care database
├── milk10k/               NOT in git - MILK10k dataset (CC-BY-NC, 368 MB)
├── milk10k.zip            NOT in git - the original download; safe to delete
├── Dummy/                 NOT in git - legacy A-G model weights (still used, see §4)
├── Old_system/            NOT in git - the previous project, kept as reference
└── outputs/               NOT in git - everything scripts generate
```

---

## 2. `src/` — the library

Ordered bottom-up, the way the data flows.

| Package | What it does | Key files |
|---|---|---|
| `constants.py` | Class names (PAD-UFES 6, MILK10k 11), modality order, malignant classes | — |
| `data/` | Loading and splitting datasets | `milk10k.py` (primary), `case_loader.py` (PAD-UFES), `metadata_preprocessor.py` |
| `models/` | Legacy A–G specialist architectures and their loader | `architectures.py`, `backends.py` |
| `native/` | **The trained MILK10k model, served one case at a time** | `engine.py` |
| `evidence/` | Evidential opinions and Dempster–Shafer fusion; legacy quality/reliability scores | `evidential.py` |
| `fusion/` | Legacy iterative diagnosis-adaptive fusion (ported from the old system), consensus ladder | `adaptive_fusion.py`, `consensus.py` |
| `calibration/` | Conformal sets and Learn-then-Test | `binary.py`, `conformal.py`, `ltt.py` |
| `decision/` | **PROCEED / ACQUIRE / DEFER policy** and the patient/doctor wording | `policy.py`, `explain.py` |
| `agents/` | The agent pipeline: specialists, scribe, librarian, devil's advocate, EADE, orchestrator | `orchestrator.py`, `eade.py` |
| `retrieval/` | FAISS similar-case index (explanation only) | `rag_index.py` |
| `llm/` | LLM client, report generation, stage-2 verification | `report_gen.py`, `verification.py` |
| `care/` | **Patient / doctor platform**: database, workflow, follow-ups, audit | `service.py`, `store.py` |
| `eval/` | Metrics: AUGRC, acquisition efficiency, calibration, statistics | `selective.py`, `stats.py` |
| `stress/` | DERMA-STRESS corruptions and benchmark runner | `corruption.py`, `benchmark.py` |
| `main.py` | FastAPI server for the legacy pipeline | — |

---

## 3. `scripts/` — what you run

| Script | Purpose | Needs |
|---|---|---|
| `milk10k_audit.py` | Audit MILK10k: classes, prevalence, duplicates, split power | `milk10k/` |
| `milk_native_infer.py` | Run the trained model on val + test; verify it matches Kaggle | `runs/…/best.pt`, `milk10k/` |
| `run_eade_milk.py` | **Main experiment**: two-stage decision engine vs baselines | output of the above |
| `demo_care.py` | End-to-end story in the terminal (panel backup) | output of the above |
| `build_rag_index.py` | Build and verify the similar-case index | `data/processed/features_v2.pt` |
| `build_features.py` | Reproducible PAD-UFES features (verifies at cosine 1.000000) | `data/raw/` |
| `baseline_probe.py` | Honest PAD-UFES baselines, 3 seeds | features |
| `run_eade.py` | Decision engine in bridge mode over the legacy models | `Dummy/` weights |
| `run_stress.py` | DERMA-STRESS benchmark | `Dummy/` weights |
| `evaluate_ladder.py` | Legacy A–G ladder evaluation | `Dummy/` weights |
| `demo.py` | One case through the full agent pipeline | `Dummy/` weights |
| `cleanup.py` | Find and remove verified-redundant files (report-only by default) | — |
| `01_prepare_data.py`, `02_generate_texts.py` | Legacy PAD-UFES preparation and synthetic-text generation (kept for provenance) | — |

---

## 4. Is `Dummy/` still needed?

**Yes — keep it.** It looks like leftovers, but two things in it are live:

| Path | Used by |
|---|---|
| `Dummy/Project/saved_models/` (2.3 GB) | the A–G specialists behind `demo.py`, `run_eade.py`, `run_stress.py`, the FastAPI server |
| `Dummy/Project/data/generated_clinical_texts.csv` | the synthetic PAD-UFES narratives used by `build_features.py` and the stress benchmark |

What was removed from it: a byte-identical third copy of the 2,298 images (3.4 GB) and a
nested git repository. If you ever retire the legacy pipeline entirely, `Dummy/` can go;
the native MILK10k system does not use it.

## 5. Is `Old_system/` needed?

Not by any code. It is your previous project (React/Express/MongoDB web app, iterative
fusion, evidence manager), kept locally as reference. What was worth keeping has been
ported: the iterative diagnosis-adaptive fusion (`src/fusion/adaptive_fusion.py`), the
agent roles, the evidence profile, and the web workstation idea (now `app/dguard_app.py`).
It is gitignored. Delete it whenever you no longer need to look at it.

---

## 6. What was cleaned up (2026-10-11)

| Removed | Size | Why |
|---|---|---|
| `data/raw/imgs_part_1..3` | 3.5 GB | original download archives; all 2,298 images byte-identical to `data/raw/images` (SHA-256) |
| `Dummy/Project/data/images` | 3.4 GB | third copy of the same images; referenced by nothing |
| `outputs/checkpoints/` | 5.7 GB | checkpoints from the invalidated July run; used by nothing after the legacy evaluator was removed |
| `Dummy/Project/.git` | 33 MB | nested repository inside a vendored tree |
| `notebooks/dguard_results/dguard_out.zip` | 91 MB | duplicate of the extracted run |
| `scripts/06_evaluate.py`, `08_analysis.py`, `train_derma_guard.py`, `dataset_audit.py` | — | legacy training/evaluation of the invalid pipeline; replaced by the Kaggle trainer and `milk10k_audit.py` |
| `app/patient_app.py`, `app/clinician_console.py` | — | replaced by `app/dguard_app.py` |
| empty `src/api`, `src/config`, `src/explain`, `src/repositories`, `src/services`, `configs/experiment`, `models/` | — | placeholders with no code |

About **12.7 GB** freed. Verified afterwards: 470 tests pass, demo runs all three paths.

Moved: the Kaggle run → `runs/milk10k_resnet50_seed42/`; the v2 specification, literature
survey and planning notes → `docs/specs_v2/`.

---

## 7. What is in git

| In git | Not in git (`.gitignore`) |
|---|---|
| `src/`, `scripts/`, `app/`, `tests/`, `notebooks/`, `configs/`, `docs/`, `paper/` | `data/`, `milk10k/`, `milk10k.zip` — datasets (MILK10k is CC-BY-NC: never republish) |
| `runs/*/metrics.json`, `manifest.json`, `history.json`, `predictions_test.csv` | `runs/*/checkpoints/` — model weights |
| | `Dummy/`, `Old_system/`, `outputs/`, `.env` |

---

## 8. Where to start reading code

1. `src/care/service.py` — the top-of-file diagram is the whole user-facing system
2. `src/native/engine.py` — how one case is assessed
3. `src/decision/policy.py` — how the action is chosen
4. `src/evidence/evidential.py` — the maths everything rests on
5. `scripts/run_eade_milk.py` — how it was evaluated
