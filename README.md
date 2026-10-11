# D-GUARD

**Risk-Controlled, Cost-Aware Evidence Acquisition and Deferral for Multimodal Dermatology AI**

> ⚠️ Research prototype. Not clinically validated. Not for medical diagnosis.

A skin-lesion AI trained on **MILK10k** (11 diagnoses, every lesion with a clinical photo
**and** a dermoscopy image). For each patient it gives one of four outcomes — and around it
sits a care platform where patients, doctors and follow-ups meet.

| | Outcome | Share of test patients |
|---|---|---|
| ✅ | **Answer — HIGH confidence** · every safety check passed | 35% |
| 📷 | **Ask for dermoscopy** · then answer | 6% |
| ⚠️ | **Answer — LOW confidence** · best diagnosis, clearly marked, a doctor confirms | 55% |
| 🧑‍⚕️ | **Defer** · photo and details contradict each other | 10% |

**9 in 10 patients get an answer.**

---

## Results (787 held-out MILK10k lesions)

| | HIGH-confidence answers | Standard approach, same coverage |
|---|---|---|
| Accuracy (11 classes) | **83.8%** | 84.2% |
| Balanced accuracy | **0.355** | 0.300 |
| **Missed cancers** | **1.8%** | 4.7% |
| Dermoscopy needed | 6.4% of patients | — |

- **A third of the missed cancers** at the same coverage and accuracy.
- LOW-confidence answers: true diagnosis in the top 3 for **82%** of patients.
- Dermoscopy raises cancer-vs-benign AUROC **0.80 → 0.86** and cuts the model's ignorance
  **0.68 → 0.44**; a learned estimator decides *when* it is worth asking (acquisition
  efficiency **+0.27** vs **+0.01** for "ask the most uncertain").
- The missed-cancer bound is **statistically certified** (Learn-then-Test).

Reported against us, and in the docs: combining all three inputs scores below the best pair
(0.379 vs 0.403 macro-F1); on average cost, always answering is cheaper; one training seed.
Everything: **[docs/05_results.md](docs/05_results.md)**.

---

## How it works

```
 photo ──────┐   ┌──────────────────┐   ┌──────────────┐   ┌─────────────┐   HIGH answer
 dermoscopy ─┼──►│ evidential head  │──►│ Dempster–    │──►│ safety gates│──► ask dermoscopy
 details ────┘   │ per input:       │   │ Shafer fusion│   │ + cost      │   LOW answer
                 │ belief +         │   │ belief,      │   │             │   defer
                 │ ignorance u      │   │ ignorance,   │   │ conformal,  │   + tentative
                 └──────────────────┘   │ CONFLICT     │   │ Learn-then- │     opinion
                                        └──────────────┘   │ Test        │
                                                           └─────────────┘
   agents:   quality check · similar confirmed cases (RAG) · report + claim verifier
   platform: patient app · doctor queue · questions · follow-ups + change check · audit · REST API
```

1. **Each input says how much it knows** — belief in each diagnosis plus ignorance `u`.
2. **Dynamic fusion** — Dempster–Shafer: a missing input changes nothing; a weak one gets
   little weight, per patient, automatically; disagreement becomes a separate number.
3. **Calibrated gates** — no false reassurance while cancer is possible; a certified
   threshold on missed cancers; contradictions go to a human.
4. **Cheapest action** — answer, ask for dermoscopy, or review, priced with an explicit cost
   matrix.

From zero: **[docs/02_learning_path.md](docs/02_learning_path.md)**.

---

## Run it

```bash
pip install -r requirements.txt
# Windows/Anaconda: set KMP_DUPLICATE_LIB_OK=TRUE

python scripts/demo_care.py            # four patients end to end (terminal)
streamlit run app/dguard_app.py        # web app: patient / doctor / system
uvicorn src.api:app --port 8000        # REST API, docs at /docs
pytest tests -q                        # 217 tests, ~6 s
```

Needs `./milk10k` (the dataset) and `runs/milk10k_resnet50_seed42/checkpoints/best.pt`
(the trained weights). To rebuild everything derived from them:

```bash
python scripts/milk_native_infer.py                    # ~2.5 min
python scripts/run_eade_milk.py --missed-cost 5 --out outputs/eade_milk/cost_5
python scripts/build_milk_index.py                     # ~3 min
```

---

## Status

| | |
|---|---|
| Dataset | **MILK10k only** (PAD-UFES-20 belonged to the earlier prototype — [docs/01_overview.md](docs/01_overview.md) §2) |
| Model | trained on Kaggle, runs locally, matches the Kaggle result |
| Decision engine, tiers, agents, platform, app, API | ✅ working |
| Required training or downloads | **none** |
| Optional | seeds 7 & 2026, DermLIP encoder — [docs/09_todo.md](docs/09_todo.md) |

---

## Documentation

| | |
|---|---|
| Start | [docs/00_index.md](docs/00_index.md) · [01_overview.md](docs/01_overview.md) |
| Learn | [02_learning_path.md](docs/02_learning_path.md) · [08_study_plan.md](docs/08_study_plan.md) |
| System | [03_system.md](docs/03_system.md) · [04_methods.md](docs/04_methods.md) · [10_project_structure.md](docs/10_project_structure.md) |
| Results | [05_results.md](docs/05_results.md) · [what_did_not_work.md](docs/what_did_not_work.md) · [limitations.md](docs/limitations.md) |
| Present | [07_panel_guide.md](docs/07_panel_guide.md) · [paper/main.tex](paper/main.tex) |

---

## Positioning

The individual techniques — evidential learning, Dempster–Shafer fusion, conformal
prediction, Learn-then-Test, learning to defer — are established. The contribution is their
combination into one calibrated decision with confidence tiers, the **measured** value of
asking for dermoscopy on paired data, and a working care platform around it.

MILK10k is CC-BY-NC (credit: the MILK study team, ISIC). No images or patient data are in
this repository.
