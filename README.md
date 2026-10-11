# DERMA-GUARD

**Evidence-Aware Specialist Consensus with Risk-Controlled Evidence Acquisition and
Deferral for Multimodal Dermatology**

> ⚠️ Research prototype. Not clinically validated. Not for medical diagnosis.

A team of AI specialists looks at a skin lesion — the clinical photo, a dermoscopy image
and the patient's details. A **Devil's Advocate** checks how good and how reliable each
specialist's evidence is; an **iterative diagnosis-adaptive fusion** decides, per patient,
whom to trust; and a calibrated policy gives one of four outcomes. A care platform connects
patients, doctors and follow-ups around it.

| | Outcome | Share of test patients |
|---|---|---|
| ✅ | **Answer — HIGH confidence** · every safety check passed | **60%** |
| 📷 | **Ask for dermoscopy** · then decide again | 27% |
| ⚠️ | **Answer — LOW confidence** · best diagnosis, clearly marked, a doctor confirms | 29% |
| 🧑‍⚕️ | **Defer** · the evidence contradicts itself | 10% |

**9 in 10 patients get an answer; 6 in 10 get a confident one.** Dataset: **MILK10k**
(11 diagnoses, every lesion with a clinical photo **and** a dermoscopy image).

---

## Results (787 held-out lesions)

| | D-GUARD (HIGH answers) | Confidence threshold, same coverage |
|---|---|---|
| Patients answered confidently | **60.2%** | 60.2% |
| Accuracy (11 classes) | **76.2%** | 72.4% |
| **Missed cancers** | **3.0%** | 9.3% |
| Mean cost per case | **1.013** | 1.037 |

- **Adaptive specialist fusion: macro-F1 0.376 → 0.453** (+20%) over evidential fusion alone.
- A simple average of the same specialists is *worse* — the gain is the evidence-aware weighting.
- The original project's **PAD-UFES Model A** is in the team; the Devil's Advocate learns it
  transfers poorly to MILK10k and gives it ~2% weight.
- Dermoscopy is requested selectively (acquisition efficiency **+0.33** vs −0.03 for
  "ask the most uncertain").
- Missed-cancer bound **certified** by Learn-then-Test; confidences shown are **calibrated**.

Reported against us: always answering is cheaper on mean cost; occluded photos and missing
details still cause confident errors; one training seed. Full results:
**[docs/05_results.md](docs/05_results.md)** · paper: **[paper/main.tex](paper/main.tex)**.

---

## How it works

```
  clinical photo ──► S1 evidential head ─┐
  dermoscopy ──────► S2 evidential head ─┤      ┌───────────────────┐    ┌──────────────────┐
  patient details ─► S3 evidential head ─┼────► │ DEVIL'S ADVOCATE  │──► │ ITERATIVE        │
  clinical photo ──► S4 PAD-UFES Model A ┤      │ quality Q         │    │ DIAGNOSIS-       │
  embeddings ──────► S5 similar-case vote┘      │ learned reliab. R │    │ ADAPTIVE FUSION  │
                                                │ uncertainty U     │    │ (EM, measured    │
                                                │ consistency C     │    │  priors, ≤3 its) │
                                                │ missingness M     │    └────────┬─────────┘
                                                │ OOD (Mahalanobis) │             │
                                                └───────────────────┘             ▼
   Dempster–Shafer ignorance + conflict ──────────────────────────────► SAFETY + COST POLICY
   conformal (benign/malignant) · Learn-then-Test · quality & OOD gates · cost matrix
                                                                                  │
                         HIGH answer · ACQUIRE dermoscopy · LOW answer · DEFER ◄──┘
   agents: quality check · librarian (similar confirmed cases) · report + claim verifier
   platform: patient app · doctor queue · questions · follow-ups + change check · audit · API
```

From zero, in plain English: **[docs/02_learning_path.md](docs/02_learning_path.md)**.

---

## Run it

```bash
pip install -r requirements.txt          # Windows/Anaconda: set KMP_DUPLICATE_LIB_OK=TRUE

python scripts/demo_care.py              # four patients end to end (terminal)
streamlit run app/dguard_app.py          # web app: patient / doctor / system
uvicorn src.api:app --port 8000          # REST API, docs at /docs
pytest tests -q                          # 217 tests, ~6 s
```

Needs `./milk10k`, the trained weights in `runs/*/checkpoints/` and the fitted artifacts in
`outputs/`. Regenerate everything: [docs/05_results.md](docs/05_results.md) §10.

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

Built from two designs: the original agentic multi-model consensus with diagnosis-adaptive
fusion (`paper/old1.tex`), and the risk-controlled acquisition and deferral design
(`paper/old2.tex`). The individual techniques are established; the contribution is their
integration — learned, calibrated and measured on paired clinical–dermoscopic data — plus a
working care platform.

MILK10k is CC-BY-NC (credit: the MILK study team, ISIC). No images or patient data are in
this repository.
