# DERMA-GUARD

**Evidence-Aware Specialist Consensus with Risk-Controlled Evidence Acquisition and
Deferral for Multimodal Dermatology**

> ⚠️ Research prototype. Not clinically validated. Not for medical diagnosis.

A team of AI specialists looks at a skin lesion — the clinical photo, a dermoscopy image
and the patient's details. A **Devil's Advocate** profiles how good and how reliable each
specialist's evidence is and **learns, per patient, whom to trust**; a calibrated policy
then answers, **asks for more evidence** (dermoscopy, missing details, a better photo), or
hands the case to a doctor. A care platform connects patients, doctors and follow-ups
around it, and every answer comes with an explanation of who said what and why.

| | Outcome | Share of test patients |
|---|---|---|
| ✅ | **Answer — HIGH confidence** · every safety check passed | **57%** |
| ⚠️ | **Answer — LOW confidence** · best diagnosis, clearly marked, a doctor confirms | 32% |
| 📷 | **Ask for dermoscopy first** · then decide again | 30% |
| 🧑‍⚕️ | **Defer** · the evidence contradicts itself | 11% |

**9 in 10 patients get an answer; more than half get a confident one.** Dataset:
**MILK10k** (11 diagnoses, every lesion with a clinical photo **and** a dermoscopy image).
External tests: **PAD-UFES-20** (phone photos) and **Derm7pt** (photo + dermoscopy).

---

## Results (787 held-out lesions, everything fitted on 786 validation lesions)

| | D-GUARD HIGH tier | Confidence threshold, same coverage | Learned rejector, same coverage |
|---|---|---|---|
| Patients answered confidently | 57.1% | 57.1% | 57.1% |
| Accuracy (11 classes) | 77.1% | 81.5% | 76.4% |
| **Missed cancers** | **3.1%** | 3.8% | 4.9% (p = 0.011) |

- **Six specialists, seven fusion rules compared** (cross-fitted); the learned Devil's
  Advocate (v2) wins on validation and lifts photo-only macro-F1 from 0.286 (evidential
  fusion) to **0.377**; with dermoscopy: accuracy 0.731, cancer AUROC **0.885**.
- **The safety policy is robust to the fusion under it:** across four fusion back-ends it
  misses 3.0–3.5% of cancers; a plain confidence threshold swings from 3.5% to **9.3%** —
  it is only as safe as the probabilities it is given.
- Missed-cancer bound **certified** by Learn-then-Test; shown confidences are **calibrated**.
- **DERMA-STRESS:** blur, darkness, noise and occlusion are all caught by the quality agent
  (0 confident answers on damaged photos); a photo without details triggers a request.
- **External tests, no refitting:** PAD-UFES-20 — 0 cancers missed among confident
  answers (11% of patients); Derm7pt — the system asked for dermoscopy for 39% of lesions
  (accuracy 0.40 → 0.48) and missed 3.8% among confident answers (threshold: 5.4%). The
  11-class diagnosis does not transfer; the safety decision does.
- **The old project re-scored:** its reported 84–90% came from a leaky split and
  predictions adjusted toward targets; honestly its models reach 57–75% on unseen patients.
- Reported against us: the threshold is more accurate; always answering is cheaper on mean
  cost; lighter skin tones are worse served; one training seed (more on the way).

Full results: **[docs/05_results.md](docs/05_results.md)** · how it was built and checked:
**[docs/12_how_it_was_built.md](docs/12_how_it_was_built.md)** · paper:
**[paper/main.tex](paper/main.tex)** ([compiled PDF](paper/main.pdf)).

---

## How it works

```
  clinical photo ──► S1 evidential head ─┐
  dermoscopy ──────► S2 evidential head ─┤      ┌─────────────────────┐   ┌──────────────────────┐
  patient details ─► S3 evidential head ─┼────► │ DEVIL'S ADVOCATE    │──►│ FUSION RULE picked   │
  clinical photo ──► S4 PAD-UFES Model A ┤      │ quality · learned   │   │ on validation:       │
  embeddings ──────► S5 similar-case vote┤      │ reliability · uncer-│   │ v2 learned gate /    │
  embeddings ──────► S6 embedding classif.┘      │                     │   │                      │
                                                │ tainty · track rec. │   │ v1 iterative EM / …  │
                                                │ consistency · OOD   │   └──────────┬───────────┘
                                                └─────────────────────┘              ▼
   Dempster–Shafer ignorance + conflict ─────────────────────────────────► SAFETY + COST POLICY
   conformal (benign/malignant) · Learn-then-Test · quality, OOD, missing-details gates
                                                                                     │
          HIGH answer · LOW answer · ACQUIRE (dermoscopy / details / retake) · DEFER ◄┘
   agents: quality · scribe · librarian · explainer · report + claim verifier
   platform: patient app · doctor queue · messages · follow-ups · data deletion · audit · API
```

From zero, in plain English: **[docs/02_learning_path.md](docs/02_learning_path.md)** ·
the app, view by view: **[docs/11_ui_guide.md](docs/11_ui_guide.md)**.

---

## Run it

```bash
pip install -r requirements.txt          # Windows/Anaconda: set KMP_DUPLICATE_LIB_OK=TRUE

python scripts/demo_care.py              # four patients end to end (terminal)
streamlit run app/dguard_app.py          # web app: patient / doctor / compare / stress / research
uvicorn src.api:app --port 8000          # REST API, docs at /docs
pytest tests -q                          # 230 tests, ~12 s
python scripts/make_figures.py           # every figure, from saved results
python scripts/build_report.py           # the project report (.docx + .pdf) in the VIT template
python scripts/build_slides.py           # the 16-slide presentation (.pptx)
```

Needs `./milk10k`, the trained weights in `runs/*/checkpoints/` and the fitted artifacts in
`outputs/`. Regenerate everything: [docs/12_how_it_was_built.md](docs/12_how_it_was_built.md) Part 6.

---

## Documentation

| | |
|---|---|
| Start | [docs/00_index.md](docs/00_index.md) · [01_overview.md](docs/01_overview.md) |
| Learn | **[13_simple_guide.md](docs/13_simple_guide.md)** (start here) · **[14_base_papers_and_contributions.md](docs/14_base_papers_and_contributions.md)** · [02_learning_path.md](docs/02_learning_path.md) · [08_study_plan.md](docs/08_study_plan.md) · [12_how_it_was_built.md](docs/12_how_it_was_built.md) |
| System | [03_system.md](docs/03_system.md) · [04_methods.md](docs/04_methods.md) · [10_project_structure.md](docs/10_project_structure.md) · [11_ui_guide.md](docs/11_ui_guide.md) |
| Results | [05_results.md](docs/05_results.md) · [what_did_not_work.md](docs/what_did_not_work.md) · [limitations.md](docs/limitations.md) |
| Present | [07_panel_guide.md](docs/07_panel_guide.md) · [paper/main.tex](paper/main.tex) |

---

## Positioning

**Base papers:** HOT AI (Mehta et al., JEADV 2025) — an AI that asks for dermoscopy — and
Learn-then-Test (Angelopoulos et al., 2025) — thresholds with a written guarantee. The project
also implements our own two earlier drafts: the agentic multi-model consensus
(`paper/old1.tex`) and the risk-controlled acquisition and deferral plan (`paper/old2.tex`). The individual techniques are established; the contribution is their
integration — learned, calibrated and measured on paired clinical–dermoscopic data — plus a
working care platform.

MILK10k is CC-BY-NC (credit: the MILK study team, ISIC). No images or patient data are in
this repository.
