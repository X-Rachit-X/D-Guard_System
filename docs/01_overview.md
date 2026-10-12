# D-GUARD — Project Overview (final)

**Risk-Controlled, Cost-Aware Evidence Acquisition and Deferral for Multimodal Dermatology AI**

This is the one-page answer to "what is this project, what does it use, and is it done?"

---

## 1. What it is, in one paragraph

A patient photographs a skin spot. D-GUARD looks at the photo and the patient's details
and gives one of four outcomes:

| Outcome | Meaning | Share of test patients |
|---|---|---|
| ✅ **Answer — HIGH confidence** | every safety check passed; calibrated answer | 54% |
| 📷 **Ask for dermoscopy** | a close-up through a skin lens is expected to help; then it answers | 27% (then answered) |
| ⚠️ **Answer — LOW confidence** | it still gives its best diagnosis, clearly marked low confidence, and asks a doctor to confirm | 35% |
| 🧑‍⚕️ **Defer** | the photo and the details point to *different* diagnoses — a human must decide | 10% |

**9 in 10 patients get an answer and more than half a confident one.** HIGH-confidence
answers are **77% accurate on 11 diagnoses** and miss **3.5%** of cancers — a rate that
stays the same whichever fusion is used, while a plain confidence threshold ranges from
9.3% to 3.5%. A **team of five specialists**, weighted by a learned Devil's Advocate
(picked from seven rules on validation), raises photo-only macro-F1 from 0.286 to **0.332**.
The system **asks** for a dermoscopy image, missing details or a better photo when needed. Around it sits a care platform: patient app, doctor review queue,
instructions, questions, follow-ups with change detection, full audit log.

---

## 2. Which dataset? — **MILK10k. Only MILK10k.**

| | MILK10k | PAD-UFES-20 |
|---|---|---|
| Role in the **final** system | **the only dataset** — trained, calibrated, tested, demoed on it | **none** |
| Where it appears | everywhere | only in [reference/](reference/) as the history of the earlier prototype |
| Lesions / images | 5,240 / 10,480 | 2,298 images |
| Classes | **11** (multiclass) | 6 |
| Dermoscopy | **every lesion has one** | none |

Why the switch: the project's core question is *"is it worth asking for a dermoscopy
photo?"*. PAD-UFES-20 has no dermoscopy, so that question could only be simulated there.
MILK10k pairs every clinical photo with a real dermoscopy image, so it is **measured**.

The earlier prototype was built on PAD-UFES-20. Its code and data have been moved out of
the project to `../HealthSphere_archive/`. Any PAD-UFES number you see is history, not a
result of this system.

**Yes, it is multiclass**: 11 classes — AKIEC, BCC, BEN_OTH, BKL, DF, INF, MAL_OTH, MEL,
NV, SCCKA, VASC — of which 5 are malignant.

---

## 3. Old project → final project

| From the **old** system (kept, improved) | What it became |
|---|---|
| Multiple specialist models voting | one evidential head **per input**, combined by Dempster–Shafer |
| Hand-tuned dynamic fusion weights | weights set **per case by evidence** (`w = 1 − u`), shown in the app |
| Hand-typed "which input to trust per diagnosis" table | **measured** from data (e.g. dermoscopy 0.74 vs photo 0.36 for SCC) |
| Evidence manager (quality, reliability) | image-quality agent with thresholds fitted on MILK10k photos + ignorance `u` |
| Librarian / RAG | similar **confirmed** past cases with their photos (training lesions only) |
| Scribe / report + verification | report agent; optional LLM rewrite **verified claim by claim** |
| 4-level consensus ladder, fixed thresholds | calibrated gates: conformal set, Learn-then-Test, conflict, cost |
| React/Express/MongoDB workstation | Streamlit app (patient / doctor / system) + REST API + SQLite |

| **New** in the final system |
|---|
| Ask-for-dermoscopy decision, with a learned estimate of its value |
| Confidence tiers: answer with HIGH or LOW confidence instead of silently deferring |
| Statistical guarantee on missed cancers (Learn-then-Test) |
| Follow-ups with automatic change detection |
| Trained on MILK10k, every number regenerable from the repo |

---

## 3a. The specialist team (old design, now trained and measured)

| Specialist | Source |
|---|---|
| S1 clinical-photo head · S2 dermoscopy head · S3 patient-details head | evidential heads trained on MILK10k |
| S4 **PAD-UFES Model A** | the **original project's** image model, reused unchanged |
| S5 similar-case vote | the 15 most similar training lesions |

The **Devil's Advocate** profiles each one (quality Q, *learned* reliability R, uncertainty U,
consistency C, missingness M, out-of-distribution distance). Version 1 combines them with
the old **iterative diagnosis-adaptive fusion** (trust table now *measured*); version 2
**learns** how every signal should set the weights. Seven combination rules are compared
under cross-fitting and the best on validation is used (v2). Details:
[03_system.md](03_system.md) §4a, results: [05_results.md](05_results.md) §2–4.

## 4. The methodology (v3), in five steps

1. **Evidence per input.** Photo, dermoscopy and details each get an *evidential head*
   that outputs belief in each diagnosis **and** ignorance `u` (1 = knows nothing).
2. **Fusion.** Dempster–Shafer combination. A missing input changes nothing; a weak one has
   little influence; disagreement becomes a separate number, **conflict `C`**.
3. **Safety gates.** A conformal set over benign/malignant stops false reassurance; a
   Learn-then-Test threshold bounds missed cancers; high conflict sends the case to a human.
4. **Cheapest action.** Compare the expected cost of answering, asking for dermoscopy and
   deferring, with a missed cancer priced 5× a false alarm.
5. **Explain.** Quality check, similar cases, a verified report, and plain-English messages
   for the patient and the doctor.

Full explanation from zero: [02_learning_path.md](02_learning_path.md). Formulas:
[04_methods.md](04_methods.md).

---

## 5. Is it trained? Is anything left to download?

| | Status |
|---|---|
| MILK10k downloaded and audited | ✅ |
| Model trained (Kaggle, ResNet50, seed 42) | ✅ runs locally, matches Kaggle |
| Calibration, acquisition estimator, retrieval index | ✅ built locally from the trained model |
| App, API, demo | ✅ working |
| **Anything required to run the system?** | **No.** |

Two optional Kaggle runs would make the paper's numbers final and stronger — seeds 7 and
2026 (error bars), and the DermLIP image encoder (accuracy). The system works without
them. Details: [09_todo.md](09_todo.md).

---

## 6. Run it

```bash
python scripts/demo_care.py          # three patients end to end, in the terminal
streamlit run app/dguard_app.py      # the app
uvicorn src.api:app --port 8000      # the REST API (docs at /docs)
```
