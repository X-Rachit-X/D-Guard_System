# D-GUARD

**Risk-Controlled, Cost-Aware Evidence Acquisition and Deferral for Multimodal Dermatology AI**

> ⚠️ **Research prototype. Not clinically validated. Do not use for medical diagnosis.**

A skin-lesion AI that decides, for each patient, whether to **answer**, **ask for a
dermoscopy photo**, or **hand the case to a doctor** — and a care platform around it where
patients upload photos, doctors review a prioritised queue, and follow-ups are checked for
change.

| Action | When | What the patient gets |
|---|---|---|
| ✅ **PROCEED** | confident, all safety gates pass | an assessment, instructions, a follow-up date |
| 📷 **ACQUIRE** | a dermoscopy photo is expected to change the answer | a request for that photo, then a new assessment |
| 🧑‍⚕️ **DEFER** | conflict, too little evidence, or cancer cannot be ruled out | a **tentative** opinion, a strong "see a doctor" instruction, and a place in the doctor's queue |

---

## Result (MILK10k, trained model, 787 held-out lesions)

| Policy | answered alone | dermoscopy taken | accuracy | **missed cancers** |
|---|---|---|---|---|
| Always answer from the photo | 100% | 0% | 64.4% | 5.2% |
| Confidence-based abstention, same coverage | 36.8% | 0% | 84.1% | 5.2% |
| **D-GUARD** | **36.8%** | **6.4%** | **83.8%** | **1.7%** |

**Same coverage, same accuracy, a third of the missed cancers** — at lower mean cost than
the abstention baseline, asking for dermoscopy for only 6% of patients.

Also measured: dermoscopy raises malignant-vs-benign AUROC from **0.80 to 0.86** and cuts
the model's ignorance from **0.68 to 0.44**; the acquisition estimator captures far more of
dermoscopy's benefit than asking the most uncertain cases (AE **+0.27** vs **+0.01**); the
Learn-then-Test risk threshold is **certified** at both stages.

**Reported against us:** it answers alone for only ~37% of patients; on *mean* cost,
always answering is cheaper (deferral is priced at 1.5); using all three sources scores
below the best pair (0.379 vs 0.403 macro-F1); one training seed so far. Full results and
every caveat: **[docs/14_results.md](docs/14_results.md)**.

---

## How it works

```
 photo ──────┐   ┌────────────────┐   ┌──────────────┐   ┌─────────────┐   ┌───────────┐
 dermoscopy ─┼──►│ one evidential │──►│ Dempster–    │──►│ safety      │──►│ PROCEED   │
 details ────┘   │ head per source│   │ Shafer fusion│   │ gates       │   │ ACQUIRE   │
                 │ belief +       │   │ belief,      │   │ conformal   │   │ DEFER     │
                 │ ignorance u    │   │ ignorance,   │   │ LTT, conflict│  │ +tentative│
                 └────────────────┘   │ CONFLICT     │   │ cost matrix │   │  opinion  │
                                      └──────────────┘   └─────────────┘   └─────┬─────┘
      agents: specialists · evidence auditor · librarian (similar cases) · report + verifier
      platform: patient app · doctor queue · instructions · questions · follow-ups · audit
```

1. **Each source says how much it knows.** An evidential head outputs *belief* and
   *ignorance* `u` (1 = no evidence). Softmax cannot say "I don't know"; this can.
2. **Dynamic fusion.** Dempster–Shafer combination: a missing source is total ignorance
   and changes nothing; a weak source has little evidence and little weight — per case,
   automatically. The app shows each case's weights.
3. **Conflict ≠ ignorance.** Sources confidently disagreeing (→ a human decides) is a
   different number from nobody knowing (→ get more evidence).
4. **Calibrated gates.** A conformal set over benign/malignant blocks false reassurance; a
   Learn-then-Test threshold carries a statistical bound on missed cancers.
5. **Cheapest action.** Compare the expected cost of answering, acquiring and deferring.

Plain-English explanation of every concept: **[docs/11_learning_path.md](docs/11_learning_path.md)**.

---

## Quickstart

```bash
pip install -e ".[dev]"
# Windows/Anaconda: set KMP_DUPLICATE_LIB_OK=TRUE first

pytest tests -q                          # 470 passed, 3 skipped

# the trained model, end to end (needs ./milk10k and runs/*/checkpoints/best.pt)
python scripts/milk_native_infer.py      # verify it matches the Kaggle run
python scripts/run_eade_milk.py --missed-cost 5 --out outputs/eade_milk/cost_5
python scripts/demo_care.py              # three patients: PROCEED, ACQUIRE, DEFER

streamlit run app/dguard_app.py          # the app: patient, doctor, system views
```

Training on a GPU: **[docs/06_training_guide.md](docs/06_training_guide.md)** with
[`notebooks/kaggle_train_dguard.py`](notebooks/kaggle_train_dguard.py).

---

## Status

| | |
|---|---|
| MILK10k evidential model, trained (Kaggle, seed 42) and verified locally | ✅ |
| Two-stage decision engine + 6 baselines + cost sweep | ✅ |
| Tentative opinion on every DEFER/ACQUIRE | ✅ |
| Care platform: patients, doctor queue, questions, follow-ups, change check, audit | ✅ |
| Web app + terminal demo | ✅ |
| RAG similar-case retrieval (explanation only, leak-guarded) and 7-agent pipeline | ✅ |
| DERMA-STRESS robustness benchmark | ✅ |
| Seeds 7 and 2026, DermLIP encoder | ⬜ [docs/09](docs/09_what_you_need_to_do.md) |
| Metrics from the original checkpoints | ❌ invalid — [docs/evaluation_integrity.md](docs/evaluation_integrity.md) |

---

## Documentation

Start at **[docs/00_index.md](docs/00_index.md)**.

| For | Read |
|---|---|
| Learning the project | [15_study_plan.md](docs/15_study_plan.md) → [11_learning_path.md](docs/11_learning_path.md) |
| One case end to end | [12_system_walkthrough.md](docs/12_system_walkthrough.md) |
| The numbers | [14_results.md](docs/14_results.md) |
| Presenting it | [13_panel_guide.md](docs/13_panel_guide.md) |
| What is left | [09_what_you_need_to_do.md](docs/09_what_you_need_to_do.md) |
| Folder layout | [16_project_structure.md](docs/16_project_structure.md) |
| What failed | [what_did_not_work.md](docs/what_did_not_work.md) |
| The paper | [paper/main.tex](paper/main.tex) |

---

## Datasets

| | Role | |
|---|---|---|
| **MILK10k** (ISIC, `10.34970/648456`) | primary | 5,240 lesions, each with a clinical photo **and** a dermoscopy image; 11 classes; 71.7% malignant; 95.7% histopathology; CC-BY-NC |
| **PAD-UFES-20** (`10.1016/j.dib.2020.106221`) | shift set, legacy pipeline | 2,298 smartphone images, no dermoscopy; CC BY 4.0 |

No patient data is committed to this repository.

---

## Honest positioning

**Not novel on their own:** evidential deep learning, Dempster–Shafer combination,
conformal prediction, Learn-then-Test, learning to defer, recommending dermoscopy.

**Our contribution:** their combination into one calibrated three-action policy; measuring
the value of acquiring dermoscopy on real paired data; the stress benchmark; and a working
care platform around it.

**We do not claim:** "first", "state of the art", "clinically safer", or "guaranteed safe".
The guarantee is statistical, assumes test patients resemble calibration patients, and does
not hold under distribution shift.
