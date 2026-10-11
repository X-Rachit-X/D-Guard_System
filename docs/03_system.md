# The System — Architecture, Agents, Platform

How the final D-GUARD fits together, end to end, with one real patient traced through it.

---

## 1. The whole system on one page

```
╔═════════════════════════════════════════════════════════════════════════════════════╗
║  ENTRY POINTS     app/dguard_app.py (Streamlit)  ·  src/api.py (REST)  ·  scripts/   ║
╚══════════════════════════════════════╦══════════════════════════════════════════════╝
                                       ▼
╔═════════════════════════════════════════════════════════════════════════════════════╗
║  CARE PLATFORM    src/care/service.py  +  src/care/store.py (SQLite, audit log)       ║
║  intake → assess → route (HIGH / ACQUIRE / LOW / DEFER) → doctor queue → follow-up    ║
╚══════════════════════════════════════╦══════════════════════════════════════════════╝
                                       ▼  assess(photo, dermoscopy, details)
╔═════════════════════════════════════════════════════════════════════════════════════╗
║  AGENT PIPELINE   src/agents/pipeline.py                                              ║
║   ┌───────────────┐  ┌────────────────────┐  ┌───────────────┐  ┌─────────────────┐  ║
║   │ Quality agent │─►│ DECISION ENGINE    │─►│ Librarian     │─►│ Report agent    │  ║
║   │ sharp? lit?   │  │ (the only one that │  │ similar       │  │ structured +    │  ║
║   │ glare?        │  │  decides)          │  │ confirmed     │  │ verified LLM    │  ║
║   └───────────────┘  └─────────┬──────────┘  │ training cases│  │ rewrite         │  ║
║                                │             └───────────────┘  └─────────────────┘  ║
╚════════════════════════════════╪════════════════════════════════════════════════════╝
                                 ▼
╔═════════════════════════════════════════════════════════════════════════════════════╗
║  DECISION ENGINE  src/native/engine.py                                                ║
║                                                                                       ║
║   photo ─► ResNet50 ─► head ─┐                                                        ║
║   dermo ─► ResNet50 ─► head ─┼─► Dempster–Shafer ─► belief b, ignorance u, conflict C ║
║   details ─► MLP ───► head ─┘     (missing input = vacuous = changes nothing)         ║
║                                    dynamic weights w_m = (1−u_m)/Σ                    ║
║                                            │                                          ║
║   src/decision/policy.py ◄─────────────────┘   + calibration bundle:                  ║
║     conflict > p90?            → DEFER          binary conformal set (Mondrian LAC)   ║
║     false reassurance? 1−u<λ̂?  → gate            LTT threshold λ̂                       ║
║     cheapest of answer / ask for dermoscopy / review                                  ║
║     → PROCEED HIGH | ACQUIRE | PROCEED LOW | DEFER   + tentative opinion              ║
║                                                                                       ║
║   src/decision/explain.py → patient wording, doctor wording                           ║
╚═════════════════════════════════════════════════════════════════════════════════════╝
```

**The rule that holds everywhere:** only the decision engine decides. Agents check, find,
and write. The platform routes. Only a doctor's recorded verdict overrides the engine.

---

## 2. Components

| Layer | File | Job |
|---|---|---|
| Model | `notebooks/kaggle_train_dguard.py` | training (Kaggle GPU); also defines the model the engine loads |
| | `runs/milk10k_resnet50_seed42/` | the trained run: metrics, predictions (weights not in git) |
| Data | `src/data/milk10k.py` | exact MILK10k loader, lesion-level splits |
| Evidence | `src/evidence/evidential.py` | reference implementation of opinions and DS fusion |
| Calibration | `src/calibration/binary.py`, `ltt.py`, `conformal.py` | benign/malignant sets, Learn-then-Test |
| Decision | `src/decision/policy.py`, `explain.py` | tiers, gates, cost; plain-English messages |
| Engine | `src/native/engine.py` | serves the trained model one case at a time |
| Agents | `src/agents/quality.py`, `librarian.py`, `report.py`, `pipeline.py` | quality, retrieval, report, orchestration |
| Retrieval | `src/retrieval/rag_index.py` | FAISS cosine index with spec checking |
| Platform | `src/care/service.py`, `store.py` | workflow, database, audit |
| Interfaces | `app/dguard_app.py`, `src/api.py` | web app, REST API |
| Evaluation | `scripts/run_eade_milk.py`, `run_stress_milk.py`, `src/eval/selective.py` | experiments, metrics |

---

## 3. The four outcomes and what happens next

| Outcome | When | Patient | Case goes to |
|---|---|---|---|
| ✅ **PROCEED — HIGH** | every gate passed, answering is cheapest | result + next steps + follow-up date | doctor *sign-off* queue (routine) |
| 📷 **ACQUIRE** | blocked, but dermoscopy is expected to pay off | "please add a dermoscopy photo" + how | waits for the patient, then re-assessed (once) |
| ⚠️ **PROCEED — LOW** | blocked or deferral cheaper, no acquisition left | best diagnosis, **marked low confidence**, "a doctor will confirm", urgency | doctor *confirm* queue |
| 🧑‍⚕️ **DEFER** | sources contradict (conflict) or no evidence | tentative opinion + "see a doctor soon" | doctor *review* queue (first) |

Queue order: review → confirm → sign-off; within each, URGENT first, then oldest.
**URGENT** = P(malignant) ≥ 0.30 or a malignant class still in the conformal set.

On the test set: HIGH 60.2% · LOW 29.4% · DEFER 10.4% · dermoscopy requested 27.3%.

---

## 4. The agents

### Quality agent — `src/agents/quality.py`
Measures sharpness (variance of the Laplacian), brightness, contrast, glare and
resolution. **Thresholds are the 5th/95th percentiles of the same measurements on 300 real
MILK10k training images** — separately for clinical photos and dermoscopy images — so
"blurry" means "blurrier than 95% of what the model learned from". It gives the patient a
concrete tip ("tilt the camera to avoid reflections") and never changes a probability; the
model's own ignorance `u` is what down-weights a poor photo.

### Librarian — `src/agents/librarian.py`
Finds the 6 most similar **training** lesions (3,667 indexed) by cosine similarity of the
trained model's own 256-d photo and metadata embeddings, and returns them with their
histopathology-backed diagnoses and photos. The doctor sees a gallery. The index refuses to
load unless it was built from the training split, and the build asserts no validation or
test lesion is inside — a held-out lesion would leak its true label into what a doctor
reads. Retrieval never changes the decision.

### Report agent — `src/agents/report.py`
Always writes a structured report: assessment and tier, differential, P(malignant),
ignorance, conflict, weights, source disagreement, quality issues, similar cases,
confirmation requirement. If a local LLM is running (Ollama), it asks for a readable
rewrite and **verifies** it: every diagnosis named must be in the evidence, every
percentage must match an evidence value within 1 point, and the confidence tier and
"confirm" sentence must survive. Any failure → the structured report is used.

### Pipeline — `src/agents/pipeline.py`
`quality → engine → librarian → report`. Same `assess()` signature as the engine, so the
platform does not care which it is given. Missing pieces degrade gracefully: no index → no
gallery; no LLM → structured report.

---

## 4a. The specialist team, Devil's Advocate and adaptive fusion

Code: `src/specialists/`, `src/fusion/adaptive.py`, `src/fusion/runtime.py`.
Fitted by `scripts/run_team.py` on validation; used per case by the engine.

**Specialists** — S1–S3 evidential heads (MILK10k), **S4 the original PAD-UFES Model A**
(ResNet50 + channel attention + MHSA ViT; its 6 classes mapped onto MILK10k), S5 a vote of
the 15 most similar training lesions.

**Devil's Advocate** (per specialist, per case):

| Signal | How it is computed |
|---|---|
| Quality Q | share of sharpness / brightness / contrast / glare inside the 5–95% range of training photos |
| Reliability R | **logistic regression fitted on validation**: P(this specialist is right) from Q, U, confidence, agreement |
| Uncertainty U | normalised entropy |
| Consistency C | 1 − mean pairwise Jensen–Shannon divergence |
| Missingness M | share of specialists absent |
| OOD | Mahalanobis distance to the nearest class (Ledoit–Wolf), flagged above validation p99 |

**Iterative diagnosis-adaptive fusion** (the old EM algorithm):

```
w(0) = softmax(a·log(Q·R))                         start from evidence quality × reliability
p = Σ w·p_m ;  y = argmax p                        working diagnosis
repeat ≤ 3:
   π_m = measured P(specialist m right | diagnosis y)      E-step (measured, not hand-typed)
   κ_m = 1 if m agrees with y else ρ                        conflict penalty
   w = softmax(a·log(Q·R) + b·log π + g·log κ)              M-step
   p = Σ w·p_m
until y is stable and weights move < 0.05
```

a, b, g, ρ and a temperature are fitted on validation. The result feeds the decision
policy; Dempster–Shafer ignorance and conflict still drive the gates. The app shows the
**calibrated** confidence (probability the stated diagnosis is right) and the **consensus
level** of the original four-level hierarchy (1 strong · 2 moderate · 3 uncertain ·
4 critical conflict).

## 5. One real patient, end to end

Held-out MILK10k lesion **IL_0449677** (true diagnosis: NV, a benign mole), traced through
the full agent pipeline (`DGuardPipeline.assess_paths`):

```
STAGE A — phone photo + details
  quality agent   clinical photo: too bright, low contrast → tip sent to patient
  each source     photo says NV · details say BCC          ← they disagree
  engine          ignorance u 0.836 · conflict C 0.007 (disagreement, but weak evidence)
                  weights: photo 0.59, details 0.41
                  P(malignant) 43% → cancer not ruled out → gate fires
                  acquisition model: dermoscopy is worth more than it costs
  OUTCOME         📷 ACQUIRE — "please provide a dermoscopy photo"
                  early, unconfirmed impression: NV

STAGE B — + the real dermoscopy image
  engine          ignorance u 0.463 · P(malignant) 24%
                  weights: photo 0.16, DERMOSCOPY 0.73, details 0.11   ← trust moved on its own
                  dermoscopy says NV · every statistical gate passes
  quality gate    the clinical photo has a SEVERE issue (low contrast, outside the
                  1st-99th percentile of training photos) → HIGH is not allowed
  OUTCOME         ⚠️ PROCEED (LOW) — Melanocytic Nevus (NV), doctor confirms   ← correct
  librarian       6 similar confirmed cases shown to the doctor
  platform        answer shown to the patient as low confidence + photo tip;
                  doctor confirm queue
```

That is the whole project in one case: weak evidence is recognised, the right extra
evidence is requested, trust shifts to it automatically, and the answer is right — yet
because the original photo was poor, the system will not call it HIGH confidence and
asks a doctor to confirm. (`scripts/demo_care.py` shows a case that ends HIGH.)

---

## 6. Care platform — `src/care/`

**Tables** (SQLite, `data/care/dguard_care.sqlite`): patients, cases, evidence,
assessments (with the policy that produced them), reviews, messages, follow-ups, audit.

**Case states:** `NEW → AWAITING_PATIENT (ACQUIRE) | AWAITING_DOCTOR (LOW / DEFER) |
ASSESSED (HIGH) → FOLLOW_UP → CLOSED`.

**Doctor actions:** CONFIRM (defaults to the system's diagnosis), OVERRIDE (must state a
diagnosis), NEED_IN_PERSON; plus a note, an instruction to the patient, and a follow-up.

**Follow-up change check:** a new photo is re-assessed and compared with the last
assessment. P(malignant) up by ≥ 0.15, or a benign call turning malignant, sends the case
straight back to the doctor as URGENT.

**Patient conversation:** questions go to the doctor's queue (shown as unanswered until the
doctor replies).

---

## 7. Interfaces

### Web app
```bash
streamlit run app/dguard_app.py
```
**Patient**: register, open a case, use a real held-out demo lesion or upload a photo,
see the outcome and messages, add dermoscopy when asked, ask questions, follow-ups.
**Doctor**: prioritised queue, images, report, evidence panel (u, C, weights, votes,
differential, reliability), similar-case gallery, confirm/override form, replies, timeline.
**System**: results tables, measured reliability, platform activity, audit log.

### REST API
```bash
uvicorn src.api:app --port 8000        # interactive docs: http://localhost:8000/docs
```

| Method | Path | Does |
|---|---|---|
| POST | `/patients` | register |
| POST | `/patients/{id}/cases` | open a case |
| POST | `/cases/{id}/photo`, `/dermoscopy` | upload images |
| POST | `/cases/{id}/details` | age, sex, site, skin tone |
| POST | `/cases/{id}/assess` | run the pipeline → outcome, tier, messages, report, similar cases |
| POST | `/cases/{id}/followup` | new photo + change check |
| POST | `/cases/{id}/questions` | patient question |
| GET | `/cases/{id}`, `/cases/{id}/timeline` | case view, full history |
| GET | `/doctor/queue` | prioritised queue |
| POST | `/cases/{id}/review`, `/reply` | doctor verdict, reply |
| GET | `/stats` | platform totals |

This is the hook for the planned upgrade — a patient chat assistant or mobile app talking
to the same workflow.

---

## 8. What runs where

| Needs | Runs on |
|---|---|
| Training | Kaggle GPU (`notebooks/kaggle_train_dguard.py`) |
| Everything else — inference, calibration, experiments, app, API | a laptop CPU |
| Optional LLM prose | local Ollama (`ollama pull qwen2.5:7b`) |
