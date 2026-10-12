# System Architecture and Design

This chapter describes DERMA-GUARD as a whole: its components, how a single case flows through them, and how the care platform around the model is organised. The methods inside each component are explained in Chapter 6.

## Design Principles

Five principles guided every design decision.

- **Only one component decides.** The neural specialists, the Devil's Advocate, the agents and the language model all provide evidence or text; the outcome of a case is chosen by a single, deterministic, versioned decision policy. This makes every decision reproducible and auditable.
- **Every number is fitted on validation data and measured on test data.** Thresholds, weights, calibration maps and the choice between methods are never set by hand and never tuned on the test set.
- **Uncertainty is a first-class output.** Each specialist reports how much it does not know; the policy treats "not enough evidence" (ask for more) differently from "contradictory evidence" (ask a human).
- **Fail safe.** Any doubt about the input (poor photograph, missing details, unusual case) can only make the system more careful, never more confident.
- **The patient always gets something useful.** Even a deferred case shows the tentative opinion, a short differential, the probability of malignancy and an urgency flag.

## Overall Architecture

![Overall architecture of DERMA-GUARD. Six specialists are profiled by the Devil's Advocate and combined by the fusion rule chosen on validation; the gatekeeper and cost matrix choose one of four outcomes; the dashed loop is the request for more evidence.](paper/figures/fig_architecture.png)

The system has four layers.

1. **Evidence layer.** A clinical photograph, an optional dermoscopic image, patient details and an optional free-text description of symptoms. The quality agent measures each image, and the scribe reads the text for warning signs.
2. **Specialist layer.** Six specialists each produce a probability distribution over the eleven diagnoses (Section 6.1).
3. **Evidence management and fusion layer.** The Devil's Advocate profiles every specialist; a learned gate converts the profile into per-case weights and fuses the opinions; in parallel, the three evidential heads are combined by Dempster's rule to obtain ignorance and conflict (Sections 6.2 to 6.4).
4. **Decision and care layer.** The gatekeeper applies the safety checks and the cost matrix and chooses an outcome; the explainer, librarian and report agent write the explanation; the care platform routes the case to the patient, a doctor, or a follow-up (Sections 6.5, 6.6 and 5.5).

Table: Main components and where they are implemented.
| Component | Responsibility | Source file |
|---|---|---|
| Data loader | Exact MILK10k loader and lesion-level split | src/data/milk10k.py |
| Evidential network | Three heads (photo, dermoscopy, details) and Dempster-Shafer fusion | notebooks/kaggle_train_dguard.py, src/native/engine.py |
| Legacy specialist | The earlier prototype's Model A | src/specialists/ |
| Case-based specialists | Similar-case vote and embedding classifier | src/fusion/adaptive.py |
| Devil's Advocate v1 | Quality, learned reliability, uncertainty, consistency, missingness, OOD | src/fusion/adaptive.py |
| Devil's Advocate v2 | Learned evidence gate, cross-fitting | src/fusion/gating.py |
| Fusion rules | Seven combination rules behind one interface | src/fusion/methods.py |
| Single-case runtime | Team fusion for one patient; comparison of all rules | src/fusion/runtime.py |
| Calibration | Conformal sets, Learn-then-Test | src/calibration/ |
| Decision policy | Four outcomes, tiers, gates, cost matrix | src/decision/policy.py |
| Agents | Quality, scribe, librarian, explainer, report writer | src/agents/ |
| Care platform | Workflow, storage, audit log | src/care/ |
| Interfaces | Web application, REST API | app/dguard_app.py, src/api.py |

## The Life of One Case

The clearest way to understand the architecture is to follow one patient through it.

1. **Intake.** The patient registers, opens a case for one spot, uploads a photograph and fills in age, sex and body site. Optionally they describe the spot in their own words, for example "it has started bleeding and got bigger over three weeks".
2. **Quality check.** The quality agent measures sharpness, brightness, contrast, glare, graininess and whether the centre of the photograph is covered, and compares each measurement with the range seen in 300 training photographs.
3. **Specialists.** The evidential network computes an opinion for the photograph and for the details; the legacy Model A, the similar-case vote and the embedding classifier each produce a distribution.
4. **Devil's Advocate and fusion.** Each specialist is profiled; the learned gate assigns weights; the fused distribution is temperature-scaled and calibrated.
5. **Gatekeeper.** The policy checks conflict, the benign/malignant conformal set, the Learn-then-Test threshold, quality, out-of-distribution distance and missing details, and compares the expected cost of answering, asking for dermoscopy and deferring. Suppose the photograph alone is not enough: the outcome is **ACQUIRE dermoscopy**.
6. **Request.** The case moves to "awaiting patient" and the patient receives instructions on how to obtain a dermoscopic image. When they upload it, the whole assessment runs again, now with the dermoscopy specialist included.
7. **Answer.** The second assessment passes every check: **HIGH-confidence answer**, for example basal cell carcinoma. Because it is malignant, the urgency is set to URGENT and a follow-up is scheduled to check that a dermatologist was seen. The scribe found "bleeding" and "growth" in the patient's text, which would also have raised urgency on its own.
8. **Explanation and report.** The explainer writes who said what and why; the librarian attaches the most similar confirmed training lesions; the report agent writes a structured report, optionally polished by a local language model and accepted only if a verifier finds no invented diagnosis or number.
9. **Doctor.** The case appears in the doctor's queue for sign-off. The doctor sees every piece of evidence, confirms or overrides, writes an instruction to the patient and can schedule a follow-up.
10. **Follow-up.** On the follow-up date the patient uploads a new photograph; if the probability of malignancy has risen by 0.15 or more, or the diagnosis has become malignant, the case returns to the doctor as urgent.

## The Four Outcomes

Table: The four outcomes, when they occur and what happens next.
| Outcome | When | What the patient sees | Next step in the platform |
|---|---|---|---|
| HIGH answer | All safety checks pass and answering is cheapest | The diagnosis, calibrated confidence, urgency, advice | Assessed; automatic follow-up |
| LOW answer | A check blocks a confident answer but the evidence does not contradict itself | The best diagnosis, clearly marked low confidence, top-3 differential | Doctor's queue ("confirm") |
| ACQUIRE | Dermoscopy, missing details or a better photo would likely change the decision | What to provide and how | Awaiting patient; reassessed on upload |
| DEFER | The sources confidently contradict each other, or there is no usable evidence | The tentative opinion and an urgency flag | Doctor's queue ("review") |

## Care Platform Design

The care platform turns individual assessments into a workflow. It is a thin service layer (src/care/service.py) over a SQLite database (src/care/store.py). Every write also creates an audit record, so the history of each case, including which policy version made each decision, can be reconstructed.

Table: Database tables of the care platform.
| Table | Purpose | Key fields |
|---|---|---|
| patients | Registered patients | name, age, sex, skin tone |
| cases | One case per lesion | patient, site, complaint, status, urgency |
| evidence | Photos, dermoscopy and details submitted | kind, file path, payload |
| assessments | Every run of the decision system | stage, action, tentative diagnosis, P(malignant), full result, policy version |
| reviews | Doctors' verdicts | doctor, verdict (confirm / override / in person), diagnosis, note |
| messages | Conversation and system instructions | sender, kind (result, instruction, question, reply), body |
| followups | Scheduled re-checks | due date, reason, status |
| audit | Append-only event log | actor, event, detail |

A case moves through a small state machine: OPEN when created; AWAITING_PATIENT when the system asks for evidence; AWAITING_DOCTOR after a deferral, a LOW answer or a worrying change; ASSESSED after a HIGH answer; FOLLOW_UP when a doctor schedules a re-check; and CLOSED after a final review. The doctor's queue orders cases by kind (review before confirm before sign-off), then urgency, then waiting time.

Personal data can be erased. Deleting a case or a patient removes every related record and the photographs uploaded through the platform (never the dataset images used by the demonstration); the audit log keeps only the fact that a deletion took place.

## Interfaces

The platform is exposed in two ways. The **web application** (Streamlit) has seven views: Patient, Doctor, Compare systems, Stress Lab, Research, Architecture and System, described with screenshots in Section 7.6 and Appendix II. The **REST API** (FastAPI) exposes the same workflow for integration with other software.

Table: REST API endpoints.
| Method | Endpoint | Purpose |
|---|---|---|
| GET | /health | Service status and model version |
| POST | /patients | Register a patient |
| POST | /patients/{pid}/cases | Open a case |
| POST | /cases/{cid}/photo | Upload a clinical photograph |
| POST | /cases/{cid}/dermoscopy | Upload a dermoscopic image |
| POST | /cases/{cid}/details | Submit patient details and symptoms |
| POST | /cases/{cid}/assess | Run the decision system and route the case |
| POST | /cases/{cid}/followup | Submit a follow-up photograph (change check) |
| POST | /cases/{cid}/questions | Patient asks a question |
| GET | /cases/{cid} | Case status, latest assessment and messages |
| GET | /cases/{cid}/timeline | Full event history |
| GET | /doctor/queue | Cases waiting for a clinician |
| POST | /cases/{cid}/review | Doctor's verdict |
| POST | /cases/{cid}/reply | Doctor replies to the patient |
| DELETE | /cases/{cid}/messages | Clear a case's conversation |
| DELETE | /cases/{cid} | Delete a case and its uploads |
| DELETE | /patients/{pid} | Delete a patient and all their data |
| GET | /stats | Platform statistics |
