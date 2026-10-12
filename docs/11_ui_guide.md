# How the app works — a guided tour

```bash
streamlit run app/dguard_app.py        # opens http://localhost:8501
```

Links open a view directly: `?view=research`, `?view=compare&run=1` (runs the comparison
on load), `?view=stress&damage=blur&run=1`.

The first load takes ~20 s (it loads six specialists and the calibration). After that,
one assessment takes about a second. Everything you see is computed live from the real
trained models — there are no canned answers.

The left sidebar switches between **seven views**:

| View | Who it is for | What you can do |
|---|---|---|
| **Patient** | the person with the spot | register, open a case, send photos/details, read the result, answer requests, message the doctor, delete your data |
| **Doctor** | the dermatologist | work the queue, see every piece of evidence and *why*, reply, confirm/override |
| **Compare systems** | examiners, researchers | run one lesion through **every** system; see the test-set comparison tables; what every metric means |
| **Stress Lab** | anyone | damage a photo (blur, dark, noise, JPEG, cover, remove details) and watch the system react |
| **Research** | examiners | every experiment result with confidence intervals |
| **Architecture** | anyone | the pipeline as a diagram, and which file implements each part |
| **System** | admin | activity counts, audit log, reset the demo database |

---

## 1. Patient view — a full case in 2 minutes

1. **Register a patient** (name, age, sex, skin tone).
2. **Open a case** — one case = one spot.
3. **Evidence.** Choose *Real held-out MILK10k lesion (demo)* — these are test lesions the
   model never trained on; the true diagnosis is shown to *you* but hidden from the system.
   Two switches let you see the system **ask** for evidence:
   - *Include the dermoscopy photo from the start* — leave **off**. If the photo alone is
     not enough, the result is **📷 ACQUIRE — dermoscopy requested**, and a button appears
     to send it. The case is then assessed again with both images.
   - *Include patient details* — turn **off**. With only a photo the system answers
     **📝 Details requested** and shows a small form (age, sex, body area). Send it and the
     spot is re-assessed straight away.
4. **Result.** A coloured badge shows the outcome:

   | Badge | Meaning | What happens next |
   |---|---|---|
   | ✅ PROCEED (HIGH) | every safety check passed | answer + follow-up date |
   | ⚠️ PROCEED LOW | best answer, clearly marked unreliable | a doctor must confirm |
   | 📷 / 📝 ACQUIRE | more evidence would change the decision | you send it, it re-assesses |
   | 🧑‍⚕️ DEFER | the evidence contradicts itself | a doctor decides |

   Under it: the plain-language message, the **calibrated confidence** bar, any warning
   signs found in your own description, and **🔍 Why this result** (see §4).
5. If the quality agent judged the photo poor (blurry, dark, grainy, covered…), a
   **Retake the photo** box appears.
6. **Messages.** A real conversation thread:
   - filter: *All* / *Results & next steps* / *Questions & replies*;
   - coloured tags show what each message is (result, next step, question, doctor's reply);
   - one-click questions (*What does my result mean?*, *How urgent is this?*, …) or type
     your own; a counter shows questions still waiting for a doctor;
   - **🗑️ Clear conversation** deletes the messages of this case (the assessment stays).
7. **🔒 My data — delete.** Delete one case, or the patient and *all* their cases. Uploaded
   photos are removed from disk; dataset images are never touched. The audit log keeps only
   the fact that a deletion happened, not the content.

## 2. Doctor view

The **queue** lists every case that needs a human, most urgent first:
*review* (deferred), *confirm* (LOW-confidence answer), *sign-off* (system answered).
Open a case to get six tabs:

| Tab | Shows |
|---|---|
| Assessment | images, the doctor-facing summary, calibrated confidence, P(malignant), ignorance, conflict, differential, image-quality warnings, the patient's own words |
| **Why — explanation** | who said what, why some specialists disagree, where the weight went (§4) |
| Specialist team & gates | the Devil's Advocate table (Q, R, U, track record, weight), consensus level, and the **Gatekeeper**: every safety check with ✅ passed / ⛔ fired |
| Similar cases & report | the 5 most similar *training* lesions with their confirmed diagnosis; the written report (LLM-polished only if the verifier approves) |
| Messages | the same thread, with quick-reply templates |
| Timeline | every event on the case, and the assessment history |

Then **Your decision**: CONFIRM / OVERRIDE (with your diagnosis) / NEED IN PERSON, a
note, an instruction to the patient and an optional follow-up.

## 3. Compare systems — use every system

**One lesion, every system.** Pick a held-out lesion and the evidence (photo + details, or
+ dermoscopy). *Run every system* gives one table:

| Group | Systems |
|---|---|
| Specialists alone | photo head, dermoscopy head, details head, PAD-UFES Model A, similar-case vote, embedding classifier |
| Fusion rules | simple average, original formula (Q·R·U), best head per case, adaptive EM (v1), stacking, Devil's Advocate v2 learned gate (mixture / product) |
| Decision systems | always answer, B3 confidence threshold, B4 learned rejector, **D-GUARD** |

For each: what it says, ✅/❌ against the hidden truth, its confidence, the probability it
gave the *true* diagnosis (bar chart), whether it answers or abstains, and — for fusion
rules — the weight it gave each specialist.

**Test-set comparison.** The same systems over all 787 test lesions: decision systems
(coverage, accuracy, missed cancers — pick what to plot), fusion rules (validation
out-of-fold score used to pick the winner, and test scores), the oracle ceiling, the
**subset study** (every combination of specialists), and each specialist alone.

**Evaluation methods explained.** One table: every metric and check used in the project,
the question it answers, and where to find it.

## 4. "Why this result" — the explanation

Built by `src/agents/explainer.py` from numbers already in the result:

1. **Who said what** — each specialist's own top diagnosis and how sure it was, and what it
   looks at (the details head never sees the spot; Model A was trained on another dataset).
2. **Why some disagree** — for every dissenting specialist, the *measured* reasons:
   its track record for this diagnosis on validation lesions, how spread out its opinion is,
   input quality, its learned reliability, and whether it is actually strongly against.
3. **Where the weight went** — the three biggest weights and what earned them.
4. **What the confidence means** — "of past validation cases given this confidence, about
   N% were right".

If a local LLM (Ollama) is running, it may rewrite the text for readability, but the
rewrite is **only shown if a verifier finds no diagnosis or percentage that is not in the
facts**; otherwise you see the structured version and the source line says so.

## 5. Stress Lab

Pick a lesion and a damage. *Run both* assesses the original and the damaged version side
by side through the **full pipeline** (all agents and gates): badge, tentative diagnosis,
confidence, ignorance, conflict, P(malignant), quality-agent findings, fusion weights and
the gatekeeper. Things to try:

- **blur** — ignorance *drops* (the model is fooled), but the quality gate blocks HIGH;
- **occlusion** — the *covered-centre* check fires;
- **noise** — the *grainy* check fires;
- **remove patient details** — the system asks for them.

## 6. Research, Architecture, System

- **Research** — headline numbers with 95% CIs, paired tests, fusion tables and what the
  learned gate listens to, decision vs baselines at three miss costs, DERMA-STRESS, PAD-UFES
  shift, Derm7pt external test, skin-tone breakdown.
- **Architecture** — the pipeline diagram and a component → file table.
- **System** — counts, the audit log, and **Reset the demo database** (asks for
  confirmation; deletes every patient, case, message and uploaded photo).

---

**Tip — the database is locked?** Another Streamlit instance has it open. Stop it, or run
with a separate database: `DGUARD_DB=data/care/demo2.sqlite streamlit run app/dguard_app.py`.
