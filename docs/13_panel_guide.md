# Panel Guide — Presenting and Defending D-GUARD

How to present the project in 15 minutes, run the live demo, and answer the questions a
panel will ask. Plain English. Every claim below is backed by a file you can open in front
of the panel.

---

## 1. The one sentence

> **D-GUARD is a skin-lesion AI that decides, for each patient, whether to answer, ask for
> a dermoscopy photo, or hand the case to a doctor — and when it does answer, it misses
> three times fewer cancers than the standard approach at the same coverage.**

Learn it word for word. Everything else in the talk supports it.

---

## 2. The story in five beats

| # | Beat | Say | Show |
|---|---|---|---|
| 1 | **Problem** | "AI gives a confident answer even when it should not — a blurry photo, missing details, sources that disagree. Softmax always picks something." | One slide: same "0.82 BCC" for a clear and a blurry photo |
| 2 | **Idea** | "Let each source say how much it actually knows, combine them so ignorant sources fall silent, and let the system choose between three actions." | The architecture diagram (§4) |
| 3 | **Guarantee** | "Thresholds are not hand-picked. They are calibrated with statistical tests that come with a written error bound." | One slide: PROCEED / ACQUIRE / DEFER with the gates |
| 4 | **Evidence** | "Trained on MILK10k. At the same coverage it misses 1.7% of cancers vs 5.2%, and asks for dermoscopy for only 6% of patients." | Results table (§5) |
| 5 | **Demo** | "Here is a real held-out patient." | The app (§6) |

Close with **honest limits** (§7). Panels trust a project more when you state its limits
before they find them.

---

## 3. Slide plan (12 slides, ~15 minutes)

| # | Slide | Content | Time |
|---|---|---|---|
| 1 | Title | D-GUARD · Risk-Controlled, Cost-Aware Evidence Acquisition and Deferral for Multimodal Dermatology AI · team names | 0:20 |
| 2 | The problem | Confident wrong answers; softmax cannot say "I don't know"; missed melanoma is the costly error | 1:00 |
| 3 | Why more data is not enough | PAD-UFES-20: adding text made the model **worse** (0.653 → 0.633). Naive fusion cannot ignore a weak source | 1:00 |
| 4 | Our question | Answer, ask for dermoscopy, or defer — chosen per patient, with a guarantee | 0:40 |
| 5 | Architecture | The diagram in §4 | 1:30 |
| 6 | Evidence, not probability | Evidential heads: belief + ignorance `u`. Missing source = `u = 1`, changes nothing | 1:20 |
| 7 | Conflict ≠ ignorance | Two sources confidently disagree → defer. Nobody knows → acquire. JS divergence cannot tell these apart; Dempster–Shafer can | 1:20 |
| 8 | Safety gates | Conformal set over benign/malignant (no false reassurance) · Learn-then-Test threshold · conflict threshold · cost matrix | 1:20 |
| 9 | Results | Table §5 + "3× fewer missed cancers, dermoscopy for 6%" | 1:30 |
| 10 | Live demo | §6 | 3:00 |
| 11 | What did not work | 3 honest points from §7 | 1:00 |
| 12 | Future work | Better encoder, more seeds, prospective study, patient app | 0:40 |

**Tip:** put the integrity audit in the backup slides, not the main talk. Bring it up only
if asked about earlier numbers (§8, Q12).

---

## 4. The architecture slide

Draw it as five boxes left to right. Keep the words this short:

```
 ┌──────────────┐  ┌───────────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐
 │  1. INPUTS   │─►│ 2. EVIDENCE   │─►│ 3. FUSION    │─►│ 4. GATES    │─►│ 5. ACTION    │
 │ photo        │  │ one head per  │  │ Dempster–    │  │ conformal   │  │ PROCEED      │
 │ dermoscopy   │  │ source:       │  │ Shafer:      │  │ set, LTT    │  │ ACQUIRE      │
 │ details      │  │ belief +      │  │ belief,      │  │ threshold,  │  │ DEFER        │
 │              │  │ ignorance u   │  │ ignorance,   │  │ conflict,   │  │ + tentative  │
 │              │  │               │  │ CONFLICT     │  │ cost        │  │   opinion    │
 └──────────────┘  └───────────────┘  └──────────────┘  └─────────────┘  └──────┬───────┘
                                                                                │
           agents: specialists · evidence auditor · librarian (similar cases) · report + verifier
           care platform: patient app · doctor queue · instructions · follow-ups · audit log
```

Say: *"The model never decides alone whether it is allowed to answer. That is the job of
steps 4 and 5, and they are calibrated, not tuned."*

---

## 5. The results slide

Use exactly this table (from [14_results.md](14_results.md)):

| Policy | answered alone | dermoscopy | accuracy | missed cancers |
|---|---|---|---|---|
| Always answer from the photo | 100% | 0% | 64% | 5.2% |
| Standard abstention, same coverage | 37% | 0% | 84% | 5.2% |
| **D-GUARD** | **37%** | **6%** | **84%** | **1.7%** |

Then say the three numbers out loud: **"Same coverage. Same accuracy. A third of the missed
cancers."**

Second line, if time: dermoscopy improves cancer detection from AUROC 0.80 to 0.86, and the
system knows when it is worth asking — its acquisition estimator scores +0.27 on
acquisition efficiency against +0.01 for "ask the most uncertain cases".

---

## 6. The live demo (3 minutes)

### 6.1 Before the panel

```bash
# 30 minutes before
streamlit run app/dguard_app.py          # first load takes ~10 s, then it is fast
python scripts/demo_care.py              # backup: same story, in the terminal
```

Open the app once so the model is loaded. Have the terminal backup open in a second
window. **Never demo on a cold start.**

### 6.2 The script

**Patient view**

1. Register "Demo Patient", open a case (forearm).
2. Leave *Use a real held-out MILK10k lesion* on. Say: *"This lesion is from the test set —
   the model never saw it during training."*
3. Pick a lesion and press **Submit photo + details and assess**.

Now one of three things happens. Narrate whichever it is:

| You see | Say |
|---|---|
| ✅ **PROCEED** | "Confident. It states the diagnosis, tells the patient what to do next, and books a follow-up." Point at the dynamic fusion weights: "This is how much each source contributed to *this* case." |
| 📷 **ACQUIRE** | "It is not sure, and it thinks a dermoscopy photo will help." Click **Upload the dermoscopy photo**. "Watch the ignorance number drop and the weight shift to dermoscopy." |
| 🧑‍⚕️ **DEFER** | "It will not decide alone — but it does not go silent. It gives a tentative opinion and tells the patient to see a doctor soon." |

4. Type a question in the chat box: *"Should I be worried?"*

**Doctor view**

5. Switch to **Doctor**. "Deferred cases come first, urgent before routine."
6. Open the case: images, the tentative opinion, ignorance, conflict, which source said
   what, and how reliable each source has been for that diagnosis.
7. Reply to the question, choose **CONFIRM**, write an instruction, set a follow-up of 90
   days, submit.

**Back to the patient**

8. Switch to **Patient**: the doctor's message and the follow-up date are there. Press
   **Re-submit the same photo** to show the follow-up change check.

**System view**

9. Show the results table and the audit log: "Every step is recorded, including which
   version of the policy made each decision."

### 6.3 If something goes wrong

| Problem | Do |
|---|---|
| App will not start | Switch to the terminal: `python scripts/demo_care.py` shows all three paths |
| A lesion gives a boring result | Pick another — the 24 demo lesions cover all three actions |
| Laptop is slow | The terminal demo needs no browser |

---

## 7. Honest limits — say these yourself

1. **It answers alone for about a third of patients.** The rest get a tentative opinion and
   a doctor. That is limited by the image model, which is a standard ResNet50; a
   dermatology-specific encoder is the next step.
2. **Using every source did not beat the best pair** (0.379 vs 0.403). Our fusion reduced
   the damage a weak source does but did not remove it.
3. **The costs are a judgement.** We priced a missed cancer at 5× a false alarm and showed
   what changes at 2, 3 and 10.
4. **One training seed so far**, and MILK10k is a specialist-referral population (74%
   malignant), so nothing here is a screening claim.
5. **Research prototype**, not a medical device; no clinical study.

---

## 8. Questions the panel will ask — and answers

**Q1. What is new here? Evidential learning, Dempster–Shafer and conformal prediction all
exist.**
> Correct, and we cite them. The contribution is **combining them into one decision rule
> with three actions** — answer, acquire, defer — calibrated with a statistical guarantee,
> and **measuring the value of acquiring dermoscopy on real paired data** rather than
> simulating it. MILK10k is one of very few datasets where every lesion has both images,
> so the "should I ask for dermoscopy?" question can be answered with an oracle bound.

**Q2. Why not just use a confidence threshold?**
> We did — it is baseline B3. At the same coverage it misses 5.2% of cancers; ours misses
> 1.7%. Confidence cannot see a case where the model is 85% sure it is a mole but melanoma
> is still plausible. Our benign/malignant conformal set catches exactly that.

**Q3. What is the difference between ignorance and conflict, and why does it matter?**
> Ignorance: nobody knows — get more evidence (ACQUIRE). Conflict: the sources confidently
> disagree — a human must decide (DEFER). Older methods measure "how different are the
> predictions" and report two ignorant sources as *agreeing*. Dempster–Shafer gives two
> separate numbers.

**Q4. Is this "dynamic multimodal fusion"?**
> Yes. Each source's influence is set per case by how much evidence it produced. A blurry
> photo produces little evidence and its weight drops on its own — the app shows these
> weights for every case. We also measured which source is reliable for which diagnosis:
> dermoscopy for SCC (0.74 vs 0.36 for the photo), the photo for melanoma.

**Q5. Your system defers 63% of cases. Isn't that useless?**
> It is not silent on those cases — it gives a tentative diagnosis, a differential, a
> cancer probability and an urgency flag, and routes them to a doctor's queue, urgent
> first. Deferral is a triage decision, not a refusal. And the deferral rate is set by the
> image model's quality; the same policy on a stronger encoder lets more through.

**Q6. Then why is your average cost higher than "always answer"?**
> Because each deferral is priced at 1.5. If throughput is all you care about, always
> answering is cheaper — and misses 5.2% of cancers. The right comparison for a deferral
> system is at matched coverage, where we are both safer and cheaper. We report both.

**Q7. What is the guarantee, exactly?**
> Learn-then-Test: with probability at least 90% over the calibration data, the
> cancer-miss rate among answered cases is at most the target. It assumes the test patients
> come from the same population as the calibration patients, and it does not hold under
> distribution shift — measuring that is part of our stress tests.

**Q8. Why MILK10k?**
> Every lesion has a clinical photo and a dermoscopy image, so acquiring dermoscopy is a
> real experiment. 95.7% histopathology-confirmed. We audited it: 11 classes, 74%
> malignant, zero duplicate images.

**Q9. How do you prevent data leakage?**
> Splits are by lesion, so both images of a lesion are on the same side, and the split is
> asserted at runtime. MILK10k has no patient IDs, so one patient could appear in two
> splits — we state that. The retrieval index holds training cases only, verified by a
> test, because a leaked neighbour would carry its true label into the written report.

**Q10. What does the LLM do?**
> It writes the report. It never diagnoses. A verifier checks every claim in the report
> against the evidence and escalates if it fails twice. With no LLM available the system
> writes a deterministic summary and keeps working.

**Q11. How would this be used in practice?**
> As triage in a dermatology referral pathway: patients upload a photo, the system answers
> the clear cases, asks for dermoscopy when it would help, and puts the rest in a doctor's
> queue ordered by urgency, with follow-ups and a change check on the next photo.

**Q12. Earlier versions reported much higher accuracy. What happened?**
> We audited the original training code and found the test set overlapped the training
> set and predictions were rewritten toward target numbers. We discarded those results,
> rebuilt the pipeline, and every number we report now regenerates from the code. The
> audit is in `docs/evaluation_integrity.md`.

**Q13. Is this clinically validated?**
> No. It is a retrospective research prototype. Validation would need a prospective study.

**Q14. Skin tone fairness?**
> MILK10k has a six-level skin-tone label, but it is very unbalanced — tone 0 has 6
> lesions. We report results per tone where there is enough data and give no per-group
> guarantee where there is not.

---

## 9. Words to use and avoid

| Use | Avoid |
|---|---|
| "decision support", "research prototype" | "diagnoses", "replaces a dermatologist" |
| "3× fewer missed cancers at the same coverage" | "safer" (without the condition) |
| "statistical guarantee under exchangeability" | "guaranteed safe" |
| "we combine known methods into a new decision rule" | "first ever", "state of the art", "novel algorithm" |
| "one training seed so far" | quoting a number without its seed and split |

---

## 10. Files to have open

| For | Open |
|---|---|
| Any number | [14_results.md](14_results.md) |
| How it works | [11_learning_path.md](11_learning_path.md), [12_system_walkthrough.md](12_system_walkthrough.md) |
| Failures | [what_did_not_work.md](what_did_not_work.md) |
| Integrity | [evaluation_integrity.md](evaluation_integrity.md) |
| The paper | `paper/main.tex` |
