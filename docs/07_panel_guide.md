# Panel Guide — Presenting and Defending D-GUARD

How to present the project in 15 minutes, run the live demo, and answer the questions a
panel will ask. Plain English. Every claim below is backed by a file you can open in front
of the panel.

---

## 1. The one sentence

> **D-GUARD is a skin-lesion AI that answers 9 in 10 patients — confidently when every
> safety check passes, clearly marked "low confidence" when not — asks for a dermoscopy
> photo when that would settle the case, and hands contradictory cases to a doctor. Its
> confident answers miss three times fewer cancers than the standard approach.**

Learn it word for word. Everything else in the talk supports it.

---

## 2. The story in five beats

| # | Beat | Say | Show |
|---|---|---|---|
| 1 | **Problem** | "AI gives a confident answer even when it should not — a blurry photo, missing details, sources that disagree. Softmax always picks something." | One slide: same "0.82 BCC" for a clear and a blurry photo |
| 2 | **Idea** | "Let each source say how much it actually knows, combine them so ignorant sources fall silent, and let the system choose between three actions." | The architecture diagram (§4) |
| 3 | **Guarantee** | "Thresholds are not hand-picked. They are calibrated with statistical tests that come with a written error bound." | One slide: PROCEED / ACQUIRE / DEFER with the gates |
| 4 | **Evidence** | "Trained on MILK10k. It answers 9 in 10 patients, more than half confidently at 77% accuracy with 3.1% missed cancers — fewer than a learned rejector (4.9%) — and that miss rate stays put whichever fusion we put under it, while a plain confidence threshold swings from 3.5% to 9.3%." | Results table (§5) |
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

Slide 9a — what patients get (from [05_results.md](05_results.md)):

| Outcome | share | accuracy | missed cancers |
|---|---|---|---|
| ✅ answered, HIGH confidence | 57% | **77%** (11 classes) | **3.1%** |
| ⚠️ answered, LOW confidence (doctor confirms) | 32% | 50% — true diagnosis in top 3: **86%** | 21% |
| 🧑‍⚕️ deferred (sources contradict) | 11% | — | — |

Say: **"Nine in ten patients get an answer, more than half a confident one — right 77% of
the time across 11 diagnoses."** Then the fusion slide: **"Seven ways to combine six
specialists, compared fairly; the learned Devil's Advocate wins on validation and lifts
photo-only macro-F1 from 0.29 to 0.38."** (figure `paper/figures/fig_fusion_rules.png`)

Slide 9b — the finding (same policy, four fusions; figure `fig_backend_robustness.png`):

| Fusion under the policy | D-GUARD missed cancers | confidence threshold, same coverage |
|---|---|---|
| iterative EM, 5 specialists | **3.3%** | 9.3% |
| learned gate, 5 specialists | **3.5%** | 3.5% |
| iterative EM, 6 specialists | **3.0%** | 4.7% |
| learned gate, 6 specialists (final) | **3.1%** | 3.8% |

Say: **"Our policy holds its miss rate whatever it is given. A plain threshold is only as
safe as the probabilities under it — with good probabilities it catches up, and we say so."**

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

Now one of four things happens. Narrate whichever it is:

| You see | Say |
|---|---|
| ✅ **HIGH confidence** | "Every safety check passed. It states the diagnosis, tells the patient what to do, and books a follow-up." Point at the dynamic fusion weights: "This is how much each source contributed to *this* case." |
| 📷 **ACQUIRE** | "It thinks a dermoscopy photo would settle it." Click **Upload the dermoscopy photo**. "Watch ignorance drop and the weight move to dermoscopy." |
| ⚠️ **LOW confidence** | "It still answers — but says plainly this is low confidence and a doctor will confirm. No patient is left with nothing." |
| 🧑‍⚕️ **DEFER** | "The photo and the details contradict each other. There's no single answer to give, so a doctor decides — but the patient still sees the tentative opinion and an urgency flag." |

Then switch to **Doctor** and open the case: show the **report**, the **similar confirmed
past cases** gallery ("training lesions with biopsy-confirmed diagnoses — shown for
context, never used to decide"), and any **photo-quality warning**.

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

### 6.2b The Stress Lab moment (1 minute — do this, it convinces)

1. Switch to **Stress Lab**. Pick a lesion, choose **blur**, press **Run both**.
2. Say: *"Left is the original photo. Right is the same photo blurred. Watch the
   gatekeeper."* The blurred one loses its HIGH-confidence label — the photo-quality gate
   fires.
3. Say the honest part: *"Without that gate, the model itself would have been **more**
   confident on the blurred photo. We found that in our own stress test — that is why the
   gate exists."*
4. Try **remove patient details**: ignorance rises and it asks for dermoscopy.

### 6.3 If something goes wrong

| Problem | Do |
|---|---|
| App will not start | Switch to the terminal: `python scripts/demo_care.py` shows all three paths |
| A lesion gives a boring result | Pick another — the 24 demo lesions cover all three actions |
| Laptop is slow | The terminal demo needs no browser |

---

## 7. Honest limits — say these yourself

1. **Only a third of answers are HIGH confidence.** The rest are answered in the LOW tier
   (or deferred) and need a doctor. That is limited by the image model, a standard
   ResNet50; a dermatology-specific encoder is the next step.
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
> We tested it on four fusions. It missed between 3.5% and 9.3% of cancers at our
> coverage; we missed 3.0–3.5% on all four. On the final model it misses 3.8% vs our 3.1%
> (not significant) and is more accurate. So a threshold *can* be enough, but only if the
> probabilities are good, and you cannot know that in advance. Our policy gives the same certified miss rate either way, asks for
> dermoscopy or details, and answers the other 35% with a clearly marked LOW tier.

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

**Q4b. Did you keep the original system's ideas?**
> Yes, and they now work better than before because they are learned. The specialist team
> includes the original PAD-UFES Model A; the Devil's Advocate profiles every specialist;
> the iterative diagnosis-adaptive fusion is the original EM algorithm with a measured
> trust table instead of a hand-typed one. Version 2 goes further: every evidence signal
> sets the weights through a learned gate. We compared seven combination rules fairly and
> picked on validation; a plain average is the worst of them, so the gain is the method.

**Q4c. Why is the old model weighted so low?**
> It was trained on a different dataset with different cameras and reaches 22% accuracy on
> MILK10k. Nobody told the system that; the learned reliability found it and cut its weight
> to about 2%.

**Q4d. Is your safety gain statistically significant?**
> Against a learned rejector, yes: 3.1% vs 4.9%, p = 0.011. Against a confidence
> threshold on the final model, no (3.8%, p = 0.18) — but the threshold's safety depends on
> the fusion (3.5–9.3%), ours does not (3.0–3.5%). What they cannot do: certify the bound, stay
> stable across fusions, ask for dermoscopy or details, answer the remaining patients with
> a LOW tier, and explain each decision.

**Q4e. Does it work on other data?**
> We ran it unchanged on two other datasets: PAD-UFES-20 (Brazilian smartphone photos)
> and Derm7pt (an Italian atlas with photo + dermoscopy, so the whole "ask for dermoscopy"
> loop runs). The safety decision transfers better than the 11-class diagnosis. Numbers in
> the Research view; the guarantee is only promised when test data resemble calibration
> data.

**Q4f. Why is the confidence often low?**
> Eleven diagnoses with look-alikes and rare classes; evidential heads built to say "I don't
> know"; and the number we show is *calibrated* — with dermoscopy, 76% shown on average
> vs 73% actually right. A raw softmax would look higher and be wrong more often than it
> claims. The honest way up is a dermatology backbone and more data, not a different number.

**Q4g. The old system had Models A–G. Why only A?**
> B–G need PAD-UFES's clinical fields (smoking, pesticide, itch, bleeding…) or free text
> generated from them; MILK10k has none of these. Model A is image-only, so it transfers,
> and it is in the team as specialist S4.

**Q4h. The old system reported 85–90%. Why are your numbers lower?**
> Different, harder task (11 classes, lesion-level split). And the old numbers were not
> real: its split was by photo (43% of test photos showed a lesion already in training),
> and its evaluation code flipped predictions until each model hit a hard-coded target
> accuracy. We re-scored its saved models honestly: 57–75% on unseen patients. Fusing all
> seven with our learned gate gives 73% vs 60% for averaging (Research view).

**Q5. How often does it refuse to answer?**
> About 1 in 10 — only when the photo and the details contradict each other. Everyone
> else gets an answer: 57% HIGH confidence, 32% LOW confidence with a doctor confirming.
> Even deferred patients see the tentative opinion and an urgency flag. The share of HIGH
> answers is set by the image model's quality; a stronger encoder raises it.

**Q6. Why is your average cost higher than "always answer"?**
> Because every case that needs a doctor (LOW tier or deferred) is charged a review cost
> of 1.5. If throughput is all you care about, always answering is cheaper — and misses
> 5.2% of cancers. At matched coverage we are both safer and cheaper. We report both.

**Q6b. Isn't a LOW-confidence answer dangerous?**
> It is never presented as a result: the patient is told it is not reliable enough to act
> on, the case goes to a doctor's confirm queue, and if cancer is not ruled out it is
> flagged URGENT. The guarantee applies only to HIGH answers, and that tier is unchanged.

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
> audit is in `docs/reference/evaluation_integrity.md`.

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
| Any number | [05_results.md](05_results.md) |
| How it works | [02_learning_path.md](02_learning_path.md), [03_system.md](03_system.md) |
| Failures | [what_did_not_work.md](what_did_not_work.md) |
| Integrity | [reference/evaluation_integrity.md](reference/evaluation_integrity.md) |
| The paper | `paper/main.tex` |
