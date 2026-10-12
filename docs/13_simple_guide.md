# D-GUARD in plain words — the whole project in 15 minutes

No maths, no jargon. If you read only one document, read this one. Every idea here has a
"deeper" link if you want more.

---

## 1. The problem, as a story

A patient photographs a spot on their arm and asks: *"Is this dangerous?"*

A normal AI model **always** answers. Blurry photo? It answers. Photo of the wrong spot?
It answers. It even says "92% sure" when it is guessing. In medicine that is the worst
habit possible: a confident wrong "it's harmless" can delay a cancer diagnosis.

A good doctor behaves differently. They say *"I'm sure"* when they are, *"let me look
closer with a dermatoscope"* when they need more information, *"I think it's X, but let's
get a second opinion"* when unsure, and *"this doesn't add up, I'm referring you"* when the
evidence contradicts itself.

**D-GUARD is an AI that behaves like that doctor.**

---

## 2. The idea in one picture

```
   photo ─┐                                     ┌──► ✅ HIGH: "It's X." (and it's usually right)
details ──┼─► 6 specialists ─► Devil's ─► Gate- ┼──► ⚠️ LOW: "Probably X — a doctor will confirm."
dermo* ───┘   each give an     Advocate   keeper ├──► 📷 ASK: "Please send a dermoscopy photo / your details."
              opinion          (who to    (safety└──► 🧑‍⚕️ DEFER: "The evidence disagrees — a doctor decides."
                               trust?)     checks)
* only if the system asks for it
```

---

## 3. The six specialists — "a team, not one model"

Think of a hospital case meeting. Each specialist sees something different:

| Specialist | Looks at | Like… |
|---|---|---|
| S1 photo expert | the normal photo | a GP looking at the spot |
| S2 dermoscopy expert | the magnified skin-surface image | a dermatologist with a dermatoscope |
| S3 details expert | age, sex, body site, skin tone | an epidemiologist ("at this age, on the face, it's usually…") |
| S4 the old project's Model A | the photo, but trained on a different dataset | a visiting doctor from another country |
| S5 similar-case vote | the 15 most similar past patients | "I've seen cases like this before" |
| S6 embedding classifier | the same past patients, but draws one clean line between diagnoses | a careful statistician |

S1–S3 are **evidential**: besides an opinion, they say *how much they actually know*. A
blank or blurry input produces "I don't know", not a random guess.
Deeper: [02_learning_path.md](02_learning_path.md) Parts 2–4.

---

## 4. The Devil's Advocate — "who should we trust today?"

Not every specialist deserves the same say on every patient. The Devil's Advocate asks
about each one:

- **Is its input good?** (sharp photo or blurry?)
- **Is it usually right in cases like this?** (it *learned* this from past cases)
- **How sure is it?**
- **What is its track record for the diagnosis we suspect?** (dermoscopy is great for
  squamous cell carcinoma, the photo is better for melanoma)
- **Does it agree with the others?**
- **Is this patient unlike anyone we've seen?** (out-of-distribution)

Then it hands out **weights** — how much each opinion counts for *this* patient.

We tried **seven ways** of turning those answers into weights (from "everyone counts the
same" to a learned rule) and let the data choose — on cases set aside for choosing, never
on the final exam cases. The learned rule (we call it **v2**) won.
Deeper: [12_how_it_was_built.md](12_how_it_was_built.md), [03_system.md](03_system.md) §4a.

---

## 5. The Gatekeeper — safety checks before any answer

Even with a good combined opinion, the answer must pass checks — like a pilot's checklist:

| Check | Plain meaning | If it fails |
|---|---|---|
| Contradiction | do the sources strongly disagree? | DEFER to a doctor |
| Cancer ruled out? | can we honestly exclude cancer before saying "harmless"? | no confident answer |
| Certified risk | is confidence high enough that, on past cases, at most 5% of cancers were missed? | no confident answer |
| Photo quality | blurry, dark, grainy, covered? | ask for a better photo |
| Unusual case | unlike anything seen in training? | no confident answer |
| Details missing | no age / site and no dermoscopy? | ASK for the details |
| Cost | is answering cheaper than a review, counting a missed cancer as 5× worse than a false alarm? | LOW answer or ask |

The "certified risk" check comes with a **mathematical guarantee** (called Learn-then-Test):
if future patients resemble past ones, the missed-cancer rate stays under the target.

---

## 6. Asking for more — the part that makes it feel like a doctor

When the photo alone is not enough, D-GUARD **asks**:
- for a **dermoscopy photo** (about 30% of patients), then decides again;
- for **missing details** (we found the system misses many more cancers without them);
- for a **better photo** when the quality is poor.

---

## 7. What happened — the results in one table

787 patients the system had never seen (MILK10k test set):

| What patients got | Share |
|---|---|
| ✅ Confident answer — right 77% of the time, 3.1% of cancers missed | 57% |
| ⚠️ Low-confidence answer, a doctor confirms — true diagnosis in its top 3 for ~86% | 32% |
| 🧑‍⚕️ Sent to a doctor because the evidence disagreed | 11% |

Is that good? Three honest comparisons:
- **vs. "answer only when confident"** (the standard trick): it misses 3.8% — slightly
  more than us, but the difference could be chance. With a weaker combination method it
  missed 9.3%, while ours stayed near 3%. **Our safety does not depend on luck with the
  model; the simple trick's does.**
- **vs. a learned "know when you're wrong" model**: it misses 4.9%, significantly more.
- **vs. always answering**: always answering is cheaper — but misses 6–7% of cancers.

Deeper: [05_results.md](05_results.md).

---

## 8. Why the confidence numbers look modest

- There are **11 diagnoses**, several look alike, and some are rare.
- The confidence shown is **calibrated** — when it says 70%, about 70% of such answers were
  right. Many AI apps show a flattering number instead.
- The image model is a general-purpose ResNet50, not a skin-specialist model. A skin
  foundation model is the honest way to raise accuracy (the Kaggle script supports it).

---

## 9. The old project — what we kept and what we learned

We kept its ideas: the specialist team, the Devil's Advocate, diagnosis-dependent trust,
and its Model A. When we re-checked the old code we found that its evaluation **adjusted
predictions toward preset target accuracies**, and that its split put photos of the same
patient in both training and test. Re-scored honestly, the old models reach 57–75% on
unseen patients instead of the reported 84–90%. The new project fixes both problems:
nothing is adjusted, and no lesion is in two splits. We even fused the old seven models
with the new Devil's Advocate — it does better than averaging them (73% vs 60%).
Deeper: [what_did_not_work.md](what_did_not_work.md) §40.

---

## 10. Where things are

| I want to… | Go to |
|---|---|
| see it working | `streamlit run app/dguard_app.py` → [11_ui_guide.md](11_ui_guide.md) |
| understand the concepts | [02_learning_path.md](02_learning_path.md) |
| know how it was built and checked | [12_how_it_was_built.md](12_how_it_was_built.md) |
| see every number | [05_results.md](05_results.md) |
| present it | [07_panel_guide.md](07_panel_guide.md) |
| read the paper | [../paper/main.tex](../paper/main.tex) |

---

## 11. Ten words to know

| Word | Meaning |
|---|---|
| Modality | a kind of input: photo, dermoscopy, details |
| Evidential | a model that also says how much it doesn't know |
| Ignorance (u) | "I don't know" amount, 0 to 1 |
| Conflict | sources confidently disagree |
| Fusion | combining several opinions into one |
| Calibrated | a 70% confidence is right ~70% of the time |
| Coverage | share of patients given a (confident) answer |
| Missed cancer | a cancer the system called harmless — the number we fight hardest |
| Cross-fitting | fit on part of the data, check on the rest, so you don't fool yourself |
| Out-of-distribution | a case unlike anything in the training data |
