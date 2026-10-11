# Learning Path — Understand D-GUARD From Zero

For someone who has never seen the project. Read top to bottom; every idea is explained
before it is used. Where a formula appears, the sentence before it says what it means, so
you can skip the maths and still follow.

All numbers come from the trained MILK10k model on 787 held-out test lesions
([05_results.md](05_results.md)).

---

## Part 0 — The one-paragraph version

A doctor looks at a skin spot and decides: *could this be cancer?* An AI can guess too, but
ordinary AI **guesses confidently even when it should not** — on a blurry photo, or when
the photo says one thing and the patient's details say another.

D-GUARD gives one of four outcomes instead of always guessing:

| | Outcome | Plain meaning |
|---|---|---|
| ✅ | **Answer — HIGH confidence** | "I'm sure enough; here is my diagnosis." |
| 📷 | **Ask for dermoscopy** | "A close-up through a skin lens would settle this — please take one." |
| ⚠️ | **Answer — LOW confidence** | "My best guess is X, but don't act on it until a doctor confirms." |
| 🧑‍⚕️ | **Defer** | "The photo and your details contradict each other — a doctor must decide." |

And for the HIGH-confidence answers it gives a **statistical promise**: the rate of
missed cancers is bounded, with a proof — not a hope.

---

## Part 1 — Why normal AI is not enough

A classifier turns an image into one number per disease, and the biggest wins:

```
  photo  ─►  [ neural network ]  ─►  BCC   0.82   ◄── answer = BCC
                                     MEL   0.11
                                     NV    0.07
```

These numbers come from **softmax**, which forces them to add up to 1.0. So **something
always wins** — even for a photo of a car. "0.82" does not mean "82% likely BCC"; it means
"of the options I was given, BCC scored highest".

| Situation | Softmax says | Should say |
|---|---|---|
| Sharp photo, textbook BCC | BCC 0.82 | "BCC, confident" |
| Blurry photo | BCC 0.82 | "I can't tell" |

You cannot tell these apart from the numbers. That is the problem D-GUARD fixes.

---

## Part 2 — Concept 1: Evidence and ignorance (Evidential Deep Learning)

### Two kinds of "unsure"

| Type | Meaning | Example | Does more evidence help? |
|---|---|---|---|
| **Aleatoric** | the case is genuinely ambiguous | a lesion that truly looks half-way between two diseases | **no** |
| **Epistemic** | the model lacks information | blurry photo, missing details | **yes** |

This decides the action: if more evidence would help → **ask for dermoscopy**; if not → a
doctor should look.

### Count evidence instead of forcing percentages

The network outputs **evidence** for each disease — any non-negative amount, where zero
means "no support at all":

```
  Lots of evidence:              Almost none:
  BCC  evidence 40               BCC  evidence 0.1
  MEL  evidence  5               MEL  evidence 0.1
  NV   evidence  3               NV   evidence 0.1
  total 48  → confident          total 0.3  → "I don't know"
```

**The total amount of evidence is exactly what softmax throws away.**

### The four formulas

With `K` diseases (here 11) and evidence `e`:

| Formula | Name | Meaning |
|---|---|---|
| `α = e + 1` | alpha | evidence plus a "start from nothing" baseline |
| `S = Σα` | strength | total evidence |
| `b = e / S` | **belief** | share of support for each disease |
| `u = K / S` | **ignorance** | how much is NOT known — 1.0 = nothing at all |

`u = 1` is the **vacuous opinion**: complete ignorance. Belief plus ignorance always add to
1. Remember `u = 1`; Part 4 needs it.

**In our results:** with just the photo and details, average ignorance is **0.68**. Add the
dermoscopy image and it drops to **0.44**. The model *knows* it learned something.

Code: `src/evidence/evidential.py`. Paper: Sensoy et al., NeurIPS 2018.

---

## Part 3 — Concept 2: Multimodal (several kinds of input)

"Modality" just means a kind of information:

| Input | What it is | Alone, on MILK10k (macro-F1) |
|---|---|---|
| Clinical photo | an ordinary close-up | 0.268 |
| **Dermoscopy** | a photo through a lit magnifying lens | **0.396** |
| Details (metadata) | age, sex, body site, skin tone, 7 concept scores | 0.245 |

**Macro-F1** scores every disease equally, so a model that only gets the common disease
right scores badly. On MILK10k, 48% of lesions are one disease (BCC), so plain accuracy
would be misleading.

### The fusion problem

| Combination | macro-F1 |
|---|---|
| dermoscopy + details | **0.403** ← best |
| all three | 0.379 |

Adding the clinical photo to the best pair made it **slightly worse**. The usual way to
combine inputs — glue them into one long vector — gives every input a fixed slot every
time, so a weaker input keeps adding noise. We need combination that decides **per patient**
how much to trust each input. That is Part 4. (Honest note: our method shrank this effect
but did not remove it — see [what_did_not_work.md](what_did_not_work.md) §17.)

---

## Part 4 — Concept 3: Dempster–Shafer fusion (the heart of the project)

Each input forms its **own** opinion (belief + ignorance). The opinions are combined with
**Dempster–Shafer's rule**:

```
  clinical photo ──► opinion  b=[NV .5, BCC .2, ...]  u=0.45
  details        ──► opinion  b=[BCC .3, NV .1, ...]  u=0.60
  dermoscopy     ──► (missing) vacuous: b=0, u=1
                             │
                       Dempster–Shafer
                             ▼
              belief  +  ignorance u  +  CONFLICT C
```

### Property 1 — weak inputs quiet themselves (dynamic fusion)

An opinion with high ignorance barely moves the result. Nobody writes "distrust blurry
photos" — it follows from the maths. Each input's influence on *this* patient is shown as a
**dynamic fusion weight**, `w = (1 − u) / Σ(1 − u)`.

Real example (lesion IL_0449677): photo + details → weights photo 0.59 / details 0.41.
Add dermoscopy → photo **0.16**, dermoscopy **0.73**, details 0.11. The trust moved to the
better evidence on its own.

### Property 2 — missing inputs vanish exactly

A missing input is the vacuous opinion, and in Dempster–Shafer that is the **identity
element** — like adding zero. No placeholder images, no special code for "no dermoscopy".

### Property 3 — conflict is not ignorance

```
  CASE A: real disagreement         CASE B: nobody knows
  photo:   MEL, confident           photo:   no idea
  details: NV,  confident           details: no idea
  → conflict HIGH, ignorance LOW    → conflict LOW, ignorance HIGH
  → a HUMAN must decide (DEFER)     → get MORE EVIDENCE (ask for dermoscopy)
```

These need **opposite** responses. Older methods ("how different are the two
predictions?") call case B *agreement*. Dempster–Shafer keeps the two numbers apart:
`C = Σ_{i≠j} b₁ᵢ b₂ⱼ` — belief one source puts on a disease while the other puts belief on
a *different* one.

### The old system's idea, measured

The old project typed in a table of "which input to trust for which disease". We now
**measure** it (how often each input alone names the right diagnosis, on validation):

| Diagnosis | photo | dermoscopy | details |
|---|---|---|---|
| SCC / keratoacanthoma | 0.36 | **0.74** | 0.40 |
| Melanoma | **0.52** | 0.40 | 0.43 |
| BCC | 0.82 | **0.93** | 0.92 |

The doctor's screen shows this row for the system's diagnosis.

Code: `src/evidence/evidential.py`, `src/native/engine.py`. Paper: Han et al., ICLR 2021.

---

## Part 5 — Concept 4: Conformal prediction (a set with a guarantee)

Instead of one answer, output a **set** guaranteed to contain the truth a chosen share of
the time — here, 90%.

D-GUARD asks the one question that matters for safety — **cancer or not?** — so its set is
over two outcomes:

```
  { malignant }          cancer is likely, benign ruled out
  { benign }             benign is likely, cancer ruled out
  { benign, malignant }  cannot rule cancer out
```

How it is built (three steps): hold out calibration lesions the model never trained on;
score how "surprised" the model was by each true answer (`1 − p(true)`); take the 90th
percentile, separately for cancers and non-cancers. Doing it **per outcome** guarantees
that **at least 90% of cancers keep "malignant" in their set**.

### The rule: no false reassurance

If the system would say a **benign** diagnosis while "malignant" is still in the set, it may
not give that as a confident answer. Saying "likely BCC, see a dermatologist" while benign
is also possible is the safe direction and is allowed.

Code: `src/calibration/binary.py`.

---

## Part 6 — Concept 5: Learn-then-Test (a threshold with a promise)

Old systems answered "if confidence > 0.80" — a number someone chose. **Learn-then-Test**
chooses the threshold by statistical testing: for each candidate, "can I prove the miss rate
is low enough?" Only provable thresholds survive.

```
   P( miss rate ≤ α )  ≥  1 − δ          α = 5%,  δ = 10%
```

"With at least 90% probability over the calibration data, the cancer-miss rate of answers
above the threshold is at most 5%." If no threshold can be proven, it **says so** rather
than guessing.

On MILK10k (786 calibration lesions) a threshold was certified at both stages. On the
earlier PAD-UFES prototype (432 cases) it could not be — too few cases to prove anything.
This is why calibration needs enough data.

Code: `src/calibration/ltt.py`. Paper: Angelopoulos et al., Ann. Appl. Stat. 2025.

---

## Part 7 — Concept 6: Choosing the action by cost

Mistakes are not equal. Telling a melanoma patient "you're fine" is far worse than a false
alarm. So the costs are written down:

```
   missing a cancer        5      (10 in the stricter setting)
   false alarm             2
   doctor review           1.5
   ask for dermoscopy      0.5
   correct                 0
```

For each option, compute the expected cost and take the cheapest (Chow's rule, 1970):

```
   answer:      Σ P(disease) × cost of saying our diagnosis if that disease is true
   dermoscopy:  0.5 + expected cost after the new photo  (estimated by a small model)
   defer:       1.5
```

The *stated* diagnosis is always the most likely one. The costs only decide whether it can
be stated confidently. (Using costs to choose the label itself made the system call 98.5%
of lesions cancer — [what_did_not_work.md](what_did_not_work.md) §18.)

### Should we ask for dermoscopy?

A small model learns, from validation patients, how much dermoscopy *actually* helped
given what the photo showed. On test patients it captures **far more** of dermoscopy's
benefit than "ask the most uncertain patients" (acquisition efficiency +0.27 vs +0.01).

Code: `src/decision/policy.py`.

---

## Part 8 — Concept 7: Confidence tiers (why we rarely defer)

Earlier versions **deferred** whenever a safety check fired — 63% of patients. A deferred
patient waiting for a doctor with no information is a bad experience. So D-GUARD now
**answers** those cases too, in a separate tier:

| Tier | When | The patient sees |
|---|---|---|
| **HIGH** | every check passed | "Result: X." + next steps |
| **LOW** | a check fired | "Low-confidence assessment: X (52% likely). Not reliable enough to act on by itself — a doctor will confirm." + urgency |
| **DEFER** | sources contradict | "A doctor must decide. Unconfirmed impression: X." |

Results: **90% of patients get an answer.** HIGH answers: 84% accurate, 1.8% missed
cancers. LOW answers: 51% exactly right, but the true diagnosis is in the top 3 for **82%** —
useful for the reviewing doctor, and never presented as certain.

The promise from Part 6 applies to the HIGH tier, which is identical whether or not the LOW
tier exists.

---

## Part 9 — Concept 8: Measuring a system that can say "not sure"

Accuracy alone is wrong: a system that answers only easy cases looks perfect. So we measure:

- **Coverage** — share of patients answered (HIGH tier).
- **Missed-cancer rate** among answered patients.
- **AUGRC** — area under the error-vs-coverage curve, divided by *all* cases (lower is
  better; Traub et al., NeurIPS 2024). Native ignorance ranks errors as well as softmax
  (0.0900 vs 0.0893).

And we always compare at **matched coverage**: the standard approach, answering the same
35% of patients, misses 4.7% of cancers; D-GUARD misses 1.8%.

Code: `src/eval/selective.py`.

---

## Part 10 — Concept 9: The agents

Agents are helpers with one job each. **Only the decision engine decides**; the others
explain, warn and write.

```
     photo(s) + details
            │
   ┌────────▼────────┐
   │ Quality agent   │  sharp? well lit? glare? (thresholds fitted on MILK10k photos)
   └────────┬────────┘
   ┌────────▼────────┐
   │ Decision engine │  evidence → Dempster–Shafer → gates → cost → tier
   └────────┬────────┘
   ┌────────▼────────┐
   │ Librarian (RAG) │  6 most similar CONFIRMED past cases, with photos
   └────────┬────────┘
   ┌────────▼────────┐
   │ Report agent    │  structured report; optional LLM rewrite, checked claim by claim
   └─────────────────┘
```

**RAG** = retrieval-augmented generation: find similar past patients and show them. The
index holds **training lesions only** — a test lesion in the index would leak its true
diagnosis into what the doctor reads. Retrieval **never** changes the decision: the
guarantees in Parts 5–6 assume the decision does not depend on a changing collection.

**The report verifier** rejects an LLM rewrite that names a diagnosis not in the evidence,
quotes a percentage that matches no evidence value, or drops "confirm with a doctor". With
no LLM, the structured report is used.

Code: `src/agents/`.

---

## Part 11 — The care platform

```
 PATIENT                         SYSTEM                          DOCTOR
 open case, upload photo  ──►  assess
                               ├ HIGH  ─► result + follow-up date ─► sign-off queue
                               ├ dermoscopy? ◄── patient uploads ──┘
                               ├ LOW   ─► answer + "doctor will confirm" ─► confirm queue
                               └ DEFER ─► tentative opinion ─────────────► review queue (urgent first)
 asks a question  ───────────────────────────────────────────────────►  replies
 gets instructions + follow-up  ◄──────────────────────────────────────  confirms / overrides
 follow-up photo  ──►  re-assess + CHANGE CHECK  ──► worse? back to doctor, URGENT
```

Every step is written to an audit log with the policy version that produced it.

Code: `src/care/`, `app/dguard_app.py`, `src/api.py`.

---

## Part 12 — Check your understanding

1. Why can't softmax say "I don't know"?
2. Aleatoric vs epistemic — which one does dermoscopy help with?
3. Why did adding the clinical photo to dermoscopy + details lower macro-F1?
4. What does `u = 1` mean, and why does combining with it change nothing?
5. Give a case with high conflict and low ignorance. What should happen?
6. Why is the conformal set over benign/malignant rather than over 11 diseases?
7. What does Learn-then-Test promise, and when does it refuse?
8. Why is the stated diagnosis the most likely one, not the cheapest one?
9. What is the difference between a LOW-confidence answer and a deferral?
10. Why must retrieval never influence the decision?

Answers: Parts 1, 2, 3, 2+4, 4, 5, 6, 7, 8, 10.

Next: the 10-day plan in [08_study_plan.md](08_study_plan.md).
