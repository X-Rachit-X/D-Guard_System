# Learning Path — Understand This Project From Zero

Written for someone who has never seen this project. Read top to bottom. Every idea is
explained before it is used. No step assumes the next one.

Plain English throughout. Where a formula appears, the sentence before it says what it
means, so you can skip the maths and still follow.

---

## Part 0 — The one-paragraph version

A doctor looks at a mole and decides: *is this skin cancer?* An AI can guess too. The
problem is that AI guesses **confidently even when it should not** — on a blurry photo, or
when the photo says one thing and the patient's history says another.

This project builds an AI that can say one of three things instead of always guessing:

1. **PROCEED** — "I am confident enough, here is my answer."
2. **ACQUIRE** — "I am not sure. Take a dermoscope photo and I will look again."
3. **DEFER** — "Send this to a human doctor."

And crucially, it comes with a **mathematical promise**: *"when I do answer, I will miss
cancer no more than 5% of the time."* Not a hope. A promise with a proof.

---

## Part 1 — Why normal AI is not enough

### 1.1 What a classifier does

A classifier takes an image and produces numbers, one per disease. The biggest number
wins.

```
  photo  ─►  [ neural network ]  ─►  BCC   0.82   ◄── biggest, so answer = BCC
                                     MEL   0.11
                                     NEV   0.07
```

Those numbers come from **softmax**, which forces them to add up to 1.0.

### 1.2 The flaw: softmax can never say "I don't know"

Because the numbers must add to 1.0, **something always wins**. Show the network a
photo of a car and it still outputs a skin disease with high confidence.

> **This is the central problem.** "0.82" does not mean "82% likely to be BCC". It means
> "of the options I was given, BCC scored highest" — even when none of them fit.

Two completely different situations produce the same-looking output:

| Situation | What softmax says | What it should say |
|---|---|---|
| Clear photo, textbook BCC | BCC 0.82 | "BCC, confident" |
| Blurry photo, no idea | BCC 0.82 | **"I cannot tell"** |

You cannot tell these apart from the numbers. That is what this project fixes.

---

## Part 2 — Concept 1: Evidential Deep Learning (separating two kinds of "unsure")

### 2.1 Two different reasons to be unsure

This distinction is the foundation of everything else:

| Type | Plain meaning | Example | Can more data fix it? |
|---|---|---|---|
| **Aleatoric** | The case is genuinely ambiguous | A lesion that truly looks half-way between two diseases | **No** |
| **Epistemic** | *The model* lacks knowledge | Blurry photo; a disease it never trained on | **Yes** |

Why this matters practically: if the uncertainty is **epistemic**, asking for a better
photo helps → **ACQUIRE**. If it is **aleatoric**, a better photo changes nothing → send
it to a human → **DEFER**.

Softmax mashes both into one number. Evidential learning keeps them apart.

### 2.2 How: count evidence instead of forcing percentages

Instead of outputting probabilities that must add to 1, the network outputs **evidence** —
a non-negative "how much support did I find for each disease", where **zero means none**.

```
Softmax (must total 1.0):        Evidential (can total anything):

  BCC  0.82                        BCC  evidence 40    ← lots of support
  MEL  0.11                        MEL  evidence  5
  NEV  0.07                        NEV  evidence  3
  ────────                         ─────────────────
  total 1.00  (always)             total 48  → confident

                                   BCC  evidence 0.1   ← almost no support
                                   MEL  evidence 0.1      for ANYTHING
                                   NEV  evidence 0.1
                                   ─────────────────
                                   total 0.3  → "I don't know"
```

**The total amount of evidence is the signal softmax throws away.** Low total = ignorance.

### 2.3 The four formulas

With `K` diseases (here K = 11) and evidence `e` for each:

| Formula | Name | Plain meaning |
|---|---|---|
| `α = e + 1` | alpha | evidence plus one "I start from nothing" baseline |
| `S = Σα` | strength | total evidence gathered |
| `b = e / S` | belief | share of support for each disease |
| `u = K / S` | **ignorance** | **how much we do not know** |

Read `u` as the headline number:

- `S` large (lots of evidence) → `u` **small** → we know something
- `S` small (no evidence) → `u` → **1.0** → we know nothing

`u = 1.0` is the **vacuous opinion**: total ignorance. Remember this, Part 4 needs it.

Belief plus ignorance always totals 1: `Σb + u = 1`.

**Code:** [src/evidence/evidential.py](../src/evidence/evidential.py) · **Paper:** Sensoy,
Kaplan & Kandemir, NeurIPS 2018

---

## Part 3 — Concept 2: Multimodal (several kinds of input)

"Modality" just means a kind of information. We have four:

```
   ┌──────────────────────┬───────────────────────────────┬──────────┐
   │ Modality             │ What it is                    │ Strength │
   ├──────────────────────┼───────────────────────────────┼──────────┤
   │ 1. Clinical image    │ ordinary close-up photo       │ 0.477    │
   │ 2. Dermoscopy image  │ special magnifying lens photo │ (needs   │
   │                      │                               │ training)│
   │ 3. Metadata          │ age, sex, body site, symptoms │ 0.648 ★  │
   │ 4. Text              │ the patient's own description │ 0.414    │
   └──────────────────────┴───────────────────────────────┴──────────┘
     ★ strongest single modality — plain facts beat photos here, because
       our image encoder is not trained yet (see 09_what_you_need_to_do.md)
```

Numbers are measured macro-F1 on PAD-UFES-20. **Macro-F1** averages the score of every
disease equally, so a model that only gets the common disease right scores badly. With 48%
of MILK10k being one disease (BCC), plain accuracy would be misleading — this is why we
never quote it.

### 3.1 The problem: more inputs made it worse

Measured on real data, three seeds:

```
  image only                0.477
  metadata only             0.648
  text only                 0.414

  image + metadata          0.653   ◄── BEST
  image + text              0.559
  text + metadata           0.587

  ALL THREE                 0.633   ◄── WORSE than the best pair!
```

**Adding information made the system worse.** This is the single fact that motivates the
whole project.

Why? The standard approach is **concatenation** — glue all the features into one long
vector and feed it to a network:

```
   [ image 2048 ][ text 768 ][ meta 114 ]  ─►  network  ─►  answer
```

The text channel is nearly useless (0.414, barely above guessing). But concatenation
gives it a fixed slot **every single time**. The network cannot say "ignore text on this
case" — the slot is always there, always contributing noise.

A human designer can notice this and delete text. **A deployed system cannot** — on *some*
patients the narrative is the only informative thing. We need something that decides
**per case**, automatically.

---

## Part 4 — Concept 3: Dempster–Shafer (combining opinions properly)

This is the heart of the project.

### 4.1 The idea

Each modality forms its own **opinion** — its own belief + ignorance from Part 2. Then we
combine the opinions using a rule from the 1960s called **Dempster–Shafer combination**.

```
   clinical image ──► opinion:  b=[BCC .7, MEL .1, ...]  u=0.15
   metadata       ──► opinion:  b=[BCC .6, MEL .2, ...]  u=0.18
   text           ──► opinion:  b=[BCC .02, MEL .01,...] u=0.95  ← knows nothing
                                        │
                                        ▼
                             Dempster–Shafer combine
                                        │
                                        ▼
                      final belief + ignorance + CONFLICT
```

### 4.2 Property 1: weak modalities silence themselves

Text has `u = 0.95` — almost total ignorance. In the DS formula, an opinion with high
ignorance **barely moves the result**. Nobody wrote a rule saying "distrust text." It
follows from the algebra.

> **This is why the design works.** The 0.633-worse-than-0.653 problem disappears, because
> a modality with no evidence has no influence — automatically, per case.

### 4.3 Property 2: missing modalities vanish for free

A missing modality gets the vacuous opinion (`b = 0`, `u = 1`) from Part 2.

In DS combination, the vacuous opinion is the **identity element** — like multiplying by
1, or adding 0. Combining with it changes **nothing, exactly**.

```
   anything  ⊕  (totally ignorant)  =  anything, unchanged
```

So "patient has no dermoscopy photo" needs no special code path, no imputation, no
zero-filling. It is **algebraically** a no-op. Compare the usual approach — mask and
renormalise — which needs separate handling for all 15 combinations of 4 present/absent
modalities, each a chance to introduce a bug.

### 4.4 Property 3: conflict ≠ ignorance (the key insight)

DS gives us a third number softmax has no equivalent for: **conflict mass `C`**, how much
the sources actively *disagree*.

Two situations that look identical to older methods:

```
  CASE A — real disagreement          CASE B — shared ignorance
  image says: MEL, confident          image says:  "no idea"  (u high)
  meta  says: NEV, confident          meta  says:  "no idea"  (u high)

  → conflict C HIGH, ignorance LOW    → conflict C LOW, ignorance HIGH
  → the sources contradict            → nobody knows anything
  → RIGHT ACTION: defer to a human    → RIGHT ACTION: get more evidence
         DEFER                                  ACQUIRE
```

**These need opposite responses.** The older method in this project used *JS divergence*
(a measure of how different two distributions are). JS **cannot** tell A from B: in case B
both are near-uniform, so they look *similar*, and JS reports "consistent" — exactly
backwards.

DS separates them because it tracks ignorance and conflict as **separate numbers**.

`C = Σ_{i≠j} b₁ᵢ · b₂ⱼ` — "how much belief did source 1 put on one disease while source 2
put belief on a different one."

**Code:** [src/evidence/evidential.py](../src/evidence/evidential.py) `ds_combine()` ·
**Paper:** Han et al., ICLR 2021 (Trusted Multi-View Classification)

---

## Part 5 — Concept 4: Conformal Prediction (sets instead of single answers)

### 5.1 The idea

Instead of one answer, output a **set** of answers guaranteed to contain the truth a
chosen percentage of the time.

```
  easy case  →  { BCC }                      small set = confident
  hard case  →  { BCC, SCC, AKIEC }          big set   = uncertain
```

Set at 90%: **the true disease is in the set for 90% of patients.** That is a measured,
distribution-free guarantee — no assumption about the network being well-calibrated.

### 5.2 How (three steps)

1. Hold out a **calibration set** the model never trained on.
2. For each calibration case, compute a **non-conformity score** — how badly the model
   ranked the true answer.
3. Take the 90th percentile of those scores, call it `q̂`. At test time, include every
   disease scoring below `q̂`.

The exact percentile uses `⌈(n+1)(1−α)⌉/n`, a small correction for having finite
calibration data.

### 5.3 Why we use it: catching the dangerous cases

Our actual measured result. Of 464 test cases, **210 produced a set containing both a
benign and a malignant disease**:

```
   set = { NEV (benign), MEL (malignant) }
          └─ "this might be a harmless mole, or might be melanoma"
```

That is **exactly** the case you must not answer. And max-softmax confidence **cannot see
it** — it might report 0.85 confidence for NEV while the set reveals melanoma is still
live. Our gate catches all 210.

**Code:** [src/calibration/conformal.py](../src/calibration/conformal.py)

---

## Part 6 — Concept 5: Learn-then-Test (turning a threshold into a promise)

### 6.1 The problem with hand-picked thresholds

The old system said: "answer if confidence > 0.80." Where did 0.80 come from? Someone
chose it. There is no promise attached.

### 6.2 The fix

**Learn-then-Test (LTT)** treats threshold choice as **statistical hypothesis testing**.
For each candidate threshold, ask: *"can I reject the hypothesis that this threshold has
unacceptable risk?"* Only thresholds that survive are allowed.

The output is a real guarantee:

```
   P( risk(λ̂) ≤ α )  ≥  1 − δ
   └─ probability that the true miss rate is under our target
      is at least 90%, over the randomness of the calibration draw
```

With `α = 0.05` (miss at most 5% of cancers) and `δ = 0.10` (90% sure).

### 6.3 It honestly refused

On PAD-UFES-20: 432 calibration cases, 9.95% of them malignant. **No threshold could be
certified.** The module reported `controlled = False` and **declined to return a
threshold**, rather than returning a confident-looking guess.

Why: with only 432 cases the statistical slack is 0.047, so the measured miss rate would
have to be below 0.31% — not achievable.

**On MILK10k this is fixed.** 1,048 calibration lesions → slack 0.033 → the bar is 1.69%,
and our measured rate is 0.50%. **This is why the calibration split was changed from 10%
to 20%** before anyone looked at test data.

**Code:** [src/calibration/ltt.py](../src/calibration/ltt.py) · **Paper:** Angelopoulos
et al., *Annals of Applied Statistics* 2025

---

## Part 7 — Concept 6: Cost-aware decisions

### 7.1 Mistakes are not equal

Telling a melanoma patient "you're fine" is far worse than telling a healthy person "let's
check." So we write the costs down explicitly:

```
   missing a cancer       10      ← by far the worst
   false alarm             2
   defer to a human        1.5    ← costs a doctor's time
   ask for dermoscopy      0.5    ← cheap
   correct answer          0
```

### 7.2 Pick the cheapest action

For each of the three actions, compute the **expected loss** (probability × cost, summed),
then take the smallest. This is **Chow's rule** (1970).

```
   E[loss | PROCEED]  = Σ  P(disease) × cost(saying our answer when truth is that disease)
   E[loss | ACQUIRE]  = 0.5  + expected loss after the new photo arrives
   E[loss | DEFER]    = 1.5

   action = whichever is smallest
```

**Honest weakness:** those five numbers are *declared, not derived*. But that is still
better than the old system, which hid the same judgement inside three unexplained
thresholds (0.80 / 0.50 / 0.30). Now it is one reviewable table, and we run a mandatory
sensitivity sweep over it. See [limitations.md](limitations.md).

**Code:** [src/decision/policy.py](../src/decision/policy.py)

---

## Part 8 — Concept 7: Measuring a system that can refuse

### 8.1 Accuracy is the wrong metric

A system that refuses every case has **100% accuracy on the cases it answered**. Useless.
We need to measure accuracy **and** how often it answers.

- **Coverage** — the fraction of cases it answered.
- **Selective risk** — error rate among answered cases.

### 8.2 AUGRC — our primary metric

Plot error against coverage, sweeping the threshold, and measure the area underneath. The
older version of this is **AURC**; we use **AUGRC**.

The difference is the denominator. AURC divides errors by the number of *covered* cases;
AUGRC divides by **all** cases:

```
   GR(τ) = (1/N) Σ loss_i · 1[confidence_i ≥ τ]
                  └─ divided by ALL N cases, not just the covered ones
```

Why it matters: AURC over-weights mistakes made at very high confidence and **changes
which model looks best on 5 of 6 benchmark datasets**. An earlier version of our own plan
recommended AURC — that was our error, and it is recorded in
[what_did_not_work.md](what_did_not_work.md).

AUGRC is bounded in `[0, ½]`. Lower is better.

**Code:** [src/eval/selective.py](../src/eval/selective.py) · **Paper:** Traub et al.,
NeurIPS 2024 (spotlight)

---

## Part 9 — Putting it together

```
   STEP 1    Four inputs
             clinical photo · dermoscopy photo · metadata · patient text
                 │
   STEP 2    Encode each one separately
             each modality gets its OWN network and its OWN evidential head
                 │
   STEP 3    Each produces an opinion          (Part 2)
             belief b  +  ignorance u
             missing modality → vacuous (b=0, u=1)
                 │
   STEP 4    Dempster–Shafer combine           (Part 4)
             weak modalities silence themselves
             missing ones vanish by algebra
             → final belief, ignorance u, CONFLICT C
                 │
   STEP 5    Safety checks
             ├─ conformal set spans benign AND malignant?  → DEFER   (Part 5)
             ├─ conflict C above calibrated threshold?      → DEFER   (Part 4)
             └─ LTT threshold certified?                    (Part 6)
                 │
   STEP 6    Cheapest action                   (Part 7)
             PROCEED        ACQUIRE               DEFER
             answer now     ask for dermoscopy    send to human
                 │
   STEP 7    Write the report
             LLM writes prose; a verifier checks every claim against the
             evidence. Similar past cases are shown for context but
             CANNOT change the decision. (Part 10)
```

---

## Part 10 — Concept 8: RAG (showing similar past cases)

**RAG** = Retrieval-Augmented Generation. Find similar past patients and show them
alongside the answer, so a doctor can judge for themselves.

How: turn every training case into a vector, then find the 8 nearest vectors to the new
case using **cosine similarity** (the angle between vectors — 1.0 = identical direction).

```
   new case ──► vector ──► search 1,403 past cases ──► 8 nearest
                                                          │
                            "SEK 0.81, BCC 0.70, ACK 0.70, SCC 0.69..."
                                                          │
                                                   shown in the report
```

### The rule that matters: retrieval is explanation ONLY

It **never** touches the decision. This is a requirement, not caution.

The guarantees in Parts 5 and 6 assume **exchangeability** — calibration and test data
drawn alike. A corpus that changes between queries breaks that, and the promise is void.

So the index holds **training cases only**, and `--verify` proves no calibration or test
case can be retrieved.

> **Why this is strict:** if a test-split neighbour were retrieved, it would carry its
> **true label into the written report**. No metric would ever show it, because the leak
> travels through *prose*, not numbers. That is the hardest kind of leak to catch later,
> so it is blocked structurally.

**Code:** [scripts/build_rag_index.py](../scripts/build_rag_index.py) · **Detail:**
[10_retrieval.md](10_retrieval.md)

---

## Part 11 — Concept 9: The agent pipeline

Seven specialised components, each with one job:

```
                          PatientCase
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
   Vision Team            Scribe Agent          Librarian Agent
   runs 7 specialist      reads text and        finds 8 similar
   models (A–G)           metadata quality      past cases
        │                      │                      │
        └──────────┬───────────┴──────────┬───────────┘
                   ▼                      │
            Devil's Advocate              │ explanation only
            audits each modality:         │ (may fail without
            quality, conflict, gaps       │  harming diagnosis)
                   │                      │
                   ▼                      │
            EADE decision engine          │
            DS → conformal → LTT → cost   │
                   │                      │
                   ▼                      │
          PROCEED / ACQUIRE / DEFER       │
                   │                      │
                   ▼                      │
            Report generator  ◄───────────┘
                   │
                   ▼
            Stage-2 verifier
            rejects claims the evidence does not support;
            one conservative retry, then escalate
```

The LLM **writes**; it does not diagnose. With no LLM running, the system emits a
deterministic summary and keeps working.

---

## Part 12 — Reading order for the rest of the docs

| Order | Doc | Why |
|---|---|---|
| 1 | [evaluation_integrity.md](evaluation_integrity.md) | **Most important.** Why no number from before 2026-10-11 can be trusted |
| 2 | [01_project_overview.md](01_project_overview.md) | The project in context |
| 3 | [12_system_walkthrough.md](12_system_walkthrough.md) | One case, traced end to end with real numbers |
| 4 | [04_methods.md](04_methods.md) | Every formula derived properly |
| 5 | [02_research_design.md](02_research_design.md) | Hypotheses H1–H5, what is novel and what is not |
| 6 | [05_experimental_protocol.md](05_experimental_protocol.md) | Datasets, splits, baselines |
| 7 | [10_retrieval.md](10_retrieval.md) | RAG and the agents |
| 8 | [what_did_not_work.md](what_did_not_work.md) | **Read this.** Everything that failed |
| 9 | [09_what_you_need_to_do.md](09_what_you_need_to_do.md) | What is left |
| 10 | [13_panel_guide.md](13_panel_guide.md) | Presenting and defending it |

---

## Part 13 — Check your understanding

If you can answer these, you understand the project:

1. Why can't softmax say "I don't know"?
2. What is the difference between aleatoric and epistemic uncertainty, and why does it
   change which **action** we take?
3. Why does adding the text modality make naive fusion **worse**?
4. What does `u = 1` mean, and why does combining with it change nothing?
5. Give a case where conflict is high but ignorance is low. What should the system do?
6. Why is a conformal set containing both a benign and a malignant disease dangerous?
7. Why did Learn-then-Test refuse to return a threshold on PAD-UFES-20?
8. Why is AUGRC preferred over AURC?
9. Why must retrieval never influence the decision?
10. At 71.7% malignant prevalence, why is "0% miss rate" almost meaningless on its own?

Answers, in order: Part 1.2 · 2.1 · 3.1 · 2.3 and 4.3 · 4.4 · 5.3 · 6.3 · 8.2 · 10 ·
[limitations.md](limitations.md).
