# System Walkthrough — One Case, End to End

Follow a single patient through the whole system, with the real numbers the code actually
produces. Read [11_learning_path.md](11_learning_path.md) first if any concept here is
unfamiliar.

---

## The whole system on one page

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  INPUT                                                                       ║
║                                                                              ║
║   clinical photo    dermoscopy photo    metadata          patient text        ║
║   (smartphone)      (special lens)      age/sex/site      "itchy, growing"    ║
║        │                  │                  │                  │            ║
╚════════╪══════════════════╪══════════════════╪══════════════════╪════════════╝
         │                  │                  │                  │
╔════════╪══════════════════╪══════════════════╪══════════════════╪════════════╗
║  STAGE 1 — ENCODE   (each modality separately; no mixing yet)                ║
║        ▼                  ▼                  ▼                  ▼            ║
║    ResNet50 /         ResNet50 /         residual MLP       Bio_ClinicalBERT  ║
║    DermLIP            DermLIP                                                 ║
║        │                  │                  │                  │            ║
║     2048-d            2048-d              114-d             768-d            ║
╚════════╪══════════════════╪══════════════════╪══════════════════╪════════════╝
         │                  │                  │                  │
╔════════╪══════════════════╪══════════════════╪══════════════════╪════════════╗
║  STAGE 2 — EVIDENCE   (one evidential head per modality)                     ║
║        ▼                  ▼                  ▼                  ▼            ║
║   evidence e ≥ 0     evidence e ≥ 0     evidence e ≥ 0    evidence e ≥ 0     ║
║   α = e+1            α = e+1            α = e+1           α = e+1            ║
║   b = e/S            b = e/S            b = e/S           b = e/S            ║
║   u = K/S            u = K/S            u = K/S           u = K/S            ║
║        │                  │                  │                  │            ║
║   ┌────┴──────────────────┴──────────────────┴──────────────────┴──────┐     ║
║   │ MISSING MODALITY → vacuous opinion (b=0, u=1) = DS IDENTITY        │     ║
║   │ no imputation, no zero-filling, no special code path               │     ║
║   └────────────────────────────┬───────────────────────────────────────┘     ║
╚════════════════════════════════╪═════════════════════════════════════════════╝
                                 │
╔════════════════════════════════╪═════════════════════════════════════════════╗
║  STAGE 3 — DEMPSTER–SHAFER FUSION                                            ║
║                                 ▼                                            ║
║         belief b        ignorance u        conflict C                        ║
║      "what we think"   "what we don't    "how much the sources               ║
║                          know"             disagree"                         ║
║                                                                              ║
║   high u, low C  → nobody knows        → get more evidence                   ║
║   low u, high C  → sources contradict  → send to a human                     ║
╚════════════════════════════════╪═════════════════════════════════════════════╝
                                 │
╔════════════════════════════════╪═════════════════════════════════════════════╗
║  STAGE 4 — SAFETY GATES   (any one can veto answering)                       ║
║                                 ▼                                            ║
║   GATE 1  conformal set spans benign AND malignant?   → DEFER                ║
║   GATE 2  conflict C > calibrated p90?                 → DEFER                ║
║   GATE 3  prediction set empty?                        → DEFER                ║
║   GATE 4  LTT-certified threshold λ̂ not met?           → DEFER                ║
║                                                                              ║
║   Gates only ever INCREASE caution. None can turn a DEFER into a PROCEED.    ║
╚════════════════════════════════╪═════════════════════════════════════════════╝
                                 │
╔════════════════════════════════╪═════════════════════════════════════════════╗
║  STAGE 5 — COST-AWARE CHOICE   (Chow's rule: pick the cheapest)              ║
║                                 ▼                                            ║
║     E[PROCEED]           E[ACQUIRE]            E[DEFER]                      ║
║     Σ P(d)·cost(d)       0.5 + E[after]        1.5                           ║
║          │                    │                   │                         ║
║          ▼                    ▼                   ▼                         ║
║      PROCEED              ACQUIRE              DEFER                         ║
║    answer now        ask for dermoscopy     send to human                    ║
╚════════════════════════════════╪═════════════════════════════════════════════╝
                                 │
╔════════════════════════════════╪═════════════════════════════════════════════╗
║  STAGE 6 — REPORT                                                            ║
║                                 ▼                                            ║
║   Librarian: 8 similar past cases ──┐   (TRAINING SPLIT ONLY;                ║
║                                     │    explanation only; cannot            ║
║                                     ▼    change the decision)                ║
║                          LLM writes the narrative                            ║
║                                     │                                        ║
║                                     ▼                                        ║
║                          Stage-2 verifier                                    ║
║                   every claim checked against the evidence;                  ║
║                   one conservative retry, then escalate                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## Now the real trace

Run it yourself:

```bash
python scripts/demo.py
```

### Input

```
clinical photo : data/raw/images/PAT_8_15_820.png
metadata       : age 55, female, forearm, itches, has grown, does not bleed
patient text   : "A rough patch on my forearm that has grown over two months."
dermoscopy     : NOT AVAILABLE   ← PAD-UFES-20 has no dermoscopy at all
```

### Stage 1–2 — encode, then form opinions

Each modality goes to its own encoder and its own head. Dermoscopy is absent, so it gets
the **vacuous opinion** `(b = 0, u = 1)`. No placeholder image is invented.

### Stage 3 — fusion

Measured fusion weights on this case:

```
   metadata        0.414    ← trusted most
   clinical_image  0.301
   text            0.285
   dermoscopy      —        absent, vanished by algebra
```

Metadata dominates. Nobody coded that preference — metadata produced the most evidence,
so it has the lowest `u`, so DS gives it the most influence. On a different case with a
sharp diagnostic image, these weights move on their own.

### Stage 4 — gates

```
   conformal set      { BCC }          single class, benign/malignant not spanned → pass
   conflict C          below p90                                                  → pass
   set non-empty       yes                                                        → pass
```

### Stage 5 — decision

```
   action      : ACCEPT (consensus level 1)
   diagnosis   : BCC — Basal Cell Carcinoma
   confidence  : 0.945
   specialists : A=BCC B=BCC C=SCC D=BCC E=BCC F=BCC G=BCC    (6 of 7 agree)
```

Specialist C disagrees. The system records that rather than hiding it — and because 6 of 7
agree with low conflict, it still answers.

### Stage 6 — report

```
   retrieval   : 8 similar cases, agreement 0.250
                 SEK@0.813, BCC@0.701, ACK@0.699, SCC@0.688, BCC@0.681
```

Two of eight neighbours are BCC, matching the answer → agreement 0.250. This is **shown,
not used**. Final report:

```
   Consensus Action: ACCEPT (level 1)
   Predicted Diagnosis: BCC - Basal Cell Carcinoma (confidence: 94.5%)
   Specialist agreement: BCC=6/7, SCC=1/7
   Evidence weighting: metadata 41%, clinical_image 30%, text 28%
   Cross-modal consistency: 80%
   Missing evidence: dermoscopy_image
   Research prototype - not clinically validated.
```

With no LLM running, this deterministic summary is produced instead of generated prose —
and the pipeline keeps working.

---

## The aggregate result

One case shows the mechanism. This shows whether it helps. From
`python scripts/run_eade.py`, over 464 test cases with 432 held out for calibration:

| | coverage | **malignancy miss rate** | AUGRC |
|---|---|---|---|
| **D-GUARD** | 0.433 | **0.0050** | 0.1634 |
| max-softmax abstention, *same coverage* | 0.433 | 0.0597 | **0.1276** |

**A 12× lower miss rate while answering exactly as many cases.** The driver is Gate 1:
**210 of 464** cases produced a prediction set containing both a benign and a malignant
disease — which confidence alone cannot detect.

Where the cases went:

```
   PROCEED   201   43.3%
   ACQUIRE   132   28.4%     "send me a dermoscopy photo"
   DEFER     131   28.2%

   reasons:  AMBIGUOUS_SET 210 · CONFLICT 41 · COST 12
```

### Reported against us

**The baseline wins on AUGRC** (0.1276 vs our 0.1634). The ignorance score `1−u` is a
*worse ranking signal* than max-softmax, even though the policy built on it is far safer.
Both numbers belong in the table — see
[what_did_not_work.md](what_did_not_work.md) §7.

Conformal coverage landed at **0.8944** against a 0.90 target. LTT reported **not
controlled** at α=0.05 and said so rather than inventing a threshold.

### What this does and does not show

✅ The machinery works: evidential fusion, conformal gating, cost-aware routing, distinct
deferral reasons, honest failure reporting.

❌ **Not a generalisation estimate.** The legacy specialists were trained on data
overlapping this test split. These numbers verify the *mechanism*, not accuracy. Closing
that gap needs the GPU run in [09_what_you_need_to_do.md](09_what_you_need_to_do.md).

---

## Two modes: bridge vs native

The system runs the same decision logic either way.

| | **Bridge mode** (legacy PAD-UFES models) | **Native mode** (MILK10k model) |
|---|---|---|
| Opinions from | softmax, lifted via `u = H(p)/log K` | trained Dirichlet heads |
| Separates aleatoric/epistemic? | **no** — entropy conflates them | **yes** |
| Trained? | legacy weights | ✅ Kaggle, seed 42 |
| Runs in | `scripts/demo.py`, `run_eade.py` | `scripts/demo_care.py`, `run_eade_milk.py`, **the app** |
| Status | ✅ running and verified | ✅ **running and verified** |

### A native-mode trace (real held-out MILK10k lesion IL_0449677, truth NV)

From `python scripts/demo_care.py`:

```
STAGE A  photo + details
  ACTION   : ACQUIRE   (urgency URGENT)
  tentative: NV   P(malignant) 43%   ignorance u 0.836   conflict C 0.007
  dynamic fusion weights: {clinical photo: 0.588, patient details: 0.412}
  each source alone:      {clinical photo: NV, patient details: BCC}     <- they disagree
  patient sees: "please provide a dermoscopy photo ... Early, unconfirmed impression: NV"

STAGE B  + the real dermoscopy image
  ACTION   : PROCEED   (urgency ROUTINE)
  tentative: NV   P(malignant) 24%   ignorance u 0.463   conflict C 0.036
  dynamic fusion weights: {clinical photo: 0.16, dermoscopy: 0.727, patient details: 0.112}
  patient sees: "Result: the system's assessment is Melanocytic Nevus (NV) ..."
```

Ignorance falls from 0.836 to 0.463, the weight moves to dermoscopy **on its own**, and
the answer is correct. That is the whole project in one case.

Bridge mode exists so the decision layer could be built and tested before any GPU was
available. **It is an approximation and is labelled as one everywhere** — normalised
entropy mixes the two uncertainty types that the whole design depends on separating. Every
bridge-mode number in these docs carries that caveat.

---

## Where each stage lives in the code

| Stage | File |
|---|---|
| Encode | [src/models/backends.py](../src/models/backends.py) |
| Evidential heads, DS fusion | [src/evidence/evidential.py](../src/evidence/evidential.py) |
| Conformal sets | [src/calibration/conformal.py](../src/calibration/conformal.py) |
| LTT risk control | [src/calibration/ltt.py](../src/calibration/ltt.py) |
| Three-action policy | [src/decision/policy.py](../src/decision/policy.py) |
| Metrics (AUGRC etc.) | [src/eval/selective.py](../src/eval/selective.py) |
| Decision engine | [src/agents/eade.py](../src/agents/eade.py) |
| Agent pipeline | [src/agents/orchestrator.py](../src/agents/orchestrator.py) |
| Retrieval | [src/retrieval/rag_index.py](../src/retrieval/rag_index.py) |
| End-to-end run | [scripts/run_eade.py](../scripts/run_eade.py) |

---

## Robustness: DERMA-STRESS

A decision system must be tested on *broken* inputs, not just clean ones. 18 conditions
across 7 families:

```
   A  image degradation      blur · noise · brightness · JPEG · occlusion (×3 severities)
   B  missing evidence       drop dermoscopy / metadata / text / combinations
   C  contradictory context  permuted body site · shifted age · conflicting symptom text
   D  distribution shift     test on PAD-UFES-20 after training on MILK10k
   E  leave-class-out OOD    remove a disease from training — does it defer on it?
   F  combined failures      degraded image + missing metadata, etc.
   G  operational            timeout · worker restart · corrupt upload · LLM unavailable
```

Family **F** is the one most papers skip and real deployments hit first. Family **C** is
where the conflict-vs-ignorance distinction earns its place: a contradiction should raise
`C`, not `u`.

```bash
python scripts/run_stress.py --limit 50
```

**Note:** centre-held-out shift is *not* possible — the MILK10k release publishes no centre
column. That corrects an earlier plan of ours; see
[what_did_not_work.md](what_did_not_work.md) §10.
