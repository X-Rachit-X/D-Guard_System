# Study Plan — Learn the Whole Project in 10 Days

A day-by-day plan for understanding D-GUARD well enough to explain and defend every part.
Each day has: **read** (docs), **run** (commands), **open** (code), **check** (questions you
should be able to answer). About 2–3 hours a day.

Concepts are explained in [11_learning_path.md](11_learning_path.md); this plan tells you
*when* to read each part and how to connect it to the code.

---

## The map you are learning

```
                         ┌─────────────────────────────────────────┐
  LAYER 4  PLATFORM      │ app/dguard_app.py   src/care/            │  patients, doctors,
                         │ instructions, follow-ups, audit log      │  follow-ups
                         └────────────────────┬────────────────────┘
                         ┌────────────────────┴────────────────────┐
  LAYER 3  DECISION      │ src/decision/  src/calibration/          │  PROCEED / ACQUIRE /
           (EADE)        │ policy, conformal sets, LTT, explain     │  DEFER + guarantee
                         └────────────────────┬────────────────────┘
                         ┌────────────────────┴────────────────────┐
  LAYER 2  EVIDENCE      │ src/evidence/  src/agents/  src/retrieval│  belief, ignorance,
           & AGENTS      │ DS fusion, specialists, librarian (RAG)  │  conflict, reports
                         └────────────────────┬────────────────────┘
                         ┌────────────────────┴────────────────────┐
  LAYER 1  MODELS        │ notebooks/kaggle_train_dguard.py         │  encoders + evidential
           & DATA        │ src/native/  src/data/milk10k.py         │  heads, MILK10k
                         └─────────────────────────────────────────┘
  ACROSS ALL LAYERS      src/eval/ (AUGRC, AE)  src/stress/ (DERMA-STRESS)  tests/
```

Learn bottom-up: you cannot understand a decision until you understand the numbers it is
made from.

---

## Day 1 — The problem and the honesty baseline

**Read**
- [11_learning_path.md](11_learning_path.md) Parts 0–1 (why softmax cannot say "I don't know")
- [evaluation_integrity.md](evaluation_integrity.md) — why old numbers are invalid
- [01_project_overview.md](01_project_overview.md)

**Run**
```bash
pip install -e ".[dev]"
pytest tests -q                    # expect 470+ passed
```

**Check**
1. In one sentence, what does D-GUARD decide?
2. Name the three actions and when each is used.
3. Name two ways the original training code produced invalid metrics.

---

## Day 2 — The data

**Read** — [05_experimental_protocol.md](05_experimental_protocol.md) §1 (datasets)

**Run**
```bash
python scripts/milk10k_audit.py
```

**Open** — `src/data/milk10k.py` (`load_milk10k`, `lesion_level_split`), `src/constants.py`
(the `MILK_*` block)

**Understand**
- Every lesion has **two images**: clinical close-up + dermoscopy. That is why the
  acquisition question can be measured.
- 11 classes, **71.7% malignant** — a referral population.
- Splits are by **lesion**, not image, so both images of a lesion stay together.
- Why the split uses SHA-256 hashing instead of Python's `hash()` (salted per process).

**Check**
1. Why is "0% missed cancers" nearly meaningless on this dataset?
2. Why can't we run a centre-shift experiment?
3. Which classes cannot get a class-conditional guarantee, and why?

---

## Day 3 — Evidential learning (belief and ignorance)

**Read** — [11_learning_path.md](11_learning_path.md) Part 2

**Open**
- `src/evidence/evidential.py`: `Opinion`, `opinion_from_evidence`, `EvidentialHead`, `edl_loss`
- `notebooks/kaggle_train_dguard.py`: `EvidentialHead`, `edl_loss`

**Try** — in Python:
```python
import torch
from src.evidence.evidential import opinion_from_evidence
lots = opinion_from_evidence(torch.tensor([[40., 5., 3.]]))
none = opinion_from_evidence(torch.tensor([[0.1, 0.1, 0.1]]))
print(lots.u, none.u)        # small vs close to 1
```

**Check**
1. Write `α`, `S`, `b`, `u` from memory.
2. What does `u = 1` mean? Why is it called the vacuous opinion?
3. Aleatoric vs epistemic: which one does acquiring dermoscopy help with?

---

## Day 4 — Fusion (Dempster–Shafer) and "dynamic" weights

**Read** — [11_learning_path.md](11_learning_path.md) Parts 3–4,
[14_results.md](14_results.md) §4 and §6

**Open**
- `src/evidence/evidential.py`: `ds_combine`, `fold_opinions`
- `src/native/engine.py`: the fold in `assess()` and the `weights` computation
- `src/fusion/adaptive_fusion.py`: the **old** iterative, diagnosis-adaptive fusion, ported

**Understand**
- Combining with the vacuous opinion changes nothing → missing modalities need no special code.
- Conflict `C` is carried as the **max over the fold**. The Kaggle export once reported only
  the last step and got `C = 0` everywhere ([what_did_not_work.md](what_did_not_work.md) §16).
- **Dynamic fusion weights** `w_m = (1 − u_m) / Σ(1 − u_m)`: each source's share of the
  evidence for *this* case.
- The old system's hand-typed "diagnosis prior" vs the measured reliability table.

**Check**
1. Give an example where conflict is high and ignorance is low. What should happen?
2. Why can't JS divergence tell "they disagree" from "nobody knows"?
3. H1 failed. What does that mean, and what was the gap?

---

## Day 5 — Conformal prediction

**Read** — [11_learning_path.md](11_learning_path.md) Part 5,
[14_results.md](14_results.md) §5.3

**Open**
- `src/calibration/conformal.py` (APS/RAPS, 11-class)
- `src/calibration/binary.py` (LAC, benign/malignant — what the system uses)

**Understand**
- A set guaranteed to contain the truth (1 − α) of the time.
- Why the gate uses a **benign/malignant** set, not an 11-class set.
- Why APS collapsed to "everything in every set" on two classes, and why LAC does not.
- Mondrian = one threshold per outcome, so the guarantee holds for malignant lesions specifically.

**Check**
1. What does `⌈(n+1)(1−α)⌉/n` correct for?
2. What is "false reassurance", and which gate setting blocks it?

---

## Day 6 — Learn-then-Test and the cost matrix

**Read** — [11_learning_path.md](11_learning_path.md) Parts 6–7,
[04_methods.md](04_methods.md)

**Open**
- `src/calibration/ltt.py`: `calibrate_threshold`, `hoeffding_bentkus_p_value`
- `src/decision/policy.py`: `CostMatrix`, `ThreeActionPolicy.decide`, `label_rule`, `ambiguity_gate`

**Understand**
- LTT gives `P(risk ≤ α) ≥ 1 − δ` — and returns "not controlled" instead of guessing.
- Why PAD-UFES (432 calibration cases) could not be certified and MILK10k can.
- Chow's rule: pick the cheapest of PROCEED / ACQUIRE / DEFER.
- Why stating the **lowest-cost** label called 98.5% of lesions malignant, and why the
  system now states the **most likely** label.

**Check**
1. What does δ mean in plain English?
2. Recite the cost matrix. Which value is a judgement call, and what changes if it moves?

---

## Day 7 — The decision engine end to end

**Read** — [12_system_walkthrough.md](12_system_walkthrough.md), [14_results.md](14_results.md) §5

**Run**
```bash
python scripts/milk_native_infer.py
python scripts/run_eade_milk.py --missed-cost 5 --out outputs/eade_milk/cost_5
```

**Open** — `scripts/run_eade_milk.py` (two-stage protocol, baselines), `src/decision/explain.py`

**Understand**
- Stage A (photo + details) → maybe ACQUIRE → Stage B (+ dermoscopy).
- The **tentative opinion** on every decision, and why `prediction` stays empty unless PROCEED.
- The baselines B1–B5 and why matched coverage is the fair comparison.
- Acquisition efficiency: 1 = oracle, 0 = random.

**Check**
1. Quote the headline result with its condition.
2. Why does "always answer" have lower mean cost, and why is that not the right comparison?

---

## Day 8 — Agents, RAG and the report

**Read** — [10_retrieval.md](10_retrieval.md), [03_architecture.md](03_architecture.md)

**Run**
```bash
python scripts/build_rag_index.py --verify
python scripts/demo.py
```

**Open** — `src/agents/orchestrator.py`, `src/agents/librarian.py`,
`src/retrieval/rag_index.py`, `src/llm/verification.py`

**Understand**
- The seven agents and which ones may affect the diagnosis.
- Retrieval is **explanation only**, built from training cases only — and why a leak there
  would be invisible.
- The stage-2 verifier rejects report claims the evidence does not support.

**Check**
1. Why must retrieval never influence the decision?
2. What happens when no LLM is available?

---

## Day 9 — The care platform and the demo

**Read** — [13_panel_guide.md](13_panel_guide.md) §6

**Run**
```bash
python scripts/demo_care.py
streamlit run app/dguard_app.py
```

**Open** — `src/care/service.py` (the workflow diagram at the top), `src/care/store.py`,
`src/native/engine.py`, `app/dguard_app.py`

**Do** — go through one full case as a patient, then as a doctor, then submit a follow-up.

**Check**
1. Where does a DEFER case go, and what does the patient see?
2. What triggers a follow-up change alert?
3. What does the audit log record for one assessment?

---

## Day 10 — Robustness, limits, and the defence

**Read** — [what_did_not_work.md](what_did_not_work.md), [limitations.md](limitations.md),
[13_panel_guide.md](13_panel_guide.md) §7–9, `paper/main.tex`

**Run**
```bash
python scripts/run_stress.py --limit 50
```

**Do** — answer every question in [13_panel_guide.md](13_panel_guide.md) §8 out loud,
without notes.

**Check**
1. Name three things that did not work.
2. Name three limitations you would state before being asked.

---

## Glossary

| Term | Meaning |
|---|---|
| **Modality** | A kind of input: clinical photo, dermoscopy, metadata, text |
| **Dermoscopy** | A photo through a lit magnifying lens that shows structures under the skin surface |
| **Evidence `e`** | Non-negative support a head finds for each class |
| **Belief `b`** | `e / S` — share of support for each class |
| **Ignorance `u`** | `K / S` — how much the model does not know; 1 = nothing |
| **Vacuous opinion** | `b = 0, u = 1`; the identity element of Dempster–Shafer combination |
| **Conflict `C`** | How much two sources put belief on *different* classes |
| **Pignistic probability** | `b + u/K` — belief with ignorance spread evenly; what the policy reads |
| **Dynamic fusion weight** | A source's share of the evidence for one case |
| **Conformal set** | A set of labels that contains the truth (1 − α) of the time |
| **LAC / APS** | Two ways to score how "surprising" the true label is in conformal prediction |
| **Mondrian** | Conformal calibration done separately per class/outcome |
| **LTT** | Learn-then-Test: choose a threshold by hypothesis testing, with an error bound |
| **α, δ** | Target risk; allowed probability that the guarantee fails |
| **Exchangeability** | Calibration and test data are drawn the same way — what the guarantees assume |
| **Coverage** | Fraction of cases the system answers alone |
| **Selective risk** | Error rate among answered cases |
| **AUGRC** | Area under the generalised risk–coverage curve; lower is better |
| **Acquisition efficiency** | How much of the possible benefit of extra evidence a policy captures (0 random, 1 oracle) |
| **Chow's rule** | Choose the action with the lowest expected cost |
| **Tentative opinion** | The system's best guess on a DEFER/ACQUIRE case, always labelled unconfirmed |
| **Bridge mode** | Running the decision layer over softmax models via an entropy approximation |
| **Native mode** | Running it over trained evidential heads (the MILK10k model) |
| **RAG** | Retrieval-augmented generation; here, showing similar past cases in the report |
| **Macro-F1 / balanced accuracy** | Metrics that weight every class equally, so a majority class cannot hide failures |
