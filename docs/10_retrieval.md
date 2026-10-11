# Retrieval (RAG) and the Agentic Pipeline

How similar-case retrieval works, why it is deliberately kept out of the decision, and
what the seven agents actually do. Updated 2026-10-11.

---

## 1. The one rule

**Retrieval is explanation only. It never enters the decision path.**

This is not caution, it is a requirement of the guarantee. Conformal prediction and
Learn-then-Test both assume the calibration and test data are **exchangeable**. A corpus
that grows, is re-ranked, or varies between inference calls breaks exchangeability, and
the risk bound `P(R(λ̂) ≤ α) ≥ 1−δ` stops holding. So the retrieved neighbours reach the
report generator and nothing else: no logit, no fusion weight, no threshold, and no
action depends on them.

Enforced in code at three points:

| Where | What it does |
|---|---|
| `scripts/build_rag_index.py` | indexes the **training split only**; writes the split name into the artifact and an audit trail of every excluded patient |
| `--verify` | asserts no calibration or test patient is in the index, and that a real query returns only training cases |
| `OrchestratorAgent.process_case` | the librarian stage is the **one** stage allowed to fail soft — losing it degrades the narrative, never the diagnosis |

### Why the leak would be invisible without this

A neighbour drawn from the test split carries its **true label** into the prose. The
metrics would not move, because nothing numeric changed — the leak flows through the
report a human reads. That is the hardest class of leak to detect after the fact, which is
why it is blocked structurally rather than checked for afterwards.

---

## 2. How it is built

```bash
python scripts/build_rag_index.py --verify
```

```
features   : 2298 rows
spec       : image_text = {clinical: 2048, text: 768} -> dim 2816
patients   : train 824  calibration 275  test 274
indexing   : 1403 training cases (895 calibration/test cases EXCLUDED)
class mix  : {BCC: 517, ACK: 437, NEV: 151, SEK: 158, SCC: 114, MEL: 26}
```

FAISS `IndexFlatIP` over L2-normalised vectors, so inner product **is** cosine
similarity. Exact search, no approximation — at 1,403 cases there is nothing to gain from
an approximate index and an exact one cannot silently degrade.

### What is indexed, and what is not

The index is built over the **raw pre-projection** features: ResNet50's 2048-d penultimate
layer and Bio_ClinicalBERT's 768-d `[CLS]`. Not the 256-d projections, for one reason —
the raw features reproduce at **cosine 1.000000** from `scripts/build_features.py`, and the
legacy projectors do not reproduce at all ([evaluation_integrity.md](evaluation_integrity.md)).
Indexing a space that cannot be regenerated would make the index unreproducible too.

**Metadata is excluded, and this costs something measurable.** The preprocessor emits
**116** columns fitted on the full CSV but **114** fitted on the training split only — two
one-hot columns correspond to categories absent from the training subset. The live
pipeline's `encode_metadata` returns 116. So a 114-wide index simply cannot be queried in
production, and padding it would misalign every column after the gap.

Measured consequence, on the same query case:

| index | neighbour agreement |
|---|---|
| image + text + metadata (2930-d, offline only) | **0.875** |
| image + text (2816-d, runtime-queryable) | **0.500** |

Metadata carries much of the retrieval signal — consistent with it being the strongest
single modality (0.648 macro-F1 against 0.477 for images). **The runtime index is the
weaker one**, and that is a real cost of making it queryable rather than a detail. Fixing
it properly means persisting the train-fitted preprocessor alongside the index; it is not
done yet and is recorded in [limitations.md](limitations.md).

`--spec raw3` builds the stronger 2930-d index for offline analysis. The script states
that it is not runtime-queryable rather than letting you discover it later.

### The spec cannot drift

The embedding spec is written into the artifact and re-checked on load. An index built in
one space and queried in another **raises**:

```
ValueError: index width 2816 disagrees with its saved embedding spec (2930);
refusing to serve queries in the wrong space
```

A nearest-neighbour search in the wrong space does not error on its own — it returns
confident nonsense. This turns that into a crash.

---

## 3. The agreement rate, and a bug worth knowing about

`retrieval_agreement` is the fraction of neighbours sharing the issued diagnosis. It is
computed **after** consensus, in `_retrieval_agreement`, not inside the Librarian.

The Librarian runs at step 3; the diagnosis does not exist until step 4. Its own
`agreement_rate` is therefore **structurally 0.0** on every real case — it is handed
`predicted_diagnosis=None` because there is nothing else to hand it. The first live run
printed `agreement 0.000` for exactly this reason, which looked like "retrieval finds
nothing useful" and was really "this number cannot be computed here".

`None` and `0.0` are now kept distinct: `None` means **not measured**, `0.0` means
**measured, nothing agreed**. A metric that cannot distinguish those two is worse than no
metric.

---

## 4. The seven agents

```
                        PatientCase
                             │
   ┌─────────────────────────┼─────────────────────────┐
   │                         │                         │
Vision Team            Scribe Agent            Librarian Agent
(specialists A–G)      (text + metadata)       (similar cases)
   │                         │                         │
   └─────────────┬───────────┴─────────────┬───────────┘
                 │                         │
         Devil's Advocate            (explanation only,
      (per-modality evidence          may fail soft)
       profile, conflict, gaps)
                 │
      ┌──────────┴──────────┐
      │                     │
 EADE engine          Legacy ladder
 (DS fusion →         (consensus levels
  conformal →          1–4, behind a flag)
  LTT → policy)
      │
      └──────────┬──────────┐
                 │          │
          PROCEED/ACQUIRE/DEFER
                 │
         Report generator ← similar cases
                 │
         Stage-2 verifier
     (claims checked against the
      evidence; one conservative
      retry, then escalate)
```

| Agent | Module | Role | Can it affect the diagnosis? |
|---|---|---|---|
| Vision Team | `agents/vision_team.py` | runs specialists A–G, emits logits + embeddings | **yes** |
| Scribe | `agents/scribe.py` | symptom extraction, metadata quality | **yes** |
| Librarian | `agents/librarian.py` | k=8 similar training cases | **no — by design** |
| Devil's Advocate | `agents/devils_advocate.py` | per-modality evidence profile, conflict, missingness | **yes** |
| EADE | `agents/eade.py` | evidential fold, conformal gate, cost-aware action | **yes** |
| Report generator | `llm/report_gen.py` | narrative from evidence + neighbours | no (prose only) |
| Stage-2 verifier | `llm/verification.py` | rejects claims the evidence does not support | **yes — can escalate** |

The LLM writes reports. It does not diagnose, and no diagnostic credit is attributed to
it. With no LLM reachable, the pipeline emits a deterministic evidence summary and keeps
working — verified, since the demo runs that way.

---

## 5. What was actually wrong before

`RAGIndex` and `LibrarianAgent` were both implemented and tested, and `faiss` was
installed. But:

1. **Nothing ever built an index.** No script referenced `RAGIndex`, so on every real case
   the Librarian returned `{"similar_cases": [], "note": "RAG index not loaded"}`.
2. **`build()` never constructed a Librarian**, so even a built index would not have been
   attached.
3. **The call site used the wrong embedding space** — the 256-d projections, where the
   index needed raw features.
4. **Retrieval never reached the caller.** `knowledge` went to the report generator and
   was dropped; `PipelineResult` had no field for it, so an explanation-only channel was
   not auditable.

All four are fixed, with 17 tests in `tests/test_rag_index_build.py`. The honest summary:
the retrieval component was **written and unit-tested but never connected**, which is a
failure mode worth naming — a green test suite over an unreachable code path.

---

## 6. Reproduce

```bash
python scripts/build_features.py --fit-metadata-on-train
python scripts/build_rag_index.py --verify
python scripts/demo.py                    # look for the `retrieval :` line
pytest tests/test_rag_index_build.py -q   # 17 passed
```

Expected demo output:

```
retrieval   : 8 similar cases, agreement 0.250
              SEK@0.813, BCC@0.701, ACK@0.699, SCC@0.688, BCC@0.681
```

---

## 7. Open

- Persist the train-fitted metadata preprocessor so the stronger 2930-d index becomes
  runtime-queryable (recovers agreement 0.875 vs 0.500).
- Rebuild on MILK10k after training — the current index is PAD-UFES-20, 6 classes.
- **Whether retrieval adds anything over the ignorance score `u` alone is untested.** It is
  listed as an open question in [what_did_not_work.md](what_did_not_work.md) and is not
  claimed as a contribution.
