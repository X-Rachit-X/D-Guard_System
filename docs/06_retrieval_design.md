# Similar-Case Retrieval and Case Grounding

Two retrieval channels, with a hard separation between them. One may influence the
prediction; the other may not.

---

## 1. Why retrieval at all

Three reasons it earns a place, rather than being decoration:

1. **Clinicians reason by precedent.** "This resembles the pigmented BCC we saw last
   month" is how dermatology is actually practised. A system that surfaces comparable
   cases fits the workflow.
2. **Neighbour agreement is an uncertainty signal.** If a case's nearest neighbours in
   embedding space carry mixed labels, the case sits near a decision boundary. That is
   usable information, and it is *complementary* to the model's own confidence — a model
   can be confidently wrong in a region where the training data disagrees.
3. **Explanations need evidence.** A generated report asserting "features consistent with
   BCC" is unfalsifiable. One that says "features consistent with BCC; 6 of 8 nearest
   training cases were BCC (ids …)" is checkable by a human.

---

## 2. Channel A — in-dataset kNN retrieval (may influence the prediction)

### Index construction

Per the audit fix already in the codebase (`proj.md` Part 10), index the **concatenation
of per-modality embeddings**, never a fused one:

```
v(x) = [ z_image (256) ; z_dermoscopy (256, zeros) ; z_metadata (114) ; z_text (256) ]
     → L2-normalise → FAISS IndexFlatIP      (inner product = cosine)
```

Indexing a *fused* embedding would be circular: fusion consumes retrieval, so retrieval
must not consume fusion. The current implementation gets this right.

**Fix needed:** blocks have very different scales and dimensionalities (114 vs 256), so a
single global L2 normalisation lets the image block dominate the distance. Normalise
**per block**, then concatenate with learned or tuned block weights `β_m`:

```
v(x) = [ β_img · ẑ_img ; β_meta · ẑ_meta ; β_text · ẑ_text ],   ẑ_m = z_m / ‖z_m‖
```

with `β` chosen on validation by retrieval precision@k. Absent modalities contribute a
zero block, and `β` is renormalised over present blocks so distances stay comparable
across availability patterns.

### Using the result correctly

Let the `k` neighbours have labels `y₁..y_k` and similarities `s₁..s_k`.

```
agreement(ŷ) = Σᵢ sᵢ · 1{yᵢ = ŷ} / Σᵢ sᵢ        (similarity-weighted)
neighbour entropy = H( similarity-weighted label distribution )
```

**Use these as conformity scores feeding the deferral rule, not as votes.**

The distinction is the whole design. If retrieval casts a vote, it becomes a weak
classifier whose errors compound with the model's — and because neighbours are drawn from
the *training* set, a vote imports training-set bias directly into the prediction. As a
*conformity score* it only affects whether the system answers, which is exactly where an
"are we in a well-supported region?" signal belongs:

```
s(x) = f( 1 − u(x),  agreement(ŷ),  −neighbour_entropy )
```

fitted on validation, then calibrated by Learn-then-Test
([04 §5](04_mathematical_foundations.md)). This way retrieval can only make the system
*more* cautious, never more confident in a wrong answer.

### Leakage rules

1. The index contains **training-split cases only**. Indexing val or test leaks.
2. **Patient-level exclusion:** when querying, drop neighbours sharing the query's
   `patient_id`. PAD-UFES-20 has multiple lesions per patient, so without this the
   top neighbour is frequently the same patient's other lesion — which inflates
   agreement and measures patient identity, not disease similarity.
3. Rebuild the index per seed, since the train split changes.

Rule 2 is easy to miss and would quietly produce excellent, meaningless retrieval
metrics.

### Evaluation

- precision@k and label-agreement@k on validation
- does `agreement` add anything over `u` alone? — compare AURC with and without it in
  `s(x)`. If it does not, say so (`what_did_not_work.md` already commits the project to
  reporting a neutral Librarian).

---

## 3. Channel B — literature / web case retrieval (must NOT influence the prediction)

The request was for "similar past cases retrieval from web". This is worth building, with
one firm constraint.

### The constraint

> Web- and literature-retrieved content may be surfaced **alongside** a prediction for a
> human to read. It may **never** enter the computation of the prediction, the
> uncertainty, or the deferral decision.

This is not excessive caution; three concrete failure modes motivate it:

1. **No provenance guarantee.** An arbitrary web page describing "a pigmented lesion that
   turned out to be melanoma" has unknown reliability, unknown population, and unknown
   selection bias. Case reports are published *because* they are unusual — the retrieved
   distribution is adversely selected by construction.
2. **Prompt injection into a clinical path.** If retrieved text reaches an LLM that
   influences the output, retrieved content becomes an injection surface. The codebase
   already treats external content as untrusted data in other places; the same rule
   applies here.
3. **Unfalsifiable evaluation.** A guarantee from conformal calibration holds for a
   fixed predictor. If the predictor's input includes a live web corpus that changes
   between calibration and deployment, the exchangeability assumption breaks and the
   guarantee is void. **This is the decisive argument** — admitting web content into the
   prediction forfeits C4, the project's main safety contribution.

### Design

```
prediction path   ──────────────────────────────►  ŷ, u, C, deferral
   (Channel A only, calibrated, guaranteed)
                                                        │
                                                        ▼
explanation path  ◄──── Channel B ──────────  report, clearly attributed
   (literature/web, provenance-tagged, no influence on ŷ)
```

Sources, in descending order of trustworthiness — prefer the top:

1. **Curated, versioned corpora** — DermNet NZ, open dermatology atlases, PubMed/PMC
   open-access abstracts. Snapshot and version them, so retrieval is reproducible.
2. **PubMed via E-utilities** — structured, citable, rate-limited.
3. **Open web** — last resort, and only with the source URL shown.

Implementation requirements:

- every retrieved item carries `{source, url, retrieved_at, snapshot_hash}`
- the report renders them in a visually distinct, clearly-labelled section
- Stage-2 verification must assert that **no diagnostic claim in the report traces
  solely to Channel B** — this is a natural extension of the existing rule 3
  ("every claim traceable to a field in evidence_json")
- a snapshot is pinned per experiment, so results are reproducible

### Honest assessment

Channel B is a **product feature**, not a research contribution. It improves how a report
reads and gives a clinician something to check. It should be presented that way, and it
should not appear in an ablation table as though it moved a metric — because by design it
cannot.

---

## 4. A genuinely novel option: retrieval-augmented conformal calibration

If a stronger retrieval contribution is wanted, this is the direction worth exploring.

Standard conformal prediction calibrates one global threshold. But prediction difficulty
is not uniform — a case in a dense, label-consistent neighbourhood is easier than one in
a sparse, mixed region. **Localised / neighbourhood-weighted conformal prediction** uses
retrieval to weight the calibration set by similarity to the query, giving an
*adaptive* threshold:

```
λ̂(x) = quantile over calibration points, weighted by  k(x, xᵢ)
```

Guarantees for localised conformal prediction exist in the literature under
weighted-exchangeability arguments (**[verify]** — check the localised/weighted conformal
literature carefully before claiming this, the conditions are subtle and easy to state
incorrectly).

If it works, the claim is attractive: *retrieval does not vote on the diagnosis; it
determines how much confidence is warranted in this region of the input space.* That
keeps retrieval out of the prediction while still giving it real work to do, and it fits
the project's evidence-management theme exactly.

Scope note: this is a stretch goal for week 13, not a core deliverable. The core
retrieval work is Channel A as a conformity score.

---

## 5. Implementation status

| Piece | State |
|---|---|
| FAISS index over concatenated per-modality embeddings | Built (`src/retrieval/rag_index.py`), dims corrected to 884 |
| `LibrarianAgent` wrapper, agreement rate | Built, 22 tests |
| Per-block normalisation + `β` weights | **To do** (§2) |
| Patient-level neighbour exclusion | **To do — required before any retrieval metric is believable** |
| Retrieval as conformity score in `s(x)` | To do (needs C4 first) |
| Channel B literature retrieval | Not started; design above |
| Localised conformal | Stretch goal |

The patient-level exclusion is the one that must be done first — without it, every
retrieval number is inflated by same-patient lesions.
