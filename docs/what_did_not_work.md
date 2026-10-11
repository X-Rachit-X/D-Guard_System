# What Did Not Work

Mandatory section. Negative results are reported, not quietly dropped.

---

## 1. The text modality does not work

**Measured.** Model C (Bio_ClinicalBERT on template-generated narratives) scores
**27.2%** on data it was trained on — near chance for 6 imbalanced classes. Its reported
83.4% came from the prediction-rewriting block. On honest linear probes over reproducible
features it reaches **0.414 macro-F1**, the weakest of the three modalities.

**Consequence.** Adding text to image+metadata *lowers* macro-F1 (0.653 → 0.633). The
tri-modal architecture is, in practice, image + metadata.

**What the system does about it.** Nothing by hand. Text produces little evidence, so
`u_text` lands at 0.80–1.00, approaching the vacuous opinion, and the DS fold gives it
almost no influence. That is the mechanism working as designed — but the honest summary is
still that the synthetic-narrative pipeline did not produce a usable signal.

## 2. Naive multimodal concatenation is worse than the best pair

0.633 (all three) vs 0.653 (image + metadata), replicated on 2 of 3 seeds and on the mean.
A designer can notice this and drop text; a deployed system cannot, because on *some*
cases the narrative is the only informative modality. This is the motivating gap for the
whole project, and it is a negative result about the obvious baseline.

## 3. Untrained "learned" components were worse than closed forms

Four evidence-layer components were randomly-initialised MLPs whose outputs were being
thresholded as if meaningful. Measured: with text quality 0.30 and reliability 0.30, the
untrained weighting network still assigned text **0.68** of the fusion weight. Closed-form
replacements (`log(Q·R·U)`, weighted means, and later the DS fold) are strictly better
than an untrained network.

## 4. The consensus hierarchy as specified was not exhaustive

The v4 spec leaves a gap between Level 1 (agreement >= 3 **and** confidence >= 0.80) and
Level 2 (agreement **== 2**). A case with 6-of-7 agreement at confidence 0.79 matched
neither and fell through to "unresolvable disagreement" — the harshest verdict available,
for one of the least ambiguous cases. It happened on the first real case tested. Reported
because the **specification**, not just the implementation, was wrong.

## 5. Safety metrics that could not fail

`recovery_rate` counted every deferral as a save, so a system refusing every case scored
1.000. Redefined to count only deferrals whose withheld answer would have been wrong, with
deferrals as the denominator. A metric that a degenerate policy maxes out is not measuring
safety.

## 6. AURC was the wrong primary metric, and we recommended it

An earlier version of this plan proposed AURC/E-AURC. Traub et al. (NeurIPS 2024) show
AURC is built on selective risk, which is unsuitable for aggregation across rejection
thresholds, over-weights high-confidence failures, and **changes metric rankings on 5 of 6
datasets**. AUGRC supersedes it. Recorded because it was our error, caught by the v2
survey.

## 7. The ignorance score ranks worse than max-softmax

In the verified end-to-end run, D-GUARD attains a **12x lower malignancy miss rate at
matched coverage** (0.0050 vs 0.0597) but a **worse AUGRC** (0.1634 vs 0.1276). So
`1 - u` is a poorer *ranking* signal than max-softmax even though the policy built on it
is far safer. Both numbers go in the results table; the discrepancy is itself a finding
about what AUGRC rewards.

## 8. Learn-then-Test could not certify a threshold at the target

With n=432 calibration cases, alpha=0.05, delta=0.10 and a 9.95% malignancy-miss base
rate, no threshold on the grid was certifiable. The module reported `controlled=False`
with the reason rather than returning a best guess. The low realised miss rate came from
the conformal ambiguity and conflict gates instead.

Honest consequence: **at this calibration size the alpha=0.05 guarantee is not
attainable.** A larger calibration split or a looser alpha is required.

## 9. The previous feature cache is unrecoverable

`cached_multimodal_features.pt` image features cannot be regenerated from the shipped
images: cosine <= 0.07 across six preprocessing variants, using the exact backbone stored
inside the checkpoints — which proved byte-identical to fresh ImageNet-V2
(`num_batches_tracked` 673830 in both). Row order and label order were verified correct,
and the two image directories are byte-identical on sampled files. No further diagnosis
was possible; the cache was replaced rather than recovered.

## 10. The planned centre-shift experiment is impossible, and we had documented it

Three versions of this protocol specified a centre-held-out shift experiment on MILK10k,
citing its 5 contributing centres (Vienna, Ankara, UQ, MSKCC, Skopje). On extracting the
real release, `attribution` turns out to be the single constant value `MILK study team`
across all 10,480 rows — **there is no centre column**. The 5 centres are a property of
how the study was assembled, not of the data we were given.

The experiment is cut. Distribution shift is now measured across datasets only (MILK10k →
PAD-UFES-20), which is weaker, and the label spaces do not nest (11 vs 6 classes), so even
that is restricted to binary malignancy.

Recorded because it is our error, and the cause is specific: we wrote a protocol around a
dataset's published description instead of its released columns.
`tests/test_milk10k.py::test_real_dataset_has_no_centre_column` pins it.

## 11. The cost matrix was designed for the wrong base rate

MILK10k is **71.7% malignant**, with BCC alone at 48.1%. The cost matrix
(`ℓ_miss = 10`, `ℓ_fa = 2`) was chosen while reasoning about a screening-like population
where misses are rare and expensive. At this prevalence the asymmetry is pointed the wrong
way round: a trivial always-malignant policy attains a **0% malignancy miss rate**, so the
headline safety metric is nearly uninformative on its own.

Nothing here was mis-implemented — the measured consequence is that **miss rate must
always be quoted beside coverage and the base rate**, and that the cost sweep is not
optional. A referral-population cost matrix and a screening one are different objects, and
the project only has data for the former.

## 12. Our documentation said 6 classes in places; MILK10k has 11

Minor beside the two above, but the same failure mode: PAD-UFES-20's 6-class vocabulary
leaked into text describing MILK10k. The real label set is 11 classes from
`supplements/training_gt.csv`. `src/constants.py` now carries both vocabularies
separately, and `tests/test_milk10k.py` asserts the 11 names are in the sorted order the
official one-hot columns use — because a transposed label map is exactly how four of six
PAD-UFES classes were once silently mislabelled.

## 13. The retrieval component was written, unit-tested, and never connected

`RAGIndex` and `LibrarianAgent` were both implemented, `faiss` was installed, and 22 tests
passed. The retrieval channel was nonetheless **completely inert in the running system**,
for four independent reasons:

1. **Nothing ever built an index.** No script in `scripts/` referenced `RAGIndex`, so the
   Librarian returned `{"similar_cases": [], "note": "RAG index not loaded"}` on every
   real case.
2. `OrchestratorAgent.build()` never constructed a Librarian, so even a built index would
   not have been attached.
3. The call site passed the **256-d projected** embeddings where the index needed the raw
   2048-d and 768-d features.
4. `knowledge` reached the report generator and was then dropped — `PipelineResult` had no
   field for it, so an explanation-only channel could not be audited.

The tests passed throughout, because they exercised `RAGIndex` directly with synthetic
cases. **A green suite over an unreachable code path.** This is the same shape as the dead
fusion path in `evaluation_integrity.md`: the unit was correct, the wiring was absent, and
nothing in the test suite could tell the difference. The lesson that generalises is that a
component needs at least one test that reaches it *through the pipeline*, not only
directly.

All four are fixed, with 17 new tests, and `demo.py` now prints the retrieved neighbours.

## 14. The runtime retrieval index is the weaker one

The metadata preprocessor emits **116** columns fitted on the full CSV but **114** fitted
on the training split only — two one-hot columns for categories absent from the training
subset. The live pipeline emits 116, so a 114-wide index cannot be queried in production
and padding it would misalign every subsequent column.

Measured on the same query case:

| index | neighbour agreement |
|---|---|
| image + text + metadata (2930-d) | **0.875** |
| image + text (2816-d, runtime-queryable) | **0.500** |

So the index the system can actually use retrieves **substantially worse** neighbours.
Metadata carries much of the retrieval signal, which is consistent with it being the
strongest single modality (0.648 vs 0.477 macro-F1 for images). Reported because the
convenient framing — "retrieval works" — would hide a factor-of-almost-two drop in the
only quality signal available. The fix is to persist the train-fitted preprocessor; it is
not done.

## 15. An agreement metric that could only ever read zero

`LibrarianAgent.retrieve` computes `agreement_rate` against `predicted_diagnosis`, but the
Librarian runs at pipeline step 3 and the diagnosis does not exist until step 4. It was
therefore always called with `None`, and the rate was **structurally 0.0 on every case**.
The first live run printed `agreement 0.000`, which reads as "retrieval finds nothing
useful" and actually meant "this number cannot be computed here".

Now computed after consensus, and `None` (not measured) is kept distinct from `0.0`
(measured, nothing agreed). Same family as the `recovery_rate` defect in §5: a metric
whose value is determined by where it is called rather than by the system's behaviour.

## 16. Still unknown

Not yet answerable, and marked as such rather than guessed:

- whether the acquisition policy beats random/greedy (needs MILK10k paired data)
- whether per-modality `u_m` genuinely predicts per-modality error (**H4**)
- whether DS conflict mass fires on real contradictions rather than merely hard cases (**H5**)
- whether the guarantee degrades gracefully under shift (**H3**)
- whether retrieval adds anything over `u` alone

All require the GPU training run and a valid held-out evaluation.
