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

## 16. The exported conflict score was zero for every lesion

The first MILK10k training run exported `conflict_C = 0.0` for **all 787** test lesions.
The Kaggle trainer had its own copy of the Dempster–Shafer fold that reported only the
conflict of the **last** pairwise step. On MILK10k the last modality is the absent text,
its opinion is vacuous, and conflict with a vacuous opinion is exactly zero by
construction — so every real conflict from the earlier steps was thrown away.

The library version (`src/evidence/evidential.py`) was correct all along: it carries the
**maximum** conflict over the fold. The trainer had drifted from it. Conflict was not part
of the training loss, so the trained weights were unaffected; `scripts/milk_native_infer.py`
recomputes conflict from the checkpoint (now non-zero on 787/787 lesions) and the trainer
is fixed for future runs.

Lesson: two implementations of the same formula will drift. The trainer is self-contained
on purpose (it must run on Kaggle without this repository), which makes a shared test the
only defence.

## 17. Using every source did not beat the best pair on MILK10k either (H1)

| | macro-F1 |
|---|---|
| dermoscopy + metadata | **0.403** |
| all three | 0.379 |

Evidence-aware fusion was supposed to let a weaker source fall silent. It reduced the
damage — but the clinical photo still pulls the result down by 0.024. Same shape as the
PAD-UFES-20 text result (§2), smaller gap. **H1 is not supported by this run.** One seed;
seeds 7 and 2026 may move it either way.

## 18. Chow's rule, as written, called 98.5% of lesions cancer

The policy first stated the label with the lowest expected cost. With a missed cancer
priced at 10 and a false alarm at 2, the cost-optimal label is "malignant" whenever
P(malignant) > 1/6. On a population that is 73.6% malignant, that is almost every lesion.

The result looked excellent — **0.000 missed cancers** for "always answer" — for a useless
reason: it called 98.5% of lesions malignant. A metric a degenerate policy maxes out is not
measuring safety (the same lesson as `recovery_rate`, §5).

Fix: the system states the **most likely** diagnosis (`label_rule="argmax"`); the cost
matrix decides only **whether** it may state it alone. Both options remain in the code so
the comparison is reproducible.

## 19. The first safety gate deferred 100% of test lesions

Two independent errors, found together:

1. **APS on two classes degenerates.** Without randomisation, any lesion whose true class the
   model ranks second gets a score of exactly 1.0. The model calls about half of benign
   lesions malignant, so the benign quantile became 1.0 and every set contained both
   outcomes. Fixed with the LAC score `1 − p(true)`.
2. **The gate blocked both directions.** Its purpose is to stop *false reassurance* ("it's
   harmless" while cancer is possible). It was also blocking "likely cancer, see a
   dermatologist" while benign was possible — the safe direction. Fixed with
   `ambiguity_gate="reassurance"`.

The earlier 11-class version of the gate had deferred 70% for a related reason: an 11-class
set from a 0.38-macro-F1 model is large almost everywhere.

## 20. At the pre-registered cost, the system can only ever say "cancer"

With a missed cancer priced at 10, saying "benign" needs P(malignant) below about 0.15, and
this model rarely gets that low. So at cost 10 **every** answer D-GUARD gives is a cancer
diagnosis (100% of answers, false-alarm rate 1.000 among the few benign lesions it answers).
It never misses a cancer — and never reassures anyone. Cost 5 is the balanced point
(1.7% missed, 86% of answers malignant). The cost is a clinical judgement; we report both.

## 21. On mean cost, "always answer" beats D-GUARD

Mean cost per case: always answer from the photo **0.703**, D-GUARD **1.086** (cost 5).
Each deferral costs 1.5 and D-GUARD defers 63%. D-GUARD wins at **matched coverage** —
fewer missed cancers *and* lower cost than confidence-based abstention — which is the fair
comparison for a deferral system. But a reader who values throughput above safety among
answered cases would prefer always answering, and should be told so.

## 22. Rare classes are never predicted

BEN_OTH (4 test lesions), INF (6) and MAL_OTH (1) get F1 = 0 — the model never predicts
them. Class weighting in the loss was not enough at this data size. No claim is made about
these classes.

## 23. Post-hoc imbalance correction overfit the validation set

Logit adjustment (dividing probabilities by the training class prior to the power τ, a
standard long-tail fix) raised **validation** macro-F1 from 0.361 to 0.439 at τ = 0.1 — and
**lowered test** macro-F1 from 0.376 to 0.359. The best τ was chosen on a handful of
rare-class validation lesions and did not transfer. Not used. Real accuracy gains need a
better image encoder (DermLIP), not post-hoc tuning.

## 24. The evidential model gets MORE confident on blurred photos

Under DERMA-STRESS, Gaussian blur **lowered** mean ignorance (0.68 → 0.61) and **raised** the
HIGH-tier rate (35% → 46%). Darkening a photo left ignorance nearly unchanged while HIGH
answers missed 12.5% of cancers. So `u` alone does not detect these failures — blur removes
texture the model reads as "less ambiguous". The image-quality agent flagged 100% of both.
Fix: a severe quality issue (outside the 1st/99th percentile of training photos) downgrades
a HIGH answer to LOW; with it, blurred and dark photos produce no HIGH answers.

## 25. A setting saved in the wrong place

`defer_mode` was written into the policy settings twice and never into the calibration
bundle, so the app silently ran in the old standard mode while the experiment reported
minimal mode. Caught because the demo deferred a case for low confidence, which minimal
mode cannot do. The bundle now records it and the engine reads it.

## 26. The original project's model does not transfer to MILK10k

PAD-UFES Model A — the original system's image specialist — reaches **21.9% accuracy** on
MILK10k clinical photos (macro-F1 0.077), below always guessing BCC. It was trained on
smartphone photos from another dataset (and under the flawed training documented in
`reference/evaluation_integrity.md`). It stays in the team because the Devil's Advocate
handles it correctly — learned reliability gives it ~2% weight — but it adds little.

## 27. A plain average of the team is worse than one good fusion

Averaging the five specialists gives macro-F1 0.317, below Dempster–Shafer of the three
heads (0.376). Only the evidence-aware adaptive fusion (0.453) beats it. More models do not
help by themselves; weighting them by evidence does.

## 28. Occlusion and missing details still cause confident errors

With the team, occluded photos keep 14% HIGH answers that miss 9.3% of cancers, and with
details withheld the HIGH miss rate rises from 4.3% to 6.4%. Neither the evidential
signals nor the quality statistics fully register these. Open.

## 29. More confident answers means more false alarms

With the team the HIGH tier grows from 35% to 60% of patients — and HIGH answers call 53% of
the benign lesions they answer malignant (35% before). With a 5:2 cost ratio, "possibly
cancer, see a dermatologist" is the cheaper statement under doubt. Safe, but it costs
specificity.

## 30. A learned rejector is as safe as our HIGH tier

A gradient-boosted error predictor trained on validation (L2D-style), abstaining at the same
60.2% coverage, misses 3.6% of cancers against D-GUARD's 3.0% — difference 0.6 points,
95% CI −0.7 to +2.0, p = 0.18. **Not significant.** The large, significant gap is against
a plain confidence threshold (9.3%). D-GUARD's case therefore rests on what it adds beyond
safety-by-abstention — certification, acquisition, answering the rest, explanations — not
on beating every deferral method on miss rate.

## 31. Lighter skin tones fare worse

Skin tone 2 (67 test lesions): 8.3% missed cancers in HIGH answers vs 2.2% for tone 3; tone
1 has only 14 lesions. MILK10k is dominated by tone 3. No mitigation attempted.

## 32. The diagnosis does not survive a change of dataset

On PAD-UFES-20, unchanged, overall accuracy falls to 38.5% and HIGH-tier accuracy to 47.6% —
below a plain confidence threshold at the same coverage (55.7%). Missed cancers stay at 0%
and malignancy AUROC holds (0.80), so the safety decision transfers; the fine-grained
diagnosis does not.

## 33. Still unknown

Not yet answerable, and marked as such rather than guessed:

- whether the results hold across seeds 7 and 2026
- whether a dermatology encoder (DermLIP) changes H1
- whether per-modality `u_m` genuinely predicts per-modality error (**H4**)
- whether DS conflict fires on real contradictions rather than merely hard cases (**H5**) —
  testable now that conflict is computed correctly; one demo case (true melanoma, photo
  says NV, details disagree) is encouraging but anecdotal
- **answered:** under PAD-UFES-20 shift the miss-rate target held (0%), accuracy did not (§32)
- whether retrieval adds anything over `u` alone

**Answered since the last version:** the acquisition policy **does** beat random and
entropy-based acquisition (AE +0.27 vs +0.01 at cost 5, +0.40 vs −0.06 at cost 10).
