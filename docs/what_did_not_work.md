# What Did Not Work — DERMA-GUARD v4.0

**This is a mandatory section.** Negative results and failed approaches are reported here to strengthen reproducibility and credibility. Update this file as experiments complete.

---

## 1. Iterative Fusion: Does 3 Iterations Add Value Over 1?

**Status**: TBD — fill in after running ablation.

**Hypothesis**: EM-style re-weighting should converge to a better solution than a single forward pass.

**Expected failure mode**: If the WeightLogitNetwork is not well-calibrated early in training, the diagnosis-adaptive prior W_prior[diag] may point weights in the wrong direction, making iteration k+1 worse than k=0. This would mean the `max_iterations=3` does not help and may hurt on uncertain cases.

**If this fails**: Report that convergence was not beneficial beyond 1 iteration. Ablation chart will show this clearly.

---

## 2. Metadata Utility (U_metadata)

**Status**: TBD — fill in after utility analysis.

**Hypothesis**: Metadata fields (age, location, etc.) provide meaningful discriminative signal.

**Expected failure mode**: For skin diseases with overlapping demographic distributions (e.g., seborrheic keratosis vs BCC), metadata may add noise, not signal. If the ablation shows that "+metadata" does not improve macro-F1, report it.

**If this fails**: The baseline ladder will show B2 (image only) ≈ B4 (multimodal), and the ablation will show "+metadata" as flat or negative.

---

## 3. Librarian (RAG) Adding Value

**Status**: TBD.

**Hypothesis**: Retrieving similar training cases improves confidence calibration via agreement_rate signal.

**Expected failure mode**: With only 413 training cases (Derm7pt), the retrieved cases may be too few and too dissimilar to add meaningful signal. The RAG agreement_rate may be close to random for rare classes.

**If this fails**: Report that RAG is neutral. The ablation will show "+Librarian" as flat.

---

## 4. Acquisition Policy vs. Random Selection

**Status**: TBD.

**Hypothesis**: GradientBoostingRegressor predicts which additional modality reduces uncertainty most.

**Expected failure mode**: With small training data (413 cases), the GBR may not have enough examples of "acquire dermo → uncertainty drops" patterns. It may not outperform a simple heuristic ("always ask for dermoscopy if missing").

**If this fails**: Report that acquisition policy = random selection. Recommend replacing with a simple priority rule.

---

## 5. PTIS v2 (Validated Regressor) vs. PTIS v1 (Heuristic)

**Status**: TBD.

**Hypothesis**: A regressor trained on human-rated text samples will better predict text informativeness than a hand-crafted heuristic.

**Expected failure mode**: If the 100 rated samples are not diverse enough, PTIS v2 may overfit to rater biases. It may perform no better than PTIS v1 on held-out text.

---

## 6. Synthetic Text Validation (regex leakage)

**Known issue**: Even with regex diagnosis-term scanning, synthetic text generated from age/location/symptoms may inadvertently encode diagnostic cues. For example, "elderly patient with facial lesion that bleeds and crusts" is strongly suggestive of BCC even without mentioning it.

**Mitigation**: Embedding diversity check (cosine < 0.97) helps but does not fully solve this.

---

## Template for Future Entries

```
## N. [Component or experiment]
**Status**: Did not work / Partially worked / Uncertain
**Expected**: [What you hoped would happen]
**Observed**: [What actually happened, with numbers if available]
**Interpretation**: [Why you think it failed]
**Impact on claims**: [Which paper claims are affected]
```


---

# Addendum - findings from the 2026-09-15 integration pass

These are measured, not hypothesised. Full detail in
[evaluation_integrity.md](evaluation_integrity.md).

## A. The text modality does not work

Model C (Bio_ClinicalBERT on generated narratives) scores **27.23%** on data it was
trained on - near chance for 6 imbalanced classes. Its reported 83.44% came from the
prediction-rewriting block. Its predictive entropy on real cases runs ~1.56 against a
maximum of log(6) ~= 1.79, i.e. it is close to uniform.

This is the clearest negative result in the project: the synthetic-narrative pipeline did
not produce a useful text signal. The system now handles this correctly rather than hiding
it - the evidence layer assigns text low reliability and fusion down-weights it - but the
honest summary is that text contributes little and the tri-modal architecture is, in
practice, image + metadata.

## B. Untrained "learned" components were worse than closed-form ones

Four evidence-layer components were randomly-initialised MLPs whose outputs were being
thresholded as if they carried signal. Measured: with text quality 0.30 and reliability
0.30, the untrained weighting network still assigned text 0.68 of the fusion weight. The
closed-form replacements (`log(Q*R*U)`, weighted means) are strictly better than an
untrained network, and are what the deployed system uses. The learned forms remain
available but are opt-in.

## C. The consensus hierarchy as specified was not exhaustive

`proj.md` Part 6 leaves a gap between Level 1 (agreement >= 3 AND confidence >= 0.80) and
Level 2 (agreement == 2). A case with 6-of-7 agreement at confidence 0.79 matched neither
and fell through to "Unresolvable disagreement". This was not a rare corner - it happened
on the first real case. Reported because the specification, not just the implementation,
was wrong.

## D. Safety metrics that could not fail

`recovery_rate` counted every deferral as a save, so a system refusing every case scored
1.000. Redefined to count only deferrals where the withheld answer would have been wrong.
A metric that a degenerate policy maxes out is not measuring safety.

## E. Still unknown

Whether iterative fusion beats a single pass, whether the Librarian adds value, and
whether the acquisition policy beats random **cannot currently be answered**. All three
require a valid held-out evaluation, which does not exist for these checkpoints. Marking
these TBD rather than guessing.
