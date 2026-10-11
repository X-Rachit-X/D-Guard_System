# Limitations

Read [evaluation_integrity.md](evaluation_integrity.md) first — the single largest
limitation is that no model metric produced before 2026-10-11 is valid.

---

## 1. Data

**Synthetic text.** PAD-UFES-20 narratives are template-generated from structured fields,
not written by patients. Any claim about patient-authored text is a claim about a
simulation. **Largest external-validity threat.** MILK10k ships no narrative at all, so
the text modality is absent there — cleaner, but it means the tri-modal claim rests on the
weaker dataset.

**No patient identifiers in MILK10k.** Splits are by lesion. If one patient contributed
several lesions they can straddle the boundary. Residual leakage risk, stated and
unquantifiable from the published metadata. What *was* checkable has been checked: all
10,480 images are byte-distinct by SHA-256, so there is no exact-duplicate leakage, and
each lesion's two images always land on the same side of every split boundary.

**Rare classes.** Measured on the real release: MILK10k MAL_OTH has **9 lesions in
total** (5 after a 60/20/20 split), and BEN_OTH 44, VASC 47, INF 50, DF 52. PAD-UFES-20
MEL has 52 of 2298. **5 of 11 MILK10k classes cannot receive a class-conditional
guarantee** at α=0.05 — they have fewer than ⌈1/α⌉=20 calibration points — and the
implementation withholds a threshold for them rather than fabricating one. Per-class
conclusions for these classes are not supportable, which is why the guarantee is stated on
**binary malignancy**.

**Prevalence: 71.7% of MILK10k lesions are malignant, and BCC alone is 48.1%.** This is
the single most consequential property of the dataset and it was not anticipated when the
cost matrix was designed. It is a tertiary-referral population, not a screening one. A
trivial always-malignant policy achieves a 0% miss rate here, so **the malignancy miss
rate must always be reported beside coverage and beside the base rate** — alone it is
nearly meaningless on this dataset. No screening claim is supportable, and the cost matrix
must be re-derived before any such claim.

**No centre-shift experiment is possible.** Corrected 2026-10-11: although the MILK10k
study describes 5 contributing centres, the released metadata publishes **no centre
column** — `attribution` is the single constant value "MILK study team" across all 10,480
rows. An earlier version of these docs planned a centre-held-out experiment on that
basis; it cannot be run. `tests/test_milk10k.py` pins the finding. Distribution shift is
therefore measured **across datasets only** (MILK10k → PAD-UFES-20), which is a weaker
design than planned and is reported as such. Published evidence that shift matters:
training on PAD-UFES-20 6-class and testing on HIBA gives **46.3% balanced accuracy**,
which those authors call no better than random guessing.

**Label-space mismatch between the two datasets.** MILK10k has 11 classes, PAD-UFES-20
has 6, and they do not nest cleanly. Cross-dataset transfer is therefore evaluated on
**binary malignancy**, not on the diagnostic label — a real restriction on the shift
analysis, not a presentational choice.

**Label uncertainty.** MILK10k is ~96% histopathology-confirmed, which mitigates but does
not remove it. PAD-UFES-20 is 58.4% biopsy-proven overall (100% of cancers).

## 2. Method

**The guarantee assumes exchangeability.** `P(R(λ̂) <= α) >= 1 - δ` holds over the
calibration draw when calibration and test data are exchangeable. It **does not hold under
distribution shift.** Measuring the size of the violation under each stress family is a
planned result, not a caveat to bury.

**Class-conditional guarantees need data.** A stratum needs roughly `1/α` calibration
points — 20 per class at α=0.05. Rare classes cannot reach it. The implementation reports
`underpowered_classes` / `underpowered_groups` and withholds a threshold rather than
fabricating one.

**The cost matrix is a judgement.** `ℓ_miss = 10`, `ℓ_fa = 2`, `c_acq = 0.5`,
`c_def = 1.5` are declared, not derived. The sensitivity sweep is mandatory because
conclusions depend on them. Making them explicit is an improvement over hiding the same
assumption inside three confidence thresholds — but it does not make them right.

**Acquisition benefit is only directly measurable where the modality exists.** On MILK10k
it does for every lesion, which is why that dataset was chosen. Any simulated acquisition
must be labelled a simulation.

**Bridge mode is an approximation.** Lifting a softmax into an opinion via normalised
entropy conflates aleatoric with epistemic uncertainty, which a trained Dirichlet head
separates. Bridge numbers must be labelled and never presented as native results.

**Conflict detection is calibrated, not validated.** The threshold is the validation p90
of DS conflict mass. That it fires on genuine contradictions rather than merely on hard
cases is an assumption still to be tested (H5).

**Native results rest on one seed and one encoder.** The MILK10k numbers in
[14_results.md](14_results.md) come from a single ResNet50 run (seed 42). Seeds 7 and 2026
are required before they are final, and validation was used both to pick the epoch and to
calibrate the policy.

**The system answers alone for only about a third of patients.** At the balanced cost
(missed cancer = 5) it defers 63%. Every deferred patient gets a tentative opinion and a
place in the doctor queue, but the deferral rate is set by the classifier's quality and is
the main practical limitation.

**At the pre-registered cost the system cannot reassure anyone.** With a missed cancer
priced at 10, every answer it gives is a cancer diagnosis. See
[what_did_not_work.md](what_did_not_work.md) §20.

**MONET concept scores are unavailable for new uploads.** They come from an external model
run on the image; for a user's own photo the engine substitutes the training median.

## 3. Scope

- No prospective or clinical validation; no clinical-utility claim.
- No reader study comparing the deferral policy against clinicians.
- Fairness is reported per skin tone, but no bias **mitigation** is attempted. The
  strata are severely imbalanced (measured: tone 3 = 60.6% of images; tone 0 = 6 lesions,
  ~1 calibration point), so tone 0 receives **no** per-group guarantee and tone 1 is
  borderline at 24 calibration lesions against the 20 required.
- Retrieval is explanation-only and deliberately outside the decision path: admitting a
  live external corpus would break exchangeability and forfeit the risk guarantee. The
  index is built from the **training split only** and `--verify` asserts no calibration or
  test patient is present, because a retrieved neighbour would carry a held-out label into
  the narrative where no metric would detect it.
- **The runtime retrieval index excludes metadata** and is measurably weaker for it
  (neighbour agreement 0.500 vs 0.875). The preprocessor emits 116 columns at runtime but
  114 fitted on train only, so the stronger index is not runtime-queryable until the
  train-fitted preprocessor is persisted. Not done.
- **Whether retrieval adds anything over the ignorance score `u` alone is untested.** It
  is not claimed as a contribution.
- The retrieval index is currently built on **PAD-UFES-20** (6 classes) and must be
  rebuilt on MILK10k after training.
- The LLM writes reports; it does not diagnose, and no diagnostic credit is attributed to
  it.

## 4. Environment

- Local torch is **CPU-only**, so all training is off-box; results depend on a Kaggle
  environment not fully pinned here.
- `torchvision.ResNet50_Weights.DEFAULT` silently moved from V1 to V2 between releases.
  It is now pinned explicitly, because that drift is part of why the previous feature
  cache is irreproducible.
- DermLIP_PanDerm is **CC-BY-NC-ND**; "ND" arguably forbids fine-tuning, so linear probing
  is the safer reading. Its pretraining corpus must be checked for overlap with our
  evaluation datasets before any clean-probe claim.
