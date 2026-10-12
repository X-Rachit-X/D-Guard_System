# Discussion

## What the Results Mean

The project set out to build an AI system that behaves like a careful clinician: answer when the evidence is good, ask for more when it is not, mark provisional opinions clearly, and hand over when the evidence contradicts itself. The results show that this is achievable with today's tools, and that it changes what a patient receives. Instead of one label for everyone, 57% of patients receive a confident answer that is right three times out of four and misses about one cancer in thirty, 32% receive a clearly marked provisional answer that a doctor confirms, 11% are referred because their evidence disagrees, and nearly a third are asked for a dermoscopic image on the way. Every confidence shown is calibrated, and every decision comes with an explanation.

The results also show what such a system can and cannot claim. Its central safety number, the missed-cancer rate of confident answers, is lower than a learned rejector's at the same coverage and is certified by Learn-then-Test. It is only slightly lower than plain confidence thresholding on our final model, a difference that could be chance. The strongest argument for the calibrated policy is not that it beats thresholding on one model, but that it does not depend on the model: across four different fusion back-ends its miss rate stayed between 3.0% and 3.5%, while thresholding ranged from 3.5% to 9.3%. In practice, nobody deploying a skin-cancer app knows in advance whether their probabilities are of the "good" or the "bad" kind. A method that is safe in both cases is worth more than one that is safe only in the lucky case.

## Comparison with the Base Papers

**Base paper 1, HOT AI** [@mehta]. DERMA-GUARD adds what the gap matrix marks as open for HOT AI: explicit costs, a separate defer action, a certified miss rate and combined-failure testing on public paired data. The HOT-style rule built from it (ask the most uncertain patients for dermoscopy) turned out to be a strong baseline: on the final model it chose whom to ask better than our learned estimator. The evidence therefore favours a simple uncertainty trigger for *acquisition*, combined with the calibrated policy for *answering*, which is where DERMA-GUARD's safety comes from (3.1% missed vs 5.6% when everyone is answered after HOT-style acquisition).

**Base paper 2, Learn-then-Test** [@ltt]. It delivered what it promises: the certified bound held on MILK10k, and in our external tests it also held on PAD-UFES-20 and Derm7pt, even though the theory only covers the first.

**Other methods.** A learned rejector in the spirit of learning to defer [@l2d] was significantly less safe; plain confidence thresholding [@geifman] was close on the final model but swings between 3.5% and 9.3% across fusions; AUGRC [@traub] showed softmax ranking errors better than ignorance on the final model; and the reader study on MILK10k [@muller] was borne out, with dermoscopy and the clinical photograph each adding information.

## What the Original Design Got Right

The earlier prototype was built around three ideas: consult several specialists, have a sceptical component question their evidence, and trust each source according to the suspected diagnosis. All three survived honest evaluation. Every structured way of weighting the team, including the original multiplicative formula once its numbers were learned rather than typed in, beat both the evidential network on its own and a plain average of the specialists. The measured diagnosis-conditioned reliabilities confirmed the clinical intuition that dermoscopy matters more for squamous cell carcinoma (0.74 vs 0.36 for the photograph on validation) while the photograph is better for melanoma (0.52 vs 0.40). And the learned reliability automatically discounted the weak external Model A without being told it was weak.

What the prototype got wrong was not its ideas but its evidence. Its split let the same patients appear in training and test sets; its evaluation adjusted predictions toward target accuracies; and its demonstration looked up the true label. Re-scored honestly, its best model reaches 75% on unseen patients rather than 90%, and its multimodal models are worse than its metadata model alone. The new project kept the ideas, replaced every hand-set number with a fitted one, and made every evaluation a script that anyone can re-run.

## What Did Not Work

Several things that seemed promising did not survive measurement. They are reported here because they are as informative as the successes.

- **Chow's rule as written.** Stating the minimum-expected-cost diagnosis, the textbook form of cost-sensitive classification, labelled 98.5% of lesions malignant on this referral population, because with a 5:2 cost ratio and 72% prevalence "malignant" is almost always the cheaper guess. The stated label is therefore the most probable class, and costs are used only for choosing between answering, asking and deferring.
- **Adaptive prediction sets for the safety gate.** The first version of the benign/malignant gate used adaptive prediction sets; their calibrated threshold collapsed to the full set, so every set contained both outcomes and every patient was deferred. Least-ambiguous sets with separate (Mondrian) calibration for benign and malignant lesions gave small, useful sets.
- **Post-hoc logit adjustment for class imbalance** raised validation macro-F1 from 0.36 to 0.44 but lowered test macro-F1 from 0.376 to 0.359, a clear case of overfitting the validation set. It was dropped.
- **In-sample fitting of the fusion.** Fitting the fusion on the same validation lesions that later calibrated the policy made the results look better than they were (60.2% HIGH answers instead of 57.2%). Cross-fitting fixed this.
- **Evidential ignorance as a quality detector.** Blur lowers ignorance instead of raising it, and darkness and occlusion pass unnoticed. Explicit image checks are necessary.
- **Learned acquisition.** The gradient-boosted estimator of dermoscopy's value chose patients far better than "ask the most uncertain" on the iterative fusion, but not on the final learned gate. Once the probabilities are good, entropy is a good enough guide.
- **Stacking.** The textbook way to learn a combination of models was the worst learned fusion rule; with 786 validation lesions there is not enough data for an unstructured meta-classifier.
- **Rare classes.** Classes with fewer than ten test lesions are almost never predicted. More data, not a better fusion rule, is the remedy.

## Lessons About Evaluation

The most valuable lessons of the project were about how to measure, not what to build.

1. **Split by the unit you will meet in practice.** For skin lesions that is at least the lesion, ideally the patient. A random split by photograph inflated our earlier accuracies by 10 to 30 points.
2. **Never touch predictions after the model makes them.** Any post-processing that uses the label, however it is motivated, makes the number meaningless.
3. **Choose on validation, report on test, once.** Every choice in this project, including which fusion rule to use and which specialists to include, was made on validation data; when the test set disagreed with that choice (for example, the iterative fusion scored a higher test macro-F1 with dermoscopy), the pre-declared choice was kept.
4. **Compare against simple baselines on the same probabilities.** The most important finding of the project, that a confidence threshold can be as safe as a sophisticated policy on one model and three times worse on another, appears only when both are evaluated on the same fused probabilities and on more than one model.
5. **Report what goes wrong.** Skin-tone disparities, failures under shift, and wins for simple baselines are part of the result.

## Ethical Considerations

DERMA-GUARD is a research prototype. It has not been validated in a clinical setting, and it should not be used to make decisions about real patients. Several design choices nevertheless follow from ethical considerations that a real deployment would face.

- **Safety over throughput.** The cost matrix makes a missed cancer five times worse than a false alarm, so the system prefers to refer. This costs specificity (half of the benign lesions given a HIGH answer are called possibly malignant) and doctor time, which is the right trade-off for a triage tool but must be stated openly.
- **Fairness.** The system performs worse for lighter skin tones, which are under-represented in MILK10k. This must be fixed with more representative data before any deployment.
- **Transparency.** Every decision records the policy version and the evidence behind it, and every outcome comes with an explanation in plain language; a language model may only reword explanations and is checked against the evidence.
- **Privacy.** Patients can delete their cases and their whole record, including uploaded photographs; the audit log retains only the fact of deletion. Dataset images are never redistributed.
- **Human oversight.** Provisional answers and deferrals always go to a doctor, and doctors can override any decision.

## Limitations

- **One training seed.** All results come from one trained network. Each additional seed also redraws the split, so seed-to-seed variation will include split variation; the pipeline to aggregate additional seeds is in place.
- **A general-purpose backbone.** The image encoder is an ImageNet ResNet50. Dermatology foundation models are likely to improve every specialist.
- **Validation does a lot of work.** 786 lesions choose the epoch, fit every fusion rule, calibrate every guarantee and choose between methods; cross-fitting limits but does not remove the resulting optimism.
- **A referral population.** 72% of MILK10k lesions are malignant, far more than in screening. No claim is made for screening use.
- **No patient or centre identifiers.** Splits are by lesion; patient-level leakage within MILK10k cannot be excluded, and centre shift cannot be tested within the dataset.
- **Declared costs.** The cost matrix expresses design choices, not measured clinical utilities.
- **Guarantees assume exchangeability.** Under distribution shift they do not hold; the external tests show the safety target being met on two datasets, which is encouraging but not a guarantee.
- **Heuristic image checks.** The quality thresholds come from 300 training photographs and may miss kinds of damage not included in DERMA-STRESS.
