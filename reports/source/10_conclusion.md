# Conclusion and Future Work

## Conclusion

This project built DERMA-GUARD, a multimodal decision-support system for skin lesions that behaves like a careful clinician rather than a classifier that always answers. Six specialists examine the clinical photograph, the dermoscopic image and the patient's details; a Devil's Advocate profiles each of them on every case, and a learned gate decides how much each opinion should count. A calibrated, cost-aware policy then gives a high-confidence answer, a clearly marked provisional answer, a request for more evidence, or a referral, with a certified bound on missed cancers. A care platform turns these decisions into a workflow with patient and doctor interfaces, explanations, messaging, follow-ups, data deletion and an audit trail.

On 787 held-out lesions of MILK10k, the system answers 89% of patients and gives 57% a confident answer that is 77% accurate across eleven diagnoses and misses 3.1% of cancers, fewer than a learned rejector at the same coverage. Its miss rate is stable across four fusion back-ends while plain confidence thresholding varies threefold, which is the project's most important finding: a threshold is only as safe as the probabilities beneath it. Learned fusion of the specialists raises photograph-only macro-F1 from 0.286 to 0.377. Explicit image-quality checks and a request for missing details close failures that evidential uncertainty alone does not notice, and on two external datasets the system becomes far more cautious and keeps its missed-cancer rate within target, although its fine-grained diagnosis does not transfer.

Just as importantly, the project replaced an earlier prototype whose reported accuracies of 84% to 90% turned out to rest on a leaky split and on predictions adjusted toward targets. Every number in this report is produced by a script from saved predictions, on a lesion-level split, with every choice made on validation data. The result is a system whose numbers are more modest and whose claims can be checked.

## Future Work

- **A dermatology foundation model** as the image backbone, which the training script already supports, is the most promising way to raise accuracy for every specialist.
- **More training seeds** to report the variability of every result; the aggregation pipeline is ready.
- **Better specialist selection.** The oracle shows that choosing the right specialist per case could raise macro-F1 from 0.44 to 0.59; a gate trained with a larger validation set, or with knowledge distillation from the oracle, is a natural next step.
- **Fairness by design.** Group-conditional calibration by skin tone, and data collection for under-represented tones.
- **Learned quality assessment.** Replacing the hand-designed image checks with a model trained on many kinds of real-world damage, while keeping the guarantee that quality issues can only make the system more careful.
- **A free-text specialist.** The scribe currently changes only urgency; a validated language specialist could let patients' descriptions inform the diagnosis.
- **Prospective evaluation** with dermatologists, measuring how the four outcomes affect clinical decisions and workload, under appropriate ethical approval.
