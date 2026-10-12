# Experiments and Results

This chapter reports every experiment. Unless stated otherwise, results are on the 787 held-out MILK10k test lesions, which were scored once after every design decision had been fixed on validation data. The operating point of the decision policy is a missed-cancer cost of 5 with the "minimal deferral" mode and all gates switched on. Results that do not favour DERMA-GUARD are reported alongside those that do.

## The Specialists

Table: Each specialist alone on the test split (with dermoscopy available).
| Specialist | Accuracy | Macro-F1 | Balanced accuracy | Malignant AUROC |
|---|---|---|---|---|
| S1 photo head (evidential) | 0.595 | 0.264 | 0.266 | 0.787 |
| S2 dermoscopy head (evidential) | 0.705 | 0.400 | 0.411 | 0.854 |
| S3 details head (evidential) | 0.600 | 0.246 | 0.252 | 0.749 |
| S4 earlier prototype's Model A | 0.219 | 0.077 | 0.096 | 0.553 |
| S5 similar-case vote | 0.719 | 0.431 | 0.402 | 0.866 |
| S6 embedding classifier | 0.700 | 0.440 | **0.447** | **0.875** |

The specialists are very different. The dermoscopy head is clearly better than the photograph head, confirming that dermoscopy carries real extra information: adding it lowers the evidential network's mean ignorance from 0.676 to 0.439 and raises its malignant-vs-benign AUROC from 0.799 to 0.856. The details head is surprisingly competitive on accuracy, because age and body site alone predict basal cell carcinoma well in a dataset where half the lesions are BCC. The legacy Model A transfers poorly (21.9% accuracy): it was trained on smartphone photographs from Brazil and can name only six of the eleven classes. The two case-based specialists, which reuse the network's learned embeddings, are the strongest single sources; the class-balanced embedding classifier has the best balanced accuracy, because it is the only specialist explicitly designed not to ignore rare classes.

## Fusion: Comparing Seven Rules

Table: The seven combination rules. "Val" is the out-of-fold macro-F1 on validation used to choose the rule; all other columns are test.
| Rule | Val | Accuracy | Macro-F1 | Balanced acc. | AUROC |
|---|---|---|---|---|---|
| *Photo + details (stage A)* | | | | | |
| Dempster-Shafer (evidential heads only) | 0.274 | 0.644 | 0.286 | 0.280 | 0.799 |
| Simple average | 0.344 | 0.637 | 0.343 | 0.338 | 0.824 |
| Original formula (Q R U) | 0.380 | 0.673 | 0.381 | 0.369 | 0.836 |
| Best head per case (highest R) | 0.319 | 0.672 | 0.337 | 0.323 | 0.788 |
| Iterative EM (v1) | 0.374 | 0.670 | **0.383** | **0.375** | 0.832 |
| Stacking | 0.272 | 0.644 | 0.308 | 0.296 | 0.822 |
| Learned gate, mixture (v2) | 0.396 | **0.677** | 0.371 | 0.355 | 0.833 |
| **Learned gate, product (v2), selected** | **0.412** | 0.657 | 0.377 | 0.360 | **0.838** |
| Oracle best head (uses the label) | - | 0.799 | 0.542 | - | - |
| *With dermoscopy (stage B)* | | | | | |
| Dempster-Shafer (evidential heads only) | 0.361 | 0.690 | 0.376 | 0.378 | 0.856 |
| Simple average | 0.457 | 0.714 | 0.422 | 0.400 | 0.870 |
| Original formula (Q R U) | 0.496 | 0.719 | **0.460** | **0.446** | 0.872 |
| Best head per case (highest R) | 0.509 | 0.727 | 0.449 | 0.427 | 0.816 |
| Iterative EM (v1) | 0.492 | 0.717 | 0.459 | 0.444 | 0.862 |
| Stacking | 0.413 | 0.720 | 0.372 | 0.361 | 0.853 |
| Learned gate, mixture (v2) | 0.485 | **0.731** | 0.452 | 0.431 | 0.878 |
| **Learned gate, product (v2), selected** | **0.517** | **0.731** | 0.438 | 0.418 | **0.885** |
| Oracle best head (uses the label) | - | 0.840 | 0.594 | - | - |

![Validation (out-of-fold) and test macro-F1 of every combination rule; the dotted line is the oracle that always picks the best specialist.](paper/figures/fig_fusion_rules.png)

Five observations stand out.

1. **Every learned way of weighting the team beats the evidential network on its own.** Photograph-only macro-F1 rises from 0.286 to 0.377 for the selected rule (+32% relative), and malignant AUROC from 0.799 to 0.838. With dermoscopy the selected rule gives the best accuracy (0.731) and the best AUROC (0.885) of all rules.
2. **The learned gate was selected in both stages** because it had the best out-of-fold validation score. On test it is not always the best on macro-F1: the iterative EM and the original formula score slightly higher. Macro-F1 over eleven classes, several with fewer than ten test lesions, is noisy, and we report the pre-declared choice rather than the test winner.
3. **Stacking, the standard machine-learning answer, is the worst learned rule.** With six specialists, eleven classes and only 786 validation lesions, a free logistic regression over 66 inputs overfits; the evidence-structured rules generalise better.
4. **The original formula, once its reliability is learned, is a strong baseline.** The prototype's idea was sound; its numbers were not.
5. **The oracle shows large headroom.** If the best specialist could be identified for each case, macro-F1 would rise to 0.59. Selection, not the specialists themselves, is the main bottleneck.

![Mean fusion weight given to each specialist on the test set, without and with dermoscopy.](paper/figures/fig_weights.png)

The weights show what the gate learned. Without dermoscopy, the details head receives the largest weight (0.41), followed by the embedding classifier (0.22), the photograph head (0.16) and the similar-case vote (0.14). With dermoscopy, weight moves to the dermoscopy head (0.18) and the embedding classifier (0.21). The external Model A always receives a small weight (0.07). The features the gate relies on most are confidence, margin, agreement, consistency and the out-of-distribution distance.

![What the learned gate listens to: mean absolute coefficient per evidence feature (standardised).](paper/figures/fig_gate_importance.png)

**Which specialists are worth having?** Refitting the selected rule on all 31 subsets of specialists in stage A and 63 in stage B shows that the full team is close to the best subset on validation (0.412 vs 0.419 in stage A; 0.517 vs 0.550 in stage B). The best stage-B subset (dermoscopy, details, Model A and embedding classifier) also scored higher on test (0.460 vs 0.438), but it was not chosen, because choosing among 63 subsets on 786 validation lesions would itself overfit. The more important lesson is that the S6 embedding classifier appears in every top subset: adding it raised photograph-only macro-F1 of the selected rule from 0.332 to 0.377 and accuracy with dermoscopy from 0.719 to 0.731 compared with the five-specialist team.

**Calibration.** The isotonic map makes the shown confidence honest: on test, the mean confidence shown is 0.757 against an actual accuracy of 0.731 with dermoscopy, and 0.660 against 0.657 without.

![Reliability diagram on the test set: stated confidence against actual accuracy for the raw fused probability and the calibrated confidence that the application shows.](paper/figures/fig_calibration.png)

![Confusion matrix of the selected fusion with dermoscopy (test set). Rows are true diagnoses with their number of lesions.](paper/figures/fig_confusion.png)

The confusion matrix shows where the errors are: actinic keratoses are confused with BCC and SCC, benign keratoses with BCC and nevi, and the four rare classes (BEN_OTH, INF, MAL_OTH and VASC with 1 to 8 test lesions) are almost never predicted.

![Per-class F1 on the test set: the evidential network alone against the full D-GUARD team. The team gains most on BKL (+0.35), BEN_OTH (+0.29), AKIEC (+0.23) and INF (+0.18), the classes the evidential network handled worst. It loses on VASC (0.88 to 0.36, 8 test lesions), where the dermoscopy head alone was best, and MAL_OTH is never predicted.](paper/figures/fig_per_class_f1.png)

## The Decision Policy

Table: What patients receive (test set, missed-cancer cost 5).
| Outcome | Share of patients | Accuracy | True diagnosis in top 3 | Missed cancers |
|---|---|---|---|---|
| HIGH-confidence answer | **57.1%** | **77.1%** | - | **3.1%** |
| LOW-confidence answer (doctor confirms) | 32.0% | 49.6% | 85.7% | 21.0% |
| Deferred (evidence contradicts itself) | 10.9% | - | - | - |
| Answered in total | 89.1% | 67.2% | | |
| Dermoscopy requested on the way | 29.5% | | | |

Nine in ten patients receive an answer; more than half receive a confident one, which is right 77% of the time across eleven diagnoses and calls 3.1% of the cancers it answers benign. LOW answers are much less reliable (49.6% exactly right, 21% of their cancers called benign), which is precisely why they are marked as such and always go to a doctor; even so, the true diagnosis is in their top three for 86% of patients, which is useful to the confirming doctor. Learn-then-Test certified a threshold at both stages (lambda-hat = 0.218 without and 0.276 with dermoscopy), so the 5% bound on missed cancers among HIGH answers holds with 90% probability if future patients resemble the calibration patients.

Table: The decision policy against baselines on the same probabilities (test, cost 5). Coverage is the share answered; B3 and the learned rejector are matched to DERMA-GUARD's coverage.
| Policy | Answered | Dermoscopy | Accuracy | Balanced acc. | Missed cancers | False alarms | Mean cost |
|---|---|---|---|---|---|---|---|
| B1 always answer, photo only | 100% | 0% | 0.657 | 0.360 | 7.2% | 47.6% | **0.759** |
| B2 always acquire dermoscopy | 100% | 100% | 0.731 | 0.418 | 6.4% | 38.0% | 1.124 |
| B3 confidence threshold | 57.1% | 0% | **0.815** | 0.413 | 3.8% | 39.3% | 0.881 |
| B3 + dermoscopy for everyone | 57.1% | 100% | **0.893** | **0.520** | **2.4%** | 32.9% | 1.293 |
| B4 learned rejector (L2D style) | 57.1% | 0% | 0.764 | - | 4.9% | - | - |
| B5 random dermoscopy requests | 100% | 29.5% | 0.682 | 0.361 | 6.5% | 44.7% | 0.842 |
| B5 dermoscopy for the most uncertain | 100% | 29.5% | 0.700 | 0.386 | 5.6% | 45.2% | 0.790 |
| **DERMA-GUARD (HIGH tier)** | **57.1%** | **29.5%** | 0.771 | 0.305 | **3.1%** | 50.0% | 1.047 |

![Risk-coverage view: share of answered cancers called benign as a function of the share of patients answered. The curves are abstention by confidence and by 1 - ignorance; the points are DERMA-GUARD's HIGH tier and the learned rejector at the same coverage.](paper/figures/fig_risk_coverage.png)

At the same coverage, DERMA-GUARD's HIGH answers miss fewer cancers than every abstention baseline: 3.1% against 3.8% for confidence thresholding and 4.9% for a learned rejector trained to predict the model's own errors. The paired bootstrap shows that the difference from the learned rejector is significant (1.8 points, 95% CI 0.2 to 3.5, one-sided p = 0.011) and that the difference from the confidence threshold is not (0.7 points, CI -0.7 to 2.0, p = 0.18). Against us: confidence thresholding is *more accurate* (81.5% vs 77.1%), because it answers only the photograph's easiest cases, whereas DERMA-GUARD's HIGH tier also includes harder cases that became answerable after dermoscopy; giving dermoscopy to everyone and then thresholding is better still on every metric, but costs a dermoscopy for every patient; and always answering is cheapest on mean cost, because every case that goes to a doctor is charged 1.5. HIGH answers also raise more false alarms (50%), because with a 5:2 cost ratio "possibly malignant, see a dermatologist" is the safer statement under doubt.

**Is the gap to simple thresholding real?** The most important analysis in this report repeats the identical policy on four fusion back-ends: the iterative EM and the learned gate, each with five specialists (before S6 was added) and with six.

![The same decision policy on four fusion back-ends. DERMA-GUARD's miss rate stays between 3.0% and 3.5%; confidence thresholding at the same coverage swings from 3.5% to 9.3%.](paper/figures/fig_backend_robustness.png)

Table: The policy across four fusion back-ends (cost 5, all cross-fitted).
| Fusion back-end | HIGH share | HIGH accuracy | DERMA-GUARD missed | Threshold missed | AE policy | AE most-uncertain |
|---|---|---|---|---|---|---|
| Iterative EM, 5 specialists | 57.2% | 0.762 | 3.3% | 9.3% | +0.30 | -0.04 |
| Learned gate, 5 specialists | 54.4% | 0.769 | 3.5% | 3.5% | +0.20 | +0.20 |
| Iterative EM, 6 specialists | 59.2% | 0.777 | 3.0% | 4.7% | +0.19 | +0.02 |
| Learned gate, 6 specialists | 57.1% | 0.771 | 3.1% | 3.8% | +0.07 | +0.26 |

DERMA-GUARD's miss rate hardly moves (3.0% to 3.5%). The threshold's moves by a factor of almost three (3.5% to 9.3%). A confidence threshold is exactly as safe as the ranking it is given; with one fusion it is as safe as DERMA-GUARD, with another it misses nearly three times as many cancers, and there is no way to know which situation one is in without test data. The calibrated policy delivers a stable, certified miss rate whichever fusion it receives. The same table shows a result against us: the learned acquisition estimator chose whom to ask for dermoscopy much better than "ask the most uncertain" on the iterative fusion (acquisition efficiency +0.30 vs -0.04), but on the selected six-specialist gate the simple rule is better (+0.26 vs +0.07). Similarly, ignorance ranked errors better than softmax confidence on the iterative fusion (AUGRC 0.0874 vs 0.0918) but not on the final gate (0.0820 vs 0.0757). Once the fused probabilities are good, simple uncertainty becomes a good enough guide.

## Sensitivity to the Cost of a Missed Cancer

Table: Effect of the missed-cancer cost c_miss (learned gate, six specialists).
| c_miss | Answered | HIGH share | HIGH accuracy | Missed (HIGH) | Threshold missed |
|---|---|---|---|---|---|
| 2 | 88.8% | 61.6% | 0.763 | 4.1% | 4.1% |
| 3 | 88.9% | 59.1% | 0.768 | 3.9% | 4.3% |
| **5** | **89.1%** | **57.1%** | **0.771** | **3.1%** | **3.8%** |
| 10 | 89.2% | 53.4% | 0.786 | 1.9% | 3.8% |

![Outcome shares (bars) and the missed-cancer rate of HIGH answers (line) as the cost of a missed cancer increases.](paper/figures/fig_outcomes_cost.png)

The cost of a missed cancer is a dial: raising it from 2 to 10 lowers the HIGH share from 62% to 53% and halves the miss rate (4.1% to 1.9%), while the share of patients who get some answer stays near 89%. At cost 10 the policy misses half as many cancers as thresholding at the same coverage.

## Statistics and Fairness

Table: Headline quantities with 95% bootstrap confidence intervals (2,000 resamples of the test set).
| Quantity | Value | 95% CI |
|---|---|---|
| Answered | 89.1% | 86.8% - 91.2% |
| HIGH-confidence share | 57.1% | 53.5% - 60.6% |
| HIGH accuracy | 77.1% | 73.0% - 80.8% |
| Missed cancers, HIGH | 3.1% | 1.6% - 4.8% |
| Missed cancers, confidence threshold | 3.8% | 2.1% - 5.6% |
| Missed cancers, learned rejector | 4.9% | 3.0% - 7.0% |
| Team fusion macro-F1 (stage B) | 0.438 | 0.360 - 0.503 |
| Dempster-Shafer macro-F1 (stage B) | 0.376 | 0.333 - 0.414 |

Table: Paired bootstrap tests.
| Comparison | Difference | 95% CI | One-sided p |
|---|---|---|---|
| Missed: learned rejector minus DERMA-GUARD | +1.8 points | +0.2 to +3.5 | **0.011** |
| Missed: confidence threshold minus DERMA-GUARD | +0.7 points | -0.7 to +2.0 | 0.18 |
| Macro-F1: team fusion minus Dempster-Shafer | +0.062 | -0.006 to +0.121 | **0.038** |
| Missed: learned gate minus iterative EM | +0.1 points | -0.5 to +0.7 | n.s. |
| HIGH share: learned gate minus iterative EM | -2.2 points | -3.8 to -0.4 | significant |

Table: Results by MILK10k skin-tone class (test set).
| Skin tone | Lesions | HIGH share | HIGH accuracy | Missed cancers (HIGH) |
|---|---|---|---|---|
| 1 (lightest) | 14 | 14% | 50% | 0% (too few to judge) |
| 2 | 67 | 39% | 81% | **11.5%** |
| 3 | 489 | 64% | 77% | 2.2% |
| 4 | 167 | 53% | 76% | 4.5% |
| 5 | 50 | 42% | 76% | 0% |

![Share of HIGH answers and missed cancers among them, by MILK10k skin-tone class (test set).](paper/figures/fig_skin_tone.png)

Fairness is a real weakness. Patients with skin tone 2 receive fewer confident answers, and among those answers 11.5% of cancers are missed, against 2.2% for tone 3. Tone 3 dominates MILK10k (62% of test lesions), and the calibration is fitted on the whole validation set, so its guarantee is an average that does not protect small groups. Class-conditional (Mondrian) calibration by skin tone is the obvious remedy, but with 14 to 67 lesions per light-tone group there is not enough data to fit it; we report the gap rather than hide it.

**Net benefit.** For the decision "refer as possibly malignant", DERMA-GUARD's rule (refer unless a HIGH-confidence benign answer is given) has a net benefit of 0.628 at a threshold probability of 0.3, slightly above a tuned probability threshold (0.626) and referring everyone (0.622); at 0.5 it is more conservative (0.507 vs 0.548).

## Robustness: DERMA-STRESS

To test whether the system notices bad evidence, the photograph of 300 test lesions was damaged in six ways without any retraining: Gaussian blur, darkening, noise, heavy JPEG compression, a black square over the centre (occlusion), and removing the patient details.

Table: DERMA-STRESS on 300 test lesions (photograph + details). "Model alone" is without the pipeline's quality and missing-details checks; "Full system" is with them.
| Condition | Mean ignorance | HIGH (model alone) | Missed (model alone) | HIGH (full system) | Missed (full system) | Photo flagged |
|---|---|---|---|---|---|---|
| Clean | 0.679 | 46% | 4.3% | 43% | 4.7% | 8% |
| Blur | **0.605 (down)** | **50% (up)** | 2.0% | **0%** | - | 100% |
| Dark | 0.723 | 22% | **12.3%** | **0%** | - | 100% |
| Noise | 0.824 | 16% | 6.1% | **0%** | - | 100% |
| JPEG | 0.741 | 27% | 5.0% | 22% | 1.5% | 24% |
| Occlusion | 0.634 | 16% | **14.3%** | **0%** | - | 100% |
| No details | 0.754 | 48% | 5.6% | **0%** (details requested) | - | 8% |

![DERMA-STRESS: share of HIGH answers and missed cancers among them, for the model alone and with DERMA-GUARD's checks.](paper/figures/fig_stress.png)

The experiment exposes a real limitation of evidential uncertainty. A blurred photograph makes the network *more* confident: mean ignorance falls from 0.679 to 0.605 and the share of HIGH answers rises from 46% to 50%. Under-exposed and occluded photographs keep 16% to 22% HIGH answers that miss 12% to 14% of cancers. Ignorance alone notices none of this. The explicit checks do: the quality agent flags every blurred, dark, noisy and occluded photograph, and the missing-details gate turns every case without details into a request. With these checks, no damaged photograph receives a HIGH answer. The price is that 8% of clean photographs are also flagged and receive a LOW answer and a retake prompt instead of a HIGH answer.

In an earlier version of the system, before the occlusion, grain and missing-details checks existed, occluded photographs kept HIGH answers that missed 9% of cancers and, with the five-specialist learned gate, removing the details led to 17.6% missed cancers among HIGH answers, because the gate relied on the similar-case vote whose embedding includes the details. These failures are why the checks were added.

## External Tests Without Retraining

The calibration guarantees assume that future patients resemble the calibration patients. To see what happens when they do not, the complete system was run, unchanged, on two other datasets.

Table: DERMA-GUARD on two external datasets without any retraining or recalibration.
| | MILK10k test | PAD-UFES-20 | Derm7pt |
|---|---|---|---|
| Lesions / photographs | 787 | 2,298 photos | 1,003 |
| Dermoscopy available | yes | no | yes (when requested) |
| Malignant share | 72% | 79% (MILK definition) | 29% |
| Mean ignorance, photo stage | 0.676 | 0.748 | 0.797 |
| Photographs failing the quality check | 8% | 28% | 29% |
| Out-of-distribution flags | 1% | 6% | 2% |
| Dermoscopy requested | 30% | 28% (cannot be provided) | 39% |
| Accuracy, all cases | 0.731 (with dermoscopy) | 0.367 | 0.479 (photo only: 0.402) |
| Malignant AUROC | 0.885 | 0.840 | 0.738 |
| HIGH-confidence share | 57.1% | 11.4% | 13.0% |
| Accuracy of HIGH answers | 0.771 | 0.517 | 0.708 |
| Missed cancers, HIGH | 3.1% | **0.0%** | 3.8% |
| Threshold at same coverage: missed / accuracy | 3.8% / 0.815 | 0.0% / 0.574 | 5.4% / 0.846 |

![MILK10k and the two external datasets: share of HIGH answers, their accuracy, and missed cancers among them.](paper/figures/fig_shift.png)

The system **notices** both shifts: ignorance, quality flags and deferrals rise, and the share of confident answers falls from 57% to 11-13%. Among the confident answers that remain, the safety target holds on both datasets: no cancer was missed on PAD-UFES-20, and 3.8% on Derm7pt (below the 5% target and below thresholding's 5.4%). On Derm7pt, the only external dataset with dermoscopy, the "ask for dermoscopy" loop does what it was designed to do: the system asked for it for 39% of lesions and overall accuracy rose from 0.402 to 0.479. What does **not** transfer is the fine-grained diagnosis. On PAD-UFES-20 only half of the confident answers name the right diagnosis, and on Derm7pt melanoma recall is 31% and dermatofibromas and vascular lesions are almost never recognised. A confidence threshold at the same coverage is more accurate on both datasets. DERMA-GUARD therefore protects the decision that matters most for safety (do not call a cancer benign with confidence) but not the eleven-class diagnosis, when the population changes.

## The Earlier Prototype's Models Revisited

Section 4.5 showed that the earlier prototype's seven models reach 57% to 75% on patients never seen in training, against reported accuracies of 84% to 90%. Those models cannot join the MILK10k team (Models B to G need PAD-UFES-20's questionnaire fields), but they can be fused with DERMA-GUARD's own learned gate on PAD-UFES-20. Using only the 132 test photographs whose patients were never in training, and five-fold cross-validation to fit the gate, the fused seven models reach **72.7%** accuracy, against **59.8%** for a simple average of the same models; the best single model (metadata only) reaches 75.0% on the same photographs. The learned gate therefore recovers almost all of the value of the best model without being told which one it is, while naive averaging is dragged down by the weak text and image models. This is the same pattern seen on MILK10k, now on an independent team of models.

## Do the Warning Signs Mean Anything?

The scribe raises triage urgency when the patient mentions bleeding, growth, change, itching or pain, a new spot or a raised lump. MILK10k has no symptoms, but PAD-UFES-20 recorded patients' answers to exactly these questions next to the biopsy-confirmed diagnosis.

Table: Cancer (BCC, SCC or melanoma) rate with and without each warning sign, PAD-UFES-20, one row per lesion (1,641 lesions).
| Warning sign | Lesions with the sign | Cancer if yes | Cancer if no | Relative risk (95% CI) |
|---|---|---|---|---|
| Bleeding | 400 | 85.0% | 32.0% | 2.66 (2.43 - 2.91) |
| Rapid growth | 583 | 52.7% | 17.5% | 3.00 (2.51 - 3.59) |
| Change | 113 | 61.1% | 30.8% | 1.98 (1.67 - 2.35) |
| Itching | 1,024 | 52.8% | 32.1% | 1.65 (1.45 - 1.87) |
| Pain | 270 | 84.8% | 37.2% | 2.28 (2.09 - 2.48) |
| Raised | 1,009 | 60.1% | 21.1% | 2.84 (2.43 - 3.34) |
| **Any of the scribe's signs** | 1,288 | **48.7%** | **12.4%** | **3.91 (2.77 - 5.52)** |

![Relative risk of cancer for each warning sign on the PAD-UFES-20 questionnaire, with 95% confidence intervals.](paper/figures/fig_symptoms.png)

Every warning sign raises the cancer rate, so using them to raise urgency is justified; a lesion with any sign is almost four times as likely to be malignant. "Raised" was not originally a flag and was added to the scribe as a result of this analysis. The evidence has limits: these are questionnaire answers collected by clinicians, not free text written by patients, and the population is a Brazilian referral programme. As designed, the signs change only urgency, never the diagnosis.

## Summary of Findings

Table: Summary of the main findings.
| Question | Answer | Evidence |
|---|---|---|
| Does weighting a team of specialists help? | Yes: macro-F1 0.286 to 0.377 without dermoscopy; best accuracy and AUROC with it | Section 8.2 |
| Is learned weighting better than averaging? | Yes, every structured rule beats averaging; free stacking does not | Section 8.2 |
| How many patients get an answer? | 89% in total, 57% with high confidence | Section 8.3 |
| Are confident answers safe? | 3.1% of cancers missed, certified below 5% with 90% probability | Section 8.3 |
| Is the policy safer than abstention? | Than a learned rejector, yes (p = 0.011); than thresholding, not significantly | Sections 8.3, 8.5 |
| Is its safety robust? | Yes: 3.0-3.5% across four fusion back-ends; thresholding 3.5-9.3% | Section 8.3 |
| Does it notice damaged input? | Only with explicit checks; ignorance alone does not | Section 8.6 |
| Does it work on other populations? | Safety yes; fine diagnosis no | Section 8.7 |
| Is it fair across skin tones? | No: tone 2 has 11.5% missed cancers | Section 8.5 |
