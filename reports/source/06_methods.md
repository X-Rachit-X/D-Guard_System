# Methodology

This chapter describes each component of the system in the order in which a case passes through it: the specialists, the Devil's Advocate, the fusion rules and how one is chosen, the Dempster-Shafer signals, calibration, the decision policy, and the agents.

## The Specialist Team

### S1 to S3: the evidential network

A single network serves three specialists. A ResNet50 [13], initialised from ImageNet weights, is shared by the clinical photograph and the dermoscopic image: each image is resized to 224 x 224 pixels and mapped to a 2,048-dimensional feature, then projected to a 256-dimensional embedding by a modality-specific layer. The patient details (age, sex, general and specific body site, skin tone and the seven MONET scores, 31 numbers after one-hot encoding and standardisation) pass through a residual multilayer perceptron to another 256-dimensional embedding. Each of the three embeddings feeds its own evidential head, which outputs non-negative evidence for the eleven classes through a softplus. The three opinions are combined by Dempster's rule (Section 3.2) to produce the network's fused opinion, its ignorance u and its conflict C.

Table: Training configuration of the evidential network (Kaggle GPU).
| Setting | Value |
|---|---|
| Backbone | ResNet50, ImageNet initialisation, shared by both image types |
| Input size | 224 x 224, standard augmentation (flips, rotation, colour jitter) |
| Embedding size | 256 per modality |
| Loss | Evidential loss [28] with KL regulariser annealed over 10 epochs, class weights |
| Optimiser | AdamW, learning rate 1e-3 (heads) and 1e-4 (backbone), weight decay 1e-4 |
| Schedule | Cosine annealing over 25 epochs; backbone frozen for the first 3 |
| Modality dropout | 0.25 (a modality is randomly hidden so the network learns to work without it) |
| Batch size | 32, mixed precision |
| Model selection | Best validation macro-F1 (epoch 23 of 25), early stopping patience 8 |

Modality dropout matters for this project: because a modality is sometimes hidden during training, the network learns to give an honest "I don't know" for an absent input, which is what makes stage A (no dermoscopy) and stage B (with dermoscopy) consistent.

### S4: the legacy model

The image model of our earlier prototype (a ResNet50 feature extractor followed by channel attention and a two-layer self-attention encoder) was trained on PAD-UFES-20 smartphone photographs. It is applied unchanged to MILK10k clinical photographs. Its six classes are mapped onto six of the eleven MILK10k classes, and the distribution is mixed with 10% uniform probability so that the five classes it cannot name are "unknown" rather than "impossible".

### S5: the similar-case vote

For a new case, the embeddings computed by the network (photograph and details, plus dermoscopy when available) are compared by cosine similarity with those of all 3,667 training lesions. The 15 most similar lesions vote for their confirmed diagnosis with weight exp(5 x similarity), with Laplace smoothing. The mean similarity of the neighbours is also used as this specialist's quality signal.

### S6: the embedding classifier

The similar-case vote looks only at the nearest neighbours. The sixth specialist draws one linear boundary through *all* training embeddings: a multinomial logistic regression with class weights inversely proportional to class frequency, so that rare diagnoses are not drowned out by basal cell carcinoma. Its regularisation strength (C = 0.01 for both stages) was chosen from four candidates by validation macro-F1. On validation it scores 0.525 macro-F1 with dermoscopy, against 0.465 for the similar-case vote, and it turned out to be the most useful addition to the team (Section 8.2).

## The Devil's Advocate

The Devil's Advocate is the component that asks, for every specialist m and every case x, *how much should this opinion count here?* It computes the following signals.

Table: Evidence signals computed by the Devil's Advocate.
| Signal | Meaning | How it is computed |
|---|---|---|
| Quality Q | Is the input good? | Share of four image statistics within the 5th-95th percentile of training photographs; 1 for metadata; neighbour similarity for S5 |
| Reliability R | Is this specialist usually right in situations like this? | Logistic regression fitted on validation, predicting correctness from Q, U, confidence and agreement |
| Uncertainty U | How spread out is its opinion? | Normalised entropy |
| Confidence, margin | How sure is it? | Top probability; gap between top two |
| Agreement | Do the others agree with it? | Share of other specialists with the same top class |
| Track record | Is it good at the diagnosis we suspect? | Measured P(specialist correct given true class), Laplace-smoothed, at the provisional diagnosis |
| Consistency C | Does the team agree overall? | 1 minus mean pairwise Jensen-Shannon divergence |
| Missingness M | Is anything missing? | Share of absent specialists |
| OOD distance | Is this case unusual? | Mahalanobis distance to nearest class mean, divided by the validation 99th percentile |

The original prototype multiplied hand-set versions of these signals into a score E_m = Q R U C S M and normalised the scores into weights. Two problems are visible in that formula. Its numbers were typed in by hand, and consistency C and sufficiency S are the same for every specialist of a case, so they cancel when the weights are normalised; in practice only Q, R and the utility prior ever changed the weights. Version 2 lets every signal act and learns how.

## Fusion Rules

### Iterative diagnosis-adaptive fusion (v1)

The trustworthiness of a source depends on the diagnosis (dermoscopy is very reliable for squamous cell carcinoma and much less so for benign keratoses), but the diagnosis is not known until the sources are combined. The earlier prototype resolved this circularity with an EM-style loop [8], which we kept and made measurable:

1. Start with weights w_m proportional to (Q_m R_m)^a and fuse: p = sum of w_m p_m. Take the provisional diagnosis y-hat = argmax p.
2. Look up each specialist's measured track record pi_m(y-hat) and a penalty kappa_m = 1 if the specialist agrees with y-hat, otherwise rho.
3. Recompute w_m = softmax over m of ( a log(Q_m R_m) + b log pi_m + g log kappa_m ) and fuse again.
4. Repeat at most three times, until the diagnosis is stable and no weight moves by more than 0.05.

The coefficients a, b, g, rho are chosen by grid search on validation macro-F1, and a temperature T by validation log-likelihood.

### Learned evidence gate (v2)

Version 2 replaces the hand-shaped formula with a learned gate. For each specialist it builds the feature vector of all ten Devil's Advocate signals, phi_m = [log Q, log R, U, confidence, margin, agreement, log track record, C, M, OOD distance], standardises it, and computes

$$ w_m(x) = softmax over present specialists of ( beta_m + theta_m . phi_m(x) )

Two ways of pooling the opinions are compared: a **mixture**, p = sum of w_m p_m, and a **product of experts**, p proportional to the product of p_m raised to w_m. The parameters theta, beta and a temperature are fitted by minimising the negative log-likelihood of the true diagnosis on validation, with an L2 penalty, using L-BFGS. Because each specialist has its own theta, the gate can learn, for example, that the dermoscopy head should be trusted more when its margin is large, while the details head should be trusted when the track record for the suspected diagnosis is high.

### The seven rules and how one is chosen

Table: The seven combination rules compared.
| Rule | What decides the weights | Learned? |
|---|---|---|
| Simple average | Nothing; every specialist counts equally | No |
| Original formula | E = Q R U (C and S cancel), with learned R | Partly |
| Best head per case | The single specialist with the highest R | Partly |
| Iterative EM (v1) | a log(QR) + b log(track record) + g log(penalty) | 4 parameters |
| Stacking | Multinomial logistic regression on all log-probabilities | Yes |
| Learned gate, mixture (v2) | All ten signals, per specialist | Yes |
| Learned gate, product (v2) | All ten signals, product pooling | Yes |

All seven rules are fitted the same way. The 786 validation lesions are split into five folds; each rule, together with the reliability models and track-record table it depends on, is fitted on four folds and applied to the fifth. The resulting **out-of-fold** probabilities serve two purposes. They are used to **select** the rule (best out-of-fold macro-F1, decided before any test result was seen), and they are the probabilities on which the decision policy is later **calibrated**, so the conformal sets and the Learn-then-Test threshold never see probabilities from a model fitted on the same lesions. The test set receives probabilities from rules fitted on the whole validation set.

Two further analyses help to understand the team. The **subset study** refits the selected rule on every non-empty subset of specialists (31 for stage A, 63 for stage B) to see whether any specialist is dead weight. The **oracle** picks, for each test case, the specialist that gave the true diagnosis the most probability; it uses the label and is therefore not a usable method, but it shows how much better per-case selection could still become.

## Ignorance, Conflict and Calibrated Confidence

In parallel with the team fusion, the three evidential heads are combined by Dempster's rule, giving the ignorance u and the conflict C (Section 3.2). Conflict is taken as the maximum over the successive pairwise combinations, so a contradiction between any two modalities is not hidden by a third. These two numbers drive the gates; the team's fused distribution provides the diagnosis.

The confidence shown to users is calibrated by isotonic regression [34] fitted on the out-of-fold validation predictions of both stages: it maps the top fused probability to the observed probability that the stated diagnosis is correct.

## The Decision Policy

The policy is implemented in src/decision/policy.py and is the only component that chooses an outcome. It operates in two stages: stage A uses the photograph and details; stage B adds dermoscopy, either because the patient provided it or because stage A asked for it.

### Gates

Table: Safety gates of the decision policy.
| Gate | Fires when | Effect |
|---|---|---|
| Conflict | Dempster-Shafer conflict above its validation 90th percentile | DEFER |
| False reassurance | The benign/malignant conformal set (alpha = 0.1, Mondrian LAC) still contains "malignant" while the answer would be benign | Blocks HIGH |
| Risk threshold | Confidence 1 - u below the Learn-then-Test threshold (alpha = 5%, delta = 10%) | Blocks HIGH |
| Photo quality | A severe quality issue (outside the 1st-99th percentile of training photographs, or a covered centre, or grain) | Blocks HIGH; app offers a retake |
| Out-of-distribution | Mahalanobis distance above the validation 99th percentile | Blocks HIGH |
| Missing details | No age, sex or site and no dermoscopy | ACQUIRE details |
| Cost | Expected cost of answering exceeds that of a review (1.5) | Blocks HIGH |

### Choosing the outcome

The stated diagnosis is the most probable class. Using the minimum-cost label instead (the classic form of Chow's rule) labelled 98.5% of lesions malignant on this referral population, so it was rejected (Section 9.3). The outcome is then chosen as follows.

1. If the conflict gate fires, or there is no usable evidence, the outcome is **DEFER**.
2. If no gate fires and answering is cheapest, the outcome is a **HIGH** answer.
3. Otherwise, if dermoscopy is still available and its expected cost r_acquire = r_answer - h-hat + 0.5 is lower than the cost of a review, the outcome is **ACQUIRE**. Here h-hat is the predicted reduction in cost from seeing the dermoscopic image, estimated by a gradient-boosted regressor fitted on validation lesions, for which the real dermoscopy is known.
4. Otherwise the outcome is a **LOW** answer: the diagnosis is stated, marked unreliable, and the case is queued for a doctor to confirm.

This "minimal deferral" design means that the system defers only on contradiction. An earlier design deferred on any failed check, so every patient without a HIGH answer went to a doctor with no answer at all; the LOW tier keeps those patients informed while still requiring confirmation. The HIGH tier is identical to a conventional defer-on-any-gate policy, so every guarantee and every comparison with baselines concerns exactly the same set of answers.

### Consensus levels

The four-level consensus hierarchy of the original design is kept as an interpretable summary: Level 1 (strong consensus) for a HIGH answer with at least three agreeing specialists, Level 2 for any other HIGH answer, Level 3 for an acquisition or a LOW answer, and Level 4 for a deferral.

## The Agents

Around the decision policy, five agents check, retrieve and write. None of them can change the decision itself, with the single exception that the quality and missing-details gates can only make it more careful.

- **Quality agent.** Measures sharpness (variance of the Laplacian), brightness, contrast, glare, graininess, and the share of perfectly flat 8 x 8 blocks in the central 60% of the image (a sticker, finger or covered lens). Thresholds are percentiles of 300 training photographs, with separate thresholds for dermoscopy. Issues beyond the 5th/95th percentile produce advice; issues beyond the 1st/99th are "severe" and block a HIGH answer.
- **Scribe.** Reads the patient's own description for warning signs (bleeding, rapid growth, change, a non-healing sore, itching or pain, a new spot, a raised lump), with negation handling so that "it does not bleed" is not a flag. A warning sign raises triage urgency; it never changes the diagnosis, because the model has no text input.
- **Librarian.** Retrieves the most similar confirmed training lesions with their photographs from a FAISS index built only from the training split; the index refuses to load if it was built from any other split, so no held-out label can leak into what a clinician reads.
- **Explainer.** Turns the Devil's Advocate's numbers into plain sentences: who said what, why dissenting specialists disagree (their measured track record for the suspected diagnosis, how spread out their opinion is, their input quality, what they can and cannot see), where the weight went, and what the confidence means.
- **Report agent.** Writes a structured report. If a local language model (Ollama) is available, it may rewrite the report and the explanation for readability, but a verifier rejects the rewrite if it names a diagnosis or quotes a percentage that is not in the evidence, or drops the low-confidence label or the requirement for confirmation.
