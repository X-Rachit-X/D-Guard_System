# Theoretical Background

This chapter explains the ideas the system is built on. Each section starts with the intuition and then gives the formula actually used in the code, so that the later chapters can refer back to it.

## Evidence Instead of Probabilities

A normal classifier ends with a softmax layer. For eleven diagnoses it outputs eleven numbers that are positive and add up to one. That constraint is the root of over-confidence: even when the input is a black square, the numbers still add to one, and one of them will be the largest.

Evidential deep learning [@edl] changes what the network outputs. Instead of probabilities, it outputs a non-negative amount of **evidence** for each class, e_k, typically through a softplus activation. These are turned into the parameters of a Dirichlet distribution, a distribution over possible probability vectors:

$$ alpha_k = e_k + 1,     S = sum over k of alpha_k

From these, subjective logic defines a **belief** for every class and one extra number, the **ignorance** (also called uncertainty mass):

$$ b_k = e_k / S,     u = K / S,     with  sum of b_k + u = 1

When the network sees strong evidence, S is large, the beliefs are large and u is small. When it sees nothing useful, all e_k are near zero, S is close to K, and u is close to 1: the model is saying "I don't know" instead of picking a class at random. For decisions that need a probability, the belief is converted to the **pignistic probability**, which spreads the ignorance evenly:

$$ p_k = b_k + u / K

![Evidence in practice: belief per class (blue) and ignorance u (red) of the photograph head for two real test lesions, the one with the most evidence and the one with the least. The bars always sum to one; when the head has little evidence, u is large instead of a random class looking likely.](paper/figures/fig_evidential_example.png)

The network is trained with the evidential loss of Sensoy et al. [@edl], an expected cross-entropy under the Dirichlet plus a regulariser that removes evidence for wrong classes. The regulariser is introduced gradually ("annealed") over the first ten epochs.

## Combining Opinions: Dempster's Rule

Suppose the photograph and the dermoscopic image each give an opinion (b_1, u_1) and (b_2, u_2). Dempster-Shafer theory [@shafer] combines them with Dempster's rule, written for subjective-logic opinions as in TMC [@tmc]. First the **conflict** is measured: how much belief the two sources put on *different* classes,

$$ C = sum over i != j of b_1,i * b_2,j

and then the combined opinion is

$$ b_k = ( b_1,k b_2,k + b_1,k u_2 + b_2,k u_1 ) / (1 - C),     u = u_1 u_2 / (1 - C)

Three properties make this rule attractive here. First, a source with high ignorance contributes little: if u_2 = 1 the second source is the identity element and the result equals the first opinion exactly, which is how a missing modality is handled. Second, two ignorant sources do not "agree"; their combination stays ignorant. Third, conflict is a separate number from ignorance: two sources that confidently say different things produce high C, which our policy treats as a reason to defer to a human, whereas high ignorance is a reason to ask for more evidence.

## Measuring Agreement and Unusualness

Two simpler measures appear throughout the system. The **Jensen-Shannon divergence** between two probability vectors p and q is a symmetric, bounded version of the Kullback-Leibler divergence,

$$ JS(p, q) = 1/2 KL(p || m) + 1/2 KL(q || m),     m = (p + q) / 2

and the **consistency** of a team of M specialists is one minus their average pairwise divergence (normalised by log 2). The **normalised entropy** of a distribution, H(p) / log K, is used as a per-specialist uncertainty U between 0 (certain) and 1 (uniform).

For **out-of-distribution** detection, each training lesion's embedding z (the vector the network computes just before its classification heads) is summarised by class means mu_c and a shared covariance Sigma, estimated with Ledoit-Wolf shrinkage. A new case's score is its distance to the nearest class mean [@lee],

$$ d(z) = min over c of sqrt( (z - mu_c)^T Sigma^-1 (z - mu_c) )

and a case is flagged when d exceeds the 99th percentile of validation distances.

## Calibration

A confidence is *calibrated* if, among all answers given with confidence 0.7, about 70% are correct. Modern networks are usually over-confident [@guo]. The project uses two tools. **Temperature scaling** divides log-probabilities by a learned T before renormalising; T > 1 softens over-confident outputs. **Isotonic regression** [@zadrozny] learns a non-decreasing step function from a raw score to the observed accuracy on validation data. The confidence that the application shows to patients and doctors is the isotonic estimate of "the probability that the stated diagnosis is correct".

## Conformal Prediction

Conformal prediction wraps any classifier and produces a set of labels guaranteed to contain the true label with probability at least 1 - alpha, for a user-chosen alpha. The recipe for the least-ambiguous set (LAC) [@lac] is:

1. On n calibration cases, compute the score s_i = 1 - p(true label of case i).
2. Take q-hat as the ceiling((n + 1)(1 - alpha)) / n quantile of these scores.
3. For a new case, include every label y with 1 - p(y) <= q-hat.

The only assumption is that calibration and test cases are exchangeable (roughly, drawn from the same distribution). DERMA-GUARD applies conformal prediction to the question that matters for safety, **benign or malignant?**, with separate (Mondrian) quantiles for each group [@lu]. If the set contains "malignant", the system is not allowed to give a confident benign answer.

## Learn-then-Test Risk Control

Conformal prediction controls coverage. A clinical system needs to control a specific risk instead: among the cases it answers confidently, how many cancers does it call benign? Learn-then-Test (LTT) [@ltt] answers this. For a family of thresholds lambda on a confidence score, it tests the null hypothesis "the risk at lambda is above alpha" on calibration data using a valid p-value (here the Hoeffding-Bentkus bound), and keeps only thresholds where the null is rejected. Choosing the most permissive surviving threshold gives the guarantee

$$ P( risk(lambda-hat) <= alpha ) >= 1 - delta

with alpha = 5% (at most one in twenty answered cancers missed) and delta = 10%. Like conformal prediction, the guarantee assumes exchangeability; Chapter 8 tests what happens on other datasets.

## Cost-Sensitive Decisions

Not all errors are equal. The policy uses an explicit cost matrix L(y, y-hat): 0 for a correct answer, 1 for confusing two diagnoses of the same group (two benign or two malignant), c_miss for calling a malignant lesion benign, and 2 for calling a benign lesion malignant. Acquiring a dermoscopic image costs 0.5 and a clinician review 1.5. The expected cost of answering y-hat under the predicted distribution p is

$$ r_answer = sum over y of p_y * L(y, y-hat)

and Chow's rule [@chow] would answer only when this is below the cost of deferring. The expected cost of acquiring dermoscopy is the predicted cost after acquisition plus 0.5. The main experiments use c_miss = 5; results for 2, 3 and 10 are reported to show the trade-off.

## Evaluating a System That Can Abstain

Ordinary accuracy is meaningless for a system that may refuse to answer: it can reach 100% by answering only the easiest case. Selective-prediction work [@geifman], [@traub] therefore reports **coverage** (the share of cases answered) together with the **risk** among answered cases, and compares systems at the *same* coverage. We use the following metrics:

- **Coverage** of high-confidence answers and **accuracy** among them.
- **Missed-cancer rate**: of the malignant lesions that received a confident answer, the share called benign.
- **Macro-F1** and **balanced accuracy**, which weight every diagnosis equally and therefore reveal whether rare diagnoses are handled.
- **Malignant-vs-benign AUROC**, which measures ranking independent of any threshold.
- **AUGRC** [@traub], the area under the generalised risk-coverage curve, summarising abstention quality across all coverages.
- **Acquisition efficiency**, AE = (G_policy - G_random) / (G_oracle - G_random), where G is the reduction in cost obtained by giving dermoscopy to a chosen set of patients; 1 means the oracle's choice, 0 means no better than random.
- **Bootstrap confidence intervals** (2,000 resamples of the test set) and paired bootstrap tests for differences between systems.
