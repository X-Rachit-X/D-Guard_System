# Literature Survey

This chapter reviews the research that the project builds on. It is organised by the questions the system has to answer: how to classify skin lesions from images and patient data, how a model can express what it does not know, how several sources of evidence can be combined, how to turn a prediction into a safe decision, and what existing systems do and do not provide.

## Deep Learning for Skin-Lesion Classification

Esteva et al. [9] trained a GoogLeNet Inception-v3 network on about 130,000 clinical images and showed that it classified keratinocyte carcinomas and melanomas as well as 21 board-certified dermatologists on biopsy-proven test images. This result triggered a wave of work, supported by the International Skin Imaging Collaboration (ISIC) challenges [5] and public datasets such as HAM10000 [31], which contains 10,015 dermoscopic images across seven diagnostic categories. Residual networks [13] became the standard backbone; their identity shortcuts allow very deep networks to train reliably, and ImageNet-pretrained ResNet50 remains a common starting point for medical transfer learning.

More recently, dermatology foundation models have appeared. PanDerm [33] was pre-trained on more than two million dermatology images across several imaging modalities and reports strong results on many downstream tasks. Such models are the natural next backbone for this project; we did not use one for the main experiments because the licence terms of the available weights restrict fine-tuning, and because we wanted every component to be trainable within free GPU limits.

A recurring weakness of this literature, pointed out by several reviews, is evaluation. Splits made at the image level rather than the patient or lesion level let photographs of the same lesion appear in both training and test sets and inflate accuracy. Our own earlier prototype made exactly this mistake (Chapter 4).

## Multimodal Dermatology

Clinicians never diagnose from an image alone; they also use the patient's age, the lesion's location, its history and, when available, dermoscopy. PAD-UFES-20 [24] provides smartphone photographs together with rich patient metadata from a Brazilian dermatology programme. MetaBlock [25] showed that metadata can modulate image features through an attention-like gating mechanism and improves accuracy on PAD-UFES-20. Derm7pt [15] pairs a clinical photograph and a dermoscopic image for about one thousand lesions and annotates the seven-point checklist used by dermatologists; Kawahara et al. trained a multitask network that predicts the diagnosis and the checklist criteria from both images. MILK10k [22], released in 2025, is the largest public dataset in which every lesion has both a clinical close-up and a dermoscopic image, with eleven diagnostic classes and largely histopathological confirmation. Multi-task systems such as those reviewed by Mehta et al. [21] can also recommend when dermoscopy is needed.

These works improve prediction. None of them decides *whether* a prediction should be issued at all, or asks for missing evidence when it would help.

## Uncertainty and Evidential Learning

A standard softmax classifier cannot distinguish "I have strong evidence for two classes" from "I have no evidence at all"; both can produce similar probability vectors. Evidential Deep Learning (EDL) [28] addresses this by treating the network's outputs as evidence for each class and placing a Dirichlet distribution over the class probabilities. The total amount of evidence determines how concentrated the Dirichlet is, and the remaining mass is interpreted as ignorance. EDL is grounded in subjective logic [14], which itself derives from the Dempster-Shafer theory of evidence [7], [29].

Calibration is a related but separate problem: a model may know what it knows and still report numbers that are too high. Guo et al. [11] showed that modern deep networks are systematically over-confident and that a single temperature parameter fixes much of this. Isotonic regression [34] is a non-parametric alternative that maps scores to observed frequencies. For detecting inputs unlike the training data, the Mahalanobis distance of a feature vector to class means under a shared covariance [17] is simple and effective, and the Ledoit-Wolf estimator [16] makes the covariance estimate stable in high dimensions.

## Combining Evidence from Several Sources

The simplest way to combine modalities is to concatenate their features. Its weakness is that every modality gets a fixed slot: an uninformative input adds noise on every case. Trusted Multi-View Classification (TMC) [12] instead lets each view produce an evidential opinion and combines the opinions with Dempster's rule. A view with little evidence then has little influence, and the rule produces a conflict mass when confident views disagree. This is the fusion we use inside the evidential network.

Beyond fixed rules, the classical machine-learning toolbox offers stacking (learning a meta-classifier on the outputs of base models), mixture-of-experts gating (learning per-input weights), and the Expectation-Maximisation algorithm [8] for problems where a latent variable (here, the unknown diagnosis) and parameters (here, how much to trust each source) must be estimated together. Our earlier prototype used an EM-style alternation with hand-set priors. Nearest-neighbour classification [6] and retrieval-augmented generation [18] provide case-based evidence; we use both, but keep retrieved documents out of the calibrated decision.

## From Prediction to Decision: Abstention, Deferral and Risk Control

Chow [4] gave the optimal rule for when a classifier should reject an input: reject when the expected loss of the best answer exceeds the cost of rejecting. Selective classification [10] applies the same idea to deep networks by abstaining on low-confidence inputs, trading coverage for accuracy. Learning to defer [23] trains a separate rejector to decide when a human expert should answer instead. Traub et al. [30] showed that the common area-under-risk-coverage metric is flawed and proposed AUGRC, which we report.

Conformal prediction [32], [1] produces prediction *sets* that contain the true label with a user-chosen probability, without assumptions about the model, as long as the calibration and test data are exchangeable. Adaptive prediction sets [26] adapt to difficulty, least-ambiguous sets [27] are smallest on average, and class-conditional (Mondrian) calibration protects subgroups; Lu et al. [20] apply this to fairness in medical imaging. Learn-then-Test [2] and conformal risk control [3] extend the guarantee from coverage to general risks, such as the rate of missed cancers among confident answers. This is exactly the kind of guarantee a clinical decision system needs.

## Comparison of Existing Approaches

Table: Capabilities of related systems compared with DERMA-GUARD (yes = present, partial, no = absent), as described in the cited work.
| System | Multimodal | Per-case weights | Evidential uncertainty | Asks for evidence | Defers | Risk guarantee | Stress tests | Care platform |
|---|---|---|---|---|---|---|---|---|
| Esteva et al. [9] | no | no | no | no | no | no | no | no |
| MetaBlock [25] | yes | partial | no | no | no | no | no | no |
| Derm7pt multitask [15] | yes | no | no | no | no | no | no | no |
| TMC [12] | yes | yes | yes | no | no | no | no | no |
| Selective classification [10] | no | no | no | no | yes | partial | no | no |
| Learning to defer [23] | no | no | no | no | yes | no | no | no |
| Learn-then-Test [2] | no | no | no | no | partial | yes | no | no |
| Our earlier prototype | yes | yes (hand-set) | no | partial | yes | no | partial | partial |
| **DERMA-GUARD (this work)** | **yes** | **yes (learned)** | **yes** | **yes** | **yes** | **yes** | **yes** | **yes** |

## Research Gap

The survey shows that the individual building blocks exist: evidential learning, Dempster-Shafer fusion, conformal prediction, risk control, deferral and multimodal datasets. What is missing is their **integration into one calibrated decision** that (i) weighs multiple specialists per case with learned trust, (ii) chooses between answering, asking for evidence, answering provisionally and deferring, (iii) comes with a certified bound on the error that matters most, (iv) is measured honestly against strong but simple baselines on the same probabilities, and (v) is wrapped in a usable clinical workflow. Paired clinical-dermoscopic data, now available through MILK10k, also makes it possible for the first time to measure the real value of asking for dermoscopy instead of simulating it. This project addresses that gap.
