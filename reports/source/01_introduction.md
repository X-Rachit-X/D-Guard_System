# Introduction

## Background

Skin cancer is the most common cancer in the world. Basal cell carcinoma and squamous cell carcinoma are diagnosed in millions of people every year, and melanoma, although less common, causes most skin-cancer deaths. All three share one fact that makes them unusual among cancers: they are visible. A trained dermatologist can often recognise a suspicious lesion simply by looking at it, and with a dermatoscope (a hand-held magnifier with its own light that shows structures below the skin surface) the accuracy rises further. The problem is not that skin cancer is hard to see. The problem is that there are not enough trained eyes. In many regions, including large parts of India, a patient may wait months for a dermatology appointment, and a general practitioner without dermoscopy training has to decide alone whether a spot needs urgent attention.

This gap is why automated skin-lesion analysis has attracted so much research. Since the landmark result of Esteva et al. in 2017, deep convolutional networks have matched dermatologists on curated image sets, large public archives such as ISIC have grown to hundreds of thousands of images, and mobile apps that classify a photographed mole are already available to the public. On paper, the technology is ready.

In practice it is not, and the reason is less about accuracy than about behaviour. A classifier trained with a softmax output must assign probabilities that sum to one, so it always produces an answer, and usually a confident-looking one. It does so for a sharp, well-lit photograph and for a blurred one taken in a dark room. It does so when the patient's age and the lesion's location point to a different diagnosis from the image. It does so for a lesion unlike anything in its training data. And the number it reports as its confidence is, in most deployed systems, not calibrated: "92%" does not mean that 92% of such answers are right.

## Motivation

Consider how a careful dermatologist actually works. When the evidence is clear, they give a diagnosis. When it is not, they do one of three things. They **look more closely**, usually with a dermatoscope. They **give a provisional opinion** and ask a colleague or a biopsy to confirm it. Or, when the clinical picture and the history contradict each other, they **refer** the patient to someone with more information. What they almost never do is announce a confident diagnosis that the evidence does not support.

This project started from a simple question: *can an AI system be built that behaves like that?* Not one that is merely more accurate, but one that knows when its evidence is good enough to answer, asks for the missing piece when it is not, says clearly when an answer is only provisional, and hands over to a human when the evidence disagrees with itself.

There was also a more personal motivation. This report is the second version of the project. The first version, built earlier by our team, already had many of the right ideas: a panel of specialist models, a component that questioned the evidence (which we called the Devil's Advocate), and a fusion step that weighted each source by how trustworthy it was for the suspected diagnosis. When we came back to that prototype to extend it, we audited its evaluation and found that the numbers it reported could not be trusted (Chapter 4 explains why). Rebuilding the system so that every number is real, reproducible and fairly compared became as important a goal as the system itself.

The project is positioned against two **base papers** chosen from the literature (Chapter 2): HOT AI [@mehta], the closest task, which recommends a dermoscopic image when the clinical photograph is insufficient, and Learn-then-Test [@ltt], which provides the statistical guarantee. Both are either used directly or turned into a baseline that runs on the same probabilities as DERMA-GUARD; the related methods we compare against are described in Chapter 2.

## Problem Statement

Given a skin lesion described by a clinical photograph, optionally a dermoscopic image, and patient details (age, sex, body site, skin tone), the system must choose one of four outcomes:

1. **Answer with high confidence**, stating the most likely of eleven diagnoses, only when calibrated evidence supports it and the rate of cancers wrongly called benign among such answers is controlled.
2. **Ask for more evidence**: a dermoscopic image, missing patient details, or a better photograph, when that evidence is likely to change the decision.
3. **Answer with low confidence**, giving the best diagnosis and a short differential, clearly labelled as unreliable and sent to a doctor for confirmation.
4. **Defer** to a clinician when the sources of evidence contradict each other.

Every outcome must come with an explanation that a patient and a doctor can understand, and the whole decision must be measured honestly against simpler alternatives on data the system has never seen.

## Objectives

The concrete objectives of the project were:

- To build a **team of complementary specialist models** for the clinical photograph, the dermoscopic image and the patient details, and to reuse the image model of our earlier prototype as an external specialist.
- To design a **Devil's Advocate** evidence manager that profiles each specialist on each case and to **learn**, rather than hand-set, how its signals translate into trust.
- To **compare fusion strategies fairly**, choosing between them on validation data only, with cross-fitting so that later calibration is not contaminated.
- To build a **calibrated four-outcome decision policy** with a certified bound on missed cancers, explicit costs, and the ability to request evidence.
- To **stress-test** the system with damaged inputs and to evaluate it on **two external datasets** without retraining.
- To deliver a **working care platform**: patient and doctor interfaces, messaging, follow-ups, deletion of personal data, explanations, and a REST API.
- To **audit the earlier prototype** and report its true performance.

## Scope of the Project

The project uses three public datasets: MILK10k (5,240 lesions, each with a clinical and a dermoscopic image, eleven diagnoses) for training, calibration and testing; and PAD-UFES-20 and Derm7pt only for external testing. All model training was done on a Kaggle GPU; everything else runs on an ordinary laptop CPU. The system is a **research prototype**. It has not been validated in a clinic, it is not a medical device, and nothing in this report should be read as a claim that it is safe for real patients. The cost values used by the decision policy are declared design choices, not clinically derived.

## Contributions

The main contributions of this work are:

- A **six-specialist team** combining evidential heads, an external legacy model, a case-based vote and an embedding classifier, with a **learned evidence gate** (Devil's Advocate v2) that lets every evidence signal set the per-case weights.
- A **fair comparison of seven combination rules** with five-fold cross-fitting, an exhaustive study of every subset of specialists, and an oracle ceiling.
- A **calibrated four-outcome policy** with Dempster-Shafer conflict detection, a benign/malignant conformal set, a Learn-then-Test risk bound, quality, out-of-distribution and missing-information gates, and cost-aware evidence acquisition.
- The finding that **confidence thresholding is only as safe as the probabilities beneath it**, whereas the calibrated policy's miss rate is stable across fusion back-ends.
- **DERMA-STRESS**, a controlled degradation benchmark, showing where evidential uncertainty fails and how explicit checks close those gaps.
- **External evaluation** on PAD-UFES-20 and Derm7pt, the latter exercising the full "ask for dermoscopy" loop on a different population.
- A **re-evaluation of the earlier prototype's seven models**, and an empirical check of the symptom-based triage rule on real questionnaire data.
- A **complete, tested software platform** (230 automated tests) with an interactive web application that exposes every system variant for side-by-side comparison.

## Organisation of the Report

Chapter 2 reviews related work. Chapter 3 explains the theory the system rests on, in as plain language as the mathematics allows. Chapter 4 describes the datasets, how they were split, and what the audit of the earlier prototype found. Chapter 5 presents the architecture of the whole system, and Chapter 6 the methods inside it. Chapter 7 covers the implementation, from training on Kaggle to the web application and the test suite. Chapter 8 reports every experiment, including the results that did not favour our system. Chapter 9 discusses what the results mean and what did not work, and Chapter 10 concludes. The appendices contain selected source code, a user guide to the application, and the commands needed to reproduce every number in this report.
