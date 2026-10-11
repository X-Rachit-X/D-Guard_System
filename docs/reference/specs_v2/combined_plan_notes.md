Yes, you should combine both versions into one unified system, not build two separate projects. Keep the original DERMA-GUARD’s dynamic multimodal fusion, agent team, Evidence Manager and RAG, and integrate them with the upgraded project’s research framework. The vision specialists analyze clinical images, dermoscopic images and available metadata; the Evidence Manager evaluates evidence quality, uncertainty, reliability and conflicts; the RAG/Librarian agent retrieves relevant medical references; and the other agents help verify evidence and prepare traceable explanations. All these components should work together through a shared orchestrator, with clear boundaries and auditable outputs.

The final research contribution should be the EADE decision engine and DERMA-STRESS evaluation framework. EADE makes the controlled decision to PROCEED, ACQUIRE additional evidence when useful, or DEFER for human review, using a calibrated, cost-aware policy—not an unrestricted LLM decision. Use MILK10k as the primary dataset, PAD-UFES-20 for suitable external clinical-image shift tests, and Derm7pt for optional replication. Implement everything in the existing D-GUARD repository as one modular system, but evaluate each component separately against meaningful baselines. This gives you one coherent project: multimodal AI interprets evidence, agents and RAG support its analysis, EADE controls the decision, and DERMA-STRESS tests its reliability.

### 1\. The final combined project

### DERMA-GUARD / D-GUARD

Recommended final research title

Risk-Controlled, Cost-Aware Evidence Acquisition and Deferral for Multimodal Dermatology AI

A multimodal, agent-assisted dermatology research system that evaluates skin-lesion evidence, retrieves relevant medical knowledge, and decides whether to proceed with a prediction, acquire additional evidence, or defer the case for human review.

The final combination should have four tightly connected layers.

### Layer 1 — Multimodal intelligence

Clinical-image encoder + dermoscopy encoder + metadata encoder, with evidence-aware fusion. Optional text input is supported only when real or clearly labelled synthetic text is available.

Purpose: estimate what each source indicates and how much evidence it provides.

### Layer 2 — Evidence and agent team

* Vision specialists analyze the available modalities.

* Evidence Manager measures uncertainty, ignorance, disagreement, missingness and reliability.

* Devil’s Advocate checks for conflicts and unsupported conclusions.

* Librarian retrieves relevant medical references using hybrid RAG.

* Scribe prepares a traceable, source-grounded report.

    Purpose: interpret and organize evidence, not independently authorize a diagnosis.

### Layer 3 — EADE decision engine

PROCEED

Policy permits a prediction

ACQUIRE

Request useful missing evidence

DEFER

Human review is required

Purpose: apply the versioned cost model, calibration and risk-control rules. The LLM orchestrator cannot override these rules.

### Layer 4 — Evaluation and platform

DERMA-STRESS + baselines + ablations + acquisition-value measurements + audit logs + case history + reproducible experiment reports.

Purpose: demonstrate what works, where it fails, and whether the added complexity is justified.

This architecture is consistent with the repository’s documented separation between multimodal evidence, Dempster–Shafer combination, conformal prediction, Learn-then-Test calibration and the three-action policy.

### 2\. What exactly should we keep from each version?

| Component                               | Final decision                          | Role in the combined project                                                                                    |
| --------------------------------------- | --------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Dynamic multimodal fusion               | Keep                                    | Combines complementary image and metadata evidence                                                              |
| Old diagnosis-adaptive iterative fusion | Optional experiment, not a requirement  | Compare against the current evidence-combination approach only if implementation and evaluation justify it      |
| Evidence Manager                        | Keep                                    | Tracks quality, reliability, uncertainty, conflict and missing evidence                                         |
| Agentic architecture                    | Keep                                    | Coordinates specialists, checks evidence and generates explanations                                             |
| RAG + medical knowledge base            | Keep                                    | Retrieves relevant clinical references with provenance                                                          |
| LLM gateway                             | Keep                                    | Provides a controlled interface to LLM providers, with structured outputs, timeouts, retries and usage tracking |
| EADE: PROCEED / ACQUIRE / DEFER         | Central research contribution candidate | Makes a cost-aware, risk-controlled decision                                                                    |
| DERMA-STRESS                            | Keep                                    | Evaluates degradation, missing evidence, contradictions, distribution shift and operational failures            |
| Human review and audit trail            | Keep                                    | Makes decisions inspectable and records the reasoning and policy versions                                       |

One important technical distinction: the repository currently documents evidential heads combined using Dempster–Shafer theory. That is a form of evidence-aware multimodal combination, but it is not automatically the same as the older iterative, diagnosis-adaptive fusion algorithm. We should preserve the old algorithm only if we implement it as a clearly defined comparator or ablation.

Likewise, RAG should support the explanation and reference-retrieval pathway. Retrieved medical documents must not silently change the classifier’s probabilities or bypass EADE’s risk controls.

### 3\. Final dataset strategy

![Samples of the 2019 ISIC dataset. (a) Melanoma – MEL, (b) Melanocytic... | Download Scientific Diagram](https://images.openai.com/static-rsc-4/enTXTWcp4C2bLOQnvPTGUDU64DMhBytpHF9WkFhPZekbvcx8IjJCy695Gb8BucRwKihQr83ggJNL0juwlvOtgP5_PohLoKuam-FWRoqg0o6Aa1CfWde1uewYGdjnZrmc5S30Ho3OJMmx02b4wf2kKhBOU61ol6xOnPSK6e-7CJ4eTSw1kTHOiS_7NGZfsX-i?purpose=fullsize)

Primary: MILK10k

Primary training and evaluation

5,240 lesions, each with paired clinical and dermoscopic images, plus structured metadata and 11 diagnostic categories. This is the best match for the acquisition question.

[Official dataset](https://api.isic-archive.com/doi/milk10k/)

![A sample of each type of skin lesion present in PAD-UFES-20 dataset. | Download Scientific Diagram](https://images.openai.com/static-rsc-4/Rov4DMZOKezDwLJr8N3dousinGADPP1hTpdpTK45hL3iLg_RHyeslqAP6U0MLTYga-1xWQe8igRtDrScKy5LIeBfe-yUJfgcUUMlGb7EHxyDh5qHicd0tUjyANWQrGYa7mCltmCSoahnAm79C8dkOVgsxKeOnjb2No3cWVswAhntjKt6COivsnRX6IgNAQOs?purpose=fullsize)

Secondary: PAD-UFES-20

Use for external clinical-image shift and robustness tests. It does not contain paired dermoscopy, so it cannot serve as the equivalent of the primary acquisition experiment.

[Dataset page](https://data.mendeley.com/datasets/zr7vgbcyr2/1)

![Dermoscopic (left) and clinical (right) images of a lesion from the... | Download Scientific Diagram](https://images.openai.com/static-rsc-4/1dAS16UR3eDZebjoPha91LnEWZG8B604Zd-h3o4fpXP9Uw-9zXfOimqr1V5JEPHtSmNFRui1UqpbPpFrwxGOqmKs47ovD8S1kb9AcBZPRH2pF3yUjoXTq8y61ook7An_mX9mbEmAhX3iRB6Wb1GkiolEoSwNw44RhhYF0CtODSvj5gLjmafDNNwwEgthBnu4?purpose=fullsize)

Optional: Derm7pt

Add this only for a separate replication experiment after MILK10k is working reproducibly. It should not delay the primary pipeline.

Two important data constraints:

* MILK10k is licensed CC BY-NC. Follow its attribution and non-commercial-use conditions, and do not commit the images to the public repository.

* MILK10k does not list patient identifiers. Use lesion-level splits, deduplicate using the available identifiers and image checks, and explicitly acknowledge that residual patient-level leakage may remain. Do not claim a true patient-level split for this dataset.

### 4\. One issue we must address in the existing repository

The repository itself flags an important evaluation-integrity problem: older checkpoints were evaluated with training/test leakage and predictions were rewritten toward hardcoded target accuracies. It explicitly says those older metrics are invalid and must not be cited. Its current documentation also marks the MILK10k migration and native evidential-head training as incomplete.

That means we should not treat the previous headline scores as validated research results. The good news is that the repository documents a reproducible feature pipeline, tests, a decision-policy implementation and a stress-testing framework. Those are useful foundations, but the models and final comparisons still need clean, reproducible evaluation.

I would prioritize the implementation in this order:

* Audit the current code, data splits, feature extraction and saved model artifacts; preserve the integrity audit.

* Complete the MILK10k data pipeline and train a reproducible clinical-only baseline.

* Add dermoscopy and metadata fusion, then evaluate missing-modality handling and calibration.

* Validate EADE against confidence-based abstention, always-fuse and relevant acquisition/deferral baselines.

* Integrate the agent team and RAG as bounded, auditable supporting components.

* Run DERMA-STRESS, ablations, cost sensitivity and external-shift experiments; publish all failures as well as successes.

### Final verdict

Combine the two designs into one system, using the upgraded research question as the backbone and retaining the old multimodal fusion, agents and RAG as supporting capabilities. The strongest defensible research direction is not “we built a more complicated dermatology agent.” It is:

> We investigate whether a risk-controlled, cost-aware evidence-acquisition and deferral policy can make more appropriate decisions about when to predict, when to request additional dermoscopic evidence and when to defer—compared with simpler baselines—and how that policy behaves under controlled evidence failures.

That is a coherent research question with a useful engineering platform around it. It remains a hypothesis to test, not a proven clinical benefit or guaranteed novelty claim.
