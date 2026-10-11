# DERMA-GUARD --- Final Project Specification

**Project type:** Three-member academic research and
software-engineering project\
**Working title:** *DERMA-GUARD: An Evidence-Adaptive Dermatology
Intelligence Framework for Risk-Aware Decision-Making, Selective
Evidence Acquisition, and Robustness Evaluation*\
**Version:** 1.0 --- Project scope locked\
**Status:** Proposed implementation and evaluation plan; research
outcomes and novelty are not yet established

------------------------------------------------------------------------

## 1. Executive summary

DERMA-GUARD is a multimodal dermatology research platform that studies
how an AI system should act when the evidence available for a
skin-lesion case is incomplete, low quality, uncertain, contradictory,
or outside the distribution on which its models were evaluated.

Most conventional classification pipelines focus on producing a label or
probability from the supplied input. DERMA-GUARD adds an explicit
evidence-management and decision layer. It evaluates image quality,
model reliability, predictive uncertainty, cross-modal disagreement,
missing information, and the availability and estimated cost of
additional evidence. A deterministic policy then chooses among three
actions:

1.  **Proceed:** return a bounded model assessment with its limitations
    and supporting evidence.
2.  **Acquire:** request an additional input that is actually available
    and potentially useful.
3.  **Defer:** abstain from a definitive automated result and route the
    case for appropriate human review.

The project also includes **DERMA-STRESS**, a reproducible evaluation
protocol for testing individual and combined evidence failures. It will
compare the proposed policy against established fusion and
selective-prediction baselines and measure reliability, coverage,
evidence-acquisition benefit, latency, and computational cost.

The integrated application provides case intake, evidence storage, model
execution, source-grounded evidence summaries, persistent assessment
history, review workflows, experiment tracking, and auditability. RAG,
LLMs, and agents are supporting tools---not the central scientific
contribution.

**Central research question:** Can explicit evidence-management
decisions improve the reliability--coverage--cost trade-off of
multimodal dermatology AI under controlled evidence failures, compared
with simpler fusion and abstention policies?

This is a research prototype, not a medical device or an autonomous
diagnostic authority. The proposed method may or may not outperform its
baselines; results must be reported honestly.

------------------------------------------------------------------------

## 2. Final scope and non-goals

### 2.1 In scope

-   A patient-case-centred web application.
-   A reproducible image-classification baseline.
-   Additional vision models or multimodal fusion when data, compute,
    licences, and evaluation protocols support them.
-   Image-quality estimation and explicit missingness tracking.
-   Calibration, uncertainty, model disagreement, and OOD signals where
    feasible.
-   A versioned, deterministic proceed/acquire/defer policy.
-   Hybrid medical-information retrieval with source attribution.
-   Controlled corruption, missing-modality, contradictory-context,
    distribution-shift, and combined-failure tests.
-   Comparisons against standard fusion, adaptive-fusion, and
    confidence-threshold abstention.
-   Ablation studies, metrics, and experiment reports.
-   Persistent case history, model-run records, review actions, and
    audit events.
-   Secure, documented, reproducible deployment.

### 2.2 Out of scope for the initial release

-   Training a foundation model from scratch.
-   Claiming that the system replaces dermatologists.
-   Autonomous diagnosis or treatment recommendations.
-   Claiming clinical benefit or regulatory readiness from public
    benchmark results.
-   Microservices, Kafka, or Kubernetes unless a demonstrated
    requirement justifies them.
-   Unrestricted LLM authority over model weights, routing, or clinical
    conclusions.
-   Using synthetic text as if it were genuine patient history.
-   Building every possible agent or dashboard feature before the
    research baseline works.

### 2.3 Project success philosophy

A functioning product and a successful scientific hypothesis are
different outcomes. The platform can be a substantial engineering
achievement even if the proposed decision policy fails to outperform
baselines. If the policy does not improve the pre-specified metrics, the
team will report that finding and emphasize the benchmark, ablation
results, and lessons learned rather than inventing a positive
conclusion.

------------------------------------------------------------------------

## 3. Problem statement and motivation

Dermatology cases may contain several kinds of evidence: clinical
photographs, dermoscopic images, structured patient information, symptom
descriptions, and medical references. Each source can be incomplete or
unreliable.

Examples include:

-   A blurry or poorly illuminated photograph.
-   Missing metadata or an unavailable dermoscopic image.
-   Two models producing contradictory predictions.
-   A symptom description that conflicts with the available image.
-   A model encountering a case that differs substantially from its
    evaluation data.
-   An inference service timing out.
-   A retrieval system returning sources that do not support a generated
    claim.

Simply adding more models does not guarantee reliability. A larger
ensemble may cost more, share correlated errors, or become confidently
wrong. A RAG-generated explanation is not independent confirmation of a
visual prediction. Confidence alone is also not a complete measure of
evidence quality.

DERMA-GUARD studies the decision surrounding the prediction: **what
evidence should be trusted, whether the current evidence is sufficient
for the chosen policy, whether more available evidence is worth
obtaining, and when the system should defer.**

This problem is related to existing research in multimodal fusion,
uncertainty estimation, OOD detection, missing-modality learning,
triage, and selective prediction. DERMA-GUARD does not claim to invent
these fields. Its proposed contribution is the explicit integration and
evaluation of an evidence-action policy under controlled combined
failures and a measured reliability--coverage--cost trade-off. Whether
that exact framing is novel enough for publication must be assessed
through a full related-work audit.

------------------------------------------------------------------------

## 4. Research objectives and hypotheses

### Objective 1 --- Evidence-Adaptive Decision Engine (EADE)

Develop a reproducible policy that uses input quality, model
reliability, predictive uncertainty, cross-modal consistency,
missingness, and estimated evidence value to choose between proceeding,
requesting available additional evidence, and deferring.

**H1:** Compared with confidence-only abstention and ordinary fusion,
the evidence-aware policy will improve risk at a pre-specified coverage
level under at least some controlled evidence failures, without an
unacceptable increase in cost.

### Objective 2 --- DERMA-STRESS

Develop a versioned protocol for evaluating individual and combined
evidence failures, using consistent cases, baselines, severity levels,
and metrics.

**H2:** The proposed policy will handle combined evidence failures
better than selected baseline policies evaluated on the same cases and
under the same conditions.

### Objective 3 --- Evidence value and acquisition

Evaluate whether requesting an additional input that is genuinely
available can reduce decision risk enough to justify the added
interaction, inference, or computational cost.

**H3:** A validation-tuned evidence-acquisition policy can achieve a
better risk--coverage--cost trade-off than fixed-input,
confidence-threshold, and reliability-only policies.

### Engineering objective --- Auditable platform

Build a persistent case workflow that stores evidence, model versions,
results, decisions, source provenance, review events, and experiment
configurations.

This is an engineering acceptance objective, not a scientific novelty
claim.

### Hypothesis discipline

All hypotheses are predictions, not results. Thresholds and policy
parameters must be chosen using training/validation data. The final test
set must not be used to tune the method or select whichever metric looks
best.

------------------------------------------------------------------------

## 5. Proposed research contributions

### Contribution A --- Evidence-Adaptive Decision Engine

The EADE connects evidence assessment to explicit actions. Its input
signals are kept separate initially:

-   **Quality (Q):** technical usability of an input.
-   **Reliability (R):** empirically validated behavior of a model or
    output.
-   **Uncertainty (U):** ambiguity of a prediction, estimated with a
    documented method.
-   **Consistency (C):** agreement or disagreement between available
    modalities or models.
-   **Missingness (M):** explicit record of unavailable inputs or
    fields.
-   **Evidence value (V):** estimated benefit of a specific available
    next input relative to its cost.
-   **OOD signal (D):** evidence that the input may differ from the
    model's evaluated distribution.

These signals should not be collapsed into an arbitrary hand-designed
trust score and treated as ground truth. Start with interpretable
features, establish baselines, tune any policy on validation data, and
use ablations to determine which signals add value.

The action space is:

-   `PROCEED`: return the model assessment with confidence/calibration
    caveats, sources where relevant, and known limitations.
-   `ACQUIRE`: request a specified input or step that is available and
    has an estimated value under the policy.
-   `DEFER`: stop automated decision-making for this case and request
    appropriate human review.

The policy is deterministic and versioned for reproducibility. LLMs may
assist with structured extraction and source-grounded summaries, but do
not have unrestricted authority to change model weights or invent
clinical conclusions.

### Contribution B --- DERMA-STRESS

DERMA-STRESS is a reproducible evaluation protocol, not merely a folder
of damaged images. It specifies:

-   Dataset and split versions.
-   Input corruption types and severity.
-   Missing-modality patterns.
-   Contradiction-generation rules.
-   External-dataset or acquisition-setting tests.
-   Combined-failure scenarios.
-   Baseline policies and model versions.
-   Metrics and confidence intervals.
-   Random seeds and configuration files.
-   Saved predictions and experiment outputs.

It will evaluate not only classification accuracy, but also failure
recognition, confident errors, abstention, recovery after new evidence,
acquisition cost, latency, and operational behavior.

### Contribution C --- Evidence value and acquisition analysis

The system evaluates whether an additional input improves the result
enough to justify its cost. Acquisition benefit can only be measured
directly when the additional input is actually available for the test
case. If the team simulates an unavailable input, that result must be
labelled as a simulation, not as demonstrated clinical benefit.

Two optional proposed metrics are:

**Minimum Sufficient Evidence (MSE):** the smallest available evidence
subset that satisfies a pre-specified, validation-derived
risk/coverage/calibration criterion under a defined cost function. If no
subset qualifies, report that no sufficient subset was found. This is a
project-defined metric and must be compared with simpler
subset-selection statistics.

**Negative Information Rate (NIR):** the proportion of evaluated cases
where adding a modality increases a pre-defined loss. For case (i), with
current evidence subset (S), additional modality (m), reference label
(y_i), and loss (L), an adverse change occurs when:

`L(prediction(S ∪ {m}), y_i) > L(prediction(S), y_i)`

NIR is the proportion of evaluated cases with an adverse change. Also
report the magnitude of the loss increase. NIR is a proposed metric, not
an established clinical standard; retain it only if it adds useful
information beyond simpler degradation statistics.

------------------------------------------------------------------------

## 6. Literature positioning and base-paper strategy

The literature contains substantial prior work on multimodal
dermatology, adaptive fusion, uncertainty, OOD detection, missing
modalities, and triage. The project must acknowledge this explicitly.

### Primary comparison papers

1.  **DermFormer: Nested Multi-Modal Vision Transformers for Robust Skin
    Cancer Detection (2025).** Dermatology-specific multimodal fusion
    using clinical images, dermoscopic images, and metadata, with
    robustness evaluation. Use as a primary architecture/robustness
    comparator where data and protocol allow.
2.  **Zuo et al., adaptive multimodal fusion for multi-label skin-lesion
    classification (2025; associated with CosCatNet).** A close
    comparison for adaptive and uncertainty-related fusion. Inspect the
    full paper and code before choosing an implementation.
3.  **HOT AI (2025).** Relevant precedent for hierarchical prediction,
    OOD alerts, and recommendations for additional dermoscopy. It
    prevents a broad claim that additional-evidence triage is new.
4.  **QA-MoE (IJCAI 2026).** A quality-aware routing and
    missing-modality comparator in non-dermatology clinical tasks. It is
    a conceptual baseline, not a directly comparable dermatology result
    unless reimplemented on compatible data.
5.  **SAGE (npj Digital Medicine, 2026).** Relevant to multi-criterion
    uncertainty and distribution-shift detection in skin-lesion
    classification.

Additional related work to inspect includes PanDerm (dermatology
foundation model), DermEVAL (multimodal dermatology benchmark),
DermFusion and UAGF (recent uncertainty/adaptive fusion or triage work),
selective-prediction calibration, conformal prediction, missing-modality
fusion, and dataset-quality/fairness studies.

### Baseline categories

-   Best single-modality classifier.
-   Standard concatenation or late fusion, where compatible modalities
    exist.
-   A documented adaptive-fusion method.
-   Confidence-threshold abstention.
-   Reliability-aware policy without acquisition.
-   Full EADE with proceed/acquire/defer.
-   Optional OOD-specific baseline.

### Reproduction rules

A published score copied from a paper is not our experimental result.
Clearly label each result as: - `paper-reported`, -
`our-reimplementation`, - or `our-original-method`.

Document differences in datasets, splits, preprocessing, labels, and
metrics. Do not claim a faithful reproduction unless the protocol is
sufficiently comparable.

### Novelty wording

Use: "We propose and evaluate an evidence-adaptive decision policy under
controlled evidence failures."

Do not use "first," "state of the art," or "proven novel" unless a
systematic search and results justify the wording. The exact research
novelty remains a hypothesis until the full related-work audit and
experiments are complete.

------------------------------------------------------------------------

## 7. Full system architecture

### A. Case intake

Responsibilities: - Create a case with a stable identifier. - Accept
clinical photographs and dermoscopic images where available. - Accept
structured symptoms and permitted metadata. - Record asset source,
upload time, file type, and provenance. - Validate file type and size. -
Record missing inputs explicitly. - Record consent/access information
where required.

Distinguish `MISSING` from `PRESENT_BUT_LOW_QUALITY`. Do not silently
impute missing clinical facts.

### B. Vision intelligence

Responsibilities: - Run a baseline image classifier. - Add a second
model only when it contributes to a meaningful comparison. - Calculate
image-quality indicators. - Store probability outputs and model
versions. - Estimate uncertainty using a documented method. - Calculate
model disagreement when multiple models are available. - Optionally
calculate OOD scores. - Return structured errors and execution status.

Do not train multiple heavyweight models from scratch at the beginning.
First establish a working baseline and a leakage-safe evaluation
protocol. Consider a pretrained dermatology model only after checking
availability, compute requirements, licence, input format, and
evaluation compatibility.

### C. Clinical evidence intelligence

Responsibilities: - Convert symptom descriptions into a constrained
schema. - Ingest a curated collection of appropriate medical
documents. - Chunk and index documents while preserving source
metadata. - Retrieve passages with lexical (BM25) and vector search. -
Optionally rerank retrieved passages. - Generate a concise evidence
summary from retrieved material. - Link claims to exact
sources/passages. - Flag unsupported claims and retrieval failures. -
Store document/index/model versions.

RAG does not prove a model prediction correct. Retrieved material is
supporting context, not independent clinical validation. Never invent
citations.

### D. Evidence Manager

Responsibilities: - Aggregate quality, reliability, uncertainty,
consistency, missingness, OOD signals, and provenance. - Preserve the
raw signal values. - Produce a versioned feature record for each
decision. - Expose the signals to the decision policy. - Log which
signals were unavailable and why.

### E. Evidence-Adaptive Decision Engine

Responsibilities: - Apply a deterministic, versioned policy. - Decide
whether to proceed, request a specific available input, or defer. -
Apply validation-derived thresholds. - Prevent unlimited acquisition
loops. - Enforce a maximum number of policy iterations. - Record the
policy version and reasons for the action. - Defer when evidence remains
insufficient or no suitable input is available.

### F. Persistent case and audit layer

Responsibilities: - Store case details, evidence assets, assessments,
model runs, evidence sources, jobs, reviews, and audit events. -
Preserve prior assessments rather than overwriting history. - Record who
or what triggered a transition. - Link each output to its exact evidence
snapshot and model version. - Make failures inspectable.

### G. DERMA-STRESS experiment runner

Responsibilities: - Load a fixed dataset split. - Apply versioned
stress-test configurations. - Run the same cases through each baseline
and the proposed method. - Save predictions and policy actions. -
Calculate metrics and uncertainty intervals. - Generate machine-readable
results and human-readable reports. - Record configuration, random seed,
versions, and execution environment.

------------------------------------------------------------------------

## 8. User roles and main workflows

### Roles

-   **Researcher/admin:** configures models, datasets, experiment runs,
    and permitted system settings.
-   **Case operator:** creates cases, uploads permitted evidence, and
    requests assessments.
-   **Reviewer:** inspects results, records review decisions, and
    documents corrections.
-   **Viewer:** views authorized case and experiment results without
    modifying them.

Role separation can be simplified for a classroom demo, but
authorization checks must still exist in the backend.

### Workflow 1 --- Create and assess a case

1.  User creates a case.
2.  User uploads one or more evidence assets.
3.  Backend validates and stores the assets.
4.  Case metadata records which inputs are available or missing.
5.  User requests an assessment.
6.  A background job runs the baseline model and supporting checks.
7.  Evidence Manager stores quality and reliability signals.
8.  Decision policy chooses `PROCEED`, `ACQUIRE`, or `DEFER`.
9.  Assessment and audit records are persisted.
10. UI shows status, outputs, limitations, and next action.

### Workflow 2 --- Acquire more evidence

1.  Policy identifies an additional input that is both available to
    request and potentially useful.
2.  The UI explains what is being requested and why, without presenting
    the request as a medical instruction.
3.  The operator supplies the input, or marks it unavailable.
4.  A new assessment is run against a new evidence snapshot.
5.  The system records whether the added input changed the prediction,
    risk, action, and cost.
6.  The acquisition loop stops after the configured limit or when the
    policy defers.

### Workflow 3 --- Human review

1.  A case is marked for review.
2.  Reviewer sees available evidence, model outputs, quality signals,
    sources, and the reason for deferral.
3.  Reviewer records a review outcome and optional rationale.
4.  The system appends a review event; it does not silently rewrite the
    earlier model output.
5.  The case timeline shows the sequence.

### Workflow 4 --- Run DERMA-STRESS

1.  Researcher selects a dataset/split and a versioned configuration.
2.  Experiment worker applies the stress condition.
3.  Baselines and the proposed policy process the same cases.
4.  Predictions, actions, runtime and errors are saved.
5.  Metrics and confidence intervals are calculated.
6.  The UI presents comparable results and links them to the experiment
    configuration.

------------------------------------------------------------------------

## 9. Dataset strategy

### Primary starting dataset: Derm7pt

It includes clinical and dermoscopic images and associated
checklist/metadata information, making it useful for initial multimodal
experiments. Verify the exact release, labels, patient identifiers,
licence, and compatible modalities before finalizing the protocol.

### Secondary dataset: PAD-UFES-20

It contains smartphone clinical images and patient/clinical information.
Verify the fields, identifiers, class mapping, licence, and suitability
for each intended experiment. It may not support every multimodal
comparison.

### Optional external image benchmark: HAM10000 or an appropriate ISIC release

Use only for compatible image-only baselines or external evaluation. Do
not treat it as a multimodal benchmark if the necessary modalities are
absent.

### Data-splitting and leakage rules

-   Split by patient whenever identifiers are available.
-   Do not place images from the same patient across train and test.
-   Use official splits when reproducing a published benchmark.
-   Fit preprocessing, calibration, and policy thresholds on
    training/validation data only.
-   Keep the final test set untouched until the protocol is frozen.
-   Keep synthetic clinical text separate from genuine patient
    information.
-   Do not create synthetic text that leaks the ground-truth diagnosis.
-   Document class imbalance and label uncertainty.
-   Check licences and data-use conditions before storing or
    redistributing assets.

### Feasibility rule

Start with one dataset and one classifier. Add a second dataset only
after the first end-to-end pipeline and evaluation are reproducible.

------------------------------------------------------------------------

## 10. DERMA-STRESS protocol

### Stress family A --- Image degradation

Examples: - Blur. - Compression artefacts. - Brightness/contrast
changes. - Resolution reduction. - Other justified image-quality
transformations.

Record the transformation, severity, random seed, and original asset ID.
Do not assume every artificial corruption is clinically realistic.

### Stress family B --- Missing evidence

Examples: - Remove metadata fields. - Remove a modality that exists in
the original case. - Vary missingness patterns.

Distinguish genuinely missing inputs from low-quality inputs. Report
performance and policy behavior by pattern.

### Stress family C --- Contradictory context

Examples: - Controlled conflicts between structured fields and symptom
text. - Contradictory non-diagnostic context where ethically and
methodologically appropriate.

Clearly label synthetic changes. Prevent ground-truth leakage and avoid
claiming that synthetic contradictions represent the full complexity of
clinical records.

### Stress family D --- Distribution shift

Evaluate on a suitable external dataset or distinct acquisition setting.
Keep this separate from artificial corruption tests. Check label
compatibility and avoid comparing incompatible diagnostic taxonomies
without an explicit mapping.

### Stress family E --- Combined failures

Examples: - Degraded image + missing metadata. - Missing dermoscopy +
conflicting context. - Distribution shift + image degradation.

Combined tests are central to our proposed evaluation, but their novelty
must not be assumed. Search existing benchmarks and document precisely
what is different about our protocol.

### Stress family F --- Operational failures

Simulate: - Vision-model timeout. - Evidence-retrieval timeout. - Worker
restart or job failure. - Invalid asset or corrupted file. - LLM
provider unavailable.

Report operational resilience separately from diagnostic accuracy.

### Reproducibility record

Every run should record: - Dataset and split version. - Configuration
file and hash if used. - Corruption type and severity. - Random seed. -
Model and policy versions. - Software/environment versions. - Saved
predictions and actions. - Start/end times, latency, errors and resource
use. - Metrics and confidence intervals.

------------------------------------------------------------------------

## 11. Experimental design and metrics

### Primary endpoint

**Risk at a pre-specified coverage level** for the cases on which the
policy proceeds.

Coverage is the proportion of evaluated cases accepted for automated
output. Risk is the selected loss/error measure among accepted cases.
Define both precisely before the final test run.

Also report coverage at a pre-specified risk target.

### Secondary metrics

**Classification** - Macro-F1. - Balanced accuracy. - Per-class
sensitivity and precision. - Confusion matrix. - Calibration metrics
appropriate to the task.

**Selective prediction and reliability** - Risk--coverage curve. - Area
under the risk--coverage curve where meaningful. - Coverage at fixed
risk. - Calibration error and/or Brier score, with methodology stated. -
OOD detection metrics when OOD labels are defined.

**Acquisition** - Acquisition frequency. - Benefit after the additional
input is supplied. - Change in loss/risk before versus after
acquisition. - Unnecessary request rate under the defined policy. -
Latency and computational cost per case. - NIR and MSE if their
definitions prove useful.

**Operational** - Job success/failure rate. - Retry and recovery
behavior. - End-to-end latency. - Traceability completeness. - Cost per
assessment or experiment where measurable.

### Required baselines

1.  Best single-modality model.
2.  Standard fusion where compatible.
3.  A feasible adaptive-fusion baseline.
4.  Confidence-threshold abstention.
5.  Reliability-aware policy without acquisition.
6.  Full EADE policy.
7.  OOD baseline where relevant.

### Required ablations

-   Remove image-quality signals.
-   Remove reliability/calibration signals.
-   Remove consistency features.
-   Remove OOD signal.
-   Remove acquisition.
-   Full method.

### Statistical discipline

-   Evaluate methods on the same cases.
-   Use paired comparisons where appropriate.
-   Report confidence intervals and class-wise results.
-   Repeat seed-dependent experiments where feasible.
-   Select thresholds on validation data only.
-   Avoid claims based on one metric while hiding regressions in others.
-   Distinguish paper-reported results from our own reimplementations.
-   If compute prevents a strong baseline from being reproduced, state
    the limitation and use a feasible alternative without claiming
    equivalence.

### Interpreting results

A plausible positive outcome is a better risk--coverage--cost trade-off
under some failures, even if the method does not improve clean-set
macro-F1. A negative result is also meaningful: it may show that added
signals or acquisition costs do not justify the complexity. Do not
assume or fabricate results in advance.

------------------------------------------------------------------------

## 12. Proposed metric definitions

### Minimum Sufficient Evidence (MSE)

Let (E) be the set of evidence items available for a case and (S
`\subseteq `{=tex}E) a candidate subset. A subset qualifies only if it
satisfies a pre-specified validation-derived criterion, such as a risk
bound and a minimum coverage/calibration requirement under a defined
cost function.

MSE is the smallest qualifying subset under that cost function. If none
qualifies, report no sufficient subset.

Important constraints: - Thresholds are selected using validation
data. - A smaller subset is not automatically clinically sufficient. -
The metric depends on the chosen sufficiency criterion. - Report
sensitivity to the criterion and compare against simple subset-selection
baselines.

### Negative Information Rate (NIR)

NIR is the proportion of evaluated cases for which adding a modality
increases the chosen loss. Also report the magnitude of the change,
since tiny and severe degradations should not be treated as equivalent.

NIR is a project-defined research metric, not a recognized clinical
standard. Keep it only if it offers insight beyond simpler measurements
of performance degradation after modality addition.

### Evidence acquisition value

Where a candidate additional input exists, compare the current decision
with the decision after the input is revealed. Estimate the change in
loss and the cost of acquiring/processing it.

A simulated input may support a simulation study but cannot establish
the real-world clinical value of acquiring that input.

------------------------------------------------------------------------

## 13. Technology stack

### Frontend

-   Next.js.
-   React.
-   TypeScript.
-   Tailwind CSS.

### Backend and schemas

-   Python.
-   FastAPI.
-   Pydantic.
-   PostgreSQL.

### ML and image processing

-   PyTorch.
-   torchvision.
-   scikit-learn.
-   OpenCV.
-   Additional libraries only when justified by the selected model and
    dataset.

### Retrieval and GenAI

-   BM25 lexical retrieval.
-   Vector retrieval using Qdrant or a comparable vector store.
-   Optional reranker.
-   LLM gateway for provider abstraction, structured outputs,
    timeout/retry controls, and usage/cost tracking.
-   LangGraph only if its explicit stateful orchestration is useful; do
    not introduce it merely for appearance.

### Storage and background work

-   S3-compatible object storage for images/documents.
-   PostgreSQL for case and workflow metadata.
-   Redis-backed queue/worker system if needed for asynchronous jobs.

### Deployment and observability

-   Docker.
-   CI checks and automated tests.
-   Structured logging.
-   OpenTelemetry instrumentation where practical.
-   Prometheus/Grafana if the deployment benefits from dashboards.

### Architecture decision

Start with a modular monolith plus background workers. Avoid
microservices, Kafka, and Kubernetes until there is a concrete scaling
or reliability requirement.

------------------------------------------------------------------------

## 14. Data model

### PatientCase

-   `id`
-   `created_at`
-   `updated_at`
-   `status`
-   `created_by`
-   `consent/access metadata` where applicable
-   permitted, minimal case metadata

### EvidenceAsset

-   `id`
-   `case_id`
-   `asset_type`
-   `storage_key`
-   `source/provenance`
-   `uploaded_at`
-   `quality_status`
-   `metadata`
-   `content_hash` where appropriate

### Assessment

-   `id`
-   `case_id`
-   `evidence_snapshot_id`
-   `policy_version`
-   `status`
-   `action`
-   `result_summary`
-   `created_at`

### ModelRun

-   `id`
-   `assessment_id`
-   `model_name`
-   `model_version`
-   `preprocessing_version`
-   `output_scores`
-   `quality/uncertainty signals`
-   `runtime`
-   `status`
-   `error_details`

### EvidenceSource

-   `id`
-   `assessment_id`
-   `document_id`
-   `source_url`
-   `passage_reference`
-   `retrieval_score`
-   `index_version`
-   `retrieved_at`

### WorkflowJob

-   `id`
-   `case_id`
-   `job_type`
-   `status`
-   `attempt_count`
-   `started_at`
-   `finished_at`
-   `error_code`
-   `error_message`

### ReviewDecision

-   `id`
-   `case_id`
-   `assessment_id`
-   `reviewer_id`
-   `decision`
-   `rationale`
-   `created_at`

### AuditEvent

-   `id`
-   `case_id`
-   `actor_type`
-   `actor_id`
-   `event_type`
-   `event_payload`
-   `created_at`

### ExperimentRun

-   `id`
-   `dataset_version`
-   `split_version`
-   `configuration`
-   `model_versions`
-   `policy_version`
-   `metrics`
-   `started_at`
-   `finished_at`
-   `status`

The schema will evolve, but each assessment must remain linked to the
exact evidence snapshot, model version, and policy version that produced
it.

------------------------------------------------------------------------

## 15. API design

Initial proposed endpoints:

  Method   Endpoint                        Purpose
  -------- ------------------------------- --------------------------------
  POST     `/api/cases`                    Create a case
  GET      `/api/cases`                    List authorized cases
  GET      `/api/cases/{id}`               Read case details
  POST     `/api/cases/{id}/assets`        Upload/register evidence
  POST     `/api/cases/{id}/assessments`   Request an assessment
  GET      `/api/assessments/{id}`         Read status/result
  GET      `/api/cases/{id}/timeline`      Read case event history
  POST     `/api/cases/{id}/reviews`       Record a review
  POST     `/api/experiments`              Start a controlled experiment
  GET      `/api/experiments/{id}`         Read experiment status/results

Implementation requirements: - Validate requests with typed schemas. -
Authenticate and authorize every case operation. - Use asynchronous jobs
for long-running inference. - Add idempotency or duplicate-job
protection where appropriate. - Return clear status and error codes. -
Do not expose object-storage keys or secrets unnecessarily. - Version
APIs when contracts change. - Preserve prior assessments rather than
overwriting them.

------------------------------------------------------------------------

## 16. Frontend pages

1.  **Overview dashboard:** case counts, job states, recent activity,
    experiment summaries.
2.  **Case list:** filters by status, creation date, and permitted
    metadata.
3.  **Case creation:** structured fields and evidence upload.
4.  **Case detail:** evidence assets, quality indicators, current
    assessment, history.
5.  **Assessment detail:** model outputs, calibrated
    confidence/uncertainty, missingness, disagreement, evidence sources,
    policy action and limitations.
6.  **Acquisition/review screen:** requested input, availability status,
    review decision form.
7.  **Case timeline:** chronological assessments, assets, jobs, and
    reviewer actions.
8.  **DERMA-STRESS lab:** configuration selection, experiment status,
    baseline comparison, metrics and ablations.
9.  **Evidence-source view:** retrieved passages, document metadata,
    source links and support/verification status.
10. **Admin/settings:** model/policy versions and access control,
    restricted to authorized users.

The UI must not imply that a heatmap, source citation, or confident
score is proof of clinical correctness.

------------------------------------------------------------------------

## 17. Security, privacy, and clinical safeguards

-   Authenticate users and enforce role-based authorization.
-   Validate uploads and restrict permitted formats and sizes.
-   Store secrets in environment/secret-management facilities, never in
    source control.
-   Keep LLM provider keys server-side.
-   Do not transmit patient images or clinical text to external services
    without an explicitly permitted data-handling arrangement.
-   Use encrypted transport and suitable storage protections.
-   Log access and important state transitions without unnecessarily
    duplicating sensitive data.
-   Avoid placing sensitive clinical content in ordinary logs.
-   Provide retention/deletion behavior appropriate to institutional
    requirements.
-   Separate public datasets, synthetic test cases, and any identifiable
    real-world data.
-   Seek appropriate institutional/ethics approval before using real
    patient data.
-   State clearly that the prototype is for research/decision support,
    not autonomous diagnosis.
-   Route concerning or insufficient cases to appropriate human review.
-   Document demographic/subgroup limitations and dataset bias where the
    data allow such analysis.

------------------------------------------------------------------------

## 18. Testing strategy

### Unit tests

-   Input schema validation.
-   Image-quality functions.
-   Missingness handling.
-   Metric calculations.
-   Policy transition rules.
-   Evidence-source mapping.
-   Data-access controls.

### Integration tests

-   Case creation to stored evidence.
-   Evidence upload to model run.
-   Model failure to visible failed-job state.
-   Retrieval to cited-source record.
-   Assessment to timeline event.
-   Review action to immutable history.
-   Experiment configuration to saved metrics.

### Research tests

-   Patient-level split validation.
-   Reproducibility of corruption configurations.
-   Baseline consistency on identical cases.
-   Threshold selection only on validation data.
-   Ablation correctness.
-   Metric calculation verification.
-   External-dataset mapping checks.

### End-to-end acceptance tests

1.  Create case, upload an image, run baseline, save and display output.
2.  Submit an invalid file and receive a clear error.
3.  Simulate inference timeout and verify that the case remains
    inspectable.
4.  Remove a modality and verify explicit missingness handling.
5.  Run a stress test and retrieve its configuration and metrics.
6.  Record a review and verify that prior assessment history remains
    unchanged.

------------------------------------------------------------------------

## 19. Three-member responsibility split

### Member 1 --- ML and reliability

Owns: - Dataset preparation and leakage-safe splits. - Vision baseline
and compatible additional models. - Quality estimation, calibration,
uncertainty and OOD features. - Fusion and evidence-action policy. -
DERMA-STRESS generation and quantitative evaluation. - Ablations and
research result tables.

Deliverable: reproducible Python package with documented inputs/outputs,
baseline scripts, policy implementation, and experiment reports.

### Member 2 --- GenAI and evidence intelligence

Owns: - Structured symptom extraction. - Curated document ingestion. -
Hybrid retrieval and optional reranking. - LLM gateway and
schema-constrained outputs. - Source attribution and claim-support
checks. - Evidence summaries and integration with the decision workflow.

Deliverable: a traceable evidence subsystem that returns structured
information and sources, and can flag unsupported claims or retrieval
failures.

### Member 3 --- Platform and systems

Owns: - Next.js interface. - FastAPI APIs and typed contracts. -
PostgreSQL and object storage. - Background jobs and workflow states. -
Case history, audit events, authentication and authorization. -
Deployment, observability, integration and system testing.

Deliverable: a deployable application that persists cases, executes
jobs, presents assessments, and supports review and experiment
inspection.

### Shared responsibilities

All members participate in research design, code review, integration,
end-to-end testing, interpretation of results, limitations,
documentation, and presentation. Each member must be able to explain the
complete architecture at a high level.

------------------------------------------------------------------------

## 20. Twelve-week target plan

### Weeks 1--2: Vertical slice

-   Repository, environments, data access and basic CI.
-   One image classifier and evaluation script.
-   Case creation, asset upload, one inference job, saved result.
-   Visible success and failure states.

### Weeks 3--4: Persistent platform and baseline

-   Case history, model-run records, background jobs and API validation.
-   Establish leakage-safe splits and baseline metrics.
-   Document dataset/label compatibility.

### Weeks 5--6: Evidence subsystem and fusion

-   Structured clinical context.
-   Initial hybrid retrieval and source attribution.
-   Quality indicators.
-   Standard and feasible adaptive-fusion baselines.

### Weeks 7--8: Decision engine

-   Versioned proceed/acquire/defer policy.
-   Calibration and uncertainty where feasible.
-   Consistency and OOD signals.
-   First policy comparisons and ablations.

### Weeks 9--10: DERMA-STRESS

-   Image degradation and missing-evidence tests.
-   Contradictory-context and combined-failure tests.
-   Acquisition-policy experiments.
-   Reproducible result saving and reporting.

### Week 11: Hardening

-   Security and authorization tests.
-   Failure recovery, provenance, monitoring and repeatability.
-   Remove nonessential features that are unstable or not useful.

### Week 12: Final evaluation and demo

-   Freeze the test protocol.
-   Run final comparisons and confidence intervals.
-   Document limitations and threats to validity.
-   Complete report, diagrams, setup guide, and presentation.

This is a target schedule, not a guarantee that every stretch feature
will be finished. Preserve evaluation quality before adding features.

------------------------------------------------------------------------

## 21. Milestones and go/no-go decisions

### Milestone 1 --- Working vertical slice

**Go:** a case can be created, an image uploaded, inference run, result
persisted, and failure displayed. **No-go:** do not add multiple models
or agents until this works.

### Milestone 2 --- Trustworthy baseline

**Go:** dataset split, labels, preprocessing and baseline metrics are
reproducible. **No-go:** fix leakage or data errors before policy
experiments.

### Milestone 3 --- Feasible comparison

**Go:** at least one strong baseline and a simple confidence-threshold
policy can be evaluated fairly. **No-go:** simplify the method if
comparisons cannot be reproduced.

### Milestone 4 --- Policy value

**Go:** the proposed policy produces a measurable improvement on at
least one pre-specified reliability--coverage--cost endpoint without
unacceptable regressions. **No-go:** if not, emphasize benchmark
findings and simplify or revise the policy transparently.

### Milestone 5 --- Integrated system

**Go:** end-to-end workflow, provenance, failure handling and experiment
reporting pass acceptance tests.

------------------------------------------------------------------------

## 22. Final demo scenario

A high-quality demonstration should show the whole system rather than a
static collection of model outputs.

1.  Create a new research case.
2.  Upload a permitted sample image.
3.  Run the baseline model and display the output and model version.
4.  Show input quality and uncertainty as separate signals.
5.  Apply a controlled corruption or remove an available modality.
6.  Re-run the system and show whether its action changes.
7.  Supply an additional input when available and compare outcomes.
8.  Display retrieved evidence and source passages for the generated
    summary.
9.  Open the case timeline and inspect the exact evidence snapshot,
    model run, decision and review history.
10. Open DERMA-STRESS and compare baselines across the same test cases.
11. Show risk--coverage, acquisition and cost results, including
    regressions or limitations.
12. Demonstrate a simulated service failure and show that the case
    remains inspectable.

Never use a single handpicked successful case as proof of overall
reliability.

------------------------------------------------------------------------

## 23. Expected deliverables

1.  Working web application.
2.  Versioned backend API and database schema.
3.  Evidence asset storage and case lifecycle.
4.  Baseline vision model and evaluation scripts.
5.  Evidence Manager and decision policy.
6.  RAG/source-attribution subsystem.
7.  DERMA-STRESS configuration and experiment runner.
8.  Baseline comparison and ablation results.
9.  Dataset/split documentation and reproducibility instructions.
10. System architecture diagram and API documentation.
11. Security/privacy and limitation statement.
12. Final project report and presentation.
13. Optional paper submission if the contribution and results justify
    it.

------------------------------------------------------------------------

## 24. Paper structure

1.  Introduction and problem formulation.
2.  Related work: multimodal dermatology, fusion, uncertainty, selective
    prediction, evidence acquisition and robustness.
3.  Proposed evidence-adaptive policy.
4.  DERMA-GUARD platform architecture.
5.  DERMA-STRESS design.
6.  Datasets, baselines, metrics and experimental protocol.
7.  Main results.
8.  Ablation studies.
9.  Evidence-acquisition and computational-cost analysis.
10. Limitations, ethics, and threats to validity.
11. Conclusion.

The paper should distinguish engineering implementation from
methodological contribution. The dashboard and agents demonstrate the
platform; the controlled experiments support the research claims.

------------------------------------------------------------------------

## 25. Risks and mitigation

  -----------------------------------------------------------------------
  Risk                                Mitigation
  ----------------------------------- -----------------------------------
  Closely related methods already     Narrow the claim to the tested
  exist                               decision policy and benchmark;
                                      compare the closest papers

  Dataset lacks required modalities   Verify fields early; use
                                      modality-compatible subsets; keep
                                      image-only experiments separate

  Limited GPU/compute                 Start with one baseline and
                                      pretrained models; cache inference;
                                      reduce ensemble size

  Calibration/uncertainty does not    Report negative result; remove
  help                                unsupported complexity

  RAG produces unsupported claims     Curate sources, cite exact
                                      passages, validate outputs, flag
                                      unsupported claims

  Synthetic stress tests look         Document transformations, use
  unrealistic                         multiple severities, separate
                                      synthetic from external tests

  Acquisition benefit cannot be       Restrict direct claims to cases
  measured                            with available additional inputs;
                                      label simulations clearly

  Integration takes too long          Build vertical slice first; modular
                                      monolith plus workers

  Team members work in silos          Shared interfaces, code review,
                                      integration milestones and joint
                                      demo preparation

  Overclaiming novelty or clinical    Use hypothesis language; report all
  benefit                             outcomes and limitations honestly
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 26. Final project statement

**DERMA-GUARD is an evidence-adaptive dermatology AI research platform
that evaluates whether explicit decisions about evidence quality,
reliability, acquisition and deferral can improve the
reliability--coverage--cost trade-off of multimodal predictions under
controlled evidence failures.**

Its three research pillars are:

1.  Evidence-Adaptive Decision Engine.
2.  DERMA-STRESS robustness evaluation.
3.  Evidence-value and acquisition analysis.

Its engineering pillars are:

-   PatientCase-centred workflow.
-   Vision intelligence.
-   Clinical evidence retrieval and source attribution.
-   Persistent model/assessment history.
-   Review and audit workflow.
-   Reproducible experiment infrastructure.

The central gamble is that making evidence-management decisions explicit
will improve the way the system behaves when evidence is imperfect. The
project will measure whether that hypothesis is supported rather than
assume it is true.

**The next action is to implement the vertical slice:** one dataset, one
baseline classifier, one case record, one saved model run, a visible
result, and a recoverable failure state. Only after that works should
the team expand the fusion policy, RAG, additional models, and full
DERMA-STRESS experiments.
