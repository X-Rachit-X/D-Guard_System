# Architecture

---

## 1. Decision path

```
                    clinical image   dermoscopy   metadata   text
                          │              │           │         │
                    ┌─────▼──────┐  ┌────▼─────┐ ┌───▼───┐ ┌──▼────┐
                    │ DermLIP /  │  │  same    │ │  MLP  │ │ BERT  │
                    │ ResNet50   │  │ encoder  │ │(train-│ │ [CLS] │
                    └─────┬──────┘  └────┬─────┘ │  fit) │ └──┬────┘
                          │              │       └───┬───┘    │
                    EvidentialHead  EvidentialHead   │   EvidentialHead
                     softplus→Dir   softplus→Dir  Evid.Head  softplus→Dir
                          │              │           │         │
                    (b,u)_clin     (b,u)_dermo  (b,u)_meta (b,u)_text
                          │              │           │         │
                          │         [ABSENT → vacuous (b=0,u=1) = IDENTITY]
                          └──────────────┴─────┬─────┴─────────┘
                                               │
                              ╔════════════════▼═════════════════╗
                              ║  Dempster–Shafer fold  (⊕)       ║
                              ║  associative · order-free        ║
                              ║  → belief b, ignorance u,        ║
                              ║    conflict C                    ║
                              ╚════════════════╤═════════════════╝
                                               │
                                    p = b + u/K   (pignistic)
                                               │
        ┌──────────────────────────────────────┼──────────────────────────────┐
        │                                      │                              │
  conformal set                        expected loss                   LTT λ̂
  (APS/RAPS,                   R_proceed / R_acquire / R_defer      P(R≤α)≥1−δ
   class-conditional)          under the cost matrix
        │                                      │                              │
        └──────────────────────────────────────┼──────────────────────────────┘
                                               │
                         ┌─────────────────────┼─────────────────────┐
                    PROCEED(ŷ)          ACQUIRE(dermoscopy)      DEFER(reason)
                                                                 CONFLICT
                                                                 AMBIGUOUS_SET
                                                                 RISK_BOUND
                                                                 COST
```

Three gates on PROCEED, each able only to increase caution. **Conflict short-circuits
ACQUIRE** — a contradiction between existing sources is not resolved by collecting more.

---

## 2. Platform

Modular monolith plus one background worker. No microservices, no Kafka, no Kubernetes.

```
  intake ──► validate ──► store asset ──► enqueue job
                                             │
                      ┌──────────────────────┼──────────────────────┐
              vision encoders         consistency module      evidence manager
              (+ evidential heads)    (DS conflict C)         (versioned record)
                      └──────────────────────┼──────────────────────┘
                                             ▼
                                    EADE (src/agents/eade.py)
                                             ▼
                        PROCEED | ACQUIRE | DEFER  + reason codes
                                             ▼
                     append-only assessment + audit event ──► UI / API
```

| Layer | Choice |
|---|---|
| API | FastAPI + Pydantic (`src/main.py`) |
| UI | Streamlit now; Next.js optional later |
| DB | PostgreSQL (+ pgvector if retrieval is enabled) |
| Jobs | Redis + RQ, one queue |
| ML | PyTorch, torchvision, timm, scikit-learn |
| Conformal / risk | in-house (`src/calibration/`), ~400 lines, easier to explain than a dependency |
| Trees | LightGBM / sklearn GBM (acquisition estimator) |
| Experiments | YAML configs + JSON logs |

Cut from the v2 plan: OpenTelemetry/Grafana, Qdrant (pgvector instead), frontend
overview/evidence-source/admin pages. The research core is roughly 20% of the original
engineering list; evaluation quality comes before features.

---

## 3. Persistence

Every assessment links to the exact evidence snapshot, model versions, calibration
version, cost-matrix version and policy version that produced it. Append-only — prior
assessments are never overwritten.

| Table | Key fields |
|---|---|
| `PatientCase` | id, status, created_by, access_metadata |
| `EvidenceAsset` | case_id, asset_type, storage_key, content_hash, quality_scores |
| `EvidenceSnapshot` | case_id, asset_ids[], missingness_record |
| `Assessment` | snapshot_id, policy_version, calibration_version, cost_matrix_version, `lambda_hat`, action, reason_codes[], conformal_set, expected_losses{} |
| `ModelRun` | model_name/version, preprocessing_version, evidence, `u`, `C`, ood_score, runtime |
| `ReviewDecision` | reviewer_id, decision, rationale |
| `AuditEvent` | actor_type, actor_id, event_type, payload |
| `ExperimentRun` | dataset_version, split_version, stress_config, config_hash, seed, metrics |

---

## 4. What is kept from the earlier system

| Component | Why |
|---|---|
| `TrainedBackends` encode-once seam | clean artifact/logic separation; the stress benchmark depends on it |
| Specialist routing by availability | correct and useful; becomes the baseline ladder |
| `src/constants.py` single source of truth | stops the axis-length and class-ordering bugs recurring |
| Stage-1 / Stage-2 report verification | good engineering for report safety |
| DERMA-STRESS | a genuine contribution |
| Orchestrator + explicit failure policy | `action="ERROR"` naming the stage, never a fabricated prediction |
| Legacy 4-level ladder | **kept behind `use_eade=False`** — "heuristic ladder vs calibrated rule" is one of the results |
| FastAPI + Streamlit | product surface |

## 5. What was replaced

| Was | Now |
|---|---|
| `Q`, `R`, `U` hand-tuned formulas | Dirichlet ignorance `u_m` |
| mask + renormalise | vacuous opinion ⊕ DS identity |
| `1 − mean JS` consistency | DS conflict mass `C` |
| `WeightLogitNetwork` (untrained) | closed-form DS fold |
| 4-level ladder (0.80/0.50/0.30) | expected loss + LTT `λ̂` + 3 gates |
| AURC | AUGRC |
| 3-seed ensemble | single-pass EDL |

## 6. Dropped from the v4 specification

| Dropped | Reason |
|---|---|
| Derm7pt as primary | MILK10k is ~5× larger, paired, and 96% histopath-confirmed |
| 5-model foundation-model ensemble | CPU-only environment; shipped with random heads voting on noise |
| Iterative diagnosis-adaptive fusion | derivation attached to a question nobody asked |
| LLM orchestration as a *scientific* claim | kept as engineering; it writes reports, it does not diagnose |
| RAG in the decision path | kept as an explanation feature only — admitting a live corpus breaks exchangeability and forfeits the risk guarantee |

**Reinstated:** conformal prediction (v1 had it, v4 dropped it), now as conformal sets
plus LTT risk control; and **MetaBlock** as the literature baseline.

---

## 7. Shapes

```
clinical  (PIL) → Resize(224,224) → ToTensor → ImageNet-norm → (1,3,224,224)
                → encoder → (1,D) → proj → (1,256) → EvidentialHead → α (1,K)
dermoscopy      → same encoder (separate head)
metadata (dict) → engineer → impute → scale ⊕ one-hot → (1,M) → (1,256) → α
text     (str)  → tokenizer(128) → BERT → [CLS] (1,768) → (1,256) → α

fold: (b,u)_clin ⊕ (b,u)_dermo ⊕ (b,u)_meta ⊕ (b,u)_text → b (1,K), u (1,1), C (1,1)
```

Two facts that have each already caused a bug:

- **Metadata width depends on where the preprocessor was fitted.** Train-only fitting
  gives 114 dims on PAD-UFES-20, not 116, because two one-hot levels are absent from
  train. Correct behaviour; specialists must be built with the matching
  `meta_input_dim`.
- **`[CLS]`, not mean-pooling.** The trained text projector was fitted on `[CLS]`;
  mean-pooled vectors put it off-distribution. A deliberate deviation from the v4 spec.
