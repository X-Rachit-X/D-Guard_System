# Architecture

Two architectures are documented here: what exists now (and which parts of it are being
kept), and the proposed target. The point of showing both is that the migration is
incremental — most of the engineering survives; the evidence layer is what changes.

---

## 1. Current architecture (as built)

```
                              PatientCase
                   (clinical image | metadata dict | free text)
                                    │
                        ┌───────────┴────────────┐
                        │   TrainedBackends      │   encode once, share everywhere
                        │  ResNet50→img_proj→256 │
                        │  BERT [CLS]→text_proj→256
                        │  MetadataPreprocessor→116
                        └───────────┬────────────┘
                                    │
          ┌─────────────┬───────────┼───────────┬──────────────┐
          │             │           │           │              │
    SpecialistTeam  ScribeAgent  Librarian  DevilsAdvocate  (LLM client)
    routes to A–G   PTIS,        FAISS kNN  Q, R, U, C, M,
    by availability symptoms                 Mahalanobis OOD
          │             │           │           │
          └─────────────┴───────────┴───────────┘
                                    │
                            Stage-1 verification
                          (OOD / conflict / no-evidence gate)
                                    │
                      DiagnosisAdaptiveFusion  (4 modalities)
                   w ∝ softmax(log(Q·R·U)) ⊙ mask, renormalised
                   q ∝ ∏ₘ pₘ^{wₘ}  (log-opinion pool over A/B/C)
                        3 EM-ish iterations, Δw < 0.05
                                    │
                        ConsensusHierarchy (4 levels)
              L1 ACCEPT · L2 ARBITRATE · L3 ACQUIRE · L4 ESCALATE
                                    │
                    ┌───────────────┼───────────────┐
            AcquisitionPolicy   report_gen     Stage-2 verification
            (Δuncertainty/cost)  (LLM or        (5 rules + LLM,
                                 deterministic)  conservative retry)
```

**Specialists A–G** — each is routed in only if its modalities are present:

| | modalities | architecture |
|---|---|---|
| A | image | ResNet50 + channel attention + MHSA ViT |
| B | metadata | residual MetaBlock-style MLP |
| C | text | Bio_ClinicalBERT |
| D/E/F | pairs | gated fusion (`AdaptiveGatedFusion`) |
| G | all three | gated fusion |

### What is kept

| Component | Why |
|---|---|
| `TrainedBackends` encode-once seam | Clean separation between artifacts and logic; the stress benchmark depends on it |
| Specialist routing by availability | Correct and useful; becomes the baseline ladder |
| `src/constants.py` single source of truth | What stops the axis-length and class-ordering bugs recurring |
| Stage-1 / Stage-2 verification | Good engineering for report safety |
| Acquisition policy | The "what to collect next" surface is genuinely useful |
| DERMA-STRESS | A real contribution |
| Orchestrator + explicit failure policy | `action="ERROR"` naming the stage, never a fabricated prediction |
| FastAPI + Streamlit | Product surface |

### What is replaced

| Component | Replaced by | Reason |
|---|---|---|
| `Q`, `R`, `U` hand-tuned formulas | Dirichlet ignorance `u_m` | Coefficients were invented; four of these quantities came from untrained MLPs |
| mask + renormalise | vacuous opinion ⊕ DS identity | Masking is not marginalisation |
| `1 − mean JS` consistency | DS conflict mass `C` | JS cannot separate conflict from shared ignorance |
| `WeightLogitNetwork` | closed-form `w` | Removes the last untrained network from the safety path |
| 4-level ladder, thresholds 0.80/0.50/0.30 | conformal risk control + Chow rule | No guarantee → distribution-free guarantee |
| hard MAP diagnosis prior | soft expectation over `q` | Restores monotone convergence |

---

## 2. Target architecture (D-GUARD v5, "evidential")

```
                              PatientCase
                                    │
                        ┌───────────┴────────────┐
                        │   TrainedBackends      │   (unchanged)
                        └───────────┬────────────┘
                                    │
        ┌───────────────────┬───────┴───────┬───────────────────┐
        │                   │               │                   │
   image encoder      metadata encoder   text encoder      [absent modality]
  (FINE-TUNED)        (MLP, train-fit)  (BERT [CLS])            │
        │                   │               │                   │
   EvidentialHead      EvidentialHead   EvidentialHead      vacuous opinion
   softplus → Dir(α)   softplus → Dir(α) softplus → Dir(α)   b=0, u=1
        │                   │               │                   │
    (b_img, u_img)     (b_meta, u_meta) (b_text, u_text)    (identity)
        └───────────────────┴───────┬───────┴───────────────────┘
                                    │
                    ╔═══════════════▼════════════════╗
                    ║  Dempster–Shafer fold  (⊕)     ║  associative, order-free
                    ║  → belief b, ignorance u,      ║  absent modality = identity
                    ║    conflict mass C             ║
                    ╚═══════════════╤════════════════╝
                                    │
                    ╔═══════════════▼════════════════╗
                    ║  EM refinement (monotone in J) ║  E: q ∝ exp(Σ wₘ log pₘ + β…)
                    ║  diagnosis-adaptive prior W    ║  M: wₘ ∝ exp(λ·E_q[log pₘ]…)
                    ╚═══════════════╤════════════════╝
                                    │
                    ╔═══════════════▼════════════════╗
                    ║  Risk-controlled deferral      ║
                    ║  λ̂ from Learn-then-Test,       ║  P(R(λ̂) ≤ α) ≥ 1−δ
                    ║  class-conditional (Mondrian)  ║
                    ║  + Chow rule on cost matrix    ║
                    ╚═══════════════╤════════════════╝
                                    │
              ┌─────────────────────┼─────────────────────┐
         predict ŷ            defer: CONFLICT        defer: IGNORANCE
                              (→ human review)       (→ acquire evidence)
                                                     defer: COST
                                                     (→ escalate by risk)
                                    │
                      report_gen + Stage-2 verification  (unchanged)
```

### The three structural changes

**1. Evidence is produced, not computed from heuristics.** Each encoder emits
non-negative evidence; belief and ignorance follow from the Dirichlet. No formula with
hand-chosen coefficients remains.

**2. Absence is algebraic.** The vacuous opinion `(b=0, u=1)` is the identity element of
`⊕`, so a missing modality literally changes nothing. This deletes the mask vector, the
renormalisation step, and the all-missing zero-division guard — roughly 40 lines of
special-casing replaced by an associative fold.

**3. Deferral is calibrated, and the reasons are distinct.** One threshold, fitted to
achieve a stated risk level with high probability, plus three *separable* deferral
causes. The current ladder gives four levels whose meanings overlap; conflict, ignorance
and cost call for different clinical responses and should not share a code path.

---

## 3. Migration path

Incremental, each step independently testable. Nothing below requires discarding the
current system.

| Step | Change | Touches | Risk |
|---|---|---|---|
| 1 | Reproducible features + manifest | `scripts/build_features.py` | **Done** |
| 2 | Honest baselines | `scripts/baseline_probe.py` | **Done** |
| 3 | Fine-tune image encoder on GPU | `notebooks/kaggle_train_dguard.py` | Low — isolated |
| 4 | Add `EvidentialHead` beside existing heads | new module | Low — additive |
| 5 | Add `ds_combine` as an alternative fusion op | `src/fusion/` | Low — selected by config |
| 6 | Soft diagnosis prior in the EM loop | `adaptive_fusion.py` | Low — small edit, §4.3 of maths doc |
| 7 | Conformal calibration on saved predictions | new script | Low — pure post-processing |
| 8 | Swap the ladder for the calibrated rule | `consensus.py` | Medium — keep the ladder behind a flag for comparison |
| 9 | Re-run DERMA-STRESS on both | `scripts/run_stress.py` | Low |

Keeping the ladder behind a config flag matters: the comparison *heuristic ladder vs
calibrated rule* is one of the results (**H3**), so the old path must stay runnable.

---

## 4. Data flow and shapes

```
image   (PIL) → Resize(224,224) → ToTensor → ImageNet-norm → (1,3,224,224)
              → backbone → (1,2048) → img_proj → (1,256) → head → α_img (1,6)

metadata(dict)→ engineer() → impute → scale ⊕ one-hot → (1,114)
              → meta_proj → (1,256) → head → α_meta (1,6)

text    (str) → tokenizer(max_len=128) → BERT → [CLS] (1,768)
              → text_proj → (1,256) → head → α_text (1,6)

fuse:   (b,u)_img ⊕ (b,u)_meta ⊕ (b,u)_text → b (1,6), u (1,1), C (1,1)
        q = b + u/6
```

Two shape facts worth writing down, because both have already caused bugs:

- **Metadata is 114-d when the preprocessor is fitted on train only**, not 116. Two
  one-hot levels do not occur in the training split. This is correct behaviour, and
  specialists must be constructed with the matching `meta_input_dim`.
- **`[CLS]`, not mean-pooling.** The trained text projector was fitted on `[CLS]`;
  mean-pooled vectors put it off-distribution. This deviates from `proj.md` Part 3 and
  the deviation is deliberate.

---

## 5. Components explicitly dropped from the v4 specification

| Dropped | Reason |
|---|---|
| Derm7pt dataset | Needs dermoscopy, which PAD-UFES lacks; the two never shared a pipeline. One dataset done rigorously beats two done partially |
| DermLIP-PanDerm, PanDerm | Weights unpublished/unavailable **[verify]**; shipped with random heads and voted on noise |
| Swin-Tiny ×2, ResNet152 | CPU-only environment; untrained heads. Superseded by one fine-tuned backbone |
| LLM orchestration as a *scientific* claim | Kept as engineering. Attributing diagnostic credit to the LLM would be unsupportable — it writes reports, it does not diagnose |
| Conformal prediction *as originally dropped* | **Reinstated**, upgraded to conformal risk control |

The v4 spec's instincts were right and its scope was not. The target architecture keeps
every idea that had a defensible mechanism and discards the ones that existed only as
configuration files.
