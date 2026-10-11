# Documentation Index

**D-GUARD v3** — *Risk-Controlled, Cost-Aware Evidence Acquisition and Deferral for
Multimodal Dermatology AI*

This set consolidates the v2.0 specification (MILK10k, three-action policy, Learn-then-Test,
AUGRC, B1–B9 baselines) with the evidential-fusion mechanism, and restores the conformal
guarantee that v1 had and the v4 spec discarded. It **supersedes** all earlier docs.

---

## Read in this order

| # | Doc | Why |
|---|---|---|
| 0 | **[evaluation_integrity.md](evaluation_integrity.md)** | **Start here.** Why no pre-2026-10-11 number can be trusted, with the evidence |
| 1 | [01_project_overview.md](01_project_overview.md) | What this is, four-generation lineage, honest reference numbers, status |
| 2 | [02_research_design.md](02_research_design.md) | Question, gap, contributions A–D, hypotheses H1–H5, gap matrix, novelty position |
| 3 | [03_architecture.md](03_architecture.md) | Decision path, platform, persistence, what was kept vs replaced |
| 4 | [04_methods.md](04_methods.md) | Every quantity derived — EDL, DS, conformal, LTT, cost, AUGRC |
| 5 | [05_experimental_protocol.md](05_experimental_protocol.md) | Datasets, splits, B1–B9, ablations, metrics, statistics, 12-week plan |
| 6 | [06_training_guide.md](06_training_guide.md) | **GPU work — the code you run on Kaggle** |
| 7 | [07_literature.md](07_literature.md) | Verified citations, comparable numbers, corrections, reading order |
| 8 | [08_implementation_status.md](08_implementation_status.md) | What is built and verified; how to reproduce |

## Supporting

| Doc | Contents |
|---|---|
| [prereg.yaml](prereg.yaml) | Pre-registration — commit before touching the test split |
| [limitations.md](limitations.md) | Known limitations |
| [what_did_not_work.md](what_did_not_work.md) | Negative results, including the measured failure of the text modality |
| [results/baseline_probe.json](results/baseline_probe.json) | Raw honest-baseline output, 3 seeds |

---

## Quick answers

**What is this?** A dermatology AI that decides whether to **answer now**, **ask for a
dermoscopic image**, or **defer to a human** — with a statistical bound on how often it
misses a malignancy when it answers. A safety project, not an accuracy project.

**What is the core proposal?** Per-modality Dirichlet evidence combined by
Dempster–Shafer, so absent modalities vanish by algebra and *conflict* is separable from
*ignorance*; feeding a cost-aware three-action rule whose threshold is calibrated by
Learn-then-Test for `P(risk ≤ α) ≥ 1−δ`.

**Does it work?** The machinery does, verified: at matched coverage (0.433) the policy's
malignancy miss rate is **0.0050 vs 0.0597** for max-softmax abstention. Conformal
coverage 0.8944 against a 0.90 target. 425 tests pass. But those numbers run over legacy
weights trained on overlapping data, so they are **not** generalisation estimates.

**Can I trust the old results?** No. The shipped training code evaluated on a test set
that was 100% training data and rewrote predictions toward hardcoded targets. Doc 0.

**What needs a GPU?** Training the evidential heads and the image encoder — the only
remaining blocker. The image pathway is encoder-limited: frozen ImageNet reaches only
0.477 macro-F1 against 0.648 for metadata alone. The trainer is smoke-tested end to end on
CPU against the real MILK10k images, so what ships to the GPU is known to run. Doc 6.

**Why MILK10k and not PAD-UFES-20?** Every MILK10k lesion has **both** a clinical and a
dermoscopic image, so "should I acquire dermoscopy?" is a real experiment with an oracle
upper bound. PAD-UFES-20 has no dermoscopy at all. It stays as the distribution-shift set.

**Is MILK10k in hand?** Yes — downloaded, audited and wired in on 2026-10-11. Verified on
the real release: **11 classes** (not 6), **71.7% malignant**, 95.7% histopathology, zero
duplicate images, and **no centre column**, which kills a centre-shift experiment we had
documented three times. `python scripts/milk10k_audit.py` reproduces all of it. The one
remaining blocker is GPU time — Doc 6.

**What is genuinely novel?** The integration and the benchmark — not the individual
techniques. Evidential deep learning, Dempster–Shafer, conformal prediction,
Learn-then-Test, learning to defer and recommending dermoscopy all have prior art.
Doc 2 §7 is explicit, and names the prohibited claims.

**What happens if the main hypothesis fails?** The paper centres on DERMA-STRESS and the
guarantee-violation analysis. That is a legitimate contribution and needs no change to
the rest of the protocol.

---

## Deleted in this consolidation

The v4 specification (`proj.md`, `det.md`, `detail.md`), the v1 context document, and the
nine superseded doc files — all recoverable from git history. They described models and
datasets that were never built here; where they conflicted with this set, this set wins.
