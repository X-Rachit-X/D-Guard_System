# Kaggle Training Guide

Everything that needs a GPU, packaged so it can be run on Kaggle and the artifacts
handed back.

Script: [`notebooks/kaggle_train_dguard.py`](../notebooks/kaggle_train_dguard.py)
(single file, no local imports — paste into one cell or add as a notebook utility
script).

---

## 1. Why this has to run off-box

| Job | CPU (this machine) | Kaggle GPU |
|---|---|---|
| Frozen-feature extraction | ~4.5 min total | — (already done locally) |
| Linear probes on frozen features | seconds | — (already done locally) |
| **Fine-tuning the image encoder** | ~10 min/epoch × 25 = **4+ hours** | **~1.5 min/epoch, ~40 min total** |
| Fine-tuning BERT as well | infeasible | ~2 hours |

The reason this matters is specific, not generic. With a **frozen** ImageNet ResNet50 the
image modality reaches only **0.477 macro-F1**, while metadata alone reaches **0.648**
([01_system_analysis.md](01_system_analysis.md) §4). ImageNet features are simply not
adapted to dermoscopic texture. Fine-tuning the backbone is the single largest expected
gain in the whole project, and it is the one thing that cannot be done here.

---

## 2. Dataset

**PAD-UFES-20** — Pacheco et al., *Data in Brief*, 2020.
Authoritative source: Mendeley Data, DOI `10.17632/zr7vgbcyr2.1`.
On Kaggle: search "PAD-UFES-20". Several mirrors exist and **their directory layouts
differ**, which the script tries to auto-detect but cannot guarantee.

### Strongly recommended: upload your own copy

You already have the exact bytes at `data/raw/`. Upload those as a **private Kaggle
dataset**:

```
Kaggle → Datasets → New Dataset → Upload
    metadata.csv                        (from data/raw/metadata.csv)
    images/                             (all 2298 files from data/raw/images/)
    generated_clinical_texts.csv        (optional; from Dummy/Project/data/)
```

This is not fussiness. The previous generation of this project was invalidated partly
because cached features could not be reconciled with **any** available image copy —
cosine similarity topped out at 0.07 against six preprocessing variants
([evaluation_integrity.md](evaluation_integrity.md) §2). Training on the identical bytes
you hold locally removes that entire failure mode, and lets the returned model be
verified against local features.

~3.4 GB; expect 20–40 min to upload.

---

## 3. Running it

1. New Notebook → **Accelerator: GPU T4 ×2** (or P100) → **Internet: On**
   (needed once, to pull Bio_ClinicalBERT and the ImageNet weights).
2. Add your dataset.
3. Set the path and run:

```python
!pip -q install transformers

import sys; sys.path.append('/kaggle/input/<your-code-dataset>')   # or paste the file inline
import kaggle_train_dguard as T

T.CFG.update(
    DATA_DIR="/kaggle/input/<your-dataset-slug>",
    OUT_DIR="/kaggle/working/dguard_out",
    EPOCHS=25,
    BATCH_SIZE=32,
    SEED=42,
)
results = T.run(T.CFG)
```

Then repeat for `SEED=7` and `SEED=2026` into separate `OUT_DIR`s — the protocol requires
three seeds ([07_experimental_protocol.md](07_experimental_protocol.md) §2).

**Sanity checks in the first minute of output:**

```
cases=2298  patients=1373          <- both numbers must match
train n=1623 patients=961          <- patients, not cases, are split
metadata dim = 114                 <- 114 not 116 is CORRECT (fit on train only)
class weights: [0.45, 0.52, 1.57, ...]   <- MEL weight should be the largest
backbone frozen for first 3 epochs
```

If `cases` is much below 2298, the metadata filter dropped rows — check the `diagnostic`
column values.

---

## 4. What to send back

The whole `OUT_DIR` (a few hundred MB, mostly the checkpoint):

| File | Contents | Why it is needed |
|---|---|---|
| `checkpoints/best.pt` | weights + config + `meta_dim` | To load and run locally |
| `metrics.json` | all test/val metrics, risk–coverage, availability ablation | The results |
| `predictions_test.csv` | per-case `y_true`, `y_pred`, class probs, `u`, `C` | **Most important** — lets any metric be recomputed, and conformal calibration be fitted, without retraining |
| `history.json` | real per-epoch loss and val macro-F1 | Honest training curves |
| `manifest.json` | versions, seeds, split sizes, config | Reproducibility |

`predictions_test.csv` is the one to prioritise if upload size is a problem: conformal
risk control, cost-sensitive thresholds, calibration diagnostics and the risk–coverage
analysis all operate on saved predictions, so most of the remaining work needs only that
file.

---

## 5. What the script does

```
PAD-UFES-20
   ├── image   → ResNet50 (fine-tuned, last stages) → 256-d → evidential head → Dir(α_img)
   ├── metadata→ residual MLP (fit on train only)   → 256-d → evidential head → Dir(α_meta)
   └── text    → Bio_ClinicalBERT [CLS] (frozen)    → 256-d → evidential head → Dir(α_text)
                                                                    │
                            Dempster–Shafer fold (vacuous = identity for absent)
                                                                    │
                                          belief b, ignorance u, conflict C
                                                                    │
                                            pignistic prob = b + u/K
```

Implements: evidential Dirichlet heads (§2 of the maths doc), DS combination (§3),
modality dropout at 0.25, class-weighted EDL loss with annealed KL, gradient clipping at
1.0, mixed precision, backbone frozen for 3 epochs then unfrozen at 0.1× LR, and early
stopping on **validation macro-F1**.

**Not yet implemented in the script** (deliberately — these run locally on the returned
predictions): the EM diagnosis-adaptive refinement (§4), conformal risk control (§5), and
cost-sensitive deferral (§6).

### Honesty guarantees in the code

- Patient-level disjointness is **asserted at runtime**, not assumed
- metadata preprocessor is fitted on **train only**
- metrics are recorded **as measured** — no clipping, no target-matching, no prediction
  rewriting
- per-case predictions are saved so every number can be independently recomputed

Those four lines exist because each corresponds to something that went wrong before.

---

## 6. Expected results

From the honest frozen-feature baselines, fine-tuning should move the image modality
substantially. Reasonable expectations:

| Configuration | Frozen (measured) | Fine-tuned (expected) |
|---|---|---|
| image only | 0.477 macro-F1 | 0.58 – 0.68 |
| metadata only | 0.648 | ~0.65 (unchanged — no backbone) |
| text only | 0.414 | ~0.42 (BERT frozen) |
| image + metadata | 0.653 | **0.70 – 0.78** |
| all three (DS fusion) | 0.633 naive | should **match or beat** image+metadata |

The last row is the test of **H1**: the DS fusion must not be dragged down by the weak
text channel the way naive concatenation is. Check this in the
`availability_ablation` block of `metrics.json` — specifically whether
`image+meta+text ≥ image+meta`.

**If `image+meta+text` is clearly worse than `image+meta`, H1 has failed** and that is a
genuine, reportable negative result, not a bug to tune away. Report it in
`what_did_not_work.md`.

Also worth watching: mean ignorance `u` rises during the first epochs (the annealed KL
suppressing evidence) and should then fall as the model becomes confident. If `u` stays
near 1.0 at the end, the KL weight is too strong — reduce `EDL_ANNEAL_EPOCHS` or scale
the KL term down.

---

## 7. Optional extras, in priority order

1. **Three seeds** (42 / 7 / 2026) — required by the protocol; do this before anything
   else.
2. **MetaBlock baseline** — the reference method on this dataset (Pacheco & Krohling);
   a reviewer will ask. Needs a small extra model class.
3. **Unfreeze BERT** (`FREEZE_TEXT=False`, `EPOCHS=15`) — tests whether the text channel
   is weak because of the encoder or because the synthetic narratives carry little
   signal. Given Model C's measured 27%, the narratives are the more likely cause, and
   that is worth establishing.
4. **Backbone sweep** — `efficientnet_b0`, `convnext_tiny` via `IMAGE_BACKBONE`.
5. **Image size 320** — lesion texture may benefit; costs ~2× time.

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `metadata.csv not found` | The script globs for it; check the printed directory listing and set `DATA_DIR` deeper |
| `no image directory found` | Set `IMAGES_SUBDIR` to the actual folder name |
| All predictions one class | Class weights not applied, or LR too high. Check the printed weights — MEL's should be largest |
| `u` pinned at 1.0 | KL anneal too aggressive; lower `EDL_ANNEAL_EPOCHS` |
| CUDA OOM | `BATCH_SIZE=16`, or `IMAGE_SIZE=192` |
| Val macro-F1 flat at ~0.1 | Backbone never unfroze — confirm the "backbone unfrozen" line appears |
| Much worse than the frozen baseline | Fine-tuning diverged; lower `LR_BACKBONE` to 5e-5 |

---

## 9. After the model comes back

Locally, with `predictions_test.csv`:

```bash
# calibrate the deferral threshold with a distribution-free risk guarantee
python scripts/calibrate_conformal.py --predictions dguard_out/predictions_test.csv --alpha 0.10

# risk-coverage curves, AURC, E-AURC
python scripts/analyze_selective.py --predictions dguard_out/predictions_test.csv

# robustness under the 18 corruption conditions
python scripts/run_stress.py --limit 300
```

The first two scripts are the next thing to write — they are pure post-processing on the
saved predictions and need no GPU. Steps 9–10 of the semester plan
([07_experimental_protocol.md](07_experimental_protocol.md) §7).
