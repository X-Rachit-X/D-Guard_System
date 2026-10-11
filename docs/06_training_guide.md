# Training Guide (GPU work)

Everything that needs a GPU, packaged to run on Kaggle and hand the artifacts back.

Script: [`notebooks/kaggle_train_dguard.py`](../notebooks/kaggle_train_dguard.py) —
single file, no local imports, smoke-tested end-to-end on CPU. Supports **MILK10k**
(primary) and **PAD-UFES-20** (shift set) via `CFG["DATASET"]`.

---

## 1. Why this runs off-box

| Job | CPU (this machine) | Kaggle GPU |
|---|---|---|
| Frozen-feature extraction | ~4.5 min | done locally |
| Linear probes | seconds | done locally |
| EADE calibrate + evaluate | ~2 min | done locally |
| **Training the evidential heads + encoder** | ~10 min/epoch × 25 = **4+ h** | **~1.5 min/epoch, ~40 min** |

The specific reason: with a **frozen** ImageNet ResNet50 the image modality reaches only
**0.477 macro-F1**, while metadata alone reaches **0.648**. The image pathway is
*encoder-limited*, and fixing that is the single largest expected gain in the project.

---

## 2. Datasets

### MILK10k — primary

| | |
|---|---|
| DOI | `10.34970/648456` (benchmark: `10.34970/262082`) |
| Licence | **CC-BY-NC** — non-commercial, credit the MILK study team |
| Size | 344.7 MB bundle · 10,480 images / 5,240 lesions |
| Access | **ISIC account required** — the bundle URL is authenticated (403 without it) |
| Status | **downloaded, audited and wired in** (2026-10-11, extracted at `./milk10k`) |

**The layout is known exactly, so the loader reads it exactly** — the old
`milk_detect_columns` pattern-matcher has been deleted. Upload the extracted directory as
a **private Kaggle dataset** (CC-BY-NC: private only, never public) and point `DATA_DIR`
at it. Required structure:

```
<DATA_DIR>/metadata.csv                      10,480 rows, one per IMAGE
<DATA_DIR>/supplements/training_gt.csv         5,240 rows, one-hot over 11 classes
<DATA_DIR>/supplements/training_input.csv     10,480 rows, skin_tone_class + 7 MONET
<DATA_DIR>/images/<isic_id>.jpg               10,480 files
```

Anything missing raises immediately rather than training on a partial dataset.

**Verified composition** — measured, not quoted. Run `python scripts/milk10k_audit.py`
to reproduce all of it:

| | |
|---|---|
| classes | **11**: `AKIEC BCC BEN_OTH BKL DF INF MAL_OTH MEL NV SCCKA VASC` |
| pairing | every lesion has exactly one `clinical: close-up` and one `dermoscopic` image |
| malignant | **71.7%** (BCC alone 48.1%) — referral population, not screening |
| histopathology | 95.7% · duplicates **none** (all 10,480 byte-distinct) |
| rarest class | MAL_OTH, **9 lesions total** |
| metadata | age, sex, site, skin tone 0–5, 7 MONET scores → **26-d** encoded |

Two things to know before you read the output:

- **Do not expect the 6-class PAD-UFES numbers.** K=11 with a 48% majority class. Quote
  **balanced accuracy and macro-F1**, never accuracy.
- **Miss rate alone is meaningless here.** At 71.7% malignant, always-predict-malignant
  scores a 0% miss rate. Always quote it beside coverage and the base rate.

### PAD-UFES-20 — shift set

Already prepared locally. Upload `data/raw/` as a private Kaggle dataset and set
`DATASET="pad_ufes"`. **Use your own copy** — the previous generation of this project was
invalidated partly because cached features could not be reconciled with any available
image copy (cosine ≤ 0.07 across six preprocessing variants). Training on the identical
bytes you hold removes that failure mode entirely.

---

## 3. Running it

1. New Notebook → **GPU T4 ×2** or P100 → **Internet: On** (to fetch weights once).
2. Add your dataset.
3. For `IMAGE_BACKBONE="dermlip_panderm"`: accept the terms at
   [the model page](https://huggingface.co/redlessone/DermLIP_PanDerm-base-w-PubMed-256)
   and add your HF token under **Add-ons → Secrets** as `HF_TOKEN`.

```python
!pip -q install transformers open_clip_torch

import sys; sys.path.append('/kaggle/input/<your-code-dataset>')
import kaggle_train_dguard as T

T.CFG.update(
    DATASET="milk10k",
    DATA_DIR="/kaggle/input/<your-milk10k-slug>",
    OUT_DIR="/kaggle/working/dguard_milk_s42",
    IMAGE_BACKBONE="dermlip_panderm",   # or "resnet50"
    WARMUP_FROZEN_EPOCHS=25,            # == EPOCHS => linear probe (see §4)
    EPOCHS=25, BATCH_SIZE=32, SEED=42,
)
T.run(T.CFG)
```

Repeat for `SEED=7` and `SEED=2026` into separate `OUT_DIR`s — the protocol requires
three seeds.

**Sanity checks in the first minute.** This is the real output — anything different
means the dataset upload is wrong, so stop rather than let it train:

```
[milk10k] 5240 lesions, 11 classes, 71.7% malignant
[milk10k] K=11 classes: ['AKIEC', 'BCC', 'BEN_OTH', 'BKL', 'DF', 'INF', 'MAL_OTH',
                         'MEL', 'NV', 'SCCKA', 'VASC']
class counts: {'BCC': 2522, 'NV': 746, 'BKL': 544, 'SCCKA': 473, 'MEL': 450,
               'AKIEC': 303, 'DF': 52, 'INF': 50, 'VASC': 47, 'BEN_OTH': 44, 'MAL_OTH': 9}
ground truth: {'histopathology': 5016, 'single contributor clinical assessment': 224}
skin tone   : {'0': 6, '1': 105, '2': 506, '3': 3174, '4': 1066, '5': 383}
  train n= 3668 lesions= 3668              <- lesions, not images, are split
metadata dim = 26 (MILK10k schema, fitted on train only)
class weights: [...]                       <- rarest class should have the largest
backbone frozen for first 25 epochs        <- linear probing
```

The counts above are fixed properties of the release. If `lesions` is not 5240 or
`class counts` differs, the upload is incomplete.

**Smoke-test it first.** The script takes `DGUARD_<KEY>` environment overrides, so the
exact file you are about to run on a GPU can be proven to work on a CPU in a few minutes:

```bash
DGUARD_DATA_DIR=milk10k DGUARD_OUT_DIR=outputs/smoke DGUARD_EPOCHS=1 DGUARD_LIMIT_LESIONS=66 DGUARD_BATCH_SIZE=8 DGUARD_AMP=0 DGUARD_NUM_WORKERS=0 DGUARD_WARMUP_FROZEN_EPOCHS=1 python notebooks/kaggle_train_dguard.py
```

That path is verified. Its *metrics are meaningless* — 46 training lesions over 11
classes — but it proves the loader, the encoder, the training loop, the 8-way
availability ablation and every output file. Delete `outputs/smoke` afterwards.

---

## 4. DermLIP_PanDerm — three caveats

Weights are public and the repo id is exactly the one the original spec named. Verified.

1. **Gated.** Needs an HF token. The original implementation failed silently here and
   fell back to ImageNet — its own error message said
   `run huggingface-cli login and accept the model license`.
2. **Licence labelled inconsistently.** `cc-by-nc-4.0` in the repo metadata vs
   **CC-BY-NC-ND** in the model details. **"ND" (NoDerivatives) arguably forbids
   fine-tuning**, so set `WARMUP_FROZEN_EPOCHS >= EPOCHS` to keep the tower frozen and
   train only the heads. The authors list linear probing as a supported use, so this is
   also the sanctioned path.
3. **Possible pretraining overlap.** PanDerm was trained on >2M images from 11
   institutions. **Check the paper's data list for MILK10k and PAD-UFES-20 before
   claiming a clean probe** — this is the same class of problem that invalidated the
   previous results. If there is overlap, report ResNet50 numbers as the clean headline
   and DermLIP as an upper bound.

---

## 5. What to send back

The whole `OUT_DIR` per seed:

| File | Why |
|---|---|
| `checkpoints/best.pt` | weights + config + `meta_dim` |
| **`predictions_test.csv`** | **highest priority** — per-case `y_true`, `y_pred`, class probs, `u`, `C` |
| `metrics.json` | test/val metrics, risk–coverage, availability ablation |
| `history.json` | real per-epoch loss and val macro-F1 |
| `manifest.json` | versions, seeds, split sizes, config |

`predictions_test.csv` is the one that matters most: **conformal calibration, LTT
threshold selection, cost-sensitive thresholds, AUGRC and the risk–coverage analysis all
run locally on saved predictions** with no GPU. If upload size is a problem, send that
file alone and the remaining analysis can proceed.

---

## 6. What the script does

```
clinical image ─┐
dermoscopy    ─┤─► shared backbone ─► proj 256 ─► EvidentialHead ─► Dir(α)  (separate head each)
metadata      ─┤─► residual MLP (train-fit) ─────► EvidentialHead ─► Dir(α)
text          ─┘─► Bio_ClinicalBERT [CLS] ───────► EvidentialHead ─► Dir(α)
                                   │
                [ABSENT modality → vacuous (b=0, u=1) = DS identity]
                                   │
                   Dempster–Shafer fold → belief b, ignorance u, conflict C
                                   │
                          pignistic p = b + u/K
```

Implements: evidential Dirichlet heads, DS combination over **4 modalities**, modality
dropout 0.25, class-weighted EDL loss with annealed KL, gradient clipping 1.0, mixed
precision, early stopping on **validation macro-F1**, and an **8-way availability
ablation** on test.

**Not in the script** (runs locally on the returned predictions): conformal calibration,
LTT thresholds, cost-sensitive policy, DERMA-STRESS.

### Honesty guarantees in the code

- lesion/patient disjointness **asserted at runtime**
- metadata preprocessor fitted on **train only**
- metrics recorded **as measured** — no clipping, no target-matching, no rewriting
- per-case predictions saved so every number is independently recomputable

Each of those lines exists because the corresponding thing went wrong before.

---

## 7. Expected results and the key check

**Read this before comparing anything.** The reference column below is **PAD-UFES-20,
6 classes**. MILK10k is **11 classes with a 48% majority (BCC)**, so its macro-F1 will be
numerically *lower* for an equally good model — a weaker number is not a regression.
Across datasets, compare only:

- **balanced accuracy** and **macro-F1** (never accuracy — a 48% majority makes it useless)
- the **shape** of the availability ablation, which is what H1 is about
- **AUGRC** and coverage-matched miss rate, quoted beside the 71.7% base rate

| Configuration | frozen ImageNet (measured, PAD-UFES) | expected |
|---|---|---|
| image only | 0.477 macro-F1 | 0.58–0.68 fine-tuned; **0.65–0.75 DermLIP probe** |
| metadata only | 0.648 | ~0.65 (no backbone) |
| text only | 0.414 | ~0.42 (BERT frozen) |
| image + metadata | 0.653 | 0.70–0.78 |
| **all modalities (DS fusion)** | 0.633 *naive concat* | **must match or beat the best pair** |

**The check that matters** — in `metrics.json → availability_ablation`:

> Is `clin+derm+meta` **≥** the best two-modality combination?

On MILK10k this comparison is *stronger* than it was on PAD-UFES-20, because
`clin+derm+meta` and `clin+derm` differ by a genuinely present dermoscopic image rather
than by a simulated absence. This is the whole reason the dataset was switched.

That is **H1**. Naive concatenation fails it (0.633 vs 0.653 on PAD-UFES). Evidence-aware
fusion should pass it *without being told which modality is weak*, because a weak
modality produces little evidence, hence `u → 1`, hence the vacuous opinion, hence no
influence.

**If `clin+derm+meta` is clearly worse than the best pair, H1 has failed.** That is a
genuine, reportable negative result — put it in `what_did_not_work.md`, do not tune it
away.

Also watch mean `u`: it rises in the first epochs (the annealed KL suppressing evidence)
then should fall as the model becomes confident. If `u` stays near 1.0 at the end, lower
`EDL_ANNEAL_EPOCHS`.

---

## 8. Priority order

1. **Three seeds** (42 / 7 / 2026) on MILK10k with `resnet50` — the protocol baseline.
2. **DermLIP_PanDerm linear probe**, same seeds — the expected large win.
3. **MetaBlock baseline (B9)** — the reference method a reviewer will ask for.
4. **PAD-UFES-20 run** for the distribution-shift family.
5. Optional: unfreeze BERT (tests whether text is weak because of the encoder or the
   synthetic narratives — given Model C's measured 27%, the narratives are the likelier
   cause); backbone sweep; image size 320.

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `detected columns` has `None` | print `df.columns`, rename explicitly |
| `no metadata CSV under ...` | set `DATA_DIR` deeper; the loader globs recursively |
| images all load as zeros | check `IMAGES_SUBDIR` and whether filenames need an extension |
| all predictions one class | class weights not applied, or LR too high |
| `u` pinned at 1.0 | KL anneal too aggressive — lower `EDL_ANNEAL_EPOCHS` |
| CUDA OOM | `BATCH_SIZE=16`, or `IMAGE_SIZE=192` |
| val macro-F1 flat | backbone never unfroze (expected for a linear probe) |
| gated repo 401 | accept the model terms, add `HF_TOKEN` as a Kaggle Secret |

---

## 10. After the model returns

```bash
# conformal sets + LTT threshold with a distribution-free risk guarantee,
# then the three-action policy, AUGRC and guarantee violation
python scripts/run_eade.py --alpha 0.05 --delta 0.10

# robustness across the 18 corruption conditions
python scripts/run_stress.py --limit 300

# the A-G / baseline ladder
python scripts/baseline_probe.py --seeds 42 7 2026
```

`scripts/run_eade.py` currently runs in **bridge** mode (softmax lifted into opinions via
normalised entropy) because the legacy specialists have no evidential heads. Once native
heads come back from Kaggle, point it at the new checkpoints and the `mode` becomes
`native` — the decision path is unchanged.
