# Training (Kaggle GPU)

Training is the only step that needs a GPU. **One run is already done** (MILK10k,
ResNet50, seed 42) and is what the system uses. Further runs are optional: they make the
paper's numbers final (seeds) and the model stronger (DermLIP).

---

## 1. What is already trained

| | |
|---|---|
| Run | `runs/milk10k_resnet50_seed42/` |
| Data | MILK10k, 11 classes, lesion-level split 3,667 / 786 / 787 |
| Model | ResNet50 (ImageNet) shared by both image types → 256-d → separate evidential head per input (clinical, dermoscopy, metadata) → Dempster–Shafer fusion |
| Training | 25 epochs, backbone frozen for 3, modality dropout 0.25, evidential loss with annealed KL, class weights; best epoch 23 by validation macro-F1 |
| Test | macro-F1 0.379, accuracy 0.693, macro AUROC 0.824 |

The weights (`checkpoints/best.pt`) are kept out of git; everything else in the run folder
is committed.

---

## 2. How to run on Kaggle (paste method)

You paste the whole script into one notebook cell. That works — but two things follow:

- It is **not a module**, so `import kaggle_train_dguard` fails (`ModuleNotFoundError`).
  You don't need to import it: after the cell runs, `run`, `run_seeds`, `run_dermlip`
  and `CFG` already exist in the notebook.
- Inside the script, `T` means `torchvision.transforms`, so `T.CFG` fails
  (`AttributeError`). Never write `T.` in your own cells.

### Cell 1 — the script, without auto-training

1. Paste the **current** `notebooks/kaggle_train_dguard.py` (download it from the repo —
   the old copy has the conflict bug and no helpers).
2. Find the line `AUTORUN = True` (just below the `CFG = dict(...)` block) and change it to
   `AUTORUN = False`. Otherwise every run of this cell trains seed 42 again for an hour.
3. Check `DATA_DIR` in `CFG` is your dataset path (yours:
   `/kaggle/input/datasets/minchou01/milk10k`).
4. Run the cell. It finishes in seconds — it only defines things.

### Cell 2 — seeds 7 and 2026 (~1 hour each)

```python
results = run_seeds((7, 2026), data_dir="/kaggle/input/datasets/minchou01/milk10k")
```

Writes `/kaggle/working/dguard_milk_s7/` and `/kaggle/working/dguard_milk_s2026/`.
Each seed also draws a different split, so three seeds measure robustness, not three
looks at one test set.

### Cell 3 — DermLIP encoder (~1–2 hours)

Before running: accept the terms on **both** model pages (the loader falls back from the
first to the second), then Kaggle → Add-ons → Secrets → add `HF_TOKEN`, and switch the
secret on for this notebook.

- https://huggingface.co/redlessone/DermLIP_PanDerm-base-w-PubMed-256
- https://huggingface.co/redlessone/DermLIP_ViT-B-16

```python
!pip -q install open_clip_torch
from kaggle_secrets import UserSecretsClient
import os
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

res = run_dermlip(seed=42, data_dir="/kaggle/input/datasets/minchou01/milk10k")
```

The first lines of output must say `[backbone] DermLIP vision tower loaded from ...`. If
neither model can be loaded the run **stops with the reason** — it never silently falls
back to ImageNet. Images are normalised with CLIP statistics automatically.

### Cell 4 — zip the results for download

```python
!cd /kaggle/working && zip -qr dguard_runs.zip dguard_milk_s7 dguard_milk_s2026 dguard_milk_dermlip_s42
```

Or from your PC:
`kaggle kernels output <your-user>/<notebook-slug> -p <folder>`.

## 3. Check the first minute, then walk away

```
[milk10k] 5240 lesions, 11 classes, 71.7% malignant
class counts: {'BCC': 2522, 'NV': 746, ..., 'MAL_OTH': 9}
  train n= 3668 lesions= 3668
metadata dim = 26 (MILK10k schema, fitted on train only)
```

If `lesions` is not 5240, the upload is incomplete — stop.

Smoke-test the exact file on a CPU first if you changed anything:

```bash
DGUARD_DATA_DIR=milk10k DGUARD_OUT_DIR=outputs/smoke DGUARD_EPOCHS=1 \
DGUARD_LIMIT_LESIONS=66 DGUARD_BATCH_SIZE=8 DGUARD_AMP=0 DGUARD_NUM_WORKERS=0 \
DGUARD_WARMUP_FROZEN_EPOCHS=1 python notebooks/kaggle_train_dguard.py
```

---

## 4. What to send back

The whole `OUT_DIR` (zip it). Contents: `checkpoints/best.pt`, `metrics.json`,
`history.json`, `manifest.json`, `predictions_test.csv`.

---

## 5. After a run comes back

```bash
# 1. put it at runs/<name>/ then
python scripts/milk_native_infer.py --run runs/<name> --out outputs/milk_native_<name>
python scripts/run_eade_milk.py --native outputs/milk_native_<name> --missed-cost 5 \
       --out outputs/eade_milk/<name>_cost_5
```

To make the app use the new model, point `DEFAULT_RUN` and `DEFAULT_BUNDLE` in
`src/native/engine.py` at the new run and bundle, then rebuild the retrieval index
(`python scripts/build_milk_index.py`).

---

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'kaggle_train_dguard'` | the script was pasted into a cell, not uploaded as a module — don't import it; call `run_seeds(...)` / `run_dermlip(...)` directly (§2) |
| `module 'torchvision.transforms' has no attribute 'CFG'` | `T` is `torchvision.transforms` inside the script — never write `T.CFG`; use the helpers |
| A cell retrains seed 42 for an hour | `AUTORUN` is still `True` in the pasted script |
| `lesions` ≠ 5240 | incomplete dataset upload |
| DermLIP falls back to ImageNet | HF token missing or terms not accepted |
| CUDA out of memory | `BATCH_SIZE=16` |
| Mean `u` stays near 1.0 at the end | lower `EDL_ANNEAL_EPOCHS` |
| Local result differs from Kaggle by > 0.02 macro-F1 | the split or metadata encoder differs — `milk_native_infer.py` refuses to run on a mismatched split |
