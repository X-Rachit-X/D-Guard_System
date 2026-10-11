# What You Need To Do

Everything that needs a human or a GPU, in order, with the exact commands. Updated
2026-10-11.

**Short version:** one GPU training run is the only thing blocking the headline result.
Everything else on this page is optional or is a decision only you can make.

---

## 0. Confirm the system runs here (5 minutes)

Do this first, so that if anything later fails you know it was not already broken.

```bash
pip install -e ".[dev]"
# Windows/Anaconda: set KMP_DUPLICATE_LIB_OK=TRUE first, or torch and MKL both
# load OpenMP and the process aborts with no useful message.

pytest tests -q                 # expect: 442 passed, 3 skipped
python scripts/demo.py          # expect: ACCEPT, BCC, confidence ~0.945
```

If `pytest` passes and `demo.py` prints a diagnosis, the engineering is sound and
anything that breaks after this came from a change you made.

---

## 1. THE BLOCKER — train on MILK10k (one GPU session, ~1 hour per seed)

This is the only item that gates the project's main claim. Nothing else on this page
substitutes for it.

**Why it cannot be done here.** Local torch is CPU-only. The image pathway is the
bottleneck: a frozen ImageNet encoder reaches **0.477** macro-F1 while metadata alone
reaches **0.648**, so the images are currently contributing less than a table of
checkboxes. Training the encoder and the evidential heads is what fixes that.

### Steps

1. Upload `./milk10k` (the extracted directory, not the zip) as a **private** Kaggle
   dataset. It is CC-BY-NC — private only, never public.
2. Upload this repo's `notebooks/kaggle_train_dguard.py` as a second dataset, or paste it
   into a cell.
3. New Notebook → **GPU T4 ×2** or P100 → **Internet: On**.
4. Run:

```python
!pip -q install transformers open_clip_torch

import sys; sys.path.append('/kaggle/input/<your-code-dataset>')
import kaggle_train_dguard as T

T.CFG.update(
    DATASET="milk10k",
    DATA_DIR="/kaggle/input/<your-milk10k-slug>",
    OUT_DIR="/kaggle/working/dguard_milk_s42",
    IMAGE_BACKBONE="resnet50",
    EPOCHS=25, BATCH_SIZE=32, SEED=42,
)
T.run(T.CFG)
```

5. Repeat with `SEED=7` and `SEED=2026` into **separate** `OUT_DIR`s. Three seeds is a
   protocol requirement, not a nicety — single-seed deep-learning numbers are noise.

### Check these in the first minute, then walk away

```
[milk10k] 5240 lesions, 11 classes, 71.7% malignant
class counts: {'BCC': 2522, ..., 'MAL_OTH': 9}
  train n= 3668 lesions= 3668
metadata dim = 26 (MILK10k schema, fitted on train only)
```

Those are fixed properties of the release. **If `lesions` is not 5240 or the class counts
differ, stop** — the upload is incomplete, and letting it train wastes the session.

### What to send back

The whole `OUT_DIR`. The file that matters most is **`predictions_test.csv`** — it carries
`y_true`, `y_pred`, `ignorance_u`, `conflict_C` and the 11 class probabilities, which is
everything the decision engine needs. With it I can run the conformal calibration, the
LTT threshold, the three-action policy and every metric without a GPU.

### Two things that will look like failures and are not

- **Macro-F1 lower than the PAD-UFES numbers.** MILK10k has 11 classes with a 48% majority
  class. Quote **balanced accuracy** and **macro-F1**, never accuracy.
- **`clin+derm+meta` not beating the best pair.** That is hypothesis **H1** genuinely
  failing, which is a real reportable result. Do not tune it away — it goes in
  `what_did_not_work.md`.

---

## 2. Decide two things only you can decide

### 2a. Should the code be public?

Right now the GitHub repo holds the full codebase. You originally said **documentation
only for now**, and a code commit reached the remote when a later docs push carried it
along (a plain `git push` publishes the whole branch, not just the staged files). Nothing
sensitive went out — no images, weights, CSVs or `.env`, verified with
`git ls-tree -r --name-only origin/main`.

**Your call:** leave it, or I strip that commit from the remote. Either is a one-line fix.

### 2b. Is the cost matrix right?

```yaml
missed_malignant: 10      # cost of calling a cancer benign
false_alarm: 2            # cost of calling a benign lesion cancer
acquire_dermoscopy: 0.5
defer_to_human: 1.5
```

These are **declared, not derived**, and they drive every PROCEED/ACQUIRE/DEFER decision.
They were also chosen while assuming a screening population — and MILK10k is **71.7%
malignant**, a referral population. A reviewer will ask where the numbers came from.
Either justify them from a clinical source or accept that the mandatory sensitivity sweep
is what carries the argument. See [limitations.md](limitations.md).

---

## 3. Optional, in value order

| # | Task | Why | Effort |
|---|---|---|---|
| 1 | **DermLIP_PanDerm linear probe** | Dermatology foundation model; the single largest expected accuracy win. Needs an HF token (gated repo) and `WARMUP_FROZEN_EPOCHS >= EPOCHS` — its licence is CC-BY-NC-**ND**, and "ND" arguably forbids fine-tuning | 1 GPU session |
| 2 | **MetaBlock baseline (B9)** | Published by the PAD-UFES-20 authors. A reviewer will expect this comparison | 1 GPU session |
| 3 | **PAD-UFES-20 run** | The distribution-shift arm. Compare on **binary malignancy only** — 11 and 6 classes do not nest | 1 GPU session |
| 4 | **Start an LLM for narrative reports** | `ollama serve && ollama pull qwen2.5:7b`. Without it the report falls back to a deterministic evidence summary, which is correct but plain | 10 min |
| 5 | **Unfreeze BERT** | Tests whether text is weak because of the encoder or the synthetic narratives. Model C scores 27% on its own training data, so the narratives are the likelier cause | 1 GPU session |

---

## 4. What is already done — do not redo it

| | |
|---|---|
| MILK10k downloaded, audited, loader written and tested | ✅ `scripts/milk10k_audit.py` |
| Evidential + Dempster–Shafer fusion | ✅ 34 tests |
| Conformal prediction (APS/RAPS, class-conditional) | ✅ coverage 0.8944 vs 0.90 target |
| Learn-then-Test risk control | ✅ validity simulation passes |
| Three-action policy + cost model | ✅ 31 tests |
| AUGRC / acquisition efficiency / calibration metrics | ✅ 32 tests |
| **RAG similar-case retrieval, built and wired in** | ✅ 1,403 training cases indexed, leak-guarded |
| 7-agent pipeline (specialists, scribe, librarian, devil's advocate, orchestrator) | ✅ runs end to end |
| DERMA-STRESS, 18 conditions | ✅ 35 tests |
| FastAPI + 2 Streamlit apps | ✅ |
| Reproducible features | ✅ verifies at cosine 1.000000 |
| Honest baselines, 3 seeds | ✅ `docs/results/baseline_probe.json` |

Full detail: [08_implementation_status.md](08_implementation_status.md).

---

## 5. If you only have 20 minutes

```bash
pytest tests -q                              # the system is intact
python scripts/milk10k_audit.py              # what the dataset actually contains
python scripts/build_rag_index.py --verify   # retrieval works and does not leak
python scripts/run_eade.py                   # the decision engine, end to end
python scripts/demo.py                       # one case, start to finish
```

Then read [evaluation_integrity.md](evaluation_integrity.md) — it is the single most
important document here, because it explains why no number produced before 2026-10-11 can
be cited.

---

## 6. Honest status

**Works and is verified:** every piece of machinery, on real data, with tests.

**Not established:** that any of it generalises. The verified end-to-end numbers run over
the legacy specialists, which were trained on data overlapping the test split. They
demonstrate that the mechanism behaves as designed; they are **not** accuracy estimates.

That gap closes with step 1 and nothing else.
