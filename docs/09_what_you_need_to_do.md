# What You Need To Do

Status of every part of the project — trained or not, done or not — and the exact steps
left. Updated 2026-10-11, after the first MILK10k training run came back.

---

## 1. Is the system trained?

| Component | Trained? | Notes |
|---|---|---|
| **MILK10k evidential model** (clinical + dermoscopy + metadata heads, DS fusion) | ✅ **Yes** — Kaggle, seed 42, 25 epochs | `runs/milk10k_resnet50_seed42/`. Runs locally and matches Kaggle (macro-F1 0.3758 vs 0.3794) |
| Calibration (conformal sets, LTT thresholds, conflict thresholds) | ✅ Yes — fitted on 786 validation lesions | `outputs/eade_milk/cost_5/calibration.joblib` |
| Acquisition-value estimator (when is dermoscopy worth it?) | ✅ Yes — fitted on validation | AE +0.27 vs +0.01 for entropy |
| Legacy A–G specialists (PAD-UFES-20) | ✅ Trained long ago | Metrics invalid ([evaluation_integrity.md](evaluation_integrity.md)); used only by the legacy pipeline and bridge mode |
| RAG similar-case index | ✅ Built (PAD-UFES-20 features) | Rebuild on MILK10k is optional |
| **Seeds 7 and 2026** | ❌ **Not yet** | Required before any number is called final |
| **DermLIP / PanDerm image encoder** | ❌ **Not yet** | Expected to be the biggest accuracy gain |
| PAD-UFES-20 run of the new trainer (shift test) | ❌ Not yet | Optional for the panel; needed for the paper's shift section |

**So: the system works end to end on a real trained model today.** What is missing makes the
numbers stronger and final; it does not block the demo.

---

## 2. What is done

| | |
|---|---|
| MILK10k downloaded, audited, loader + 38 tests | ✅ |
| Evidential fusion, conformal, LTT, three-action policy, metrics | ✅ |
| Native inference of the trained model (verified against Kaggle) | ✅ `scripts/milk_native_infer.py` |
| Two-stage decision experiment with 6 baselines + cost sweep | ✅ `scripts/run_eade_milk.py` |
| Tentative opinion on DEFER/ACQUIRE + patient and doctor wording | ✅ `src/decision/explain.py` |
| Care platform: patients, doctor queue, instructions, questions, follow-ups, change check, audit | ✅ `src/care/` |
| Web app (patient / doctor / system) | ✅ `streamlit run app/dguard_app.py` |
| Terminal demo (panel backup) | ✅ `python scripts/demo_care.py` |
| RAG, 7-agent pipeline, DERMA-STRESS | ✅ |
| Docs, study plan, panel guide, results, IEEE paper draft | ✅ |
| Cleanup (12.7 GB freed, structure documented) | ✅ |
| 473 tests (470 pass, 3 skip without optional data) | ✅ |

---

## 3. What you need to do — in priority order

### ① Run seeds 7 and 2026 on Kaggle  — *needed for the paper* (~1 hour each)

Same notebook, same dataset as your seed-42 run. The trainer now has the conflict fix
built in, so the exported `conflict_C` will be correct this time.

```python
!pip -q install transformers

import sys; sys.path.append('/kaggle/input/<your-code-dataset>')
import kaggle_train_dguard as T

for seed in (7, 2026):
    T.CFG.update(
        DATASET="milk10k",
        DATA_DIR="/kaggle/input/<your-milk10k-slug>",
        OUT_DIR=f"/kaggle/working/dguard_milk_s{seed}",
        IMAGE_BACKBONE="resnet50",
        EPOCHS=25, BATCH_SIZE=32, SEED=seed,
    )
    T.run(T.CFG)
```

**Upload the updated `notebooks/kaggle_train_dguard.py` first** — the one on Kaggle is the
old copy with the conflict bug.

**Send back:** both `OUT_DIR` folders (zip them). I will put them in `runs/` and report
mean ± spread across the three seeds.

> Note: changing `SEED` also changes the train/val/test split, by design. That is what makes
> three seeds a real robustness check rather than three runs on the same test set.

### ② DermLIP encoder — *the biggest expected improvement* (~1–2 hours)

1. Accept the terms at https://huggingface.co/redlessone/DermLIP_PanDerm-base-w-PubMed-256
2. Kaggle → Add-ons → Secrets → add `HF_TOKEN`
3. Run:

```python
!pip -q install transformers open_clip_torch
from kaggle_secrets import UserSecretsClient
import os; os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

T.CFG.update(
    DATASET="milk10k", DATA_DIR="/kaggle/input/<your-milk10k-slug>",
    OUT_DIR="/kaggle/working/dguard_milk_dermlip_s42",
    IMAGE_BACKBONE="dermlip_panderm",
    WARMUP_FROZEN_EPOCHS=25,   # = EPOCHS -> linear probe (the licence says No-Derivatives)
    EPOCHS=25, BATCH_SIZE=32, SEED=42,
)
T.run(T.CFG)
```

**Send back:** the `OUT_DIR`. If it beats ResNet50, every number improves, and the PROCEED
rate rises because the model becomes confident on more cases.

### ③ Decide the cost of a missed cancer  — *your call, 10 minutes*

| missed-cancer cost | answers alone | missed cancers |
|---|---|---|
| 5 (demo uses this) | 37% | 1.7% |
| 10 (pre-registered) | 32% | 0.0% — but every answer is a cancer diagnosis |

Pick one for the paper's headline and keep the other as the sensitivity result. Then set
`frozen: true` in `docs/prereg.yaml`. Details: [14_results.md](14_results.md) §5.2.

### ④ Practise the panel demo — *before presenting* (1 hour)

Follow [13_panel_guide.md](13_panel_guide.md) §6 twice: once in the app, once with the
terminal backup.

### ⑤ Fill in the paper's author block and compile it

Open `paper/main.tex` in Overleaf, put in names and affiliations, compile. Update numbers
after ① and ② come back.

### ⑥ Optional

| Task | Why |
|---|---|
| PAD-UFES-20 run with the new trainer (`DATASET="pad_ufes"`) | the distribution-shift section of the paper |
| `ollama serve` + `ollama pull qwen2.5:7b` | narrative reports instead of the deterministic summary |
| Delete `milk10k.zip` (345 MB) and `Old_system/` | disk space; neither is used by code |
| Rebuild the RAG index on MILK10k | retrieval currently uses PAD-UFES cases |

---

## 4. What I need from you

| Item | Blocks |
|---|---|
| `OUT_DIR` for seeds 7 and 2026 | final numbers with error bars |
| `OUT_DIR` for the DermLIP run | the stronger model |
| Your choice of missed-cancer cost (5 or 10) | freezing the pre-registration |
| Author names / affiliations / guide name | the paper title block |
| Panel date and time limit | trimming the slide plan |

Nothing else is blocked on you.

---

## 5. Quick check that everything works

```bash
pytest tests -q                                   # 470 passed, 3 skipped
python scripts/demo_care.py                        # PROCEED, ACQUIRE->PROCEED, DEFER
streamlit run app/dguard_app.py                    # the app
```
