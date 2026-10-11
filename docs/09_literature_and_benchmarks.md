# Verified Literature and Comparable Benchmarks

Every citation below was **checked against a primary source on 2026-10-11**. This
supersedes the "citations from memory, unverified" caveats in docs 02, 04, 05 and 06.

Two of my earlier claims were **wrong** and are corrected in §3.

---

## 1. Verified citations

### Dataset

**PAD-UFES-20** — Pacheco, A. G. C. et al., *Data in Brief*, vol. 32, 106221 (2020).
DOI [10.1016/j.dib.2020.106221](https://doi.org/10.1016/j.dib.2020.106221).
Data: Mendeley [10.17632/zr7vgbcyr2.1](https://data.mendeley.com/datasets/zr7vgbcyr2/1).
Preprint [arXiv:2007.00478](https://arxiv.org/abs/2007.00478).

Confirmed figures — all match what this project assumed:

- **1373 patients, 1641 lesions, 2298 images, 6 diagnostic classes**
- 58.4% of lesions biopsy-proven, **100% of the skin cancers**
- collected across 11 cities in Espírito Santo, Brazil, via the UFES *Programa de
  Assistência Dermatológica*
- six classes arise after **clustering Bowen's disease into SCC** — worth stating in a
  paper, since it affects the SCC class definition
- feature count is reported inconsistently across versions (21 in *Data in Brief*, 22 on
  arXiv/Mendeley, up to 26 in the Mendeley description). Our pipeline uses the Mendeley
  CSV; state which you used.

### Fusion baseline

**MetaBlock** — Pacheco, A. G. C. & Krohling, R. A., "An Attention-Based Mechanism to
Combine Images and Metadata in Deep Learning Models Applied to Skin Cancer
Classification", *IEEE Journal of Biomedical and Health Informatics*, **25(9),
3554–3563 (2021)**.

Confirmed: metadata is used to **enhance the most relevant image features throughout the
classification pipeline**; compared against **MetaNet** and feature concatenation; better
in **6 of 10 scenarios** across two skin-lesion datasets. This is the reference method on
this dataset and remains a required baseline.

### Evidential / trusted fusion

**Evidential Deep Learning** — Sensoy, M., Kaplan, L. & Kandemir, M., "Evidential Deep
Learning to Quantify Classification Uncertainty", *NeurIPS 31* (2018), pp. 3179–3189.
[arXiv:1806.01768](https://arxiv.org/abs/1806.01768).
Confirmed: Dirichlet over class probabilities, deterministic net emits evidence,
subjective-logic framing, strong OOD and adversarial robustness. The authors describe
their own loss analysis as *preliminary* — relevant, since §2.2 of the maths doc picks
the squared-error Bayes-risk form.

**Trusted Multi-View Classification (TMC)** — **Zongbo Han**, Changqing Zhang, Huazhu Fu
& Joey Tianyi Zhou, *ICLR 2021*. [arXiv:2102.02051](https://arxiv.org/abs/2102.02051).
Confirmed: per-view Dirichlet evidence combined at the **evidence level** via
Dempster–Shafer; induces accurate uncertainty, giving reliability and robustness on
out-of-distribution samples. This is the closest prior work to our C1+C2 and must be
cited as such — our contribution is the *application and the safety layer on top*, not
the operator.

### Risk control

**Learn then Test** — Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan, M. I. &
Lei, L., "Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control",
*Annals of Applied Statistics* (2025), DOI
[10.1214/24-AOAS1998](https://doi.org/10.1214/24-AOAS1998).

Confirmed, and it directly validates the choice in maths doc §5.2: LTT exists
specifically because earlier methods **required λ to be one-dimensional and the risk
monotone in λ**. Selective risk is *not* monotone in λ, so LTT is the correct instrument
and conformal risk control alone is not sufficient. Procedure confirmed: per-λ hypothesis
test, **Hoeffding–Bentkus** p-values, familywise correction, yielding
`P(R(T_λ) ≤ α) ≥ 1 − δ`.

**Conformal Risk Control** — Angelopoulos, Bates, **Fisch, Lei & Schuster**.
⚠️ **Correction:** docs 02/04 attributed this to "Angelopoulos, Bates, Candès, Jordan,
Lei" — that is the *Learn then Test* author list. Fixed.

**Distribution-free, risk-controlling prediction sets** — Bates et al. The precursor LTT
generalises; cite as the earlier work.

Reference text: Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction and
Distribution-Free Uncertainty Quantification",
[arXiv:2107.07511](https://arxiv.org/abs/2107.07511).

Implementation note: **MAPIE** ships an LTT-style risk-control module — worth using
rather than hand-rolling the p-values.

### Still unverified

`Dempster (1967)`, `Shafer (1976)`, `Jøsang` (subjective logic), `Chow (1970)`,
`Hinton (2002)` product of experts, `Zadeh et al. (2017)` TFN, `Liu et al. (2018)` LMF,
`Tsai et al. (2019)` MulT, `Arevalo et al.` GMU, `Wu & Goodman (2018)` MVAE,
`Sutter et al. (2021)` MoPoE, `Lin et al. (2017)` focal loss, `Guo et al. (2017)`
temperature scaling, `Lakshminarayanan et al. (2017)` deep ensembles, `Vovk` Mondrian
conformal. These are long-established and low-risk, but verify before submission.

---

## 2. Comparable published numbers on PAD-UFES-20

⚠️ **Not a leaderboard.** Papers differ in metric (accuracy / balanced accuracy /
macro-F1), in class count (6-class vs binary), and — critically — in **splitting
protocol**. Treat these as scattered reports.

| Method | Task | Reported | Source |
|---|---|---|---|
| DiffMIC (diffusion) | 6-class | **BACC 0.646** | arXiv:2504.00026 |
| DiffMIC | binary cancer/non-cancer | BACC 0.836 | same |
| ViT + multimodal cross-attention | unclear | BACC 0.775 | Hyper.ai listing |
| CNN + MetaBlock | unclear | BACC 0.770 | Hyper.ai listing |
| LiwTERM | unclear | BACC 0.740 | Hyper.ai listing |
| XGBoost, **metadata only** | 6-class | **acc 81.6% ± 3.2, macro-F1 0.746** | TRACE, arXiv:2411.08701 |
| nnMLP, metadata only | 6-class | acc 79.4%, macro-F1 0.721 | same |
| MetaNet + MetaBlock | PAD-UFES-20 + ISIC 2019 | BACC 81.4% | Frontiers in Surgery 2022 |

### How our honest baselines compare

| | Ours (patient-level, 3 seeds) | Published |
|---|---|---|
| metadata only | acc 0.690, macro-F1 0.648 | XGBoost acc 0.816, macro-F1 0.746 |
| image + metadata | acc 0.723, macro-F1 0.653 | CNN+MetaBlock BACC 0.770 |
| 6-class best | macro-F1 0.653 | DiffMIC BACC 0.646 |

**Our baselines sit below the published metadata results, and that is expected** —
they are deliberately weak baselines (linear probes on *frozen* ImageNet features, no
fine-tuning, no gradient boosting). Two further reasons for the gap, both worth stating:

1. **Our split is stricter.** We split disjointly by `patient_id`. Several published
   results do not state patient-level grouping, and PAD-UFES-20 has multiple lesions per
   patient (1641 lesions / 1373 patients), so an image-level split leaks a patient's
   other lesions across the boundary and inflates scores. Numbers from different
   splitting protocols are not comparable.
2. **No fine-tuning yet.** That is the Kaggle job.

**Actionable target:** the metadata-only XGBoost result (macro-F1 0.746) is the number to
beat, and it suggests our metadata pathway has ~0.10 macro-F1 of headroom from a better
model class alone. Try gradient boosting on metadata before anything exotic —
`baseline_probe.py --model gbm` already supports it.

### External validity — a strong confirmation of our limitation

The DiffMIC authors trained on PAD-UFES-20 6-class and tested on **HIBA**: average
balanced accuracy **46.32%**, which they describe as *no better than random guessing*.

This is direct published evidence for limitation #2 ("one dataset, one centre, no
cross-site evidence"). Cite it — it converts a hedge into a documented finding about the
dataset.

---

## 3. Corrections to my earlier claims

### ❌ "PanDerm weights are unpublished" — WRONG

**PanDerm is published with public weights.**

**A Multimodal Vision Foundation Model for Clinical Dermatology** — Yan, S. et al.,
*Nature Medicine* (2025). [arXiv:2410.15038](https://arxiv.org/abs/2410.15038).
Code: [github.com/SiyuanYan1/PanDerm](https://github.com/SiyuanYan1/PanDerm).
Self-supervised pretraining on **>2 million real-world skin-disease images from 11
institutions across 4 imaging modalities**.

Released checkpoints:

| Name | Release | Arch | Location |
|---|---|---|---|
| **DermLIP_PanDerm** | 07/2025 | ViT-B/16 | HF `redlessone/DermLIP_PanDerm-base-w-PubMed-256` |
| PanDerm_Base | 04/2025 | ViT-B/16 | Google Drive (`panderm_bb_data6_checkpoint-499.pth`) |
| PanDerm (paper) | 10/2024 | ViT-L/16 | Google Drive (`panderm_ll_data6_checkpoint-499.pth`) |

The HF repo ID is **exactly** the one in `proj.md` Part 2. The v4 specification was
accurate; the *implementation* simply never loaded it. The old code's own error message
("run `huggingface-cli login` and accept the model license") shows why: **the repo is
gated.**

**This changes a recommendation.** See §4.

### ❌ Conformal Risk Control author list — WRONG

Corrected in §1.

### ✅ Everything else held

Dataset counts, MetaBlock's role and venue, EDL, TMC (including that Han is first
author), LTT's procedure and its monotonicity rationale — all confirmed as written.

---

## 4. Revised recommendation: use DermLIP_PanDerm as the image encoder

Our image modality reaches only **0.477 macro-F1** with frozen ImageNet ResNet50 features.
A foundation model pretrained on 2M dermatology images is the obvious fix, and it is
available.

```python
# pip install open_clip_torch
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms(
    'hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256')
tokenizer = open_clip.get_tokenizer('hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256')
```

Confirmed details: **open_clip** (not `transformers`/`timm`), **224×224** input, text
encoder is PubMedBERT-256, and the zero-shot template is
`'This is a skin image of {c}'` — again exactly matching `proj.md`.

### Caveats, all of which matter

1. **Gated.** Accept the terms on the model page and `huggingface-cli login`. On Kaggle,
   store the token as a Secret. **This is the single most likely cause of the original
   implementation's silent fallback.**
2. **Licence is non-commercial and inconsistently labelled.** The page metadata says
   `cc-by-nc-4.0`; Model Details and the access section say **`cc-by-nc-nd-4.0`**. The
   repo says CC-BY-NC-ND 4.0 for model and code.
   **"ND" (NoDerivatives) arguably restricts fine-tuning.** Linear probing is the safer
   reading, and conveniently the authors state the model "requires fine-tuning or linear
   probing" — so **prefer linear probing / frozen-feature extraction**, and check the
   official terms before fine-tuning. For a non-commercial semester project with
   attribution this is fine either way, but the ambiguity should be resolved in writing.
3. **Overlap risk.** PanDerm was pretrained on 2M images from 11 institutions. If
   PAD-UFES-20 is in that corpus, "linear probe on PanDerm features" is partly
   train-on-test. **Check the paper's data list before claiming a clean result** — this
   is exactly the class of problem that invalidated the previous generation here.

### Expected effect

| Image encoder | macro-F1 (image only) |
|---|---|
| ImageNet ResNet50, frozen | 0.477 (measured) |
| ImageNet ResNet50, fine-tuned | 0.58–0.68 (estimated) |
| **DermLIP_PanDerm, linear probe** | **plausibly 0.65–0.75** |

If that holds, the image pathway stops being the weak link and the fusion story becomes
much stronger — the interesting case becomes *two strong modalities plus one weak one*,
which is a better test of evidence-aware fusion than *one strong and two weak*.

Cheapest decisive experiment: extract DermLIP features for all 2298 images and re-run
`baseline_probe.py`. No fine-tuning, no GPU strictly required, and it immediately tells us
whether the image pathway is encoder-limited or data-limited. **Do this first.**

---

## Sources

- [PAD-UFES-20, Data in Brief / PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7479321/) · [arXiv:2007.00478](https://arxiv.org/abs/2007.00478) · [Mendeley](https://data.mendeley.com/datasets/zr7vgbcyr2/1)
- [MetaBlock (Pacheco & Krohling, IEEE JBHI 2021)](https://hyper.ai/en/papers/an-attention-based-mechanism-to-combine)
- [Evidential Deep Learning, NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/hash/a981f2b708044d6fb4a71a1463242520-Abstract.html) · [arXiv:1806.01768](https://arxiv.org/abs/1806.01768)
- [Trusted Multi-View Classification, ICLR 2021](https://arxiv.org/abs/2102.02051) · [OpenReview](https://openreview.net/pdf/4ae336db914c13c1db09afbb3dea3d948ad4aa37.pdf)
- [Learn then Test](https://academicnewsletter.sufe.edu.cn/info/440069) · [Gentle Introduction to Conformal Prediction](https://arxiv.org/pdf/2107.07511) · [MAPIE risk control](https://mapie.readthedocs.io/en/latest/theory/risk-control/)
- [PanDerm, Nature Medicine 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12353815/) · [arXiv:2410.15038](https://arxiv.org/abs/2410.15038) · [GitHub](https://github.com/SiyuanYan1/PanDerm) · [DermLIP on HuggingFace](https://huggingface.co/redlessone/DermLIP_PanDerm-base-w-PubMed-256)
- [DiffMIC on PAD-UFES-20 + HIBA cross-dataset](https://arxiv.org/pdf/2504.00026)
- [TRACE metadata baselines](https://arxiv.org/pdf/2411.08701)
- [MetaNet+MetaBlock, Frontiers in Surgery 2022](https://www.frontiersin.org/journals/surgery/articles/10.3389/fsurg.2022.1029991/full)
