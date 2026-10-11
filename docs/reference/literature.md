# Literature

Merged survey. **Verification status is recorded per entry** — several claims that were
carried for a while turned out to be wrong, so the marks matter.

| Mark | Meaning |
|---|---|
| ✅ | existence, venue and the key claim used here checked against a primary source on 2026-10-11 |
| 📄 | has a DOI/link but **not independently checked**; open it before citing |
| 📚 | long-established; bibliographic details standard, still confirm before formal citation |
| ⚠️ | could not be located — do not cite until someone confirms |

**Evidence rule.** An abstract is not enough to claim a method is *absent* from a paper.
Before any novelty claim, read the full text, supplement and code. Gap-matrix cells
([02_research_design.md](research_design.md) §5) filled from abstracts are provisional.

**Result labelling.** Every number in a table is tagged `paper-reported`,
`our-reimplementation`, or `our-method`.

---

## 1. Base papers

| Role | Paper | Status |
|---|---|---|
| Closest task (acquire/defer in dermatology) | **HOT AI** — Mehta D., Primiero C., Betz-Stablein B., Nguyen T.D., Gal Y., Bowling A., Haskett M., Sashindranath M., Bonnington P., Mar V., Soyer H.P., Ge Z. *Multi-task AI models in dermatology.* **JEADV 39(12):2121–2133, 2025**, doi 10.1111/jdv.20551. Preprint arXiv:2311.01009 (different title) | ✅ record; ⚠️ **content unverified** |
| Risk control | **Learn then Test** — Angelopoulos, Bates, Candès, Jordan & Lei, **Ann. Appl. Stat. 2025**, doi 10.1214/24-AOAS1998, arXiv:2110.01052 | ✅ |
| Risk control | **Conformal Risk Control** — Angelopoulos, Bates, **Fisch, Lei & Schuster**, arXiv:2208.02814 | ✅ |
| Evaluation metric | **AUGRC** — Traub, Bungert, Lüth, Baumgartner, Maier-Hein, Maier-Hein & Jäger. *Overcoming Common Flaws in the Evaluation of Selective Classification Systems.* **NeurIPS 2024 (spotlight)**, arXiv:2407.01032 | ✅ |
| Evidential fusion | **TMC** — **Zongbo Han**, Changqing Zhang, Huazhu Fu & Joey Tianyi Zhou. *Trusted Multi-View Classification.* **ICLR 2021**, arXiv:2102.02051 | ✅ |
| Evidential uncertainty | **EDL** — Sensoy, Kaplan & Kandemir. **NeurIPS 2018**, pp. 3179–3189, arXiv:1806.01768 | ✅ |
| Fusion baseline | **MetaBlock** — Pacheco & Krohling. **IEEE JBHI 25(9):3554–3563, 2021** | ✅ |
| Dataset | **MILK10k** (ISIC), doi 10.34970/648456 | ✅ |
| Clinical motivation | **Müller C. et al.**, *JAAD* 2026, doi 10.1016/j.jaad.2026.03.069 | 📄 |
| Acquisition formulation | **CAMA** — Rheude T. et al., arXiv:2505.16791 | 📄 |
| Image encoder | **PanDerm** — Yan S. et al., *Nature Medicine* 2025, arXiv:2410.15038 | ✅ |

### Verified details worth knowing

**AUGRC.** Defines 5 requirements for multi-threshold selective-classification metrics;
shows existing ones — **AURC specifically** — fail them. Generalized risk is normalised
by `N`, selective risk by the number accepted. AUGRC ∈ [0, ½], lower better, read as the
*average risk of undetected failures*. **Changes metric rankings on 5 of 6 datasets.**
AURC is built on selective risk, which "is not suitable for aggregation over rejection
thresholds" because it over-weights high-confidence failures; e-AURC inherits the flaws.

**Learn then Test.** Exists specifically because earlier methods *required* `λ` to be
one-dimensional and the risk monotone in `λ`. Procedure: per-`λ` hypothesis test,
Hoeffding–Bentkus p-values, familywise correction → `P(R ≤ α) ≥ 1−δ`. An LTT-style
module ships in **MAPIE**.

**TMC.** Per-view Dirichlet evidence combined **at the evidence level** via
Dempster–Shafer; induces accurate uncertainty giving reliability and robustness on OOD
samples. This is the closest prior work to our evidence layer — our contribution is the
safety policy on top, not the operator.

**MetaBlock.** Metadata enhances the most relevant image features throughout the
pipeline; compared against MetaNet and concatenation; better in **6 of 10 scenarios**
across two datasets. Published by the PAD-UFES-20 authors — a reviewer will expect it.

**MILK10k.** 10,480 images / 5,240 lesions, each with paired clinical close-up **and**
dermatoscopic image; **CC-BY-NC**; 344.7 MB bundle; metadata includes age bins, sex, site,
**6-level skin tone**, `method of ground truth`, MONET concept probabilities (7 term
groups); benchmark of 479 lesions with hidden labels.

*Verified on the extracted release, 2026-10-11* — figures measured, not quoted:
**11 classes** (AKIEC, BCC, BEN_OTH, BKL, DF, INF, MAL_OTH, MEL, NV, SCCKA, VASC);
**71.7% of lesions malignant**, BCC alone 48.1%; **95.7%** histopathology-confirmed
(so "~96%" is right); rarest class MAL_OTH has **9** lesions; all 10,480 images
byte-distinct. The 5 centres (Vienna, Ankara, UQ, MSKCC, Skopje) are described by the
study but **not identifiable in the release** — `attribution` is one constant value, so
no centre-held-out experiment is possible. See `scripts/milk10k_audit.py`.

**PAD-UFES-20.** Pacheco et al., *Data in Brief* 32:106221, 2020. 1373 patients / 1641
lesions / 2298 images / 6 classes; 58.4% biopsy-proven, **100% of cancers**; six classes
arise after **clustering Bowen's disease into SCC** — state this, it affects the SCC
definition. Feature count is reported inconsistently across versions (21/22/26); say
which you used.

**PanDerm.** Self-supervised on **>2M skin images from 11 institutions, 4 modalities**.
Released weights: **DermLIP_PanDerm** (07/2025, ViT-B/16) at HF
`redlessone/DermLIP_PanDerm-base-w-PubMed-256`; PanDerm_Base (04/2025, ViT-B/16) and
PanDerm ViT-L/16 on Google Drive. Loaded with **open_clip**, 224×224, text encoder
PubMedBERT-256, zero-shot template `'This is a skin image of {c}'`.
**Three caveats:** the repo is **gated** (HF token required); the licence is labelled
inconsistently (`cc-by-nc-4.0` in metadata vs **CC-BY-NC-ND** in the model details — "ND"
arguably forbids fine-tuning, so prefer **linear probing**); and **check whether MILK10k
or PAD-UFES-20 is inside the pretraining corpus** before claiming a clean probe.

---

## 2. Comparable published numbers

⚠️ **Not a leaderboard.** Papers differ in metric, class count and — critically —
splitting protocol.

| Method | Task | Reported | Source |
|---|---|---|---|
| DiffMIC (diffusion) | PAD-UFES 6-class | BACC 0.646 | arXiv:2504.00026 📄 |
| DiffMIC | binary | BACC 0.836 | same |
| ViT + multimodal cross-attention | unclear | BACC 0.775 | Hyper.ai listing 📄 |
| CNN + MetaBlock | unclear | BACC 0.770 | Hyper.ai listing 📄 |
| XGBoost, **metadata only** | PAD-UFES 6-class | **acc 81.6% ± 3.2, macro-F1 0.746** | TRACE, arXiv:2411.08701 📄 |
| MetaNet + MetaBlock | PAD-UFES + ISIC 2019 | BACC 81.4% | Front. Surg. 2022 📄 |

**Our honest baselines** (patient-level, 3 seeds, frozen features): metadata-only
macro-F1 0.648, image+metadata 0.653. Below the published metadata result — expected,
because ours are deliberately weak linear probes on a **stricter** split. Several
published results do not state patient-level grouping, and PAD-UFES-20 has 1641 lesions
from 1373 patients, so an image-level split leaks a patient's other lesions.

**Actionable:** the metadata XGBoost result suggests ~0.10 macro-F1 of headroom from
model class alone. `baseline_probe.py --model gbm` already supports this — do it before
anything exotic.

**External validity.** DiffMIC trained on PAD-UFES 6-class and tested on HIBA: balanced
accuracy **46.32%**, which those authors call *no better than random guessing*. Direct
published evidence for the single-centre limitation. 📄

---

## 3. By category

### A — Dermatology triage, deferral, acquisition
HOT AI ✅/⚠️ · ViT UQ for biopsy recommendation, *Front. Bioeng. Biotechnol.* 2026,
doi 10.3389/fbioe.2026.1859844 📄 · Roy et al., *Does Your Dermatology Classifier Know
What It Doesn't Know?*, hierarchical outlier detection with a clinical cost metric 📚 ·
**DermFusion** (listed as *Sci. Rep.* 2026, doi 10.1038/s41598-026-72311-6) ⚠️ **could
not be located — confirm or delete.**

### B — Selective prediction, calibration, risk control
Geifman & El-Yaniv, NeurIPS 2017 📚 · Guo et al., ICML 2017 (temperature scaling) 📚 ·
Angelopoulos & Bates, arXiv:2107.07511 (intro) ✅ · RAPS, arXiv:2009.14193 📚 · LTT ✅ ·
CRC ✅ · AUGRC ✅ · Fayyad et al., arXiv:2312.07460 — CP vs MC-dropout vs EDL on skin
lesions, CP most robust ✅ · Lu et al., arXiv:2109.04392 — group-conditional CP by skin
type ✅ · Zoravar et al., arXiv:2505.15997 — conformal ensemble of ViTs under shift ✅ ·
CHIL 2026 calibration/selective prediction in multimodal clinical classification 📄.

### C — Learning to defer
Mozannar & Sontag, ICML 2020 📚. *Action item: search for learning-to-defer applied to
dermatology during the novelty audit — none recorded yet.*

### D — Active feature / modality acquisition
CAMA 📄 · ACO (Valancius et al., arXiv:2302.13960) 📄 · Shim, Hwang & Yang, NeurIPS 2018
📄 · RL for active modality selection, MICCAI 2022 📄 · SAGEAgent, arXiv:2607.09521 📄 ·
SDM-Q, arXiv:2605.31014 📄.

### E — Multimodal dermatology fusion (backbone comparators)
MetaBlock ✅ · DermFormer, *Pattern Anal. Appl.* 2025, doi 10.1007/s10044-025-01572-0 📄 ·
Zuo et al. / CosCatNet, *Artif. Intell. Med.* 2025 📄 · **JI-ADF**, arXiv:2604.27343 —
fusion on MILK10k with the official hidden test, so a reference on **our exact dataset**
📄 · SkinM2Former, WACV 2025 📄 · MHDF-Net 📄 · UAGF, *ESWA* 332:133657, 2026 — note this
is **single-modality multi-backbone**, not multimodal 📄 · QA-MoE, IJCAI 2026,
doi 10.24963/ijcai.2026/766 — note it is evaluated on **ADNI and MIMIC-IV, not
dermatology**, and there is a same-name ACL 2026 paper on sentiment analysis; always cite
by DOI 📄 · Kawahara et al., *IEEE JBHI* 2019 (Derm7pt origin) 📚.

### F — Uncertainty, OOD, shift
SAGE, *npj Digit. Med.* 2026 📄 (name clash with SAGEAgent) · eBioMedicine 2025
generalisation gap 📄 · Energy OOD (Liu et al. 2020) and Mahalanobis (Lee et al. 2018)
📚 · Deep ensembles (Lakshminarayanan et al. 2017) 📚 · UQ review, *Med. Image Anal.*
2024 📄.

### G — Robustness, fairness, data quality, label uncertainty
Robustness scoping review, *npj Digit. Med.* 2025 📄 · Frontier-model robustness,
*Nature Medicine* 2026 📄 · DermaMNIST/Fitzpatrick17k quality, *Sci. Data* 2025 📄 ·
**DDI**, *Sci. Adv.* 2022, doi 10.1126/sciadv.abq6147 📄 · Uncertain ground truth,
*Med. Image Anal.* 2025 📄 · Vickers & Elkin, *Med. Decis. Making* 2006 (decision curve
analysis) 📚.

### I — Language models (context only, not baselines)
PanDerm ✅ · DermEVAL, WACV 2026 📄 · Clinical freetext + dermoscopy, *Commun. Med.* 2026
📄 · Synthesized clinical notes, *npj Digit. Med.* 2026 📄 · SkinGPT-4, *Nat. Commun.*
2024 📄 · Esteva et al., *Nature* 2017 📚.

---

## 4. Corrections to earlier claims in this repository

| Claim | Status |
|---|---|
| "PanDerm weights are unpublished" | ❌ **WRONG.** Published, *Nature Medicine* 2025, public weights at the exact HF repo `proj.md` named. The repo is **gated**, which is why the original code silently fell back to ImageNet |
| Conformal Risk Control authors | ❌ **WRONG.** CRC is Angelopoulos, Bates, **Fisch, Lei & Schuster**; the list given was Learn-then-Test's |
| AURC as the primary selective metric | ❌ **WRONG.** Traub et al. show AURC fails their requirements; AUGRC supersedes it |
| "no web access available" | ❌ **WRONG.** WebSearch/WebFetch were available as deferred tools throughout |
| PAD-UFES-20 counts, MetaBlock venue, EDL, TMC (Han first author), LTT procedure | ✅ all confirmed as written |

---

## 5. Reading order

**Round 1 (weeks 1–2, blocking):** HOT AI (journal + preprint) · Learn then Test ·
Conformal Risk Control · AUGRC · TMC · EDL · MetaBlock · MILK10k terms · Müller et al. ·
Mozannar & Sontag · JI-ADF · **confirm or delete DermFusion** ⚠️

**Round 2:** DermFormer · CosCatNet · Roy et al. · SAGE · Zoravar · CAMA · ACO ·
QA-MoE · biopsy-recommendation UQ · CHIL 2026 · robustness review · eBioMedicine · DDI.

**Round 3:** MHDF-Net · SkinM2Former · UAGF · PanDerm data list (**overlap check**) ·
DermEVAL · synthesized notes · SAGEAgent · SDM-Q.

**Novelty-audit searches** (record every hit in `docs/related_work_audit.md`):
*dermoscopy acquisition* · *selective prediction dermatology* · *conformal deferral
dermatology* · *learning to defer dermatology* · *active modality acquisition skin* ·
*risk control medical imaging triage* · plus **all forward citations of HOT AI**.

### Extraction template → `docs/extractions/<slug>.md`

```
Citation / DOI / venue / date:
Verification: ✅ 📄 ⚠️   — checked by ____ on ____
Task, labels, dataset(s), size, modalities:
Split and leakage controls:
Backbone / fusion:
Uncertainty / calibration / OOD method:
Action space (predict / abstain / defer / acquire):
Explicit costs? (Y/N, how):
Statistical guarantee? (Y/N, which):
Stress / shift protocol:    Subgroup analysis:
Key reported numbers (paper-reported):
Code / weights / licence:
Author-stated limitations:   Our inferred limitations (marked as inference):
Gap-matrix cells to update:   Experiment it informs:
Can we reproduce it fairly? (Y/N, why):
```
