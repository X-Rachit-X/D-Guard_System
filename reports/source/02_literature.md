# Literature Survey

This chapter presents the two base papers of the project, then the wider literature it draws on, and ends with a gap matrix that shows where DERMA-GUARD sits. The selection follows the project's literature dossier, which concluded that the contribution of this work is a decision policy on top of a model, not a new fusion architecture.

## Base Papers

From the literature reviewed, two papers are the bases of this project: one for the task and one for the method.

Table: The two base papers.
| Role | Base paper | What it gives the project |
|---|---|---|
| Closest task | HOT AI, Mehta et al., JEADV 2025 [@mehta] | An AI that recommends a dermoscopic image when the clinical photograph is insufficient |
| Core method | Learn-then-Test, Angelopoulos et al. [@ltt] | Decision thresholds with a finite-sample guarantee on a chosen risk |

### Base paper 1: HOT AI

Mehta et al. [@mehta] describe a multi-task model that, from a clinical image, produces a hierarchical diagnosis, an out-of-distribution alert and a recommendation to take a dermoscopic image when the clinical image alone is not enough; when the recommendation is followed, the two images are fused for the final diagnosis. It is the paper closest to ours in task, and it rules out "an AI that asks for dermoscopy" as a new idea.

DERMA-GUARD differs in four ways that can be measured. It puts explicit costs on acquiring dermoscopy and on deferring to a clinician; it has a separate defer action for contradictory evidence; it certifies the rate of cancers called benign among its confident answers; and it is evaluated on a public paired dataset with combined-failure stress tests. Because HOT AI's model and in-house data are not public, we compare against a faithful HOT-style rule: request dermoscopy for the most uncertain patients, at the same rate as our policy, then answer.

### Base paper 2: Learn-then-Test

Angelopoulos et al. [@ltt] show how to choose thresholds on a held-out calibration set so that a user-chosen risk is controlled with a finite-sample, distribution-free guarantee, valid when calibration and future cases are exchangeable. We do not change the method. We apply it to a clinically meaningful risk (the share of answered cancers called benign) inside a three-action policy, and we measure what happens to the guarantee on two other datasets where exchangeability no longer holds.

## Related Work

### Data and clinical motivation

MILK10k [@milk] is the first large public dataset in which every lesion has both a clinical close-up and a dermoscopic image, which makes it possible to measure the real value of asking for dermoscopy instead of simulating it. Müller et al. [@muller] used it in a reader study with 283 participants and found that dermoscopy had higher sensitivity for malignancy while the clinical close-up had higher specificity, and that each added independent value. That is the clinical reason to ask for dermoscopy selectively rather than always or never. JI-ADF [@jiadf] reports multimodal fusion results on MILK10k's official hidden test, which we did not use, so its numbers are not directly comparable with ours.

### Skin-lesion classification and multimodal fusion

Esteva et al. [@esteva] showed that a deep network can match dermatologists on biopsy-proven images; residual networks [@resnet] remain a common backbone, and dermatology foundation models such as PanDerm [@panderm] are the current frontier. Clinicians combine the image with the patient's age, the lesion's location and history: PAD-UFES-20 [@pad] provides smartphone photographs with rich metadata, MetaBlock [@metablock] lets metadata modulate image features, and Derm7pt [@derm7pt] pairs clinical and dermoscopic images for about one thousand lesions. These works improve prediction; none decides whether a prediction should be issued at all. A recurring weakness is evaluation: splitting by image rather than by lesion or patient inflates accuracy.

### Uncertainty, evidential learning and calibration

A softmax classifier cannot say "no class is supported". Evidential deep learning [@edl] places a Dirichlet over class probabilities and yields an explicit ignorance mass, grounded in the Dempster–Shafer theory of evidence [@shafer]. Trusted multi-view classification [@tmc] combines per-view opinions with Dempster's rule so that uninformative views fall silent. Temperature scaling [@guo] and isotonic regression [@zadrozny] calibrate confidences, and Mahalanobis distances [@lee] detect unusual inputs. The EM algorithm [@em] underlies the iterative fusion of the original design.

### Conformal prediction, abstention and deferral

Least-ambiguous conformal sets [@lac] give label sets with distribution-free coverage; Fayyad et al. [@fayyad] found conformal prediction the most robust uncertainty method for skin lesions, and Lu et al. [@lu] made it group-conditional by skin type. In DERMA-GUARD the conformal set drives an action (no confident benign answer while "malignant" remains in the set) rather than only quantifying uncertainty. Chow's rule [@chow] rejects when answering costs more than rejecting, selective classification [@geifman] applies abstention to deep networks, and learning to defer [@l2d] trains a rule that hands cases to an expert; a learned rejector in that spirit is one of our baselines. For evaluation we use AUGRC [@traub], which corrects flaws in the usual risk–coverage metric, and for acquisition we follow CAMA [@cama] in bracketing a learned acquisition rule between random and oracle acquisition at the same rate.

## Gap Matrix and Comparison

The gap matrix asks which parts of the problem each related work covers. The DERMA-GUARD row is filled in from what this project actually measured.

Table: Gap matrix (yes = present, no = absent, ? = not determinable from the abstract).
| Work | Paired clinical + dermoscopy | Defer / abstain | Acquire a modality | Explicit costs | Statistical risk guarantee | Stress tests | Skin-tone subgroups |
|---|---|---|---|---|---|---|---|
| HOT AI [@mehta] (base) | yes | OOD alert | yes | ? | ? | ? | ? |
| Learn-then-Test [@ltt] (base) | n/a | via thresholds | no | no | yes | no | no |
| Fayyad et al. [@fayyad] | no | set-based | no | no | coverage | ? | ? |
| Lu et al. [@lu] | no | set-based | no | no | group coverage | no | yes |
| JI-ADF [@jiadf] | yes (MILK10k) | ? | no | no | no | ? | ? |
| CAMA [@cama] | n/a | no | yes | budget | no | no | no |
| Learning to defer [@l2d] | n/a | yes | no | defer cost | no | no | no |
| **DERMA-GUARD (measured)** | **yes** | **LOW tier + DEFER** | **dermoscopy, details, retake** | **5:2 miss/false alarm; 0.5 acquire; 1.5 review** | **Learn-then-Test, certified** | **6 damage types + 2 external datasets** | **reported; gap found** |

Every individual column is covered by someone; the niche is the row. The table below shows what was measured against the base papers and the other methods we compare with.

Table: Base papers and comparison methods, with what was measured in this project (MILK10k test, missed-cancer cost 5).
| Work | Comparison in this project | Measured result |
|---|---|---|
| HOT AI (base) | HOT-style rule: dermoscopy for the most uncertain patients at the same rate, then answer | That rule chose patients better than our learned estimator (acquisition efficiency +0.26 vs +0.07), but answering everyone afterwards missed 5.6% of cancers; DERMA-GUARD's confident answers missed 3.1% |
| Learn-then-Test (base) | Applied to the miss rate of the HIGH tier | Certified at both stages (lambda-hat 0.218 and 0.276); the target held on PAD-UFES-20 (0.0%) and Derm7pt (3.8%) |
| Learning to defer | L2D-style learned rejector at the same coverage | Missed 4.9% vs 3.1% (p = 0.011) |
| Selective classification | Confidence threshold at the same coverage | 3.8% vs 3.1% (not significant); 3.5% to 9.3% across four fusions |
| AUGRC | Ranking of errors by ignorance vs softmax | 0.0820 vs 0.0757: softmax better on the final model |
| MILK10k + reader study | Per-modality specialists | Dermoscopy head AUROC 0.854 vs photograph 0.787; fused 0.885 |

## Research Gap

The base papers establish the task (HOT AI) and the guarantee (Learn-then-Test); the related work supplies sound evaluation, a formulation of acquisition, paired data and clinical evidence that both modalities matter. The gap this project addresses is their **integration into one calibrated decision** that weighs several specialists per case with learned trust; chooses between answering, asking for evidence, answering provisionally and deferring; certifies the error that matters most; is compared with simple baselines on the same probabilities; is stress-tested under combined failures and on other populations; and is wrapped in a usable clinical workflow.
