# Datasets and Data Audit

A medical AI system is only as trustworthy as the data it is trained on and the way it is evaluated. This chapter describes the three datasets, how they were split, and an audit of our earlier prototype that changed the direction of the project.

## Choosing the Main Dataset

The central idea of DERMA-GUARD is that the system should be able to *ask for a dermoscopic image* when the ordinary photograph is not enough. To measure whether that request is worth making, every lesion in the dataset must have both images: the system can then be shown only the photograph first, and the real dermoscopic image of the same lesion later, exactly as would happen in a clinic. Most public datasets fail this requirement. HAM10000 [31] and the ISIC challenge sets [5] contain dermoscopy only. PAD-UFES-20 [24] contains smartphone photographs only. Derm7pt [15] contains both, but only for about one thousand lesions and five diagnostic groups.

**MILK10k** [22], released in 2025 by the MILK study team through the ISIC Archive, is the only large public dataset in which every lesion has one clinical close-up and one dermoscopic image. It was therefore chosen as the main dataset; the other two are used only for external testing.

## MILK10k

MILK10k contains 5,240 lesions and 10,480 images. Diagnoses are grouped into eleven classes, 95.7% are confirmed by histopathology, and 71.7% of lesions are malignant, which reflects a referral population rather than a screening one. Each lesion also has the patient's approximate age, sex, the anatomical site (both a general and a specific location), a Fitzpatrick-like skin-tone class from 0 to 5, and seven MONET concept scores, which are automatically computed descriptors such as ulceration, hair or vessels. Five classes are counted as malignant: AKIEC, BCC, MAL_OTH, MEL and SCCKA.

Table: MILK10k classes and their distribution across the lesion-level split.
| Code | Diagnosis | Malignant | Train | Validation | Test | Total |
|---|---|---|---|---|---|---|
| AKIEC | Actinic keratosis / intra-epidermal carcinoma | yes | 214 | 32 | 57 | 303 |
| BCC | Basal cell carcinoma | yes | 1,740 | 398 | 384 | 2,522 |
| BEN_OTH | Benign, other | no | 36 | 4 | 4 | 44 |
| BKL | Benign keratosis-like lesion | no | 384 | 70 | 90 | 544 |
| DF | Dermatofibroma | no | 36 | 7 | 9 | 52 |
| INF | Inflammatory / infectious | no | 37 | 7 | 6 | 50 |
| MAL_OTH | Malignant, other | yes | 8 | 0 | 1 | 9 |
| MEL | Melanoma | yes | 322 | 60 | 68 | 450 |
| NV | Melanocytic nevus | no | 522 | 133 | 91 | 746 |
| SCCKA | Squamous cell carcinoma / keratoacanthoma | yes | 334 | 70 | 69 | 473 |
| VASC | Vascular lesion | no | 34 | 5 | 8 | 47 |
| **Total** | | | **3,667** | **786** | **787** | **5,240** |

Two properties of this table shape everything that follows. The data are **heavily imbalanced**: basal cell carcinoma alone is almost half the dataset, while four classes have fewer than ten test lesions, so accuracy can look good while rare diagnoses are never predicted; this is why macro-F1 and balanced accuracy are reported throughout. And the skin-tone distribution is narrow: in the test set, 489 of 787 lesions are tone 3, and only 14 are tone 1, which limits what can be said about fairness.

## Splitting Without Leakage

MILK10k does not publish patient identifiers, so the finest grouping available is the lesion. The data were split **by lesion** into 70% training, 15% validation and 15% test, so the clinical and dermoscopic images of a lesion always land in the same split. The split is seeded and reproduced exactly by every script, and the code asserts that no lesion identifier appears in two splits. Each split has a single purpose:

- **Training (3,667 lesions)**: fitting the neural network and the case-based specialists.
- **Validation (786 lesions)**: choosing the training epoch, fitting every fusion rule, learned reliability, calibration map, conformal quantile, Learn-then-Test threshold and acquisition estimator, and choosing between methods.
- **Test (787 lesions)**: scored once at the end. No decision of any kind was made by looking at test results.

Because the validation lesions are used both to fit fusion rules and to calibrate the safety guarantees, the fusion rules are **cross-fitted**: each rule is fitted on four fifths of the validation set and applied to the remaining fifth, five times, so the calibration sees only out-of-fold probabilities (Chapter 6).

## External Datasets

**PAD-UFES-20** [24] contains 2,298 smartphone photographs of 1,641 lesions from 1,373 patients collected in Brazil, with six diagnoses (BCC, actinic keratosis, nevus, seborrheic keratosis, SCC and melanoma) and extensive patient questionnaires, including whether the lesion itches, has grown, hurts, has changed, bleeds or is raised. It has no dermoscopy. Its six classes map onto MILK10k as BCC to BCC, ACK to AKIEC, NEV to NV, SEK to BKL, SCC to SCCKA and MEL to MEL.

**Derm7pt** [15] contains 1,011 lesions from an Italian dermoscopy atlas, each with a clinical and a dermoscopic image, sex and body site but no age. Its diagnoses were mapped as follows: every nevus subtype to NV, every melanoma to MEL, seborrheic keratosis, lentigo and melanosis to BKL, BCC to BCC, dermatofibroma to DF and vascular lesions to VASC; eight "miscellaneous" lesions were dropped, leaving 1,003. It is used to test the complete "ask for dermoscopy" loop on a different population.

Both external datasets are used **without any retraining or recalibration**. Their images are not redistributed with this project.

## Audit of the Earlier Prototype

The first version of DERMA-GUARD was built on PAD-UFES-20 with seven models: A (image), B (metadata), C (free text), D (image and metadata), E (image and text), F (text and metadata) and G (all three). It reported accuracies between 83.5% and 89.9%. Before building on those models, we re-examined how the numbers had been produced. Three problems emerged.

### The split was by photograph

The prototype split the 2,298 photographs at random. Because many lesions were photographed more than once, and many patients had several lesions, photographs of the same lesion or patient ended up on both sides. Re-creating the exact split showed that **43% of the test photographs came from a lesion that was also in training, and 62% from a patient who was**. A model can score well on such a test set simply by recognising the lesion or the patient.

### The evaluation adjusted predictions toward target accuracies

The prototype's evaluation function contained a table of target accuracies for each model (for example 85.6% for Model A and 89.8% for Model G). After the model made its predictions, the function compared the number of correct predictions with the target. If there were too few, it took wrong predictions whose probability for the true class was close to the winner and swapped the two probabilities, so that they became correct; if there were too many, it did the reverse. The reported accuracies were therefore set in advance rather than measured. The free-text "patient narratives" used by Models C, E, F and G had also been generated from the same questionnaire fields that Model B used.

### The demonstration looked up the true label

In the live demonstration, when an uploaded file name matched a PAD-UFES-20 image, Model A's output was replaced by 92.5% on the *true* diagnosis, and three "foundation model" panel members were small arithmetic variations of Model A rather than separate models.

### What the models really achieve

Because the prototype had cached every input feature, we could load the saved weights of all seven models unchanged and score them without any adjustment, on the original test photographs and on the subset whose patient never appeared in training.

Table: The earlier prototype's seven models on PAD-UFES-20, as reported and as re-scored without adjustment.
| Model | Inputs | Reported | Same test photographs | Lesion never in training (197) | Patient never in training (132) |
|---|---|---|---|---|---|
| A | image | 85.5% | 61.4% | 55.3% | 56.8% |
| B | metadata | 84.1% | 75.9% | 68.5% | 75.0% |
| C | free text | 83.5% | 23.5% | 25.4% | 27.3% |
| D | image + metadata | 88.4% | 63.5% | 57.4% | 58.3% |
| E | image + text | 88.1% | 61.4% | 55.8% | 57.6% |
| F | text + metadata | 87.2% | 72.2% | 67.0% | 72.7% |
| G | image + text + metadata | 89.9% | 62.0% | 55.8% | 57.6% |

![The earlier prototype's models: reported accuracy versus honest re-scoring. The dashed line is the seven models fused by DERMA-GUARD's learned gate on unseen patients (cross-validated).](paper/figures/fig_legacy_models.png)

The honest picture is very different from the reported one. The metadata model is the strongest; the text model is close to chance; and the multimodal models are *worse* than the metadata model alone, the opposite of what the prototype claimed. These findings set three rules for the new project: split by lesion, never touch predictions after the model makes them, and keep every evaluation script in the repository so that any number can be regenerated.

The audit also produced something useful. Model A, which uses only the image, can run on MILK10k photographs and is kept as specialist S4. Models B to G need PAD-UFES-20's questionnaire fields or text generated from them, which MILK10k does not have, so they cannot join the MILK10k team; instead they are used in a separate experiment (Section 8.8) and are available in the application's research view. The PAD-UFES-20 questionnaire itself was used to check whether the warning signs our triage component looks for really signal cancer (Section 8.10).
