"""
src/constants.py - Canonical vocabulary for D-GUARD.

Single source of truth for class labels, malignancy, and modality widths.
Dataset: MILK10k (ISIC, DOI 10.34970/648456) - 11 classes, every lesion with a
clinical photo AND a dermoscopy image. This is the only dataset the final system
uses. (PAD-UFES-20 belonged to the earlier prototype; see docs/history/.)
"""
from __future__ import annotations

from typing import Dict, List

# Input modalities the evidential model understands, in fold order.
MODALITIES: List[str] = ["clinical_image", "dermoscopy_image", "metadata", "text"]

# Default embedding widths for the retrieval index (src/retrieval/rag_index.py).
# The final MILK10k index uses its own 256-d spec (src/agents/librarian.py).
MODALITY_DIMS: Dict[str, int] = {
    "clinical_image": 256,
    "dermoscopy_image": 256,
    "metadata": 116,
    "text": 256,
}


# ---------------------------------------------------------------------------
# MILK10k (11 classes) - the primary dataset from v3 onward
# ---------------------------------------------------------------------------
# Verified against milk10k/supplements/training_gt.csv on 2026-10-11: the file
# is a one-hot lesion-level table over exactly these 11 columns, 5240 rows.
# Ordering here is the alphabetical column order of that file, so index i always
# means the same class as column i of the official ground truth.
MILK_CLASS_NAMES: List[str] = [
    "AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF",
    "MAL_OTH", "MEL", "NV", "SCCKA", "VASC",
]
MILK_NUM_CLASSES: int = len(MILK_CLASS_NAMES)
MILK_CLASS_TO_INDEX: Dict[str, int] = {n: i for i, n in enumerate(MILK_CLASS_NAMES)}

MILK_CLASS_FULL_NAMES: Dict[str, str] = {
    "AKIEC": "Actinic Keratosis / Intraepithelial Carcinoma",
    "BCC": "Basal Cell Carcinoma",
    "BEN_OTH": "Benign - Other",
    "BKL": "Benign Keratosis-like Lesion",
    "DF": "Dermatofibroma",
    "INF": "Inflammatory / Infectious",
    "MAL_OTH": "Malignant - Other",
    "MEL": "Melanoma",
    "NV": "Melanocytic Nevus",
    "SCCKA": "Squamous Cell Carcinoma / Keratoacanthoma",
    "VASC": "Vascular Lesion",
}

#: Malignant MILK10k classes. AKIEC is counted malignant: the official
#: diagnosis_1 column labels its members "Malignant" (in-situ carcinoma), and
#: the safety layer must escalate them.
MILK_MALIGNANT_CLASSES: List[str] = ["AKIEC", "BCC", "MAL_OTH", "MEL", "SCCKA"]

#: Measured lesion-level prevalence, milk10k/supplements/training_gt.csv.
#: 3757 of 5240 lesions (71.7%) are malignant and BCC alone is 48.1%. This is a
#: tertiary-referral population, NOT a screening one -- any cost matrix or risk
#: target calibrated on it inherits that base rate. See docs/05.
MILK_MALIGNANT_PREVALENCE: float = 0.717

#: skin_tone_class values in supplements/training_input.csv: 6 ordinal levels,
#: heavily imbalanced (level 3 = 60.6% of images, level 0 = 12 images). Groups
#: below ceil(1/alpha) calibration points get NO per-group guarantee.
MILK_SKIN_TONE_CLASSES: List[str] = ["0", "1", "2", "3", "4", "5"]

#: MONET concept-activation columns shipped per image. Continuous in [0,1],
#: present for all 10480 images. Usable as auxiliary metadata features.
MILK_MONET_COLUMNS: List[str] = [
    "MONET_ulceration_crust",
    "MONET_hair",
    "MONET_vasculature_vessels",
    "MONET_erythema",
    "MONET_pigmented",
    "MONET_gel_water_drop_fluid_dermoscopy_liquid",
    "MONET_skin_markings_pen_ink_purple_pen",
]

#: image_type values. Every one of the 5240 lesions has exactly one of each,
#: which is why acquisition is directly measurable on this dataset.
MILK_CLINICAL_TYPE: str = "clinical: close-up"
MILK_DERMOSCOPIC_TYPE: str = "dermoscopic"


def milk_malignant_indices() -> List[int]:
    """Indices of malignant classes in MILK_CLASS_NAMES order."""
    return [MILK_CLASS_TO_INDEX[c] for c in MILK_MALIGNANT_CLASSES]
