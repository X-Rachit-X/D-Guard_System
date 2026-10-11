"""src/evidence/missingness.py - Missingness flags (M) for DERMA-GUARD v4.0.

The mask this module emits is consumed directly by DiagnosisAdaptiveFusion, so
both import their ordering from src.constants.MODALITIES. Do not redefine the
list here.
"""
from __future__ import annotations

from typing import Dict, List

import torch

from src.constants import MODALITIES

# Re-exported under its historical name for callers that already import it.
ALL_MODALITIES: List[str] = list(MODALITIES)


def compute_missingness(clinical_available: bool, dermo_available: bool,
                        meta_available: bool, text_available: bool) -> Dict:
    """Build the availability flags and the fusion mask.

    Returns a dict with:
        flags:         {modality: bool}   True = available
        missing_list:  names of absent modalities
        missing_count: int
        all_missing:   True when the pipeline must ABSTAIN
        missing_mask:  torch.Tensor (len(MODALITIES),), 1.0 = available,
                       ordered exactly as src.constants.MODALITIES
    """
    flags = {
        "clinical_image": bool(clinical_available),
        "dermoscopy_image": bool(dermo_available),
        "metadata": bool(meta_available),
        "text": bool(text_available),
    }
    missing_list = [m for m in ALL_MODALITIES if not flags[m]]
    mask = torch.tensor([float(flags[m]) for m in ALL_MODALITIES], dtype=torch.float32)

    return {
        "flags": flags,
        "missing_list": missing_list,
        "missing_count": len(missing_list),
        "all_missing": len(missing_list) == len(ALL_MODALITIES),
        "missing_mask": mask,
    }


def mask_from_available(available: List[str]) -> torch.Tensor:
    """Build the same mask from a list of present modality names."""
    have = set(available)
    return torch.tensor([float(m in have) for m in ALL_MODALITIES], dtype=torch.float32)
