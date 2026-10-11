"""
src/agents/vision_team.py - Specialist team (DERMA-GUARD v4.0).

Runs every trained specialist (A-G) whose required modalities are present and
reports their independent opinions, plus the per-modality embeddings the fusion
and retrieval layers consume.

Deviation from proj.md Part 2, recorded in docs/evaluation_integrity.md: the
spec names DermLIP-PanDerm, PanDerm, two Swin-Tiny instances and ResNet152. None
of those are trainable or even loadable in this environment (PanDerm weights are
not published, PAD-UFES-20 contains no dermoscopy images, and torch here is
CPU-only), and the previous implementation shipped them with randomly-initialised
heads - meaning five "specialists" that voted on noise. The team below instead
exposes the models that actually carry trained weights.

Image quality still uses the hand-crafted statistics from the spec, but scored by
the deterministic weighted formula rather than the untrained ImageQualityMLP;
an untrained MLP would emit an arbitrary number that the safety layer then
thresholds at 0.30, which is worse than no signal at all.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.constants import CLASS_NAMES, NUM_CLASSES
from src.models.architectures import SPECIALIST_MODALITIES
from src.models.backends import CaseEncoding, TrainedBackends, get_backends


# ---------------------------------------------------------------------------
# Per-image quality statistics
# ---------------------------------------------------------------------------

class ImageQualityMLP(nn.Module):
    """Maps the 5 image statistics to a quality score.

    Retained for the trainable path only. `score_image_quality` does NOT use it
    by default - see the module docstring.
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_image_stats(img_bgr: np.ndarray, expected_size: int = 224) -> Dict[str, float]:
    """Five quality statistics from an OpenCV BGR image, each scaled to [0,1].

    occlusion_score is returned already inverted (1.0 = nothing occluded) so all
    five features point the same way: higher is better.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean()) / 255.0
    contrast = float(gray.std()) / 128.0
    resolution_ratio = min(1.0, min(h, w) / expected_size)
    dark_fraction = float(np.sum(gray < 10)) / (h * w)

    return {
        "blur_score": min(1.0, blur / 200.0),   # >200 var(Laplacian) is sharp
        "brightness": brightness,
        "contrast": min(1.0, contrast),
        "resolution_ratio": resolution_ratio,
        "occlusion_score": 1.0 - dark_fraction,
    }


# Weights from proj.md Part 4; they sum to 1.0 so Q_image lands in [0,1].
_QUALITY_WEIGHTS = {
    "blur_score": 0.25, "brightness": 0.15, "contrast": 0.20,
    "resolution_ratio": 0.25, "occlusion_score": 0.15,
}


def score_image_quality(image: Image.Image | np.ndarray | torch.Tensor,
                        quality_mlp: Optional[ImageQualityMLP] = None) -> float:
    """Q_image in [0,1]. Uses the deterministic formula unless a *trained* MLP is given."""
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu()
        if arr.dim() == 4:
            arr = arr[0]
        arr = (arr.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    elif isinstance(image, Image.Image):
        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        bgr = image

    stats = compute_image_stats(bgr)
    if quality_mlp is not None:
        feat = torch.tensor([[stats[k] for k in
                              ("blur_score", "brightness", "contrast",
                               "resolution_ratio", "occlusion_score")]], dtype=torch.float32)
        with torch.no_grad():
            return float(quality_mlp(feat).item())
    return float(sum(_QUALITY_WEIGHTS[k] * stats[k] for k in _QUALITY_WEIGHTS))


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class SpecialistReport:
    """Independent opinions from every applicable specialist."""

    model_predictions: Dict[str, torch.Tensor]    # specialist code -> (1, K) logits
    model_embeddings: Dict[str, torch.Tensor]     # modality name   -> embedding
    model_confidences: Dict[str, float]           # specialist code -> max softmax
    uncertainty_estimates: Dict[str, float]       # specialist code -> predictive entropy
    quality_scores: Dict[str, float]              # modality name   -> Q
    modality_logits: Dict[str, torch.Tensor] = field(default_factory=dict)
    image_stats: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def top1(self) -> Dict[str, int]:
        return {name: int(lg.argmax(dim=-1).flatten()[0].item())
                for name, lg in self.model_predictions.items()}


# Backwards-compatible alias: the pipeline used to call this a VisionReport.
VisionReport = SpecialistReport


# ---------------------------------------------------------------------------
# SpecialistTeam
# ---------------------------------------------------------------------------

class SpecialistTeam(nn.Module):
    """Routes a case to every compatible trained specialist and collects votes."""

    #: which specialist supplies each modality's standalone opinion for fusion
    UNIMODAL_FOR = {"clinical_image": "A", "metadata": "B", "text": "C"}

    def __init__(self, backends: Optional[TrainedBackends] = None,
                 class_names: Optional[List[str]] = None,
                 quality_mlp: Optional[ImageQualityMLP] = None):
        super().__init__()
        self.backends = backends or get_backends()
        self.class_names = class_names or list(CLASS_NAMES)
        self.num_classes = NUM_CLASSES
        # Left as None deliberately: an untrained MLP is worse than the formula.
        self.quality_mlp = quality_mlp
        self._loaded = False

    def load(self) -> "SpecialistTeam":
        """Warm the artifact cache so the first real case is not slow."""
        missing = [c for c in SPECIALIST_MODALITIES
                   if c not in self.backends.available_specialists()]
        if missing:
            warnings.warn(f"[SpecialistTeam] checkpoints absent for: {missing}")
        self._loaded = True
        return self

    @staticmethod
    def _entropy(logits: torch.Tensor) -> float:
        p = F.softmax(logits, dim=-1)
        return float(-(p * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean().item())

    def process_case(self, image: Optional[Image.Image | torch.Tensor] = None,
                     metadata: Optional[Dict] = None,
                     text: Optional[str] = None,
                     encoding: Optional[CaseEncoding] = None) -> SpecialistReport:
        """Encode the case, run every compatible specialist, and report.

        Passing a pre-computed `encoding` skips re-encoding, which the stress
        benchmark relies on when replaying one case under many corruptions.
        """
        enc = encoding or self.backends.encode_case(image=image, metadata=metadata, text=text)
        notes = list(enc.notes)

        # Per-modality quality: images get the statistical score; metadata and
        # text are scored by the Scribe / evidence layer, not here.
        quality_scores: Dict[str, float] = {}
        image_stats: Dict[str, float] = {}
        if image is not None and enc.image is not None:
            try:
                image_stats = compute_image_stats(
                    pil_to_bgr(image) if isinstance(image, Image.Image)
                    else (image.detach().cpu()[0].permute(1, 2, 0).numpy() * 255)
                    .clip(0, 255).astype(np.uint8)[:, :, ::-1])
                quality_scores["clinical_image"] = score_image_quality(image, self.quality_mlp)
            except Exception as exc:
                notes.append(f"image quality scoring failed: {exc}")
                quality_scores["clinical_image"] = 0.5

        available = enc.available()
        codes = self.backends.specialists_for(available)

        model_predictions: Dict[str, torch.Tensor] = {}
        for code in codes:
            try:
                model_predictions[code] = self.backends.run_specialist(code, enc)
            except Exception as exc:
                notes.append(f"specialist {code} failed: {exc}")

        if not model_predictions:
            notes.append("no specialist could run for the supplied modalities")

        # One standalone opinion per modality, for DiagnosisAdaptiveFusion.
        modality_logits: Dict[str, torch.Tensor] = {}
        for modality, code in self.UNIMODAL_FOR.items():
            src = "image" if modality == "clinical_image" else modality
            if getattr(enc, src, None) is not None and code in model_predictions:
                modality_logits[modality] = model_predictions[code]

        model_embeddings: Dict[str, torch.Tensor] = {}
        if enc.image is not None:
            model_embeddings["clinical_image"] = enc.image.squeeze(0).detach().cpu()
        if enc.metadata is not None:
            model_embeddings["metadata"] = enc.metadata.squeeze(0).detach().cpu()
        if enc.text is not None:
            model_embeddings["text"] = enc.text.squeeze(0).detach().cpu()
        # Raw pre-projection features are carried too, because the retrieval
        # index is built over THOSE (2048-d ResNet50, 768-d BERT [CLS]) rather
        # than the 256-d projections: the raw features are the ones that
        # reproduce bit-for-bit from scripts/build_features.py, while the legacy
        # projectors do not (docs/evaluation_integrity.md).
        if getattr(enc, "image_raw", None) is not None:
            model_embeddings["clinical_image_raw"] = enc.image_raw.squeeze(0).detach().cpu()
        if getattr(enc, "text_raw", None) is not None:
            model_embeddings["text_raw"] = enc.text_raw.squeeze(0).detach().cpu()

        return SpecialistReport(
            model_predictions=model_predictions,
            model_embeddings=model_embeddings,
            model_confidences={c: float(F.softmax(lg, dim=-1).max().item())
                               for c, lg in model_predictions.items()},
            uncertainty_estimates={c: self._entropy(lg)
                                   for c, lg in model_predictions.items()},
            quality_scores=quality_scores,
            modality_logits=modality_logits,
            image_stats=image_stats,
            notes=notes,
        )


# Backwards-compatible alias.
VisionSpecialistTeam = SpecialistTeam
