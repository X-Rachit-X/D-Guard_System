"""src/evidence/reliability.py — Reliability (R) computation for DERMA-GUARD v4.0.

R_m = sigmoid(MLP([Q_m, H_m, S_m, A_m]))
  H_m = predictive entropy of single-modality branch
  S_m = 1 - variance_of_prediction_under_TTA(5 augmentations)
  A_m = mean cosine similarity between modality m prediction and all others
MLP: 4 inputs → 8 hidden → 1 output. Trained jointly.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import NUM_CLASSES


class ReliabilityMLP(nn.Module):
    """4 → 8 → 1 MLP for reliability estimation. Output is sigmoid-activated."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 4) → (B, 1) reliability in [0,1]."""
        return self.net(x)


def compute_predictive_entropy(logits: torch.Tensor) -> float:
    """H_m = predictive entropy of single-modality prediction."""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return float(-(probs * log_probs).sum(dim=-1).mean().item())


def compute_tta_stability(
    model_fn,
    image: torch.Tensor,
    n_augmentations: int = 5,
    augmentation_fn=None,
) -> float:
    """S_m = 1 - variance_of_prediction_under_TTA(5 augmentations).

    Per spec: S_m = 1 - var(softmax predictions under 5 random augmentations).
    Higher = more stable = more reliable.

    Args:
        model_fn: callable (image: Tensor) → logit Tensor
        image:    (C, H, W) image tensor, values in [0,1] or normalised
        n_augmentations: 5 per spec
        augmentation_fn: optional; uses random flips + gaussian noise if None

    Returns:
        S_m ∈ [0, 1]
    """
    preds = []
    for _ in range(n_augmentations):
        if augmentation_fn is not None:
            aug_img = augmentation_fn(image)
        else:
            aug_img = image.clone()
            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                aug_img = torch.flip(aug_img, dims=[2])
            # Random vertical flip
            if torch.rand(1).item() > 0.5:
                aug_img = torch.flip(aug_img, dims=[1])
            # Gaussian noise (σ = 0.02, consistent with training-time jitter)
            aug_img = (aug_img + 0.02 * torch.randn_like(aug_img)).clamp(0.0, 1.0)
        try:
            with torch.no_grad():
                logits = model_fn(aug_img.unsqueeze(0))
                probs = F.softmax(logits, dim=-1).squeeze(0)
            preds.append(probs)
        except Exception:
            continue

    if len(preds) < 2:
        return 1.0  # can't estimate variance with <2 samples

    preds_stack = torch.stack(preds, dim=0)  # (n_aug, K)
    # Mean variance across classes (per spec: variance of prediction)
    variance = preds_stack.var(dim=0).mean().item()
    return float(max(0.0, 1.0 - variance * 100))  # scale: variance is small, ~0.001-0.01



def compute_agreement_score(
    target_logits: torch.Tensor,
    other_logits_list: List[torch.Tensor],
) -> float:
    """A_m = mean cosine similarity between modality m prediction and all others.

    Uses probability vectors (softmax) for cosine similarity.
    """
    if not other_logits_list:
        return 1.0  # no others to disagree with

    def as_prob_row(logits: torch.Tensor) -> torch.Tensor:
        """(K,) or (1,K) or (B,K) -> a single (1,K) probability row.

        Callers pass logits in both shapes; unconditionally unsqueezing turned
        an already-batched (1,K) into (1,1,K) and made cosine_similarity return
        a vector instead of a scalar.
        """
        p = F.softmax(logits, dim=-1)
        if p.dim() > 2:
            p = p.reshape(-1, p.shape[-1])
        if p.dim() == 2:
            p = p.mean(dim=0)
        return p.reshape(1, -1)

    target_probs = as_prob_row(target_logits)
    similarities = [
        float(F.cosine_similarity(target_probs, as_prob_row(other)).item())
        for other in other_logits_list
    ]
    return float(sum(similarities) / len(similarities))


# Weights for the closed-form reliability estimate. Certainty (1 - normalised
# entropy) dominates because it is the signal that most directly separates a
# specialist that knows the answer from one that is guessing.
_RELIABILITY_WEIGHTS = {"certainty": 0.40, "agreement": 0.30, "quality": 0.20, "stability": 0.10}


def compute_reliability(
    Q_m: float,
    H_m: float,
    S_m: float,
    A_m: float,
    reliability_mlp: Optional[ReliabilityMLP] = None,
    num_classes: int = NUM_CLASSES,
) -> float:
    """R_m from quality, entropy, TTA stability and cross-modal agreement.

    With `reliability_mlp=None` (the default, and what the deployed system uses)
    this is the closed-form weighted mean below. The spec's
    `sigmoid(MLP([Q,H,S,A]))` is only meaningful once that MLP has been trained
    jointly with fusion; an untrained MLP returns a value near 0.5 that is
    uncorrelated with its inputs, which would make the whole evidence layer
    decorative. Pass a trained MLP to use the learned form.

    Entropy is normalised by log(num_classes), the true maximum for a K-way
    predictive distribution - the previous hard-coded divisor of 2.0 was wrong
    for every K except 7.
    """
    import math

    H_max = math.log(max(num_classes, 2))
    H_norm = float(min(1.0, max(0.0, H_m / H_max)))

    if reliability_mlp is not None:
        feat = torch.tensor([[Q_m, H_norm, S_m, A_m]], dtype=torch.float32)
        with torch.no_grad():
            return float(reliability_mlp(feat).item())

    w = _RELIABILITY_WEIGHTS
    score = (w["certainty"] * (1.0 - H_norm)
             + w["agreement"] * float(max(0.0, min(1.0, A_m)))
             + w["quality"] * float(max(0.0, min(1.0, Q_m)))
             + w["stability"] * float(max(0.0, min(1.0, S_m))))
    return float(max(0.0, min(1.0, score)))
