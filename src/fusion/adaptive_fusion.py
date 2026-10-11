"""
src/fusion/adaptive_fusion.py - Iterative EM-style DiagnosisAdaptiveFusion.

Fuses MODALITY embeddings (clinical image / dermoscopy image / metadata / text),
not per-model outputs. That distinction is the fix for the defect described in
docs/evaluation_integrity.md: this module previously declared five *vision
models* as its fusion axis while the evidence layer produced a four-*modality*
mask, so every real call raised a shape error that the orchestrator swallowed.
Both sides now index `src.constants.MODALITIES`, so a future mismatch is a
single-source-of-truth violation rather than a silent runtime failure.

Algorithm (proj.md Part 5):

    iteration 0   w = softmax(alpha * f([Q,R,U,C])) * mask, renormalised
                  z = sum_m w_m * z_m ; diag = argmax(classifier(z))
    iteration k   w = softmax(alpha * f([Q,R,U,C])
                              + beta  * W_prior[diag]
                              + gamma * log(conflict_penalty)) * mask, renormalised
                  stop when the diagnosis is stable and max|dw| < threshold

Audit fixes carried over:
  1. Missing modalities are masked then renormalised; all-missing raises
     InsufficientEvidenceError rather than dividing by zero.
  2. Class imbalance handled by FocalLossWithWeights (gamma=2.0, inverse-freq).
  3. Callers clip gradients to max_norm=1.0.
  4. Training uses mixed precision where CUDA is available.
  5. Early stopping is on validation macro-F1, not loss.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import MODALITIES, MODALITY_DIMS, NUM_CLASSES, NUM_MODALITIES


class InsufficientEvidenceError(Exception):
    """Raised when every modality is missing - the pipeline must ABSTAIN."""


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

class ModalityProjector(nn.Module):
    """Project heterogeneous modality embeddings into one common space."""

    def __init__(self, input_dims: Dict[str, int], d_common: int = 256):
        super().__init__()
        self.projectors = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, d_common),
                nn.LayerNorm(d_common),
                nn.ReLU(),
            )
            for name, dim in input_dims.items()
        })
        self.d_common = d_common

    def forward(self, embeddings: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for name, emb in embeddings.items():
            if name in self.projectors and emb is not None:
                vec = emb if emb.dim() == 2 else emb.unsqueeze(0)
                out[name] = self.projectors[name](vec)
        return out


class WeightLogitNetwork(nn.Module):
    """f: [Q_agg, R_agg, U_agg, C] (4 scalars) -> one weight logit per modality.

    Per proj.md Part 5 this is a single call producing a vector over modalities,
    not one scalar-output call per modality.
    """

    def __init__(self, num_modalities: int = NUM_MODALITIES):
        super().__init__()
        self.num_modalities = num_modalities
        self.net = nn.Sequential(
            nn.Linear(4, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, num_modalities),
        )

    def forward(self, evidence_vector: torch.Tensor) -> torch.Tensor:
        return self.net(evidence_vector.unsqueeze(0)).squeeze(0)


class FocalLossWithWeights(nn.Module):
    """Focal loss (gamma=2.0) with inverse-frequency class weights."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, labels, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()

    @staticmethod
    def compute_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        """weight_c = N / (num_classes * n_c)."""
        n = len(labels)
        weights = torch.zeros(num_classes)
        for c in range(num_classes):
            n_c = int((labels == c).sum().item())
            weights[c] = n / (num_classes * max(n_c, 1))
        return weights


@dataclass
class FusionResult:
    prediction: torch.Tensor        # (1, K) logits
    diagnosis: int
    confidence: float
    weights: torch.Tensor           # (M,) in MODALITIES order; 0 where missing
    fused_embedding: torch.Tensor   # (d_common,)
    iterations_used: int
    converged: bool

    def weight_dict(self) -> Dict[str, float]:
        """Per-modality weights by name - what the evidence trace and UI show."""
        return {name: float(self.weights[i]) for i, name in enumerate(MODALITIES)}

    def to_dict(self) -> Dict:
        return {
            "diagnosis": self.diagnosis,
            "confidence": self.confidence,
            "weights": self.weight_dict(),
            "iterations_used": self.iterations_used,
            "converged": self.converged,
        }


# ---------------------------------------------------------------------------
# DiagnosisAdaptiveFusion
# ---------------------------------------------------------------------------

class DiagnosisAdaptiveFusion(nn.Module):
    """Iterative, diagnosis-adaptive fusion over modalities."""

    MODALITY_NAMES = list(MODALITIES)
    EMBEDDING_DIMS = dict(MODALITY_DIMS)

    def __init__(self, num_modalities: int = NUM_MODALITIES,
                 num_classes: int = NUM_CLASSES, d_common: int = 256,
                 max_iterations: int = 3, convergence_threshold: float = 0.05,
                 use_learned_weighting: bool = False):
        super().__init__()
        if num_modalities != len(self.MODALITY_NAMES):
            raise ValueError(
                f"num_modalities={num_modalities} contradicts src.constants.MODALITIES "
                f"({len(self.MODALITY_NAMES)}). Change the constant, not this argument."
            )
        self.num_modalities = num_modalities
        self.num_classes = num_classes
        self.d_common = d_common
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        # False until WeightLogitNetwork has been trained - see _base_weight_logits.
        self.use_learned_weighting = use_learned_weighting

        self.f = WeightLogitNetwork(num_modalities)
        self.W_prior = nn.Parameter(torch.randn(num_classes, num_modalities) * 0.01)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.gamma_param = nn.Parameter(torch.tensor(0.5))
        self.projector = ModalityProjector(self.EMBEDDING_DIMS, d_common)
        self.classifier = nn.Linear(d_common, num_classes)

    # ------------------------------------------------------------------
    def _evidence_vector(self, quality: Dict[str, float], reliability: Dict[str, float],
                         utility: Dict[str, float], consistency: float) -> torch.Tensor:
        """Aggregate Q/R/U over modalities into the 4-scalar input for f."""
        device = next(self.f.parameters()).device

        def agg(d: Dict[str, float], default: float) -> float:
            vals = [float(d.get(n, default)) for n in self.MODALITY_NAMES]
            return sum(vals) / len(vals)

        vec = [agg(quality, 0.5), agg(reliability, 0.5), agg(utility, 0.5), float(consistency)]
        return torch.tensor(vec, dtype=torch.float32, device=device)

    def _base_weight_logits(self, quality: Dict[str, float], reliability: Dict[str, float],
                            utility: Dict[str, float], consistency: float) -> torch.Tensor:
        """Weight logits before the diagnosis prior is applied.

        Two modes:

        `use_learned_weighting=False` (default) - closed-form evidence product,
            logit_m = log(Q_m * R_m * U_m), i.e. w proportional to the evidence
            score E_m = Q_m*R_m*U_m, renormalised over available modalities. This
            needs no training, so the deployed system weights modalities by
            measured evidence rather than by a randomly-initialised MLP. A
            low-quality, unreliable modality is down-weighted, which is the whole
            point of the evidence layer.

        `use_learned_weighting=True` - use the WeightLogitNetwork f. Only correct
            once f has actually been trained; an untrained f produces arbitrary
            weights that ignore the evidence entirely.
        """
        device = next(self.parameters()).device
        if self.use_learned_weighting:
            return self.f(self._evidence_vector(quality, reliability, utility, consistency))

        eps = 1e-6
        scores = []
        for name in self.MODALITY_NAMES:
            q = float(quality.get(name, 0.5))
            r = float(reliability.get(name, 0.5))
            u = float(utility.get(name, 0.5))
            scores.append(torch.log(torch.tensor(max(q * r * u, eps))))
        return torch.stack(scores).to(device)

    def _normalise(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """softmax -> mask -> renormalise, with an even-split fallback."""
        w = F.softmax(logits, dim=-1) * mask
        total = w.sum()
        if total < 1e-8:
            # Every surviving weight underflowed: fall back to an even split over
            # whatever is actually available rather than emitting NaNs.
            return mask / mask.sum().clamp(min=1e-8)
        return w / total

    def _weighted_sum(self, projected: Dict[str, torch.Tensor],
                      weights: torch.Tensor) -> torch.Tensor:
        b = next(iter(projected.values())).shape[0] if projected else 1
        device = weights.device
        z = torch.zeros(b, self.d_common, device=device)
        used = torch.zeros((), device=device)
        for i, name in enumerate(self.MODALITY_NAMES):
            if name in projected and weights[i] > 1e-8:
                z = z + weights[i] * projected[name].to(device)
                used = used + weights[i]
        if used.item() > 1e-8:
            z = z / used
        return z

    def _per_modality_logits(self, projected: Dict[str, torch.Tensor],
                             modality_logits: Optional[Dict[str, torch.Tensor]]
                             ) -> Dict[str, torch.Tensor]:
        """Class logits for each available modality.

        Prefers `modality_logits` - the outputs of the TRAINED unimodal
        specialists (A image / B metadata / C text). Falls back to the internal
        linear head only when none are supplied, which is the trainable
        spec-faithful path; that head is randomly initialised until trained, so
        production callers should always pass modality_logits.
        """
        if modality_logits:
            out = {}
            for name in self.MODALITY_NAMES:
                lg = modality_logits.get(name)
                if lg is None or name not in projected:
                    continue
                out[name] = lg if lg.dim() == 2 else lg.unsqueeze(0)
            if out:
                return out
        return {name: self.classifier(emb) for name, emb in projected.items()}

    def compute_conflict_penalty(self, per_modality: Dict[str, torch.Tensor],
                                 diag: int) -> torch.Tensor:
        """penalty_m = 1 - P(diag | modality m); missing modalities score 0."""
        penalties = []
        for name in self.MODALITY_NAMES:
            if name in per_modality:
                p = F.softmax(per_modality[name], dim=-1)[:, diag].mean()
                penalties.append(float(1.0 - p.item()))
            else:
                penalties.append(0.0)
        device = next(self.parameters()).device
        return torch.tensor(penalties, dtype=torch.float32, device=device)

    def _pool(self, per_modality: Dict[str, torch.Tensor],
              weights: torch.Tensor) -> torch.Tensor:
        """Weighted log-opinion pool of the per-modality class distributions.

        Summing weighted log-probabilities (rather than probabilities) means a
        modality that confidently rules a class out can veto it, which is the
        behaviour the safety layer relies on when modalities disagree.
        """
        device = weights.device
        acc, used = None, 0.0
        for i, name in enumerate(self.MODALITY_NAMES):
            if name not in per_modality or weights[i] <= 1e-8:
                continue
            logp = F.log_softmax(per_modality[name].to(device), dim=-1)
            term = weights[i] * logp
            acc = term if acc is None else acc + term
            used += float(weights[i].detach())
        if acc is None:
            raise InsufficientEvidenceError("No modality contributed to the pool.")
        return acc / max(used, 1e-8)

    # ------------------------------------------------------------------
    def fuse(self, embeddings: Dict[str, torch.Tensor],
             quality: Dict[str, float], reliability: Dict[str, float],
             utility: Dict[str, float], consistency: float,
             missing_mask: torch.Tensor,
             modality_logits: Optional[Dict[str, torch.Tensor]] = None) -> FusionResult:
        """Run EM-style fusion. `missing_mask` is (M,), 1 = available.

        Args:
            embeddings:      modality -> embedding tensor (pre-projection)
            quality/reliability/utility: per-modality evidence scores
            consistency:     global cross-modality agreement C
            missing_mask:    (M,) 1 = available, ordered as src.constants.MODALITIES
            modality_logits: modality -> (1, K) logits from the trained unimodal
                             specialists. Strongly recommended: without it the
                             prediction comes from this module's untrained head.

        Raises InsufficientEvidenceError when nothing is available.
        """
        if missing_mask.numel() != self.num_modalities:
            raise ValueError(
                f"missing_mask has {missing_mask.numel()} entries but fusion expects "
                f"{self.num_modalities} ({self.MODALITY_NAMES}). Build it with "
                f"src.evidence.missingness.compute_missingness()."
            )
        if float(missing_mask.sum()) == 0.0:
            raise InsufficientEvidenceError(
                "All modalities are missing; cannot fuse. Pipeline should ABSTAIN.")

        device = next(self.parameters()).device
        mask = missing_mask.to(device).float()

        # Only project modalities that are both supplied AND unmasked, so a
        # modality that was dropped by a stress condition cannot leak in.
        supplied = {n: e for n, e in embeddings.items()
                    if e is not None and n in self.EMBEDDING_DIMS
                    and mask[self.MODALITY_NAMES.index(n)] > 0}
        if not supplied:
            raise InsufficientEvidenceError(
                "Mask marks modalities available but no embeddings were supplied.")
        projected = self.projector(supplied)

        per_modality = self._per_modality_logits(projected, modality_logits)
        base_logits = self._base_weight_logits(quality, reliability, utility, consistency)

        # ---- iteration 0: no diagnosis prior ----
        w = self._normalise(self.alpha * base_logits, mask)
        z = self._weighted_sum(projected, w)
        pred = self._pool(per_modality, w)
        diag = int(pred.argmax(dim=-1).flatten()[0].item())

        # ---- iterations 1..max ----
        converged = False
        iterations = 0
        for _ in range(self.max_iterations):
            iterations += 1
            pi = self.W_prior[diag]
            penalty = self.compute_conflict_penalty(per_modality, diag)

            logits_new = (self.alpha * base_logits
                          + self.beta * pi
                          + self.gamma_param * torch.log(penalty + 1e-8))
            w_new = self._normalise(logits_new, mask)
            z_new = self._weighted_sum(projected, w_new)
            pred_new = self._pool(per_modality, w_new)
            diag_new = int(pred_new.argmax(dim=-1).flatten()[0].item())

            stable = (diag_new == diag
                      and torch.max(torch.abs(w_new - w)).item() < self.convergence_threshold)
            w, z, pred, diag = w_new, z_new, pred_new, diag_new
            if stable:
                converged = True
                break

        confidence = float(F.softmax(pred, dim=-1).max(dim=-1).values.mean().item())
        return FusionResult(
            prediction=pred,
            diagnosis=diag,
            confidence=confidence,
            weights=w.detach().cpu(),
            fused_embedding=z.mean(dim=0).detach(),
            iterations_used=iterations,
            converged=converged,
        )

    def forward(self, embeddings, quality, reliability, utility, consistency, missing_mask):
        result = self.fuse(embeddings, quality, reliability, utility, consistency, missing_mask)
        return {
            "logits": result.prediction,
            "weights": result.weights,
            "fused_embedding": result.fused_embedding,
            "iterations": result.iterations_used,
        }
