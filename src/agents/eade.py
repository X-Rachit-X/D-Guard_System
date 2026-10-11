"""
src/agents/eade.py - Evidence-Adaptive Decision Engine.

Turns a SpecialistReport into a PROCEED / ACQUIRE / DEFER decision:

    per-modality opinions  ->  DS fold  ->  (belief, ignorance, conflict)
                                    |
                      conformal set + LTT threshold + cost matrix
                                    |
                     PROCEED | ACQUIRE(target) | DEFER(reason)

Two modes, because the trained specialists predate the evidential heads:

  * `native`  - modality encoders carry trained EvidentialHeads and emit
                Dirichlet evidence directly. This is the target design; it needs
                the GPU training run (notebooks/kaggle_train_dguard.py).
  * `bridge`  - lift an existing softmax specialist into an opinion using its
                normalised predictive entropy as the ignorance mass:

                    u_m = H(p_m) / log K ,   S_m = K / u_m ,   e_m = p_m S_m - 1

                A near-uniform prediction gives u -> 1, i.e. the vacuous opinion,
                so a useless modality drops out of the DS fold automatically.
                This is exactly the "DS on softmax" ablation from the protocol,
                and it lets the whole decision path run and be tested today
                against the already-trained A-G ladder.

The bridge is an approximation: entropy is aleatoric+epistemic combined, whereas
a trained Dirichlet head separates them. Numbers from `bridge` must be labelled
as such and must not be presented as the native method's result.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from src.constants import CLASS_NAMES, MALIGNANT_CLASSES, MODALITIES, NUM_CLASSES
from src.decision.policy import Action, CostMatrix, Decision, ThreeActionPolicy
from src.evidence.evidential import (
    Opinion,
    fold_opinions,
    opinion_from_evidence,
    vacuous,
)


def malignant_indices(class_names: Sequence[str] = CLASS_NAMES) -> List[int]:
    return [i for i, c in enumerate(class_names) if c in set(MALIGNANT_CLASSES)]


# ---------------------------------------------------------------------------
# softmax -> opinion bridge
# ---------------------------------------------------------------------------

def opinion_from_logits(logits: torch.Tensor, floor: float = 1e-3) -> Opinion:
    """Lift softmax logits into an evidential opinion via normalised entropy.

    u = H(p)/log K in [0,1]: 0 for a one-hot prediction, 1 for uniform.
    Then S = K/u and e = p*S - 1, clamped at zero.

    `floor` keeps u strictly positive so S stays finite for a perfectly
    confident prediction.
    """
    lg = logits if logits.dim() == 2 else logits.unsqueeze(0)
    p = torch.softmax(lg, dim=-1)
    k = p.shape[-1]
    h = -(p * torch.log(p.clamp(min=1e-12))).sum(-1, keepdim=True)
    u = (h / math.log(k)).clamp(min=floor, max=1.0)
    S = k / u
    return opinion_from_evidence((p * S - 1.0).clamp(min=0.0))


# ---------------------------------------------------------------------------
# result
# ---------------------------------------------------------------------------

@dataclass
class EvidenceAssessment:
    decision: Decision
    probability: np.ndarray
    ignorance: float
    conflict: float
    per_modality_ignorance: Dict[str, float] = field(default_factory=dict)
    modalities_used: List[str] = field(default_factory=list)
    mode: str = "bridge"

    @property
    def action(self) -> Action:
        return self.decision.action

    def to_dict(self) -> Dict:
        return {
            **self.decision.to_dict(),
            "probability": self.probability.tolist(),
            "ignorance": self.ignorance,
            "conflict": self.conflict,
            "per_modality_ignorance": self.per_modality_ignorance,
            "modalities_used": self.modalities_used,
            "mode": self.mode,
        }


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------

class EvidenceAdaptiveEngine:
    """Evidential fusion + risk-controlled three-action policy.

    `conformal` and `policy.lambda_hat` come from calibration on a held-out
    split; leaving them None runs the engine uncalibrated, which is fine for a
    smoke test but carries no guarantee. `is_calibrated` says which it is, so a
    caller can never quietly report guaranteed behaviour from an uncalibrated
    run.
    """

    #: which specialist speaks for each modality when bridging
    UNIMODAL_FOR = {"clinical_image": "A", "metadata": "B", "text": "C"}

    def __init__(self, policy: Optional[ThreeActionPolicy] = None,
                 conformal=None, class_names: Sequence[str] = CLASS_NAMES,
                 mode: str = "bridge",
                 acquirable: Sequence[str] = ("dermoscopy_image",)):
        self.class_names = list(class_names)
        self.mode = mode
        self.acquirable = list(acquirable)
        self.conformal = conformal
        self.policy = policy or ThreeActionPolicy(
            cost=CostMatrix(),
            malignant_classes=malignant_indices(class_names),
            num_classes=len(class_names),
        )

    @property
    def is_calibrated(self) -> bool:
        return self.policy.lambda_hat is not None and self.conformal is not None

    # ------------------------------------------------------------------
    def opinions_from_report(self, report) -> Dict[str, Opinion]:
        """Per-modality opinions, with absent modalities as vacuous."""
        k = len(self.class_names)
        out: Dict[str, Opinion] = {}
        native = getattr(report, "modality_opinions", None) or {}
        logits = getattr(report, "modality_logits", {}) or {}

        for m in MODALITIES:
            if m in native:
                out[m] = native[m]
            elif m in logits and logits[m] is not None:
                out[m] = opinion_from_logits(logits[m])
            else:
                out[m] = vacuous(1, k)
        return out

    # ------------------------------------------------------------------
    def assess(self, report, *, n_acquisitions_so_far: int = 0,
               predicted_benefit: Optional[Dict[str, float]] = None,
               available: Optional[Sequence[str]] = None) -> EvidenceAssessment:
        """Produce a decision for one case from a SpecialistReport."""
        ops = self.opinions_from_report(report)
        fused = fold_opinions([ops[m] for m in MODALITIES])

        probs = fused.probability.squeeze(0).detach().cpu().numpy()
        ignorance = float(fused.u.squeeze().item())
        conflict = float(fused.conflict.squeeze().item()) if fused.conflict is not None else 0.0

        pset = None
        if self.conformal is not None:
            try:
                pset = self.conformal.predict_sets(probs[None, :])[0]
            except Exception:
                pset = None

        present = set(available if available is not None
                      else [m for m, o in ops.items() if o.u.squeeze().item() < 0.999])
        still_acquirable = [m for m in self.acquirable if m not in present]

        decision = self.policy.decide(
            probs,
            ignorance=ignorance,
            conflict=conflict,
            prediction_set=pset,
            acquirable=still_acquirable,
            predicted_benefit=predicted_benefit,
            n_acquisitions_so_far=n_acquisitions_so_far,
        )
        return EvidenceAssessment(
            decision=decision, probability=probs, ignorance=ignorance,
            conflict=conflict,
            per_modality_ignorance={m: float(o.u.squeeze().item()) for m, o in ops.items()},
            modalities_used=sorted(present), mode=self.mode,
        )

    # ------------------------------------------------------------------
    def calibrate(self, probs: np.ndarray, labels: np.ndarray,
                  score: np.ndarray, alpha: float = 0.05, delta: float = 0.10,
                  conformal_alpha: float = 0.10,
                  class_conditional: bool = False) -> Dict:
        """Fit the conformal sets and the LTT threshold on a calibration split.

        The controlled loss is the MALIGNANCY MISS: a malignant lesion predicted
        benign. That is the clinically meaningful risk, and it is bounded in
        {0,1} as the Hoeffding-Bentkus bound requires.
        """
        from src.calibration.conformal import ConformalCalibrator
        from src.calibration.ltt import calibrate_threshold

        mal = set(malignant_indices(self.class_names))
        pred = probs.argmax(1)
        miss = np.array([1.0 if (t in mal and p not in mal) else 0.0
                         for t, p in zip(labels, pred)])

        self.conformal = ConformalCalibrator(
            alpha=conformal_alpha, method="aps",
            class_conditional=class_conditional).fit(probs, labels)

        ltt = calibrate_threshold(score, miss, alpha=alpha, delta=delta,
                                  conditional=True, min_coverage=0.05)
        self.policy.lambda_hat = ltt.lambda_hat
        return {"ltt": ltt.to_dict(),
                "conformal": self.conformal.to_dict(),
                "controlled_risk": "malignancy_miss_rate_on_proceeded",
                "miss_base_rate": float(miss.mean()),
                "is_calibrated": self.is_calibrated}

    def to_dict(self) -> Dict:
        return {"mode": self.mode, "is_calibrated": self.is_calibrated,
                "policy": self.policy.to_dict(),
                "conformal": None if self.conformal is None else self.conformal.to_dict(),
                "acquirable": self.acquirable}
