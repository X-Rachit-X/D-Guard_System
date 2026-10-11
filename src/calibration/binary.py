"""
src/calibration/binary.py - Conformal sets over {benign, malignant}.

The safety gate asks one question - can malignancy be ruled out? - so the set
it needs is over those two outcomes, not over all K diagnoses.

Score is LAC, s = 1 - p(true outcome) (Sadinle, Lei & Wasserman, JASA 2019),
which yields the smallest sets on average. Non-randomised APS degenerates on two
classes: any lesion whose true outcome the model ranks second scores exactly
1.0, the quantile collapses to 1.0, and every set holds both outcomes. Measured
on MILK10k, that version deferred 100% of test lesions.

Class-conditional (Mondrian): one quantile per outcome. The guarantee that
matters clinically is the malignant one - at least (1 - alpha) of malignant
lesions keep "malignant" in their set - because that is what makes false
reassurance rare.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence

import numpy as np


def binary_probs(probs: np.ndarray, malignant: Sequence[int]) -> np.ndarray:
    """(N, K) class probabilities -> (N, 2) [P(benign), P(malignant)]."""
    probs = np.asarray(probs, dtype=float)
    if probs.ndim == 1:
        probs = probs[None, :]
    pm = probs[:, list(malignant)].sum(1)
    return np.column_stack([1.0 - pm, pm])


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Split-conformal quantile with the finite-sample (n+1) correction."""
    scores = np.asarray(scores, dtype=float)
    n = scores.size
    if n == 0:
        return 1.0
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


@dataclass
class BinarySetCalibrator:
    alpha: float = 0.10
    malignant: Sequence[int] = ()
    num_classes: int = 0
    q: Dict[int, float] = field(default_factory=dict)
    n_per_outcome: Dict[int, int] = field(default_factory=dict)

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "BinarySetCalibrator":
        probs = np.asarray(probs, dtype=float)
        self.num_classes = probs.shape[1]
        b = binary_probs(probs, self.malignant)
        yb = np.isin(np.asarray(labels).ravel(), list(self.malignant)).astype(int)
        for c in (0, 1):
            self.q[c] = conformal_quantile(1.0 - b[yb == c, c], self.alpha)
            self.n_per_outcome[c] = int((yb == c).sum())
        return self

    @property
    def qhat(self) -> float:
        return self.q.get(1, 1.0)

    def binary_sets(self, probs: np.ndarray) -> np.ndarray:
        """(N, 2) boolean [benign in set, malignant in set]."""
        b = binary_probs(probs, self.malignant)
        return np.column_stack([1.0 - b[:, c] <= self.q[c] for c in (0, 1)])

    def predict_sets(self, probs: np.ndarray) -> np.ndarray:
        """(N, K) boolean: the binary set expanded to every class in each outcome."""
        b = self.binary_sets(probs)
        mal = np.zeros(self.num_classes, dtype=bool)
        mal[list(self.malignant)] = True
        out = np.zeros((b.shape[0], self.num_classes), dtype=bool)
        out[b[:, 0]] |= ~mal
        out[b[:, 1]] |= mal
        return out

    def evaluate(self, probs: np.ndarray, labels: np.ndarray) -> Dict:
        b = self.binary_sets(probs)
        yb = np.isin(np.asarray(labels).ravel(), list(self.malignant)).astype(int)
        covered = b[np.arange(len(yb)), yb]
        return {
            "coverage": float(covered.mean()),
            "coverage_malignant": float(covered[yb == 1].mean()) if (yb == 1).any() else None,
            "coverage_benign": float(covered[yb == 0].mean()) if (yb == 0).any() else None,
            "both_in_set": float((b[:, 0] & b[:, 1]).mean()),
            "empty": float((~b[:, 0] & ~b[:, 1]).mean()),
        }

    def to_dict(self) -> Dict:
        return {"alpha": self.alpha, "malignant": list(self.malignant),
                "num_classes": self.num_classes,
                "q": {str(k): v for k, v in self.q.items()},
                "n_per_outcome": {str(k): v for k, v in self.n_per_outcome.items()}}
