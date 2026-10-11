"""
src/calibration/conformal.py - Conformal prediction sets (APS / RAPS).

Produces label SETS with marginal coverage >= 1 - alpha, which the decision
policy consumes through one rule:

    if the set contains at least one benign AND at least one malignant class,
    PROCEED is not permitted.

That rule is the reason sets are worth building at all: a point prediction of
"nevus, 0.62" hides the fact that melanoma is still in the plausible set, while
a set {nevus, melanoma} makes the ambiguity explicit and actionable.

References (standard; verify before formal citation):
  Vovk, Gammerman & Shafer - conformal prediction.
  Romano, Sesia & Candès - APS (adaptive prediction sets).
  Angelopoulos, Bates, Malik & Jordan - RAPS (arXiv:2009.14193).
  Vovk - Mondrian / class-conditional conformal prediction.

Validity assumes calibration and test data are EXCHANGEABLE. It does not hold
under distribution shift; measuring the size of the violation under shift is a
deliberate experiment, not a caveat to bury (see docs/05_experimental_protocol.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# scores
# ---------------------------------------------------------------------------

def aps_scores(probs: np.ndarray, labels: Optional[np.ndarray] = None,
               randomize: bool = True, rng: Optional[np.random.Generator] = None
               ) -> np.ndarray:
    """Adaptive-prediction-set nonconformity scores.

    For a label `y`, the score is the cumulative probability mass of all labels
    ranked at least as high as `y`. With `labels` given, returns the (N,) score
    of the true label; otherwise the full (N, K) score matrix.

    `randomize` subtracts a uniform fraction of the true label's own mass, which
    makes coverage exact rather than conservative.
    """
    probs = np.asarray(probs, dtype=float)
    n, k = probs.shape
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    rng = rng or np.random.default_rng(0)

    if labels is None:
        scores = np.empty_like(probs)
        u = rng.random((n, k)) if randomize else np.zeros((n, k))
        contrib = cum - u * sorted_p
        np.put_along_axis(scores, order, contrib, axis=1)
        return scores

    labels = np.asarray(labels).ravel()
    rank = np.argmax(order == labels[:, None], axis=1)
    s = cum[np.arange(n), rank]
    if randomize:
        s = s - rng.random(n) * sorted_p[np.arange(n), rank]
    return s


def raps_scores(probs: np.ndarray, labels: Optional[np.ndarray] = None,
                k_reg: int = 2, lam: float = 0.1, randomize: bool = True,
                rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """RAPS scores: APS plus a penalty on set size.

    The penalty `lam * max(0, rank - k_reg)` discourages long tails, which keeps
    sets small enough to act on. `lam = 0` recovers APS.
    """
    probs = np.asarray(probs, dtype=float)
    n, k = probs.shape
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    ranks = np.arange(1, k + 1)[None, :]
    penalty = lam * np.maximum(0, ranks - k_reg)
    rng = rng or np.random.default_rng(0)

    if labels is None:
        scores = np.empty_like(probs)
        u = rng.random((n, k)) if randomize else np.zeros((n, k))
        np.put_along_axis(scores, order, cum - u * sorted_p + penalty, axis=1)
        return scores

    labels = np.asarray(labels).ravel()
    rank = np.argmax(order == labels[:, None], axis=1)
    s = cum[np.arange(n), rank] + lam * np.maximum(0, (rank + 1) - k_reg)
    if randomize:
        s = s - rng.random(n) * sorted_p[np.arange(n), rank]
    return s


# ---------------------------------------------------------------------------
# calibrator
# ---------------------------------------------------------------------------

@dataclass
class ConformalCalibrator:
    """Split-conformal calibrator for APS/RAPS sets.

    `class_conditional=True` gives Mondrian conformal prediction: a separate
    quantile per true class, so coverage holds within each class instead of only
    on average. That matters when one class is rare - a pooled guarantee lets
    the majority classes absorb the error budget.

    IMPORTANT: a class stratum needs roughly n_k >= 1/alpha calibration points
    for a non-vacuous quantile. `fit` records which classes fall short in
    `underpowered_classes` instead of silently producing a meaningless bound.
    """

    alpha: float = 0.10
    method: str = "aps"            # "aps" | "raps"
    k_reg: int = 2
    lam: float = 0.1
    class_conditional: bool = False
    randomize: bool = True
    seed: int = 42

    qhat: Optional[float] = field(default=None, init=False)
    qhat_per_class: Dict[int, float] = field(default_factory=dict, init=False)
    n_calib: int = field(default=0, init=False)
    underpowered_classes: List[int] = field(default_factory=list, init=False)
    num_classes: int = field(default=0, init=False)

    # -- internals ------------------------------------------------------
    def _score(self, probs, labels=None):
        rng = np.random.default_rng(self.seed)
        if self.method == "raps":
            return raps_scores(probs, labels, self.k_reg, self.lam, self.randomize, rng)
        return aps_scores(probs, labels, self.randomize, rng)

    @staticmethod
    def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
        """The finite-sample corrected (1-alpha) quantile, ceil((n+1)(1-alpha))/n."""
        n = scores.size
        if n == 0:
            return float("inf")
        level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
        return float(np.quantile(scores, level, method="higher"))

    # -- api ------------------------------------------------------------
    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "ConformalCalibrator":
        probs = np.asarray(probs, dtype=float)
        labels = np.asarray(labels).ravel()
        self.num_classes = probs.shape[1]
        self.n_calib = labels.size
        scores = self._score(probs, labels)

        self.qhat = self._conformal_quantile(scores, self.alpha)
        self.qhat_per_class = {}
        self.underpowered_classes = []
        min_needed = int(np.ceil(1.0 / self.alpha)) - 1

        for c in range(self.num_classes):
            m = labels == c
            if m.sum() == 0:
                self.qhat_per_class[c] = self.qhat
                self.underpowered_classes.append(c)
                continue
            if m.sum() < min_needed:
                self.underpowered_classes.append(c)
            self.qhat_per_class[c] = self._conformal_quantile(scores[m], self.alpha)
        return self

    def predict_sets(self, probs: np.ndarray) -> np.ndarray:
        """(N, K) boolean membership matrix."""
        if self.qhat is None:
            raise RuntimeError("ConformalCalibrator.fit must be called first")
        probs = np.asarray(probs, dtype=float)
        all_scores = self._score(probs)

        if not self.class_conditional:
            return all_scores <= self.qhat
        # Class-conditional: a label is included if its score is within ITS own
        # quantile, so each class carries its own error budget.
        thresh = np.array([self.qhat_per_class.get(c, self.qhat)
                           for c in range(probs.shape[1])])[None, :]
        return all_scores <= thresh

    def evaluate(self, probs: np.ndarray, labels: np.ndarray) -> Dict:
        sets = self.predict_sets(probs)
        labels = np.asarray(labels).ravel()
        covered = sets[np.arange(labels.size), labels]
        sizes = sets.sum(axis=1)
        per_class = {}
        for c in range(sets.shape[1]):
            m = labels == c
            if m.sum():
                per_class[int(c)] = {"coverage": float(covered[m].mean()),
                                     "mean_size": float(sizes[m].mean()),
                                     "n": int(m.sum())}
        return {
            "target_coverage": 1.0 - self.alpha,
            "empirical_coverage": float(covered.mean()),
            "mean_set_size": float(sizes.mean()),
            "empty_set_rate": float((sizes == 0).mean()),
            "singleton_rate": float((sizes == 1).mean()),
            "per_class": per_class,
            "underpowered_classes": self.underpowered_classes,
            "n_calib": self.n_calib,
        }

    def to_dict(self) -> Dict:
        return {"alpha": self.alpha, "method": self.method,
                "class_conditional": self.class_conditional,
                "qhat": self.qhat, "qhat_per_class": self.qhat_per_class,
                "n_calib": self.n_calib,
                "underpowered_classes": self.underpowered_classes}


# ---------------------------------------------------------------------------
# the rule the policy actually uses
# ---------------------------------------------------------------------------

def set_is_ambiguous(sets: np.ndarray, malignant_indices: Sequence[int]) -> np.ndarray:
    """True where a prediction set spans both benign and malignant classes.

    This is the conformal gate on PROCEED. An empty set also counts as ambiguous:
    it means no label was conformal at this alpha, which is a reason to stop, not
    a reason to guess.
    """
    sets = np.asarray(sets, dtype=bool)
    mal = np.zeros(sets.shape[1], dtype=bool)
    mal[list(malignant_indices)] = True
    has_mal = sets[:, mal].any(axis=1)
    has_ben = sets[:, ~mal].any(axis=1)
    empty = ~sets.any(axis=1)
    return (has_mal & has_ben) | empty
