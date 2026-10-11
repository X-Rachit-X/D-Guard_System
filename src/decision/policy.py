"""
src/decision/policy.py - PROCEED / ACQUIRE / DEFER by expected-loss minimisation.

Replaces the 4-level consensus ladder, whose thresholds (0.80 / 0.50 / 0.30) were
hand-set, symmetric across error types, and carried no guarantee of any kind.

The rule is Bayes-optimal under a declared cost matrix:

    R_proceed(S)    = min_yhat  sum_y  p_S(y) * L(y, yhat)
    R_acquire(S,m)  = E[ R_proceed(S + m) ] + c_acq(m)
    R_defer         = c_defer
    a*(S)           = argmin over those three

with two safety gates on PROCEED that can only make the system more cautious:

  * CONFLICT gate   - DS conflict mass above its calibrated threshold means the
                      modalities actively contradict each other. Defer to a
                      human; acquiring more evidence does not resolve a
                      contradiction.
  * AMBIGUITY gate  - the conformal set spans benign and malignant classes, so
                      the malignancy question is not settled at this alpha.
  * RISK gate       - confidence below the LTT-calibrated lambda_hat, which is
                      what carries the distribution-free risk bound.

Separating CONFLICT from IGNORANCE is the point of the evidential layer: they
call for different clinical responses (human review vs collect more evidence),
and the old single `consistency` scalar could not tell them apart.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class Action(str, Enum):
    PROCEED = "PROCEED"
    ACQUIRE = "ACQUIRE"
    DEFER = "DEFER"


class DeferReason(str, Enum):
    CONFLICT = "CONFLICT"        # modalities contradict -> human review
    IGNORANCE = "IGNORANCE"      # too little evidence, nothing left to acquire
    AMBIGUOUS_SET = "AMBIGUOUS_SET"   # conformal set spans benign/malignant
    RISK_BOUND = "RISK_BOUND"    # below the LTT-calibrated threshold
    COST = "COST"                # deferring is simply cheaper
    NO_EVIDENCE = "NO_EVIDENCE"  # nothing supplied at all
    ACQUISITION_EXHAUSTED = "ACQUISITION_EXHAUSTED"
    QUALITY = "QUALITY"          # photo outside the calibrated quality range


# ---------------------------------------------------------------------------
# cost model
# ---------------------------------------------------------------------------

@dataclass
class CostMatrix:
    """Asymmetric clinical costs. Every value is a declared, reviewable parameter.

    The point of writing these down is that a reader can disagree with them.
    Hiding them inside three confidence thresholds, as the old ladder did, makes
    the assumption that all errors cost the same - which is false, and was never
    stated.

    Defaults follow the project pre-registration template; a sensitivity sweep
    over `missed_malignant` is mandatory because conclusions depend on it.
    """

    correct: float = 0.0
    wrong_same_group: float = 1.0      # e.g. BCC called SCC: both malignant
    missed_malignant: float = 10.0     # malignant called benign - the bad one
    false_alarm: float = 2.0           # benign called malignant
    acquire: Dict[str, float] = field(default_factory=lambda: {"dermoscopy_image": 0.5})
    defer: float = 1.5

    def loss(self, y_true: int, y_pred: int, malignant: Sequence[int]) -> float:
        if y_true == y_pred:
            return self.correct
        mal = set(malignant)
        t_mal, p_mal = y_true in mal, y_pred in mal
        if t_mal and not p_mal:
            return self.missed_malignant
        if p_mal and not t_mal:
            return self.false_alarm
        return self.wrong_same_group

    def loss_matrix(self, num_classes: int, malignant: Sequence[int]) -> np.ndarray:
        """(K, K) matrix L[y_true, y_pred]."""
        return np.array([[self.loss(t, p, malignant) for p in range(num_classes)]
                         for t in range(num_classes)], dtype=float)

    def normalised_miss_loss(self) -> float:
        """missed_malignant scaled to [0,1] - the bounded loss LTT requires."""
        worst = max(self.missed_malignant, self.false_alarm,
                    self.wrong_same_group, 1e-9)
        return self.missed_malignant / worst

    def to_dict(self) -> Dict:
        return {"correct": self.correct, "wrong_same_group": self.wrong_same_group,
                "missed_malignant": self.missed_malignant,
                "false_alarm": self.false_alarm, "acquire": dict(self.acquire),
                "defer": self.defer}


def expected_loss(probs: np.ndarray, loss_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Bayes risk of each candidate prediction.

    Returns (best_action_loss (N,), best_prediction (N,)).
    """
    probs = np.asarray(probs, dtype=float)
    risks = probs @ loss_matrix          # (N, K): risk of predicting each label
    return risks.min(axis=1), risks.argmin(axis=1)


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    action: Action
    prediction: Optional[int] = None
    expected_loss: Optional[float] = None
    reason: Optional[DeferReason] = None
    acquire_target: Optional[str] = None
    risks: Dict[str, float] = field(default_factory=dict)
    gates: Dict[str, bool] = field(default_factory=dict)
    confidence: Optional[float] = None
    ignorance: Optional[float] = None
    conflict: Optional[float] = None
    prediction_set: Optional[List[int]] = None
    # ---- tentative opinion ------------------------------------------------
    # Present on EVERY decision, including DEFER and ACQUIRE. Deferring means
    # "do not issue an autonomous diagnosis", not "say nothing": a reviewing
    # clinician is better served by the system's best guess, its differential
    # and its malignancy estimate - clearly marked UNCONFIRMED - than by
    # silence. `prediction` stays None unless the action is PROCEED, so no
    # caller can mistake a tentative opinion for an issued one.
    tentative: Optional[int] = None
    tentative_confidence: Optional[float] = None
    differential: List[Tuple[int, float]] = field(default_factory=list)
    p_malignant: Optional[float] = None
    urgency: Optional[str] = None          # "URGENT" | "ROUTINE"
    #: Confidence tier of a PROCEED answer.
    #:   "HIGH"  every safety gate passed - the calibrated, guaranteed answer
    #:   "LOW"   an answer given in minimal-defer mode although a gate fired or
    #:           deferral was cheaper; ALWAYS shown as low confidence and
    #:           ALWAYS requires a clinician's confirmation
    tier: Optional[str] = None
    #: Why a LOW-tier answer is low confidence (the gate or cost reason that
    #: would have deferred it in standard mode).
    low_reason: Optional["DeferReason"] = None

    @property
    def requires_confirmation(self) -> bool:
        return self.action != Action.PROCEED or self.tier == "LOW"

    @property
    def is_autonomous(self) -> bool:
        """True only for a HIGH-tier answer: the system may state it on its own."""
        return self.action == Action.PROCEED and self.tier != "LOW"

    def to_dict(self) -> Dict:
        return {
            "action": self.action.value,
            "prediction": self.prediction,
            "expected_loss": self.expected_loss,
            "reason": self.reason.value if self.reason else None,
            "acquire_target": self.acquire_target,
            "risks": self.risks, "gates": self.gates,
            "confidence": self.confidence, "ignorance": self.ignorance,
            "conflict": self.conflict, "prediction_set": self.prediction_set,
            "tentative": self.tentative,
            "tentative_confidence": self.tentative_confidence,
            "differential": [[int(c), float(q)] for c, q in self.differential],
            "p_malignant": self.p_malignant,
            "urgency": self.urgency,
            "autonomous": self.is_autonomous,
            "tier": self.tier,
            "low_reason": self.low_reason.value if self.low_reason else None,
            "requires_confirmation": self.requires_confirmation,
        }


@dataclass
class ThreeActionPolicy:
    """Deterministic, versioned proceed/acquire/defer policy.

    Thresholds come from calibration, not from hand-tuning:
      lambda_hat        <- src.calibration.ltt.calibrate_threshold
      conflict_threshold<- validation quantile of DS conflict mass
      prediction sets   <- src.calibration.conformal.ConformalCalibrator
    """

    cost: CostMatrix = field(default_factory=CostMatrix)
    malignant_classes: Sequence[int] = (0, 4, 5)   # BCC, SCC, MEL for PAD-UFES ordering
    num_classes: int = 6
    lambda_hat: Optional[float] = None
    conflict_threshold: float = 0.5
    #: Triage flag for the HUMAN REVIEW QUEUE only - it orders deferred cases,
    #: it never changes an action. A deferred case is URGENT when the
    #: probability of malignancy reaches this level or a malignant class is
    #: still inside the conformal prediction set. Declared, not calibrated.
    urgent_malignancy_threshold: float = 0.30
    differential_size: int = 3
    #: How the stated label is chosen.
    #:   "bayes"  the label with the lowest expected cost (Chow's rule). With an
    #:            asymmetric cost matrix this SHIFTS the stated diagnosis toward
    #:            malignant classes: on MILK10k (73.6% malignant, miss:false-alarm
    #:            10:2) it calls 98.5% of lesions malignant, so "never misses a
    #:            cancer" becomes true for a trivial reason.
    #:   "argmax" the most probable diagnosis is what gets stated; the cost
    #:            matrix then decides only WHETHER the system may state it alone,
    #:            via the PROCEED-vs-DEFER comparison and the safety gates. This
    #:            keeps the diagnosis honest and puts the caution where it belongs.
    label_rule: str = "bayes"
    #: Which ambiguous prediction sets block PROCEED.
    #:   "symmetric"    any set holding both a benign and a malignant class, or
    #:                  an empty set. Blocks "likely cancer" as hard as "likely
    #:                  harmless" - with a weak model that defers nearly
    #:                  everything (measured: 100% on MILK10k).
    #:   "reassurance"  block only when the system would state a BENIGN label
    #:                  while a malignant class is still in the set (or the set
    #:                  is empty). The gate exists to stop false reassurance;
    #:                  stating a malignant diagnosis while benign is still
    #:                  possible errs in the safe direction - the patient is
    #:                  sent onward either way - and its false-alarm cost is
    #:                  already priced by the cost matrix.
    ambiguity_gate: str = "symmetric"
    #: "standard": any blocked case is DEFERRED.
    #: "minimal":  a blocked case is ANSWERED in the LOW tier - explicitly low
    #:             confidence, confirmation required - instead of deferred.
    #:             Only contradiction between sources (conflict) and the absence
    #:             of any evidence still DEFER: a contradiction has no single best
    #:             answer to state, and no evidence has none at all.
    #: The HIGH tier is identical in both modes, so the calibrated guarantees
    #: apply to exactly the same answers either way.
    defer_mode: str = "standard"
    max_acquisitions: int = 1
    version: str = "eade-v3.0"

    def __post_init__(self):
        self._L = self.cost.loss_matrix(self.num_classes, self.malignant_classes)

    # ------------------------------------------------------------------
    def decide(
        self,
        probs: np.ndarray,
        *,
        ignorance: Optional[float] = None,
        conflict: Optional[float] = None,
        prediction_set: Optional[np.ndarray] = None,
        acquirable: Sequence[str] = (),
        predicted_benefit: Optional[Dict[str, float]] = None,
        n_acquisitions_so_far: int = 0,
        confidence: Optional[float] = None,
    ) -> Decision:
        """Decide for ONE case.

        Args:
            probs:            (K,) fused posterior (pignistic probability).
            ignorance:        DS ignorance mass u in [0,1].
            conflict:         DS conflict mass C in [0,1].
            prediction_set:   (K,) boolean conformal membership.
            acquirable:       modality names that could still be obtained.
            predicted_benefit: expected loss REDUCTION per acquirable modality.
            confidence:       score compared against lambda_hat.
                              Defaults to 1 - ignorance when available, else max prob.
        """
        p = np.asarray(probs, dtype=float).ravel()
        if p.size != self.num_classes:
            raise ValueError(f"probs has {p.size} entries, expected {self.num_classes}")
        if p.sum() <= 0:
            return Decision(action=Action.DEFER, reason=DeferReason.NO_EVIDENCE,
                            urgency="ROUTINE")
        p = p / p.sum()

        if confidence is None:
            confidence = (1.0 - float(ignorance)) if ignorance is not None else float(p.max())

        label_risks = p @ self._L          # expected cost of stating each label
        if self.label_rule == "argmax":
            pred = int(np.argmax(p))
        elif self.label_rule == "bayes":
            pred = int(np.argmin(label_risks))
        else:
            raise ValueError(f"label_rule must be 'bayes' or 'argmax', got {self.label_rule!r}")
        r_proceed = float(label_risks[pred])

        # --- acquisition option -------------------------------------------
        r_acquire, best_m = np.inf, None
        if n_acquisitions_so_far < self.max_acquisitions:
            for m in acquirable:
                benefit = (predicted_benefit or {}).get(m, 0.0)
                cand = (r_proceed - float(benefit)) + self.cost.acquire.get(m, 0.5)
                if cand < r_acquire:
                    r_acquire, best_m = cand, m

        r_defer = float(self.cost.defer)
        risks = {"proceed": r_proceed, "defer": r_defer}
        if best_m is not None:
            risks["acquire"] = float(r_acquire)

        # --- safety gates on PROCEED --------------------------------------
        gate_conflict = conflict is not None and float(conflict) > self.conflict_threshold
        gate_ambiguous = False
        pset: Optional[List[int]] = None
        if prediction_set is not None:
            s = np.asarray(prediction_set, dtype=bool).ravel()
            pset = [int(i) for i in np.flatnonzero(s)]
            mal = set(self.malignant_classes)
            has_mal = any(i in mal for i in pset)
            has_ben = any(i not in mal for i in pset)
            if self.ambiguity_gate == "reassurance":
                states_benign = pred not in mal
                gate_ambiguous = (states_benign and has_mal) or len(pset) == 0
            elif self.ambiguity_gate == "symmetric":
                gate_ambiguous = (has_mal and has_ben) or len(pset) == 0
            else:
                raise ValueError(f"ambiguity_gate must be 'symmetric' or "
                                 f"'reassurance', got {self.ambiguity_gate!r}")
        gate_risk = self.lambda_hat is not None and confidence < self.lambda_hat

        gates = {"conflict": gate_conflict, "ambiguous_set": gate_ambiguous,
                 "below_risk_threshold": gate_risk}
        blocked = gate_conflict or gate_ambiguous or gate_risk

        mal_idx = [i for i in self.malignant_classes if i < p.size]
        p_mal = float(p[mal_idx].sum()) if mal_idx else 0.0
        order = np.argsort(-p)[: self.differential_size]
        mal_in_set = pset is not None and any(i in set(mal_idx) for i in pset)
        urgency = ("URGENT" if (p_mal >= self.urgent_malignancy_threshold or mal_in_set)
                   else "ROUTINE")

        common = dict(risks=risks, gates=gates, confidence=float(confidence),
                      ignorance=None if ignorance is None else float(ignorance),
                      conflict=None if conflict is None else float(conflict),
                      prediction_set=pset,
                      tentative=pred, tentative_confidence=float(p[pred]),
                      differential=[(int(i), float(p[i])) for i in order],
                      p_malignant=p_mal, urgency=urgency)

        # Conflict is never resolved by collecting more evidence - a
        # contradiction needs a human. It therefore short-circuits ACQUIRE.
        if gate_conflict:
            return Decision(action=Action.DEFER, reason=DeferReason.CONFLICT, **common)

        if blocked:
            if best_m is not None and r_acquire < r_defer:
                return Decision(action=Action.ACQUIRE, acquire_target=best_m,
                                expected_loss=float(r_acquire),
                                reason=(DeferReason.AMBIGUOUS_SET if gate_ambiguous
                                        else DeferReason.RISK_BOUND), **common)
            reason = (DeferReason.AMBIGUOUS_SET if gate_ambiguous else
                      DeferReason.RISK_BOUND if gate_risk else DeferReason.IGNORANCE)
            if n_acquisitions_so_far >= self.max_acquisitions and not acquirable:
                reason = DeferReason.ACQUISITION_EXHAUSTED
            if self.defer_mode == "minimal":
                return Decision(action=Action.PROCEED, prediction=pred, tier="LOW",
                                low_reason=reason, expected_loss=r_proceed, **common)
            return Decision(action=Action.DEFER, reason=reason, **common)

        # --- unblocked: plain expected-loss comparison --------------------
        if best_m is not None and r_acquire < min(r_proceed, r_defer):
            return Decision(action=Action.ACQUIRE, acquire_target=best_m,
                            expected_loss=float(r_acquire), **common)
        if r_defer < r_proceed:
            if self.defer_mode == "minimal":
                return Decision(action=Action.PROCEED, prediction=pred, tier="LOW",
                                low_reason=DeferReason.COST, expected_loss=r_proceed,
                                **common)
            return Decision(action=Action.DEFER, reason=DeferReason.COST, **common)
        return Decision(action=Action.PROCEED, prediction=pred, tier="HIGH",
                        expected_loss=r_proceed, **common)

    # ------------------------------------------------------------------
    def decide_batch(self, probs: np.ndarray, **kw) -> List[Decision]:
        """Vectorised convenience wrapper; per-case arrays are indexed positionally."""
        probs = np.asarray(probs, dtype=float)
        n = probs.shape[0]

        def at(key, i):
            v = kw.get(key)
            if v is None:
                return None
            arr = np.asarray(v)
            return arr[i] if arr.ndim and arr.shape[0] == n else v

        out = []
        for i in range(n):
            out.append(self.decide(
                probs[i],
                ignorance=at("ignorance", i), conflict=at("conflict", i),
                prediction_set=(None if kw.get("prediction_set") is None
                                else np.asarray(kw["prediction_set"])[i]),
                acquirable=kw.get("acquirable", ()),
                predicted_benefit=kw.get("predicted_benefit"),
                confidence=at("confidence", i),
            ))
        return out

    def to_dict(self) -> Dict:
        return {"version": self.version, "cost": self.cost.to_dict(),
                "malignant_classes": list(self.malignant_classes),
                "lambda_hat": self.lambda_hat,
                "conflict_threshold": self.conflict_threshold,
                "max_acquisitions": self.max_acquisitions}


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------

def confidence_threshold_baseline(probs: np.ndarray, threshold: float) -> np.ndarray:
    """B3: accept iff max softmax >= threshold. Returns a boolean accept mask."""
    return np.asarray(probs, dtype=float).max(axis=1) >= threshold


def entropy_acquisition_baseline(probs: np.ndarray, threshold: float) -> np.ndarray:
    """B5, HOT-style: request the extra modality iff predictive entropy > threshold.

    Baseline for the published precedent that an AI can recommend dermoscopy when
    the clinical image is insufficient (Mehta et al., JEADV 2025). What this
    project adds on top is the cost model, the risk guarantee, and the stress
    protocol - not the idea of asking.
    """
    p = np.asarray(probs, dtype=float)
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(axis=1)
    return ent > threshold
