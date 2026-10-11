"""
tests/test_tentative_opinion.py - DEFER still says what the system thinks.

Deferring means "a human must confirm this", not "say nothing". Every decision
carries a tentative opinion, but only PROCEED may ever be phrased as a result.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.decision.explain import explain_doctor, explain_patient
from src.decision.policy import Action, DeferReason, ThreeActionPolicy

NAMES = ["BCC", "ACK", "NEV", "SEK", "SCC", "MEL"]
MAL = (0, 4, 5)


def policy(**kw):
    return ThreeActionPolicy(malignant_classes=MAL, num_classes=6, **kw)


def confident(cls=2, p=0.94):
    v = np.full(6, (1 - p) / 5)
    v[cls] = p
    return v


def test_proceed_has_prediction_and_matching_tentative():
    d = policy().decide(confident(2), ignorance=0.05, conflict=0.0,
                        prediction_set=np.eye(6, dtype=bool)[2])
    assert d.action == Action.PROCEED
    assert d.prediction == d.tentative == 2
    assert d.is_autonomous


def test_defer_carries_a_tentative_opinion_but_no_prediction():
    d = policy(conflict_threshold=0.2).decide(confident(5, 0.6), ignorance=0.1,
                                              conflict=0.9)
    assert d.action == Action.DEFER
    assert d.reason == DeferReason.CONFLICT
    assert d.prediction is None, "a deferred case must never carry an issued answer"
    assert d.tentative == 5
    assert d.tentative_confidence == pytest.approx(0.6, abs=1e-6)
    assert not d.is_autonomous


def test_differential_is_sorted_and_sized():
    p = np.array([0.40, 0.25, 0.15, 0.10, 0.06, 0.04])
    d = policy().decide(p, ignorance=0.5, conflict=0.0)
    probs = [q for _, q in d.differential]
    assert len(d.differential) == 3
    assert probs == sorted(probs, reverse=True)
    assert d.differential[0][0] == 0


def test_p_malignant_sums_malignant_classes():
    p = np.array([0.30, 0.20, 0.20, 0.10, 0.10, 0.10])
    d = policy().decide(p, ignorance=0.5)
    assert d.p_malignant == pytest.approx(0.30 + 0.10 + 0.10)


def test_urgent_when_malignancy_in_prediction_set():
    pset = np.zeros(6, dtype=bool)
    pset[[2, 5]] = True                      # NEV and MEL both live
    d = policy().decide(confident(2, 0.85), ignorance=0.1, conflict=0.0,
                        prediction_set=pset)
    assert d.action != Action.PROCEED
    assert d.urgency == "URGENT"


def test_routine_when_clearly_benign():
    d = policy().decide(confident(2, 0.97), ignorance=0.05, conflict=0.0,
                        prediction_set=np.eye(6, dtype=bool)[2])
    assert d.urgency == "ROUTINE"


def test_patient_message_on_defer_states_opinion_and_urges_review():
    pset = np.zeros(6, dtype=bool)
    pset[[2, 5]] = True
    d = policy().decide(confident(2, 0.85), ignorance=0.1, conflict=0.0,
                        prediction_set=pset)
    msg = explain_patient(d, NAMES)
    assert "Unconfirmed impression" in msg or "unconfirmed impression" in msg
    assert "NEV" in msg
    assert "doctor" in msg.lower()
    assert "SOON" in msg


def test_patient_message_on_proceed_is_a_result():
    d = policy().decide(confident(2), ignorance=0.05, conflict=0.0,
                        prediction_set=np.eye(6, dtype=bool)[2])
    msg = explain_patient(d, NAMES)
    assert msg.startswith("Result:")


def test_doctor_message_labels_tentative_unconfirmed():
    d = policy(conflict_threshold=0.2).decide(confident(5, 0.6), ignorance=0.1,
                                              conflict=0.9)
    msg = explain_doctor(d, NAMES)
    assert "UNCONFIRMED" in msg
    assert "conflict" in msg.lower()
    assert "Differential" in msg


def test_to_dict_exposes_tentative_fields():
    d = policy().decide(confident(1, 0.5), ignorance=0.6)
    out = d.to_dict()
    for k in ("tentative", "differential", "p_malignant", "urgency", "autonomous"):
        assert k in out
