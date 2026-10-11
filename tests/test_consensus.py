"""
tests/test_consensus.py - Consensus hierarchy tests (DERMA-GUARD v4.0)

Covers the Level 1-4 boundaries, empty evidence, escalation under conflict, and
the exhaustiveness property: every reachable input must land on some level.

These use real dataclasses rather than MagicMock. Mocks previously let
`max(weighted_votes, ...)` compare two MagicMocks, which raised TypeError the
moment arbitration was genuinely exercised.
"""
import pytest
import torch

from src.agents.devils_advocate import EvidenceProfile
from src.agents.vision_team import SpecialistReport
from src.constants import NUM_CLASSES
from src.evidence.missingness import mask_from_available
from src.fusion.adaptive_fusion import FusionResult
from src.fusion.consensus import classify_consensus

ALL_CODES = ["A", "B", "C", "D", "E", "F", "G"]


def make_vision_report(agreement=3, n_models=5, majority_class=2):
    """SpecialistReport where `agreement` specialists vote for majority_class."""
    predictions = {}
    for i, code in enumerate(ALL_CODES[:n_models]):
        logits = torch.full((1, NUM_CLASSES), -5.0)
        cls = majority_class if i < agreement else (majority_class + i + 1) % NUM_CLASSES
        logits[0, cls] = 5.0
        predictions[code] = logits
    return SpecialistReport(
        model_predictions=predictions,
        model_embeddings={},
        model_confidences={c: 0.9 for c in predictions},
        uncertainty_estimates={c: 0.2 for c in predictions},
        quality_scores={},
    )


def make_evidence_profile(quality=None, conflict_flag=False, consistency=0.9,
                          model_reliability=None, missing=None):
    missing = missing or ["dermoscopy_image"]
    present = [m for m in ("clinical_image", "metadata", "text") if m not in missing]
    return EvidenceProfile(
        quality=quality or {"clinical_image": 0.8, "metadata": 0.9, "text": 0.6},
        reliability={m: 0.8 for m in present},
        utility={m: 0.5 for m in present},
        consistency=consistency,
        conflict_flag=conflict_flag,
        conflict_details={},
        missing_modalities=missing,
        missing_mask=mask_from_available(present),
        ood_score=0.0,
        overall_risk=0.3,
        model_reliability=model_reliability or {c: 0.8 for c in ALL_CODES},
    )


def make_fusion_result(confidence=0.90, diagnosis=2):
    return FusionResult(
        prediction=torch.zeros(1, NUM_CLASSES), diagnosis=diagnosis,
        confidence=confidence, weights=mask_from_available(("metadata",)),
        fused_embedding=torch.zeros(8), iterations_used=1, converged=True,
    )


class TestLevel1Accept:
    def test_strong_consensus_returns_level1(self):
        r = classify_consensus(make_fusion_result(0.92), make_evidence_profile(),
                               make_vision_report(agreement=4))
        assert r.level == 1 and r.action == "ACCEPT"
        assert r.diagnosis == 2

    def test_low_confidence_does_not_level1(self):
        """Below 0.80 must not ACCEPT - it should arbitrate instead."""
        r = classify_consensus(make_fusion_result(0.60), make_evidence_profile(),
                               make_vision_report(agreement=4))
        assert r.level != 1
        assert r.action == "ARBITRATE"

    def test_low_quality_blocks_level1(self):
        q = {"clinical_image": 0.1, "metadata": 0.9, "text": 0.6}
        r = classify_consensus(make_fusion_result(0.95),
                               make_evidence_profile(quality=q),
                               make_vision_report(agreement=4))
        assert r.level != 1


class TestLevel2Arbitrate:
    def test_moderate_consensus_returns_level2(self):
        r = classify_consensus(make_fusion_result(0.65), make_evidence_profile(),
                               make_vision_report(agreement=2))
        assert r.level == 2 and r.action == "ARBITRATE"
        assert r.diagnosis is not None

    def test_strong_agreement_moderate_confidence_arbitrates(self):
        """REGRESSION: the spec's `== 2` left this case with no level at all.

        6-of-7 agreement at confidence 0.79 previously fell past every branch to
        the Level 4 'unresolvable disagreement' escalation.
        """
        r = classify_consensus(make_fusion_result(0.79), make_evidence_profile(),
                               make_vision_report(agreement=6, n_models=7))
        assert r.action == "ARBITRATE" and r.level == 2

    def test_arbitration_weights_by_quality_and_reliability(self):
        """The winning vote should follow evidence weight, not raw count."""
        preds = {}
        for code, cls in [("A", 0), ("B", 1), ("C", 1)]:
            lg = torch.full((1, NUM_CLASSES), -5.0); lg[0, cls] = 5.0
            preds[code] = lg
        vr = SpecialistReport(model_predictions=preds, model_embeddings={},
                              model_confidences={}, uncertainty_estimates={},
                              quality_scores={})
        # B and C read metadata/text, which are near-worthless here; A reads a
        # pristine image, so A's minority vote should win arbitration.
        ep = make_evidence_profile(
            quality={"clinical_image": 1.0, "metadata": 0.02, "text": 0.02},
            model_reliability={"A": 1.0, "B": 0.02, "C": 0.02})
        r = classify_consensus(make_fusion_result(0.70), ep, vr)
        assert r.action == "ARBITRATE"
        assert r.diagnosis == 0


class TestLevel3Acquire:
    def test_no_agreement_triggers_acquire(self):
        r = classify_consensus(make_fusion_result(0.40), make_evidence_profile(),
                               make_vision_report(agreement=1, n_models=3))
        assert r.action == "ACQUIRE" and r.level == 3

    def test_low_quality_triggers_acquire(self):
        q = {"clinical_image": 0.1, "metadata": 0.2, "text": 0.15}
        r = classify_consensus(make_fusion_result(0.40),
                               make_evidence_profile(quality=q),
                               make_vision_report(agreement=1, n_models=3))
        assert r.action == "ACQUIRE"


class TestLevel4Escalate:
    def test_critical_conflict_escalates(self):
        r = classify_consensus(
            make_fusion_result(0.40),
            make_evidence_profile(conflict_flag=True, consistency=0.1),
            make_vision_report(agreement=1, n_models=3))
        assert r.action == "ESCALATE" and r.level == 4

    def test_no_predictions_escalates(self):
        vr = SpecialistReport(model_predictions={}, model_embeddings={},
                              model_confidences={}, uncertainty_estimates={},
                              quality_scores={})
        r = classify_consensus(make_fusion_result(0.9), make_evidence_profile(), vr)
        assert r.action == "ESCALATE" and r.level == 4


class TestExhaustiveness:
    @pytest.mark.parametrize("agreement", [1, 2, 3, 5, 7])
    @pytest.mark.parametrize("confidence", [0.10, 0.45, 0.55, 0.79, 0.85, 0.99])
    def test_every_input_reaches_a_level(self, agreement, confidence):
        """No combination may fall through the ladder into a spurious verdict."""
        r = classify_consensus(make_fusion_result(confidence),
                               make_evidence_profile(),
                               make_vision_report(agreement=agreement, n_models=7))
        assert r.level in (1, 2, 3, 4)
        assert r.action in ("ACCEPT", "ARBITRATE", "ACQUIRE", "ESCALATE")
        # A verdict that names a diagnosis must actually carry one.
        if r.action in ("ACCEPT", "ARBITRATE"):
            assert r.diagnosis is not None

    def test_high_agreement_never_reports_unresolvable(self):
        for confidence in (0.51, 0.60, 0.70, 0.79, 0.80, 0.95):
            r = classify_consensus(make_fusion_result(confidence),
                                   make_evidence_profile(),
                                   make_vision_report(agreement=6, n_models=7))
            assert r.reason != "Unresolvable disagreement between models", (
                f"6/7 agreement at confidence {confidence} must not be called unresolvable")
