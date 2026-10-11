"""
tests/test_orchestrator.py - Pipeline integration tests (DERMA-GUARD v4.0)

Covers the orchestrator's control flow with stubbed agents (fast, no weights),
plus the failure policy: a broken stage must surface as ERROR, never as a
fabricated low-confidence diagnosis.

See tests/test_integration_trained.py for the end-to-end run against the real
trained artifacts.
"""
import pytest
import torch
from unittest.mock import MagicMock

from src.agents.devils_advocate import EvidenceProfile
from src.agents.orchestrator import OrchestratorAgent, PatientCase, PipelineResult
from src.agents.scribe import ScribeReport
from src.agents.vision_team import SpecialistReport
from src.constants import CLASS_NAMES, MODALITY_DIMS, NUM_CLASSES
from src.evidence.missingness import compute_missingness

CODES = ["A", "B", "C", "D", "E", "F", "G"]


def make_vision_report(agreement=4, n_models=5, majority=2):
    preds = {}
    for i, code in enumerate(CODES[:n_models]):
        lg = torch.full((1, NUM_CLASSES), -5.0)
        lg[0, majority if i < agreement else i % NUM_CLASSES] = 5.0
        preds[code] = lg
    modality_logits = {
        "clinical_image": preds.get("A"),
        "metadata": preds.get("B"),
        "text": preds.get("C"),
    }
    return SpecialistReport(
        model_predictions=preds,
        model_embeddings={m: torch.randn(MODALITY_DIMS[m])
                          for m in ("clinical_image", "metadata", "text")},
        model_confidences={c: 0.85 for c in preds},
        uncertainty_estimates={c: 0.3 for c in preds},
        quality_scores={"clinical_image": 0.8},
        modality_logits={k: v for k, v in modality_logits.items() if v is not None},
    )


def make_scribe_report(ptis=0.7):
    return ScribeReport(
        symptoms={"itching": "true", "growth": "unknown"}, ptis=ptis,
        risk_factors=[], summary="Test summary",
        text_embedding=torch.randn(768),
        metadata_embedding=torch.randn(MODALITY_DIMS["metadata"]),
        metadata_quality=0.9,
    )


def make_evidence_profile(ood_score=5.0, conflict=False, consistency=0.85,
                          available=("clinical_image", "metadata", "text")):
    miss = compute_missingness("clinical_image" in available, "dermoscopy_image" in available,
                               "metadata" in available, "text" in available)
    return EvidenceProfile(
        quality={m: 0.8 for m in available},
        reliability={m: 0.8 for m in available},
        utility={m: 0.5 for m in available},
        consistency=consistency, conflict_flag=conflict, conflict_details={},
        missing_modalities=miss["missing_list"], missing_mask=miss["missing_mask"],
        ood_score=ood_score, overall_risk=0.2,
        model_reliability={c: 0.8 for c in CODES},
        uncertainty={c: 0.3 for c in CODES},
    )


def make_orchestrator(**overrides):
    vision = MagicMock(); vision.process_case.return_value = make_vision_report()
    scribe = MagicMock(); scribe.process_case.return_value = make_scribe_report()
    librarian = MagicMock(); librarian.retrieve.return_value = {"similar_cases": []}
    da = MagicMock(); da.profile.return_value = make_evidence_profile()
    orch = OrchestratorAgent(
        vision_team=vision, scribe=scribe, librarian=librarian,
        devils_advocate=da, llm_client=None, class_names=list(CLASS_NAMES),
        **overrides,
    )
    return orch, vision, scribe, da


def full_case(case_id="t"):
    return PatientCase(case_id=case_id, image=torch.randn(1, 3, 224, 224),
                       metadata={"age": 55, "region": "NECK"},
                       text="Lesion on the neck, growing for 3 months, itching.")


class TestFullPipelineIntegration:
    def test_clean_case_returns_decision(self):
        orch, _, _, _ = make_orchestrator()
        result = orch.process_case(full_case("test_001"))
        assert isinstance(result, PipelineResult)
        assert result.case_id == "test_001"
        assert result.action in ("ACCEPT", "ARBITRATE")
        assert result.diagnosis is not None
        assert result.diagnosis_name in CLASS_NAMES

    def test_fusion_actually_runs(self):
        """REGRESSION: fusion must execute, not be swallowed by the orchestrator.

        The old pipeline caught every fusion exception and substituted
        diagnosis=0 / confidence=0.0, so a permanently-broken fusion looked like
        a working-but-unsure one. A real run leaves a `fusion` trace step and
        non-degenerate weights.
        """
        orch, _, _, _ = make_orchestrator()
        result = orch.process_case(full_case())
        steps = [t["step"] for t in result.evidence_trace]
        assert "fusion" in steps
        assert "fusion_error" not in steps
        assert result.fusion_weights
        assert sum(result.fusion_weights.values()) == pytest.approx(1.0, abs=1e-5)

    def test_result_has_evidence_trace(self):
        orch, _, _, _ = make_orchestrator()
        result = orch.process_case(full_case("test_002"))
        assert isinstance(result.evidence_trace, list) and result.evidence_trace
        assert all({"step", "t", "data"} <= set(e) for e in result.evidence_trace)

    def test_specialist_votes_reported(self):
        orch, _, _, _ = make_orchestrator()
        result = orch.process_case(full_case())
        assert result.specialist_votes
        assert all(v in CLASS_NAMES for v in result.specialist_votes.values())


class TestFailurePolicy:
    def test_stage_failure_surfaces_as_error(self):
        """A broken stage must not be laundered into a confident-looking result."""
        orch, vision, _, _ = make_orchestrator()
        vision.process_case.side_effect = RuntimeError("checkpoint missing")
        result = orch.process_case(full_case())
        assert result.action == "ERROR"
        assert result.diagnosis is None
        assert "checkpoint missing" in result.escalate_reason

    def test_librarian_failure_is_non_fatal(self):
        """Retrieval is advisory; losing it must not kill the diagnosis."""
        orch, _, _, _ = make_orchestrator()
        orch.librarian.retrieve.side_effect = RuntimeError("index unavailable")
        result = orch.process_case(full_case())
        assert result.action in ("ACCEPT", "ARBITRATE")
        assert "librarian_skipped" in [t["step"] for t in result.evidence_trace]


class TestOODEscalation:
    def test_ood_case_triggers_escalation(self):
        orch, _, _, da = make_orchestrator()
        da.profile.return_value = make_evidence_profile(ood_score=200.0)
        result = orch.process_case(full_case("ood"))
        assert result.action == "ESCALATE"
        assert result.diagnosis is None


class TestMissingModality:
    def test_no_modality_returns_abstain(self):
        orch, _, _, _ = make_orchestrator()
        result = orch.process_case(PatientCase(case_id="empty"))
        assert result.action == "ABSTAIN"
        assert result.diagnosis is None

    def test_availability_inferred_from_payload(self):
        case = PatientCase(case_id="x", metadata={"age": 40})
        avail = case.availability()
        assert avail["metadata"] is True
        assert avail["clinical_image"] is False
        assert avail["text"] is False
        assert avail["dermoscopy_image"] is False

    def test_explicit_override_wins(self):
        case = PatientCase(case_id="x", metadata={"age": 40}, meta_available=False)
        assert case.availability()["metadata"] is False
