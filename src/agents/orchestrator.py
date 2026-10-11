"""
src/agents/orchestrator.py - Orchestrator Agent (DERMA-GUARD v4.0).

7-step pipeline:
  1. Encode the case once, then run the agents (specialists, scribe, librarian,
     devil's advocate) over that shared encoding
  2. Stage 1 verification (pre-fusion gating)
  3. Dynamic fusion (DiagnosisAdaptiveFusion over modalities)
  4. 4-level consensus hierarchy
  5. Report generation (LLM)
  6. Stage 2 verification (rule + LLM, conservative retry)
  7. Log and return

CRITICAL CONSTRAINT: the Orchestrator uses structured tool-calling only. The LLM
writes the report; it never decides the diagnosis.

FAILURE POLICY: a broken stage produces an explicit ERROR/ESCALATE result naming
the stage. It never degrades to a fabricated prediction. The previous version
caught fusion exceptions and substituted `diagnosis=0, confidence=0.0`, which
made a hard-failing pipeline look like a working one that was merely unsure -
and let the whole test suite pass while fusion never ran. See
docs/evaluation_integrity.md.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from src.constants import CLASS_NAMES, MODALITIES
from src.fusion.acquisition import AcquisitionPolicy
from src.fusion.adaptive_fusion import DiagnosisAdaptiveFusion, InsufficientEvidenceError
from src.fusion.consensus import ConsensusResult, classify_consensus
from src.llm.report_gen import generate_report
from src.llm.verification import stage1_verify, stage2_verify


# ---------------------------------------------------------------------------
# PatientCase
# ---------------------------------------------------------------------------

@dataclass
class PatientCase:
    """Unified input to the Orchestrator pipeline."""
    case_id: str
    image: Optional[Image.Image | torch.Tensor] = None
    metadata: Optional[Dict] = None
    text: Optional[str] = None
    #: PAD-UFES-20 carries no dermoscopy, so this is present for interface
    #: completeness and is always treated as unavailable unless supplied.
    dermoscopy_image: Optional[Image.Image | torch.Tensor] = None

    # Explicit overrides; when None, availability is inferred from the payload.
    clinical_available: Optional[bool] = None
    dermo_available: Optional[bool] = None
    meta_available: Optional[bool] = None
    text_available: Optional[bool] = None

    def availability(self) -> Dict[str, bool]:
        """Resolve which modalities this case actually carries."""
        return {
            "clinical_image": (self.image is not None
                               if self.clinical_available is None else self.clinical_available),
            "dermoscopy_image": (self.dermoscopy_image is not None
                                 if self.dermo_available is None else self.dermo_available),
            "metadata": (bool(self.metadata)
                         if self.meta_available is None else self.meta_available),
            "text": (bool(self.text and self.text.strip())
                     if self.text_available is None else self.text_available),
        }


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Structured output from OrchestratorAgent.process_case()."""
    case_id: str
    action: str                        # ACCEPT|ARBITRATE|ACQUIRE|ESCALATE|ABSTAIN|ERROR
    diagnosis: Optional[int] = None
    diagnosis_name: Optional[str] = None
    confidence: Optional[float] = None
    consensus_level: Optional[int] = None
    report: Optional[str] = None
    evidence_used: List[str] = field(default_factory=list)
    evidence_missing: List[str] = field(default_factory=list)
    fusion_weights: Dict[str, float] = field(default_factory=dict)
    specialist_votes: Dict[str, str] = field(default_factory=dict)
    acquire_target: Optional[str] = None
    escalate_reason: Optional[str] = None
    #: What fusion would have answered, recorded even when the safety layer
    #: declined to issue it. This is what makes "did deferring actually avoid an
    #: error?" measurable instead of assumed - see src/stress/benchmark.py.
    fused_diagnosis: Optional[int] = None
    fused_confidence: Optional[float] = None
    #: Similar prior cases shown to the report generator, and the fraction of
    #: them sharing the issued diagnosis. EXPLANATION ONLY - neither value
    #: enters the decision, and both are surfaced so a reader can audit what
    #: the narrative was allowed to draw on.
    similar_cases: List[Dict] = field(default_factory=list)
    retrieval_agreement: Optional[float] = None
    evidence_trace: List[Dict] = field(default_factory=list)
    processing_time_s: float = 0.0
    model_version: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "case_id": self.case_id,
            "action": self.action,
            "diagnosis": self.diagnosis,
            "diagnosis_name": self.diagnosis_name,
            "confidence": self.confidence,
            "consensus_level": self.consensus_level,
            "report": self.report,
            "evidence_used": self.evidence_used,
            "similar_cases": self.similar_cases,
            "retrieval_agreement": self.retrieval_agreement,
            "evidence_missing": self.evidence_missing,
            "fusion_weights": self.fusion_weights,
            "specialist_votes": self.specialist_votes,
            "acquire_target": self.acquire_target,
            "escalate_reason": self.escalate_reason,
            "fused_diagnosis": self.fused_diagnosis,
            "fused_confidence": self.fused_confidence,
            "processing_time_s": self.processing_time_s,
            "model_version": self.model_version,
        }


def _consensus_from_assessment(assessment, class_names) -> ConsensusResult:
    """Adapt an EADE assessment onto the ConsensusResult the report layer expects.

    The level field is retained only for display continuity: 1 = PROCEED,
    3 = ACQUIRE, 4 = DEFER. The decision itself comes from the expected-loss
    rule and its calibrated gates, not from the old thresholds.
    """
    d = assessment.decision
    action_map = {"PROCEED": ("ACCEPT", 1), "ACQUIRE": ("ACQUIRE", 3),
                  "DEFER": ("ESCALATE", 4)}
    action, level = action_map[d.action.value]
    reason = None
    if d.reason is not None:
        reason = (f"{d.reason.value} (u={assessment.ignorance:.2f}, "
                  f"C={assessment.conflict:.2f})")
    return ConsensusResult(
        level=level, action=action,
        diagnosis=d.prediction,
        confidence=d.confidence,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# OrchestratorAgent
# ---------------------------------------------------------------------------

def _retrieval_agreement(knowledge, diagnosis):
    """Fraction of retrieved neighbours sharing `diagnosis`.

    Returns None when there was no retrieval or no diagnosis, so "not measured"
    is distinguishable from "measured as zero agreement".
    """
    cases = (knowledge or {}).get("similar_cases") or []
    if not cases or diagnosis is None:
        return None
    from src.retrieval.rag_index import proportion_sharing_predicted_diagnosis
    return round(proportion_sharing_predicted_diagnosis(cases, int(diagnosis)), 4)


class OrchestratorAgent:
    """Central coordinator for the DERMA-GUARD v4.0 pipeline."""

    def __init__(self, vision_team=None, scribe=None, librarian=None,
                 devils_advocate=None, fusion_model: Optional[DiagnosisAdaptiveFusion] = None,
                 acquisition_policy: Optional[AcquisitionPolicy] = None,
                 llm_client=None, class_names: Optional[List[str]] = None,
                 model_version: Optional[str] = None, backends=None,
                 strict: bool = True, eade=None):
        """
        Args:
            strict: when True (default) an unexpected failure in a pipeline stage
                is surfaced as an ERROR result. Set False only for batch jobs that
                must keep going; the failure is still recorded in the trace.
        """
        self.backends = backends
        self.vision_team = vision_team
        self.scribe = scribe
        self.librarian = librarian
        self.devils_advocate = devils_advocate
        self.fusion = fusion_model or DiagnosisAdaptiveFusion()
        self.acquisition = acquisition_policy or AcquisitionPolicy()
        self.llm = llm_client
        self.class_names = class_names or list(CLASS_NAMES)
        self.model_version = model_version
        self.strict = strict
        # When present, the Evidence-Adaptive Decision Engine replaces the
        # 4-level consensus ladder: PROCEED/ACQUIRE/DEFER from an expected-loss
        # rule with calibrated safety gates. The ladder stays reachable
        # (eade=None) because "heuristic ladder vs calibrated rule" is one of
        # the experimental comparisons.
        self.eade = eade

        self.tools = {
            "vision_specialist": self._call_vision_team,
            "scribe": self._call_scribe,
            "librarian": self._call_librarian,
            "devils_advocate": self._call_evidence_profiler,
        }

    # ------------------------------------------------------------------
    @classmethod
    def build(cls, llm_client=None, weights_dir=None, use_eade: bool = True,
              rag_index_path: str | None = "outputs/rag/index_train",
              **kwargs) -> "OrchestratorAgent":
        """Assemble a fully-wired orchestrator against the trained artifacts."""
        from src.agents.devils_advocate import DevilsAdvocateAgent
        from src.agents.scribe import ScribeAgent
        from src.agents.vision_team import SpecialistTeam
        from src.models.backends import get_backends

        backends = get_backends(weights_dir=weights_dir)
        eade = None
        if use_eade:
            from src.agents.eade import EvidenceAdaptiveEngine
            eade = EvidenceAdaptiveEngine(mode="bridge")

        # Similar-case retrieval, attached only if an index has actually been
        # built. It stays None otherwise rather than installing an agent that
        # answers "index not loaded" on every case, which is what used to
        # happen. Build one with: python scripts/build_rag_index.py --verify
        librarian = kwargs.pop("librarian", None)
        if librarian is None and rag_index_path is not None:
            from pathlib import Path as _Path

            from src.agents.librarian import LibrarianAgent
            if _Path(str(rag_index_path)).with_suffix(".index").exists():
                librarian = LibrarianAgent(index_path=str(rag_index_path))

        return cls(
            backends=backends,
            vision_team=SpecialistTeam(backends=backends).load(),
            scribe=ScribeAgent(llm_client=llm_client, backends=backends),
            devils_advocate=DevilsAdvocateAgent(),
            librarian=librarian,
            llm_client=llm_client,
            eade=eade,
            **kwargs,
        )

    # ------------------------------------------------------------------
    def process_case(self, case: PatientCase) -> PipelineResult:
        """Execute the 7-step pipeline for one case."""
        t0 = time.time()
        trace: List[Dict] = []

        def _log(step: str, data: Any) -> None:
            trace.append({"step": step, "t": round(time.time() - t0, 4), "data": str(data)[:500]})

        def _fail(stage: str, exc: Exception) -> PipelineResult:
            _log(f"{stage}_error", repr(exc))
            return PipelineResult(
                case_id=case.case_id, action="ERROR",
                escalate_reason=f"{stage} failed: {exc}",
                evidence_trace=trace, processing_time_s=time.time() - t0,
                model_version=self.model_version,
            )

        avail = case.availability()
        _log("start", avail)

        if not any(avail.values()):
            _log("abstain", "no modality supplied")
            return PipelineResult(
                case_id=case.case_id, action="ABSTAIN",
                escalate_reason="No evidence available: the case carries no modality.",
                evidence_missing=list(MODALITIES),
                evidence_trace=trace, processing_time_s=time.time() - t0,
                model_version=self.model_version,
            )

        # ============================ STEP 1 ============================
        try:
            encoding = None
            if self.backends is not None:
                encoding = self.backends.encode_case(
                    image=case.image if avail["clinical_image"] else None,
                    metadata=case.metadata if avail["metadata"] else None,
                    text=case.text if avail["text"] else None,
                )
                if encoding.notes:
                    _log("encoding_notes", encoding.notes)

            vision_report = self.vision_team.process_case(
                image=case.image if avail["clinical_image"] else None,
                metadata=case.metadata if avail["metadata"] else None,
                text=case.text if avail["text"] else None,
                encoding=encoding,
            )
            _log("specialists", {"ran": sorted(vision_report.model_predictions),
                                 "notes": vision_report.notes})
        except Exception as exc:
            if self.strict:
                return _fail("specialist_team", exc)
            return _fail("specialist_team", exc)

        try:
            scribe_report = self.scribe.process_case(
                text=case.text or "",
                metadata=case.metadata if avail["metadata"] else None,
                metadata_vector=encoding.metadata if encoding is not None else None,
            )
            _log("scribe", {"ptis": round(scribe_report.ptis, 3),
                            "meta_quality": round(scribe_report.metadata_quality, 3)})
        except Exception as exc:
            return _fail("scribe", exc)

        knowledge = None
        if self.librarian is not None:
            try:
                knowledge = self._call_librarian(vision_report, scribe_report)
                _log("librarian", {"similar_cases": len(knowledge.get("similar_cases", []))})
            except Exception as exc:
                # Retrieval is advisory: losing it degrades the report, not the
                # diagnosis, so this is the one stage that may fail soft.
                _log("librarian_skipped", repr(exc))

        try:
            evidence_profile = self.devils_advocate.profile(
                vision_report=vision_report, scribe_report=scribe_report,
                clinical_available=avail["clinical_image"],
                dermo_available=avail["dermoscopy_image"],
                meta_available=avail["metadata"],
                text_available=avail["text"],
            )
            _log("devils_advocate", {"risk": round(evidence_profile.overall_risk, 3),
                                     "C": round(evidence_profile.consistency, 3),
                                     "conflict": evidence_profile.conflict_flag})
        except Exception as exc:
            return _fail("evidence_profiler", exc)

        # ============================ STEP 2 ============================
        v1 = stage1_verify(evidence_profile)
        if not v1.passed:
            reason = v1.reason or "stage 1 verification failed"
            action = ("ESCALATE" if "Out-of-distribution" in reason
                      else "ABSTAIN" if "No evidence" in reason else "ACQUIRE")
            _log("stage1_fail", reason)
            return PipelineResult(
                case_id=case.case_id, action=action, escalate_reason=reason,
                evidence_missing=evidence_profile.missing_modalities,
                evidence_trace=trace, processing_time_s=time.time() - t0,
                model_version=self.model_version,
            )
        _log("stage1", "pass")

        # ============================ STEP 3 ============================
        try:
            fusion_result = self.fusion.fuse(
                embeddings=vision_report.model_embeddings,
                quality=evidence_profile.quality,
                reliability=evidence_profile.reliability,
                utility=evidence_profile.utility,
                consistency=evidence_profile.consistency,
                missing_mask=evidence_profile.missing_mask,
                modality_logits=vision_report.modality_logits,
            )
            _log("fusion", fusion_result.to_dict())
        except InsufficientEvidenceError as exc:
            _log("fusion_abstain", str(exc))
            return PipelineResult(
                case_id=case.case_id, action="ABSTAIN", escalate_reason=str(exc),
                evidence_missing=evidence_profile.missing_modalities,
                evidence_trace=trace, processing_time_s=time.time() - t0,
                model_version=self.model_version,
            )
        except Exception as exc:
            return _fail("fusion", exc)

        # ============================ STEP 4 ============================
        if self.eade is not None:
            assessment = self.eade.assess(
                vision_report,
                available=[m for m, ok in avail.items() if ok])
            consensus = _consensus_from_assessment(assessment, self.class_names)
            _log("eade", assessment.to_dict())
        else:
            consensus = classify_consensus(fusion_result, evidence_profile, vision_report)
        _log("consensus", consensus.to_dict())

        votes = {c: self.class_names[d] for c, d in vision_report.top1().items()}
        common = dict(
            case_id=case.case_id,
            fused_diagnosis=fusion_result.diagnosis,
            fused_confidence=fusion_result.confidence,
            evidence_used=[m for m, ok in avail.items() if ok],
            evidence_missing=evidence_profile.missing_modalities,
            fusion_weights=fusion_result.weight_dict(),
            specialist_votes=votes,
            # Retrieval is surfaced on every exit path, including ACQUIRE and
            # ESCALATE, so an auditor can see what the narrative drew on even
            # when no diagnosis was issued. Trimmed: the full narratives are
            # large and the index metadata already holds them.
            similar_cases=[
                {k: v for k, v in c.items() if k != "narrative"}
                for c in (knowledge or {}).get("similar_cases", [])
            ],
            # Recomputed HERE, not in the Librarian. The Librarian runs before
            # consensus, so at retrieval time there is no diagnosis to agree
            # with and its own agreement_rate is structurally 0.0. Computing it
            # once the diagnosis exists is the only way the number means
            # anything.
            retrieval_agreement=_retrieval_agreement(knowledge, fusion_result.diagnosis),
            evidence_trace=trace,
            model_version=self.model_version,
        )

        if consensus.action == "ACQUIRE":
            target = self._select_acquisition(case, avail, evidence_profile, scribe_report)
            _log("acquire", target)
            return PipelineResult(action="ACQUIRE", acquire_target=target,
                                  escalate_reason=consensus.reason,
                                  consensus_level=consensus.level,
                                  processing_time_s=time.time() - t0, **common)

        if consensus.action == "ESCALATE":
            _log("escalate", consensus.reason)
            return PipelineResult(action="ESCALATE", escalate_reason=consensus.reason,
                                  consensus_level=consensus.level,
                                  processing_time_s=time.time() - t0, **common)

        # ============================ STEP 5 ============================
        evidence_dict = evidence_profile.to_dict()
        evidence_dict.update({
            "consensus_action": consensus.action,
            "consensus_level": consensus.level,
            "confidence": consensus.confidence,
            "fusion_weights": fusion_result.weight_dict(),
            "specialist_votes": votes,
        })

        try:
            report = generate_report(
                vision_report=vision_report, scribe_report=scribe_report,
                knowledge=knowledge, evidence_profile=evidence_dict,
                consensus=consensus.to_dict(), llm_client=self.llm,
                class_names=self.class_names,
            )
            _log("report", f"{len(report)} chars")
        except Exception as exc:
            return _fail("report_generation", exc)

        # ============================ STEP 6 ============================
        v2 = stage2_verify(report, evidence_dict, self.llm)
        if not v2.passed:
            _log("stage2_fail", v2.reason)
            report = generate_report(
                vision_report=vision_report, scribe_report=scribe_report,
                knowledge=knowledge, evidence_profile=evidence_dict,
                consensus=consensus.to_dict(), llm_client=self.llm,
                class_names=self.class_names, conservative=True,
            )
            retry = stage2_verify(report, evidence_dict, self.llm)
            if not retry.passed:
                _log("stage2_escalate", retry.reason)
                return PipelineResult(
                    action="ESCALATE",
                    escalate_reason=f"Report failed verification twice: {retry.reason}",
                    consensus_level=consensus.level,
                    processing_time_s=time.time() - t0, **common)
            _log("stage2", "conservative report passed")
        else:
            _log("stage2", "pass")

        # ============================ STEP 7 ============================
        diag = consensus.diagnosis
        result = PipelineResult(
            action=consensus.action,
            diagnosis=diag,
            diagnosis_name=self.class_names[diag] if diag is not None else None,
            confidence=consensus.confidence,
            consensus_level=consensus.level,
            report=report,
            processing_time_s=time.time() - t0,
            **common,
        )
        _log("done", {"action": result.action, "diagnosis": result.diagnosis_name})
        return result

    # ------------------------------------------------------------------
    def _select_acquisition(self, case, avail, evidence_profile, scribe_report) -> str:
        """Ask the acquisition policy what to request next."""
        unc = evidence_profile.uncertainty or {}
        features = {
            "current_entropy": (sum(unc.values()) / len(unc)) if unc else 0.5,
            "missing_count": len(evidence_profile.missing_modalities),
            "metadata_completeness": evidence_profile.quality.get("metadata", 0.0),
            "ptis": scribe_report.ptis,
            "ood_score": evidence_profile.ood_score,
            "consistency": evidence_profile.consistency,
        }
        try:
            return self.acquisition.select_action(
                features, [m for m, ok in avail.items() if ok])
        except Exception as exc:
            warnings.warn(f"[Orchestrator] acquisition policy failed: {exc}")
            missing = evidence_profile.missing_modalities
            return missing[0] if missing else "ask_growth"

    # -- tool wrappers ---------------------------------------------------
    def _call_vision_team(self, case: PatientCase):
        avail = case.availability()
        return self.vision_team.process_case(
            image=case.image if avail["clinical_image"] else None,
            metadata=case.metadata if avail["metadata"] else None,
            text=case.text if avail["text"] else None,
        )

    def _call_scribe(self, case: PatientCase):
        return self.scribe.process_case(text=case.text or "", metadata=case.metadata)

    def _call_librarian(self, vision_report, scribe_report):
        """Query the similar-case index.

        Uses the RAW pre-projection features, which is the space
        scripts/build_rag_index.py indexes. The projected 256-d embeddings would
        be the wrong width and FAISS would reject them - which is the failure we
        want, rather than a silent nearest-neighbour lookup in the wrong space.

        Metadata is deliberately absent from the key: the preprocessor emits 116
        columns at runtime but 114 when fitted on the training split only, so it
        cannot be part of a runtime-queryable index. See SPECS in
        scripts/build_rag_index.py.
        """
        emb = vision_report.model_embeddings
        return self.librarian.retrieve({
            "clinical_embedding": emb.get("clinical_image_raw"),
            "text_embedding": emb.get("text_raw"),
        })

    def _call_evidence_profiler(self, case: PatientCase, vision_report, scribe_report):
        avail = case.availability()
        return self.devils_advocate.profile(
            vision_report=vision_report, scribe_report=scribe_report,
            clinical_available=avail["clinical_image"],
            dermo_available=avail["dermoscopy_image"],
            meta_available=avail["metadata"],
            text_available=avail["text"],
        )
