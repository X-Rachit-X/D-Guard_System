"""src/llm/report_gen.py — Clinical report generation from evidence bundle."""
from __future__ import annotations

import json
import warnings
from typing import Dict, List, Optional

from src.llm.prompts import (
    REPORT_GEN_SYSTEM, synthesis_prompt,
    CONSERVATIVE_REPORT_PROMPT, conservative_prompt,
)


def generate_report(
    vision_report: Dict,
    scribe_report: Dict,
    knowledge: Optional[Dict],
    evidence_profile: Dict,
    consensus: Dict,
    llm_client=None,
    class_names: Optional[List[str]] = None,
    conservative: bool = False,
) -> str:
    """Generate a clinical report from the evidence bundle.

    Args:
        vision_report:    VisionReport dict or dataclass
        scribe_report:    ScribeReport dict or dataclass
        knowledge:        RAG retrieval result dict
        evidence_profile: EvidenceProfile dict
        consensus:        ConsensusResult dict
        llm_client:       OllamaClient instance
        class_names:      list of class name strings
        conservative:     if True, use conservative prompt

    Returns:
        Report text string
    """
    # Convert dataclasses to dicts for prompt building
    def _to_dict(obj):
        if hasattr(obj, "__dataclass_fields__"):
            import dataclasses
            d = dataclasses.asdict(obj)
            # Convert tensors
            for k, v in d.items():
                if hasattr(v, "tolist"):
                    d[k] = v.tolist()
            return d
        return obj if isinstance(obj, dict) else {}

    vr = _to_dict(vision_report)
    sr = _to_dict(scribe_report)
    ep = _to_dict(evidence_profile)
    cr = _to_dict(consensus)

    if llm_client is None:
        return _fallback_report(cr, ep, sr, class_names)

    if conservative:
        evidence_json = json.dumps(ep, indent=2, default=str)
        try:
            return llm_client.generate(
                conservative_prompt(evidence_json),
                system_prompt=CONSERVATIVE_REPORT_PROMPT,
                temperature=0.0,
            )
        except Exception as e:
            warnings.warn(f"[ReportGen] Conservative report generation failed: {e}")
            return _fallback_report(cr, ep, sr)

    prompt = synthesis_prompt(vr, sr, knowledge, ep, cr, class_names=class_names)
    try:
        return llm_client.generate(prompt, system_prompt=REPORT_GEN_SYSTEM, temperature=0.0)
    except Exception as e:
        warnings.warn(f"[ReportGen] Report generation failed: {e}. Using fallback.")
        return _fallback_report(cr, ep, sr)


def _fallback_report(consensus: Dict, evidence: Dict, scribe: Dict,
                     class_names: Optional[List[str]] = None) -> str:
    """Deterministic factual report, used whenever no LLM is reachable.

    Every number here is copied straight from the evidence bundle, so the report
    stays traceable under stage-2 rule 3. ASCII only: the report is written to
    files and consoles whose encoding we do not control.
    """
    from src.constants import CLASS_FULL_NAMES, CLASS_NAMES

    names = class_names or list(CLASS_NAMES)
    action = consensus.get("action", "ESCALATE")
    diag = consensus.get("diagnosis", None)
    conf = consensus.get("confidence", 0.0) or 0.0
    missing = evidence.get("missing_modalities", []) or []
    weights = evidence.get("fusion_weights", {}) or {}
    votes = evidence.get("specialist_votes", {}) or {}

    lines = ["DERMA-GUARD v4.0 - Automated Report",
             f"Consensus Action: {action} (level {evidence.get('consensus_level', '?')})"]

    if action in ("ACCEPT", "ARBITRATE") and diag is not None:
        label = names[diag] if isinstance(diag, int) and 0 <= diag < len(names) else str(diag)
        full = CLASS_FULL_NAMES.get(label, label)
        lines.append(f"Predicted Diagnosis: {label} - {full} (confidence: {conf:.1%})")
    elif action == "ACQUIRE":
        lines.append("No diagnosis issued: additional evidence requested.")
    elif action in ("ESCALATE", "ABSTAIN"):
        lines.append("No diagnosis issued: the case was referred for human review.")

    if votes:
        agree = {}
        for v in votes.values():
            agree[v] = agree.get(v, 0) + 1
        summary = ", ".join(f"{k}={v}/{len(votes)}" for k, v in
                            sorted(agree.items(), key=lambda kv: -kv[1]))
        lines.append(f"Specialist agreement: {summary}")

    active = {k: v for k, v in weights.items() if v > 0.0}
    if active:
        lines.append("Evidence weighting: " + ", ".join(
            f"{k} {v:.0%}" for k, v in sorted(active.items(), key=lambda kv: -kv[1])))

    consistency = evidence.get("consistency")
    if isinstance(consistency, (int, float)):
        lines.append(f"Cross-modal consistency: {consistency:.0%}")

    if missing:
        lines.append(f"Missing evidence: {', '.join(missing)}")

    lines.append("Note: no LLM was reachable, so this is the deterministic "
                 "evidence summary rather than a generated narrative.")
    lines.append("Research prototype - not clinically validated. Do not use for diagnosis.")
    return "\n".join(lines)
