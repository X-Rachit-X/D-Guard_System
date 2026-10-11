"""
src/agents/report.py - Report agent and claim verifier.

Writes the clinical report for every assessment. A deterministic, structured
report is always produced. If a local LLM (Ollama) is reachable it is asked to
rewrite that report as readable prose - and the rewrite is then VERIFIED:

  * every diagnosis it names must be in the differential or among the similar
    cases;
  * every percentage it quotes must match a number in the evidence (+/- 1);
  * it must keep the confidence tier and the "confirm with a doctor" sentence
    whenever the decision requires confirmation.

A rewrite that fails any check is discarded and the deterministic report is
used. The LLM therefore writes; it never diagnoses, and it cannot introduce a
claim the evidence does not support.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

SYSTEM_PROMPT = (
    "You rewrite dermatology decision-support reports for clinicians. Use ONLY the "
    "facts given. Do not add diagnoses, numbers or advice that are not in the input. "
    "Keep every percentage exactly as given. Keep the confidence level and any "
    "instruction to confirm with a doctor. At most 120 words."
)


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * float(x):.0f}%"


class ReportAgent:
    def __init__(self, class_names: Sequence[str], full_names: Dict[str, str],
                 llm=None):
        self.classes = list(class_names)
        self.full = dict(full_names)
        self.llm = llm

    # ---------------------------------------------------------- deterministic
    def structured(self, result, quality: Dict[str, Optional[Dict]],
                   similar: Optional[Dict]) -> str:
        d = result.decision
        name = lambda i: f"{self.full.get(self.classes[i], self.classes[i])} ({self.classes[i]})"  # noqa: E731
        lines: List[str] = []
        if d.action.value == "PROCEED":
            tier_txt = ("HIGH confidence (all safety checks passed)" if d.tier != "LOW"
                        else "LOW confidence (confirmation required)")
            lines.append(f"Assessment: {name(d.prediction)} - {tier_txt}; "
                         f"probability {_pct(d.tentative_confidence)}.")
        elif d.action.value == "ACQUIRE":
            lines.append(f"Provisional: {name(d.tentative)} ({_pct(d.tentative_confidence)}); "
                         f"dermoscopy requested before a decision.")
        else:
            why = {"CONFLICT": "the sources contradict each other",
                   "NO_EVIDENCE": "no usable evidence was provided"}.get(
                       d.reason.value if d.reason else "", "the evidence is insufficient")
            lines.append(f"Deferred to a clinician: {why}. Unconfirmed impression: "
                         f"{name(d.tentative)} ({_pct(d.tentative_confidence)}).")
        if d.differential:
            lines.append("Differential: " + "; ".join(
                f"{self.classes[c]} {_pct(p)}" for c, p in d.differential) + ".")
        lines.append(f"Probability of malignancy: {_pct(d.p_malignant)}"
                     f"{' - URGENT review' if d.urgency == 'URGENT' else ''}.")
        lines.append(f"Evidence: ignorance {result.ignorance:.2f}, conflict "
                     f"{result.conflict:.2f}; weights "
                     + ", ".join(f"{k} {_pct(v)}" for k, v in result.fusion_weights.items())
                     + ".")
        votes = result.modality_votes
        if votes and len(set(votes.values())) > 1:
            lines.append("Sources disagree: " + ", ".join(f"{k} says {v}" for k, v in votes.items()) + ".")
        for kind, q in (quality or {}).items():
            if q and q.get("issues"):
                lines.append(f"Image quality ({kind}): " + ", ".join(q["issues"]) + ".")
        if similar and similar.get("similar_cases"):
            dx = [c["diagnosis"] for c in similar["similar_cases"]]
            counts = {x: dx.count(x) for x in dict.fromkeys(dx)}
            lines.append("Similar confirmed past cases: " + ", ".join(
                f"{k} x{v}" for k, v in counts.items()) + ".")
        if d.requires_confirmation:
            lines.append("Requires confirmation by a clinician.")
        lines.append("Decision support only - not a diagnosis.")
        return "\n".join(lines)

    # --------------------------------------------------------------- verify
    def verify(self, text: str, result, similar: Optional[Dict]) -> List[str]:
        """Return a list of problems; empty means the text is supported."""
        d = result.decision
        problems: List[str] = []
        allowed = {self.classes[c] for c, _ in d.differential}
        if d.tentative is not None:
            allowed.add(self.classes[d.tentative])
        for c in (similar or {}).get("similar_cases", []):
            allowed.add(c["diagnosis"])
        for code in self.classes:
            if re.search(rf"\b{re.escape(code)}\b", text) and code not in allowed:
                problems.append(f"names {code}, which is not in the evidence")
        evidence_numbers = [100 * p for _, p in d.differential]
        evidence_numbers += [100 * (d.p_malignant or 0), 100 * (d.tentative_confidence or 0)]
        evidence_numbers += [100 * v for v in result.fusion_weights.values()]
        for m in re.finditer(r"(\d{1,3}(?:\.\d+)?)\s*%", text):
            v = float(m.group(1))
            if not any(abs(v - e) <= 1.0 for e in evidence_numbers):
                problems.append(f"quotes {v}% which matches no evidence value")
        if d.requires_confirmation and not re.search(r"confirm", text, re.I):
            problems.append("drops the requirement for clinician confirmation")
        if d.tier == "LOW" and not re.search(r"low", text, re.I):
            problems.append("drops the low-confidence label")
        return problems

    # ---------------------------------------------------------------- write
    def write(self, result, quality: Dict[str, Optional[Dict]],
              similar: Optional[Dict]) -> Dict:
        base = self.structured(result, quality, similar)
        if self.llm is None:
            return {"text": base, "source": "structured", "problems": []}
        try:
            prose = self.llm.generate(base, system_prompt=SYSTEM_PROMPT)
        except Exception as e:  # LLM unreachable or timed out
            return {"text": base, "source": "structured", "problems": [f"llm: {e}"]}
        problems = self.verify(prose or "", result, similar)
        if problems or not prose:
            return {"text": base, "source": "structured (LLM rewrite rejected)",
                    "problems": problems or ["empty LLM output"]}
        return {"text": prose.strip(), "source": "llm (verified)", "problems": []}
