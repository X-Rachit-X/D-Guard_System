"""
src/decision/explain.py - Turn a Decision into plain-English messages.

Two audiences, two wordings, one decision:

  patient   short, calm, no jargon, always says what to DO next
  doctor    the tentative opinion, the differential, the evidence numbers and
            the exact reason the system did not answer on its own

The rule that holds for both: a tentative opinion is always labelled
UNCONFIRMED, and only a PROCEED decision is ever phrased as a result. On DEFER
the system still says what it thinks - deferring means "a human must confirm
this", not "I have nothing to say".
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from src.decision.policy import Action, Decision, DeferReason

REASON_TEXT = {
    DeferReason.CONFLICT: "the different kinds of evidence point to different diagnoses",
    DeferReason.IGNORANCE: "there is not enough evidence to be sure",
    DeferReason.AMBIGUOUS_SET: ("the possible diagnoses include both harmless and "
                                "cancerous conditions, so it cannot be ruled out safely"),
    DeferReason.RISK_BOUND: "confidence is below the level certified as safe",
    DeferReason.COST: "the risk of a wrong answer outweighs the cost of a human review",
    DeferReason.NO_EVIDENCE: "no usable evidence was provided",
    DeferReason.ACQUISITION_EXHAUSTED: ("it is still uncertain after the extra evidence "
                                        "was added"),
    DeferReason.QUALITY: ("the photo quality is outside the range the system was "
                          "checked on"),
    DeferReason.OUT_OF_DISTRIBUTION: ("this spot looks unlike the cases the system "
                                      "learned from"),
    DeferReason.MISSING_DETAILS: ("only a photo was given - without your age, where the "
                                  "spot is, or a dermoscopy image, the system is less "
                                  "reliable"),
}

ACQUIRE_TEXT = {
    "dermoscopy_image": "a dermoscopy photo (a close-up taken with a special skin lens)",
    "dermoscopy": "a dermoscopy photo (a close-up taken with a special skin lens)",
    "metadata": "a few details: age, where the spot is, and any symptoms",
    "text": "a short description of how the spot has changed",
    "clinical_image": "a clear, well-lit photo of the spot",
}


def _name(i: Optional[int], class_names: Sequence[str],
          full_names: Optional[Dict[str, str]]) -> str:
    if i is None:
        return "unknown"
    code = class_names[i]
    if full_names and code in full_names:
        return f"{full_names[code]} ({code})"
    return code


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def differential_lines(d: Decision, class_names: Sequence[str],
                       full_names: Optional[Dict[str, str]] = None) -> List[str]:
    return [f"{_name(c, class_names, full_names)} - {_pct(q)}" for c, q in d.differential]


def explain_patient(d: Decision, class_names: Sequence[str],
                    full_names: Optional[Dict[str, str]] = None) -> str:
    """What the person with the skin lesion reads."""
    lines: List[str] = []
    if d.action == Action.PROCEED and d.tier == "LOW":
        lines.append(f"Low-confidence assessment: "
                     f"{_name(d.prediction, class_names, full_names)} "
                     f"({_pct(d.tentative_confidence)} likely).")
        why = REASON_TEXT.get(d.low_reason, "the evidence is not strong enough")
        lines.append(f"This is NOT reliable enough to act on by itself, because {why}.")
        if d.urgency == "URGENT":
            lines.append("IMPORTANT: cancer has not been ruled out. Please see a doctor "
                         "SOON to confirm.")
        else:
            lines.append("Please have a doctor confirm it.")
        lines.append("A doctor has been asked to review your case.")
        return "\n".join(lines)

    if d.action == Action.PROCEED:
        lines.append(f"Result: the system's assessment is "
                     f"{_name(d.prediction, class_names, full_names)}.")
        if d.prediction is not None and d.urgency == "URGENT":
            lines.append("This looks like a condition that needs a specialist. Please "
                         "book an appointment with a dermatologist soon - it is "
                         "usually very treatable when handled early.")
        elif (d.p_malignant or 0) >= 0.10:
            lines.append("Because some cancer risk remains, a doctor should still "
                         "examine the spot.")
        else:
            lines.append("This looks low-risk. Keep an eye on it and take a new photo "
                         "if it changes in size, shape or colour.")
        lines.append("This is a decision-support result, not a final diagnosis. Your "
                     "doctor will confirm it.")
        return "\n".join(lines)

    if d.action == Action.ACQUIRE:
        want = ACQUIRE_TEXT.get(d.acquire_target or "", d.acquire_target or "more information")
        lines.append("The system needs one more piece of information before it can give "
                     "an assessment.")
        lines.append(f"Next step: please provide {want}.")
        if d.tentative is not None:
            lines.append(f"Early, unconfirmed impression: "
                         f"{_name(d.tentative, class_names, full_names)}.")
        return "\n".join(lines)

    # DEFER - still says what it thinks
    why = REASON_TEXT.get(d.reason, "it is not certain enough")
    if d.tentative is not None:
        lines.append(f"Unconfirmed impression: "
                     f"{_name(d.tentative, class_names, full_names)} "
                     f"(low confidence, {_pct(d.tentative_confidence)}).")
    lines.append(f"The system will not decide this on its own because {why}.")
    if d.urgency == "URGENT":
        lines.append("IMPORTANT: please see a doctor SOON. Cancer has not been ruled "
                     "out for this spot.")
    else:
        lines.append("Please have a doctor look at this spot to confirm.")
    lines.append("Your case has been sent to a doctor for review.")
    return "\n".join(lines)


def explain_doctor(d: Decision, class_names: Sequence[str],
                   full_names: Optional[Dict[str, str]] = None) -> str:
    """What the reviewing clinician reads."""
    head = {
        Action.PROCEED: ("PROCEED (LOW CONFIDENCE) - answer given, confirmation required"
                         if d.tier == "LOW" else
                         "PROCEED (HIGH CONFIDENCE) - system-issued assessment"),
        Action.ACQUIRE: "ACQUIRE - additional evidence requested",
        Action.DEFER: "DEFER - human decision required",
    }[d.action]
    lines = [head, "-" * len(head)]
    if d.action == Action.PROCEED:
        lines.append(f"Assessment      : {_name(d.prediction, class_names, full_names)}"
                     f"  [{d.tier or 'HIGH'} confidence]")
        if d.tier == "LOW" and d.low_reason is not None:
            lines.append(f"Low because     : {d.low_reason.value} - "
                         f"{REASON_TEXT.get(d.low_reason, '')}")
    else:
        lines.append(f"Tentative (UNCONFIRMED): "
                     f"{_name(d.tentative, class_names, full_names)} "
                     f"@ {_pct(d.tentative_confidence)}")
    if d.differential:
        lines.append("Differential    : " + "; ".join(
            differential_lines(d, class_names, full_names)))
    lines.append(f"P(malignant)    : {_pct(d.p_malignant)}")
    if d.urgency:
        lines.append(f"Triage          : {d.urgency}")
    if d.prediction_set is not None:
        ps = d.prediction_set
        if len(ps) == len(class_names):
            lines.append("Prediction set  : all classes - cancer can be neither ruled "
                         "in nor ruled out")
        elif len(ps) == 0:
            lines.append("Prediction set  : empty - no class is plausible enough")
        else:
            lines.append("Prediction set  : {" + ", ".join(class_names[i] for i in ps) + "}")
    lines.append(f"Ignorance u     : {d.ignorance:.3f}" if d.ignorance is not None
                 else "Ignorance u     : n/a")
    lines.append(f"Conflict C      : {d.conflict:.3f}" if d.conflict is not None
                 else "Conflict C      : n/a")
    if d.reason is not None:
        lines.append(f"Reason          : {d.reason.value} - "
                     f"{REASON_TEXT.get(d.reason, '')}")
    if d.acquire_target:
        lines.append(f"Requested       : {d.acquire_target}")
    fired = [k for k, v in (d.gates or {}).items() if v]
    lines.append("Safety gates    : " + (", ".join(fired) if fired else "none fired"))
    return "\n".join(lines)
