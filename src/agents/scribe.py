"""
src/agents/scribe.py - Scribe agent: the patient's own words -> clinical red flags.

From the original DERMA-GUARD design (Scribe + Patient Text Informativeness).
The trained model has no text input - MILK10k ships no narratives - so the Scribe
does NOT influence the diagnosis. It does two things a clinician would:

  1. extract warning signs from what the patient writes (bleeding, rapid growth,
     change in size/shape/colour, a new spot after 50, a non-healing sore,
     itching or pain), using the "E for evolving" logic of ABCDE screening; and
  2. score how informative the text is (PTIS: 0 = says nothing useful, 1 = rich).

Any red flag raises the case to URGENT triage and is shown to the doctor. It can
only ever add caution; it never makes the system more confident.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

RED_FLAGS = {
    "bleeding": r"\b(bleed\w*|blood\w*|ooz\w*)\b",
    "rapid growth": r"\b(grow\w*|grew|bigger|enlarg\w*|spread\w*|getting larger)\b",
    "change": r"\b(chang\w*|darker|different colou?r|new colou?r|shape)\b",
    "non-healing sore": r"\b(not heal\w*|won'?t heal|doesn'?t heal|ulcer\w*|open sore|crust\w*|scab\w*)\b",
    "itching or pain": r"\b(itch\w*|pain\w*|hurt\w*|sore|tender|burn\w*)\b",
    "new spot": r"\b(new|appeared|recently noticed|just noticed)\b",
    # added after scripts/analyze_pad_symptoms.py: on PAD-UFES-20 a raised lesion is
    # malignant (BCC/SCC/MEL) 60% of the time vs 21% for a flat one (RR 2.8)
    "raised lump": r"\b(raised|lump\w*|bump\w*|nodul\w*|swollen)\b",
}
DURATION = r"\b(\d+)\s*(day|week|month|year)s?\b"
NEGATION = r"\b(no|not|never|without|doesn'?t|don'?t|isn'?t|hasn'?t)\s+(\w+\s+){0,2}$"


@dataclass
class ScribeReport:
    red_flags: List[str] = field(default_factory=list)
    duration: Optional[str] = None
    informativeness: float = 0.0
    summary: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


class ScribeAgent:
    def read(self, text: Optional[str], age: Optional[float] = None) -> Optional[ScribeReport]:
        if not text or not text.strip():
            return None
        t = text.lower()
        flags = []
        for name, pat in RED_FLAGS.items():
            for m in re.finditer(pat, t):
                before = t[max(0, m.start() - 25):m.start()]
                if not re.search(NEGATION, before):     # "does not bleed" is not a flag
                    flags.append(name)
                    break
        if "new spot" in flags and not (age and age >= 50):
            flags.remove("new spot")                     # new lesions matter most after 50
        dur = re.search(DURATION, t)
        words = len(re.findall(r"\w+", t))
        info = min(1.0, 0.15 * len(set(flags)) + 0.25 * bool(dur) + min(words, 60) / 120)
        summ = ("Warning signs mentioned: " + ", ".join(flags) + "."
                if flags else "No warning signs mentioned.")
        if dur:
            summ += f" Duration: {dur.group(0)}."
        return ScribeReport(red_flags=flags, duration=dur.group(0) if dur else None,
                            informativeness=round(info, 2), summary=summ)
