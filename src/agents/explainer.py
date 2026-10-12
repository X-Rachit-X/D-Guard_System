"""
src/agents/explainer.py - Why do the specialists disagree, and who was trusted?

Turns the Devil's Advocate's numbers for ONE case into plain sentences:

  1. who said what              each specialist's own top diagnosis and how sure it was
  2. why they disagree          for every specialist that disagrees with the team, the
                                MEASURED reasons: it is rarely right for this diagnosis
                                on validation lesions, its opinion is spread out, the
                                photo is poor, it cannot see the lesion (details only),
                                it was trained on another dataset (PAD-UFES Model A), ...
  3. where the weight went      the largest fusion weights and what earned them
  4. what the confidence means  calibrated confidence, ignorance and consistency

Every sentence is built from numbers already in the result - nothing is invented.
An optional local LLM may rewrite the text for readability; the rewrite is only
used if every percentage and diagnosis it mentions appears in the facts (the same
claim verifier as the report agent), otherwise the structured text is kept.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

WHAT_IT_SEES = {
    "clinical-photo head": "the ordinary photo",
    "dermoscopy head": "the dermoscopy image (skin-surface microscopy)",
    "patient-details head": "only age, sex, body site and skin tone - it never sees the spot",
    "PAD-UFES Model A (original system)": "the photo, but it was trained on a different "
                                          "dataset (PAD-UFES-20) and knows only 6 of the 11 "
                                          "diagnoses",
    "similar-case vote": "the 15 most similar confirmed training lesions",
    "embedding classifier": "a class-balanced boundary drawn through all training lesions",
}

FEATURE_TEXT = {
    "log Q": "photo quality", "log R": "learned reliability", "uncertainty U": "uncertainty",
    "confidence": "confidence", "margin": "margin between its top two answers",
    "agreement": "agreement with the others", "log utility": "track record for the "
    "suspected diagnosis", "consistency C": "overall consistency",
    "missingness M": "missing inputs", "OOD distance": "how unusual the case is",
}

SYSTEM_PROMPT = (
    "Rewrite the following explanation for a doctor in clear, short paragraphs. "
    "Do not add any diagnosis, number or fact that is not in the text. Keep every "
    "percentage exactly as written.")


def _pct(x) -> str:
    return f"{100 * float(x):.0f}%"


class EvidenceExplainer:
    def __init__(self, class_names: Sequence[str], full_names: Dict[str, str], llm=None):
        self.classes = list(class_names)
        self.full = full_names
        self.llm = llm

    def _name(self, c: str) -> str:
        return f"{self.full.get(c, c)} ({c})"

    # ------------------------------------------------------------- structured
    def facts(self, team: Dict, decision) -> Dict:
        if not team or not team.get("votes"):
            return {}
        final = self.classes[decision.tentative] if decision.tentative is not None else None
        probs = team.get("probs") or {}
        out = {"final": final, "who": [], "dissent": [], "weight": [], "confidence": []}
        for name, vote in team["votes"].items():
            p = probs.get(name)
            own = max(p) if p else None
            out["who"].append({
                "specialist": name, "says": vote, "own_confidence": own,
                "weight": team["weights"].get(name), "reliability": team["reliability"].get(name),
                "quality": team["quality"].get(name), "uncertainty": team["uncertainty"].get(name),
                "track_record": (team.get("utility") or {}).get(name),
                "p_final": (p[self.classes.index(final)] if p and final in self.classes else None),
                "sees": WHAT_IT_SEES.get(name, "")})
        for w in out["who"]:
            if final is None or w["says"] == final:
                continue
            reasons = []
            if w["track_record"] is not None and w["track_record"] < 0.4:
                reasons.append(f"on validation lesions that were truly {final}, it was right "
                               f"only {_pct(w['track_record'])} of the time")
            if w["uncertainty"] is not None and w["uncertainty"] > 0.9:
                reasons.append("its opinion is spread over many diagnoses (uncertainty "
                               f"{w['uncertainty']:.2f} of 1)")
            if w["quality"] is not None and w["quality"] < 0.75 and "details" not in w["specialist"]:
                reasons.append(f"its input looks unusual (quality {w['quality']:.2f})")
            if "details" in w["specialist"]:
                reasons.append("it cannot see the spot at all - it only knows what is common "
                               "for this age and body site")
            if "PAD-UFES" in w["specialist"]:
                reasons.append("it was trained on Brazilian smartphone photos and cannot name "
                               "5 of the 11 diagnoses")
            if "similar-case" in w["specialist"]:
                reasons.append("the most similar past cases had a different diagnosis")
            if w["reliability"] is not None and w["reliability"] < 0.4:
                reasons.append(f"its learned reliability for this case is low "
                               f"({_pct(w['reliability'])})")
            if w["p_final"] is not None and w["p_final"] > 0.15:
                reasons.append(f"it still gives {final} {_pct(w['p_final'])}, so it is not "
                               f"strongly against it")
            out["dissent"].append({"specialist": w["specialist"], "says": w["says"],
                                   "reasons": reasons or ["no single measured reason - "
                                                          "a genuine difference of opinion"]})
        ranked = sorted(out["who"], key=lambda w: -(w["weight"] or 0))
        for w in ranked[:3]:
            why = []
            if (w["reliability"] or 0) >= 0.6:
                why.append(f"high learned reliability ({_pct(w['reliability'])})")
            if (w["track_record"] or 0) >= 0.6:
                why.append(f"a good track record for {final} ({_pct(w['track_record'])})")
            if w["says"] == final:
                why.append("it agrees with the team")
            if (w["own_confidence"] or 0) >= 0.5:
                why.append(f"it is fairly sure ({_pct(w['own_confidence'])})")
            out["weight"].append({"specialist": w["specialist"], "weight": w["weight"],
                                  "why": why})
        imp = team.get("gate_importance") or {}
        if imp:
            top = sorted(imp.items(), key=lambda kv: -kv[1])[:3]
            out["gate_listens_to"] = [FEATURE_TEXT.get(k, k) for k, _ in top]
        out["method"] = team.get("method")
        out["consistency"] = team.get("consistency")
        out["calibrated_confidence"] = decision.tentative_confidence
        return out

    def structured(self, f: Dict) -> str:
        if not f:
            return ""
        L: List[str] = []
        L.append("WHO SAID WHAT")
        for w in f["who"]:
            own = f" ({_pct(w['own_confidence'])} sure)" if w["own_confidence"] is not None else ""
            L.append(f"- {w['specialist']}: {self._name(w['says'])}{own}. Looks at {w['sees']}.")
        if f["dissent"]:
            L.append("")
            L.append(f"WHY SOME DISAGREE WITH {self._name(f['final'])}")
            for d in f["dissent"]:
                L.append(f"- {d['specialist']} says {d['says']}: " + "; ".join(d["reasons"]) + ".")
        else:
            L.append("")
            L.append(f"All specialists agree on {self._name(f['final'])}.")
        L.append("")
        L.append("WHERE THE WEIGHT WENT")
        for w in f["weight"]:
            why = (" - " + ", ".join(w["why"])) if w["why"] else ""
            L.append(f"- {w['specialist']}: {_pct(w['weight'] or 0)} of the weight{why}.")
        if f.get("gate_listens_to"):
            L.append("The Devil's Advocate weighs specialists mostly by "
                     + ", ".join(f["gate_listens_to"]) + " (learned on validation lesions).")
        L.append("")
        L.append("WHAT THE CONFIDENCE MEANS")
        if f["calibrated_confidence"] is not None:
            L.append(f"- Of past validation cases given this confidence, about "
                     f"{_pct(f['calibrated_confidence'])} were right. It is calibrated, "
                     f"not flattering: 11 diagnoses, several of them rare and look-alike.")
        if f.get("consistency") is not None:
            L.append(f"- Consistency between specialists: {f['consistency']:.2f} of 1.")
        return "\n".join(L)

    # ------------------------------------------------------------- verify / LLM
    def verify(self, text: str, f: Dict) -> List[str]:
        problems = []
        base = self.structured(f)
        allowed_pct = {m.group(1) for m in re.finditer(r"(\d{1,3})%", base)}
        for m in re.finditer(r"(\d{1,3})\s*%", text):
            if m.group(1) not in allowed_pct:
                problems.append(f"quotes {m.group(1)}% not in the evidence")
        allowed_cls = {c for c in self.classes if re.search(rf"\b{c}\b", base)}
        for c in self.classes:
            if re.search(rf"\b{c}\b", text) and c not in allowed_cls:
                problems.append(f"names {c}, not in the evidence")
        return problems

    def explain(self, team: Dict, decision) -> Dict:
        f = self.facts(team, decision)
        base = self.structured(f)
        res = {"facts": f, "text": base, "source": "structured", "problems": []}
        if not base or self.llm is None:
            return res
        try:
            prose = self.llm.generate(base, system_prompt=SYSTEM_PROMPT)
        except Exception as e:  # LLM not running
            res["problems"] = [f"llm: {e}"]
            return res
        problems = self.verify(prose or "", f)
        if problems or not prose:
            res.update(source="structured (LLM rewrite rejected)",
                       problems=problems or ["empty"])
            return res
        res.update(text=prose.strip(), source="llm (verified)")
        return res
