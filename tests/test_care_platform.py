"""
tests/test_care_platform.py - The patient / doctor workflow around the engine.

Uses a scripted stand-in engine so the workflow is tested without the trained
model. The engine contract it relies on is the one NativeEngine implements:
`assess(clinical, dermoscopy, metadata) -> result` with `.stage`, `.action`,
`.decision`, `.classes`, `.to_dict()`, `.patient_message`, `.doctor_message`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pytest
from PIL import Image

from src.care import CareService, CareStore, CaseStatus
from src.decision.policy import ThreeActionPolicy

CLASSES = ["BCC", "ACK", "NEV", "SEK", "SCC", "MEL"]
MAL = [0, 4, 5]


@dataclass
class FakeResult:
    decision: object
    stage: str
    classes: List[str] = field(default_factory=lambda: CLASSES)
    patient_message: str = "patient text"
    doctor_message: str = "doctor text"

    @property
    def action(self):
        return self.decision.action.value

    def to_dict(self) -> Dict:
        return {"action": self.action, "decision": self.decision.to_dict()}


class ScriptedEngine:
    """Returns pre-set probability vectors in order, one per assess() call."""

    malignant_names = [CLASSES[i] for i in MAL]

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def assess(self, clinical, dermoscopy, metadata):
        self.calls.append({"clinical": clinical is not None,
                           "dermoscopy": dermoscopy is not None, "meta": metadata})
        probs, kw = self.script.pop(0)
        pol = ThreeActionPolicy(malignant_classes=MAL, num_classes=6, label_rule="argmax",
                                ambiguity_gate="reassurance", conflict_threshold=0.5)
        acq = [] if dermoscopy is not None else ["dermoscopy_image"]
        d = pol.decide(np.asarray(probs), acquirable=acq,
                       n_acquisitions_so_far=1 if dermoscopy is not None else 0, **kw)
        return FakeResult(decision=d, stage="B" if dermoscopy is not None else "A")


def vec(cls, p):
    v = np.full(6, (1 - p) / 5)
    v[cls] = p
    return v


CONFIDENT_NEV = (vec(2, 0.97), {"ignorance": 0.05, "conflict": 0.0})
CONFIDENT_BCC = (vec(0, 0.95), {"ignorance": 0.05, "conflict": 0.0})
CONFLICTED = (vec(5, 0.55), {"ignorance": 0.10, "conflict": 0.90})
NEEDS_DERMO = (vec(2, 0.55), {"ignorance": 0.60, "conflict": 0.0,
                              "predicted_benefit": {"dermoscopy_image": 5.0}})


@pytest.fixture
def img():
    return Image.new("RGB", (32, 32), (180, 120, 100))


def make(tmp_path, script):
    eng = ScriptedEngine(script)
    svc = CareService(eng, CareStore(":memory:"), upload_dir=tmp_path / "up")
    pid = svc.register_patient("Test Patient", age=60, sex="female")
    cid = svc.open_case(pid, site="forearm", complaint="new spot")
    return svc, eng, cid


# ------------------------------------------------------------------ routing
def test_proceed_benign_schedules_routine_followup(tmp_path, img):
    svc, _, cid = make(tmp_path, [CONFIDENT_NEV])
    svc.submit_photo(cid, img)
    out = svc.assess(cid)
    assert out["action"] == "PROCEED"
    assert svc.store.case(cid)["status"] == CaseStatus.ASSESSED
    fu = svc.store.followups(cid)
    assert len(fu) == 1 and fu[0]["created_by"] == "system"


def test_proceed_malignant_is_urgent_and_short_followup(tmp_path, img):
    svc, _, cid = make(tmp_path, [CONFIDENT_BCC])
    svc.submit_photo(cid, img)
    out = svc.assess(cid)
    assert out["action"] == "PROCEED" and out["urgency"] == "URGENT"
    assert svc.store.case(cid)["urgency"] == "URGENT"


def test_defer_goes_to_doctor_with_tentative_opinion(tmp_path, img):
    svc, _, cid = make(tmp_path, [CONFLICTED])
    svc.submit_photo(cid, img)
    out = svc.assess(cid)
    assert out["action"] == "DEFER"
    assert out["tentative"] == "MEL", "deferral must still carry the system's opinion"
    assert svc.store.case(cid)["status"] == CaseStatus.AWAITING_DOCTOR
    q = svc.doctor_queue()
    assert q[0]["case_id"] == cid and q[0]["kind"] == "review"


def test_acquire_asks_patient_then_reassesses_with_dermoscopy(tmp_path, img):
    svc, eng, cid = make(tmp_path, [NEEDS_DERMO, CONFIDENT_NEV])
    svc.submit_photo(cid, img)
    first = svc.assess(cid)
    assert first["action"] == "ACQUIRE"
    assert svc.store.case(cid)["status"] == CaseStatus.AWAITING_PATIENT
    msgs = [m["body"] for m in svc.store.messages(cid)]
    assert any("dermoscop" in b.lower() for b in msgs)

    svc.submit_dermoscopy(cid, img)
    second = svc.assess(cid)
    assert eng.calls[-1]["dermoscopy"] is True
    assert second["action"] in ("PROCEED", "DEFER")
    assert second["action"] != "ACQUIRE", "at most one acquisition round"


def test_assess_requires_a_photo(tmp_path):
    svc, _, cid = make(tmp_path, [CONFIDENT_NEV])
    with pytest.raises(ValueError, match="photo"):
        svc.assess(cid)


def test_details_reach_the_engine(tmp_path, img):
    svc, eng, cid = make(tmp_path, [CONFIDENT_NEV])
    svc.submit_photo(cid, img)
    svc.submit_details(cid, {"age": 60, "site": "upper_extremity"})
    svc.assess(cid)
    assert eng.calls[-1]["meta"]["site"] == "upper_extremity"


# ------------------------------------------------------------ doctor side
def test_queue_puts_urgent_review_first(tmp_path, img):
    eng = ScriptedEngine([CONFIDENT_NEV, CONFLICTED])
    svc = CareService(eng, CareStore(":memory:"), upload_dir=tmp_path / "up")
    pid = svc.register_patient("A")
    c1 = svc.open_case(pid)
    svc.submit_photo(c1, img)
    svc.assess(c1)                              # PROCEED -> sign-off
    c2 = svc.open_case(pid)
    svc.submit_photo(c2, img)
    svc.assess(c2)                              # DEFER (conflict) -> review
    q = svc.doctor_queue()
    assert [r["case_id"] for r in q] == [c2, c1]


def test_confirm_defaults_to_tentative_and_closes(tmp_path, img):
    svc, _, cid = make(tmp_path, [CONFLICTED])
    svc.submit_photo(cid, img)
    svc.assess(cid)
    svc.doctor_review(cid, "Rao", "CONFIRM")
    assert svc.store.reviews(cid)[0]["diagnosis"] == "MEL"
    assert svc.store.case(cid)["status"] == CaseStatus.CLOSED


def test_override_requires_diagnosis(tmp_path, img):
    svc, _, cid = make(tmp_path, [CONFLICTED])
    svc.submit_photo(cid, img)
    svc.assess(cid)
    with pytest.raises(ValueError, match="OVERRIDE"):
        svc.doctor_review(cid, "Rao", "OVERRIDE")


def test_review_with_followup_sets_follow_up_status(tmp_path, img):
    svc, _, cid = make(tmp_path, [CONFLICTED])
    svc.submit_photo(cid, img)
    svc.assess(cid)
    svc.doctor_review(cid, "Rao", "OVERRIDE", diagnosis="NEV",
                      instruction="Benign mole. Re-photo in 3 months.", followup_days=90)
    assert svc.store.case(cid)["status"] == CaseStatus.FOLLOW_UP
    assert any(m["sender"] == "doctor" for m in svc.store.messages(cid))


def test_invalid_verdict_rejected(tmp_path, img):
    svc, _, cid = make(tmp_path, [CONFLICTED])
    svc.submit_photo(cid, img)
    svc.assess(cid)
    with pytest.raises(ValueError):
        svc.doctor_review(cid, "Rao", "MAYBE")


# --------------------------------------------------------------- follow-up
def test_followup_change_alert_returns_case_to_doctor(tmp_path, img):
    svc, _, cid = make(tmp_path, [CONFIDENT_NEV, CONFIDENT_BCC])
    svc.submit_photo(cid, img)
    svc.assess(cid)
    out = svc.submit_followup_photo(cid, img)
    assert out["change"]["alert"] is True
    assert svc.store.case(cid)["status"] == CaseStatus.AWAITING_DOCTOR
    assert svc.store.case(cid)["urgency"] == "URGENT"
    assert all(f["status"] == "DONE" for f in svc.store.followups(cid))


def test_stable_followup_does_not_alert(tmp_path, img):
    svc, _, cid = make(tmp_path, [CONFIDENT_NEV, CONFIDENT_NEV])
    svc.submit_photo(cid, img)
    svc.assess(cid)
    out = svc.submit_followup_photo(cid, img)
    assert out["change"]["alert"] is False


# --------------------------------------------------------- conversation/audit
def test_patient_question_is_flagged_for_doctor(tmp_path, img):
    svc, _, cid = make(tmp_path, [CONFLICTED])
    svc.submit_photo(cid, img)
    svc.assess(cid)
    svc.ask(cid, "Should I be worried?")
    assert svc.doctor_queue()[0]["unanswered_questions"] == 1
    svc.reply(cid, "Rao", "We will look today.")
    assert svc.doctor_queue()[0]["unanswered_questions"] == 0


def test_every_step_is_audited(tmp_path, img):
    svc, _, cid = make(tmp_path, [CONFLICTED])
    svc.submit_photo(cid, img)
    svc.assess(cid)
    svc.doctor_review(cid, "Rao", "CONFIRM")
    events = [a["event"] for a in svc.store.audit_log(cid)]
    for e in ("case_opened", "evidence_added", "assessed", "status_changed", "reviewed"):
        assert e in events


def test_assessment_records_the_policy(tmp_path, img):
    svc, _, cid = make(tmp_path, [CONFIDENT_NEV])
    svc.submit_photo(cid, img)
    svc.assess(cid)
    a = svc.store.latest_assessment(cid)
    assert "engine" in a["policy"]


def test_timeline_is_chronological(tmp_path, img):
    svc, _, cid = make(tmp_path, [CONFLICTED])
    svc.submit_photo(cid, img)
    svc.assess(cid)
    svc.doctor_review(cid, "Rao", "CONFIRM")
    t = [r["t"] for r in svc.timeline(cid)]
    assert t == sorted(t)
    types = {r["type"] for r in svc.timeline(cid)}
    assert {"evidence", "assessment", "review", "message"} <= types


def test_stats(tmp_path, img):
    svc, _, cid = make(tmp_path, [CONFIDENT_NEV])
    svc.submit_photo(cid, img)
    svc.assess(cid)
    st = svc.store.stats()
    assert st["patients"] == 1 and st["cases"] == 1
    assert st["assessments_by_action"].get("PROCEED") == 1
