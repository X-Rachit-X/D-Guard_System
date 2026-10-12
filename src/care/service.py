"""
src/care/service.py - The care workflow around the decision engine.

    patient opens a case ──► uploads photo + details ──► ASSESS
                                                          │
          ┌───────────────────────────────┬───────────────┴─────────────────┐
       PROCEED                         ACQUIRE                             DEFER
  result + instructions,       "please add a dermoscopy       tentative opinion + strong
  auto follow-up, doctor        photo" -> AWAITING_PATIENT    "see a doctor" -> doctor
  sign-off queue                 -> re-assess on upload        queue, URGENT first
          │                               │                                 │
          └──────────────► DOCTOR REVIEW: confirm / override / see in person ◄┘
                             note + instruction to patient + follow-up date
                                              │
                               FOLLOW_UP: new photo at the due date
                                              │
                          re-assess + CHANGE CHECK against the last assessment
                          (rising cancer probability or a new malignant call
                           sends the case straight back to the doctor, URGENT)

Hard rules
  * The engine's action is never overridden by the platform. Only a doctor can
    turn a DEFER into a diagnosis, and that is recorded as the doctor's verdict.
  * Every state change writes an audit row with the policy that produced it.
"""
from __future__ import annotations

import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from src.care.store import CareStore


class CaseStatus:
    NEW = "NEW"
    AWAITING_PATIENT = "AWAITING_PATIENT"      # e.g. dermoscopy requested
    AWAITING_DOCTOR = "AWAITING_DOCTOR"        # deferred to a human
    ASSESSED = "ASSESSED"                      # system answered; doctor sign-off pending
    FOLLOW_UP = "FOLLOW_UP"                    # reviewed; re-check scheduled
    CLOSED = "CLOSED"


#: Re-check intervals the SYSTEM proposes. A doctor can always change them.
AUTO_FOLLOWUP_DAYS = {
    ("PROCEED", "URGENT"): 14,     # confirm the patient was actually seen
    ("PROCEED", "ROUTINE"): 90,    # watchful waiting: new photo in 3 months
}

#: A follow-up photo that moves P(malignant) up by at least this much, or that
#: turns a benign tentative call into a malignant one, is sent back to a doctor.
CHANGE_ALERT_DELTA = 0.15

ACQUIRE_HOWTO = (
    "How to get a dermoscopy photo: a dermoscope is a small magnifying lens with "
    "its own light. Most dermatology clinics, and many pharmacies and GP surgeries, "
    "can take one in a few minutes; clip-on dermoscopes for smartphones also work. "
    "Hold it flat against the spot and keep the whole spot in frame."
)

DETAILS_HOWTO = (
    "Please add a few details about yourself and the spot: your age, sex and where "
    "on the body it is. Every case the system was checked on had these details; "
    "without them (and without a dermoscopy photo) its answers are less reliable. "
    "It takes under a minute, and the spot will be assessed again straight away."
)


def _today() -> datetime:
    return datetime.now(timezone.utc)


class CareService:
    def __init__(self, engine: Any, store: Optional[CareStore] = None,
                 upload_dir: str | Path = "data/care/uploads"):
        self.engine = engine
        self.store = store or CareStore()
        self.upload_dir = Path(upload_dir)

    # ============================================================ intake
    def register_patient(self, name: str, age: Optional[float] = None,
                         sex: Optional[str] = None, skin_tone: Optional[str] = None,
                         contact: Optional[str] = None) -> int:
        return self.store.add_patient(name, age, sex, skin_tone, contact)

    def open_case(self, patient_id: int, site: Optional[str] = None,
                  complaint: Optional[str] = None) -> int:
        if self.store.patient(patient_id) is None:
            raise KeyError(f"no patient {patient_id}")
        cid = self.store.add_case(patient_id, site, complaint, CaseStatus.NEW)
        self.store.add_message(cid, "system", "info",
                               "Your case is open. Please upload a clear, well-lit photo "
                               "of the spot and answer a few questions about it.")
        return cid

    def _save_image(self, cid: int, kind: str, image: Image.Image | str | Path) -> str:
        d = self.upload_dir / f"case_{cid}"
        d.mkdir(parents=True, exist_ok=True)
        n = len(list(d.glob(f"{kind}_*"))) + 1
        dest = d / f"{kind}_{n}.png"
        if isinstance(image, (str, Path)):
            src = Path(image)
            if src.suffix.lower() == ".png":
                shutil.copyfile(src, dest)
            else:
                Image.open(src).convert("RGB").save(dest)
        else:
            image.convert("RGB").save(dest)
        return str(dest)

    def submit_photo(self, cid: int, image, actor: str = "patient") -> int:
        return self.store.add_evidence(cid, "clinical_photo",
                                       self._save_image(cid, "clinical", image), actor=actor)

    def submit_dermoscopy(self, cid: int, image, actor: str = "patient") -> int:
        return self.store.add_evidence(cid, "dermoscopy",
                                       self._save_image(cid, "dermoscopy", image), actor=actor)

    def submit_details(self, cid: int, details: Dict, actor: str = "patient") -> int:
        return self.store.add_evidence(cid, "details", payload=details, actor=actor)

    # ========================================================== assessment
    def _policy_info(self) -> Dict:
        b = getattr(self.engine, "bundle", None) or {}
        cost = b.get("cost")
        return {
            "engine": type(self.engine).__name__,
            "label_rule": b.get("label_rule"),
            "ambiguity_gate": b.get("ambiguity_gate"),
            "set_mode": b.get("set_mode"),
            "missed_malignant_cost": getattr(cost, "missed_malignant", None),
            "bundle_version": b.get("version"),
        }

    def assess(self, cid: int) -> Dict:
        """Run the engine on the latest evidence and route the case."""
        case = self.store.case(cid)
        if case is None:
            raise KeyError(f"no case {cid}")
        photo = self.store.latest_evidence(cid, "clinical_photo")
        derm = self.store.latest_evidence(cid, "dermoscopy")
        details = self.store.latest_evidence(cid, "details")
        if photo is None and derm is None:
            raise ValueError("upload at least one photo before assessing")

        load = lambda e: Image.open(e["path"]).convert("RGB") if e else None  # noqa: E731
        result = self.engine.assess(load(photo), load(derm),
                                    (details or {}).get("payload") or {})
        d = result.decision
        cls = result.classes
        tentative = cls[d.tentative] if d.tentative is not None else None
        previous = self.store.latest_assessment(cid)

        aid = self.store.add_assessment(
            cid, result.stage, result.action, tentative, d.p_malignant, d.urgency,
            result.to_dict(), self._policy_info())

        self.store.add_message(cid, "system", "result", result.patient_message)
        change = self._change_check(previous, d.p_malignant, tentative)
        tier = getattr(d, "tier", None)
        self._last_acquire_target = getattr(d, "acquire_target", None)
        self._route(cid, result.action, d.urgency, change, tier)
        return {"assessment_id": aid, "action": result.action, "urgency": d.urgency,
                "tier": tier, "tentative": tentative, "change": change, "result": result}

    def _change_check(self, previous: Optional[Dict], p_mal: Optional[float],
                      tentative: Optional[str]) -> Optional[Dict]:
        """Compare with the last assessment of the same case (follow-up monitoring)."""
        if previous is None or p_mal is None:
            return None
        prev_p = previous.get("p_malignant")
        malignant = set(getattr(self.engine, "malignant_names", []) or [])
        became_malignant = (previous.get("tentative") not in malignant
                            and tentative in malignant) if malignant else False
        delta = None if prev_p is None else float(p_mal) - float(prev_p)
        alert = (delta is not None and delta >= CHANGE_ALERT_DELTA) or became_malignant
        return {"previous_p_malignant": prev_p, "p_malignant": p_mal, "delta": delta,
                "previous_tentative": previous.get("tentative"), "tentative": tentative,
                "alert": bool(alert)}

    def _route(self, cid: int, action: str, urgency: Optional[str],
               change: Optional[Dict], tier: Optional[str] = None) -> None:
        s = self.store
        if change and change["alert"]:
            s.set_status(cid, CaseStatus.AWAITING_DOCTOR, "URGENT")
            s.add_message(cid, "system", "instruction",
                          "Your new photo shows a change compared with last time. A "
                          "doctor will review it urgently. Please do not wait for your "
                          "next scheduled check.")
            return

        if action == "ACQUIRE":
            s.set_status(cid, CaseStatus.AWAITING_PATIENT, urgency)
            target = getattr(self, "_last_acquire_target", None)
            s.add_message(cid, "system", "instruction",
                          DETAILS_HOWTO if target == "metadata" else ACQUIRE_HOWTO)
            return

        if action == "DEFER":
            s.set_status(cid, CaseStatus.AWAITING_DOCTOR, urgency)
            when = "within 48 hours" if urgency == "URGENT" else "within 7 days"
            s.add_message(cid, "system", "instruction",
                          f"A doctor will review your case {when}. If the spot bleeds, "
                          f"grows quickly or becomes painful before then, see a doctor "
                          f"in person straight away.")
            return

        if action == "PROCEED" and tier == "LOW":
            # The patient already has the (clearly labelled) low-confidence
            # answer; a doctor must confirm it before anyone acts on it.
            s.set_status(cid, CaseStatus.AWAITING_DOCTOR, urgency)
            when = "within 48 hours" if urgency == "URGENT" else "within 7 days"
            s.add_message(cid, "system", "instruction",
                          f"Because this assessment is low confidence, a doctor will "
                          f"confirm it {when}. If the spot bleeds, grows quickly or "
                          f"becomes painful before then, see a doctor in person.")
            return

        # PROCEED, HIGH confidence
        s.set_status(cid, CaseStatus.ASSESSED, urgency)
        days = AUTO_FOLLOWUP_DAYS.get(("PROCEED", urgency or "ROUTINE"))
        if days:
            due = (_today() + timedelta(days=days)).date().isoformat()
            reason = ("check that the specialist appointment happened"
                      if urgency == "URGENT" else "routine re-photo to watch for change")
            s.add_followup(cid, due, reason, created_by="system")
            s.add_message(cid, "system", "instruction",
                          f"Follow-up: on or after {due}, please "
                          + ("confirm you have seen a dermatologist."
                             if urgency == "URGENT" else
                             "upload a new photo of the same spot, taken the same way."))

    # ======================================================= doctor side
    def doctor_queue(self) -> List[Dict]:
        """Cases waiting for a human, most urgent first.

        Deferred cases outrank system-answered ones awaiting sign-off, URGENT
        outranks ROUTINE, and older outranks newer.
        """
        rows = (self.store.cases(status=CaseStatus.AWAITING_DOCTOR)
                + self.store.cases(status=CaseStatus.ASSESSED))
        out = []
        for c in rows:
            a = self.store.latest_assessment(c["id"])
            p = self.store.patient(c["patient_id"])
            open_q = [m for m in self.store.messages(c["id"])
                      if m["sender"] == "patient" and m["kind"] == "question"]
            answered_after = [m for m in self.store.messages(c["id"])
                              if m["sender"] == "doctor"]
            out.append({
                "case_id": c["id"], "patient": p["name"] if p else "?",
                "status": c["status"], "urgency": c["urgency"] or "ROUTINE",
                "action": a["action"] if a else None,
                "tentative": a["tentative"] if a else None,
                "p_malignant": a["p_malignant"] if a else None,
                "waiting_since": c["updated_at"],
                "unanswered_questions": max(0, len(open_q) - len(answered_after)),
                "tier": (a["result"].get("decision", {}).get("tier") if a else None),
                "kind": ("sign-off" if c["status"] != CaseStatus.AWAITING_DOCTOR
                         else "confirm" if a and a["result"].get("decision", {}).get("tier") == "LOW"
                         else "review"),
            })
        rank = lambda r: ({"review": 0, "confirm": 1}.get(r["kind"], 2),  # noqa: E731
                          0 if r["urgency"] == "URGENT" else 1, r["waiting_since"])
        return sorted(out, key=rank)

    def doctor_review(self, cid: int, doctor: str, verdict: str,
                      diagnosis: Optional[str] = None, note: Optional[str] = None,
                      instruction: Optional[str] = None,
                      followup_days: Optional[int] = None) -> int:
        """verdict: CONFIRM | OVERRIDE | NEED_IN_PERSON."""
        verdict = verdict.upper()
        if verdict not in ("CONFIRM", "OVERRIDE", "NEED_IN_PERSON"):
            raise ValueError("verdict must be CONFIRM, OVERRIDE or NEED_IN_PERSON")
        if verdict == "OVERRIDE" and not diagnosis:
            raise ValueError("an OVERRIDE must state the doctor's diagnosis")
        a = self.store.latest_assessment(cid)
        if verdict == "CONFIRM" and diagnosis is None and a:
            diagnosis = a["tentative"]
        rid = self.store.add_review(cid, doctor, verdict, diagnosis, note)

        if verdict == "NEED_IN_PERSON":
            body = (instruction or "Please book an in-person appointment so a doctor can "
                    "examine the spot directly.")
        else:
            body = instruction or (f"Dr {doctor} has reviewed your case. Diagnosis: "
                                   f"{diagnosis}.")
        self.store.add_message(cid, "doctor", "instruction", body)

        if followup_days:
            due = (_today() + timedelta(days=int(followup_days))).date().isoformat()
            self.store.add_followup(cid, due, note or "doctor-requested re-check",
                                    created_by=f"doctor:{doctor}")
            self.store.set_status(cid, CaseStatus.FOLLOW_UP, actor=f"doctor:{doctor}")
        else:
            self.store.set_status(cid, CaseStatus.CLOSED, actor=f"doctor:{doctor}")
        return rid

    # ============================================== patient conversation
    # ======================================================= data control
    def _remove_uploads(self, paths) -> int:
        """Delete only files this platform saved (inside upload_dir) - never the
        dataset images used by the demo."""
        n = 0
        root = Path(self.upload_dir).resolve()
        for p in paths:
            try:
                fp = Path(p).resolve()
                if root in fp.parents and fp.exists():
                    fp.unlink()
                    n += 1
            except OSError:
                pass
        return n

    def delete_case(self, cid: int, actor: str = "patient") -> int:
        return self._remove_uploads(self.store.delete_case(cid, actor))

    def delete_patient(self, pid: int, actor: str = "patient") -> int:
        return self._remove_uploads(self.store.delete_patient(pid, actor))

    def clear_messages(self, cid: int, actor: str = "patient") -> None:
        self.store.clear_messages(cid, actor)

    def reset_all(self) -> int:
        paths = [e["path"] for c in self.store.cases() for e in self.store.evidence(c["id"])
                 if e.get("path")]
        self.store.reset_all()
        return self._remove_uploads(paths)

    def ask(self, cid: int, question: str) -> int:
        mid = self.store.add_message(cid, "patient", "question", question)
        self.store.add_message(cid, "system", "info",
                               "Thank you - your question has been passed to your doctor, "
                               "who will reply here.")
        return mid

    def reply(self, cid: int, doctor: str, text: str) -> int:
        return self.store.add_message(cid, "doctor", "info", f"Dr {doctor}: {text}")

    # ============================================================ follow-up
    def due_followups(self, on: Optional[str] = None) -> List[Dict]:
        day = on or _today().date().isoformat()
        return [f for f in self.store.followups(status="SCHEDULED") if f["due_date"] <= day]

    def submit_followup_photo(self, cid: int, image, dermoscopy=None,
                              details: Optional[Dict] = None) -> Dict:
        """New evidence at a follow-up: store it, re-assess, run the change check."""
        for f in self.store.followups(cid, status="SCHEDULED"):
            self.store.close_followup(f["id"], "DONE", actor="patient")
        self.submit_photo(cid, image)
        if dermoscopy is not None:
            self.submit_dermoscopy(cid, dermoscopy)
        if details:
            self.submit_details(cid, details)
        return self.assess(cid)

    # ============================================================== views
    def timeline(self, cid: int) -> List[Dict]:
        """Everything that happened to a case, in order - for both views."""
        s = self.store
        ev = [{"t": e["created_at"], "type": "evidence", "what": e["kind"],
               "detail": e["path"] or e["payload"]} for e in s.evidence(cid)]
        asm = [{"t": a["created_at"], "type": "assessment",
                "what": f"{a['action']} (stage {a['stage']})",
                "detail": {"tentative": a["tentative"], "p_malignant": a["p_malignant"],
                           "urgency": a["urgency"]}} for a in s.assessments(cid)]
        rv = [{"t": r["created_at"], "type": "review",
               "what": f"{r['verdict']} by Dr {r['doctor']}",
               "detail": {"diagnosis": r["diagnosis"], "note": r["note"]}}
              for r in s.reviews(cid)]
        msg = [{"t": m["created_at"], "type": "message",
                "what": f"{m['sender']} / {m['kind']}", "detail": m["body"]}
               for m in s.messages(cid)]
        fu = [{"t": f["created_at"], "type": "follow-up",
               "what": f"due {f['due_date']} ({f['status']})", "detail": f["reason"]}
              for f in s.followups(cid)]
        return sorted(ev + asm + rv + msg + fu, key=lambda r: r["t"])
