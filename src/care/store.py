"""
src/care/store.py - SQLite persistence for the care platform.

One file, no server. Every table is append-mostly, and every state change also
writes an `audit` row, so the history of a case can always be reconstructed:
who submitted what, what the system decided, which policy version decided it,
and what the doctor did about it.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL,
    age          REAL,
    sex          TEXT,
    skin_tone    TEXT,
    contact      TEXT,
    created_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cases (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id   INTEGER NOT NULL REFERENCES patients(id),
    site         TEXT,
    complaint    TEXT,
    status       TEXT NOT NULL,
    urgency      TEXT,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evidence (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id      INTEGER NOT NULL REFERENCES cases(id),
    kind         TEXT NOT NULL,          -- clinical_photo | dermoscopy | details
    path         TEXT,
    payload      TEXT,
    created_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS assessments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id      INTEGER NOT NULL REFERENCES cases(id),
    stage        TEXT NOT NULL,
    action       TEXT NOT NULL,
    tentative    TEXT,
    p_malignant  REAL,
    urgency      TEXT,
    result       TEXT NOT NULL,
    policy       TEXT,
    created_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reviews (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id      INTEGER NOT NULL REFERENCES cases(id),
    doctor       TEXT NOT NULL,
    verdict      TEXT NOT NULL,          -- CONFIRM | OVERRIDE | NEED_IN_PERSON
    diagnosis    TEXT,
    note         TEXT,
    created_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id      INTEGER NOT NULL REFERENCES cases(id),
    sender       TEXT NOT NULL,          -- system | doctor | patient
    kind         TEXT NOT NULL,          -- info | instruction | question | result
    body         TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS followups (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id      INTEGER NOT NULL REFERENCES cases(id),
    due_date     TEXT NOT NULL,
    reason       TEXT,
    created_by   TEXT NOT NULL,
    status       TEXT NOT NULL,          -- SCHEDULED | DONE | CANCELLED
    created_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id      INTEGER,
    actor        TEXT NOT NULL,
    event        TEXT NOT NULL,
    detail       TEXT,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_cases_status ON cases(status);
CREATE INDEX IF NOT EXISTS ix_assess_case ON assessments(case_id);
CREATE INDEX IF NOT EXISTS ix_msg_case ON messages(case_id);
"""


def now() -> str:
    # Microseconds, so events written in the same second still sort in order.
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class CareStore:
    """Thin, thread-safe wrapper over one SQLite file (or ':memory:' for tests)."""

    def __init__(self, path: str | Path = "data/care/dguard_care.sqlite"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(SCHEMA)
        self._db.commit()

    # ------------------------------------------------------------ primitives
    def _insert(self, table: str, row: Dict[str, Any]) -> int:
        cols = ", ".join(row)
        qs = ", ".join("?" for _ in row)
        with self._lock:
            cur = self._db.execute(f"INSERT INTO {table} ({cols}) VALUES ({qs})",
                                   tuple(row.values()))
            self._db.commit()
            return int(cur.lastrowid)

    def _rows(self, sql: str, args: tuple = ()) -> List[Dict]:
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, args).fetchall()]

    def _one(self, sql: str, args: tuple = ()) -> Optional[Dict]:
        rows = self._rows(sql, args)
        return rows[0] if rows else None

    def _exec(self, sql: str, args: tuple = ()) -> None:
        with self._lock:
            self._db.execute(sql, args)
            self._db.commit()

    # --------------------------------------------------------------- audit
    def audit(self, case_id: Optional[int], actor: str, event: str,
              detail: Optional[Dict] = None) -> None:
        self._insert("audit", {"case_id": case_id, "actor": actor, "event": event,
                               "detail": json.dumps(detail or {}, default=str),
                               "created_at": now()})

    # ------------------------------------------------------------ patients
    def add_patient(self, name: str, age: Optional[float] = None, sex: Optional[str] = None,
                    skin_tone: Optional[str] = None, contact: Optional[str] = None) -> int:
        pid = self._insert("patients", {"name": name, "age": age, "sex": sex,
                                        "skin_tone": skin_tone, "contact": contact,
                                        "created_at": now()})
        self.audit(None, "system", "patient_registered", {"patient_id": pid})
        return pid

    def patient(self, pid: int) -> Optional[Dict]:
        return self._one("SELECT * FROM patients WHERE id = ?", (pid,))

    def patients(self) -> List[Dict]:
        return self._rows("SELECT * FROM patients ORDER BY id DESC")

    # --------------------------------------------------------------- cases
    def add_case(self, patient_id: int, site: Optional[str], complaint: Optional[str],
                 status: str) -> int:
        t = now()
        cid = self._insert("cases", {"patient_id": patient_id, "site": site,
                                     "complaint": complaint, "status": status,
                                     "urgency": None, "created_at": t, "updated_at": t})
        self.audit(cid, "patient", "case_opened", {"site": site})
        return cid

    def case(self, cid: int) -> Optional[Dict]:
        return self._one("SELECT * FROM cases WHERE id = ?", (cid,))

    def cases(self, patient_id: Optional[int] = None,
              status: Optional[str] = None) -> List[Dict]:
        sql, args = "SELECT * FROM cases WHERE 1=1", []
        if patient_id is not None:
            sql += " AND patient_id = ?"
            args.append(patient_id)
        if status is not None:
            sql += " AND status = ?"
            args.append(status)
        return self._rows(sql + " ORDER BY updated_at DESC", tuple(args))

    def set_status(self, cid: int, status: str, urgency: Optional[str] = None,
                   actor: str = "system") -> None:
        old = self.case(cid)
        if urgency is None:
            self._exec("UPDATE cases SET status = ?, updated_at = ? WHERE id = ?",
                       (status, now(), cid))
        else:
            self._exec("UPDATE cases SET status = ?, urgency = ?, updated_at = ? "
                       "WHERE id = ?", (status, urgency, now(), cid))
        self.audit(cid, actor, "status_changed",
                   {"from": old["status"] if old else None, "to": status,
                    "urgency": urgency})

    # ------------------------------------------------------------ evidence
    def add_evidence(self, cid: int, kind: str, path: Optional[str] = None,
                     payload: Optional[Dict] = None, actor: str = "patient") -> int:
        eid = self._insert("evidence", {"case_id": cid, "kind": kind, "path": path,
                                        "payload": json.dumps(payload or {}, default=str),
                                        "created_at": now()})
        self.audit(cid, actor, "evidence_added", {"kind": kind, "evidence_id": eid})
        return eid

    def latest_evidence(self, cid: int, kind: str) -> Optional[Dict]:
        r = self._one("SELECT * FROM evidence WHERE case_id = ? AND kind = ? "
                      "ORDER BY id DESC LIMIT 1", (cid, kind))
        if r:
            r["payload"] = json.loads(r["payload"] or "{}")
        return r

    def evidence(self, cid: int) -> List[Dict]:
        rows = self._rows("SELECT * FROM evidence WHERE case_id = ? ORDER BY id", (cid,))
        for r in rows:
            r["payload"] = json.loads(r["payload"] or "{}")
        return rows

    # --------------------------------------------------------- assessments
    def add_assessment(self, cid: int, stage: str, action: str, tentative: Optional[str],
                       p_malignant: Optional[float], urgency: Optional[str],
                       result: Dict, policy: Dict) -> int:
        aid = self._insert("assessments", {
            "case_id": cid, "stage": stage, "action": action, "tentative": tentative,
            "p_malignant": p_malignant, "urgency": urgency,
            "result": json.dumps(result, default=str),
            "policy": json.dumps(policy, default=str), "created_at": now()})
        self.audit(cid, "system", "assessed",
                   {"assessment_id": aid, "stage": stage, "action": action,
                    "urgency": urgency})
        return aid

    def assessments(self, cid: int) -> List[Dict]:
        rows = self._rows("SELECT * FROM assessments WHERE case_id = ? ORDER BY id", (cid,))
        for r in rows:
            r["result"] = json.loads(r["result"])
            r["policy"] = json.loads(r["policy"] or "{}")
        return rows

    def latest_assessment(self, cid: int) -> Optional[Dict]:
        a = self.assessments(cid)
        return a[-1] if a else None

    # -------------------------------------------------------------- reviews
    def add_review(self, cid: int, doctor: str, verdict: str, diagnosis: Optional[str],
                   note: Optional[str]) -> int:
        rid = self._insert("reviews", {"case_id": cid, "doctor": doctor, "verdict": verdict,
                                       "diagnosis": diagnosis, "note": note,
                                       "created_at": now()})
        self.audit(cid, f"doctor:{doctor}", "reviewed",
                   {"verdict": verdict, "diagnosis": diagnosis})
        return rid

    def reviews(self, cid: int) -> List[Dict]:
        return self._rows("SELECT * FROM reviews WHERE case_id = ? ORDER BY id", (cid,))

    # ------------------------------------------------------------- messages
    def add_message(self, cid: int, sender: str, kind: str, body: str) -> int:
        mid = self._insert("messages", {"case_id": cid, "sender": sender, "kind": kind,
                                        "body": body, "created_at": now()})
        self.audit(cid, sender, "message", {"kind": kind, "message_id": mid})
        return mid

    def messages(self, cid: int) -> List[Dict]:
        return self._rows("SELECT * FROM messages WHERE case_id = ? ORDER BY id", (cid,))

    # ------------------------------------------------------------ follow-ups
    def add_followup(self, cid: int, due_date: str, reason: str, created_by: str) -> int:
        fid = self._insert("followups", {"case_id": cid, "due_date": due_date,
                                         "reason": reason, "created_by": created_by,
                                         "status": "SCHEDULED", "created_at": now()})
        self.audit(cid, created_by, "followup_scheduled",
                   {"followup_id": fid, "due": due_date})
        return fid

    def followups(self, cid: Optional[int] = None, status: Optional[str] = None) -> List[Dict]:
        sql, args = "SELECT * FROM followups WHERE 1=1", []
        if cid is not None:
            sql += " AND case_id = ?"
            args.append(cid)
        if status is not None:
            sql += " AND status = ?"
            args.append(status)
        return self._rows(sql + " ORDER BY due_date", tuple(args))

    def close_followup(self, fid: int, status: str = "DONE", actor: str = "system") -> None:
        row = self._one("SELECT * FROM followups WHERE id = ?", (fid,))
        self._exec("UPDATE followups SET status = ? WHERE id = ?", (status, fid))
        self.audit(row["case_id"] if row else None, actor, "followup_closed",
                   {"followup_id": fid, "status": status})

    # ----------------------------------------------------------------- audit
    def audit_log(self, cid: Optional[int] = None, limit: int = 200) -> List[Dict]:
        if cid is None:
            rows = self._rows("SELECT * FROM audit ORDER BY id DESC LIMIT ?", (limit,))
        else:
            rows = self._rows("SELECT * FROM audit WHERE case_id = ? ORDER BY id", (cid,))
        for r in rows:
            r["detail"] = json.loads(r["detail"] or "{}")
        return rows

    def stats(self) -> Dict[str, Any]:
        def count(sql, args=()):
            return self._one(sql, args)["n"]
        return {
            "patients": count("SELECT COUNT(*) AS n FROM patients"),
            "cases": count("SELECT COUNT(*) AS n FROM cases"),
            "by_status": {r["status"]: r["n"] for r in self._rows(
                "SELECT status, COUNT(*) AS n FROM cases GROUP BY status")},
            "assessments_by_action": {r["action"]: r["n"] for r in self._rows(
                "SELECT action, COUNT(*) AS n FROM assessments GROUP BY action")},
            "reviews_by_verdict": {r["verdict"]: r["n"] for r in self._rows(
                "SELECT verdict, COUNT(*) AS n FROM reviews GROUP BY verdict")},
            "open_followups": count("SELECT COUNT(*) AS n FROM followups "
                                    "WHERE status = 'SCHEDULED'"),
        }
