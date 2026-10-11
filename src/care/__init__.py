"""
src/care - the patient / doctor care platform built around the decision engine.

    store.py    SQLite persistence: patients, cases, evidence, assessments,
                doctor reviews, messages, follow-ups, audit log
    service.py  the workflow: intake -> assessment -> (acquire | doctor review)
                -> instructions -> follow-up -> re-assessment

The engine decides PROCEED / ACQUIRE / DEFER. The platform turns that into what
happens next for a real person: who must act, what they are told, and when the
case comes back.
"""
from src.care.service import CareService, CaseStatus
from src.care.store import CareStore

__all__ = ["CareService", "CareStore", "CaseStatus"]
