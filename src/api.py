"""
src/api.py - REST API for the D-GUARD care platform.

    uvicorn src.api:app --port 8000          # docs at http://localhost:8000/docs

The same workflow as the web app, over HTTP, so a mobile app, a chat assistant
or a hospital system can drive it: register a patient, open a case, upload a
photo, get an assessment (with its confidence tier and messages), upload the
dermoscopy the system asks for, ask the doctor questions, and - on the doctor
side - work the review queue.

The model loads on the first request that needs it (~10 s), then stays loaded.
"""
from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from src.care import CareService, CareStore

ROOT = Path(__file__).resolve().parents[1]

app = FastAPI(title="D-GUARD care platform API", version="1.0",
              description="Risk-controlled, cost-aware evidence acquisition and deferral "
                          "for multimodal dermatology. Research prototype - not for diagnosis.")


@lru_cache(maxsize=1)
def service() -> CareService:
    from src.agents.pipeline import DGuardPipeline
    return CareService(DGuardPipeline.build(),
                       CareStore(ROOT / "data" / "care" / "dguard_care.sqlite"),
                       upload_dir=ROOT / "data" / "care" / "uploads")


def _img(f: UploadFile) -> Image.Image:
    try:
        return Image.open(io.BytesIO(f.file.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"not a readable image: {e}")


def _case(cid: int) -> Dict:
    c = service().store.case(cid)
    if c is None:
        raise HTTPException(404, f"no case {cid}")
    return c


def _assessment_view(out: Dict) -> Dict:
    r = out["result"]
    d = r.decision
    return {
        "action": out["action"], "tier": out.get("tier"), "urgency": out["urgency"],
        "tentative": out["tentative"], "p_malignant": d.p_malignant,
        "differential": [[r.classes[c], p] for c, p in d.differential],
        "requires_confirmation": d.requires_confirmation,
        "ignorance": r.ignorance, "conflict": r.conflict,
        "fusion_weights": r.fusion_weights,
        "patient_message": r.patient_message,
        "report": (r.report or {}).get("text"),
        "similar_cases": [{k: c[k] for k in ("lesion_id", "diagnosis", "similarity")}
                          for c in (r.similar or {}).get("similar_cases", [])],
        "quality": r.quality,
        "change": out.get("change"),
    }


# ------------------------------------------------------------------ patient
class PatientIn(BaseModel):
    name: str
    age: Optional[float] = None
    sex: Optional[str] = None
    skin_tone: Optional[str] = None
    contact: Optional[str] = None


class CaseIn(BaseModel):
    site: Optional[str] = None
    complaint: Optional[str] = None


class DetailsIn(BaseModel):
    age: Optional[float] = None
    sex: Optional[str] = None
    site: Optional[str] = None
    anatom_site_general: Optional[str] = None
    skin_tone_class: Optional[str] = None


class QuestionIn(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/patients")
def register(p: PatientIn):
    return {"patient_id": service().register_patient(**p.model_dump())}


@app.post("/patients/{pid}/cases")
def open_case(pid: int, c: CaseIn):
    try:
        return {"case_id": service().open_case(pid, c.site, c.complaint)}
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.post("/cases/{cid}/photo")
def upload_photo(cid: int, file: UploadFile = File(...)):
    _case(cid)
    return {"evidence_id": service().submit_photo(cid, _img(file))}


@app.post("/cases/{cid}/dermoscopy")
def upload_dermoscopy(cid: int, file: UploadFile = File(...)):
    _case(cid)
    return {"evidence_id": service().submit_dermoscopy(cid, _img(file))}


@app.post("/cases/{cid}/details")
def details(cid: int, d: DetailsIn):
    _case(cid)
    return {"evidence_id": service().submit_details(
        cid, {k: v for k, v in d.model_dump().items() if v is not None})}


@app.post("/cases/{cid}/assess")
def assess(cid: int):
    _case(cid)
    try:
        return _assessment_view(service().assess(cid))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/cases/{cid}/followup")
def followup(cid: int, file: UploadFile = File(...)):
    _case(cid)
    return _assessment_view(service().submit_followup_photo(cid, _img(file)))


@app.post("/cases/{cid}/questions")
def ask(cid: int, q: QuestionIn):
    _case(cid)
    return {"message_id": service().ask(cid, q.text)}


@app.get("/cases/{cid}")
def case(cid: int):
    s = service().store
    c = _case(cid)
    a = s.latest_assessment(cid)
    return {"case": c,
            "latest_assessment": a and {k: a[k] for k in ("stage", "action", "tentative",
                                                           "p_malignant", "urgency", "created_at")},
            "messages": s.messages(cid), "followups": s.followups(cid),
            "reviews": s.reviews(cid)}


@app.get("/cases/{cid}/timeline")
def timeline(cid: int):
    _case(cid)
    return service().timeline(cid)


# ------------------------------------------------------------------- doctor
class ReviewIn(BaseModel):
    doctor: str
    verdict: str                      # CONFIRM | OVERRIDE | NEED_IN_PERSON
    diagnosis: Optional[str] = None
    note: Optional[str] = None
    instruction: Optional[str] = None
    followup_days: Optional[int] = None


class ReplyIn(BaseModel):
    doctor: str
    text: str


@app.get("/doctor/queue")
def queue():
    return service().doctor_queue()


@app.post("/cases/{cid}/review")
def review(cid: int, r: ReviewIn):
    _case(cid)
    try:
        return {"review_id": service().doctor_review(cid, **r.model_dump())}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/cases/{cid}/reply")
def reply(cid: int, r: ReplyIn):
    _case(cid)
    return {"message_id": service().reply(cid, r.doctor, r.text)}


@app.delete("/cases/{cid}/messages")
def clear_messages(cid: int):
    _case(cid)
    service().clear_messages(cid, actor="api")
    return {"cleared": cid}


@app.delete("/cases/{cid}")
def delete_case(cid: int):
    _case(cid)
    return {"deleted_case": cid, "files_removed": service().delete_case(cid, actor="api")}


@app.delete("/patients/{pid}")
def delete_patient(pid: int):
    if service().store.patient(pid) is None:
        raise HTTPException(404, f"no patient {pid}")
    return {"deleted_patient": pid, "files_removed": service().delete_patient(pid, actor="api")}


@app.get("/stats")
def stats():
    return service().store.stats()
