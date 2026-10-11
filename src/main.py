"""
src/main.py — FastAPI backend for DERMA-GUARD v4.0

Endpoints:
  GET  /            → health check
  GET  /docs        → auto-generated OpenAPI docs
  POST /predict     → submit a case for analysis
  GET  /case/{id}   → retrieve a case result
  GET  /metrics     → system metrics
"""
from __future__ import annotations

import os
# Prevent OpenMP runtime collision crash on Windows / Anaconda (libiomp5md.dll)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import torch
from PIL import Image
import io

# ---------------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DERMA-GUARD v4.0 API",
    description=(
        "Agentic Evidence-Management Framework for Reliable Multimodal Dermatological Diagnosis.\n\n"
        "**Important**: This is a research system. Results are NOT clinically validated."
    ),
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory result store (production: use a database)
_result_store: Dict[str, Dict] = {}

# Global pipeline (initialized lazily)
_pipeline = None


_pipeline_error: Optional[str] = None


def get_pipeline():
    """Lazily build the fully-wired DERMA-GUARD pipeline.

    Uses OrchestratorAgent.build(), which loads the trained specialists. A
    failure here is recorded and reported by /health and as a 503; the API does
    NOT fall back to an agent-less orchestrator, because that would answer
    requests without ever consulting a model.
    """
    global _pipeline, _pipeline_error
    if _pipeline is None and _pipeline_error is None:
        try:
            from src.agents.orchestrator import OrchestratorAgent
            from src.constants import CLASS_NAMES
            from src.llm.client import OllamaClient

            llm = OllamaClient()
            _pipeline = OrchestratorAgent.build(
                llm_client=llm if llm.is_available() else None,
                class_names=list(CLASS_NAMES),
                model_version="4.0.0",
            )
            print(f"[API] Pipeline ready. Classes: {list(CLASS_NAMES)}")
        except Exception as e:
            _pipeline_error = str(e)
            print(f"[API] Pipeline initialization failed: {e}")
    return _pipeline


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class CaseSubmission(BaseModel):
    patient_text: Optional[str] = None
    metadata: Optional[Dict] = None


class CaseResult(BaseModel):
    case_id: str
    action: str
    diagnosis: Optional[int] = None
    diagnosis_name: Optional[str] = None
    confidence: Optional[float] = None
    consensus_level: Optional[int] = None
    report: Optional[str] = None
    evidence_used: List[str] = []
    evidence_missing: List[str] = []
    fusion_weights: Dict[str, float] = {}
    specialist_votes: Dict[str, str] = {}
    acquire_target: Optional[str] = None
    escalate_reason: Optional[str] = None
    processing_time_s: float = 0.0


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "DERMA-GUARD v4.0",
        "version": "4.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    """Detailed health check."""
    pipeline = get_pipeline()
    from src.llm.client import OllamaClient
    llm = OllamaClient()
    return {
        "status": "ok",
        "pipeline_loaded": pipeline is not None,
        "ollama_available": llm.is_available(),
        "ollama_model_available": llm.ensure_model_available() if llm.is_available() else False,
        "gpu_available": torch.cuda.is_available(),
    }


@app.post("/predict", response_model=CaseResult, tags=["Inference"])
async def predict(
    clinical_image: Optional[UploadFile] = File(None),
    dermoscopy_image: Optional[UploadFile] = File(None),
    patient_text: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None),
):
    """Submit a multimodal case for dermatological analysis.

    Accepts:
    - clinical_image: optional JPEG/PNG clinical photograph
    - dermoscopy_image: optional JPEG/PNG dermoscopy image
    - patient_text: optional free-text patient description

    Returns:
    - PipelineResult with action, diagnosis, confidence, report
    """
    case_id = str(uuid.uuid4())[:8]

    # The backends own the exact preprocessing the trained projector expects
    # (Resize(224,224) -> ToTensor -> ImageNet normalise), so hand them the PIL
    # image rather than pre-transforming it here with a different recipe.
    clinical_pil = None
    dermo_pil = None

    if clinical_image is not None:
        try:
            clinical_pil = Image.open(io.BytesIO(await clinical_image.read())).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid clinical image: {e}")

    if dermoscopy_image is not None:
        try:
            dermo_pil = Image.open(io.BytesIO(await dermoscopy_image.read())).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid dermoscopy image: {e}")

    parsed_metadata = None
    if metadata:
        import json as _json
        try:
            parsed_metadata = _json.loads(metadata)
            if not isinstance(parsed_metadata, dict):
                raise ValueError("metadata must be a JSON object")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid metadata JSON: {e}")

    if clinical_pil is None and not patient_text and not parsed_metadata:
        raise HTTPException(status_code=400, detail="At least one input is required.")

    # Run pipeline
    from src.agents.orchestrator import PatientCase
    case = PatientCase(
        case_id=case_id,
        image=clinical_pil,
        dermoscopy_image=dermo_pil,
        metadata=parsed_metadata,
        text=patient_text,
    )

    pipeline = get_pipeline()
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=f"Pipeline not available: {_pipeline_error or 'unknown initialization error'}")

    try:
        result = pipeline.process_case(case)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    result_dict = result.to_dict()
    _result_store[case_id] = result_dict

    return CaseResult(
        case_id=case_id,
        action=result.action,
        diagnosis=result.diagnosis,
        diagnosis_name=result.diagnosis_name,
        confidence=result.confidence,
        consensus_level=result.consensus_level,
        report=result.report,
        evidence_used=result.evidence_used,
        evidence_missing=result.evidence_missing,
        fusion_weights=result.fusion_weights,
        specialist_votes=result.specialist_votes,
        acquire_target=result.acquire_target,
        escalate_reason=result.escalate_reason,
        processing_time_s=result.processing_time_s,
    )


@app.get("/case/{case_id}", tags=["Inference"])
def get_case(case_id: str):
    """Retrieve results for a previously submitted case."""
    if case_id not in _result_store:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found.")
    return _result_store[case_id]


@app.get("/cases", tags=["Inference"])
def list_cases():
    """List all processed case IDs."""
    return {"case_ids": list(_result_store.keys()), "total": len(_result_store)}


@app.get("/metrics", tags=["System"])
def system_metrics():
    """Basic system metrics."""
    return {
        "total_cases_processed": len(_result_store),
        "accept_count": sum(1 for r in _result_store.values() if r.get("action") == "ACCEPT"),
        "arbitrate_count": sum(1 for r in _result_store.values() if r.get("action") == "ARBITRATE"),
        "acquire_count": sum(1 for r in _result_store.values() if r.get("action") == "ACQUIRE"),
        "escalate_count": sum(1 for r in _result_store.values() if r.get("action") == "ESCALATE"),
        "abstain_count": sum(1 for r in _result_store.values() if r.get("action") == "ABSTAIN"),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
