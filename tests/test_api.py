"""
tests/test_api.py - FastAPI surface tests.

The /predict tests load the real pipeline, so they carry the `integration`
marker and skip when the trained artifacts are absent. Validation and routing
tests run without weights.
"""
import io
import json

import pytest

from src.constants import CLASS_NAMES
from src.models.backends import default_weights_dir


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from src.main import app
    return TestClient(app)


def _weights_present() -> bool:
    return (default_weights_dir() / "model_G" / "model_G.pt").exists()


requires_weights = pytest.mark.skipif(not _weights_present(), reason="trained artifacts absent")


def sample_image_bytes():
    import glob

    matches = sorted(glob.glob("data/raw/images/*.png"))
    if not matches:
        pytest.skip("no sample images available")
    with open(matches[0], "rb") as f:
        return f.read()


METADATA = {"age": 55, "gender": "FEMALE", "region": "NECK", "diameter_1": 6.0,
            "diameter_2": 5.0, "fitspatrick": 3, "itch": "TRUE", "grew": "TRUE",
            "hurt": "FALSE", "bleed": "TRUE", "elevation": "TRUE"}


class TestHealthEndpoints:
    def test_root_responds(self, client):
        assert client.get("/").status_code == 200

    def test_health_reports_status(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert "pipeline_loaded" in body


class TestValidation:
    def test_empty_request_rejected(self, client):
        assert client.post("/predict", data={}).status_code == 400

    def test_malformed_metadata_rejected(self, client):
        r = client.post("/predict", data={"patient_text": "a mole", "metadata": "{not json"})
        assert r.status_code == 400
        assert "metadata" in r.json()["detail"].lower()

    def test_non_object_metadata_rejected(self, client):
        r = client.post("/predict", data={"patient_text": "a mole", "metadata": "[1,2,3]"})
        assert r.status_code == 400

    def test_corrupt_image_rejected(self, client):
        r = client.post("/predict",
                        files={"clinical_image": ("x.png", io.BytesIO(b"not an image"), "image/png")})
        assert r.status_code == 400

    def test_unknown_case_is_404(self, client):
        assert client.get("/case/does-not-exist").status_code == 404


@pytest.mark.integration
@requires_weights
class TestPredict:
    def test_full_case(self, client):
        r = client.post(
            "/predict",
            files={"clinical_image": ("lesion.png", io.BytesIO(sample_image_bytes()), "image/png")},
            data={"patient_text": "Bleeding nodular lesion on the neck.",
                  "metadata": json.dumps(METADATA)},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["action"] in ("ACCEPT", "ARBITRATE", "ACQUIRE", "ESCALATE")
        assert body["action"] != "ERROR", body.get("escalate_reason")
        assert body["evidence_used"]
        assert "dermoscopy_image" in body["evidence_missing"]
        if body["action"] in ("ACCEPT", "ARBITRATE"):
            assert body["diagnosis_name"] in CLASS_NAMES
            assert 0.0 <= body["confidence"] <= 1.0
            assert body["specialist_votes"]
            assert sum(body["fusion_weights"].values()) == pytest.approx(1.0, abs=1e-4)

    def test_result_is_retrievable(self, client):
        r = client.post("/predict", data={"metadata": json.dumps(METADATA)})
        assert r.status_code == 200
        case_id = r.json()["case_id"]
        stored = client.get(f"/case/{case_id}")
        assert stored.status_code == 200
        assert stored.json()["case_id"] == case_id

    def test_metadata_only_case_is_handled(self, client):
        r = client.post("/predict", data={"metadata": json.dumps(METADATA)})
        assert r.status_code == 200
        assert r.json()["action"] != "ERROR"

    def test_metrics_counts_actions(self, client):
        client.post("/predict", data={"metadata": json.dumps(METADATA)})
        m = client.get("/metrics").json()
        assert m["total_cases_processed"] >= 1
        assert sum(m[k] for k in m if k.endswith("_count")) >= 1
