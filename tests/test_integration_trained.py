"""
tests/test_integration_trained.py - End-to-end tests against the real artifacts.

These load the actual trained checkpoints, so they are slower than the stubbed
orchestrator tests and are skipped automatically when the weights are absent
(e.g. on a fresh clone or in CI without the artifact directory).

Run just these with:  pytest tests/test_integration_trained.py -v
"""
import pytest
import torch
from PIL import Image

from src.constants import CLASS_NAMES, MODALITY_DIMS
from src.models.backends import TrainedBackends, default_weights_dir

pytestmark = pytest.mark.integration


def _weights_present() -> bool:
    d = default_weights_dir()
    return (d / "img_proj.pt").exists() and (d / "model_G" / "model_G.pt").exists()


requires_weights = pytest.mark.skipif(
    not _weights_present(),
    reason=f"trained artifacts not found at {default_weights_dir()}",
)

METADATA = {
    "age": 55, "gender": "FEMALE", "region": "NECK",
    "diameter_1": 6.0, "diameter_2": 5.0, "fitspatrick": 3,
    "itch": "TRUE", "grew": "TRUE", "hurt": "FALSE", "bleed": "TRUE",
    "elevation": "TRUE", "changed": "FALSE",
}
TEXT = "55-year-old female with a 6mm bleeding pigmented nodular lesion on the neck."


@pytest.fixture(scope="module")
def backends():
    return TrainedBackends()


@pytest.fixture(scope="module")
def sample_image():
    import glob
    matches = sorted(glob.glob("data/raw/images/*.png"))
    if not matches:
        pytest.skip("no sample images in data/raw/images")
    return Image.open(matches[0])


@requires_weights
class TestArtifactLoading:
    def test_all_specialists_present(self, backends):
        assert sorted(backends.available_specialists()) == list("ABCDEFG")

    @pytest.mark.parametrize("code", list("ABCDEFG"))
    def test_specialist_loads_strictly(self, backends, code):
        """A strict load is what proves architectures.py still matches the weights."""
        model = backends.specialist(code)
        assert model is not None
        assert not model.training

    def test_text_projector_loads_strictly(self, backends):
        """REGRESSION: the shipped predict.py loaded this non-strictly into the
        wrong class, silently keeping random weights."""
        proj = backends.text_proj
        out = proj(torch.zeros(1, 768))
        assert out.shape == (1, 256)
        assert torch.isfinite(out).all()


@requires_weights
class TestEncoding:
    def test_encode_all_modalities(self, backends, sample_image):
        enc = backends.encode_case(image=sample_image, metadata=METADATA, text=TEXT)
        assert enc.notes == []
        assert enc.image.shape == (1, MODALITY_DIMS["clinical_image"])
        assert enc.metadata.shape == (1, MODALITY_DIMS["metadata"])
        assert enc.text.shape == (1, MODALITY_DIMS["text"])
        assert sorted(enc.available()) == ["image", "metadata", "text"]

    def test_partial_case_encodes_only_what_is_given(self, backends):
        enc = backends.encode_case(metadata=METADATA)
        assert enc.available() == ["metadata"]
        assert enc.image is None and enc.text is None

    def test_encoding_is_deterministic(self, backends, sample_image):
        a = backends.encode_case(image=sample_image).image
        b = backends.encode_case(image=sample_image).image
        assert torch.allclose(a, b, atol=1e-5)


@requires_weights
class TestRouting:
    def test_full_case_routes_to_all_specialists(self, backends):
        assert sorted(backends.specialists_for(["image", "metadata", "text"])) == list("ABCDEFG")

    def test_image_only_routes_to_image_model(self, backends):
        assert backends.specialists_for(["image"]) == ["A"]

    def test_image_text_excludes_metadata_models(self, backends):
        codes = set(backends.specialists_for(["image", "text"]))
        assert codes == {"A", "C", "E"}
        assert not codes & {"B", "D", "F", "G"}

    def test_missing_modality_raises_not_guesses(self, backends):
        enc = backends.encode_case(metadata=METADATA)
        with pytest.raises(ValueError, match="needs image"):
            backends.run_specialist("A", enc)


@requires_weights
class TestSpecialistPredictions:
    def test_predictions_are_valid_distributions(self, backends, sample_image):
        enc = backends.encode_case(image=sample_image, metadata=METADATA, text=TEXT)
        for code in backends.specialists_for(enc.available()):
            probs = backends.probs(backends.run_specialist(code, enc))
            assert probs.shape == (len(CLASS_NAMES),)
            assert probs.min() >= 0.0
            assert probs.sum() == pytest.approx(1.0, abs=1e-5)


@requires_weights
class TestEndToEndPipeline:
    @pytest.fixture(scope="class")
    def orchestrator(self):
        from src.agents.orchestrator import OrchestratorAgent
        return OrchestratorAgent.build()

    def test_full_case_produces_a_decision(self, orchestrator, sample_image):
        from src.agents.orchestrator import PatientCase
        result = orchestrator.process_case(PatientCase(
            case_id="integration", image=sample_image, metadata=METADATA, text=TEXT))
        assert result.action in ("ACCEPT", "ARBITRATE", "ACQUIRE", "ESCALATE")
        assert result.action != "ERROR", result.escalate_reason
        steps = [t["step"] for t in result.evidence_trace]
        assert "fusion" in steps and "consensus" in steps

    def test_diagnosis_carries_a_report(self, orchestrator, sample_image):
        from src.agents.orchestrator import PatientCase
        result = orchestrator.process_case(PatientCase(
            case_id="integration", image=sample_image, metadata=METADATA, text=TEXT))
        if result.action in ("ACCEPT", "ARBITRATE"):
            assert result.diagnosis_name in CLASS_NAMES
            assert result.report and "DERMA-GUARD" in result.report
            # A named diagnosis must never be reported without its disclaimer.
            assert "not clinically validated" in result.report.lower()

    def test_empty_case_abstains(self, orchestrator):
        from src.agents.orchestrator import PatientCase
        result = orchestrator.process_case(PatientCase(case_id="empty"))
        assert result.action == "ABSTAIN"
        assert result.diagnosis is None

    def test_degraded_case_does_not_error(self, orchestrator, sample_image):
        from src.agents.orchestrator import PatientCase
        for kwargs in ({"image": sample_image}, {"metadata": METADATA}, {"text": TEXT}):
            result = orchestrator.process_case(PatientCase(case_id="degraded", **kwargs))
            assert result.action != "ERROR", result.escalate_reason
