"""
tests/test_fusion.py - DiagnosisAdaptiveFusion tests (DERMA-GUARD v4.0)

Covers:
  - convergence within max_iterations
  - weights sum to 1.0 over available modalities
  - missing modalities receive exactly zero weight
  - all-missing raises InsufficientEvidenceError rather than dividing by zero
  - REGRESSION: a mask whose length disagrees with MODALITIES must raise, not be
    silently swallowed (this is the defect that left fusion dead in production)
  - the fused prediction is driven by the supplied specialist logits
"""
import pytest
import torch
import torch.nn.functional as F

from src.constants import MODALITIES, NUM_CLASSES, NUM_MODALITIES
from src.evidence.missingness import compute_missingness, mask_from_available
from src.fusion.adaptive_fusion import DiagnosisAdaptiveFusion, InsufficientEvidenceError

D_COMMON = 64


@pytest.fixture
def fusion_model():
    return DiagnosisAdaptiveFusion(
        num_classes=NUM_CLASSES, d_common=D_COMMON,
        max_iterations=3, convergence_threshold=0.05,
    )


def make_embeddings(modalities=("clinical_image", "metadata", "text")):
    from src.constants import MODALITY_DIMS
    return {m: torch.randn(1, MODALITY_DIMS[m]) for m in modalities}


def evidence_dicts(modalities, q=0.7, r=0.8, u=0.5):
    return ({m: q for m in modalities},
            {m: r for m in modalities},
            {m: u for m in modalities})


class TestModalityContract:
    def test_default_matches_constants(self, fusion_model):
        assert fusion_model.num_modalities == NUM_MODALITIES
        assert fusion_model.MODALITY_NAMES == list(MODALITIES)

    def test_contradicting_num_modalities_raises(self):
        """Constructing with a different axis length must fail immediately."""
        with pytest.raises(ValueError, match="contradicts src.constants.MODALITIES"):
            DiagnosisAdaptiveFusion(num_modalities=5)

    def test_wrong_sized_mask_raises(self, fusion_model):
        """REGRESSION: a 5-wide mask against a 4-wide axis must raise.

        The original code paired a 5-model fusion axis with the 4-modality
        evidence mask; the broadcast error was caught by the orchestrator and
        turned into a fabricated zero-confidence prediction, so every test
        passed while fusion never actually ran.
        """
        mods = ("clinical_image", "metadata", "text")
        q, r, u = evidence_dicts(mods)
        with pytest.raises(ValueError, match="entries but fusion expects"):
            fusion_model.fuse(make_embeddings(mods), q, r, u, 0.8, torch.ones(5))


class TestFusionConvergence:
    def test_converges_within_max_iterations(self, fusion_model):
        mods = ("clinical_image", "metadata", "text")
        q, r, u = evidence_dicts(mods)
        result = fusion_model.fuse(make_embeddings(mods), q, r, u, 0.9,
                                   mask_from_available(mods))
        assert result.iterations_used <= fusion_model.max_iterations
        assert result.iterations_used >= 1

    def test_reports_convergence_flag(self, fusion_model):
        mods = ("clinical_image", "metadata")
        q, r, u = evidence_dicts(mods)
        result = fusion_model.fuse(make_embeddings(mods), q, r, u, 0.9,
                                   mask_from_available(mods))
        assert isinstance(result.converged, bool)


class TestWeightNormalization:
    def test_weights_sum_to_one(self, fusion_model):
        mods = ("clinical_image", "metadata", "text")
        q, r, u = evidence_dicts(mods)
        result = fusion_model.fuse(make_embeddings(mods), q, r, u, 0.8,
                                   mask_from_available(mods))
        assert result.weights.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_missing_modality_weight_is_zero(self, fusion_model):
        mods = ("clinical_image", "metadata")
        q, r, u = evidence_dicts(mods)
        result = fusion_model.fuse(make_embeddings(mods), q, r, u, 0.8,
                                   mask_from_available(mods))
        w = result.weight_dict()
        assert w["dermoscopy_image"] == 0.0
        assert w["text"] == 0.0
        assert w["clinical_image"] > 0.0 and w["metadata"] > 0.0

    def test_single_modality_gets_all_weight(self, fusion_model):
        mods = ("metadata",)
        q, r, u = evidence_dicts(mods)
        result = fusion_model.fuse(make_embeddings(mods), q, r, u, 1.0,
                                   mask_from_available(mods))
        assert result.weight_dict()["metadata"] == pytest.approx(1.0, abs=1e-5)

    def test_low_evidence_modality_is_downweighted(self, fusion_model):
        """Evidence scores must actually steer the weights."""
        mods = ("clinical_image", "text")
        embs = make_embeddings(mods)
        q = {"clinical_image": 0.95, "text": 0.10}
        r = {"clinical_image": 0.95, "text": 0.10}
        u = {"clinical_image": 0.5, "text": 0.5}
        w = fusion_model.fuse(embs, q, r, u, 0.8, mask_from_available(mods)).weight_dict()
        assert w["clinical_image"] > w["text"]


class TestZeroDivisionGuard:
    def test_all_missing_raises_insufficient_evidence(self, fusion_model):
        miss = compute_missingness(False, False, False, False)
        assert miss["all_missing"] is True
        with pytest.raises(InsufficientEvidenceError):
            fusion_model.fuse({}, {}, {}, {}, 1.0, miss["missing_mask"])

    def test_mask_available_but_no_embeddings_raises(self, fusion_model):
        """A mask claiming availability with nothing supplied must not divide by zero."""
        with pytest.raises(InsufficientEvidenceError):
            fusion_model.fuse({}, {}, {}, {}, 1.0, mask_from_available(("metadata",)))


class TestSpecialistPooling:
    def test_prediction_follows_supplied_logits(self, fusion_model):
        """With trained specialist logits, the pooled answer must follow them."""
        mods = ("clinical_image", "metadata", "text")
        target = 3
        logits = {}
        for m in mods:
            lg = torch.full((1, NUM_CLASSES), -5.0)
            lg[0, target] = 5.0
            logits[m] = lg
        q, r, u = evidence_dicts(mods)
        result = fusion_model.fuse(make_embeddings(mods), q, r, u, 1.0,
                                   mask_from_available(mods), modality_logits=logits)
        assert result.diagnosis == target
        assert result.confidence > 0.9

    def test_disagreement_lowers_confidence(self, fusion_model):
        mods = ("clinical_image", "metadata")
        agree, disagree = {}, {}
        for i, m in enumerate(mods):
            same = torch.full((1, NUM_CLASSES), -5.0); same[0, 1] = 5.0
            agree[m] = same
            diff = torch.full((1, NUM_CLASSES), -5.0); diff[0, i] = 5.0
            disagree[m] = diff
        q, r, u = evidence_dicts(mods)
        mask = mask_from_available(mods)
        c_agree = fusion_model.fuse(make_embeddings(mods), q, r, u, 1.0, mask,
                                    modality_logits=agree).confidence
        c_disagree = fusion_model.fuse(make_embeddings(mods), q, r, u, 1.0, mask,
                                       modality_logits=disagree).confidence
        assert c_agree > c_disagree
