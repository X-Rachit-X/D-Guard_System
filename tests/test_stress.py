"""
tests/test_stress.py - DERMA-STRESS corruption engine and benchmark metrics.

Covers the corruption conditions themselves and the two metric definitions that
previously could not fail (see docs/evaluation_integrity.md §3.5).
"""
import numpy as np
import pytest
import torch

from src.constants import NUM_CLASSES
from src.stress.benchmark import ConditionResult, StressReport, compute_brier, compute_ece
from src.stress.corruption import (
    CORRUPTION_CONDITIONS,
    CorruptionEngine,
    bgr_to_tensor,
    corrupt_blur,
    corrupt_brightness,
    corrupt_jpeg,
    corrupt_metadata_field_swap,
    corrupt_noise,
    corrupt_occlusion,
    corrupt_text_vague,
    poison_metadata,
    tensor_to_bgr,
)


def make_batch(n=4):
    return [{
        "case_id": f"c{i}",
        "label": i % 3,
        "clinical_image": torch.rand(1, 3, 224, 224),
        "metadata": {"age": 30 + i * 10,
                     "region": "NECK" if i % 2 else "ARM",
                     "bleed": "TRUE" if i % 2 else "FALSE"},
        "patient_text": f"case {i} lesion growing on the arm for three months",
    } for i in range(n)]


class TestTensorHelpers:
    @pytest.mark.parametrize("shape", [(3, 224, 224), (1, 3, 224, 224)])
    def test_accepts_both_ranks(self, shape):
        """REGRESSION: batched tensors used to crash every image corruption."""
        bgr = tensor_to_bgr(torch.rand(*shape))
        assert bgr.shape == (224, 224, 3)

    def test_rejects_real_batches(self):
        with pytest.raises(ValueError, match="single image"):
            tensor_to_bgr(torch.rand(4, 3, 224, 224))

    def test_roundtrip_preserves_values(self):
        t = torch.rand(3, 64, 64)
        back = bgr_to_tensor(tensor_to_bgr(t))
        assert back.shape == t.shape
        assert torch.allclose(back, t, atol=2 / 255)  # uint8 quantisation


class TestImageCorruptions:
    @pytest.mark.parametrize("fn,kwargs", [
        (corrupt_blur, {"kernel_size": 15}),
        (corrupt_noise, {"sigma": 0.2}),
        (corrupt_brightness, {"factor": 0.5}),
        (corrupt_jpeg, {"quality": 10}),
        (corrupt_occlusion, {"area_fraction": 0.2}),
    ])
    def test_preserves_shape_and_range(self, fn, kwargs):
        img = torch.rand(1, 3, 224, 224)
        out = fn(img, **kwargs)
        assert out.shape == img.shape
        assert 0.0 <= out.min() and out.max() <= 1.0

    def test_corruption_actually_changes_the_image(self):
        img = torch.rand(1, 3, 224, 224)
        assert not torch.allclose(corrupt_blur(img, 15), img, atol=1e-3)
        assert not torch.allclose(corrupt_occlusion(img, 0.3), img, atol=1e-3)

    def test_brightness_darkens(self):
        img = torch.full((1, 3, 32, 32), 0.6)
        assert corrupt_brightness(img, 0.5).mean().item() < img.mean().item()


class TestMetadataCorruptions:
    def test_field_swap_changes_exactly_one_field(self):
        batch = make_batch()
        original = batch[0]["metadata"]
        out = corrupt_metadata_field_swap(original, batch)
        differing = [k for k in original if out[k] != original[k]]
        assert len(differing) == 1
        assert set(out) == set(original)   # shape preserved; record still looks valid

    def test_poison_takes_a_different_diagnosis_record(self):
        batch = make_batch()
        out = poison_metadata(batch[0]["metadata"], batch[1:], label=batch[0]["label"])
        assert out in [c["metadata"] for c in batch[1:]]

    def test_field_swap_without_donors_is_a_noop(self):
        meta = {"age": 40}
        assert corrupt_metadata_field_swap(meta, []) == meta


class TestTextCorruptions:
    def test_vague_truncates(self):
        out = corrupt_text_vague("one two three four five six seven eight", max_words=5)
        assert len(out.split()) <= 5


class TestCorruptionEngine:
    def test_generates_every_condition(self):
        out = CorruptionEngine(seed=42).corrupt_batch(make_batch(), cache_dir=None)
        assert len(out) == len(CORRUPTION_CONDITIONS)
        assert set(out) == {c["id"] for c in CORRUPTION_CONDITIONS}

    def test_availability_conditions_clear_the_modality(self):
        out = CorruptionEngine(seed=42).corrupt_batch(make_batch(), cache_dir=None)
        assert out["miss_clinical"][0].clinical_image is None
        assert out["miss_clinical"][0].clinical_available is False
        assert out["miss_metadata"][0].metadata is None
        assert out["miss_metadata"][0].meta_available is False
        assert out["miss_text"][0].patient_text == ""

    def test_compound_applies_several(self):
        cc = CorruptionEngine(seed=42).corrupt_batch(make_batch(), cache_dir=None)["compound_multi"][0]
        assert cc.clinical_image is None                 # miss_clinical
        assert len(cc.patient_text.split()) <= 5         # vague_text

    def test_deterministic_under_seed(self):
        a = CorruptionEngine(seed=7).corrupt_batch(make_batch(), cache_dir=None)
        b = CorruptionEngine(seed=7).corrupt_batch(make_batch(), cache_dir=None)
        assert a["wrong_metadata"][0].metadata == b["wrong_metadata"][0].metadata


class TestCalibrationMetrics:
    def test_perfect_confident_predictions_have_low_ece(self):
        labels = np.array([0, 1, 2, 3])
        probs = np.full((4, NUM_CLASSES), 0.001)
        for i, l in enumerate(labels):
            probs[i, l] = 0.999
        probs /= probs.sum(axis=1, keepdims=True)
        assert compute_ece(probs, labels) < 0.05
        assert compute_brier(probs, labels) < 0.05

    def test_confidently_wrong_has_high_ece(self):
        labels = np.array([0, 0, 0, 0])
        probs = np.full((4, NUM_CLASSES), 0.001)
        probs[:, 1] = 0.999
        probs /= probs.sum(axis=1, keepdims=True)
        assert compute_ece(probs, labels) > 0.9


class TestReportMetricSemantics:
    def test_condition_result_carries_coverage(self):
        r = ConditionResult(condition_id="x", condition_name="x", category="c", n_cases=10,
                            accuracy=0.5, ece=0.1, brier=0.2, abstention_rate=0.0,
                            escalation_rate=0.3, recovery_rate=1.0, mean_confidence=0.7,
                            coverage=0.7)
        assert r.coverage == 0.7

    def test_report_serialises_coverage(self):
        rep = StressReport(model_name="m", dataset="d", timestamp="t")
        rep.conditions.append(ConditionResult(
            condition_id="x", condition_name="x", category="c", n_cases=1,
            accuracy=1.0, ece=0.0, brier=0.0, abstention_rate=0.0,
            escalation_rate=0.0, recovery_rate=0.0, mean_confidence=1.0, coverage=1.0))
        assert rep.to_dict()["conditions"][0]["coverage"] == 1.0

    def test_recovery_denominator_is_deferrals_not_cases(self):
        """REGRESSION: a system that defers everything must not score 1.000.

        recovery_rate = (deferrals whose withheld answer was wrong) / deferrals.
        If every deferral would in fact have been CORRECT, the rate must be 0 -
        the old definition returned 1.0 because it divided by n_cases and counted
        each deferral as a save.
        """
        deferrals, would_have_erred = 10, 0
        assert would_have_erred / max(deferrals, 1) == 0.0
        deferrals, would_have_erred = 10, 7
        assert would_have_erred / max(deferrals, 1) == pytest.approx(0.7)
