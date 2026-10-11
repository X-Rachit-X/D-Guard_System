"""
tests/test_case_loader.py - PAD-UFES-20 case loading and split integrity.

The split tests are the important ones: the shipped checkpoints were trained
under overlapping partitions (100% of the test set was training data), so these
assert the property that failure violated.
"""
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.case_loader import (
    image_for_pipeline,
    load_cases,
    load_image_tensor,
    patient_level_split,
)

METADATA_CSV = Path("data/raw/metadata.csv")
IMAGE_DIR = Path("data/raw/images")

requires_data = pytest.mark.skipif(
    not METADATA_CSV.exists(), reason=f"{METADATA_CSV} not present")


def synthetic_cases(n_patients=20, per_patient=3):
    return [{
        "case_id": f"p{p}_l{l}",
        "patient_id": f"p{p}",
        "label": p % 6,
        "diagnostic": ["BCC", "ACK", "NEV", "SEK", "SCC", "MEL"][p % 6],
        "metadata": {"age": 40 + p},
        "patient_text": "lesion",
        "clinical_image": None,
    } for p in range(n_patients) for l in range(per_patient)]


class TestPatientLevelSplit:
    def test_partitions_are_disjoint_by_case(self):
        split = patient_level_split(synthetic_cases())
        ids = {k: {c["case_id"] for c in v} for k, v in split.items()}
        assert not (ids["train"] & ids["val"])
        assert not (ids["train"] & ids["test"])
        assert not (ids["val"] & ids["test"])

    def test_no_patient_spans_two_partitions(self):
        """The property an image-level split violates: one patient's lesions
        must not straddle the boundary."""
        split = patient_level_split(synthetic_cases())
        pats = {k: {c["patient_id"] for c in v} for k, v in split.items()}
        assert not (pats["train"] & pats["val"])
        assert not (pats["train"] & pats["test"])
        assert not (pats["val"] & pats["test"])

    def test_every_case_lands_somewhere(self):
        cases = synthetic_cases()
        split = patient_level_split(cases)
        assert sum(len(v) for v in split.values()) == len(cases)

    def test_reproducible_under_seed(self):
        cases = synthetic_cases()
        a = patient_level_split(cases, seed=42)
        b = patient_level_split(cases, seed=42)
        assert [c["case_id"] for c in a["test"]] == [c["case_id"] for c in b["test"]]

    def test_different_seeds_differ(self):
        cases = synthetic_cases(n_patients=50)
        a = patient_level_split(cases, seed=1)["test"]
        b = patient_level_split(cases, seed=2)["test"]
        assert {c["case_id"] for c in a} != {c["case_id"] for c in b}

    def test_bad_ratios_rejected(self):
        with pytest.raises(ValueError, match="must sum to 1.0"):
            patient_level_split(synthetic_cases(), ratios=(0.7, 0.2, 0.2))

    def test_ratios_are_approximately_honoured(self):
        split = patient_level_split(synthetic_cases(n_patients=100), ratios=(0.7, 0.15, 0.15))
        total = sum(len(v) for v in split.values())
        assert 0.6 < len(split["train"]) / total < 0.8


class TestImageHelpers:
    def test_image_for_pipeline_roundtrip(self, tmp_path):
        arr = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        path = tmp_path / "x.png"
        Image.fromarray(arr).save(path)

        t = load_image_tensor(path)
        assert t.shape == (1, 3, 224, 224)
        assert 0.0 <= t.min() and t.max() <= 1.0

        pil = image_for_pipeline(t)
        assert isinstance(pil, Image.Image)
        assert pil.size == (224, 224)

    def test_image_for_pipeline_passes_none_through(self):
        assert image_for_pipeline(None) is None

    def test_accepts_unbatched(self):
        assert image_for_pipeline(torch.rand(3, 32, 32)).size == (32, 32)


@requires_data
class TestLoadCases:
    def test_loads_the_full_dataset(self):
        cases = load_cases(load_images=False)
        assert len(cases) == 2298
        assert all(c["label"] in range(6) for c in cases)

    def test_metadata_omits_missing_fields(self):
        """Absent values must be dropped, not imputed - Q_meta counts them."""
        cases = load_cases(load_images=False, limit=50)
        for c in cases:
            assert None not in c["metadata"].values()
            assert not any(str(v).upper() in ("UNK", "NAN", "") for v in c["metadata"].values())

    def test_dermoscopy_always_unavailable(self):
        for c in load_cases(load_images=False, limit=20):
            assert c["dermo_available"] is False

    def test_limit_keeps_rare_classes(self):
        """Stratified subsampling must not drop MEL (52 of 2298)."""
        cases = load_cases(load_images=False, limit=120)
        assert len({c["diagnostic"] for c in cases}) >= 5

    def test_real_split_is_disjoint(self):
        split = patient_level_split(load_cases(load_images=False))
        tr = {c["patient_id"] for c in split["train"]}
        te = {c["patient_id"] for c in split["test"]}
        va = {c["patient_id"] for c in split["val"]}
        assert not (tr & te) and not (tr & va) and not (va & te)
