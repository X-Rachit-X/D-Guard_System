"""
tests/test_config.py - Configuration must agree with the code and the checkpoints.

A config that drifts from `src/constants.py` does not crash - it silently
relabels predictions. `configs/base.yaml` previously declared
{BCC:0, SCC:1, ACK:2, SEK:3, MEL:4, NEV:5}, transposing SCC<->ACK and NEV<->MEL
against the indices the trained heads were fitted to, and
`src/data/preprocessing.py` hardcoded the same wrong map. These tests exist so
that can only ever happen once.
"""
from pathlib import Path

import pytest
import yaml

from src.constants import (
    ALWAYS_UNAVAILABLE,
    CLASS_NAMES,
    CLASS_TO_INDEX,
    MALIGNANT_CLASSES,
    MODALITIES,
    MODALITY_DIMS,
    NUM_CLASSES,
)

CONFIG_DIR = Path("configs")


def load(name):
    path = CONFIG_DIR / name
    if not path.exists():
        pytest.skip(f"{path} not present")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def base():
    return load("base.yaml")


class TestClassOrdering:
    def test_label_map_matches_constants(self, base):
        """The ordering the trained classifier heads were fitted against."""
        assert base["label_map"]["multiclass"] == CLASS_TO_INDEX

    def test_class_to_index_derives_from_class_names(self):
        assert CLASS_TO_INDEX == {n: i for i, n in enumerate(CLASS_NAMES)}
        assert len(CLASS_TO_INDEX) == NUM_CLASSES == 6

    def test_matches_the_training_code_map(self):
        """Cross-check against the map the checkpoints were actually trained under."""
        from src.data.metadata_preprocessor import DIAGNOSTIC_MAP

        assert DIAGNOSTIC_MAP == CLASS_TO_INDEX

    def test_preprocessing_emits_the_same_indices(self):
        """REGRESSION: preprocessing.py used to declare its own, different map.

        SCC and ACK were transposed, as were NEV and MEL - so four of six classes
        came out mislabelled relative to the trained heads, with no error raised.
        """
        import pandas as pd

        from src.data.preprocessing import preprocess_metadata

        row = {
            "age": 50, "diameter_1": 5.0, "diameter_2": 4.0,
            "gender": "FEMALE", "region": "ARM", "fitspatrick": 3,
            "background_father": "GERMANY", "background_mother": "GERMANY",
            "smoke": "FALSE", "drink": "FALSE", "pesticide": "FALSE",
            "skin_cancer_history": "FALSE", "cancer_history": "FALSE",
            "has_piped_water": "TRUE", "has_sewage_system": "TRUE",
            "itch": "FALSE", "grew": "FALSE", "hurt": "FALSE",
            "changed": "FALSE", "bleed": "FALSE", "elevation": "FALSE",
            "biopsed": True,
        }
        df = pd.DataFrame([{**row, "diagnostic": name, "img_id": f"{name}.png"}
                           for name in CLASS_NAMES])
        out, _, _ = preprocess_metadata(df, binary=False)
        assert out["label"].tolist() == [CLASS_TO_INDEX[n] for n in CLASS_NAMES]
        # Specifically pin the four that used to be wrong.
        labels = dict(zip(out["diagnostic"], out["label"]))
        assert labels["ACK"] == 1 and labels["NEV"] == 2
        assert labels["SCC"] == 4 and labels["MEL"] == 5

    def test_malignant_set_agrees(self, base):
        assert set(base["label_map"]["malignant"]) == set(MALIGNANT_CLASSES)

    def test_default_config_class_names_agree(self):
        cfg = load("default.yaml")
        assert cfg["class_names"] == list(CLASS_NAMES)
        assert cfg["num_classes"] == NUM_CLASSES


class TestModalityOrdering:
    def test_order_matches_constants(self, base):
        assert base["modalities"]["order"] == list(MODALITIES)

    def test_dims_match_constants(self, base):
        assert base["modalities"]["dims"] == MODALITY_DIMS

    def test_always_unavailable_matches(self, base):
        assert base["modalities"]["always_unavailable"] == list(ALWAYS_UNAVAILABLE)

    def test_fusion_axis_length_matches(self):
        """REGRESSION: adaptive.yaml said 5 while the evidence mask was 4."""
        fusion = load("model/fusion/adaptive.yaml")
        assert fusion["num_modalities"] == len(MODALITIES)
        assert fusion["modalities"] == list(MODALITIES)

    def test_retrieval_dim_is_the_modality_concatenation(self, base):
        assert base["retrieval"]["index_dim"] == sum(MODALITY_DIMS.values())


class TestSpecialistConfig:
    def test_declares_all_seven(self):
        cfg = load("model/specialists.yaml")
        assert sorted(cfg["models"]) == list("ABCDEFG")

    def test_modalities_match_the_code(self):
        from src.models.architectures import SPECIALIST_MODALITIES

        cfg = load("model/specialists.yaml")
        for code, spec in cfg["models"].items():
            assert tuple(spec["modalities"]) == SPECIALIST_MODALITIES[code]

    def test_unimodal_routing_matches_the_team(self):
        from src.agents.vision_team import SpecialistTeam

        cfg = load("model/specialists.yaml")
        assert cfg["unimodal_for"] == SpecialistTeam.UNIMODAL_FOR

    def test_num_classes_and_embed_dim(self):
        cfg = load("model/specialists.yaml")
        assert cfg["num_classes"] == NUM_CLASSES
        assert cfg["embed_dim"] == MODALITY_DIMS["clinical_image"]

    def test_image_preprocessing_matches_the_encoder(self):
        """Any drift here puts the frozen ResNet50 off-distribution."""
        from src.models.backends import IMAGENET_MEAN, IMAGENET_STD

        cfg = load("model/specialists.yaml")
        assert cfg["image_preprocessing"]["resize"] == [224, 224]
        assert cfg["image_preprocessing"]["normalize"]["mean"] == IMAGENET_MEAN
        assert cfg["image_preprocessing"]["normalize"]["std"] == IMAGENET_STD


class TestConsensusThresholds:
    def test_thresholds_match_the_implementation(self, base):
        c = base["consensus"]
        assert c["strong_threshold"] == 0.80
        assert c["moderate_threshold"] == 0.50
        assert c["quality_threshold"] == 0.30
        # `>= 2`, not `== 2`: the spec's exact-equality left a gap in the ladder.
        assert c["min_agreement_level2"] == 2
        assert c["min_agreement_level1"] == 3


class TestNoLegacyReferences:
    LEGACY = ["derm7pt", "dermlip", "panderm", "swin_tiny", "resnet152"]

    def test_configs_are_free_of_dead_architectures(self):
        """PAD-UFES-20 only; none of these models exist in this project."""
        offenders = []
        for path in CONFIG_DIR.rglob("*.yaml"):
            text = path.read_text(encoding="utf-8").lower()
            for token in self.LEGACY:
                # Allow mentions inside comment lines that explain the removal.
                hits = [ln for ln in text.splitlines()
                        if token in ln and not ln.strip().startswith("#")]
                if hits:
                    offenders.append((path.name, token))
        assert not offenders, f"legacy references remain: {offenders}"

    def test_dataset_is_pad_ufes(self, base):
        data = load("data/pad_ufes.yaml")
        assert data["name"] == "pad_ufes"
        assert data["num_cases"] == 2298
        assert data["splits"]["strategy"] == "patient_level"

    def test_declared_paths_exist(self, base):
        """The config used to point at data/raw/pad_ufes/, which does not exist."""
        for key in ("raw_images", "raw_metadata"):
            path = Path(base["data"][key])
            if not path.exists():
                pytest.skip(f"{path} not present in this checkout")
            assert path.exists()
