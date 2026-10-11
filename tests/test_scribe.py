"""
tests/test_scribe.py - Scribe agent: PTIS, symptom extraction, metadata quality.

The symptom-extraction tests pin the safety property from proj.md Part 3: an
unmentioned symptom must come back as `unknown`, never as `false`. Silently
turning "not stated" into "patient denies it" is how a text pipeline invents
negative findings.
"""
import pytest
import torch
from unittest.mock import MagicMock

from src.agents.scribe import (
    PTISv1,
    ScribeAgent,
    ScribeReport,
    compute_ptis_features,
)
from src.constants import MODALITY_DIMS

RICH = ("55-year-old female with a 6mm bleeding pigmented nodular lesion on the neck "
        "that has grown over three months, itches, and has irregular brown borders.")
VAGUE = "a thing"


@pytest.fixture
def scribe():
    return ScribeAgent()


class TestPTIS:
    def test_empty_text_scores_zero(self):
        assert PTISv1().score("") == 0.0
        assert PTISv1().score("   ") == 0.0

    def test_rich_beats_vague(self):
        s = PTISv1()
        assert s.score(RICH) > s.score(VAGUE)

    def test_bounded_to_unit_interval(self):
        s = PTISv1()
        for text in ("", VAGUE, RICH, RICH * 10):
            assert 0.0 <= s.score(text) <= 1.0

    def test_vagueness_is_penalised(self):
        s = PTISv1()
        plain = "The lesion on my arm has grown for three months and bleeds."
        hedged = "Maybe the lesion on my arm has kind of grown, I'm not sure, sort of bleeds."
        assert s.score(plain) > s.score(hedged)

    def test_feature_vector_shape(self):
        assert compute_ptis_features(RICH).shape == (7,)

    def test_features_are_finite_for_empty_text(self):
        assert torch.isfinite(compute_ptis_features("")).all()


class TestSymptomExtraction:
    def test_unmentioned_symptoms_are_unknown_not_false(self, scribe):
        """The safety property: absence of evidence != evidence of absence."""
        out = scribe.extract_symptoms("I have a lesion on my arm.")
        assert set(out) == set(ScribeAgent.SYMPTOM_FIELDS)
        assert out["bleeding"] == "unknown"
        assert out["pain"] == "unknown"
        assert "false" not in out.values() or all(
            v != "false" for k, v in out.items() if k in ("bleeding", "pain"))

    def test_mentioned_symptoms_are_detected(self, scribe):
        out = scribe.extract_symptoms(RICH)
        assert out["bleeding"] == "true"
        assert out["itching"] == "true"
        assert out["growth"] == "true"

    def test_every_field_always_present(self, scribe):
        for text in ("", VAGUE, RICH):
            assert set(scribe.extract_symptoms(text)) == set(ScribeAgent.SYMPTOM_FIELDS)

    def test_values_come_from_the_closed_vocabulary(self, scribe):
        for v in scribe.extract_symptoms(RICH).values():
            assert v in ("true", "false", "unknown")

    def test_llm_failure_falls_back_to_heuristics(self):
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("ollama unreachable")
        out = ScribeAgent(llm_client=llm).extract_symptoms(RICH)
        assert out["bleeding"] == "true"          # heuristic still fires

    def test_malformed_llm_json_falls_back(self):
        llm = MagicMock()
        llm.generate.return_value = "this is not json at all"
        out = ScribeAgent(llm_client=llm).extract_symptoms(RICH)
        assert set(out) == set(ScribeAgent.SYMPTOM_FIELDS)


class TestQuestionGeneration:
    def test_produces_a_question(self, scribe):
        q = scribe.generate_question("growth")
        assert isinstance(q, str) and q.strip().endswith("?")

    def test_unknown_field_still_yields_a_question(self, scribe):
        q = scribe.generate_question("some_unmapped_field")
        assert isinstance(q, str) and q.strip()


class TestMetadataCompleteness:
    def test_empty_metadata_scores_zero(self, scribe):
        quality, missing = scribe.metadata_completeness(None)
        assert quality == 0.0
        assert len(missing) == len(ScribeAgent.EXPECTED_META_FIELDS)

    def test_complete_metadata_scores_one(self, scribe):
        meta = {f: "TRUE" for f in ScribeAgent.EXPECTED_META_FIELDS}
        quality, missing = scribe.metadata_completeness(meta)
        assert quality == pytest.approx(1.0)
        assert missing == []

    def test_unknown_sentinels_count_as_missing(self, scribe):
        """PAD-UFES writes unknowns as 'UNK'; treating that as a value would
        inflate Q_meta on records that carry no information."""
        meta = {f: "TRUE" for f in ScribeAgent.EXPECTED_META_FIELDS}
        meta["age"] = "UNK"
        meta["region"] = ""
        meta["gender"] = None
        quality, missing = scribe.metadata_completeness(meta)
        assert set(missing) == {"age", "region", "gender"}
        assert quality < 1.0

    def test_quality_is_monotone_in_completeness(self, scribe):
        full = {f: "TRUE" for f in ScribeAgent.EXPECTED_META_FIELDS}
        half = dict(list(full.items())[: len(full) // 2])
        assert scribe.metadata_completeness(full)[0] > scribe.metadata_completeness(half)[0]


class TestProcessCase:
    def test_returns_a_populated_report(self, scribe):
        report = scribe.process_case(RICH, metadata={"age": 55, "region": "NECK"})
        assert isinstance(report, ScribeReport)
        assert 0.0 <= report.ptis <= 1.0
        assert 0.0 <= report.metadata_quality <= 1.0
        assert report.metadata_embedding.shape == (MODALITY_DIMS["metadata"],)
        assert report.summary

    def test_empty_case_is_handled(self, scribe):
        report = scribe.process_case("", metadata=None)
        assert report.ptis == 0.0
        assert report.metadata_quality == 0.0

    def test_precomputed_metadata_vector_is_used(self, scribe):
        vec = torch.arange(MODALITY_DIMS["metadata"], dtype=torch.float32)
        report = scribe.process_case("lesion", metadata={"age": 40}, metadata_vector=vec)
        assert torch.allclose(report.metadata_embedding, vec)

    def test_summary_reflects_informativeness(self, scribe):
        assert "low" in scribe.process_case(VAGUE, metadata={"age": 1}).summary.lower()

    def test_risk_factors_extracted(self, scribe):
        report = scribe.process_case(
            "I spend a lot of time in the sun and my father had skin cancer.",
            metadata={"age": 60})
        assert "sun_exposure" in report.risk_factors
        assert "family_history" in report.risk_factors
