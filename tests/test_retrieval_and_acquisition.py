"""
tests/test_retrieval_and_acquisition.py - RAG index, Librarian, acquisition policy
and the retrospective utility guard.

The RAG tests pin the audit fix from proj.md Part 10: the index must hold
CONCATENATED PER-MODALITY embeddings, never a fused one. Indexing fused
embeddings is circular - fusion consumes retrieval, which would then consume
fusion.
"""
import numpy as np
import pytest
import torch

from src.agents.librarian import LibrarianAgent
from src.constants import MODALITY_DIMS
from src.evidence.utility import approximate_utility_from_training, compute_utility_retrospective
from src.fusion.acquisition import (
    ACTION_COSTS,
    CANDIDATE_ACTIONS,
    AcquisitionPolicy,
    encode_case_features,
)
from src.retrieval.rag_index import (
    DEFAULT_EMBEDDING_DIMS,
    TOTAL_DIM,
    RAGIndex,
    concat_modality_embeddings,
    l2_normalize,
)


def make_case(seed=0, label=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "case_id": f"c{seed}", "label": label, "diagnosis": label,
        "clinical_embedding": torch.randn(256, generator=g),
        "dermoscopy_embedding": torch.zeros(256),
        "metadata_embedding": torch.randn(116, generator=g),
        "text_embedding": torch.randn(256, generator=g),
    }


class TestEmbeddingConcatenation:
    def test_dims_match_the_canonical_modality_widths(self):
        assert DEFAULT_EMBEDDING_DIMS["clinical_embedding"] == MODALITY_DIMS["clinical_image"]
        assert DEFAULT_EMBEDDING_DIMS["metadata_embedding"] == MODALITY_DIMS["metadata"]
        assert DEFAULT_EMBEDDING_DIMS["text_embedding"] == MODALITY_DIMS["text"]
        assert TOTAL_DIM == sum(DEFAULT_EMBEDDING_DIMS.values()) == 884

    def test_concatenation_is_the_right_width(self):
        assert concat_modality_embeddings(make_case()).shape == (TOTAL_DIM,)

    def test_absent_modality_becomes_zeros_not_an_error(self):
        case = make_case()
        del case["text_embedding"]
        emb = concat_modality_embeddings(case)
        assert emb.shape == (TOTAL_DIM,)
        assert np.allclose(emb[-256:], 0.0)

    def test_l2_normalize_produces_unit_rows(self):
        x = np.random.default_rng(0).normal(size=(5, 12))
        assert np.allclose(np.linalg.norm(l2_normalize(x), axis=1), 1.0)

    def test_zero_row_does_not_divide_by_zero(self):
        assert np.isfinite(l2_normalize(np.zeros((1, 8)))).all()


class TestRAGIndex:
    def test_query_finds_the_identical_case_first(self):
        cases = [make_case(i, label=i % 3) for i in range(24)]
        idx = RAGIndex().add_training_cases(cases)
        out = idx.query(cases[5], k=3)
        assert out["similar_cases"][0]["case_id"] == "c5"

    def test_k_is_respected(self):
        cases = [make_case(i, label=i % 3) for i in range(24)]
        out = RAGIndex().add_training_cases(cases).query(cases[0], k=7)
        assert len(out["similar_cases"]) == 7

    def test_agreement_rate_is_a_proportion(self):
        cases = [make_case(i, label=i % 3) for i in range(24)]
        out = RAGIndex().add_training_cases(cases).query(cases[0], k=6, predicted_diagnosis=0)
        assert 0.0 <= out["agreement_rate"] <= 1.0

    def test_agreement_is_one_when_every_neighbour_shares_the_label(self):
        cases = [make_case(i, label=2) for i in range(10)]
        out = RAGIndex().add_training_cases(cases).query(cases[0], k=5, predicted_diagnosis=2)
        assert out["agreement_rate"] == pytest.approx(1.0)

    def test_empty_index_is_handled(self):
        assert RAGIndex().add_training_cases([])._index is None

    def test_roundtrip_through_disk(self, tmp_path):
        cases = [make_case(i, label=i % 3) for i in range(12)]
        path = str(tmp_path / "idx")
        RAGIndex().add_training_cases(cases).save(path)

        reloaded = RAGIndex()
        reloaded.load(path)
        out = reloaded.query(cases[3], k=2)
        assert out["similar_cases"][0]["case_id"] == "c3"


class TestLibrarian:
    def test_reports_clearly_when_no_index_is_loaded(self):
        out = LibrarianAgent().retrieve({})
        assert out["similar_cases"] == []
        assert "not loaded" in out["note"]

    def test_serves_queries_once_built(self):
        cases = [make_case(i, label=i % 3) for i in range(16)]
        lib = LibrarianAgent(k=4).build_from_cases(cases)
        out = lib.retrieve(cases[2], predicted_diagnosis=cases[2]["label"])
        assert len(out["similar_cases"]) == 4


class TestAcquisitionPolicy:
    def test_feature_vector_shape(self):
        vec = encode_case_features({"current_entropy": 1.2}, "ask_growth")
        assert vec.shape == (6 + len(CANDIDATE_ACTIONS),)

    def test_action_is_one_hot(self):
        vec = encode_case_features({}, "metadata")
        onehot = vec[6:]
        assert onehot.sum() == 1.0
        assert onehot[CANDIDATE_ACTIONS.index("metadata")] == 1.0

    def test_questions_cost_less_than_imaging(self):
        """Value = Δuncertainty / cost, so the cost ordering drives the policy."""
        assert ACTION_COSTS["ask_growth"] < ACTION_COSTS["text"] < ACTION_COSTS["metadata"]
        assert ACTION_COSTS["metadata"] < ACTION_COSTS["clinical_image"] < ACTION_COSTS["dermoscopy_image"]

    def test_never_requests_something_already_present(self):
        policy = AcquisitionPolicy()
        available = ["clinical_image", "metadata", "text"]
        action = policy.select_action({"current_entropy": 1.2, "missing_count": 1}, available)
        assert action not in available

    def test_returns_a_known_action(self):
        policy = AcquisitionPolicy()
        action = policy.select_action({"current_entropy": 1.5}, ["metadata"])
        assert action in CANDIDATE_ACTIONS

    def test_handles_everything_already_available(self):
        """With no modality left to request, it must still name something actionable."""
        policy = AcquisitionPolicy()
        action = policy.select_action({"current_entropy": 1.0},
                                      ["clinical_image", "dermoscopy_image", "metadata", "text"])
        assert action in CANDIDATE_ACTIONS


class TestUtility:
    def test_deployment_fallback_uses_canonical_modality_keys(self):
        """U must be keyed like the fusion weight vector, or it silently misaligns."""
        from src.constants import MODALITIES

        assert list(approximate_utility_from_training().keys()) == list(MODALITIES)

    def test_lookup_by_diagnosis(self):
        stats = {"clinical_image": {"0": 0.9, "mean": 0.4}, "text": {"mean": 0.2}}
        out = approximate_utility_from_training(stats, diagnosis=0)
        assert out["clinical_image"] == pytest.approx(0.9)
        assert out["text"] == pytest.approx(0.2)

    def test_retrospective_utility_warns_it_needs_ground_truth(self):
        """proj.md rule 4: U must never be computed at inference time."""
        def model_fn(inp):
            return torch.stack([inp["a"].sum(), inp["b"].sum()]).reshape(1, 2)

        case = {"a": torch.ones(4), "b": torch.ones(4) * 2}
        with pytest.warns(UserWarning, match="research ONLY"):
            out = compute_utility_retrospective(
                model_fn, case, torch.tensor([0]),
                lambda logits, y: torch.nn.functional.cross_entropy(logits, y),
                ["a", "b"])
        assert set(out) == {"a", "b"}
        assert all(isinstance(v, float) for v in out.values())
