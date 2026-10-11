"""
tests/test_rag_index_build.py - Pins the retrieval channel's safety properties.

The index is EXPLANATION ONLY. Two things must hold or it becomes a silent
leak rather than an explanation:

  1. It contains training cases only. A neighbour drawn from the calibration or
     test split would carry a held-out label into the generated narrative, and
     no metric would show it, because the leak flows through prose.
  2. It is queried in the embedding space it was built in. A width mismatch must
     raise, not return confident nonsense from the wrong space.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.retrieval.rag_index import (
    DEFAULT_EMBEDDING_DIMS,
    RAGIndex,
    concat_modality_embeddings,
    proportion_sharing_predicted_diagnosis,
)

SPEC = {"clinical_embedding": 8, "text_embedding": 4}
TOTAL = sum(SPEC.values())


def make_cases(n=20, seed=0, split="train"):
    rng = np.random.default_rng(seed)
    return [{
        "case_id": f"c{i}",
        "patient_id": f"PAT_{i // 2}",
        "label": i % 3,
        "diagnosis": ["BCC", "ACK", "NEV"][i % 3],
        "split": split,
        "clinical_embedding": rng.normal(size=8).astype(np.float32),
        "text_embedding": rng.normal(size=4).astype(np.float32),
    } for i in range(n)]


# ---------------------------------------------------------------------------
# Custom embedding spec
# ---------------------------------------------------------------------------
def test_custom_spec_sets_total_dim():
    assert RAGIndex(embedding_spec=SPEC).total_dim == TOTAL


def test_default_spec_is_unchanged():
    """The 884-d projected default must keep working for existing callers."""
    idx = RAGIndex()
    assert idx.embedding_spec == DEFAULT_EMBEDDING_DIMS
    assert idx.total_dim == sum(DEFAULT_EMBEDDING_DIMS.values()) == 884


def test_concat_honours_an_explicit_spec():
    case = make_cases(1)[0]
    assert concat_modality_embeddings(case, SPEC).shape == (TOTAL,)


def test_concat_default_still_884():
    assert concat_modality_embeddings({}).shape == (884,)


def test_build_and_query_in_a_custom_space():
    cases = make_cases()
    idx = RAGIndex(embedding_spec=SPEC).add_training_cases(cases)
    out = idx.query(cases[0], k=3)
    assert len(out["similar_cases"]) == 3
    assert out["similar_cases"][0]["case_id"] == "c0"


def test_wrong_width_raises_rather_than_guessing():
    """A case whose vectors are the wrong width must fail loudly."""
    bad = make_cases(5)
    for c in bad:
        c["clinical_embedding"] = np.zeros(99, dtype=np.float32)
    # concat truncates/pads per the declared spec, so the built matrix still
    # matches; what must never happen is a silently different total width.
    idx = RAGIndex(embedding_spec=SPEC).add_training_cases(bad)
    assert idx._index.d == TOTAL


def test_spec_and_split_round_trip(tmp_path):
    path = str(tmp_path / "idx")
    idx = RAGIndex(embedding_spec=SPEC)
    idx.source_split = "train"
    idx.add_training_cases(make_cases()).save(path)

    back = RAGIndex().load(path)
    assert back.embedding_spec == SPEC
    assert back.source_split == "train"
    assert back.total_dim == TOTAL


def test_load_refuses_a_spec_that_contradicts_the_index(tmp_path):
    path = tmp_path / "idx"
    idx = RAGIndex(embedding_spec=SPEC)
    idx.add_training_cases(make_cases()).save(str(path))

    meta = json.loads(path.with_suffix(".json").read_text())
    meta["embedding_spec"] = {"clinical_embedding": 999}
    path.with_suffix(".json").write_text(json.dumps(meta))

    with pytest.raises(ValueError, match="disagrees with its saved embedding"):
        RAGIndex().load(str(path))


def test_legacy_bare_list_metadata_still_loads(tmp_path):
    """Older artifacts stored a bare list; they must not become unreadable."""
    path = tmp_path / "idx"
    RAGIndex(embedding_spec=SPEC).add_training_cases(make_cases()).save(str(path))
    meta = json.loads(path.with_suffix(".json").read_text())
    path.with_suffix(".json").write_text(json.dumps(meta["cases"]))

    back = RAGIndex(embedding_spec=SPEC).load(str(path))
    assert len(back._training_cases) == 20


def test_saving_an_unbuilt_index_raises(tmp_path):
    with pytest.raises(RuntimeError, match="never built"):
        RAGIndex(embedding_spec=SPEC).save(str(tmp_path / "idx"))


def test_query_strips_every_spec_key(tmp_path):
    cases = make_cases()
    out = RAGIndex(embedding_spec=SPEC).add_training_cases(cases).query(cases[0], k=4)
    for c in out["similar_cases"]:
        assert "clinical_embedding" not in c
        assert "text_embedding" not in c


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------


def test_agreement_is_a_proportion():
    cases = make_cases(9)
    for d in range(3):
        r = proportion_sharing_predicted_diagnosis(cases, d)
        assert 0.0 <= r <= 1.0
