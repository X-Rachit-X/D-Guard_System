"""
src/retrieval/rag_index.py — RAG Index for DERMA-GUARD v4.0.

CRITICAL FIX: Index individual modality embedding CONCATENATIONS, not fused embeddings.
  - Training: concatenate [z_clinical, z_dermo, z_meta, z_text] per case
  - Inference: same concatenation for the test case
  - NEVER index fused embeddings (circular — fusion depends on RAG which depends on fusion)

Index type: FAISS IndexFlatIP (inner product = cosine after L2 normalisation).
Default k = 8 similar cases.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# Individual embedding dims, derived from the canonical modality widths so the
# index can never silently disagree with what the encoders actually emit.
from src.constants import MODALITY_DIMS

DEFAULT_EMBEDDING_DIMS = {
    "clinical_embedding":    MODALITY_DIMS["clinical_image"],    # 256, trained image projector
    "dermoscopy_embedding":  MODALITY_DIMS["dermoscopy_image"],  # 256, same projector
    "metadata_embedding":    MODALITY_DIMS["metadata"],          # 116, engineered features
    "text_embedding":        MODALITY_DIMS["text"],              # 256, trained text projector
}

TOTAL_DIM = sum(DEFAULT_EMBEDDING_DIMS.values())  # 256 + 256 + 116 + 256 = 884


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2 normalise rows of a 2D array (for cosine similarity via inner product)."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-8)


def concat_modality_embeddings(case: Dict,
                               dims: Optional[Dict[str, int]] = None) -> np.ndarray:
    """Concatenate individual modality embeddings for indexing/querying.

    Args:
        case: dict with keys matching `dims`
        dims: {embedding_key: width}. Defaults to DEFAULT_EMBEDDING_DIMS, the
              projected 256/256/116/256 space. Pass an explicit spec to index a
              different embedding space - e.g. the raw backbone features emitted
              by scripts/build_features.py (2048/768/114), which are the only
              ones that currently reproduce bit-for-bit.

    Returns:
        1D numpy array of concatenated embeddings (sum(dims.values()),)
    """
    dims = DEFAULT_EMBEDDING_DIMS if dims is None else dims
    parts = []
    for key, dim in dims.items():
        emb = case.get(key)
        if emb is None:
            parts.append(np.zeros(dim, dtype=np.float32))
        elif isinstance(emb, torch.Tensor):
            parts.append(emb.detach().cpu().numpy().flatten()[:dim])
        else:
            arr = np.array(emb, dtype=np.float32).flatten()
            if len(arr) < dim:
                arr = np.pad(arr, (0, dim - len(arr)))
            parts.append(arr[:dim])
    return np.concatenate(parts, axis=0).astype(np.float32)


def proportion_sharing_predicted_diagnosis(
    similar_cases: List[Dict],
    predicted_diagnosis: Optional[int] = None,
) -> float:
    """Compute fraction of similar cases sharing the predicted diagnosis."""
    if not similar_cases:
        return 0.0
    if predicted_diagnosis is None:
        return 0.0
    matching = sum(
        1 for c in similar_cases
        if c.get("label") == predicted_diagnosis or c.get("diagnosis") == predicted_diagnosis
    )
    return matching / len(similar_cases)


class RAGIndex:
    """FAISS-based RAG index over individual modality embeddings.

    Usage:
        # Build at training time
        index = RAGIndex()
        index.add_training_cases(training_cases)
        index.save("outputs/rag_index")

        # Query at inference
        index.load("outputs/rag_index")
        result = index.query(test_case, k=8)
    """

    def __init__(self, total_dim: int = TOTAL_DIM,
                 embedding_spec: Optional[Dict[str, int]] = None):
        """
        Args:
            total_dim:      width of the concatenated vector
            embedding_spec: {embedding_key: width} this index is built over. It
                            is persisted alongside the index and re-checked on
                            every query, so an index built in one embedding
                            space can never be queried in another - which would
                            return confident nonsense rather than an error.
        """
        self.embedding_spec = dict(embedding_spec) if embedding_spec else dict(
            DEFAULT_EMBEDDING_DIMS)
        self.total_dim = sum(self.embedding_spec.values()) if embedding_spec else total_dim
        self._index = None
        self._training_cases: List[Dict] = []
        #: Set by build_rag_index.py. Retrieval is EXPLANATION ONLY and must
        #: never contain a case from the split being evaluated, or the
        #: explanation leaks held-out labels.
        self.source_split: Optional[str] = None

    def add_training_cases(self, cases: List[Dict]) -> "RAGIndex":
        """Build FAISS index from training cases.

        CRITICAL: uses individual modality concatenation, NOT fused embeddings.

        Args:
            cases: list of dicts, each with embedding keys and 'label'/'diagnosis'.
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")

        embeddings = []
        for case in cases:
            emb = concat_modality_embeddings(case, self.embedding_spec)
            embeddings.append(emb)

        if not embeddings:
            print("[RAGIndex] No training cases provided. Index not built.")
            return self

        emb_matrix = np.stack(embeddings, axis=0).astype(np.float32)  # (N, D)
        if emb_matrix.shape[1] != self.total_dim:
            raise ValueError(
                f"embedding width {emb_matrix.shape[1]} does not match the "
                f"declared total_dim {self.total_dim}; spec={self.embedding_spec}"
            )
        # L2 normalise for cosine similarity via inner product
        emb_matrix = l2_normalize(emb_matrix)

        self._index = faiss.IndexFlatIP(self.total_dim)
        self._index.add(emb_matrix)
        self._training_cases = list(cases)
        print(f"[RAGIndex] Indexed {len(cases)} training cases (dim={self.total_dim})")
        return self

    def query(
        self,
        case: Dict,
        k: int = 8,
        predicted_diagnosis: Optional[int] = None,
    ) -> Dict:
        """Find k most similar training cases.

        CRITICAL: query uses individual modality concatenation (same as index).

        Args:
            case:                 dict with embedding keys
            k:                    number of neighbours to retrieve
            predicted_diagnosis:  if provided, computes agreement_rate

        Returns:
            dict with:
                similar_cases:  list of k case dicts with 'distance' added
                agreement_rate: fraction sharing predicted_diagnosis
        """
        if self._index is None or not self._training_cases:
            return {"similar_cases": [], "agreement_rate": 0.0}

        query_emb = concat_modality_embeddings(case, self.embedding_spec).reshape(1, -1)
        query_emb = l2_normalize(query_emb)

        actual_k = min(k, self._index.ntotal)
        if actual_k == 0:
            return {"similar_cases": [], "agreement_rate": 0.0}

        distances, indices = self._index.search(query_emb, actual_k)

        similar_cases = []
        for i in range(actual_k):
            idx = int(indices[0][i])
            if 0 <= idx < len(self._training_cases):
                case_copy = dict(self._training_cases[idx])
                # Remove raw embedding tensors (too large for JSON / LLM context)
                for key in (*DEFAULT_EMBEDDING_DIMS, *self.embedding_spec):
                    case_copy.pop(key, None)
                case_copy["distance"] = float(distances[0][i])
                similar_cases.append(case_copy)

        agreement_rate = proportion_sharing_predicted_diagnosis(similar_cases, predicted_diagnosis)

        return {
            "similar_cases": similar_cases,
            "agreement_rate": agreement_rate,
        }

    def save(self, path: str) -> None:
        """Save index and metadata to disk."""
        import faiss
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._index is None:
            raise RuntimeError("nothing to save: the index was never built")
        faiss.write_index(self._index, str(path.with_suffix(".index")))
        meta_path = path.with_suffix(".json")
        # Save only non-tensor metadata
        safe_cases = []
        for c in self._training_cases:
            safe_c = {k: v for k, v in c.items() if not hasattr(v, "numpy")}
            safe_cases.append(safe_c)
        payload = {
            "embedding_spec": self.embedding_spec,
            "total_dim": self.total_dim,
            "source_split": self.source_split,
            "cases": safe_cases,
        }
        with open(meta_path, "w") as f:
            json.dump(payload, f, default=str)
        print(f"[RAGIndex] Saved index -> {path}.index + .json "
              f"({len(safe_cases)} cases, dim={self.total_dim}, "
              f"split={self.source_split})")

    def load(self, path: str) -> "RAGIndex":
        """Load index from disk."""
        import faiss
        path = Path(path)
        index_path = path.with_suffix(".index")
        meta_path = path.with_suffix(".json")
        if not index_path.exists():
            raise FileNotFoundError(f"RAG index not found at {index_path}")
        self._index = faiss.read_index(str(index_path))
        self.total_dim = self._index.d
        with open(meta_path) as f:
            payload = json.load(f)
        # Accept both the current payload and the older bare-list format.
        if isinstance(payload, dict) and "cases" in payload:
            self._training_cases = payload["cases"]
            self.embedding_spec = payload.get("embedding_spec") or self.embedding_spec
            self.source_split = payload.get("source_split")
        else:
            self._training_cases = payload
        spec_total = sum(self.embedding_spec.values())
        if spec_total != self.total_dim:
            raise ValueError(
                f"index width {self.total_dim} disagrees with its saved embedding "
                f"spec ({spec_total}); refusing to serve queries in the wrong space"
            )
        print(f"[RAGIndex] Loaded {len(self._training_cases)} cases "
              f"(dim={self.total_dim}, split={self.source_split})")
        return self
