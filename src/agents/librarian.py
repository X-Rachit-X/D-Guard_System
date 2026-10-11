"""
src/agents/librarian.py - Similar-case retrieval over MILK10k training lesions.

Shows the clinician confirmed past cases that look like this one - their photos
and their histopathology-backed diagnoses. Retrieval is EXPLANATION ONLY:

  * the index holds TRAINING lesions only (asserted when built), so a neighbour
    can never carry a held-out test label into what a reader sees;
  * it never changes a probability, a threshold or an action. The calibrated
    guarantees assume exchangeable data, and a decision that depended on a
    retrieved corpus would break that assumption.

Embeddings are the trained model's own 256-d projections of the clinical photo
and of the metadata, so "similar" means similar to the model, which is what a
reader needs in order to understand its opinion.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.retrieval.rag_index import RAGIndex

SPEC = {"clinical_embedding": 256, "metadata_embedding": 256}


class MilkLibrarian:
    def __init__(self, index: RAGIndex, k: int = 6):
        if index.source_split != "train":
            raise ValueError(f"retrieval index must be built from the training split, "
                             f"got source_split={index.source_split!r}")
        self.index = index
        self.k = k

    @classmethod
    def load(cls, path: str | Path, k: int = 6) -> "MilkLibrarian":
        return cls(RAGIndex().load(str(path)), k=k)

    def similar(self, clinical_emb: Optional[np.ndarray], meta_emb: Optional[np.ndarray],
                tentative: Optional[str] = None) -> Dict:
        out = self.index.query({"clinical_embedding": clinical_emb,
                                "metadata_embedding": meta_emb}, k=self.k)
        cases: List[Dict] = []
        for c in out["similar_cases"]:
            cases.append({
                "lesion_id": c.get("lesion_id"),
                "diagnosis": c.get("diagnosis"),
                "similarity": round(float(c.get("distance", 0.0)), 4),
                "clinical_path": c.get("clinical_path"),
                "dermoscopy_path": c.get("dermoscopy_path"),
                "age": c.get("age"), "site": c.get("site"),
            })
        agree = (sum(c["diagnosis"] == tentative for c in cases) / len(cases)
                 if cases and tentative else None)
        return {"similar_cases": cases, "agreement": agree}
