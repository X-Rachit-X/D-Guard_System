"""src/evidence - evidential opinions and Dempster-Shafer fusion (reference implementation)."""
from src.evidence.evidential import (
    EvidentialHead, Opinion, conflict_and_ignorance, ds_combine, edl_loss,
    fold_opinions, gate_opinion, opinion_from_evidence, opinion_from_probs, vacuous,
)

__all__ = ["EvidentialHead", "Opinion", "ds_combine", "fold_opinions", "vacuous",
           "gate_opinion", "opinion_from_evidence", "opinion_from_probs",
           "edl_loss", "conflict_and_ignorance"]
