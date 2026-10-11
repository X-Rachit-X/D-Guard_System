"""src/eval/__init__.py — Evaluation package."""
from src.eval.metrics import compute_per_class_metrics, compute_macro_f1
from src.eval.stats import bootstrap_ci, mcnemar_test, cohens_d, bonferroni_correction

__all__ = [
    "compute_per_class_metrics", "compute_macro_f1",
    "bootstrap_ci", "mcnemar_test", "cohens_d", "bonferroni_correction",
]

from src.eval.selective import (
    acquisition_efficiency, adaptive_ece, augrc, aurc, brier_decomposition,
    net_benefit, risk_coverage_curve, selective_report, selective_risk,
)
