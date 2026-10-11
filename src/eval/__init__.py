"""src/eval - selective-prediction metrics (AUGRC, acquisition efficiency, calibration)."""
from src.eval.selective import (
    acquisition_efficiency, adaptive_ece, augrc, aurc, brier_decomposition,
    net_benefit, risk_coverage_curve, selective_report, selective_risk,
)

__all__ = ["acquisition_efficiency", "adaptive_ece", "augrc", "aurc",
           "brier_decomposition", "net_benefit", "risk_coverage_curve",
           "selective_report", "selective_risk"]
