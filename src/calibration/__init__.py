"""src/calibration - conformal prediction sets and risk-controlled thresholds."""
from src.calibration.conformal import (
    ConformalCalibrator,
    aps_scores,
    raps_scores,
    set_is_ambiguous,
)
from src.calibration.ltt import (
    LTTResult,
    calibrate_per_group,
    calibrate_threshold,
    guarantee_violation,
    hoeffding_bentkus_p_value,
    simulate_coverage,
)

__all__ = ["ConformalCalibrator", "aps_scores", "raps_scores", "set_is_ambiguous",
           "LTTResult", "calibrate_threshold", "calibrate_per_group",
           "guarantee_violation", "hoeffding_bentkus_p_value", "simulate_coverage"]
