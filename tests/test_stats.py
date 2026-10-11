"""
tests/test_stats.py - Statistical analysis utilities (proj.md Part 13).

These functions produce the significance claims a write-up would rest on, so
they are checked against hand-computed values rather than only for self-
consistency. A silently wrong correction here does not crash anything - it just
yields a wrong conclusion.
"""
import numpy as np
import pytest

from src.eval.metrics import compute_abstention_metrics, compute_macro_f1, compute_per_class_metrics
from src.eval.stats import (
    bonferroni_correction,
    bootstrap_ci,
    cohens_d,
    fdr_correction,
    mcnemar_test,
)


def accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean())


class TestBootstrapCI:
    def test_ci_brackets_the_point_estimate(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 6, 300)
        y_pred = y_true.copy()
        flip = rng.choice(300, 60, replace=False)
        y_pred[flip] = (y_pred[flip] + 1) % 6

        out = bootstrap_ci(accuracy, y_true, y_pred, n_bootstrap=400)
        assert out["ci_lower"] <= out["metric_on_full"] <= out["ci_upper"]
        assert out["ci_lower"] < out["ci_upper"]
        assert out["metric_on_full"] == pytest.approx(0.8, abs=0.02)

    def test_reproducible_under_seed(self):
        y_true = np.array([0, 1, 2, 3, 4, 5] * 20)
        y_pred = np.array([0, 1, 2, 3, 4, 0] * 20)
        a = bootstrap_ci(accuracy, y_true, y_pred, n_bootstrap=200, seed=42)
        b = bootstrap_ci(accuracy, y_true, y_pred, n_bootstrap=200, seed=42)
        assert a["ci_lower"] == b["ci_lower"] and a["ci_upper"] == b["ci_upper"]

    def test_perfect_predictions_give_a_degenerate_interval(self):
        y = np.array([0, 1, 2, 3, 4, 5] * 10)
        out = bootstrap_ci(accuracy, y, y.copy(), n_bootstrap=100)
        assert out["ci_lower"] == 1.0 and out["ci_upper"] == 1.0

    def test_alpha_widens_the_interval(self):
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 6, 200)
        y_pred = rng.integers(0, 6, 200)
        narrow = bootstrap_ci(accuracy, y_true, y_pred, n_bootstrap=400, alpha=0.20)
        wide = bootstrap_ci(accuracy, y_true, y_pred, n_bootstrap=400, alpha=0.01)
        assert (wide["ci_upper"] - wide["ci_lower"]) >= (narrow["ci_upper"] - narrow["ci_lower"])


class TestMcNemar:
    def test_counts_discordant_pairs(self):
        y_true = np.array([0, 0, 0, 0, 0, 0])
        a = np.array([0, 0, 0, 0, 1, 1])   # 4 correct
        b = np.array([0, 1, 1, 1, 0, 1])   # 2 correct
        out = mcnemar_test(y_true, a, b)
        assert out["correct_a_wrong_b"] == 3    # cases 1,2,3
        assert out["wrong_a_correct_b"] == 1    # case 4

    def test_known_statistic(self):
        """b=10, c=2 -> chi2 = (|10-2|-1)^2 / 12 = 49/12 = 4.0833, p ~= 0.0433."""
        y_true = np.zeros(40, dtype=int)
        a = np.zeros(40, dtype=int)
        b_pred = np.zeros(40, dtype=int)
        b_pred[:10] = 1          # A right, B wrong  -> b = 10
        a[10:12] = 1             # A wrong, B right  -> c = 2
        out = mcnemar_test(y_true, a, b_pred)
        assert out["correct_a_wrong_b"] == 10
        assert out["wrong_a_correct_b"] == 2
        assert out["statistic"] == pytest.approx(49 / 12, abs=1e-6)
        assert out["p_value"] == pytest.approx(0.0433, abs=1e-3)
        assert out["significant_0.05"] is True

    def test_identical_models_are_not_significant(self):
        y_true = np.array([0, 1, 2, 3] * 10)
        pred = np.array([0, 1, 2, 0] * 10)
        out = mcnemar_test(y_true, pred, pred.copy())
        assert out["p_value"] == 1.0
        assert out["note"] == "No disagreements"

    def test_symmetric_disagreement_is_not_significant(self):
        y_true = np.zeros(20, dtype=int)
        a = np.zeros(20, dtype=int); a[:5] = 1     # c = 5
        b = np.zeros(20, dtype=int); b[5:10] = 1   # b = 5
        assert mcnemar_test(y_true, a, b)["p_value"] > 0.05


class TestCohensD:
    def test_zero_when_identical(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert cohens_d(x, x.copy()) == pytest.approx(0.0)

    def test_known_value(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])   # mean 3, sd 1.5811
        b = np.array([3.0, 4.0, 5.0, 6.0, 7.0])   # mean 5, same sd
        assert cohens_d(a, b) == pytest.approx(-2.0 / 1.5811, abs=1e-3)

    def test_sign_follows_argument_order(self):
        a, b = np.array([5.0, 6.0, 7.0]), np.array([1.0, 2.0, 3.0])
        assert cohens_d(a, b) > 0
        assert cohens_d(b, a) < 0

    def test_zero_variance_does_not_divide_by_zero(self):
        assert cohens_d(np.array([2.0, 2.0]), np.array([2.0, 2.0])) == 0.0


class TestMultipleComparisonCorrection:
    def test_bonferroni_scales_by_n(self):
        out = bonferroni_correction([0.01, 0.02, 0.03], alpha=0.05)
        # threshold becomes 0.05/3 = 0.0167, so only the first survives
        assert out["adjusted_threshold"] == pytest.approx(0.05 / 3)
        assert out["significant"] == [True, False, False]

    def test_bonferroni_boundary_is_inclusive(self):
        """A p-value exactly at alpha/n is rejected, not left to float rounding."""
        out = bonferroni_correction([0.01, 0.9, 0.9, 0.9, 0.9], alpha=0.05)
        assert out["significant"][0] is True

    def test_fdr_is_less_conservative_than_bonferroni(self):
        p = [0.001, 0.008, 0.02, 0.04, 0.3]
        bonf = bonferroni_correction(p, alpha=0.05)["significant"]
        fdr = fdr_correction(p, alpha=0.05)["significant"]
        assert sum(fdr) >= sum(bonf)

    def test_fdr_known_example(self):
        """Benjamini-Hochberg on p=[0.01,0.02,0.03,0.04,0.05], n=5.

        Adjusted = p * n / rank = 0.05 for every entry. The textbook rule finds
        the largest k with p_(k) <= (k/n)*alpha; here p_(5)=0.05 <= 0.05, so k=5
        and ALL five hypotheses are rejected.

        REGRESSION: the implementation used `<`, so each entry was accepted or
        rejected according to whether `p * n / rank` happened to land a few ULPs
        below 0.05 - giving [True, True, True, False, False] for a case where
        the correct answer is uniform.
        """
        out = fdr_correction([0.01, 0.02, 0.03, 0.04, 0.05], alpha=0.05)
        adj = out["adjusted_p_values"]
        assert all(a == pytest.approx(0.05, abs=1e-9) for a in adj)
        assert out["significant"] == [True] * 5

    def test_fdr_rejects_nothing_when_all_p_are_large(self):
        out = fdr_correction([0.4, 0.6, 0.8, 0.99], alpha=0.05)
        assert out["significant"] == [False] * 4

    def test_fdr_preserves_input_order(self):
        out = fdr_correction([0.5, 0.001, 0.2], alpha=0.05)
        adj = out["adjusted_p_values"]
        assert adj[1] < adj[2] < adj[0]   # the smallest raw p stays smallest

    def test_adjusted_p_values_are_probabilities(self):
        out = fdr_correction([0.4, 0.5, 0.9, 0.95], alpha=0.05)
        assert all(0.0 <= a <= 1.0 for a in out["adjusted_p_values"])


class TestMetrics:
    def test_macro_f1_perfect(self):
        y = [0, 1, 2, 3, 4, 5]
        assert compute_macro_f1(y, y) == pytest.approx(1.0)

    def test_macro_f1_penalises_rare_class_failure(self):
        """Macro-F1 must not hide a failed minority class the way accuracy does."""
        y_true = [0] * 95 + [5] * 5
        y_pred = [0] * 100          # never predicts the rare class
        assert compute_macro_f1(y_true, y_pred) < 0.6
        assert np.mean(np.array(y_true) == np.array(y_pred)) == 0.95

    def test_per_class_metrics_reports_each_class(self):
        from src.constants import CLASS_NAMES

        y_true = [0, 1, 2, 3, 4, 5, 0, 1]
        y_pred = [0, 1, 2, 3, 4, 0, 0, 1]
        out = compute_per_class_metrics(y_true, y_pred, class_names=list(CLASS_NAMES))
        assert isinstance(out, dict) and out
        # proj.md rule 9: ALWAYS report per-class metrics, not just macro.
        blob = str(out)
        assert any(name in blob for name in CLASS_NAMES)

    def test_abstention_metrics_separate_coverage_from_accuracy(self):
        results = [
            {"action": "ACCEPT", "diagnosis": 0},
            {"action": "ARBITRATE", "diagnosis": 1},
            {"action": "ESCALATE", "diagnosis": None},
            {"action": "ABSTAIN", "diagnosis": None},
        ]
        out = compute_abstention_metrics(results, y_true=[0, 1, 2, 3])
        assert out["coverage"] == pytest.approx(0.5)
        assert out["abstention_rate"] == pytest.approx(0.5)
        # Both answered cases were right, so accuracy-on-covered is 1.0 even
        # though overall accuracy is 0.5 - the two must not be conflated.
        assert out["accuracy_on_covered"] == pytest.approx(1.0)
