"""
tests/test_selective.py - Selective-prediction metrics (AUGRC, AURC, AE, ECE).

AUGRC definitions follow Traub et al., NeurIPS 2024 (arXiv:2407.01032):
generalized risk divides by N, selective risk divides by the number accepted,
and AUGRC integrates the former over coverage. Bounded [0, 1/2].
"""
import numpy as np
import pytest

from src.eval.selective import (
    acquisition_efficiency,
    acquisition_efficiency_curve,
    adaptive_ece,
    augrc,
    aurc,
    brier_decomposition,
    classwise_ece,
    negative_information_rate,
    net_benefit,
    risk_coverage_curve,
    selective_report,
    selective_risk,
)

K = 6


def perfect(n=100):
    y = np.arange(n) % K
    probs = np.full((n, K), 0.001)
    probs[np.arange(n), y] = 1 - 0.001 * (K - 1)
    return y, probs


def ranked(n=200, err=0.3, seed=0):
    """Errors concentrated in the low-confidence tail - an informative score."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, K, n)
    score = rng.random(n)
    wrong = rng.random(n) < err * (1 - score)
    pred = np.where(wrong, (y + 1) % K, y)
    return y, pred, score


# ---------------------------------------------------------------------------
class TestRiskCoverage:
    def test_perfect_predictions_give_zero_risk(self):
        y, probs = perfect()
        rc = risk_coverage_curve(y, probs.argmax(1), probs.max(1))
        assert rc.augrc == pytest.approx(0.0, abs=1e-9)
        assert rc.aurc == pytest.approx(0.0, abs=1e-9)

    def test_all_wrong_hits_the_upper_bound(self):
        """AUGRC is bounded by 1/2; all-wrong should approach it."""
        n = 200
        y = np.zeros(n, int)
        pred = np.ones(n, int)
        rc = risk_coverage_curve(y, pred, np.random.default_rng(0).random(n))
        assert rc.aurc == pytest.approx(1.0, abs=1e-9)
        assert 0.49 <= rc.augrc <= 0.5 + 1e-9

    def test_augrc_is_in_the_documented_range(self):
        for seed in range(5):
            y, pred, score = ranked(seed=seed)
            assert 0.0 <= augrc(y, pred, score) <= 0.5 + 1e-9

    def test_generalized_risk_is_normalised_by_n_not_coverage(self):
        """The definitional difference from selective risk."""
        y, pred, score = ranked(seed=1)
        rc = risk_coverage_curve(y, pred, score)
        # GR = SR * coverage, by construction
        assert np.allclose(rc.generalized_risk, rc.selective_risk * rc.coverage, atol=1e-12)

    def test_augrc_respects_its_upper_bound_exactly(self):
        """REGRESSION: a right-endpoint Riemann sum overshot 1/2 by 1/(2n)."""
        for n in (50, 200, 1000):
            y = np.zeros(n, int); pred = np.ones(n, int)
            rc = risk_coverage_curve(y, pred, np.random.default_rng(0).random(n))
            assert rc.augrc <= 0.5 + 1e-9, (n, rc.augrc)
            assert rc.augrc == pytest.approx(0.5, abs=1e-9)

    def test_full_coverage_risk_equals_error_rate(self):
        y, pred, score = ranked(seed=2)
        rc = risk_coverage_curve(y, pred, score)
        assert rc.selective_risk[-1] == pytest.approx((pred != y).mean())
        assert rc.coverage[-1] == pytest.approx(1.0)

    def test_informative_score_beats_random_score(self):
        y, pred, score = ranked(n=600, err=0.5, seed=3)
        rand = np.random.default_rng(9).random(y.size)
        assert augrc(y, pred, score) < augrc(y, pred, rand)

    def test_coverage_at_risk_is_monotone_in_alpha(self):
        y, pred, score = ranked(n=500, seed=4)
        rc = risk_coverage_curve(y, pred, score)
        covs = [rc.coverage_at_risk(a) for a in (0.02, 0.05, 0.1, 0.3)]
        assert covs == sorted(covs)

    def test_curve_is_deterministic_under_ties(self):
        y = np.array([0, 1, 0, 1]); pred = np.array([0, 0, 0, 1])
        score = np.ones(4)                    # all tied
        a = risk_coverage_curve(y, pred, score).augrc
        b = risk_coverage_curve(y, pred, score).augrc
        assert a == b

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="empty"):
            risk_coverage_curve(np.array([]), np.array([]), np.array([]))

    def test_custom_loss_is_honoured(self):
        y, pred, score = ranked(seed=5)
        heavy = np.where(pred != y, 1.0, 0.0) * 0.5
        assert augrc(y, pred, score, loss=heavy) < augrc(y, pred, score)


class TestSelectiveRisk:
    def test_pair_is_reported_together(self):
        y, pred, score = ranked(seed=6)
        risk, cov = selective_risk(y, pred, score > 0.5)
        assert 0.0 <= risk <= 1.0 and 0.0 < cov < 1.0

    def test_refusing_everything_gives_nan_not_zero(self):
        """A system that answers nothing must not score a perfect risk."""
        y, pred, _ = ranked(seed=7)
        risk, cov = selective_risk(y, pred, np.zeros(y.size, bool))
        assert np.isnan(risk) and cov == 0.0

    def test_accepting_everything_equals_error_rate(self):
        y, pred, _ = ranked(seed=8)
        risk, cov = selective_risk(y, pred, np.ones(y.size, bool))
        assert cov == 1.0 and risk == pytest.approx((pred != y).mean())


class TestCalibration:
    def test_perfect_confident_is_well_calibrated(self):
        y, probs = perfect(300)
        assert adaptive_ece(probs, y) < 0.05

    def test_confidently_wrong_is_badly_calibrated(self):
        n = 200
        y = np.zeros(n, int)
        probs = np.full((n, K), 0.001); probs[:, 1] = 1 - 0.001 * (K - 1)
        assert adaptive_ece(probs, y) > 0.9

    def test_classwise_ece_is_finite(self):
        y, probs = perfect(120)
        assert 0.0 <= classwise_ece(probs, y) < 1.0

    def test_brier_decomposition_components(self):
        y, probs = perfect(200)
        d = brier_decomposition(probs, y)
        assert set(d) == {"brier", "reliability", "resolution", "uncertainty"}
        assert d["brier"] < 0.01 and d["reliability"] < 0.05

    def test_brier_penalises_confident_errors(self):
        n = 150
        y = np.zeros(n, int)
        good = np.full((n, K), 0.001); good[:, 0] = 1 - 0.001 * (K - 1)
        bad = np.full((n, K), 0.001); bad[:, 3] = 1 - 0.001 * (K - 1)
        assert brier_decomposition(bad, y)["brier"] > brier_decomposition(good, y)["brier"]


class TestAcquisitionEfficiency:
    def test_oracle_ranking_scores_one(self):
        rng = np.random.default_rng(0)
        benefit = rng.normal(1.0, 1.0, 400)
        ae = acquisition_efficiency(benefit, policy_score=benefit, rate=0.3)
        assert ae.efficiency == pytest.approx(1.0, abs=1e-6)

    def test_random_ranking_scores_near_zero(self):
        rng = np.random.default_rng(1)
        benefit = rng.normal(1.0, 1.0, 1200)
        ae = acquisition_efficiency(benefit, rng.normal(0, 1, 1200), rate=0.3, seed=1)
        assert abs(ae.efficiency) < 0.25

    def test_inverted_ranking_scores_negative(self):
        """Actively-worse-than-random is a real outcome and must be reportable."""
        rng = np.random.default_rng(2)
        benefit = rng.normal(1.0, 1.0, 400)
        ae = acquisition_efficiency(benefit, -benefit, rate=0.3)
        assert ae.efficiency < -0.5

    def test_oracle_is_an_upper_bound(self):
        rng = np.random.default_rng(3)
        benefit = rng.normal(0.5, 1.0, 500)
        ae = acquisition_efficiency(benefit, rng.random(500), rate=0.4, seed=3)
        assert ae.gain_oracle >= ae.gain_policy - 1e-9
        assert ae.gain_oracle >= ae.gain_random - 1e-9

    def test_zero_rate_handled(self):
        ae = acquisition_efficiency(np.ones(10), np.ones(10), rate=0.0)
        assert ae.n_acquired == 0 and np.isnan(ae.efficiency)

    def test_curve_covers_requested_rates(self):
        rng = np.random.default_rng(4)
        b = rng.normal(1, 1, 300)
        c = acquisition_efficiency_curve(b, b, rates=(0.1, 0.5))
        assert set(c) == {"0.10", "0.50"}

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            acquisition_efficiency(np.ones(5), np.ones(3), rate=0.5)


class TestNegativeInformationRate:
    def test_detects_harmful_acquisitions(self):
        before = np.array([1.0, 1.0, 1.0, 1.0])
        after = np.array([0.0, 0.0, 2.0, 3.0])      # 2 of 4 got worse
        out = negative_information_rate(before, after)
        assert out["nir"] == pytest.approx(0.5)
        assert out["mean_increase_when_worse"] == pytest.approx(1.5)

    def test_zero_when_always_helpful(self):
        out = negative_information_rate(np.ones(10), np.zeros(10))
        assert out["nir"] == 0.0 and out["mean_change"] < 0


class TestNetBenefit:
    def test_shapes_and_reference_strategies(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 300)
        p = np.clip(y * 0.6 + rng.random(300) * 0.4, 0, 1)
        nb = net_benefit(y, p, thresholds=(0.1, 0.2, 0.3))
        assert len(nb["net_benefit_model"]) == 3
        assert nb["net_benefit_treat_none"] == [0.0, 0.0, 0.0]

    def test_informative_model_beats_treat_none(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 500)
        p = np.clip(y * 0.8 + rng.random(500) * 0.2, 0, 1)
        nb = net_benefit(y, p, thresholds=(0.2,))
        assert nb["net_benefit_model"][0] > 0.0


class TestSelectiveReport:
    def test_bundle_has_the_headline_fields(self):
        y, probs = perfect(200)
        rep = selective_report(y, probs, probs.max(1))
        for key in ("accuracy", "augrc", "aurc", "adaptive_ece", "brier",
                    "reliability", "risk_coverage"):
            assert key in rep

    def test_policy_block_included_when_mask_given(self):
        y, probs = perfect(100)
        acc = probs.max(1) > 0.5
        rep = selective_report(y, probs, probs.max(1), accepted=acc)
        assert rep["policy"]["coverage"] == pytest.approx(1.0)
        assert rep["policy"]["n_accepted"] == 100
