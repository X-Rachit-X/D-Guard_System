"""
tests/test_calibration.py - Conformal sets and Learn-then-Test risk control.

The load-bearing test is `TestGuaranteeValidity::test_empirical_coverage_holds`:
on synthetic exchangeable data the realised test risk must exceed alpha in at
most delta of repeated calibration draws. That is the test that catches an
invalid p-value or an off-by-one in the fixed-sequence walk - the kind of bug
that would silently invalidate every safety claim the project makes.
"""
import numpy as np
import pytest

from src.calibration.conformal import (
    ConformalCalibrator,
    aps_scores,
    raps_scores,
    set_is_ambiguous,
)
from src.calibration.ltt import (
    bentkus_p_value,
    calibrate_per_group,
    calibrate_threshold,
    guarantee_violation,
    hoeffding_bentkus_p_value,
    hoeffding_p_value,
    simulate_coverage,
)

K = 6
MALIGNANT = (0, 4, 5)   # BCC, SCC, MEL


def synth(n=600, k=K, seed=0, noise=0.6):
    """Probabilities loosely correlated with labels, so confidence is informative."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    logits = rng.normal(0, 1, (n, k))
    logits[np.arange(n), y] += rng.normal(2.0, noise, n)
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True), y


# ---------------------------------------------------------------------------
class TestPValues:
    def test_hoeffding_is_one_when_risk_exceeds_alpha(self):
        assert hoeffding_p_value(0.3, 0.1, 100) == 1.0

    def test_hoeffding_shrinks_with_n(self):
        ps = [hoeffding_p_value(0.02, 0.1, n) for n in (10, 50, 200, 1000)]
        assert ps == sorted(ps, reverse=True)

    def test_hoeffding_shrinks_as_risk_falls(self):
        ps = [hoeffding_p_value(r, 0.1, 200) for r in (0.09, 0.05, 0.02, 0.0)]
        assert ps == sorted(ps, reverse=True)

    def test_all_p_values_are_probabilities(self):
        for n in (1, 10, 500):
            for r in (0.0, 0.05, 0.5, 1.0):
                for f in (hoeffding_p_value, bentkus_p_value, hoeffding_bentkus_p_value):
                    assert 0.0 <= f(r, 0.1, n) <= 1.0

    def test_hb_is_at_most_each_component(self):
        for r in (0.0, 0.02, 0.08):
            hb = hoeffding_bentkus_p_value(r, 0.1, 150)
            assert hb <= hoeffding_p_value(r, 0.1, 150) + 1e-12
            assert hb <= bentkus_p_value(r, 0.1, 150) + 1e-12


# ---------------------------------------------------------------------------
class TestLTT:
    def test_returns_a_threshold_on_easy_data(self):
        rng = np.random.default_rng(1)
        score = rng.random(800)
        loss = (rng.random(800) < 0.25 * (1 - score)).astype(float)
        res = calibrate_threshold(score, loss, alpha=0.15, delta=0.1)
        assert res.controlled and 0.0 <= res.lambda_hat <= 1.0
        assert res.risk_at_lambda <= 0.15 + 1e-9

    def test_reports_failure_rather_than_guessing(self):
        """When nothing can be certified it must say so, not return a best guess."""
        rng = np.random.default_rng(2)
        score = rng.random(40)
        loss = np.ones(40)                 # everything wrong
        res = calibrate_threshold(score, loss, alpha=0.01, delta=0.05)
        assert not res.controlled
        assert res.lambda_hat is None
        assert "no threshold" in res.note

    def test_stricter_alpha_is_more_conservative(self):
        rng = np.random.default_rng(3)
        score = rng.random(1500)
        loss = (rng.random(1500) < 0.3 * (1 - score)).astype(float)
        loose = calibrate_threshold(score, loss, alpha=0.20, delta=0.1)
        tight = calibrate_threshold(score, loss, alpha=0.05, delta=0.1)
        assert loose.controlled and tight.controlled
        assert tight.coverage_at_lambda <= loose.coverage_at_lambda + 1e-9

    def test_min_coverage_rejects_degenerate_thresholds(self):
        """A threshold accepting 3 cases trivially passes the risk test."""
        rng = np.random.default_rng(4)
        score = rng.random(500)
        loss = (rng.random(500) < 0.3 * (1 - score)).astype(float)
        res = calibrate_threshold(score, loss, alpha=0.05, delta=0.1, min_coverage=0.5)
        if res.controlled:
            assert res.coverage_at_lambda >= 0.5

    def test_rejects_unbounded_loss(self):
        with pytest.raises(ValueError, match="bounded in"):
            calibrate_threshold(np.random.random(50), np.full(50, 3.0))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            calibrate_threshold(np.random.random(10), np.zeros(5))

    def test_joint_mode_runs(self):
        rng = np.random.default_rng(5)
        score = rng.random(600)
        loss = (rng.random(600) < 0.2 * (1 - score)).astype(float)
        assert calibrate_threshold(score, loss, alpha=0.1, conditional=False).n_calib == 600


# ---------------------------------------------------------------------------
class TestGuaranteeValidity:
    @pytest.mark.parametrize("alpha,delta", [(0.10, 0.10), (0.20, 0.10)])
    def test_empirical_coverage_holds(self, alpha, delta):
        """THE safety test: violation rate must not exceed delta.

        If this fails, the risk guarantee is not valid and no claim resting on it
        can be made.
        """
        out = simulate_coverage(n_calib=400, n_test=400, alpha=alpha, delta=delta,
                                n_trials=120, seed=7)
        assert out["violation_rate"] <= delta + 0.05, out

    def test_violation_is_measured_not_assumed(self):
        rng = np.random.default_rng(8)
        score = rng.random(500)
        loss = (rng.random(500) < 0.9).astype(float)     # shifted: much worse
        v = guarantee_violation(score, loss, lambda_hat=0.2, alpha=0.05)
        assert v["violated"] and v["violation"] > 0

    def test_violation_handles_empty_acceptance(self):
        v = guarantee_violation(np.zeros(10), np.zeros(10), lambda_hat=0.9, alpha=0.05)
        assert v["n_accepted"] == 0 and v["coverage"] == 0.0


# ---------------------------------------------------------------------------
class TestPerGroup:
    def test_underpowered_groups_are_flagged_not_fabricated(self):
        """A per-group guarantee from a handful of points must be refused."""
        rng = np.random.default_rng(9)
        n = 600
        score = rng.random(n)
        loss = (rng.random(n) < 0.2 * (1 - score)).astype(float)
        group = np.array(["big"] * 590 + ["tiny"] * 10)   # 10 < ceil(1/0.05)=20
        out = calibrate_per_group(score, loss, group, alpha=0.05, delta=0.1)
        assert "tiny" in out["underpowered_groups"]
        assert out["per_group"]["tiny"]["controlled"] is False
        assert out["per_group"]["big"]["n"] == 590

    def test_pooled_is_always_reported(self):
        rng = np.random.default_rng(10)
        score = rng.random(400)
        loss = (rng.random(400) < 0.2).astype(float)
        out = calibrate_per_group(score, loss, np.array(["a", "b"] * 200), alpha=0.2)
        assert "pooled" in out


# ---------------------------------------------------------------------------
class TestConformalScores:
    def test_aps_true_label_scores_shape(self):
        probs, y = synth(100)
        assert aps_scores(probs, y, randomize=False).shape == (100,)

    def test_aps_full_matrix_shape(self):
        probs, _ = synth(50)
        assert aps_scores(probs, randomize=False).shape == (50, K)

    def test_raps_penalty_increases_scores(self):
        probs, y = synth(200, seed=11)
        plain = raps_scores(probs, y, lam=0.0, randomize=False)
        pen = raps_scores(probs, y, lam=0.5, randomize=False)
        assert (pen >= plain - 1e-9).all()

    def test_confident_correct_scores_below_confident_wrong(self):
        """APS score of the top-ranked class equals its own mass; what matters is
        that a correct confident prediction scores far below a wrong one."""
        probs = np.full((1, K), 0.01); probs[0, 2] = 1 - 0.01 * (K - 1)
        s_right = aps_scores(probs, np.array([2]), randomize=False)[0]
        s_wrong = aps_scores(probs, np.array([5]), randomize=False)[0]
        assert s_right < s_wrong
        assert s_right == pytest.approx(probs[0, 2], abs=1e-9)


class TestConformalCalibrator:
    @pytest.mark.parametrize("method", ["aps", "raps"])
    def test_coverage_near_target(self, method):
        probs, y = synth(2000, seed=12)
        cal_p, cal_y, te_p, te_y = probs[:1000], y[:1000], probs[1000:], y[1000:]
        c = ConformalCalibrator(alpha=0.1, method=method).fit(cal_p, cal_y)
        ev = c.evaluate(te_p, te_y)
        assert ev["empirical_coverage"] >= 0.84      # target 0.90, finite-sample slack
        assert 1 <= ev["mean_set_size"] <= K

    def test_stricter_alpha_gives_larger_sets(self):
        probs, y = synth(1600, seed=13)
        cal_p, cal_y, te_p = probs[:800], y[:800], probs[800:]
        small = ConformalCalibrator(alpha=0.20).fit(cal_p, cal_y).predict_sets(te_p)
        big = ConformalCalibrator(alpha=0.02).fit(cal_p, cal_y).predict_sets(te_p)
        assert big.sum(1).mean() >= small.sum(1).mean()

    def test_class_conditional_reports_per_class(self):
        probs, y = synth(1200, seed=14)
        c = ConformalCalibrator(alpha=0.1, class_conditional=True).fit(probs[:600], y[:600])
        ev = c.evaluate(probs[600:], y[600:])
        assert len(ev["per_class"]) >= 5
        assert len(c.qhat_per_class) == K

    def test_underpowered_class_flagged(self):
        """MEL-like case: too few calibration points for a per-class guarantee."""
        probs, y = synth(400, seed=15)
        y[:] = 0
        y[:5] = 5                                   # only 5 of the rare class
        c = ConformalCalibrator(alpha=0.05, class_conditional=True).fit(probs, y)
        assert 5 in c.underpowered_classes

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit must be called"):
            ConformalCalibrator().predict_sets(synth(10)[0])

    def test_serialisable(self):
        probs, y = synth(300, seed=16)
        d = ConformalCalibrator(alpha=0.1).fit(probs, y).to_dict()
        assert d["qhat"] is not None and d["n_calib"] == 300


class TestAmbiguityRule:
    def test_mixed_set_is_ambiguous(self):
        s = np.zeros((1, K), bool); s[0, [2, 5]] = True       # NEV (benign) + MEL
        assert set_is_ambiguous(s, MALIGNANT)[0]

    def test_all_benign_is_not_ambiguous(self):
        s = np.zeros((1, K), bool); s[0, [1, 2, 3]] = True    # ACK, NEV, SEK
        assert not set_is_ambiguous(s, MALIGNANT)[0]

    def test_all_malignant_is_not_ambiguous(self):
        s = np.zeros((1, K), bool); s[0, [0, 4]] = True       # BCC, SCC
        assert not set_is_ambiguous(s, MALIGNANT)[0]

    def test_empty_set_counts_as_ambiguous(self):
        """No conformal label at this alpha is a reason to stop, not to guess."""
        assert set_is_ambiguous(np.zeros((1, K), bool), MALIGNANT)[0]

    def test_singleton_is_never_ambiguous(self):
        for c in range(K):
            s = np.zeros((1, K), bool); s[0, c] = True
            assert not set_is_ambiguous(s, MALIGNANT)[0]
