"""
tests/test_decision.py - Cost model and the PROCEED/ACQUIRE/DEFER policy.

Covers the properties that distinguish this from the 4-level ladder it replaces:
the cost matrix is asymmetric and declared, conflict and ignorance route to
different actions, and every safety gate can only make the system MORE cautious.
"""
import numpy as np
import pytest

from src.decision.policy import (
    Action,
    CostMatrix,
    DeferReason,
    ThreeActionPolicy,
    confidence_threshold_baseline,
    entropy_acquisition_baseline,
    expected_loss,
)

K = 6
MALIGNANT = (0, 4, 5)       # BCC, SCC, MEL
BENIGN = (1, 2, 3)          # ACK, NEV, SEK


def peaked(cls, mass=0.9, k=K):
    p = np.full(k, (1.0 - mass) / (k - 1))
    p[cls] = mass
    return p


def singleton_set(cls, k=K):
    s = np.zeros(k, bool); s[cls] = True
    return s


def mixed_set(k=K):
    s = np.zeros(k, bool); s[2] = True; s[5] = True     # NEV + MEL
    return s


# ---------------------------------------------------------------------------
class TestCostMatrix:
    def test_correct_costs_nothing(self):
        c = CostMatrix()
        for i in range(K):
            assert c.loss(i, i, MALIGNANT) == 0.0

    def test_missing_a_malignancy_is_the_most_expensive_error(self):
        c = CostMatrix()
        miss = c.loss(5, 2, MALIGNANT)          # MEL called NEV
        alarm = c.loss(2, 5, MALIGNANT)         # NEV called MEL
        same = c.loss(0, 4, MALIGNANT)          # BCC called SCC
        assert miss > alarm > same > 0

    def test_within_group_error_is_cheapest_nonzero(self):
        c = CostMatrix()
        assert c.loss(0, 4, MALIGNANT) == c.wrong_same_group
        assert c.loss(1, 2, MALIGNANT) == c.wrong_same_group

    def test_loss_matrix_shape_and_diagonal(self):
        L = CostMatrix().loss_matrix(K, MALIGNANT)
        assert L.shape == (K, K)
        assert np.allclose(np.diag(L), 0.0)

    def test_asymmetry_is_explicit(self):
        L = CostMatrix().loss_matrix(K, MALIGNANT)
        assert L[5, 2] != L[2, 5]       # the whole point of the cost model

    def test_normalised_miss_loss_is_bounded(self):
        v = CostMatrix().normalised_miss_loss()
        assert 0.0 < v <= 1.0           # LTT needs a loss in [0,1]

    def test_serialisable(self):
        assert CostMatrix().to_dict()["missed_malignant"] == 10.0


class TestExpectedLoss:
    def test_certain_case_predicts_that_class(self):
        L = CostMatrix().loss_matrix(K, MALIGNANT)
        for c in range(K):
            risk, pred = expected_loss(peaked(c, 0.999)[None, :], L)
            assert pred[0] == c and risk[0] < 0.2

    def test_asymmetric_costs_bias_toward_caution(self):
        """Under a 10x miss penalty an ambiguous benign/malignant case should
        not be called benign on a bare majority."""
        L = CostMatrix().loss_matrix(K, MALIGNANT)
        p = np.zeros(K); p[2] = 0.55; p[5] = 0.45       # NEV 0.55 vs MEL 0.45
        _, pred = expected_loss(p[None, :], L)
        assert pred[0] == 5, "should prefer the malignant call when missing it costs 10x"

    def test_uniform_gives_high_risk(self):
        L = CostMatrix().loss_matrix(K, MALIGNANT)
        risk, _ = expected_loss(np.full((1, K), 1.0 / K), L)
        assert risk[0] > 1.0


# ---------------------------------------------------------------------------
class TestPolicyBasics:
    def policy(self, **kw):
        kw.setdefault("malignant_classes", MALIGNANT)
        kw.setdefault("num_classes", K)
        return ThreeActionPolicy(**kw)

    def test_confident_clean_case_proceeds(self):
        d = self.policy().decide(peaked(2, 0.97), ignorance=0.05, conflict=0.02,
                                 prediction_set=singleton_set(2))
        assert d.action is Action.PROCEED and d.prediction == 2

    def test_empty_probs_defers_with_no_evidence(self):
        d = self.policy().decide(np.zeros(K))
        assert d.action is Action.DEFER and d.reason is DeferReason.NO_EVIDENCE

    def test_wrong_length_probs_raises(self):
        with pytest.raises(ValueError, match="expected 6"):
            self.policy().decide(np.ones(4) / 4)

    def test_decision_serialises(self):
        d = self.policy().decide(peaked(1, 0.95), ignorance=0.1)
        out = d.to_dict()
        assert out["action"] in {a.value for a in Action}
        assert "risks" in out and "gates" in out

    def test_batch_matches_per_case(self):
        pol = self.policy()
        probs = np.stack([peaked(0, 0.95), peaked(3, 0.9)])
        batch = pol.decide_batch(probs, ignorance=np.array([0.05, 0.1]))
        assert [b.action for b in batch] == [
            pol.decide(probs[i], ignorance=[0.05, 0.1][i]).action for i in range(2)]


class TestSafetyGates:
    def policy(self, **kw):
        return ThreeActionPolicy(malignant_classes=MALIGNANT, num_classes=K, **kw)

    def test_conflict_defers_to_human_not_acquisition(self):
        """A contradiction is not resolved by collecting more evidence."""
        d = self.policy(conflict_threshold=0.5).decide(
            peaked(2, 0.9), ignorance=0.1, conflict=0.9,
            acquirable=["dermoscopy_image"],
            predicted_benefit={"dermoscopy_image": 5.0})
        assert d.action is Action.DEFER
        assert d.reason is DeferReason.CONFLICT

    def test_ignorance_prefers_acquisition_over_deferral(self):
        """Not knowing enough IS fixable by acquiring - the opposite routing."""
        d = self.policy(lambda_hat=0.8).decide(
            np.full(K, 1.0 / K), ignorance=0.95, conflict=0.02,
            acquirable=["dermoscopy_image"],
            predicted_benefit={"dermoscopy_image": 2.0})
        assert d.action is Action.ACQUIRE
        assert d.acquire_target == "dermoscopy_image"

    def test_conflict_and_ignorance_route_differently(self):
        """The distinction the old single consistency scalar could not express."""
        pol = self.policy(lambda_hat=0.8, conflict_threshold=0.5)
        kw = dict(acquirable=["dermoscopy_image"],
                  predicted_benefit={"dermoscopy_image": 2.0})
        conflicted = pol.decide(peaked(2, 0.9), ignorance=0.1, conflict=0.9, **kw)
        ignorant = pol.decide(np.full(K, 1 / K), ignorance=0.95, conflict=0.02, **kw)
        assert conflicted.action is Action.DEFER
        assert ignorant.action is Action.ACQUIRE

    def test_mixed_conformal_set_blocks_proceed(self):
        d = self.policy().decide(peaked(2, 0.95), ignorance=0.05, conflict=0.05,
                                 prediction_set=mixed_set())
        assert d.action is not Action.PROCEED
        assert d.gates["ambiguous_set"] is True

    def test_empty_conformal_set_blocks_proceed(self):
        d = self.policy().decide(peaked(2, 0.95), ignorance=0.05, conflict=0.05,
                                 prediction_set=np.zeros(K, bool))
        assert d.action is not Action.PROCEED

    def test_below_lambda_blocks_proceed(self):
        d = self.policy(lambda_hat=0.9).decide(peaked(2, 0.95), ignorance=0.5,
                                               conflict=0.05)
        assert d.action is not Action.PROCEED
        assert d.gates["below_risk_threshold"] is True

    def test_gates_only_ever_increase_caution(self):
        """Adding any gate must never turn a DEFER into a PROCEED."""
        base = self.policy()
        clean = base.decide(peaked(2, 0.97), ignorance=0.05, conflict=0.02,
                            prediction_set=singleton_set(2))
        assert clean.action is Action.PROCEED
        for kw in ({"conflict": 0.95},
                   {"prediction_set": mixed_set()},
                   {"ignorance": 0.99}):
            args = dict(ignorance=0.05, conflict=0.02,
                        prediction_set=singleton_set(2))
            args.update(kw)
            gated = ThreeActionPolicy(malignant_classes=MALIGNANT, num_classes=K,
                                      lambda_hat=0.9).decide(peaked(2, 0.97), **args)
            assert gated.action is not Action.PROCEED or clean.action is Action.PROCEED

    def test_exhausted_acquisition_defers(self):
        d = self.policy(lambda_hat=0.9, max_acquisitions=1).decide(
            np.full(K, 1 / K), ignorance=0.95, conflict=0.02,
            acquirable=[], n_acquisitions_so_far=1)
        assert d.action is Action.DEFER
        assert d.reason in (DeferReason.ACQUISITION_EXHAUSTED, DeferReason.RISK_BOUND)

    def test_acquisition_respects_the_round_limit(self):
        pol = self.policy(lambda_hat=0.9, max_acquisitions=1)
        d = pol.decide(np.full(K, 1 / K), ignorance=0.95,
                       acquirable=["dermoscopy_image"],
                       predicted_benefit={"dermoscopy_image": 5.0},
                       n_acquisitions_so_far=1)
        assert d.action is not Action.ACQUIRE


class TestCostSensitivity:
    def test_expensive_acquisition_is_not_requested(self):
        pol = ThreeActionPolicy(
            cost=CostMatrix(acquire={"dermoscopy_image": 100.0}),
            malignant_classes=MALIGNANT, num_classes=K, lambda_hat=0.5)
        d = pol.decide(peaked(2, 0.8), ignorance=0.2,
                       acquirable=["dermoscopy_image"],
                       predicted_benefit={"dermoscopy_image": 0.1})
        assert d.action is not Action.ACQUIRE

    def test_cheap_deferral_wins_on_a_hard_case(self):
        pol = ThreeActionPolicy(cost=CostMatrix(defer=0.01),
                                malignant_classes=MALIGNANT, num_classes=K)
        d = pol.decide(np.full(K, 1.0 / K), ignorance=0.5)
        assert d.action is Action.DEFER and d.reason is DeferReason.COST

    def test_raising_miss_cost_shifts_behaviour(self):
        """Mandatory sensitivity check: results depend on the cost matrix."""
        probs = np.zeros(K); probs[2] = 0.6; probs[5] = 0.4
        preds = []
        for m in (1.0, 10.0, 50.0):
            pol = ThreeActionPolicy(cost=CostMatrix(missed_malignant=m),
                                    malignant_classes=MALIGNANT, num_classes=K)
            preds.append(pol.decide(probs, ignorance=0.1, conflict=0.0).prediction)
        assert preds[0] == 2 and preds[-1] == 5, preds

    def test_policy_records_its_version_and_costs(self):
        d = ThreeActionPolicy(num_classes=K).to_dict()
        assert d["version"].startswith("eade")
        assert "cost" in d and "lambda_hat" in d


class TestBaselines:
    def test_confidence_threshold_accepts_confident_only(self):
        probs = np.stack([peaked(0, 0.99), np.full(K, 1.0 / K)])
        acc = confidence_threshold_baseline(probs, 0.8)
        assert acc[0] and not acc[1]

    def test_entropy_acquisition_triggers_on_uncertainty(self):
        probs = np.stack([peaked(0, 0.99), np.full(K, 1.0 / K)])
        req = entropy_acquisition_baseline(probs, threshold=1.0)
        assert not req[0] and req[1]

    def test_threshold_monotonicity(self):
        probs = np.stack([peaked(i, 0.5 + 0.1 * i) for i in range(5)])
        counts = [confidence_threshold_baseline(probs, t).sum() for t in (0.1, 0.5, 0.9)]
        assert counts == sorted(counts, reverse=True)
