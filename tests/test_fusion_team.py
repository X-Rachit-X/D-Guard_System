"""tests/test_fusion_team.py - Devil's Advocate + iterative diagnosis-adaptive fusion."""
import numpy as np

from src.fusion.adaptive import AdaptiveFusion, EvidenceManager, OODDetector, js_consistency

K, M, N = 4, 3, 200
rng = np.random.default_rng(0)


def team():
    y = rng.integers(0, K, N)
    good = np.eye(K)[y] * 0.7 + 0.3 / K                    # reliable specialist
    noisy = rng.dirichlet(np.ones(K), N)                    # useless specialist
    mid = 0.5 * good + 0.5 * noisy
    return np.stack([good, noisy, mid], 1), y


def test_weights_sum_to_one_and_absent_gets_zero():
    P, y = team()
    pres = np.ones((N, M), bool)
    pres[:, 2] = False
    f = AdaptiveFusion(names=list("abc")).fit_prior(P, pres, y, K)
    p, w = f.fuse(P, np.ones((N, M)), pres)
    assert np.allclose(w.sum(1), 1) and np.allclose(w[:, 2], 0)
    assert np.allclose(p.sum(1), 1)


def test_learned_reliability_downweights_the_useless_specialist():
    P, y = team()
    pres = np.ones((N, M), bool)
    Q = np.ones((N, M))
    R = EvidenceManager(names=list("abc")).fit(P, Q, pres, y).reliability(P, Q, pres)
    assert R[:, 0].mean() > R[:, 1].mean()
    f = AdaptiveFusion(names=list("abc")).fit_prior(P, pres, y, K)
    f.fit_params(P, Q * R, pres, y, K)
    _, w = f.fuse(P, Q * R, pres)
    assert w[:, 0].mean() > w[:, 1].mean()


def test_consistency_and_ood():
    P, y = team()
    pres = np.ones((N, M), bool)
    same = np.repeat(P[:, :1], M, axis=1)
    assert js_consistency(same, pres).min() > 0.99
    Z = rng.normal(size=(N, 8)) + np.eye(K)[y] @ rng.normal(size=(K, 8)) * 3
    ood = OODDetector().fit(Z, y, K).calibrate(Z, q=99)
    assert ood.flag(Z).mean() <= 0.02
    assert ood.flag(-Z * 10).mean() > 0.5
