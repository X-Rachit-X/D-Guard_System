"""
src/eval/selective.py - Metrics for a system that is allowed not to answer.

Accuracy alone is uninterpretable for an abstaining system, so the primary
family here is risk-vs-coverage.

AUGRC is the primary multi-threshold metric, following Traub, Bungert, Lüth,
Baumgartner, Maier-Hein, Maier-Hein & Jäger, "Overcoming Common Flaws in the
Evaluation of Selective Classification Systems", NeurIPS 2024 (spotlight),
arXiv:2407.01032. Definitions taken from that paper (verified 2026-10-11):

    generalized risk   GR(tau) = (1/N) * sum_i  loss_i * 1[g(x_i) >= tau]
    selective risk     SR(tau) =  sum_i loss_i * 1[g >= tau] / sum_i 1[g >= tau]
    AUGRC              = integral_0^1  P(err, g >= tau)  d P(g >= tau)

The distinction that matters: generalized risk divides by N, selective risk
divides by the number ACCEPTED. The authors show selective risk "is not suitable
for aggregation over rejection thresholds" because it over-weights
high-confidence failures, which breaks monotonicity and ranking
interpretability; AURC and e-AURC inherit those flaws. AURC is still reported
here, for comparability with older work only.

AUGRC is bounded in [0, 1/2]; lower is better.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _as_loss(y_true: np.ndarray, y_pred: np.ndarray,
             loss: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-case loss. Defaults to 0/1 misclassification."""
    if loss is not None:
        return np.asarray(loss, dtype=float).ravel()
    return (np.asarray(y_pred).ravel() != np.asarray(y_true).ravel()).astype(float)


def _order_by_confidence(score: np.ndarray) -> np.ndarray:
    """Indices sorted most-confident first, with a deterministic tie-break."""
    score = np.asarray(score, dtype=float).ravel()
    # argsort on (-score, index) keeps ties in input order -> reproducible curves
    return np.lexsort((np.arange(score.size), -score))


# ---------------------------------------------------------------------------
# risk-coverage
# ---------------------------------------------------------------------------

@dataclass
class RiskCoverage:
    coverage: np.ndarray          # (N,) k/N for k = 1..N
    selective_risk: np.ndarray    # (N,) SR at each coverage
    generalized_risk: np.ndarray  # (N,) GR at each coverage
    augrc: float
    aurc: float
    n: int

    def risk_at_coverage(self, target: float) -> float:
        """Selective risk at the largest coverage <= target."""
        idx = np.searchsorted(self.coverage, target, side="right") - 1
        return float(self.selective_risk[max(idx, 0)])

    def coverage_at_risk(self, alpha: float) -> float:
        """Largest coverage whose selective risk is <= alpha; 0.0 if never."""
        ok = np.flatnonzero(self.selective_risk <= alpha)
        return float(self.coverage[ok[-1]]) if ok.size else 0.0

    def to_dict(self) -> Dict:
        return {
            "augrc": self.augrc, "aurc": self.aurc, "n": self.n,
            "risk_at_coverage": {f"{c:.2f}": self.risk_at_coverage(c)
                                 for c in (0.2, 0.4, 0.6, 0.8, 1.0)},
            "coverage_at_risk": {f"{a:.2f}": self.coverage_at_risk(a)
                                 for a in (0.05, 0.10, 0.20)},
        }


def risk_coverage_curve(y_true, y_pred, score, loss=None) -> RiskCoverage:
    """Full risk-coverage curve plus AUGRC and AURC.

    `score` is the confidence used to rank cases (higher = more confident), e.g.
    1 - ignorance from the evidential layer, or max softmax for a baseline.
    """
    losses = _as_loss(y_true, y_pred, loss)
    n = losses.size
    if n == 0:
        raise ValueError("empty input")

    order = _order_by_confidence(score)
    l_sorted = losses[order]
    k = np.arange(1, n + 1, dtype=float)

    cum = np.cumsum(l_sorted)
    coverage = k / n
    selective = cum / k        # divided by #accepted
    generalized = cum / n      # divided by N  <- the Traub et al. definition

    # AUGRC = integral of GR d(coverage), discretised by the TRAPEZOIDAL rule
    # with the origin (coverage=0, GR=0) included.
    #
    # A right-endpoint Riemann sum (i.e. `generalized.mean()`) overshoots by
    # 1/(2n): for an all-wrong predictor it returns (n+1)/(2n) = 0.5025 at
    # n=200, breaking the metric's documented [0, 1/2] upper bound. The
    # trapezoid gives exactly 1/2 in that case, which is the correct limit.
    gr_with_origin = np.concatenate(([0.0], generalized))
    cov_with_origin = np.concatenate(([0.0], coverage))
    augrc = float(np.trapezoid(gr_with_origin, cov_with_origin)
                  if hasattr(np, "trapezoid") else
                  np.trapz(gr_with_origin, cov_with_origin))

    # AURC keeps the conventional mean-over-thresholds form so that numbers
    # remain comparable with published AURC values. Reported for comparability
    # only - Traub et al. show selective risk is unsuitable for aggregation
    # across thresholds because it over-weights high-confidence failures.
    aurc = float(selective.mean())
    return RiskCoverage(coverage, selective, generalized, augrc, aurc, n)


def augrc(y_true, y_pred, score, loss=None) -> float:
    """Area under the generalized risk-coverage curve. Lower is better, in [0, 0.5]."""
    return risk_coverage_curve(y_true, y_pred, score, loss).augrc


def aurc(y_true, y_pred, score, loss=None) -> float:
    """Area under the (selective) risk-coverage curve. Reported for comparability only."""
    return risk_coverage_curve(y_true, y_pred, score, loss).aurc


def selective_risk(y_true, y_pred, accepted, loss=None) -> Tuple[float, float]:
    """(selective risk, coverage) for an explicit accept mask.

    This is the pair to report for a deployed policy: a risk number without its
    coverage is not interpretable, since refusing everything gives risk 0.
    """
    losses = _as_loss(y_true, y_pred, loss)
    acc = np.asarray(accepted).ravel().astype(bool)
    cov = float(acc.mean())
    if acc.sum() == 0:
        return float("nan"), 0.0
    return float(losses[acc].mean()), cov


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------

def adaptive_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """Equal-MASS binning ECE.

    Equal-width bins are a poor choice once a policy abstains: confidences bunch
    up near the threshold and most bins end up empty or overloaded.
    """
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true).ravel()
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    n = conf.size
    if n == 0:
        return float("nan")
    order = np.argsort(conf)
    total = 0.0
    for chunk in np.array_split(order, min(n_bins, n)):
        if chunk.size:
            total += (chunk.size / n) * abs(correct[chunk].mean() - conf[chunk].mean())
    return float(total)


def classwise_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """Mean ECE computed per class (one-vs-rest), then averaged.

    Reported because class-dependent miscalibration directly distorts selective
    decisions even when the pooled ECE looks acceptable.
    """
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true).ravel()
    k = probs.shape[1]
    out = []
    for c in range(k):
        p = probs[:, c]
        hit = (y_true == c).astype(float)
        order = np.argsort(p)
        e = 0.0
        for chunk in np.array_split(order, min(n_bins, p.size)):
            if chunk.size:
                e += (chunk.size / p.size) * abs(hit[chunk].mean() - p[chunk].mean())
        out.append(e)
    return float(np.mean(out))


def brier_decomposition(probs: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """Total Brier score plus its reliability / resolution / uncertainty parts.

    An evidence-management claim is specifically a claim about RELIABILITY, so
    reporting only the total hides the quantity of interest.
    """
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true).ravel()
    n, k = probs.shape
    onehot = np.eye(k)[y_true]
    total = float(((probs - onehot) ** 2).sum(axis=1).mean())

    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    base = correct.mean()
    rel = res = 0.0
    order = np.argsort(conf)
    for chunk in np.array_split(order, min(10, n)):
        if not chunk.size:
            continue
        w = chunk.size / n
        rel += w * (conf[chunk].mean() - correct[chunk].mean()) ** 2
        res += w * (correct[chunk].mean() - base) ** 2
    return {"brier": total, "reliability": float(rel), "resolution": float(res),
            "uncertainty": float(base * (1.0 - base))}


# ---------------------------------------------------------------------------
# acquisition efficiency
# ---------------------------------------------------------------------------

@dataclass
class AcquisitionEfficiency:
    rate: float
    gain_policy: float
    gain_random: float
    gain_oracle: float
    efficiency: float
    n_acquired: int

    def to_dict(self) -> Dict:
        return {"rate": self.rate, "gain_policy": self.gain_policy,
                "gain_random": self.gain_random, "gain_oracle": self.gain_oracle,
                "acquisition_efficiency": self.efficiency,
                "n_acquired": self.n_acquired}


def acquisition_efficiency(benefit: np.ndarray, policy_score: np.ndarray,
                           rate: float, seed: int = 42,
                           n_random: int = 200) -> AcquisitionEfficiency:
    """AE(r): how much of the oracle benefit a policy captures at a matched rate.

        AE(r) = (G_policy - G_random) / (G_oracle - G_random)

    AE = 1 means oracle-level targeting, 0 means no better than random, and
    negative means actively worse than random - which is a real possible outcome
    and should be reported rather than clipped.

    Args:
        benefit:      (N,) realised loss reduction from acquiring, per case.
                      Negative entries are cases where acquiring HURT.
        policy_score: (N,) the policy's predicted benefit (ranking only).
        rate:         fraction of cases to acquire for.
        n_random:     random draws to average the random baseline over.
    """
    benefit = np.asarray(benefit, dtype=float).ravel()
    policy_score = np.asarray(policy_score, dtype=float).ravel()
    if benefit.size != policy_score.size:
        raise ValueError("benefit and policy_score must have the same length")
    n = benefit.size
    m = int(round(rate * n))
    if m <= 0:
        return AcquisitionEfficiency(rate, 0.0, 0.0, 0.0, float("nan"), 0)

    g_policy = float(benefit[_order_by_confidence(policy_score)[:m]].sum())
    # Oracle acquires exactly the m cases with the largest realised benefit.
    g_oracle = float(np.sort(benefit)[::-1][:m].sum())

    rng = np.random.default_rng(seed)
    g_random = float(np.mean([benefit[rng.choice(n, m, replace=False)].sum()
                              for _ in range(n_random)]))

    denom = g_oracle - g_random
    eff = float((g_policy - g_random) / denom) if abs(denom) > 1e-12 else float("nan")
    return AcquisitionEfficiency(rate, g_policy, g_random, g_oracle, eff, m)


def acquisition_efficiency_curve(benefit, policy_score,
                                 rates: Sequence[float] = (0.1, 0.2, 0.3, 0.5, 0.7),
                                 seed: int = 42) -> Dict[str, Dict]:
    return {f"{r:.2f}": acquisition_efficiency(benefit, policy_score, r, seed).to_dict()
            for r in rates}


def negative_information_rate(loss_before: np.ndarray,
                              loss_after: np.ndarray) -> Dict[str, float]:
    """Fraction of cases where acquiring a modality INCREASED the loss.

    Magnitude is reported alongside the rate, because a marginal increase and a
    catastrophic one should not count the same.
    """
    before = np.asarray(loss_before, dtype=float).ravel()
    after = np.asarray(loss_after, dtype=float).ravel()
    worse = after > before
    return {
        "nir": float(worse.mean()),
        "mean_increase_when_worse": float((after - before)[worse].mean()) if worse.any() else 0.0,
        "max_increase": float((after - before).max()) if before.size else 0.0,
        "mean_change": float((after - before).mean()),
    }


# ---------------------------------------------------------------------------
# net benefit (decision curve analysis)
# ---------------------------------------------------------------------------

def net_benefit(y_binary: np.ndarray, p_positive: np.ndarray,
                thresholds: Sequence[float] = tuple(np.arange(0.01, 0.51, 0.01))
                ) -> Dict[str, List[float]]:
    """Decision-curve net benefit (Vickers & Elkin, 2006).

        NB(p_t) = TP/N - (FP/N) * p_t/(1-p_t)

    Reported against the two reference strategies ("treat all", "treat none") so
    a reader can see whether the model beats doing nothing.
    """
    y = np.asarray(y_binary).ravel().astype(int)
    p = np.asarray(p_positive, dtype=float).ravel()
    n = y.size
    prev = y.mean()
    nb_model, nb_all, ths = [], [], []
    for t in thresholds:
        t = float(t)
        pred = p >= t
        tp = float(np.sum(pred & (y == 1)))
        fp = float(np.sum(pred & (y == 0)))
        w = t / (1.0 - t) if t < 1.0 else np.inf
        nb_model.append(tp / n - (fp / n) * w)
        nb_all.append(prev - (1.0 - prev) * w)
        ths.append(t)
    return {"threshold": ths, "net_benefit_model": nb_model,
            "net_benefit_treat_all": nb_all,
            "net_benefit_treat_none": [0.0] * len(ths)}


# ---------------------------------------------------------------------------
# bundle
# ---------------------------------------------------------------------------

def selective_report(y_true, probs, score, accepted=None, loss=None) -> Dict:
    """Everything a selective-prediction results table needs, in one call."""
    y_true = np.asarray(y_true).ravel()
    probs = np.asarray(probs, dtype=float)
    y_pred = probs.argmax(axis=1)

    rc = risk_coverage_curve(y_true, y_pred, score, loss)
    out: Dict = {
        "n": int(y_true.size),
        "accuracy": float((y_pred == y_true).mean()),
        "augrc": rc.augrc,
        "aurc": rc.aurc,
        "risk_coverage": rc.to_dict(),
        "adaptive_ece": adaptive_ece(probs, y_true),
        "classwise_ece": classwise_ece(probs, y_true),
        **brier_decomposition(probs, y_true),
    }
    if accepted is not None:
        risk, cov = selective_risk(y_true, y_pred, accepted, loss)
        out["policy"] = {"selective_risk": risk, "coverage": cov,
                         "n_accepted": int(np.asarray(accepted).ravel().sum())}
    return out
