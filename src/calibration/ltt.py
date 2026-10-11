"""
src/calibration/ltt.py - Learn then Test: risk-controlled threshold selection.

Picks the deferral threshold so that

    P( R(lambda_hat) <= alpha )  >=  1 - delta

distribution-free, over the draw of the calibration set, assuming only
exchangeability.

Reference: Angelopoulos, Bates, Candès, Jordan & Lei, "Learn then Test:
Calibrating Predictive Algorithms to Achieve Risk Control", Annals of Applied
Statistics (2025), DOI 10.1214/24-AOAS1998. Verified 2026-10-11.

WHY LTT AND NOT CONFORMAL RISK CONTROL. CRC requires the risk to be monotone in
lambda and lambda to be one-dimensional. Our controlled risk is the malignancy
MISS RATE AMONG PROCEEDED CASES, whose denominator moves with lambda, so it is
not monotone. The LTT paper exists precisely to drop that requirement: it recasts
threshold selection as multiple hypothesis testing, so no monotonicity is needed.
That is the reason for this module rather than a CRC one.

Procedure:
  1. For each lambda on a grid, compute the empirical risk on calibration data.
  2. Form a valid p-value for H_lambda : R(lambda) > alpha
     (Hoeffding-Bentkus for a bounded loss).
  3. Walk the grid from conservative to permissive with FIXED-SEQUENCE testing,
     stopping at the first non-rejection. Fixed-sequence needs no multiplicity
     correction, which is what keeps the selected threshold from being
     needlessly conservative.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# p-values
# ---------------------------------------------------------------------------

def _h1(a: float, b: float) -> float:
    """KL between Bernoulli(a) and Bernoulli(b), used by the Hoeffding bound."""
    a = min(max(a, 1e-12), 1 - 1e-12)
    b = min(max(b, 1e-12), 1 - 1e-12)
    return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))


def hoeffding_p_value(risk_hat: float, alpha: float, n: int) -> float:
    """Hoeffding-style p-value for H: R > alpha, loss bounded in [0,1]."""
    if n <= 0:
        return 1.0
    if risk_hat >= alpha:
        return 1.0
    return float(np.exp(-n * _h1(risk_hat, alpha)))


def bentkus_p_value(risk_hat: float, alpha: float, n: int) -> float:
    """Bentkus p-value: e * P(Bin(n, alpha) <= ceil(n * risk_hat))."""
    if n <= 0:
        return 1.0
    k = int(np.ceil(n * risk_hat))
    return float(min(1.0, np.e * stats.binom.cdf(k, n, alpha)))


def hoeffding_bentkus_p_value(risk_hat: float, alpha: float, n: int) -> float:
    """min of the two bounds - valid, and tighter than either alone."""
    return float(min(hoeffding_p_value(risk_hat, alpha, n),
                     bentkus_p_value(risk_hat, alpha, n), 1.0))


# ---------------------------------------------------------------------------
# result
# ---------------------------------------------------------------------------

@dataclass
class LTTResult:
    lambda_hat: Optional[float]
    alpha: float
    delta: float
    n_calib: int
    risk_at_lambda: Optional[float]
    coverage_at_lambda: Optional[float]
    p_value: Optional[float]
    grid: List[float] = field(default_factory=list)
    risks: List[float] = field(default_factory=list)
    p_values: List[float] = field(default_factory=list)
    coverages: List[float] = field(default_factory=list)
    n_rejected: int = 0
    note: str = ""

    @property
    def controlled(self) -> bool:
        """True when some threshold achieved the target with 1-delta confidence."""
        return self.lambda_hat is not None

    def to_dict(self) -> Dict:
        return {
            "lambda_hat": self.lambda_hat, "alpha": self.alpha, "delta": self.delta,
            "n_calib": self.n_calib, "risk_at_lambda": self.risk_at_lambda,
            "coverage_at_lambda": self.coverage_at_lambda, "p_value": self.p_value,
            "controlled": self.controlled, "n_rejected": self.n_rejected,
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------

def calibrate_threshold(
    score: np.ndarray,
    loss: np.ndarray,
    alpha: float = 0.05,
    delta: float = 0.10,
    grid: Optional[Sequence[float]] = None,
    n_grid: int = 100,
    conditional: bool = True,
    min_coverage: float = 0.0,
) -> LTTResult:
    """Select the most permissive accept-threshold whose risk is provably <= alpha.

    Args:
        score: (N,) confidence; a case is ACCEPTED when score >= lambda.
        loss:  (N,) per-case loss in [0,1]. For a malignancy-miss guarantee this
               is 1 when a malignant lesion was called benign, else 0.
        alpha: target risk among accepted cases.
        delta: allowed failure probability over the calibration draw.
        conditional: if True (default) the controlled risk is the SELECTIVE risk
               (loss among accepted). If False it is the joint risk
               P(loss and accepted), which is monotone and so also usable with
               plain conformal risk control.
        min_coverage: refuse thresholds that accept less than this fraction. A
               threshold that accepts 3 cases will trivially pass the risk test
               and is useless.

    Returns an LTTResult; `controlled=False` means no threshold on the grid could
    be certified, which is itself a finding and must not be silently replaced by
    the least-bad option.
    """
    score = np.asarray(score, dtype=float).ravel()
    loss = np.asarray(loss, dtype=float).ravel()
    if score.size != loss.size:
        raise ValueError("score and loss must have the same length")
    if not (0.0 <= loss.min() and loss.max() <= 1.0):
        raise ValueError("loss must be bounded in [0, 1] for the HB bound to be valid")
    n_total = score.size

    if grid is None:
        # Descending = conservative (accept few) -> permissive (accept many).
        lo, hi = float(np.min(score)), float(np.max(score))
        grid = list(np.linspace(hi, lo, n_grid))
    else:
        grid = sorted(grid, reverse=True)

    risks, pvals, covs = [], [], []
    lambda_hat = risk_hat_sel = cov_sel = p_sel = None
    n_rejected = 0
    stopped = False
    note = ""

    for lam in grid:
        accepted = score >= lam
        n_acc = int(accepted.sum())
        cov = n_acc / n_total
        covs.append(cov)

        if n_acc == 0:
            risks.append(float("nan")); pvals.append(1.0)
            continue

        if conditional:
            r_hat = float(loss[accepted].mean())
            n_eff = n_acc              # the risk is an average over accepted cases
        else:
            r_hat = float((loss * accepted).mean())
            n_eff = n_total
        risks.append(r_hat)

        p = hoeffding_bentkus_p_value(r_hat, alpha, n_eff)
        pvals.append(p)

        if stopped:
            continue
        if p <= delta and cov >= min_coverage:
            # Rejected H_lambda: risk is controlled here. Keep walking to find a
            # MORE permissive lambda that still controls it.
            lambda_hat, risk_hat_sel, cov_sel, p_sel = lam, r_hat, cov, p
            n_rejected += 1
        elif lambda_hat is not None:
            # Fixed-sequence testing stops at the first non-rejection.
            stopped = True

    if lambda_hat is None:
        note = (f"no threshold on the grid achieved risk <= {alpha} at "
                f"confidence {1 - delta}; n={n_total}. With this calibration size "
                f"the smallest certifiable risk is bounded below by roughly "
                f"{1.0 / max(n_total, 1):.4f}.")

    return LTTResult(lambda_hat=lambda_hat, alpha=alpha, delta=delta,
                     n_calib=n_total, risk_at_lambda=risk_hat_sel,
                     coverage_at_lambda=cov_sel, p_value=p_sel,
                     grid=[float(g) for g in grid], risks=risks,
                     p_values=pvals, coverages=covs,
                     n_rejected=n_rejected, note=note)


def calibrate_per_group(
    score: np.ndarray, loss: np.ndarray, group: np.ndarray,
    alpha: float = 0.05, delta: float = 0.10, min_group_size: int = 0, **kw
) -> Dict:
    """Group-conditional (Mondrian) calibration - one threshold per group.

    Used for per-skin-tone control. A pooled threshold lets large groups absorb
    the error budget while a small group silently exceeds alpha, so for a
    fairness claim the calibration must be per group.

    Groups too small to certify are reported in `underpowered`, NOT given a
    fabricated threshold. Claiming a per-group guarantee from a handful of
    calibration points is exactly the kind of overclaim this project is trying to
    avoid; the honest move is to merge sparse groups or report the limitation.
    """
    score = np.asarray(score, dtype=float).ravel()
    loss = np.asarray(loss, dtype=float).ravel()
    group = np.asarray(group).ravel()

    need = max(min_group_size, int(np.ceil(1.0 / alpha)))
    out: Dict[str, Dict] = {}
    underpowered: List[str] = []

    for g in np.unique(group):
        m = group == g
        if m.sum() < need:
            underpowered.append(str(g))
            out[str(g)] = {"n": int(m.sum()), "controlled": False,
                           "note": f"n={int(m.sum())} < {need} required for alpha={alpha}"}
            continue
        res = calibrate_threshold(score[m], loss[m], alpha=alpha, delta=delta, **kw)
        out[str(g)] = {**res.to_dict(), "n": int(m.sum())}

    pooled = calibrate_threshold(score, loss, alpha=alpha, delta=delta, **kw)
    return {"pooled": pooled.to_dict(), "per_group": out,
            "underpowered_groups": underpowered,
            "min_group_size_required": need}


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

def guarantee_violation(score: np.ndarray, loss: np.ndarray,
                        lambda_hat: float, alpha: float) -> Dict:
    """Realised risk at a fitted threshold, minus the target.

    Positive means the guarantee was violated on this data. Under shift it will
    be positive, and measuring how large is a primary result rather than a
    footnote - the guarantee is only valid under exchangeability.
    """
    score = np.asarray(score, dtype=float).ravel()
    loss = np.asarray(loss, dtype=float).ravel()
    accepted = score >= lambda_hat
    n_acc = int(accepted.sum())
    if n_acc == 0:
        return {"coverage": 0.0, "realised_risk": float("nan"),
                "violation": float("nan"), "n_accepted": 0,
                "note": "nothing accepted at this threshold"}
    realised = float(loss[accepted].mean())
    return {"coverage": n_acc / score.size, "realised_risk": realised,
            "target_alpha": alpha, "violation": realised - alpha,
            "violated": bool(realised > alpha), "n_accepted": n_acc}


def simulate_coverage(n_calib: int = 500, n_test: int = 500, alpha: float = 0.1,
                      delta: float = 0.1, n_trials: int = 200,
                      base_rate: float = 0.3, seed: int = 0) -> Dict:
    """Sanity check on synthetic EXCHANGEABLE data.

    Draws calibration and test sets from the same distribution, calibrates, and
    counts how often the realised test risk exceeds alpha. That rate should be
    <= delta. If it is not, the implementation is wrong - this is the test that
    catches an invalid p-value or an off-by-one in the fixed-sequence walk.
    """
    rng = np.random.default_rng(seed)
    violations = 0
    uncontrolled = 0
    coverages: List[float] = []

    for _ in range(n_trials):
        def draw(n):
            s = rng.random(n)
            # higher score -> lower chance of loss, so risk is informative
            l = (rng.random(n) < base_rate * (1.0 - s)).astype(float)
            return s, l

        s_cal, l_cal = draw(n_calib)
        s_test, l_test = draw(n_test)
        res = calibrate_threshold(s_cal, l_cal, alpha=alpha, delta=delta)
        if not res.controlled:
            uncontrolled += 1
            continue
        v = guarantee_violation(s_test, l_test, res.lambda_hat, alpha)
        coverages.append(v["coverage"])
        if v.get("violated"):
            violations += 1

    done = n_trials - uncontrolled
    return {
        "n_trials": n_trials, "n_uncontrolled": uncontrolled,
        "violation_rate": violations / max(done, 1),
        "delta": delta,
        "passes": (violations / max(done, 1)) <= max(delta, 1e-9) + 0.05,
        "mean_coverage": float(np.mean(coverages)) if coverages else float("nan"),
    }
