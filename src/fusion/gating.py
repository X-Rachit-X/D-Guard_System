"""
src/fusion/gating.py - Devil's Advocate v2: a LEARNED evidence gate.

The v1 fusion (src/fusion/adaptive.py) weights a specialist by
    a*log(Q*R) + b*log(prior) + g*log(penalty)
i.e. only quality x reliability, the measured prior and the conflict penalty
reach the weights; U, C, M and OOD are shown but never used. The original
paper's formula E_m = Q*R*U*C*S*M has the same blind spot: C and S are the same
for every specialist of a case, so they cancel when the weights are normalised.

v2 lets EVERY evidence dimension move the weights, per case and per specialist,
with the mapping learned from validation lesions instead of hand-set:

    phi_m(x) = [ log Q_m, log R_m, U_m (entropy), confidence_m, margin_m,
                 agreement_m, log utility_m (= W_prior[y0, m]),
                 C(x), M(x), OOD(x) ]
    w_m(x)   = softmax_m( beta_m + theta_m . phi_m(x) )      over present specialists
    p(x)     = sum_m w_m(x) p_m(x)          ("linear" pool)
          or   normalise( prod_m p_m(x)^w_m(x) )   ("log" pool, product of experts)
    p        <- temperature-scaled p

theta, beta, T are fitted by minimising the negative log-likelihood (+ L2) on
validation lesions. Because the same validation lesions later calibrate the
conformal sets, `cross_fit` returns OUT-OF-FOLD fused probabilities for them, so
the downstream guarantees are not computed on in-sample fits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from src.fusion.adaptive import EPS, js_consistency, norm_entropy

FEATURES = ["log Q", "log R", "uncertainty U", "confidence", "margin", "agreement",
            "log utility", "consistency C", "missingness M", "OOD distance"]


def evidence_features(P, Q, R, present, W_prior, ood) -> np.ndarray:
    """(N, M, K) specialist distributions -> (N, M, F) evidence features."""
    N, M, K = P.shape
    srt = np.sort(P, axis=2)
    conf, margin = srt[:, :, -1], srt[:, :, -1] - srt[:, :, -2]
    top = P.argmax(2)
    agree = np.zeros((N, M))
    for m in range(M):
        others = present.copy()
        others[:, m] = False
        same = (top == top[:, [m]]) & others
        agree[:, m] = np.where(others.sum(1) > 0, same.sum(1) / np.maximum(others.sum(1), 1), 0.5)
    # provisional diagnosis = reliability-weighted average, for the utility look-up
    w0 = np.where(present, R + EPS, 0.0)
    y0 = ((w0 / w0.sum(1, keepdims=True))[:, :, None] * P).sum(1).argmax(1)
    util = W_prior[y0] if W_prior is not None else np.full((N, M), 0.5)
    C = js_consistency(P, present)
    Mmiss = 1.0 - present.mean(1)
    ood = np.zeros(N) if ood is None else np.clip(ood, 0, 5)
    case = np.stack([C, Mmiss, ood], 1)[:, None, :].repeat(M, 1)
    F = np.concatenate([
        np.stack([np.log(np.clip(Q, 1e-3, 1)), np.log(np.clip(R, 1e-3, 1)),
                  norm_entropy(P), conf, margin, agree, np.log(np.clip(util, 1e-3, 1))], 2),
        case], 2)
    return F.astype(np.float32)


@dataclass
class LearnedGate:
    names: List[str] = field(default_factory=list)
    pool: str = "linear"            # "linear" (mixture) or "log" (product of experts)
    l2: float = 1e-2
    theta: Optional[np.ndarray] = None   # (M, F)
    beta: Optional[np.ndarray] = None    # (M,)
    mu: Optional[np.ndarray] = None      # feature standardisation
    sd: Optional[np.ndarray] = None
    temperature: float = 1.0

    # ------------------------------------------------------------------ core
    def _weights(self, F, present, theta, beta, xp):
        z = (F - self.mu) / self.sd
        logits = (z * theta[None]).sum(2) + beta[None]
        logits = xp.where(present, logits, -1e9)
        logits = logits - logits.max(1, keepdims=True)
        e = xp.exp(logits) * present
        return e / e.sum(1, keepdims=True)

    def _pool(self, P, w, xp):
        if self.pool == "linear":
            return (w[:, :, None] * P).sum(1)
        lp = (w[:, :, None] * xp.log(xp.clip(P, 1e-6, 1))).sum(1)
        lp = lp - lp.max(1, keepdims=True)
        e = xp.exp(lp)
        return e / e.sum(1, keepdims=True)

    def fit(self, P, F, present, y):
        import torch
        N, M, Fd = F.shape
        pres = present.astype(bool)
        self.mu = F[pres].mean(0)
        self.sd = F[pres].std(0) + 1e-6
        tP = torch.tensor(P, dtype=torch.float64)
        tF = torch.tensor(F, dtype=torch.float64)
        tM = torch.tensor(pres)
        ty = torch.tensor(y, dtype=torch.long)
        theta = torch.zeros((M, Fd), dtype=torch.float64, requires_grad=True)
        beta = torch.zeros(M, dtype=torch.float64, requires_grad=True)
        logT = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        mu, sd = torch.tensor(self.mu, dtype=torch.float64), torch.tensor(self.sd, dtype=torch.float64)
        opt = torch.optim.LBFGS([theta, beta, logT], lr=0.5, max_iter=300,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            z = (tF - mu) / sd
            logits = (z * theta[None]).sum(2) + beta[None]
            logits = logits.masked_fill(~tM, -1e9)
            w = torch.softmax(logits, 1)
            if self.pool == "linear":
                p = (w[:, :, None] * tP).sum(1)
                lp = torch.log(p.clamp_min(1e-9))
            else:
                lp = (w[:, :, None] * torch.log(tP.clamp_min(1e-6))).sum(1)
            lp = torch.log_softmax(lp / torch.exp(logT), 1)
            loss = torch.nn.functional.nll_loss(lp, ty) + self.l2 * (theta ** 2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        self.theta = theta.detach().numpy()
        self.beta = beta.detach().numpy()
        self.temperature = float(np.exp(logT.item()))
        return self

    def fuse(self, P, F, present):
        w = self._weights(F, present.astype(bool), self.theta, self.beta, np)
        p = self._pool(P, w, np)
        lp = np.log(np.clip(p, EPS, 1)) / self.temperature
        p = np.exp(lp - lp.max(1, keepdims=True))
        return p / p.sum(1, keepdims=True), w

    def importance(self) -> dict:
        """Mean |theta| per evidence feature, in standardised units (what the gate listens to)."""
        imp = np.abs(self.theta).mean(0)
        return {f: round(float(v), 3) for f, v in zip(FEATURES, imp)}


def cross_fit(make, P, F, present, y, folds: int = 5, seed: int = 0):
    """Out-of-fold fused probabilities on the fitting set (for honest calibration)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    out = np.zeros((len(y), P.shape[2]))
    for k in range(folds):
        te = idx[k::folds]
        tr = np.setdiff1d(idx, te)
        g = make().fit(P[tr], F[tr], present[tr], y[tr])
        out[te] = g.fuse(P[te], F[te], present[te])[0]
    return out
