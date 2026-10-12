"""
src/fusion/methods.py - Every way of combining the specialists, behind one API.

Used by scripts/run_team.py (fit + compare + pick the winner on validation), by
the app's single-case runtime and by the Comparative-study view, so all three
run exactly the same arithmetic.

    method            what decides the weights                         learned?
    ---------------   ----------------------------------------------   --------
    average           nothing - every specialist counts the same       no
    original_formula  E_m = Q_m R_m U_m (C, S cancel), old paper eq.   no (R is)
    best_head         the single specialist with the highest R_m       no (R is)
    adaptive_em       a log(QR) + b log(prior) + g log(penalty), EM    4 params
    stacking          multinomial logistic regression on all log p_m   yes
    gate_linear       Devil's Advocate v2, mixture pool                yes
    gate_log          Devil's Advocate v2, product-of-experts pool     yes

`fit_context` builds what every method needs from a fitting set: the learned
reliability models (R) and the measured diagnosis prior W_prior[c, m].
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.fusion.adaptive import EPS, AdaptiveFusion, EvidenceManager
from src.fusion.gating import LearnedGate, evidence_features

METHODS = ["average", "original_formula", "best_head", "adaptive_em", "stacking",
           "gate_linear", "gate_log"]

LABELS = {
    "average": "Simple average",
    "original_formula": "Original formula (Q·R·U)",
    "best_head": "Best head per case (highest R)",
    "adaptive_em": "Adaptive EM fusion (v1)",
    "stacking": "Stacking (logistic regression)",
    "gate_linear": "Devil's Advocate v2 – learned gate (mixture)",
    "gate_log": "Devil's Advocate v2 – learned gate (product)",
}


@dataclass
class FusionContext:
    """Fitted pieces shared by all methods (one per stage)."""
    names: List[str]
    K: int
    manager: EvidenceManager
    prior: AdaptiveFusion                  # holds W_prior; also the fitted v1 fusion
    models: Dict[str, object] = field(default_factory=dict)

    def evidence(self, P, Q, present, ood=None):
        R = self.manager.reliability(P, Q, present)
        F = evidence_features(P, Q, R, present, self.prior.W_prior, ood)
        return R, F


def fit_context(names, K, P, Q, present, y, fit_em=True) -> FusionContext:
    mgr = EvidenceManager(names=list(names)).fit(P, Q, present, y)
    em = AdaptiveFusion(names=list(names)).fit_prior(P, present, y, K)
    if fit_em:
        R = mgr.reliability(P, Q, present)
        em.fit_params(P, Q * R, present, y, K)
    return FusionContext(names=list(names), K=K, manager=mgr, prior=em)


def _stack_X(P, present):
    return np.where(present[:, :, None], np.log(np.clip(P, 1e-6, 1)), 0.0).reshape(len(P), -1)


def fit_method(ctx: FusionContext, method: str, P, Q, present, y, ood=None):
    """Fit `method` on (P, Q, present, y); stores the fitted object in ctx.models."""
    R, F = ctx.evidence(P, Q, present, ood)
    if method == "stacking":
        from sklearn.linear_model import LogisticRegression
        ctx.models[method] = LogisticRegression(C=0.3, max_iter=3000).fit(_stack_X(P, present), y)
    elif method in ("gate_linear", "gate_log"):
        ctx.models[method] = LearnedGate(names=ctx.names, pool=method.split("_")[1]).fit(
            P, F, present, y)
    return ctx


def fuse_method(ctx: FusionContext, method: str, P, Q, present, ood=None):
    """-> fused probabilities (N, K) and per-specialist weights (N, M)."""
    R, F = ctx.evidence(P, Q, present, ood)
    N, M, K = P.shape
    pres = present.astype(bool)
    if method == "average":
        w = pres / pres.sum(1, keepdims=True)
    elif method == "original_formula":
        w0 = np.where(pres, R + EPS, 0.0)
        y0 = ((w0 / w0.sum(1, keepdims=True))[:, :, None] * P).sum(1).argmax(1)
        E = np.where(pres, Q * R * ctx.prior.W_prior[y0], 0.0) + EPS * pres
        w = E / E.sum(1, keepdims=True)
    elif method == "best_head":
        w = np.zeros((N, M))
        w[np.arange(N), np.where(pres, R, -1).argmax(1)] = 1.0
    elif method == "adaptive_em":
        return ctx.prior.fuse(P, Q * R, pres)
    elif method == "stacking":
        p = ctx.models[method].predict_proba(_stack_X(P, pres))
        full = np.zeros((N, K))
        full[:, ctx.models[method].classes_] = p
        return full, pres / pres.sum(1, keepdims=True)
    elif method in ("gate_linear", "gate_log"):
        return ctx.models[method].fuse(P, F, pres)
    else:
        raise ValueError(method)
    return (w[:, :, None] * P).sum(1), w


def oracle_best_head(P, present, y):
    """Upper bound: for every case, the specialist that gave the TRUE class most mass.
    Uses the label - a ceiling for head selection, never a usable method."""
    sc = np.where(present, P[np.arange(len(y)), :, y], -1)
    return P[np.arange(len(y)), sc.argmax(1)]
