"""
src/fusion/adaptive.py - The original DERMA-GUARD fusion, rebuilt on trained models.

Two pieces from the original paper ("Agentic Multi-Model Consensus with Dynamic
Diagnosis-Adaptive Evidence Fusion"), now fitted on data instead of hand-set:

1. EVIDENCE MANAGER (the "Devil's Advocate"). For every specialist m and case x:

       Q_m   quality       image statistics vs. the training-photo distribution;
                           1 - missing fraction for metadata; neighbour similarity for kNN
       U_m   uncertainty   normalised entropy of the specialist's distribution
       R_m   reliability   P(specialist m is right | Q, U, confidence, agreement) -
                           a LEARNED model (logistic regression on validation lesions),
                           replacing the original's untrained MLP
       C     consistency   1 - mean pairwise Jensen-Shannon divergence between specialists
       M     missingness   share of modalities absent
       OOD   Mahalanobis distance of the case embedding to the nearest class centroid
                           (Ledoit-Wolf shared covariance), threshold = validation p99

2. ITERATIVE DIAGNOSIS-ADAPTIVE FUSION (EM-style, max 3 iterations):

       w(0)  = softmax_m( a * log(Q_m R_m) )
       p     = sum_m w_m p_m ;  y = argmax p
       repeat:
         pi_m  = W_prior[y, m]            how reliable m is when the diagnosis is y
                                          (MEASURED on validation, Laplace-smoothed)
         pen_m = 1 if argmax p_m == y else rho        (conflict penalty)
         w     = softmax_m( a log(Q_m R_m) + b log pi_m + g log pen_m )
         p     = sum_m w_m p_m
       until y stable and max |dw| < 0.05

   a, b, g, rho are chosen on validation by grid search; the test split is never
   used for any choice.

Everything is vectorised over cases, so the experiment and the single-case app
run exactly the same arithmetic.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

EPS = 1e-9


# ---------------------------------------------------------------- helpers
def norm_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0)
    return -(p * np.log(p)).sum(-1) / np.log(p.shape[-1])


def _js(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p, q = np.clip(p, EPS, 1), np.clip(q, EPS, 1)
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * np.log(a / b)).sum(-1)  # noqa: E731
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def js_consistency(P: np.ndarray, present: np.ndarray) -> np.ndarray:
    """(N, M, K) distributions, (N, M) presence -> (N,) consistency in [0, 1]."""
    N, M, _ = P.shape
    tot = np.zeros(N)
    cnt = np.zeros(N)
    for i, j in itertools.combinations(range(M), 2):
        both = present[:, i] & present[:, j]
        tot += np.where(both, _js(P[:, i], P[:, j]) / np.log(2), 0.0)
        cnt += both
    return np.where(cnt > 0, 1.0 - tot / np.maximum(cnt, 1), 1.0)


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), EPS)


def knn_probs(query: np.ndarray, ref: np.ndarray, ref_y: np.ndarray, K: int,
              k: int = 15, alpha: float = 1.0):
    """Similarity-weighted vote of the k most similar TRAINING lesions.

    Returns (N, K) probabilities (Laplace-smoothed) and (N,) mean similarity.
    """
    q, r = l2n(query), l2n(ref)
    sims = q @ r.T
    idx = np.argpartition(-sims, k, axis=1)[:, :k]
    top = np.take_along_axis(sims, idx, axis=1)
    w = np.exp(5.0 * top)
    P = np.full((len(q), K), alpha, dtype=np.float64)
    for j in range(k):
        np.add.at(P, (np.arange(len(q)), ref_y[idx[:, j]]), w[:, j])
    return P / P.sum(1, keepdims=True), top.mean(1)


# --------------------------------------------------------------- OOD
@dataclass
class OODDetector:
    """Mahalanobis distance to the nearest class centroid, shared covariance."""
    means: Optional[np.ndarray] = None
    precision: Optional[np.ndarray] = None
    threshold: float = np.inf

    def fit(self, Z: np.ndarray, y: np.ndarray, K: int) -> "OODDetector":
        from sklearn.covariance import LedoitWolf
        Z = l2n(Z)
        self.means = np.stack([Z[y == c].mean(0) if (y == c).any() else Z.mean(0)
                               for c in range(K)])
        resid = Z - self.means[y]
        self.precision = LedoitWolf().fit(resid).precision_
        return self

    def score(self, Z: np.ndarray) -> np.ndarray:
        Z = l2n(Z)
        d = Z[:, None, :] - self.means[None, :, :]
        m = np.einsum("nkd,de,nke->nk", d, self.precision, d)
        return m.min(1)

    def calibrate(self, Z_val: np.ndarray, q: float = 99.0) -> "OODDetector":
        self.threshold = float(np.percentile(self.score(Z_val), q))
        return self

    def flag(self, Z: np.ndarray) -> np.ndarray:
        return self.score(Z) > self.threshold


# ------------------------------------------------------- S6 embedding classifier
@dataclass
class EmbeddingClassifier:
    """S6: class-balanced logistic regression on the trained network's embeddings.

    The similar-case vote (S5) looks at the 15 nearest training lesions; this
    specialist draws one linear boundary through ALL training embeddings, with
    class weights so rare diagnoses are not drowned out. C is chosen on validation.
    """
    keys: List[str] = field(default_factory=list)
    C: float = 0.01
    mu: Optional[np.ndarray] = None
    sd: Optional[np.ndarray] = None
    model: object = None
    K: int = 11

    def fit(self, Z: np.ndarray, y: np.ndarray) -> "EmbeddingClassifier":
        from sklearn.linear_model import LogisticRegression
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-6
        self.model = LogisticRegression(C=self.C, max_iter=3000,
                                        class_weight="balanced").fit((Z - self.mu) / self.sd, y)
        return self

    def predict(self, Z: np.ndarray) -> np.ndarray:
        p = self.model.predict_proba((Z - self.mu) / self.sd)
        out = np.full((len(Z), self.K), 1e-4)
        out[:, self.model.classes_] = p
        return out / out.sum(1, keepdims=True)


# ----------------------------------------------------------- evidence manager
def _features(P_m, Q_m, P_all, present, m):
    """Per-specialist features for the reliability model."""
    conf = P_m.max(1)
    H = norm_entropy(P_m)
    top = P_m.argmax(1)
    others = [j for j in range(P_all.shape[1]) if j != m]
    agree = np.zeros(len(P_m))
    n = np.zeros(len(P_m))
    for j in others:
        ok = present[:, j]
        agree += np.where(ok, P_all[:, j].argmax(1) == top, 0)
        n += ok
    agree = np.where(n > 0, agree / np.maximum(n, 1), 0.5)
    return np.column_stack([Q_m, H, conf, agree])


@dataclass
class EvidenceManager:
    """Learned reliability R_m per specialist (logistic regression on validation)."""
    names: List[str] = field(default_factory=list)
    models: Dict[str, object] = field(default_factory=dict)

    def fit(self, P, Q, present, y):
        from sklearn.linear_model import LogisticRegression
        for m, name in enumerate(self.names):
            ok = present[:, m]
            X = _features(P[:, m], Q[:, m], P, present, m)[ok]
            t = (P[ok, m].argmax(1) == y[ok]).astype(int)
            if len(np.unique(t)) < 2:
                self.models[name] = float(t.mean())
                continue
            self.models[name] = LogisticRegression(C=1.0, max_iter=1000).fit(X, t)
        return self

    def reliability(self, P, Q, present) -> np.ndarray:
        N, M, _ = P.shape
        R = np.zeros((N, M))
        for m, name in enumerate(self.names):
            mdl = self.models[name]
            if isinstance(mdl, float):
                R[:, m] = mdl
            else:
                R[:, m] = mdl.predict_proba(_features(P[:, m], Q[:, m], P, present, m))[:, 1]
        return np.where(present, R, 0.0)

    def profile(self, P, Q, present) -> Dict[str, np.ndarray]:
        """The five-dimensional evidence profile of the original design."""
        R = self.reliability(P, Q, present)
        U = np.where(present, norm_entropy(P), 1.0)
        return {"Q": Q, "R": R, "U": U, "C": js_consistency(P, present),
                "M": 1.0 - present.mean(1)}


# ------------------------------------------------------------- adaptive fusion
@dataclass
class AdaptiveFusion:
    names: List[str] = field(default_factory=list)
    W_prior: Optional[np.ndarray] = None     # (K, M): P(m right | true class c)
    a: float = 1.0
    b: float = 1.0
    g: float = 1.0
    rho: float = 0.55
    max_iter: int = 3
    tol: float = 0.05
    temperature: float = 1.0

    def fit_prior(self, P, present, y, K, smooth: float = 1.0):
        M = P.shape[1]
        W = np.zeros((K, M))
        for m in range(M):
            ok = present[:, m]
            hit = P[ok, m].argmax(1) == y[ok]
            for c in range(K):
                sel = y[ok] == c
                W[c, m] = (hit[sel].sum() + smooth * 0.5) / (sel.sum() + smooth)
        self.W_prior = W
        return self

    @staticmethod
    def _softmax(logits, present):
        logits = np.where(present, logits, -np.inf)
        logits = logits - logits.max(1, keepdims=True)
        e = np.where(present, np.exp(logits), 0.0)
        return e / np.maximum(e.sum(1, keepdims=True), EPS)

    def fuse(self, P, QR, present, return_trace=False):
        """P (N,M,K), QR (N,M) = Q*R, present (N,M) -> fused (N,K), weights (N,M)."""
        base = self.a * np.log(np.clip(QR, 1e-4, 1))
        w = self._softmax(base, present)
        p = (w[:, :, None] * P).sum(1)
        y = p.argmax(1)
        iters = np.zeros(len(P), dtype=int)
        for _ in range(self.max_iter):
            pi = self.W_prior[y]                                    # (N, M)
            pen = np.where(P.argmax(2) == y[:, None], 1.0, self.rho)
            logits = base + self.b * np.log(np.clip(pi, 1e-4, 1)) + self.g * np.log(pen)
            w_new = self._softmax(logits, present)
            p_new = (w_new[:, :, None] * P).sum(1)
            y_new = p_new.argmax(1)
            moving = (y_new != y) | (np.abs(w_new - w).max(1) >= self.tol)
            iters += moving
            w, p, y = w_new, p_new, y_new
            if not moving.any():
                break
        if self.temperature != 1.0:
            lp = np.log(np.clip(p, EPS, 1)) / self.temperature
            p = np.exp(lp - lp.max(1, keepdims=True))
            p = p / p.sum(1, keepdims=True)
        return (p, w, iters) if return_trace else (p, w)

    def fit_params(self, P, QR, present, y, K, metric="macro_f1"):
        """Grid search a, b, g, rho on validation; then temperature by NLL."""
        from sklearn.metrics import f1_score
        best, best_s = None, -1
        for a, b, g, rho in itertools.product((0.5, 1, 2, 4), (0, 0.5, 1, 2, 4),
                                              (0, 0.5, 1), (0.3, 0.55, 0.8)):
            self.a, self.b, self.g, self.rho, self.temperature = a, b, g, rho, 1.0
            p, _ = self.fuse(P, QR, present)
            s = f1_score(y, p.argmax(1), labels=list(range(K)), average="macro",
                         zero_division=0)
            if s > best_s:
                best, best_s = (a, b, g, rho), s
        self.a, self.b, self.g, self.rho = best
        # temperature scaling (Guo et al. 2017) on the fused distribution
        p, _ = self.fuse(P, QR, present)
        lp = np.log(np.clip(p, EPS, 1))
        nll = lambda T: -np.log(np.clip(  # noqa: E731
            np.exp(lp / T - np.log(np.exp(lp / T).sum(1, keepdims=True)))[np.arange(len(y)), y],
            EPS, 1)).mean()
        Ts = np.linspace(0.3, 2.0, 35)
        self.temperature = float(Ts[np.argmin([nll(T) for T in Ts])])
        return best_s


# ------------------------------------------------------------------ bundle
@dataclass
class TeamBundle:
    """Everything the app needs to reproduce the team fusion for one case."""
    names: List[str]
    classes: List[str]
    manager_A: EvidenceManager        # reliability models, stage A specialists
    manager_B: EvidenceManager        # reliability models, stage B specialists
    names_A: List[str]
    fusion_A: AdaptiveFusion          # stage A: no dermoscopy specialist
    fusion_B: AdaptiveFusion          # stage B: with dermoscopy
    ood: OODDetector
    knn_ref_A: np.ndarray             # train embeddings for the kNN specialist (stage A)
    knn_ref_B: np.ndarray             # (stage B, includes dermoscopy)
    knn_y: np.ndarray
    knn_k: int
    q_ref_clin: np.ndarray            # (2, F) 5th/95th percentiles of training photos
    q_ref_derm: np.ndarray
    confidence_calibrator: object = None   # isotonic: max fused p -> P(correct)
    report: Dict = field(default_factory=dict)
    ctx_A: object = None              # src.fusion.methods.FusionContext per stage
    ctx_B: object = None
    method_A: str = "adaptive_em"     # combination rule picked on validation
    method_B: str = "adaptive_em"
    clf_A: object = None              # S6 EmbeddingClassifier per stage (None = not in team)
    clf_B: object = None


def quality_score(q: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """(N, F) image statistics -> (N,) share of statistics inside the 5-95% range."""
    lo, hi = ref
    inside = (q >= lo) & (q <= hi)
    return inside.mean(1)
