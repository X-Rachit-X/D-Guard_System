"""
src/evidence/evidential.py - Evidential opinions and Dempster-Shafer combination.

Replaces the hand-tuned Q/R/U formulas and the mask-and-renormalise path with a
single operator that has three properties the decision layer needs:

  1. ABSENCE IS FREE.  An absent modality is the vacuous opinion (b=0, u=1),
     which is the identity element of DS combination.  No mask vector, no
     renormalisation, no all-missing zero-division guard.
  2. CONFLICT IS SEPARATE FROM IGNORANCE.  The DS conflict mass `C` measures
     belief assigned to mutually exclusive classes; `u` measures how little was
     known.  Jensen-Shannon divergence between predictive distributions cannot
     tell those apart - two confident modalities disagreeing and two near-uniform
     ones can give similar JS - and the three-action policy must distinguish
     them, because conflict means "defer to a human" while ignorance means
     "acquire more evidence".
  3. UNCERTAINTY IS EPISTEMIC AND CHEAP.  One forward pass per modality, no
     ensemble.  That matters when compute is the binding constraint.

References (verified 2026-10-11, see docs/07_literature_survey.md):
  Sensoy, Kaplan & Kandemir, "Evidential Deep Learning to Quantify
    Classification Uncertainty", NeurIPS 2018 (arXiv:1806.01768).
  Zongbo Han, Changqing Zhang, Huazhu Fu & Joey Tianyi Zhou, "Trusted
    Multi-View Classification", ICLR 2021 (arXiv:2102.02051).
  Dempster (1967); Shafer (1976); Jøsang (subjective logic).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


# ---------------------------------------------------------------------------
# Opinion
# ---------------------------------------------------------------------------

@dataclass
class Opinion:
    """A subjective-logic opinion over `K` classes.

    belief:   (B, K) non-negative belief masses
    u:        (B, 1) ignorance mass
    conflict: (B, 1) accumulated conflict mass from combination (0 for a leaf)

    Invariant: belief.sum(-1) + u == 1 for every row.
    """

    belief: torch.Tensor
    u: torch.Tensor
    conflict: Optional[torch.Tensor] = None

    @property
    def num_classes(self) -> int:
        return self.belief.shape[-1]

    @property
    def batch_size(self) -> int:
        return self.belief.shape[0]

    @property
    def strength(self) -> torch.Tensor:
        """Dirichlet strength S = K / u."""
        return self.belief.new_full((1,), self.num_classes) / self.u.clamp(min=EPS)

    @property
    def alpha(self) -> torch.Tensor:
        """Dirichlet parameters, alpha = evidence + 1."""
        return self.belief * self.strength + 1.0

    @property
    def probability(self) -> torch.Tensor:
        """Pignistic probability: ignorance spread uniformly over classes.

        This is what the decision layer consumes. It is deliberately NOT a
        softmax: a modality that found no evidence contributes a uniform
        distribution rather than an arbitrary confident one.
        """
        return self.belief + self.u / self.num_classes

    def check(self, tol: float = 1e-4) -> bool:
        total = self.belief.sum(-1, keepdim=True) + self.u
        return bool(torch.allclose(total, torch.ones_like(total), atol=tol))

    def detach(self) -> "Opinion":
        return Opinion(self.belief.detach(), self.u.detach(),
                       None if self.conflict is None else self.conflict.detach())

    def to_dict(self) -> Dict[str, List]:
        out = {"belief": self.belief.tolist(), "u": self.u.squeeze(-1).tolist()}
        if self.conflict is not None:
            out["conflict"] = self.conflict.squeeze(-1).tolist()
        return out


def vacuous(batch_size: int, num_classes: int,
            device=None, dtype=torch.float32) -> Opinion:
    """The opinion of an ABSENT modality: no belief, total ignorance.

    This is the identity element of `ds_combine`: combining any opinion with a
    vacuous one leaves it unchanged. That is what makes missing modalities a
    no-op rather than something to impute or mask.
    """
    return Opinion(
        belief=torch.zeros(batch_size, num_classes, device=device, dtype=dtype),
        u=torch.ones(batch_size, 1, device=device, dtype=dtype),
    )


def opinion_from_evidence(evidence: torch.Tensor) -> Opinion:
    """Non-negative evidence -> Opinion. alpha = e+1, S = sum(alpha), b = e/S, u = K/S."""
    e = evidence.clamp(min=0.0)
    k = e.shape[-1]
    alpha = e + 1.0
    S = alpha.sum(-1, keepdim=True)
    return Opinion(belief=e / S, u=torch.full_like(S, float(k)) / S)


def opinion_from_probs(probs: torch.Tensor, strength: float | torch.Tensor) -> Opinion:
    """Build an opinion from a probability vector plus a chosen total strength.

    Used to lift a plain softmax baseline into the evidential framework for
    ablations ("DS on softmax"), where `strength` controls how much the baseline
    is trusted. strength -> K reproduces the vacuous opinion.
    """
    k = probs.shape[-1]
    S = torch.as_tensor(strength, dtype=probs.dtype, device=probs.device)
    S = S.reshape(-1, 1).expand(probs.shape[0], 1).clamp(min=float(k))
    e = probs * S - 1.0
    return opinion_from_evidence(e.clamp(min=0.0))


# ---------------------------------------------------------------------------
# Dempster-Shafer combination
# ---------------------------------------------------------------------------

def ds_combine(a: Opinion, b: Opinion) -> Opinion:
    """Dempster's rule of combination for two opinions.

        C   = sum_{i != j} b_a[i] * b_b[j]          (conflict mass)
        b_k = (b_a[k] b_b[k] + b_a[k] u_b + b_b[k] u_a) / (1 - C)
        u   = (u_a u_b) / (1 - C)

    Commutative and associative, so `fold_opinions` can apply it in any order.
    Accumulated conflict is carried forward as the max over the fold, so a
    single badly-conflicting pair is not diluted by agreeing ones.
    """
    if a.num_classes != b.num_classes:
        raise ValueError(f"class count mismatch: {a.num_classes} vs {b.num_classes}")

    ba, ua, bb, ub = a.belief, a.u, b.belief, b.u

    # sum_{i != j} ba_i bb_j  =  (sum ba)(sum bb) - sum_k ba_k bb_k
    cross = ba.sum(-1, keepdim=True) * bb.sum(-1, keepdim=True)
    agree = (ba * bb).sum(-1, keepdim=True)
    conflict = (cross - agree).clamp(0.0, 1.0 - 1e-6)
    denom = (1.0 - conflict).clamp(min=EPS)

    belief = (ba * bb + ba * ub + bb * ua) / denom
    u = (ua * ub) / denom

    prior = [c for c in (a.conflict, b.conflict) if c is not None]
    acc = conflict if not prior else torch.maximum(conflict, torch.stack(prior).amax(0))
    return Opinion(belief=belief, u=u, conflict=acc)


def fold_opinions(opinions: Sequence[Opinion]) -> Opinion:
    """Combine a sequence of opinions. Absent modalities should already be vacuous."""
    if not opinions:
        raise ValueError("fold_opinions requires at least one opinion")
    out = opinions[0]
    for nxt in opinions[1:]:
        out = ds_combine(out, nxt)
    if out.conflict is None:
        out = Opinion(out.belief, out.u, torch.zeros_like(out.u))
    return out


def gate_opinion(op: Opinion, present: torch.Tensor) -> Opinion:
    """Replace per-row opinions by the vacuous one wherever `present` is 0.

    `present` is (B,) or (B,1) in {0,1}. This is how a per-case missing modality
    is expressed without branching on Python-level conditionals.
    """
    m = present.reshape(-1, 1).to(op.belief.dtype)
    return Opinion(belief=op.belief * m, u=op.u * m + (1.0 - m),
                   conflict=None if op.conflict is None else op.conflict * m)


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------

class EvidentialHead(nn.Module):
    """Modality embedding -> non-negative Dirichlet evidence -> Opinion.

    softplus (not exp) keeps evidence non-negative without exploding, and
    saturates gracefully so an uninformative input yields e -> 0, hence u -> 1.
    """

    def __init__(self, d_in: int, num_classes: int, hidden: Optional[int] = 128,
                 dropout: float = 0.2, max_evidence: float = 1e4):
        super().__init__()
        self.num_classes = num_classes
        self.max_evidence = max_evidence
        if hidden:
            self.net = nn.Sequential(
                nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden, num_classes),
            )
        else:
            self.net = nn.Linear(d_in, num_classes)

    def forward(self, z: torch.Tensor) -> Opinion:
        evidence = F.softplus(self.net(z)).clamp(max=self.max_evidence)
        return opinion_from_evidence(evidence)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def dirichlet_kl_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """KL( Dir(alpha) || Dir(1) ), per row."""
    k = alpha.shape[-1]
    S = alpha.sum(-1, keepdim=True)
    return (torch.lgamma(S).squeeze(-1)
            - torch.lgamma(alpha).sum(-1)
            - math.lgamma(k)
            + ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(S))).sum(-1))


def edl_loss(opinion: Opinion, target: torch.Tensor, anneal: float = 1.0,
             class_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Bayes risk of squared error under the Dirichlet, plus an annealed KL term.

    The squared-error form is used rather than the digamma/log form because it is
    better behaved numerically; Sensoy et al. describe their own loss analysis as
    preliminary, so this choice is recorded as a decision rather than a given.

    The KL term is applied to `alpha` with the TRUE class stripped out, so it
    drives evidence for wrong classes toward zero without penalising correct
    confidence. `anneal` should ramp 0 -> 1 over the first few epochs; applying
    it at full strength from the start suppresses all evidence and pins u at 1.
    """
    alpha = opinion.alpha
    S = alpha.sum(-1, keepdim=True)
    p = alpha / S
    y = F.one_hot(target, opinion.num_classes).to(p.dtype)

    err = (y - p) ** 2
    var = p * (1.0 - p) / (S + 1.0)
    per_class = err + var
    if class_weight is not None:
        per_class = per_class * class_weight.reshape(1, -1)
    fit = per_class.sum(-1)

    alpha_tilde = y + (1.0 - y) * alpha          # remove the true class
    reg = dirichlet_kl_to_uniform(alpha_tilde)
    return (fit + anneal * reg).mean()


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def conflict_and_ignorance(opinions: Dict[str, Opinion]) -> Dict[str, torch.Tensor]:
    """Fold a set of named opinions and report the quantities the policy needs.

    Returns `probability`, `belief`, `ignorance`, `conflict`, and the per-modality
    ignorance. The last is what makes "which modality was uninformative here?"
    answerable per case - the question the old scalar `consistency` could not
    answer.
    """
    fused = fold_opinions(list(opinions.values()))
    return {
        "probability": fused.probability,
        "belief": fused.belief,
        "ignorance": fused.u,
        "conflict": fused.conflict,
        "per_modality_ignorance": {k: v.u for k, v in opinions.items()},
    }
