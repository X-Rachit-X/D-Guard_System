# Mathematical Foundations

Formal specification of every quantity the system computes. The current
implementation replaces most of these with hand-tuned formulas
([01_system_analysis.md](01_system_analysis.md)); this document gives the derivation
each one should have.

**Citation note.** Attributions are from memory and unverified. Check every one against
the published paper before use in a submission.

---

## 0. Notation

| Symbol | Meaning |
|---|---|
| `K` | number of classes (6 for PAD-UFES-20: BCC, ACK, NEV, SEK, SCC, MEL) |
| `M` | number of modalities (image, metadata, text) |
| `x_m` | observation from modality `m`; `x_m = ∅` if absent |
| `A(x) ⊆ {1..M}` | set of modalities present for case `x` |
| `y ∈ {1..K}` | true class |
| `p_m(y)` | modality `m`'s predictive distribution |
| `q(y)` | fused posterior |
| `w ∈ Δ^{M-1}` | modality weights, `Σ_m w_m = 1` |
| `e_m ∈ R₊^K` | evidence vector from modality `m` |
| `α_m = e_m + 1` | Dirichlet parameters |
| `S_m = Σ_k α_{m,k}` | Dirichlet strength |
| `b_m, u_m` | belief masses and ignorance mass |
| `s(x)` | scalar confidence score used for deferral |
| `λ` | deferral threshold |

---

## 1. Why the current formulation is not well-posed

The deployed fusion computes

```
w = softmax( α·f([Q̄, R̄, Ū, C]) + β·W[ŷ,:] + γ·log(penalty) ) ⊙ mask,   w ← w / Σw
```

Three problems.

1. **No objective.** `w` is not the argmin/argmax of anything, so "convergence" has no
   definition. The code runs three iterations and tests whether `max|Δw| < 0.05`.
2. **`Q`, `R`, `U` are not identified.** `R = 0.4·certainty + 0.3·agreement +
   0.2·quality + 0.1·stability` has four coefficients that were chosen, not fitted.
   Nothing constrains them and nothing validates them.
3. **Masking ≠ marginalisation.** Zeroing a weight and renormalising the remainder
   asserts that the absent modality would have agreed with the others in proportion to
   their existing weights. The correct statement is that it provides *no information*,
   which is a different operation (§2.3).

---

## 2. Evidential modality heads

### 2.1 Dirichlet evidence

Following evidential deep learning (**Sensoy, Kaplan & Kandemir, NeurIPS 2018**), each
modality head emits non-negative evidence through a softplus:

```
e_m = softplus( g_m(z_m) ) ∈ R₊^K ,      α_m = e_m + 1 ,      S_m = Σ_k α_{m,k}
```

Interpreting `Dir(α_m)` as a distribution over class probabilities, subjective logic
(**Jøsang**) gives the opinion

```
b_{m,k} = e_{m,k} / S_m          (belief in class k)
u_m     = K / S_m                (ignorance)
Σ_k b_{m,k} + u_m = 1
```

The expected class probability is `p_m(k) = α_{m,k} / S_m`.

`u_m` is the quantity the current system approximates with its invented reliability
coefficients. Here it is a *consequence* of how much evidence the encoder found: an
unreadable image or a three-word narrative produces small `e_m`, hence `S_m → K` and
`u_m → 1`.

### 2.2 Training loss

Use the Bayes risk of the squared error under the Dirichlet (numerically better behaved
than the digamma form), plus an annealed KL term that drives evidence for *wrong*
classes to zero:

```
L_m(α_m, y) = Σ_k ( (y_k − p_m(k))² + p_m(k)(1 − p_m(k)) / (S_m + 1) )
            + η_t · KL( Dir(α̃_m) ‖ Dir(1) )
```

where `α̃_m = y + (1 − y) ⊙ α_m` removes the true class before penalising, and
`η_t = min(1, t / T)` anneals the regulariser over the first `T` epochs. The two terms in
the first bracket are the bias and variance of the Dirichlet mean — i.e. the loss
penalises being wrong *and* being confidently wrong.

**Class imbalance.** With MEL at 52/2298, weight `L_m` by inverse frequency
`c_k = N / (K·n_k)`, or keep focal loss (**Lin et al., 2017**) on a softmax head and fit
the evidential head as a separate output. Report which was used — they calibrate
differently.

### 2.3 Absent modalities

An absent modality yields the **vacuous opinion**

```
b_m = 0 ,    u_m = 1    (equivalently e_m = 0, α_m = 1, S_m = K)
```

This is the identity element of the combination rule in §3, so absence is handled by the
algebra rather than by masking code. This is the single cleanest improvement over the
current implementation: no mask vector, no renormalisation, no all-missing zero-division
guard, and — unlike renormalisation — it is the correct representation of "no evidence".

---

## 3. Dempster–Shafer combination

### 3.1 The rule

For two opinions (**Dempster 1967; Shafer 1976**; applied to multi-view learning in
**Trusted Multi-View Classification, Han et al., ICLR 2021**):

```
C        = Σ_{i≠j} b_{1,i} · b_{2,j}                    (conflict mass)
b_k      = ( b_{1,k} b_{2,k} + b_{1,k} u_2 + b_{2,k} u_1 ) / (1 − C)
u        = ( u_1 · u_2 ) / (1 − C)
```

Combination is commutative and associative, so `M` modalities combine by folding:
`b^(1..M) = b^(1) ⊕ b^(2) ⊕ … ⊕ b^(M)`, order-independent.

### 3.2 The three properties worth having

1. **Identity.** `b ⊕ (0, u=1) = b`. Absent modalities cost nothing. (§2.3)
2. **Conflict is intrinsic.** `C` is already in the normaliser — it is the mass assigned
   to mutually exclusive class pairs. The current system's separate
   `1 − mean pairwise JS` is an external proxy for a quantity the combination rule
   computes exactly.
3. **Uncertainty composes correctly.** Agreeing confident modalities give small `u`;
   conflicting ones give `u` inflated by `1/(1−C)`.

### 3.3 Conflict, and why JS divergence is the wrong instrument

The deployed system flags conflict when `1 − mean_pairs JS(p_i ‖ p_j) < 0.3`. Jensen–
Shannon between two *predictive distributions* cannot distinguish

- **conflict:** both modalities confident, on different classes — `JS` large, and this
  is genuinely dangerous; and
- **ignorance:** both near-uniform — `JS` small, so the case is flagged *consistent*
  when in fact nothing is known.

The DS decomposition separates these: conflict is `C`, ignorance is `u`. A safety layer
must treat them differently — conflict should escalate to a human, ignorance should
trigger evidence acquisition. The current single `consistency` number cannot express
that distinction, which is why the ladder routes some ignorant cases to `ACCEPT`.

Report both, and calibrate the conflict threshold on validation (§5) rather than fixing
it at 0.3.

---

## 4. Diagnosis-adaptive fusion as alternating minimisation

### 4.1 Objective

Let `W ∈ R₊^{K×M}` be the learned diagnosis→modality prior (the existing `W_prior`).
Define, for fixed `W`, the joint objective over `(q, w)`:

```
J(q, w) = Σ_m w_m · KL( q ‖ p_m )                        [agreement with each modality]
        − β · Σ_{y,m} q(y) · w_m · log W_{y,m}            [diagnosis-adaptive prior]
        + (1/λ) · Σ_m w_m log w_m                         [entropy regulariser on w]
```

subject to `q ∈ Δ^{K-1}`, `w ∈ Δ^{M-1}`, with `w_m = 0` for `m ∉ A(x)`.

### 4.2 The two closed-form steps

**E-step** (minimise over `q`, `w` fixed):

```
q(y) ∝ exp( Σ_m w_m log p_m(y) + β Σ_m w_m log W_{y,m} )
```

A weighted log-opinion pool, tilted by the diagnosis prior. Note that with `β = 0` this
reduces to `q ∝ ∏_m p_m^{w_m}`, the geometric pool the repaired implementation already
uses.

**M-step** (minimise over `w`, `q` fixed):

```
w_m ∝ exp( −λ [ KL(q ‖ p_m) − β Σ_y q(y) log W_{y,m} ] )
```

Since `KL(q‖p_m) = −H(q) − E_q[log p_m]` and `H(q)` is constant across `m`, this is
equivalently `w_m ∝ exp( λ [ E_q[log p_m] + β E_q[log W_{·,m}] ] )` — **weight each
modality by how well it explains the current posterior**, modulated by the prior.

### 4.3 Convergence

Each step is the exact minimiser of `J` in one block, so `J` is non-increasing and
bounded below (`KL ≥ 0`, entropy bounded on the simplex). Therefore the iteration
converges, and the stopping rule becomes principled: stop when `J^(t) − J^(t+1) < ε`.

**One required change from the current code.** The deployed version indexes the prior at
the *hard* MAP label, `π = W[argmax q, :]`. That makes the objective change between
iterations, so nothing is guaranteed and oscillation is possible. The formulation above
uses the **soft** expectation `Σ_y q(y) log W_{y,m}`, which keeps a single objective and
recovers monotonicity. This is a small code change with a real theoretical payoff.

### 4.4 Relationship to the implementation

`β` and `λ` subsume the existing `alpha`/`beta`/`gamma_param` scalars, and the conflict
penalty term drops out — conflict now enters through `C` in §3 rather than as a third
additive logit. The `WeightLogitNetwork` is removed entirely: `w` is computed in closed
form from quantities that already exist, so there is no untrained MLP left in the path.

---

## 5. Risk-controlled selective prediction

This is the component that restores the guarantee the project lost when the 4-level
ladder replaced conformal prediction.

### 5.1 Quantities

Confidence score (any of these; choose on validation):

```
s(x) = 1 − u(x)            (DS ignorance)
s(x) = max_k b_k(x)        (max belief)
s(x) = max_k b_k(x) − u(x) (belief margin against ignorance)
```

Accept when `s(x) ≥ λ`, otherwise defer. Then

```
coverage        cov(λ) = P( s(x) ≥ λ )
selective risk  R(λ)   = E[ ℓ(ŷ, y) · 1{s ≥ λ} ] / E[ 1{s ≥ λ} ]
```

`R(λ)` is the error rate **among cases the system chose to answer** — the clinically
meaningful quantity, and the one the current implementation reports without any control.

### 5.2 Calibration

Two options, in increasing rigour.

**(a) Conformal Risk Control** (**Angelopoulos, Bates, Candès, Jordan, Lei**). For a
risk that is monotone non-increasing in `λ` and bounded by `B`, pick

```
λ̂ = inf{ λ : R̂_n(λ) + B/(n+1) ≤ α }
```

on `n` calibration points; then `E[R(λ̂)] ≤ α`. Simple, but the guarantee is **in
expectation**, and note that selective risk is *not* in general monotone in `λ` because
the denominator moves — so apply CRC to the un-normalised risk, or use (b).

**(b) Learn-then-Test** (**Angelopoulos, Bates et al.**). Treat each `λ` on a finite grid
as a hypothesis `H_λ : R(λ) > α`, compute a valid p-value per `λ` (Hoeffding–Bentkos
bound on the bounded loss), and reject with fixed-sequence testing from the most
conservative `λ` downward. This yields the high-probability statement

```
P( R(λ̂) ≤ α ) ≥ 1 − δ
```

distribution-free, with no monotonicity assumption. **Use this one.**

### 5.3 Class-conditional (Mondrian) calibration, and its honest limit

Calibrating one global `λ` lets the two majority classes (BCC 845, ACK 730) dominate, so
melanoma can carry far higher error than `α` while the pooled number looks fine.
Mondrian conformal prediction (**Vovk**) fixes this by calibrating `λ_k` within each
class stratum.

**The limit must be stated.** A stratum needs roughly `n_k ≳ 1/α` calibration points for
a non-vacuous bound. With `α = 0.1` and a 15% calibration split:

| Class | n total | ~n in calibration split | Class-conditional guarantee at α=0.1 |
|---|---|---|---|
| BCC | 845 | ~127 | attainable |
| ACK | 730 | ~110 | attainable |
| NEV | 244 | ~37 | attainable, wide |
| SEK | 235 | ~35 | attainable, wide |
| SCC | 192 | ~29 | marginal |
| **MEL** | **52** | **~8** | **not attainable** |

So: report class-conditional control for the four to five feasible classes, and for MEL
either (i) group the malignant classes `{BCC, SCC, MEL}` into one stratum, or (ii) report
the per-class interval honestly as uninformative. Claiming a per-class melanoma guarantee
from eight calibration points would be exactly the kind of overclaim this project has
already been burned by.

### 5.4 Evaluation

Report the **risk–coverage curve** and its summary, not a single accuracy:

- `AURC` — area under the risk–coverage curve (lower is better)
- `E-AURC` — excess over the optimal oracle ordering, which removes the dependence on
  base accuracy and makes models with different accuracies comparable
- `coverage @ risk ≤ α` — the headline operating point
- `R(λ̂)` on test, to confirm the calibrated guarantee actually holds

A single accuracy number is not interpretable for a system that abstains. Accuracy and
coverage must always be reported as a pair.

---

## 6. Cost-sensitive deferral

Let `c(ŷ, y)` be the cost of predicting `ŷ` when the truth is `y`, and `c_def` the cost
of deferring to a clinician. Under the fused posterior `q`, the Bayes action is

```
ŷ* = argmin_ŷ Σ_y q(y) c(ŷ, y) ,      defer iff   min_ŷ Σ_y q(y) c(ŷ, y)  >  c_def
```

This is the multiclass generalisation of **Chow's rule (1970)**. For the symmetric 0–1
case it collapses to `max_k q(k) < 1 − c_def`, which is the single-threshold rule the
current system uses — so the existing design is the special case that assumes every error
costs the same.

It does not. A missed melanoma and a false alarm on a seborrheic keratosis are different
events. Specify `c` explicitly, e.g.

```
c(benign_pred, MEL) = 50 ,   c(malignant_pred, benign) = 1 ,   c_def = 2
```

and derive the thresholds, rather than hand-setting 0.80/0.50/0.30. **The cost matrix
becomes a declared, reviewable parameter of the method** instead of an implicit
assumption buried in three constants — which is itself a presentational win in a
write-up, because a reader can disagree with the numbers.

Sensitivity to `c` should be reported as a curve, since reasonable people will choose
different values.

---

## 7. Calibration diagnostics

```
ECE   = Σ_b (n_b/n) · | acc(b) − conf(b) |         (equal-width bins; report bin count)
```

Prefer **adaptive ECE** (equal-mass bins) — with abstention, confidences cluster near the
threshold and equal-width bins become badly unbalanced.

Brier score decomposes as

```
BS = reliability − resolution + uncertainty
```

Report the decomposition, not just the total: it separates *miscalibration* (reliability)
from *discrimination* (resolution), and an evidence-management claim is specifically a
claim about reliability.

Also report per-modality calibration, since the central thesis is that modalities differ
in trustworthiness. If `u_m` is meaningful, cases with high `u_m` must empirically show
higher error for that modality — that is a falsifiable prediction and should be tested
directly (reliability diagram of error rate vs `u_m`).

---

## 8. Algorithm summary

```
Input: case x, present modalities A(x)
Output: decision ∈ {predict ŷ, defer}, with calibrated risk

1  for m in 1..M:
2      if m ∈ A(x):  e_m ← softplus(g_m(encoder_m(x_m)));  α_m ← e_m + 1
3      else:         α_m ← 1                                  # vacuous
4      b_m ← e_m/S_m ;  u_m ← K/S_m
5
6  (b, u, C) ← fold_DS(b_1..b_M, u_1..u_M)                     # §3, order-independent
7
8  q ← b + u/K                                                 # pignistic init
9  repeat                                                       # §4, monotone in J
10     w_m ∝ exp( λ[ E_q log p_m + β E_q log W_{·,m} ] )        # M-step
11     q(y) ∝ exp( Σ_m w_m log p_m(y) + β Σ_m w_m log W_{y,m} ) # E-step
12 until ΔJ < ε
13
14 s ← 1 − u                                                    # §5.1
15 if C > C_calibrated:            return defer(reason=CONFLICT)   # §3.3
16 if s < λ̂_{k(q)}:               return defer(reason=IGNORANCE)  # §5.2-5.3
17 ŷ ← argmin_ŷ Σ_y q(y) c(ŷ,y)                                 # §6
18 if min cost > c_def:            return defer(reason=COST)
19 return predict ŷ
```

Lines 15–18 give **three distinct deferral reasons** where the current ladder gives four
levels with overlapping meanings. Conflict, ignorance and cost are different failure
modes requiring different clinical responses (human review, acquire more evidence,
escalate by risk), and separating them is both better engineering and a cleaner story.

---

## 9. What each piece replaces

| Current | Replacement | Gain |
|---|---|---|
| `Q` from 5 hand-weighted image stats | `u_m` from learned evidence | Fitted, not invented |
| `R = 0.4·c + 0.3·a + 0.2·q + 0.1·s` | `u_m` from the same Dirichlet | Four free coefficients removed |
| `U` (needs labels; constant 0.5 live) | Not needed; subsumed by `w_m` | Removes a quantity that was never usable at inference |
| mask + renormalise | vacuous opinion ⊕ identity | Correct marginalisation |
| `1 − mean JS`, threshold 0.3 | DS conflict mass `C`, calibrated | Separates conflict from ignorance |
| `WeightLogitNetwork` (untrained MLP) | closed-form `w` | No untrained network in the safety path |
| 3 iterations, `max|Δw| < 0.05` | monotone `J`, stop on `ΔJ < ε` | Convergence guarantee |
| 4-level ladder, thresholds 0.80/0.50/0.30 | LTT-calibrated `λ̂`, class-conditional | Distribution-free risk guarantee |
| symmetric confidence thresholds | explicit cost matrix + Chow rule | Asymmetric clinical costs made explicit |
| accuracy | risk–coverage curve, AURC, E-AURC | Interpretable for an abstaining system |

Every row removes a hand-set constant and replaces it with either a derivation or a
calibration procedure. That is the substance of the contribution.
