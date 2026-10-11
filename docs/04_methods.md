# Methods

Formal specification of every quantity the system computes. Each replaces a hand-set
constant from an earlier version with either a derivation or a calibration procedure.

Citations verified 2026-10-11 — see [07_literature.md](07_literature.md).

---

## 0. Notation

| Symbol | Meaning |
|---|---|
| `K` | classes |
| `M` | modalities (clinical image, dermoscopy, metadata, text) |
| `S ⊆ {c,d,z,t}` | evidence currently available |
| `y ∈ {1..K}` | true class; `g(y) ∈ {benign, malignant}` its malignancy group |
| `e_m ∈ R₊^K` | evidence from modality `m`; `α_m = e_m + 1`; `S_m = Σ α_m` |
| `b_m, u_m` | belief masses and ignorance mass, `Σ_k b_{m,k} + u_m = 1` |
| `C` | Dempster–Shafer conflict mass |
| `L(y, ŷ)` | clinical loss; `c_acq(m)`, `c_def` acquisition and deferral costs |
| `λ̂` | LTT-calibrated accept threshold |
| `α, δ` | target risk and its failure probability |

---

## 1. Evidential modality heads

Following **EDL** (Sensoy, Kaplan & Kandemir, NeurIPS 2018), each modality head emits
non-negative evidence through a softplus:

```
e_m = softplus(g_m(z_m)),   α_m = e_m + 1,   S_m = Σ_k α_{m,k}
b_{m,k} = e_{m,k}/S_m,      u_m = K/S_m
```

`u_m` is the per-case, per-modality ignorance. It **replaces** the previous
`R = 0.4·certainty + 0.3·agreement + 0.2·quality + 0.1·stability`, whose four
coefficients were chosen rather than fitted — and which, in the shipped system, was
computed by a randomly-initialised MLP.

**Loss.** Bayes risk of squared error under the Dirichlet plus an annealed KL:

```
L_m = Σ_k [ (y_k − p_k)² + p_k(1−p_k)/(S_m+1) ]  +  η_t · KL( Dir(α̃_m) ‖ Dir(1) )
α̃_m = y + (1−y) ⊙ α_m          (true class stripped before penalising)
η_t = min(1, t/T)
```

The squared-error form is chosen over the digamma form for numerical behaviour; Sensoy
et al. describe their own loss analysis as preliminary, so this is a recorded decision.
**`η` must anneal** — at full strength from epoch 0 it suppresses all evidence and pins
`u` at 1.

**Absent modality** → the **vacuous opinion** `(b=0, u=1)`, i.e. `α = 1`.

Implementation: `src/evidence/evidential.py` · tests: `tests/test_evidential.py`.

---

## 2. Dempster–Shafer combination

Following **TMC** (Zongbo Han, Changqing Zhang, Huazhu Fu & Joey Tianyi Zhou, ICLR 2021):

```
C   = Σ_{i≠j} b_{1,i} b_{2,j}                              (conflict mass)
b_k = ( b_{1,k} b_{2,k} + b_{1,k} u_2 + b_{2,k} u_1 ) / (1−C)
u   = ( u_1 u_2 ) / (1−C)
```

Commutative and associative, so `M` modalities fold in any order.

### The three properties that justify the choice

**(i) Vacuous is the identity.** `b ⊕ (0, u=1) = b`. A missing modality changes nothing
— handled by the algebra, not by masking and renormalising. Renormalisation implicitly
asserts the absent modality would have agreed with the others in proportion to their
weights; the identity element asserts it says nothing, which is correct.

**(ii) Conflict is intrinsic and separate from ignorance.** `C` is the mass on mutually
exclusive class pairs. Compare with the JS-divergence consistency score it replaces:

| situation | JS between posteriors | DS |
|---|---|---|
| both confident, different classes | large | `C` high, `u` **low** |
| both near-uniform (nobody knows) | small → reads as *consistent* | `C` low, `u` **high** |

A single scalar cannot express both. The policy needs the distinction because the two
demand different responses: **conflict → human review**, **ignorance → acquire**.

**(iii) Uncertainty composes.** Agreeing confident modalities reduce `u`; conflict
inflates it by `1/(1−C)`.

### Pignistic probability

```
p(k) = b_k + u/K
```

Deliberately not a softmax: a modality that found no evidence contributes a uniform
distribution rather than an arbitrary confident one.

---

## 3. Conformal prediction sets

APS / RAPS sets with marginal coverage `≥ 1−α_conf`, calibrated split-conformally.

```
qhat = Quantile_{⌈(n+1)(1−α)⌉/n} ( scores of true labels on calibration )
set(x) = { k : score_k(x) ≤ qhat }
```

**The rule the policy uses:**

> If the set contains at least one benign **and** at least one malignant class, PROCEED
> is not permitted. An **empty** set also blocks PROCEED — no label was conformal at this
> `α`, which is a reason to stop rather than to guess.

**Class-conditional (Mondrian)** calibration fits `qhat_k` per class, so coverage holds
within each class instead of only on average. A stratum needs roughly `n_k ≳ 1/α`
points; `ConformalCalibrator.fit` records shortfalls in `underpowered_classes` instead of
emitting a meaningless bound.

Implementation: `src/calibration/conformal.py`. Verified: empirical coverage 0.8944
against a 0.90 target on 464 test cases.

---

## 4. Cost model and the three-action rule

### Cost matrix — declared, not implied

| Truth \ Prediction | correct | wrong, same group | malignant → benign | benign → malignant |
|---|---|---|---|---|
| loss | 0 | `ℓ_sub` (1) | **`ℓ_miss` (10)** | `ℓ_fa` (2) |

Plus `c_acq(dermoscopy) = 0.5`, `c_def = 1.5`.

Writing these down is the point: a reader can disagree with them. The 4-level ladder's
thresholds (0.80 / 0.50 / 0.30) implicitly assumed **all errors cost the same**, which is
false and was never stated. A sensitivity sweep over `ℓ_miss ∈ {5, 10, 20}` is
**mandatory** because conclusions depend on it.

### Expected losses

```
R_proceed(S)   = min_ŷ  Σ_y p_S(y) L(y, ŷ)
R_acquire(S,m) = Ê[ R_proceed(S ∪ {m}) ] + c_acq(m)
               = R_proceed(S) − h_m(x_S) + c_acq(m)
R_defer        = c_def

a*(S) = argmin{ R_proceed, min_m R_acquire, R_defer }
```

subject to `≤ T_max` acquisition rounds, only genuinely available modalities, and the
gates in §5. This is the multiclass generalisation of **Chow's rule (1970)**; the
symmetric 0–1 case collapses to a single confidence threshold, i.e. the old ladder is the
special case that assumes symmetric costs.

### Acquisition-value estimator `h_m`

1. **Out-of-fold** (5-fold) predictions for `p_{S}` and `p_{S∪{m}}` on training lesions
   where both modalities exist — in-fold targets would be optimistic.
2. Target `t_i = R_proceed(S)_i − R_proceed(S∪{m})_i`, the realised loss reduction.
3. Fit gradient-boosted trees on features available **before** acquisition: posterior,
   entropy, `u_m`, `C`, quality scores, OOD score, metadata.

Fallbacks: entropy threshold (HOT-style, B5) and ensemble mutual information (B6).

---

## 5. Risk-controlled thresholds (Learn then Test)

**Controlled risk** — the clinically meaningful quantity:

```
R(λ) = P( g(y)=malignant ∧ g(ŷ)=benign | a*=PROCEED )
```

the malignancy miss rate among proceeded cases, bounded in {0,1}.

**Why LTT and not conformal risk control.** CRC requires the risk to be monotone in `λ`
and `λ` one-dimensional. `R(λ)` has a denominator that moves with `λ`, so it is **not
monotone**. The LTT paper exists precisely to drop that requirement by recasting
threshold choice as multiple hypothesis testing. Using CRC here would be invalid.

**Procedure.** For a grid `λ₁ > … > λ_J` (conservative → permissive):

1. empirical risk `R̂(λ_j)` on the calibration split;
2. a valid p-value for `H_j : R(λ_j) > α` via the **Hoeffding–Bentkus** bound:

```
p_H = exp(−n · h₁(R̂ ∧ α, α)),   h₁(a,b) = a log(a/b) + (1−a) log((1−a)/(1−b))
p_B = e · P( Bin(n, α) ≤ ⌈n R̂⌉ )
p_HB = min(p_H, p_B)
```

3. **fixed-sequence testing**: walk the grid while `p_j ≤ δ`, stop at the first
   non-rejection, take the most permissive rejected `λ`. Fixed-sequence needs no
   multiplicity correction, which keeps `λ̂` from being needlessly conservative.

Result: `P( R(λ̂) ≤ α ) ≥ 1 − δ`.

**Failure is reported, not papered over.** If no `λ` can be certified,
`LTTResult.controlled` is `False` with a note. Observed in practice: with `n=432`
calibration cases, `α=0.05`, `δ=0.10` and a 9.95% miss base rate, nothing was
certifiable — and the module said so rather than returning a best guess.

**Group-conditional** calibration (`calibrate_per_group`) fits `λ̂` per skin-tone group.
Groups below `⌈1/α⌉` points are listed in `underpowered_groups` and given **no**
threshold; claiming a per-group guarantee from a handful of points is exactly the
overclaim this project is avoiding.

**Validity is tested.** `simulate_coverage` draws synthetic exchangeable data and checks
the violation rate `≤ δ` across repeated calibration draws. That test is what would catch
an invalid p-value or an off-by-one in the walk.

Implementation: `src/calibration/ltt.py` · tests: `tests/test_calibration.py`.

---

## 6. Safety gates

Three gates on PROCEED, each able only to **increase** caution:

| Gate | Condition | Deferral reason | Clinical response |
|---|---|---|---|
| Conflict | `C > C_calibrated` (validation p90) | `CONFLICT` | human review — more evidence cannot resolve a contradiction |
| Ambiguity | conformal set spans benign ∧ malignant, or is empty | `AMBIGUOUS_SET` | acquire, else defer |
| Risk | `confidence < λ̂` | `RISK_BOUND` | acquire, else defer |

Conflict **short-circuits** ACQUIRE: collecting more evidence does not resolve a
contradiction between existing sources.

This is the substantive improvement over the 4-level ladder, which produced four levels
with overlapping meanings and could not say *why* it was unsure.

---

## 7. Metrics

### AUGRC — primary

Following **Traub et al., NeurIPS 2024** (spotlight):

```
selective risk     SR(τ) = Σ ℓ_i·1[g_i ≥ τ] / Σ 1[g_i ≥ τ]      (÷ accepted)
generalized risk   GR(τ) = (1/N) Σ ℓ_i·1[g_i ≥ τ]               (÷ N)
AUGRC = ∫₀¹ P(err, g ≥ τ) d P(g ≥ τ)
```

AUGRC ∈ **[0, ½]**, lower better, interpretable as the *average risk of undetected
failures*. AURC is reported for comparability only: it is built on selective risk, which
the authors show "is not suitable for aggregation over rejection thresholds" because it
over-weights high-confidence failures, and it changed rankings on 5 of 6 benchmark
datasets.

**Implementation note.** The integral is discretised by the **trapezoidal rule with the
origin included**. A right-endpoint Riemann sum overshoots by `1/(2n)` and breaks the
documented upper bound — it returned 0.5025 at `n=200` for an all-wrong predictor. Pinned
by `test_augrc_respects_its_upper_bound_exactly`.

### Acquisition efficiency

```
AE(r) = ( G_policy(r) − G_random(r) ) / ( G_oracle(r) − G_random(r) )
```

Oracle acquires exactly the `rN` cases with largest realised benefit; random averages
over draws. `AE = 1` oracle-level, `0` no better than random, **negative worse than
random** — reported, not clipped.

### Also required

- **Coverage at guaranteed risk** — the primary pre-registered endpoint
- **Realised `R(λ̂)` vs nominal `α`**, and the **violation** under each stress family
- Per-class precision/recall/F1 — always, never macro alone
- **Adaptive (equal-mass) ECE** and class-wise ECE. Equal-width bins are a poor choice
  once a policy abstains: confidences bunch near the threshold
- **Brier with its reliability/resolution/uncertainty decomposition** — an
  evidence-management claim is specifically a claim about *reliability*
- **NIR** with the magnitude of the increase, not just the rate
- **Net benefit** (Vickers & Elkin, 2006): `NB = TP/N − (FP/N)·p_t/(1−p_t)`

Coverage and accuracy are always reported **as a pair**; neither alone is interpretable
for a system that abstains.

Implementation: `src/eval/selective.py` · tests: `tests/test_selective.py`.

---

## 8. Algorithm

```
Input: case x, available modalities A(x)
Output: PROCEED(ŷ) | ACQUIRE(m) | DEFER(reason)

 1  for m in modalities:
 2      op_m = EvidentialHead_m(encode_m(x_m))   if m in A(x)
 3             vacuous(K)                        otherwise
 4  (b, u, C) = fold_DS(op_1 .. op_M)            # order-independent
 5  p = b + u/K                                   # pignistic
 6
 7  set = conformal(p)                            # calibrated, class-conditional
 8  if C > C_cal:                 return DEFER(CONFLICT)
 9  if ambiguous(set):            gate_proceed = False
10  if (1−u) < λ̂:                gate_proceed = False
11
12  R_proceed = min_ŷ Σ_y p(y) L(y,ŷ)
13  R_acquire = min_m [ R_proceed − h_m(x) + c_acq(m) ]
14  if not gate_proceed:
15      return ACQUIRE(argmin) if R_acquire < c_def else DEFER(reason)
16  return argmin{ PROCEED(ŷ*), ACQUIRE, DEFER }
```

---

## 9. What each piece replaced

| Was | Now | Gain |
|---|---|---|
| `Q` from 5 hand-weighted image stats | `u_m` from trained Dirichlet evidence | fitted, not invented |
| `R = 0.4c + 0.3a + 0.2q + 0.1s` | same `u_m` | 4 free coefficients removed |
| `U` (needed labels; constant 0.5 live) | subsumed into the cost/benefit term | removes a quantity unusable at inference |
| mask + renormalise | vacuous opinion ⊕ identity | correct marginalisation |
| `1 − mean JS`, threshold 0.3 | DS conflict `C`, calibrated at p90 | separates conflict from ignorance |
| `WeightLogitNetwork` (untrained MLP) | closed-form DS fold | no untrained network in the safety path |
| 4-level ladder, 0.80/0.50/0.30 | expected loss + LTT `λ̂` + 3 gates | distribution-free risk bound |
| symmetric thresholds | explicit cost matrix | asymmetric clinical costs made reviewable |
| accuracy | AUGRC + coverage@risk + violation | interpretable under abstention |
| AURC (earlier recommendation) | AUGRC | AURC is unsuitable for threshold aggregation |
| 3-seed deep ensemble | single-pass EDL | 3× compute saved |
