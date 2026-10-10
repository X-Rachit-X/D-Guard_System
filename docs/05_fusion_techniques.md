# Multimodal Fusion: Options, Trade-offs, and What This Project Should Use

The current system fuses by taking a weighted geometric mean of three predictive
distributions, with weights from a hand-tuned formula. That is one point in a large
design space, and not an obviously good one. This document lays out the space, states
the selection criteria that follow from *this* dataset, and commits to a choice.

**Citation note.** Attributions from memory, unverified. Check before citing.

---

## 1. Selection criteria for this problem

Fusion method choice should follow from the data, not from novelty. PAD-UFES-20 imposes
four hard constraints:

| Constraint | Consequence |
|---|---|
| **Modalities routinely absent.** Metadata fields are blank; narratives may be 3 words; dermoscopy never exists | The operator must handle `M` varying per case without retraining |
| **2298 cases, 6 classes, MEL = 52** | Parameter-heavy fusion will overfit. Low-capacity operators preferred |
| **Modality quality varies per case** | Fixed weights are wrong; weights must be input-dependent |
| **Modalities genuinely conflict** | Must represent disagreement, not average it away |

Plus one methodological constraint: whatever is chosen must produce a **calibrated
uncertainty**, because the deferral policy ([04 §5](04_mathematical_foundations.md))
consumes it.

Two consequences worth stating up front. First, **tiny data rules out most of the
attention-based literature** — MulT-class models have millions of fusion parameters and
were developed on datasets 10–100× larger. Second, **averaging is the wrong default**:
if the image says NEV at 0.9 and the text says MEL at 0.9, the mean says "0.45/0.45,
moderately confident in neither", which reads as mild uncertainty when the correct
reading is *active contradiction* and a reason to involve a human.

---

## 2. The design space

### 2.1 Early fusion (feature concatenation)

```
z = [z_img ; z_meta ; z_text] → MLP → logits
```

**Pros:** trivial; learns cross-modal interactions implicitly; strong baseline in
practice. **Cons:** fixed input width, so a missing modality must be zero-imputed — and
zero is a *meaningful* value post-normalisation, so the model cannot distinguish "absent"
from "average". No per-modality uncertainty.

**Verdict:** mandatory baseline (B4 in the ladder). Not the method.

### 2.2 Late fusion (logit/probability pooling)

Arithmetic mean, geometric mean (= weighted log-opinion pool), or majority vote.

- **Arithmetic** `Σ w_m p_m`: a mixture. One confident modality can be outvoted; robust
  to a single bad modality.
- **Geometric** `∏ p_m^{w_m}`: a product of experts (**Hinton, 2002**). Any modality
  assigning ≈0 to a class **vetoes** it. Sharper, and the veto is desirable in a safety
  setting — a confident "definitely not melanoma" from a clear image *should* suppress
  that class.

**Verdict:** geometric pooling is what the repaired implementation already does and is a
sound choice. Keep as the fusion baseline. Its weakness is that `w` has to come from
somewhere, which is exactly where the current system is unprincipled.

### 2.3 Gated fusion / Gated Multimodal Unit

```
g = σ(W_g [z_1; z_2]) ,   z = g ⊙ z_1 + (1−g) ⊙ z_2
```

(**Arevalo et al.**) Input-dependent weights, cheap. This is essentially what the trained
specialists D–G already use (`AdaptiveGatedFusion`). **Cons:** the gate is a black box —
it gives no interpretable uncertainty and no handle for a deferral policy, and `g` is not
a probability of anything.

**Verdict:** already present; keep D–G as baselines. Insufficient as the contribution
because it cannot feed a calibrated deferral rule.

### 2.4 MetaBlock / MetaNet — the dermatology-specific baseline

**MetaBlock** (**Pacheco & Krohling, IEEE JBHI 2021**) conditions image feature maps on
metadata, roughly

```
z_img' = tanh( z_img ⊙ f(z_meta) + g(z_meta) )
```

i.e. a FiLM-style affine modulation where metadata gates visual channels.

**This matters more than any other row in this table:** MetaBlock was introduced *by the
authors of PAD-UFES-20, on PAD-UFES-20*. It is the number a reviewer will expect to see
beaten or matched. The v1 project context correctly identified it as the "literature SOTA
baseline"; v4 dropped it. **Reinstate it.**

**Verdict:** required baseline. Not the contribution.

### 2.5 Tensor and low-rank fusion

Tensor Fusion Network (**Zadeh et al., 2017**) takes the outer product
`z_1 ⊗ z_2 ⊗ z_3` to capture all interaction orders; Low-rank Multimodal Fusion
(**Liu et al., 2018**) factorises it to stay tractable.

**Cons:** parameter count scales badly; designed for sentiment data with ~20k samples.
With 2298 cases and 52 melanomas this will overfit. No native missing-modality handling.

**Verdict:** reject on sample-size grounds. Worth one sentence in a related-work section.

### 2.6 Cross-attention transformers

Multimodal Transformer (**Tsai et al., 2019**) and successors use cross-modal attention
over sequences.

**Cons:** our modalities are single vectors, not sequences, so most of the architecture's
value (temporal alignment) does not apply. Large parameter count. **Verdict:** reject —
architecturally mismatched, not merely too big.

### 2.7 Product-of-experts / mixture-of-product-of-experts (generative)

MVAE (**Wu & Goodman, 2018**) and MoPoE (**Sutter et al., 2021**) give a principled
latent-variable treatment where missing modalities are genuinely marginalised out, and
PoE has the right identity behaviour.

**Pros:** correct missing-modality semantics; generative, so can impute. **Cons:**
training a multimodal VAE on 2298 cases for a discriminative task adds substantial
machinery for uncertain gain; PoE is known to be overconfident when experts disagree —
precisely our failure case.

**Verdict:** the honest runner-up. Worth noting as future work; too much machinery for one
semester.

### 2.8 Evidential / Dempster–Shafer combination — **the choice**

Per-modality Dirichlet evidence, combined with the DS rule
([04 §2–3](04_mathematical_foundations.md)). Trusted Multi-View Classification
(**Han et al., ICLR 2021**) is the reference.

Scored against the four constraints from §1:

| Constraint | How DS satisfies it |
|---|---|
| Modalities absent | Vacuous opinion is the **identity element** — absence is handled by the algebra, not by imputation or masking |
| Tiny data | One softplus head per modality (`d × K` params). Cheapest option in the table besides pooling |
| Per-case quality | `u_m` is per-case and learned, not a fixed weight |
| Conflict representable | Conflict mass `C` is **separate from** ignorance `u` — the one operator here that distinguishes "they disagree" from "nobody knows" |
| Calibrated uncertainty for deferral | `u` and `C` feed directly into the deferral rule |

**Verdict: adopt.** It is the only operator in the space that satisfies all five
requirements, and it does so with fewer parameters than the gated fusion already in use.

---

## 3. Decision

| Role | Method |
|---|---|
| **Primary (contribution)** | Evidential heads + DS combination + EM diagnosis-adaptive refinement |
| Baseline — unimodal | A (image), B (metadata), C (text) |
| Baseline — literature | **MetaBlock** (image+metadata) |
| Baseline — naive | Early-fusion concat MLP |
| Baseline — pooling | Geometric late fusion, uniform weights |
| Baseline — gated | Existing `AdaptiveGatedFusion` (specialists D–G) |
| Ablations | DS without EM; EM without DS; uniform `w`; no conflict term; hard vs soft prior |
| Future work | MoPoE; cross-attention if a larger corpus becomes available |

This ladder answers the reviewer's question — *does the fancy operator beat the simple
one?* — and is honest about the possibility that it does not.

---

## 4. Missing-modality training protocol

Whatever the operator, the model must be *trained* to handle absence, or it will only
encounter complete cases and degrade sharply at inference.

**Modality dropout.** During training, independently drop each present modality with
probability `p_drop` (0.15–0.3; v1 used 0.3), replacing it with the vacuous opinion. This
makes the absence pathway part of the training distribution. The existing
`AdaptiveGatedFusion` already does a crude version (15% chance of zeroing one modality) —
keep the idea, apply it to the evidential heads.

**Evaluate on every subset.** With `M = 3` there are 7 non-empty availability patterns.
Report the full table, not just the complete-case number. A method whose advantage only
appears with all three modalities present has not demonstrated robustness to imperfect
evidence, which is this project's entire claim.

---

## 5. Pitfalls specific to this dataset

1. **The metadata is suspiciously strong.** In honest evaluation the metadata-only model
   reached ~94% while image-only reached ~21% on live images
   ([evaluation_integrity.md §2](evaluation_integrity.md)). Before claiming multimodal
   benefit, check whether engineered metadata features leak the label. The preprocessor
   computes `bcc_score`, `mel_score`, `scc_score`, … as hand-crafted per-class
   likelihoods from symptoms and location. **These are diagnosis-shaped features**, and
   if they were designed with knowledge of the labels they are a leak dressed as feature
   engineering. **Audit this first** — it may explain the entire metadata result, and it
   would invalidate any fusion claim built on top.
2. **Fusion can only help if modalities are complementary.** Report the per-modality
   confusion matrices and their error overlap. If text adds nothing (and at ~27% it
   currently does not), the honest conclusion is a two-modality system — and that is a
   finding, not a failure.
3. **Geometric pooling with a broken modality is dangerous.** A product of experts lets
   one confidently-wrong modality veto the right answer. With Model C at near-chance,
   uniform-weight geometric pooling will actively hurt. This is the strongest argument
   for the evidential route: a near-chance modality produces low evidence, hence `u_m→1`,
   hence the vacuous opinion, hence no influence — automatically.
4. **Do not tune on test.** The temptation with 7 availability patterns × several methods
   is to report the best cell. Fix the protocol in advance and report the whole table.

---

## 6. Implementation sketch

```python
class EvidentialHead(nn.Module):
    """Modality encoder output -> Dirichlet evidence."""
    def __init__(self, d_in, n_classes):
        super().__init__()
        self.fc = nn.Linear(d_in, n_classes)

    def forward(self, z):
        e = F.softplus(self.fc(z))          # non-negative evidence
        alpha = e + 1.0
        S = alpha.sum(-1, keepdim=True)
        return {"evidence": e, "alpha": alpha,
                "belief": e / S,            # b_k
                "u": z.new_full((z.shape[0], 1), alpha.shape[-1]) / S}


def ds_combine(op_a, op_b):
    """Dempster-Shafer combination of two opinions. Associative, so fold for M>2.
    A vacuous opinion (b=0, u=1) is the identity element."""
    b_a, u_a = op_a["belief"], op_a["u"]
    b_b, u_b = op_b["belief"], op_b["u"]
    # conflict mass: belief assigned to mutually exclusive class pairs
    bb = b_a.sum(-1, keepdim=True) * b_b.sum(-1, keepdim=True) - (b_a * b_b).sum(-1, keepdim=True)
    C = bb.clamp(0.0, 1.0 - 1e-6)
    denom = 1.0 - C
    b = (b_a * b_b + b_a * u_b + b_b * u_a) / denom
    u = (u_a * u_b) / denom
    return {"belief": b, "u": u, "conflict": C}


def vacuous(batch, n_classes, device):
    """The identity element -- what an ABSENT modality contributes."""
    return {"belief": torch.zeros(batch, n_classes, device=device),
            "u": torch.ones(batch, 1, device=device)}
```

Folding over present modalities, with `vacuous()` for absent ones, replaces the entire
mask-build / mask-multiply / renormalise / guard-against-zero path in
`src/fusion/adaptive_fusion.py` — roughly 40 lines of special-casing collapse into an
associative fold with an identity element.
