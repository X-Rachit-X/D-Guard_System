# *Appendix I: Selected Source Code

This appendix reproduces the most important functions of the system exactly as they appear in the repository. The complete source code (about 14,600 lines of Python), the tests and the experiment scripts are in the project repository.

**A.1  The learned evidence gate (Devil's Advocate v2): features and fitting** (file: `src/fusion/gating.py`)

```
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
```

```
    def fuse(self, P, F, present):
        w = self._weights(F, present.astype(bool), self.theta, self.beta, np)
        p = self._pool(P, w, np)
        lp = np.log(np.clip(p, EPS, 1)) / self.temperature
        p = np.exp(lp - lp.max(1, keepdims=True))
        return p / p.sum(1, keepdims=True), w
```

**A.2  All fusion rules behind one interface** (file: `src/fusion/methods.py`)

```
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
```

**A.3  Cross-fitting and rule selection** (file: `scripts/run_team.py`)

```
def cv_folds(n, folds=5, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    return [(np.setdiff1d(idx, idx[k::folds]), idx[k::folds]) for k in range(folds)]
```

```
def run_methods(names, Pv, Qv, Mv, yv, Ov, Pt, Qt, Mt, Ot, methods, folds):
    """Out-of-fold val + full-fit test probabilities for every method."""
    oof = {m: np.zeros((len(yv), K)) for m in methods}
    for tr, te in folds:
        ctx = fit_context(names, K, Pv[tr], Qv[tr], Mv[tr], yv[tr])
        for m in methods:
            fit_method(ctx, m, Pv[tr], Qv[tr], Mv[tr], yv[tr], Ov[tr])
            oof[m][te] = fuse_method(ctx, m, Pv[te], Qv[te], Mv[te], Ov[te])[0]
    ctx = fit_context(names, K, Pv, Qv, Mv, yv)
    test, wts = {}, {}
    for m in methods:
        fit_method(ctx, m, Pv, Qv, Mv, yv, Ov)
        test[m], wts[m] = fuse_method(ctx, m, Pt, Qt, Mt, Ot)
    return oof, test, wts, ctx
```

**A.4  Occlusion check of the quality agent** (file: `src/agents/quality.py`)

```
def _centre(g: np.ndarray) -> np.ndarray:
    """Middle 60% of the photo - where the lesion is framed. Dermoscopy vignettes
    and dark backgrounds are flat too, but they sit at the edges."""
    h, w = g.shape
    return g[h // 5: 4 * h // 5, w // 5: 4 * w // 5]
```

```
def flat_fraction(g: np.ndarray, block: int = 8, tol: float = 0.004) -> float:
    """Share of the (centre of the) photo covered by perfectly flat 8x8 blocks.

    Real skin photos always carry sensor noise and texture, so a block whose
    grey levels vary by less than ~1 step out of 255 is almost never skin: it is
    a sticker, a finger in shadow, a black bar or a covered lens. This is the
    occlusion detector (DERMA-STRESS showed occlusion was the one corruption the
    other checks missed).
    """
    h, w = (g.shape[0] // block) * block, (g.shape[1] // block) * block
    if h == 0 or w == 0:
        return 0.0
    b = g[:h, :w].reshape(h // block, block, w // block, block)
    return float((b.std(axis=(1, 3)) < tol).mean())
```

**A.5  Quality and missing-details gates in the agent pipeline** (file: `src/agents/pipeline.py`)

```
def has_details(metadata: Optional[Dict]) -> bool:
    """True if at least one of age / sex / body site was given."""
    m = metadata or {}
    return any(m.get(k) not in (None, "", "unknown") and m.get(k) == m.get(k)  # NaN check
               for k in DETAIL_KEYS)
```

```
    def assess(self, clinical: Optional[Image.Image], dermoscopy: Optional[Image.Image],
               metadata: Optional[Dict]):
        q = {"clinical photo": self.quality.check(clinical),
             "dermoscopy": self.dermo_quality.check(dermoscopy)}
        metadata = dict(metadata or {})
        symptoms = metadata.pop("symptoms_text", None)       # the model has no text input
        result = self.engine.assess(clinical, dermoscopy, metadata)
        result.quality = {k: (v.to_dict() if v else None) for k, v in q.items()}

        # QUALITY GATE. The calibrated guarantees assume this photo resembles the
        # calibration photos. One with a SEVERE quality issue is, by construction,
        # outside the 1-99% range of those photos, so it cannot keep a HIGH-
        # confidence label. Measured reason (scripts/run_stress_milk.py): on blurred
        # photos the model's ignorance FALLS and its HIGH-tier rate RISES, and on
        # dark photos HIGH answers missed 12.5% of cancers - while the quality agent
        # flagged 100% of both. The gate only ever adds caution.
        #
        # MISSING-DETAILS GATE. Every calibration case had age, sex and site. With
        # none of them and no dermoscopy, DERMA-STRESS measured more missed cancers
        # among HIGH answers (6.4% vs 4.3% clean), so such an answer stays LOW.
        from src.decision.policy import DeferReason
        d = result.decision
        flagged = [k for k, v in q.items() if v is not None and v.severe]
        reason = None
        if flagged:
            reason, gate = DeferReason.QUALITY, "photo_quality"
        if dermoscopy is None and not has_details(metadata):
            # Details are the cheapest evidence there is: ASK for them (acquisition)
            # instead of answering without them. The tentative answer is kept.
            if d.action.value != "DEFER":
                from src.decision.explain import explain_doctor, explain_patient
                from src.decision.policy import Action
                d.action = Action.ACQUIRE
                d.acquire_target = "metadata"
                d.tier = None
                d.gates = {**(d.gates or {}), "missing_details": True}
                result.patient_message = explain_patient(d, self.engine.classes,
                                                         MILK_CLASS_FULL_NAMES)
                result.doctor_message = explain_doctor(d, self.engine.classes,
                                                       MILK_CLASS_FULL_NAMES)
        if reason is not None and d.action.value == "PROCEED" and d.tier != "LOW":
            from src.decision.explain import explain_doctor, explain_patient
            d.tier = "LOW"
            d.low_reason = reason
            d.gates = {**(d.gates or {}), gate: True}
            result.patient_message = explain_patient(d, self.engine.classes,
                                                     MILK_CLASS_FULL_NAMES)
            result.doctor_message = explain_doctor(d, self.engine.classes,
                                                   MILK_CLASS_FULL_NAMES)

        # SCRIBE: warning signs in the patient's own words raise triage urgency;
        # they never change the diagnosis or the confidence tier.
        sr = self.scribe.read(symptoms, metadata.get("age"))
        if sr is not None:
            result.scribe = sr.to_dict()
            if sr.red_flags and result.decision.urgency != "URGENT":
                result.decision.urgency = "URGENT"
                from src.decision.explain import explain_doctor, explain_patient
                result.patient_message = explain_patient(result.decision, self.engine.classes,
                                                         MILK_CLASS_FULL_NAMES)
                result.doctor_message = explain_doctor(result.decision, self.engine.classes,
                                                       MILK_CLASS_FULL_NAMES)
            if sr.red_flags:
                result.patient_message += ("\nYou mentioned " + ", ".join(sr.red_flags)
                                           + " - please make sure a doctor sees this spot soon.")

        tent = result.decision.tentative
        tent_name = self.engine.classes[tent] if tent is not None else None
        if self.librarian is not None and result.embeddings.get("clinical") is not None:
            result.similar = self.librarian.similar(result.embeddings.get("clinical"),
                                                    result.embeddings.get("meta"), tent_name)
        result.report = self.reporter.write(result, result.quality, result.similar)
        if self.explainer is not None and result.team:
            result.explanation = self.explainer.explain(result.team, result.decision)

        advice = [a for v in q.values() if v for a in v.advice]
        if advice:
            result.patient_message += ("\nPhoto tip: " + " ".join(dict.fromkeys(advice)))
        return result
```

**A.6  Data deletion** (file: `src/care/service.py`)

```
    def _remove_uploads(self, paths) -> int:
        """Delete only files this platform saved (inside upload_dir) - never the
        dataset images used by the demo."""
        n = 0
        root = Path(self.upload_dir).resolve()
        for p in paths:
            try:
                fp = Path(p).resolve()
                if root in fp.parents and fp.exists():
                    fp.unlink()
                    n += 1
            except OSError:
                pass
        return n
```

```
    def delete_patient(self, pid: int, actor: str = "patient") -> int:
        return self._remove_uploads(self.store.delete_patient(pid, actor))
```
