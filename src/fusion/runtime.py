"""
src/fusion/runtime.py - The specialist team for ONE case (used by the engine/app).

Same arithmetic as scripts/run_team.py, applied to a single patient:
five specialists -> Devil's Advocate profile (Q, R, U, C, M, OOD) -> the
combination rule picked on validation (src/fusion/methods.py; v2 learned gate or
v1 iterative EM) -> calibrated confidence -> consensus level.

`compare(classes)` re-fuses the LAST case with every combination rule, for the
app's Comparative-study view.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from src.fusion.adaptive import js_consistency, knn_probs, norm_entropy, quality_score

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEAM = ROOT / "outputs" / "milk_team" / "team_bundle.joblib"

DISPLAY = {
    "clinical": "clinical-photo head",
    "dermoscopy": "dermoscopy head",
    "metadata": "patient-details head",
    "pad_ufes_A": "PAD-UFES Model A (original system)",
    "knn_cases": "similar-case vote",
    "embed_clf": "embedding classifier",
}

CONSENSUS_TEXT = {
    1: "Level 1 - strong consensus",
    2: "Level 2 - moderate agreement (evidence-weighted arbitration)",
    3: "Level 3 - high uncertainty (more evidence or confirmation needed)",
    4: "Level 4 - critical conflict (human decision)",
}


class TeamRuntime:
    def __init__(self, classes, bundle_path: Path = DEFAULT_TEAM):
        import joblib

        from src.specialists import PadLegacySpecialist
        self.bundle = joblib.load(bundle_path)
        if list(self.bundle.classes) != list(classes):
            raise ValueError("team bundle was built for a different class list")
        self.K = len(classes)
        self.classes = list(classes)
        self.pad = PadLegacySpecialist(self.classes)

    def fuse_case(self, head_probs: Dict[str, np.ndarray], clinical, dermoscopy,
                  z: Dict[str, Optional[np.ndarray]], q_stats: Dict[str, Optional[np.ndarray]]):
        b = self.bundle
        stage = "B" if dermoscopy is not None else "A"
        names = b.names if stage == "B" else b.names_A
        mgr = b.manager_B if stage == "B" else b.manager_A
        fusion = b.fusion_B if stage == "B" else b.fusion_A
        K = self.K

        qc = (quality_score(q_stats["clinical"][None], b.q_ref_clin)[0]
              if q_stats.get("clinical") is not None else 0.0)
        qd = (quality_score(q_stats["dermoscopy"][None], b.q_ref_derm)[0]
              if q_stats.get("dermoscopy") is not None else 0.0)

        P, Q, pres = [], [], []
        for n in names:
            if n == "clinical":
                p, q, ok = head_probs.get("clinical"), qc, clinical is not None
            elif n == "dermoscopy":
                p, q, ok = head_probs.get("dermoscopy"), qd, dermoscopy is not None
            elif n == "metadata":
                p, q, ok = head_probs.get("metadata"), 1.0, head_probs.get("metadata") is not None
            elif n == "pad_ufes_A":
                raw = self.pad.predict(clinical) if clinical is not None else None
                p = None if raw is None else 0.9 * raw + 0.1 / K
                q, ok = qc, p is not None
            elif n == "knn_cases":
                parts = [z.get("clinical"), z.get("dermoscopy"), z.get("meta")] \
                    if stage == "B" else [z.get("clinical"), z.get("meta")]
                if any(v is None for v in parts):
                    p, q, ok = None, 0.0, False
                else:
                    ref = b.knn_ref_B if stage == "B" else b.knn_ref_A
                    pk, sim = knn_probs(np.hstack(parts)[None], ref, b.knn_y, K, k=b.knn_k)
                    p, q, ok = pk[0], float(np.clip(sim[0], 0, 1)), True
            elif n == "embed_clf":
                clf = getattr(b, "clf_B" if stage == "B" else "clf_A", None)
                parts = [z.get({"z_img": "clinical", "z_derm": "dermoscopy",
                                "z_meta": "meta"}[k]) for k in (clf.keys if clf else [])]
                if clf is None or any(v is None for v in parts):
                    p, q, ok = None, 0.0, False
                else:
                    p, q, ok = clf.predict(np.hstack(parts)[None])[0], 1.0, True
            else:
                p, q, ok = None, 0.0, False
            P.append(p if p is not None else np.full(K, 1.0 / K))
            Q.append(q)
            pres.append(ok)

        P = np.stack(P)[None]
        Q = np.array(Q, dtype=float)[None]
        pres = np.array(pres)[None]
        R = mgr.reliability(P, Q, pres)
        consistency = float(js_consistency(P, pres)[0])

        ood, ood_dist = False, None
        if z.get("clinical") is not None and z.get("meta") is not None:
            zc = np.hstack([z["clinical"], z["meta"]])[None]
            ood_dist = float(b.ood.score(zc)[0] / b.ood.threshold)
            ood = ood_dist > 1.0

        ctx = getattr(b, "ctx_B" if stage == "B" else "ctx_A", None)
        method = getattr(b, "method_B" if stage == "B" else "method_A", "adaptive_em")
        iters = np.zeros(1, dtype=int)
        if ctx is not None:
            from src.fusion.methods import fuse_method
            ood_arr = None if ood_dist is None else np.array([ood_dist])
            p, w = fuse_method(ctx, method, P, Q, pres, ood_arr)
            if method == "adaptive_em":
                _, _, iters = fusion.fuse(P, Q * R, pres, return_trace=True)
        else:
            p, w, iters = fusion.fuse(P, Q * R, pres, return_trace=True)
        p, w = p[0], w[0]
        self.last = dict(P=P, Q=Q, pres=pres, R=R, names=names, stage=stage,
                         ood=None if ood_dist is None else np.array([ood_dist]))

        y = int(p.argmax())
        votes = {DISPLAY[n]: self.classes[int(P[0, i].argmax())]
                 for i, n in enumerate(names) if pres[0, i]}
        agree = sum(1 for i in range(len(names)) if pres[0, i] and P[0, i].argmax() == y)
        conf_cal = float(b.confidence_calibrator.predict([p.max()])[0]) \
            if b.confidence_calibrator is not None else float(p.max())
        return {
            "prob": p, "stage": stage,
            "weights": {DISPLAY[n]: round(float(w[i]), 3)
                        for i, n in enumerate(names) if pres[0, i]},
            "reliability": {DISPLAY[n]: round(float(R[0, i]), 3)
                            for i, n in enumerate(names) if pres[0, i]},
            "quality": {DISPLAY[n]: round(float(Q[0, i]), 3)
                        for i, n in enumerate(names) if pres[0, i]},
            "uncertainty": {DISPLAY[n]: round(float(norm_entropy(P[0, i])), 3)
                            for i, n in enumerate(names) if pres[0, i]},
            "votes": votes, "agreeing": agree, "n_specialists": int(pres.sum()),
            "consistency": consistency, "missingness": float(1 - pres.mean()),
            "ood": ood, "ood_distance": ood_dist, "iterations": int(iters[0]),
            "method": method,
            "probs": {DISPLAY[n]: [round(float(x), 4) for x in P[0, i]]
                      for i, n in enumerate(names) if pres[0, i]},
            # measured: how often this specialist was right on validation lesions of
            # the class the team now believes (the original "diagnosis-adaptive prior")
            "utility": {DISPLAY[n]: round(float(fusion.W_prior[y, i]), 3)
                        for i, n in enumerate(names) if pres[0, i]},
            "gate_importance": (b.report.get("gate_importance") or {}).get(stage, {}),
            "calibrated_confidence": conf_cal,
        }


def compare_last(rt: "TeamRuntime") -> Dict[str, Dict]:
    """Every combination rule on the most recent case: prediction, confidence, weights."""
    from src.fusion.methods import LABELS, METHODS, fuse_method
    L = rt.last
    ctx = getattr(rt.bundle, "ctx_B" if L["stage"] == "B" else "ctx_A", None)
    if ctx is None:
        return {}
    out = {}
    for i, n in enumerate(L["names"]):
        if L["pres"][0, i]:
            p = L["P"][0, i]
            out[f"Specialist alone: {DISPLAY[n]}"] = {
                "prediction": rt.classes[int(p.argmax())], "confidence": float(p.max()),
                "p_malignant": None, "weights": None, "prob": p}
    for m in METHODS:
        if m not in ctx.models and m in ("stacking", "gate_linear", "gate_log"):
            continue
        p, w = fuse_method(ctx, m, L["P"], L["Q"], L["pres"], L["ood"])
        out[LABELS[m]] = {
            "prediction": rt.classes[int(p[0].argmax())], "confidence": float(p[0].max()),
            "weights": {DISPLAY[n]: round(float(w[0, i]), 3)
                        for i, n in enumerate(L["names"]) if L["pres"][0, i]},
            "prob": p[0]}
    return out


def consensus_level(decision, agreeing: int, n: int) -> int:
    """The original 4-level consensus hierarchy, now driven by the calibrated policy."""
    if decision.action.value == "DEFER":
        return 4
    if decision.action.value == "ACQUIRE" or decision.tier == "LOW":
        return 3
    return 1 if agreeing >= min(3, n) else 2
