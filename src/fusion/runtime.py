"""
src/fusion/runtime.py - The specialist team for ONE case (used by the engine/app).

Same arithmetic as scripts/run_team.py, applied to a single patient:
five specialists -> Devil's Advocate profile (Q, R, U, C, M, OOD) -> iterative
diagnosis-adaptive fusion -> calibrated confidence -> consensus level.
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
            else:
                p, q, ok = None, 0.0, False
            P.append(p if p is not None else np.full(K, 1.0 / K))
            Q.append(q)
            pres.append(ok)

        P = np.stack(P)[None]
        Q = np.array(Q, dtype=float)[None]
        pres = np.array(pres)[None]
        R = mgr.reliability(P, Q, pres)
        p, w, iters = fusion.fuse(P, Q * R, pres, return_trace=True)
        p, w = p[0], w[0]
        consistency = float(js_consistency(P, pres)[0])

        ood = False
        if z.get("clinical") is not None and z.get("meta") is not None:
            ood = bool(b.ood.flag(np.hstack([z["clinical"], z["meta"]])[None])[0])

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
            "ood": ood, "iterations": int(iters[0]),
            "calibrated_confidence": conf_cal,
        }


def consensus_level(decision, agreeing: int, n: int) -> int:
    """The original 4-level consensus hierarchy, now driven by the calibrated policy."""
    if decision.action.value == "DEFER":
        return 4
    if decision.action.value == "ACQUIRE" or decision.tier == "LOW":
        return 3
    return 1 if agreeing >= min(3, n) else 2
