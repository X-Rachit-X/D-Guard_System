"""
scripts/run_eade.py - Calibrate and evaluate the Evidence-Adaptive Decision Engine.

End-to-end demonstration of the full decision path on real data:

    reproducible features
      -> trained specialists A/B/C (one per modality)
      -> evidential opinions (bridge mode: entropy -> ignorance)
      -> Dempster-Shafer fold
      -> conformal sets + Learn-then-Test threshold, fitted on a CALIBRATION split
      -> PROCEED / ACQUIRE / DEFER on the TEST split
      -> AUGRC, coverage at guaranteed risk, guarantee violation, baselines

Splits are disjoint by patient: train / calibration / test. Calibration never
touches test.

HONESTY NOTE. This runs in `bridge` mode against the already-trained A-G ladder,
whose reported metrics are invalid (docs/evaluation_integrity.md) and which were
fitted on data overlapping this test split. The numbers below therefore
demonstrate that the DECISION MACHINERY works and that the guarantee is
enforced; they are not a generalisation estimate. Native-mode numbers require
the GPU training run (notebooks/kaggle_train_dguard.py).

Usage:
    python scripts/build_features.py            # once, without --fit-metadata-on-train
    python scripts/run_eade.py
    python scripts/run_eade.py --alpha 0.10 --delta 0.10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    import numpy as np
    import torch

    from src.agents.eade import (
        EvidenceAdaptiveEngine,
        malignant_indices,
        opinion_from_logits,
    )
    from src.calibration.ltt import guarantee_violation
    from src.constants import CLASS_NAMES, MODALITIES
    from src.data.case_loader import load_cases, patient_level_split
    from src.decision.policy import (
        Action,
        CostMatrix,
        ThreeActionPolicy,
        confidence_threshold_baseline,
    )
    from src.evidence.evidential import fold_opinions, vacuous
    from src.eval.selective import augrc, risk_coverage_curve, selective_risk
    from src.models.backends import get_backends

    ap = argparse.ArgumentParser(description="Calibrate + evaluate the EADE")
    ap.add_argument("--features", default="data/processed/features_v2.pt")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="target malignancy miss rate among proceeded cases")
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--conformal-alpha", type=float, default=0.10)
    ap.add_argument("--class-conditional", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/eade/eade_report.json")
    args = ap.parse_args()

    K = len(CLASS_NAMES)
    MAL = set(malignant_indices())
    b = get_backends()

    # ---- features ---------------------------------------------------------
    fpath = PROJECT_ROOT / args.features
    if not fpath.exists():
        print(f"Missing {fpath}.\nRun: python scripts/build_features.py")
        return 1
    blob = torch.load(fpath, map_location="cpu", weights_only=False)
    img_ids = blob["img_ids"]
    y_all = blob["labels"].numpy()
    print("=" * 76)
    print("Evidence-Adaptive Decision Engine - calibrate and evaluate")
    print(f"features built {blob['manifest']['created']}  n={len(img_ids)}")
    print("=" * 76)

    # ---- disjoint patient-level split: train / calib / test ---------------
    cases = load_cases(load_images=False)
    by_id = {c["case_id"]: c for c in cases}
    stub = [{"case_id": i, "patient_id": by_id[i]["patient_id"], "label": int(l)}
            for i, l in zip(img_ids, y_all) if i in by_id]
    sp = patient_level_split(stub, ratios=(0.60, 0.20, 0.20), seed=args.seed)
    pos = {c: i for i, c in enumerate(img_ids)}
    idx = {k: np.array([pos[c["case_id"]] for c in sp[k]]) for k in ("train", "val", "test")}
    calib_idx, test_idx = idx["val"], idx["test"]
    print(f"split  train={len(idx['train'])}  calibration={len(calib_idx)}  test={len(test_idx)}")
    print(f"       patients: " + "  ".join(
        f"{k}={len({c['patient_id'] for c in sp[k]})}" for k in ("train", "val", "test")))

    # ---- per-modality logits from the trained unimodal specialists --------
    print("\nrunning specialists A (image) / B (metadata) / C (text) ...")
    img_proj = b.img_proj
    text_proj = b.text_proj
    with torch.no_grad():
        z_img = img_proj(blob["img_feats"])
        z_txt = text_proj(blob["text_feats"])
    # Model B expects the preprocessor width the checkpoints were trained with.
    meta_dim = blob["meta_feats"].shape[1]
    mdl_a, mdl_b, mdl_c = b.specialist("A"), b.specialist("B"), b.specialist("C")
    if mdl_b.meta_encoder.input_dim != meta_dim:
        print(f"  metadata width {meta_dim} != checkpoint {mdl_b.meta_encoder.input_dim};"
              f" re-encoding with the fitted preprocessor")
        import pandas as pd
        df = pd.read_csv(PROJECT_ROOT / "data/raw/metadata.csv").set_index("img_id")
        rows = df.loc[img_ids].reset_index()
        z_meta = torch.tensor(b.preprocessor.transform(rows), dtype=torch.float32)
    else:
        z_meta = blob["meta_feats"]

    with torch.no_grad():
        lg_img = mdl_a(img_embed=z_img)[0]
        lg_meta = mdl_b(meta_embed=z_meta)[0]
        lg_txt = mdl_c(text_embed=z_txt)[0]

    # ---- evidential fold --------------------------------------------------
    print("folding evidential opinions (Dempster-Shafer) ...")
    op_img = opinion_from_logits(lg_img)
    op_meta = opinion_from_logits(lg_meta)
    op_txt = opinion_from_logits(lg_txt)
    n = len(img_ids)
    fused = fold_opinions([op_img, op_meta, op_txt, vacuous(n, K)])  # dermoscopy absent

    probs = fused.probability.numpy()
    ignorance = fused.u.squeeze(-1).numpy()
    conflict = fused.conflict.squeeze(-1).numpy()
    score = 1.0 - ignorance

    print(f"  mean ignorance u = {ignorance.mean():.3f}   "
          f"mean conflict C = {conflict.mean():.3f}")
    for name, op in (("image", op_img), ("metadata", op_meta), ("text", op_txt)):
        print(f"  u[{name:<8}] = {op.u.mean().item():.3f}")

    # ---- calibrate on the CALIBRATION split only --------------------------
    print(f"\ncalibrating on {len(calib_idx)} held-out cases "
          f"(alpha={args.alpha}, delta={args.delta}) ...")
    engine = EvidenceAdaptiveEngine(
        policy=ThreeActionPolicy(cost=CostMatrix(),
                                 malignant_classes=sorted(MAL), num_classes=K),
        mode="bridge")
    cal = engine.calibrate(probs[calib_idx], y_all[calib_idx], score[calib_idx],
                           alpha=args.alpha, delta=args.delta,
                           conformal_alpha=args.conformal_alpha,
                           class_conditional=args.class_conditional)
    # Conflict threshold: upper decile on calibration, so "conflicting" is defined
    # by the data rather than by a hand-picked 0.3.
    engine.policy.conflict_threshold = float(np.quantile(conflict[calib_idx], 0.90))

    ltt = cal["ltt"]
    print(f"  malignancy-miss base rate      : {cal['miss_base_rate']:.4f}")
    print(f"  LTT lambda_hat                 : {ltt['lambda_hat']}")
    print(f"  risk at lambda (calibration)   : {ltt['risk_at_lambda']}")
    print(f"  coverage at lambda (calib)     : {ltt['coverage_at_lambda']}")
    print(f"  conformal qhat (alpha={args.conformal_alpha}) : {cal['conformal']['qhat']:.4f}")
    print(f"  conflict threshold (p90)       : {engine.policy.conflict_threshold:.4f}")
    if not ltt["controlled"]:
        print(f"  NOT CONTROLLED: {ltt['note']}")

    # ---- evaluate on TEST -------------------------------------------------
    print(f"\nevaluating on {len(test_idx)} test cases ...")
    yt = y_all[test_idx]
    pt = probs[test_idx]
    sets = engine.conformal.predict_sets(pt)

    decisions = []
    for i in range(len(test_idx)):
        decisions.append(engine.policy.decide(
            pt[i], ignorance=float(ignorance[test_idx][i]),
            conflict=float(conflict[test_idx][i]), prediction_set=sets[i],
            acquirable=["dermoscopy_image"],
            predicted_benefit={"dermoscopy_image": 0.3}))

    acts = np.array([d.action.value for d in decisions])
    proceeded = acts == Action.PROCEED.value
    preds = np.array([d.prediction if d.prediction is not None else -1 for d in decisions])

    miss_t = np.array([1.0 if (t in MAL and p not in MAL) else 0.0
                       for t, p in zip(yt, preds)])
    cov = float(proceeded.mean())
    realised = float(miss_t[proceeded].mean()) if proceeded.any() else float("nan")
    acc_proceeded = float((preds[proceeded] == yt[proceeded]).mean()) if proceeded.any() else float("nan")

    print(f"\n{'action':<10}{'n':>6}{'share':>9}")
    print("-" * 26)
    for a in (Action.PROCEED, Action.ACQUIRE, Action.DEFER):
        m = acts == a.value
        print(f"{a.value:<10}{int(m.sum()):>6}{m.mean():>9.3f}")

    reasons = {}
    for d in decisions:
        if d.reason:
            reasons[d.reason.value] = reasons.get(d.reason.value, 0) + 1
    if reasons:
        print("\ndeferral / acquisition reasons:")
        for r, c in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {r:<24}{c:>5}")

    rc = risk_coverage_curve(yt, pt.argmax(1), score[test_idx])
    viol = guarantee_violation(score[test_idx], miss_t, ltt["lambda_hat"], args.alpha) \
        if ltt["controlled"] else {"note": "no threshold certified"}

    print("\n" + "=" * 76)
    print("TEST RESULTS")
    print("=" * 76)
    print(f"coverage (PROCEED share)            : {cov:.4f}")
    print(f"accuracy among proceeded            : {acc_proceeded:.4f}")
    print(f"malignancy miss rate among proceeded: {realised:.4f}   (target alpha={args.alpha})")
    if ltt["controlled"]:
        print(f"guarantee violation (realised-alpha): {viol['violation']:+.4f}"
              f"   {'VIOLATED' if viol['violated'] else 'HELD'}")
    print(f"AUGRC (primary, lower better)       : {rc.augrc:.4f}")
    print(f"AURC (comparability only)           : {rc.aurc:.4f}")
    print(f"coverage at risk<=0.05 / 0.10       : "
          f"{rc.coverage_at_risk(0.05):.3f} / {rc.coverage_at_risk(0.10):.3f}")

    cev = engine.conformal.evaluate(pt, yt)
    print(f"conformal coverage (target {cev['target_coverage']:.2f})     : "
          f"{cev['empirical_coverage']:.4f}")
    print(f"mean conformal set size             : {cev['mean_set_size']:.2f}")
    if cev["underpowered_classes"]:
        print(f"underpowered classes (too few calib): "
              f"{[CLASS_NAMES[c] for c in cev['underpowered_classes']]}")

    # ---- baseline B3: confidence-threshold abstention --------------------
    print("\nbaseline B3 (max-softmax abstention) at matched coverage:")
    thr = float(np.quantile(pt.max(1), 1.0 - cov)) if cov > 0 else 1.0
    b3 = confidence_threshold_baseline(pt, thr)
    b3_pred = pt.argmax(1)
    b3_miss = np.array([1.0 if (t in MAL and p not in MAL) else 0.0
                        for t, p in zip(yt, b3_pred)])
    b3_risk, b3_cov = selective_risk(yt, b3_pred, b3, loss=b3_miss)
    print(f"  coverage {b3_cov:.4f}   miss rate {b3_risk:.4f}   "
          f"AUGRC {augrc(yt, b3_pred, pt.max(1)):.4f}")
    print(f"  EADE     {cov:.4f}   miss rate {realised:.4f}   AUGRC {rc.augrc:.4f}")

    out = {
        "mode": "bridge",
        "warning": ("Bridge mode over the legacy A-G ladder, whose metrics are "
                    "invalid and whose training data overlaps this test split. "
                    "Demonstrates the machinery, NOT generalisation. See "
                    "docs/evaluation_integrity.md."),
        "split": {k: int(len(v)) for k, v in idx.items()},
        "calibration": cal,
        "conflict_threshold": engine.policy.conflict_threshold,
        "test": {
            "coverage": cov, "accuracy_among_proceeded": acc_proceeded,
            "malignancy_miss_rate_among_proceeded": realised,
            "target_alpha": args.alpha,
            "guarantee": viol,
            "augrc": rc.augrc, "aurc": rc.aurc,
            "risk_coverage": rc.to_dict(),
            "action_counts": {a.value: int((acts == a.value).sum())
                              for a in (Action.PROCEED, Action.ACQUIRE, Action.DEFER)},
            "reasons": reasons,
            "conformal": cev,
            "baseline_b3": {"coverage": b3_cov, "miss_rate": b3_risk,
                            "augrc": float(augrc(yt, b3_pred, pt.max(1)))},
        },
        "mean_ignorance": float(ignorance.mean()),
        "mean_conflict": float(conflict.mean()),
        "policy": engine.to_dict(),
    }
    path = PROJECT_ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved -> {path}")
    print("\nReminder: bridge mode over legacy weights. Machinery verified; "
          "generalisation NOT established.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
