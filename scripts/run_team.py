"""
scripts/run_team.py - Specialist team + Devil's Advocate + adaptive fusion on MILK10k.

Fits everything on VALIDATION lesions, then scores the TEST split once:

    specialists   S1 clinical head · S2 dermoscopy head · S3 metadata head
                  S4 original PAD-UFES Model A · S5 kNN over training lesions
    evidence      Q, learned R, U, C (JS), M, OOD (Mahalanobis) per case
    fusion        seven combination rules (src/fusion/methods.py), each fitted with
                  5-fold cross-fitting on val; the winner is picked on val
                  out-of-fold macro-F1; every subset of specialists is also scored
    confidence    isotonic map from fused confidence to P(correct)

Writes outputs/milk_team/{val,test}.npz in the same format as outputs/milk_native,
so the decision engine runs on the team's fused probabilities unchanged:

    python scripts/run_team.py
    python scripts/run_eade_milk.py --native outputs/milk_team --missed-cost 5 \
        --out outputs/eade_team/cost_5

Prerequisites: scripts/milk_native_infer.py and scripts/build_specialists.py.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score  # noqa: E402

from src.constants import MILK_CLASS_NAMES, milk_malignant_indices  # noqa: E402
from src.fusion.adaptive import (OODDetector, TeamBundle, knn_probs,  # noqa: E402
                                 quality_score)
from src.fusion.methods import (LABELS, METHODS, fit_context, fit_method,  # noqa: E402
                                fuse_method, oracle_best_head)

K = len(MILK_CLASS_NAMES)
MAL = milk_malignant_indices()
NAT = ROOT / "outputs" / "milk_native"
SPEC = ROOT / "outputs" / "specialists"
OUT = ROOT / "outputs" / "milk_team"
KNN_K = 15


def load(p):
    d = np.load(p, allow_pickle=True)
    return {k: d[k] for k in d.files}


def pign(b, u):
    return b + u / K


def metrics(y, p):
    pred = p.argmax(1)
    pm = p[:, MAL].sum(1)
    return {"acc": accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, labels=list(range(K)), average="macro",
                                 zero_division=0),
            "bal_acc": balanced_accuracy_score(y, pred),
            "mal_auroc": roc_auc_score(np.isin(y, MAL), pm)}


def build(nat, spec, train, q_ref_c, q_ref_d, stage):
    """Stack specialist distributions P (N,M,K), quality Q (N,M), presence (N,M)."""
    N = len(nat["y"])
    names, P, Q, pres = [], [], [], []

    def add(name, p, q, present=True):
        names.append(name)
        P.append(p)
        Q.append(np.broadcast_to(q, (N,)).astype(float))
        pres.append(np.full(N, present))

    qc = quality_score(spec["q_clin"], q_ref_c)
    qd = quality_score(spec["q_derm"], q_ref_d)
    add("clinical", pign(nat["mod_image_belief"], nat["mod_image_u"]), qc)
    if stage == "B":
        add("dermoscopy", pign(nat["mod_dermoscopy_belief"], nat["mod_dermoscopy_u"]), qd)
    add("metadata", pign(nat["mod_meta_belief"], nat["mod_meta_u"]), 1.0)
    if "pad_prob" in spec:
        # PAD Model A has no mass on 5 classes; mix in a little uniform so a
        # missing class is "unknown", not "impossible".
        add("pad_ufes_A", 0.9 * spec["pad_prob"] + 0.1 / K, qc)
    if stage == "A":
        zq, zr = (np.hstack([spec["z_img"], spec["z_meta"]]),
                  np.hstack([train["z_img"], train["z_meta"]]))
    else:
        zq, zr = (np.hstack([spec["z_img"], spec["z_derm"], spec["z_meta"]]),
                  np.hstack([train["z_img"], train["z_derm"], train["z_meta"]]))
    pk, sim = knn_probs(zq, zr, train["y"], K, k=KNN_K)
    add("knn_cases", pk, np.clip(sim, 0, 1))
    return names, np.stack(P, 1), np.stack(Q, 1), np.stack(pres, 1)


def cv_folds(n, folds=5, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    return [(np.setdiff1d(idx, idx[k::folds]), idx[k::folds]) for k in range(folds)]


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


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="auto", choices=["auto"] + METHODS,
                    help="combination rule; auto = best val out-of-fold macro-F1")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no-subsets", action="store_true", help="skip the subset study")
    ap.add_argument("--native", default=str(NAT), help="milk_native_infer output")
    ap.add_argument("--spec", default=str(SPEC), help="build_specialists output")
    args = ap.parse_args()
    out_dir = Path(args.out)
    nat_dir, spec_dir = Path(args.native), Path(args.spec)
    nat = {s: load(nat_dir / f"{s}.npz") for s in ("val", "test")}
    spec = {s: load(spec_dir / f"{s}.npz") for s in ("train", "val", "test")}
    for s in ("val", "test"):
        assert list(nat[s]["lesion_id"]) == list(spec[s]["lesion_id"]), f"{s} order mismatch"
    train = spec["train"]
    q_ref_c = np.percentile(train["q_clin"], [5, 95], axis=0)
    q_ref_d = np.percentile(train["q_derm"], [5, 95], axis=0)
    yv, yt = nat["val"]["y"], nat["test"]["y"]
    folds = cv_folds(len(yv))

    # OOD first: its distance is one of the Devil's Advocate's inputs
    zcat = lambda d: np.hstack([d["z_img"], d["z_meta"]])  # noqa: E731
    ood = OODDetector().fit(zcat(train), train["y"], K)
    ood.calibrate(zcat(spec["val"]), q=99.0)
    Ov = ood.score(zcat(spec["val"])) / ood.threshold
    Ot = ood.score(zcat(spec["test"])) / ood.threshold

    print("=" * 78)
    print("SPECIALIST TEAM + DEVIL'S ADVOCATE + FUSION COMPARISON (5-fold cross-fit on val)")
    print("=" * 78)
    report = {"specialists": {}, "fusion": {}, "fusion_val_oof": {}, "selected": {},
              "subsets": {}, "gate_importance": {}, "oracle": {}, "mean_weights": {}}
    team = {}
    for stage, view in (("A", "no_dermo"), ("B", "full")):
        names, Pv, Qv, Mv = build(nat["val"], spec["val"], train, q_ref_c, q_ref_d, stage)
        _, Pt, Qt, Mt = build(nat["test"], spec["test"], train, q_ref_c, q_ref_d, stage)

        print(f"\n--- stage {stage} ({'photo + details' if stage == 'A' else '+ dermoscopy'}) ---")
        print(f"{'specialist':14} {'acc':>6} {'macroF1':>8} {'balAcc':>7} {'malAUROC':>9}")
        for m, n in enumerate(names):
            r = metrics(yt, Pt[:, m])
            report["specialists"][f"{stage}:{n}"] = r
            print(f"{n:14} {r['acc']:6.3f} {r['macro_f1']:8.3f} {r['bal_acc']:7.3f} "
                  f"{r['mal_auroc']:9.3f}")

        oof, test, wts, ctx = run_methods(names, Pv, Qv, Mv, yv, Ov, Pt, Qt, Mt, Ot,
                                          METHODS, folds)
        test_rows = {"DS fusion (evidential heads)": nat["test"][f"{view}_prob"]}
        val_rows = {"DS fusion (evidential heads)": nat["val"][f"{view}_prob"]}
        for m in METHODS:
            test_rows[LABELS[m]], val_rows[LABELS[m]] = test[m], oof[m]
        print(f"\n{'fusion method':46} {'val F1':>7} | {'acc':>6} {'macroF1':>8} "
              f"{'balAcc':>7} {'malAUROC':>9}")
        for k in test_rows:
            r, rv = metrics(yt, test_rows[k]), metrics(yv, val_rows[k])
            report["fusion"][f"{stage}:{k}"] = r
            report["fusion_val_oof"][f"{stage}:{k}"] = rv
            print(f"{k:46} {rv['macro_f1']:7.3f} | {r['acc']:6.3f} {r['macro_f1']:8.3f} "
                  f"{r['bal_acc']:7.3f} {r['mal_auroc']:9.3f}")
        report["oracle"][stage] = metrics(yt, oracle_best_head(Pt, Mt, yt))
        print(f"{'(oracle best head - uses the label, ceiling)':46} {'':7} | "
              f"{report['oracle'][stage]['acc']:6.3f} {report['oracle'][stage]['macro_f1']:8.3f}")

        # pick the winner on VALIDATION (out-of-fold macro-F1) - never on test
        best = (max(METHODS, key=lambda m: metrics(yv, oof[m])["macro_f1"])
                if args.method == "auto" else args.method)
        report["selected"][stage] = best
        print(f"selected on validation: {LABELS[best]}")
        gate = best if best.startswith("gate") else "gate_linear"
        report["gate_importance"][stage] = ctx.models[gate].importance()
        report["mean_weights"][stage] = {n: float(wts[best][:, i][Mt[:, i]].mean())
                                         for i, n in enumerate(names)}
        print("mean weights on test: " + ", ".join(
            f"{n} {v:.2f}" for n, v in report["mean_weights"][stage].items()))
        print("what the learned gate listens to: " + ", ".join(
            f"{k} {v:.2f}" for k, v in report["gate_importance"][stage].items()))

        # which specialists are worth having? every subset, with the selected method
        import itertools
        for r_ in (range(1, len(names) + 1) if not args.no_subsets else ()):
            for sub in itertools.combinations(range(len(names)), r_):
                sub = list(sub)
                if len(sub) == 1:
                    pv_, pt_ = Pv[:, sub[0]], Pt[:, sub[0]]
                else:
                    o_, t_, _, _ = run_methods([names[i] for i in sub], Pv[:, sub], Qv[:, sub],
                                               Mv[:, sub], yv, Ov, Pt[:, sub], Qt[:, sub],
                                               Mt[:, sub], Ot, [best], folds)
                    pv_, pt_ = o_[best], t_[best]
                key = "+".join(names[i] for i in sub)
                report["subsets"][f"{stage}:{key}"] = {
                    "val_macro_f1": metrics(yv, pv_)["macro_f1"], **metrics(yt, pt_)}
        subs = {k: v for k, v in report["subsets"].items() if k.startswith(stage)}
        top = sorted(subs.items(), key=lambda kv: -kv[1]["val_macro_f1"])[:5]
        print("best specialist subsets (by val): " + "; ".join(
            f"{k[2:]} val {v['val_macro_f1']:.3f} test {v['macro_f1']:.3f}" for k, v in top))

        team[stage] = dict(names=names, ctx=ctx, method=best, p_val=oof[best],
                           p_test=test[best])

    print(f"\nOOD (Mahalanobis, val p99 threshold): {(Ot > 1).mean():.1%} of test lesions flagged")

    # --------------------------------------------------- confidence calibration
    from sklearn.isotonic import IsotonicRegression
    conf_v = np.concatenate([team["A"]["p_val"].max(1), team["B"]["p_val"].max(1)])
    hit_v = np.concatenate([team["A"]["p_val"].argmax(1) == yv,
                            team["B"]["p_val"].argmax(1) == yv])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(conf_v, hit_v)
    for stage in ("A", "B"):
        c = iso.predict(team[stage]["p_test"].max(1))
        hit = team[stage]["p_test"].argmax(1) == yt
        print(f"calibrated confidence stage {stage}: mean shown {c.mean():.3f} vs actual "
              f"accuracy {hit.mean():.3f}")
    report["ood_test_rate"] = float((Ot > 1).mean())

    # ---------------------------------------------------------------- outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, idx in (("val", "p_val"), ("test", "p_test")):
        d = dict(nat[split])
        d["no_dermo_prob"] = team["A"][idx].astype(np.float32)
        d["full_prob"] = team["B"][idx].astype(np.float32)
        d["ood"] = ood.flag(zcat(spec[split]))
        np.savez_compressed(out_dir / f"{split}.npz", **d)

    import joblib
    bundle = TeamBundle(
        names=team["B"]["names"], names_A=team["A"]["names"], classes=MILK_CLASS_NAMES,
        manager_A=team["A"]["ctx"].manager, manager_B=team["B"]["ctx"].manager,
        fusion_A=team["A"]["ctx"].prior, fusion_B=team["B"]["ctx"].prior, ood=ood,
        knn_ref_A=np.hstack([train["z_img"], train["z_meta"]]).astype(np.float32),
        knn_ref_B=np.hstack([train["z_img"], train["z_derm"], train["z_meta"]]).astype(np.float32),
        knn_y=train["y"], knn_k=KNN_K, q_ref_clin=q_ref_c, q_ref_derm=q_ref_d,
        confidence_calibrator=iso, report=report,
        ctx_A=team["A"]["ctx"], ctx_B=team["B"]["ctx"],
        method_A=team["A"]["method"], method_B=team["B"]["method"])
    joblib.dump(bundle, out_dir / "team_bundle.joblib")
    (out_dir / "team_report.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nsaved -> {out_dir}/(val|test).npz, team_bundle.joblib, team_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
