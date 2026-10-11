"""
scripts/run_team.py - Specialist team + Devil's Advocate + adaptive fusion on MILK10k.

Fits everything on VALIDATION lesions, then scores the TEST split once:

    specialists   S1 clinical head · S2 dermoscopy head · S3 metadata head
                  S4 original PAD-UFES Model A · S5 kNN over training lesions
    evidence      Q, learned R, U, C (JS), M, OOD (Mahalanobis) per case
    fusion        iterative diagnosis-adaptive fusion (a, b, g, rho, T fitted on val)
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
from src.fusion.adaptive import (AdaptiveFusion, EvidenceManager, OODDetector,  # noqa: E402
                                 TeamBundle, knn_probs, norm_entropy, quality_score)

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


def main() -> int:
    nat = {s: load(NAT / f"{s}.npz") for s in ("val", "test")}
    spec = {s: load(SPEC / f"{s}.npz") for s in ("train", "val", "test")}
    for s in ("val", "test"):
        assert list(nat[s]["lesion_id"]) == list(spec[s]["lesion_id"]), f"{s} order mismatch"
    train = spec["train"]
    q_ref_c = np.percentile(train["q_clin"], [5, 95], axis=0)
    q_ref_d = np.percentile(train["q_derm"], [5, 95], axis=0)
    yv, yt = nat["val"]["y"], nat["test"]["y"]

    print("=" * 78)
    print("SPECIALIST TEAM + DEVIL'S ADVOCATE + ITERATIVE DIAGNOSIS-ADAPTIVE FUSION")
    print("=" * 78)
    report = {"specialists": {}, "fusion": {}}
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

        mgr = EvidenceManager(names=names).fit(Pv, Qv, Mv, yv)
        Rv, Rt = mgr.reliability(Pv, Qv, Mv), mgr.reliability(Pt, Qt, Mt)
        heads = [i for i, n in enumerate(names) if n in ("clinical", "dermoscopy", "metadata")]

        rows = {}
        rows["DS fusion (evidential heads)"] = nat["test"][f"{view}_prob"]
        rows["simple average (team)"] = Pt.mean(1)
        fh = AdaptiveFusion(names=[names[i] for i in heads]).fit_prior(Pv[:, heads], Mv[:, heads], yv, K)
        fh.fit_params(Pv[:, heads], Qv[:, heads] * Rv[:, heads], Mv[:, heads], yv, K)
        rows["adaptive fusion (heads only)"] = fh.fuse(Pt[:, heads], Qt[:, heads] * Rt[:, heads], Mt[:, heads])[0]
        ft = AdaptiveFusion(names=names).fit_prior(Pv, Mv, yv, K)
        ft.fit_params(Pv, Qv * Rv, Mv, yv, K)
        p_team_t, w_t, it_t = ft.fuse(Pt, Qt * Rt, Mt, return_trace=True)
        p_team_v, w_v = ft.fuse(Pv, Qv * Rv, Mv)
        rows["ADAPTIVE FUSION (full team)"] = p_team_t

        print(f"\n{'fusion method':32} {'acc':>6} {'macroF1':>8} {'balAcc':>7} {'malAUROC':>9}")
        for k, p in rows.items():
            r = metrics(yt, p)
            report["fusion"][f"{stage}:{k}"] = r
            print(f"{k:32} {r['acc']:6.3f} {r['macro_f1']:8.3f} {r['bal_acc']:7.3f} {r['mal_auroc']:9.3f}")
        print(f"fitted on val: a={ft.a} b={ft.b} g={ft.g} rho={ft.rho} T={ft.temperature:.2f}; "
              f"mean weights on test: "
              + ", ".join(f"{n} {w_t[:, i][Mt[:, i]].mean():.2f}" for i, n in enumerate(names)))
        print(f"iterations used: " + ", ".join(f"{k}:{(it_t == k).mean():.0%}" for k in range(4)))
        team[stage] = dict(names=names, mgr=mgr, fusion=ft, p_val=p_team_v, p_test=p_team_t,
                           w_test=w_t, R_test=Rt, Q_test=Qt, M_test=Mt, P_test=Pt)

    # ----------------------------------------------------------------- OOD
    ood = OODDetector().fit(np.hstack([train["z_img"], train["z_meta"]]), train["y"], K)
    ood.calibrate(np.hstack([spec["val"]["z_img"], spec["val"]["z_meta"]]), q=99.0)
    ood_t = ood.flag(np.hstack([spec["test"]["z_img"], spec["test"]["z_meta"]]))
    print(f"\nOOD (Mahalanobis, val p99 threshold): {ood_t.mean():.1%} of test lesions flagged")

    # --------------------------------------------------- confidence calibration
    from sklearn.isotonic import IsotonicRegression
    conf_v = np.concatenate([team["A"]["p_val"].max(1), team["B"]["p_val"].max(1)])
    hit_v = np.concatenate([team["A"]["p_val"].argmax(1) == yv, team["B"]["p_val"].argmax(1) == yv])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(conf_v, hit_v)
    for stage in ("A", "B"):
        c = iso.predict(team[stage]["p_test"].max(1))
        hit = team[stage]["p_test"].argmax(1) == yt
        print(f"calibrated confidence stage {stage}: mean shown {c.mean():.3f} vs actual "
              f"accuracy {hit.mean():.3f}")
    report["ood_test_rate"] = float(ood_t.mean())

    # ---------------------------------------------------------------- outputs
    OUT.mkdir(parents=True, exist_ok=True)
    for split, idx in (("val", "p_val"), ("test", "p_test")):
        d = dict(nat[split])
        d["no_dermo_prob"] = team["A"][idx].astype(np.float32)
        d["full_prob"] = team["B"][idx].astype(np.float32)
        z = np.hstack([spec[split]["z_img"], spec[split]["z_meta"]])
        d["ood"] = ood.flag(z)
        np.savez_compressed(OUT / f"{split}.npz", **d)

    import joblib
    bundle = TeamBundle(
        names=team["B"]["names"], names_A=team["A"]["names"], classes=MILK_CLASS_NAMES,
        manager_A=team["A"]["mgr"], manager_B=team["B"]["mgr"],
        fusion_A=team["A"]["fusion"], fusion_B=team["B"]["fusion"], ood=ood,
        knn_ref_A=np.hstack([train["z_img"], train["z_meta"]]).astype(np.float32),
        knn_ref_B=np.hstack([train["z_img"], train["z_derm"], train["z_meta"]]).astype(np.float32),
        knn_y=train["y"], knn_k=KNN_K, q_ref_clin=q_ref_c, q_ref_derm=q_ref_d,
        confidence_calibrator=iso, report=report)
    joblib.dump(bundle, OUT / "team_bundle.joblib")
    (OUT / "team_report.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nsaved -> {OUT}/(val|test).npz, team_bundle.joblib, team_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
