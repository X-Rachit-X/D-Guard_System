"""
scripts/run_eade_milk.py - The core experiment, in native mode on MILK10k.

Question: given a clinical photo and basic details, should the system
PROCEED, ACQUIRE a dermoscopy image, or DEFER to a human - and does deciding
this way beat simpler policies?

Two-stage protocol (the way a clinic would actually use it):

    Stage A   clinical photo + metadata          -> PROCEED / ACQUIRE / DEFER
    Stage B   (ACQUIRE cases only) + the REAL     -> PROCEED / DEFER
              dermoscopy image for that lesion

Every MILK10k lesion has both images, so Stage B uses the true dermoscopy, not a
simulated one, and the value of acquiring it is measured, not assumed.

Calibration uses the VALIDATION split only; the TEST split is scored once.

    conformal sets      APS, alpha=0.10, separately per stage
    conflict threshold  validation p90 of DS conflict, per stage
    LTT threshold       Learn-then-Test on 1-u, malignancy-miss loss
    acquisition value   h(stage-A evidence) -> expected cost saved by dermoscopy,
                        fitted on validation (the h_m estimator in docs/04)

Prerequisite:
    python scripts/milk_native_infer.py

Usage:
    python scripts/run_eade_milk.py
    python scripts/run_eade_milk.py --missed-cost 5      # cost sensitivity
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.calibration.conformal import ConformalCalibrator  # noqa: E402
from src.calibration.ltt import calibrate_threshold  # noqa: E402
from src.constants import (MILK_CLASS_NAMES, MILK_CLASS_FULL_NAMES,  # noqa: E402
                           milk_malignant_indices)
from src.decision.explain import explain_doctor, explain_patient  # noqa: E402
from src.decision.policy import Action, CostMatrix, ThreeActionPolicy  # noqa: E402
from src.eval.selective import acquisition_efficiency, augrc  # noqa: E402

K = len(MILK_CLASS_NAMES)
MAL = milk_malignant_indices()


def load(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def macro_f1(y, p):
    from sklearn.metrics import f1_score
    return float(f1_score(y, p, labels=list(range(K)), average="macro", zero_division=0))


def balanced_acc(y, p):
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y, p)) if len(y) else float("nan")


def is_missed_malignant(y, pred):
    mal = np.isin(y, MAL)
    return mal & ~np.isin(pred, MAL)


def features(d, view):
    """Stage-A evidence summary used by the acquisition-value estimator."""
    p = d[f"{view}_prob"]
    ent = -(p * np.log(np.clip(p, 1e-12, 1))).sum(1) / np.log(K)
    return np.column_stack([
        d[f"{view}_u"].ravel(), d[f"{view}_conflict"].ravel(),
        p.max(1), p[:, MAL].sum(1), ent,
        d["mod_image_u"].ravel(), d["mod_meta_u"].ravel(),
    ])


from src.calibration.binary import BinarySetCalibrator as _BinarySets  # noqa: E402


def BinarySetCalibrator(alpha):
    """Library calibrator bound to MILK10k's malignant classes."""
    return _BinarySets(alpha=alpha, malignant=MAL)


SET_MODE = {"mode": "binary"}


def stated_labels(pol, p):
    """The label the policy would state for each row, under its label_rule."""
    if pol.label_rule == "argmax":
        return p.argmax(1)
    return (p @ pol._L).argmin(1)


def calibrate_stage(val, view, alpha_set, alpha_risk, delta, policy_kw):
    p = val[f"{view}_prob"]
    y = val["y"]
    if SET_MODE["mode"] == "binary":
        cc = BinarySetCalibrator(alpha_set).fit(p, y)
    else:
        cc = ConformalCalibrator(alpha=alpha_set, method="aps").fit(p, y)
    conflict_thr = float(np.quantile(val[f"{view}_conflict"], 0.90))
    score = 1.0 - val[f"{view}_u"].ravel()
    pol = ThreeActionPolicy(malignant_classes=MAL, num_classes=K, **policy_kw)
    pred = stated_labels(pol, p)
    loss = is_missed_malignant(y, pred).astype(float)
    ltt = calibrate_threshold(score, loss, alpha=alpha_risk, delta=delta,
                              min_coverage=0.05)
    return cc, conflict_thr, ltt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--native", default="outputs/milk_native")
    ap.add_argument("--alpha-set", type=float, default=0.10)
    ap.add_argument("--alpha-risk", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--missed-cost", type=float, default=10.0,
                    help="cost of calling a cancer benign. 10 is the pre-registered "
                         "value; 5 is the balanced operating point the demo uses "
                         "(see docs/14_results.md)")
    ap.add_argument("--no-ltt", action="store_true",
                    help="do not apply the LTT gate (report it only)")
    ap.add_argument("--label-rule", choices=["argmax", "bayes"], default="argmax",
                    help="argmax: state the most likely diagnosis (default). bayes: "
                         "state the lowest-cost label, which on this mostly-malignant "
                         "population calls almost everything malignant")
    ap.add_argument("--no-quality-gate", action="store_true",
                    help="do not downgrade HIGH answers on photos with a severe "
                         "quality issue (the app applies this gate)")
    ap.add_argument("--data", default="milk10k", help="dataset root (for the photos)")
    ap.add_argument("--defer-mode", choices=["minimal", "standard"], default="minimal",
                    help="minimal: blocked cases are ANSWERED in a LOW-confidence tier "
                         "that requires a doctor's confirmation; only conflict defers "
                         "(default). standard: blocked cases are deferred")
    ap.add_argument("--gate", choices=["reassurance", "symmetric"], default="reassurance",
                    help="reassurance: block only benign statements while malignancy "
                         "is still possible (default). symmetric: block any set "
                         "spanning benign and malignant")
    ap.add_argument("--set-mode", choices=["binary", "multiclass"], default="binary",
                    help="binary: conformal set over benign/malignant, the quantity "
                         "the guarantee is pre-registered on (default). multiclass: "
                         "11-class APS set")
    ap.add_argument("--out", default="outputs/eade_milk")
    ap.add_argument("--show", type=int, default=3, help="print N example messages")
    args = ap.parse_args()

    SET_MODE["mode"] = args.set_mode
    nat = ROOT / args.native
    if not (nat / "test.npz").exists():
        print("ERROR: run  python scripts/milk_native_infer.py  first.")
        return 1
    val, test = load(nat / "val.npz"), load(nat / "test.npz")
    yv, yt = val["y"], test["y"]

    cost = CostMatrix(missed_malignant=args.missed_cost)
    policy_kw = {"cost": cost, "label_rule": args.label_rule,
                 "ambiguity_gate": args.gate, "defer_mode": args.defer_mode}

    print("=" * 76)
    print("D-GUARD on MILK10k - native evidential model, two-stage acquisition")
    print("=" * 76)
    print(f"calibration (val) {len(yv)}   test {len(yt)}   classes {K}   "
          f"malignant prevalence test {np.isin(yt, MAL).mean():.3f}")
    print(f"cost: miss {cost.missed_malignant}  false alarm {cost.false_alarm}  "
          f"dermoscopy {cost.acquire['dermoscopy_image']}  defer {cost.defer}")
    print(f"label rule: {args.label_rule}   conformal set: {args.set_mode}   "
          f"gate: {args.gate}")

    # ---------------- calibration ------------------------------------------
    ccA, cthrA, lttA = calibrate_stage(val, "no_dermo", args.alpha_set,
                                       args.alpha_risk, args.delta, policy_kw)
    ccB, cthrB, lttB = calibrate_stage(val, "full", args.alpha_set,
                                       args.alpha_risk, args.delta, policy_kw)
    print()
    print("CALIBRATION")
    for tag, cc, thr, ltt in (("A clinical+meta", ccA, cthrA, lttA),
                              ("B +dermoscopy  ", ccB, cthrB, lttB)):
        lam = f"{ltt.lambda_hat:.3f}" if ltt.controlled else "NOT CONTROLLED"
        qtxt = (f"q_benign {cc.q[0]:.3f} q_malig {cc.q[1]:.3f}"
                if hasattr(cc, "q") and cc.q else f"qhat {cc.qhat:.3f}")
        print(f"  stage {tag}  {qtxt}  conflict p90 {thr:.4f}  "
              f"LTT lambda {lam}")
        if not ltt.controlled and ltt.note:
            print(f"      {ltt.note}")

    # ---------------- acquisition-value estimator h --------------------------
    from sklearn.ensemble import GradientBoostingRegressor
    polA = ThreeActionPolicy(malignant_classes=MAL, num_classes=K, **policy_kw)
    L = polA._L

    def realised_cost(d, view):
        pred = stated_labels(polA, d[f"{view}_prob"])
        return L[d["y"], pred], pred

    cA_v, _ = realised_cost(val, "no_dermo")
    cB_v, _ = realised_cost(val, "full")
    benefit_v = cA_v - cB_v
    h = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                  subsample=0.8, random_state=42)
    h.fit(features(val, "no_dermo"), benefit_v)

    cA_t, predA_t = realised_cost(test, "no_dermo")
    cB_t, predB_t = realised_cost(test, "full")
    benefit_t = cA_t - cB_t
    hat_t = h.predict(features(test, "no_dermo"))
    print()
    print("ACQUISITION-VALUE ESTIMATOR h  (fitted on val, scored on test)")
    print(f"  mean realised benefit of dermoscopy  val {benefit_v.mean():+.3f}  "
          f"test {benefit_t.mean():+.3f}")
    print(f"  corr(predicted, realised) on test    "
          f"{np.corrcoef(hat_t, benefit_t)[0, 1]:+.3f}")

    # ---------------- the policy, case by case ------------------------------
    def policy_for(stage):
        thr = cthrA if stage == "A" else cthrB
        ltt = lttA if stage == "A" else lttB
        lam = ltt.lambda_hat if (ltt.controlled and not args.no_ltt) else None
        return ThreeActionPolicy(malignant_classes=MAL, num_classes=K,
                                 conflict_threshold=thr, lambda_hat=lam, **policy_kw)

    pA, pB = policy_for("A"), policy_for("B")
    setsA = ccA.predict_sets(test["no_dermo_prob"])
    setsB = ccB.predict_sets(test["full_prob"])

    final_action, final_pred, stage_of, decisions = [], [], [], []
    acquired = np.zeros(len(yt), dtype=bool)
    for i in range(len(yt)):
        dA = pA.decide(test["no_dermo_prob"][i],
                       ignorance=float(test["no_dermo_u"][i]),
                       conflict=float(test["no_dermo_conflict"][i]),
                       prediction_set=setsA[i],
                       acquirable=["dermoscopy_image"],
                       predicted_benefit={"dermoscopy_image": float(hat_t[i])})
        d = dA
        if dA.action == Action.ACQUIRE:
            acquired[i] = True
            d = pB.decide(test["full_prob"][i],
                          ignorance=float(test["full_u"][i]),
                          conflict=float(test["full_conflict"][i]),
                          prediction_set=setsB[i],
                          acquirable=[], n_acquisitions_so_far=1)
        decisions.append(d)
        final_action.append(d.action.value)
        final_pred.append(d.prediction if d.action == Action.PROCEED else -1)
        stage_of.append("B" if acquired[i] else "A")

    # ---------------- quality gate (identical to src/agents/pipeline.py) ----------
    n_quality = 0
    if not args.no_quality_gate:
        from PIL import Image

        from src.agents.quality import ImageQualityAgent
        from src.decision.policy import DeferReason
        cache = ROOT / "outputs" / "milk_rag"
        img_dir = ROOT / args.data / "images"
        qa_c = ImageQualityAgent.fit([], cache=cache / "quality_thresholds.json")
        qa_d = ImageQualityAgent.fit([], cache=cache / "quality_thresholds_dermoscopy.json")
        for i, d in enumerate(decisions):
            if d.action != Action.PROCEED or d.tier == "LOW":
                continue
            bad = bool(qa_c.check(Image.open(img_dir / str(test["clinical_file"][i]))).severe)
            if acquired[i]:
                bad |= bool(qa_d.check(Image.open(img_dir / str(test["dermoscopy_file"][i]))).severe)
            if bad:
                d.tier, d.low_reason = "LOW", DeferReason.QUALITY
                n_quality += 1
        print(f"\nquality gate: {n_quality} HIGH answers downgraded to LOW "
              f"(severe photo-quality issue)")

    # ---------------- OOD gate (Devil's Advocate, Mahalanobis; from run_team.py) ----
    if "ood" in test:
        from src.decision.policy import DeferReason as _DR
        n_ood = 0
        for i, d in enumerate(decisions):
            if d.action == Action.PROCEED and d.tier != "LOW" and bool(test["ood"][i]):
                d.tier, d.low_reason = "LOW", _DR.OUT_OF_DISTRIBUTION
                n_ood += 1
        print(f"OOD gate: {n_ood} HIGH answers downgraded to LOW (out of distribution)")

    act = np.array(final_action)
    pred = np.array(final_pred)
    tier = np.array([d.tier or "" for d in decisions])
    answered = act == Action.PROCEED.value
    # "proceed" = the HIGH tier: answers that passed every gate. All baseline
    # comparisons use it, so they are identical in standard and minimal mode.
    proceed = answered & (tier != "LOW")
    low = answered & (tier == "LOW")
    n = len(yt)

    # Realised cost per case under the cost matrix: PROCEED pays the error
    # cost, DEFER pays the review cost, and any acquisition adds its cost.
    case_cost = np.where(proceed, L[yt, np.where(pred < 0, 0, pred)], cost.defer)
    case_cost = case_cost + acquired * cost.acquire["dermoscopy_image"]

    miss = is_missed_malignant(yt[proceed], pred[proceed])
    print()
    print("=" * 76)
    print("TEST RESULTS - D-GUARD")
    print("=" * 76)
    print(f"  PROCEED  {proceed.sum():4d}  {proceed.mean():.3f}   "
          f"HIGH tier (stage A {(proceed & ~acquired).sum()}, "
          f"after dermoscopy {(proceed & acquired).sum()})")
    print(f"  ACQUIRE  {acquired.sum():4d}  {acquired.mean():.3f}   "
          f"dermoscopy requested")
    print(f"  DEFER    {(act == 'DEFER').sum():4d}  {(act == 'DEFER').mean():.3f}")
    print()
    print(f"  accuracy among PROCEED         {(pred[proceed] == yt[proceed]).mean():.4f}")
    print(f"  balanced acc among PROCEED     {balanced_acc(yt[proceed], pred[proceed]):.4f}")
    print(f"  malignancy miss rate (PROCEED) {miss.mean() if proceed.any() else 0:.4f}   "
          f"[{miss.sum()} of {proceed.sum()}]")
    print(f"  mean cost per case             {case_cost.mean():.4f}")
    urgent = sum(1 for d in decisions if d.requires_confirmation and d.urgency == "URGENT")
    print(f"  needing a doctor, flagged URGENT {urgent}")

    low_miss = is_missed_malignant(yt[low], pred[low]) if low.any() else np.array([])
    print()
    print(f"CONFIDENCE TIERS  (defer mode: {args.defer_mode})")
    print(f"  answered, HIGH confidence  {proceed.sum():4d}  {proceed.mean():.3f}   "
          f"accuracy {(pred[proceed] == yt[proceed]).mean():.3f}   "
          f"missed cancers {miss.mean() if proceed.any() else 0:.3f}")
    if low.any():
        print(f"  answered, LOW confidence   {low.sum():4d}  {low.mean():.3f}   "
              f"accuracy {(pred[low] == yt[low]).mean():.3f}   "
              f"missed cancers {low_miss.mean():.3f}   "
              f"top-3 hit {np.mean([yt[i] in [c for c, _ in decisions[i].differential] for i in np.flatnonzero(low)]):.3f}")
    print(f"  deferred (conflict only)   {(act == 'DEFER').sum():4d}  {(act == 'DEFER').mean():.3f}")
    print(f"  => the patient gets an answer in {answered.mean():.1%} of cases; "
          f"{proceed.mean():.1%} are calibrated HIGH-confidence answers")

    # ---------------- baselines ---------------------------------------------
    print()
    print("BASELINES (same test lesions, same cost matrix)")
    rows = []

    def add(name, proc_mask, preds, acq_mask=None):
        acq_mask = np.zeros(n, bool) if acq_mask is None else acq_mask
        c = np.where(proc_mask, L[yt, np.where(preds < 0, 0, preds)], cost.defer)
        c = c + acq_mask * cost.acquire["dermoscopy_image"]
        m = is_missed_malignant(yt[proc_mask], preds[proc_mask])
        ben = ~np.isin(yt[proc_mask], MAL)
        fa = ben & np.isin(preds[proc_mask], MAL)
        any_ = proc_mask.any()
        rows.append({
            "policy": name, "coverage": float(proc_mask.mean()),
            "acquire_rate": float(acq_mask.mean()),
            "accuracy": float((preds[proc_mask] == yt[proc_mask]).mean()) if any_ else float("nan"),
            "balanced_acc": balanced_acc(yt[proc_mask], preds[proc_mask]) if any_ else float("nan"),
            "miss_rate": float(m.mean()) if any_ else 0.0,
            "false_alarm_rate": float(fa.sum() / max(ben.sum(), 1)),
            "says_malignant": float(np.isin(preds[proc_mask], MAL).mean()) if any_ else 0.0,
            "mean_cost": float(c.mean()),
        })

    allp = np.ones(n, bool)
    add("B1 always answer, photo only", allp, predA_t)
    add("B2 always take dermoscopy, answer", allp, predB_t, allp)
    cov = proceed.mean()
    for tag, view, pr in (("photo only", "no_dermo", predA_t),
                          ("with dermoscopy", "full", predB_t)):
        conf = test[f"{view}_prob"].max(1)
        thr = np.quantile(conf, 1 - cov) if cov > 0 else 1.0
        add(f"B3 max-softmax abstain, {tag}, matched coverage", conf >= thr, pr,
            allp if view == "full" else None)
    rate = acquired.mean()
    rng = np.random.default_rng(42)
    rand_acq = np.zeros(n, bool)
    rand_acq[rng.choice(n, int(round(rate * n)), replace=False)] = True
    entA = -(test["no_dermo_prob"] * np.log(np.clip(test["no_dermo_prob"], 1e-12, 1))).sum(1)
    ent_acq = np.zeros(n, bool)
    ent_acq[np.argsort(-entA)[: int(round(rate * n))]] = True
    for tag, acq in (("random", rand_acq), ("highest entropy", ent_acq)):
        preds = np.where(acq, predB_t, predA_t)
        add(f"B5 acquire {tag} at same rate, then answer", allp, preds, acq)
    add("D-GUARD (this work)", proceed, pred, acquired)

    print(f"  {'policy':50} {'cover':>5} {'acq':>4} {'acc':>5} {'balAc':>5} "
          f"{'miss':>5} {'FA':>5} {'saysM':>5} {'cost':>6}")
    for r in rows:
        print(f"  {r['policy'][:50]:50} {r['coverage']:5.2f} {r['acquire_rate']:4.2f} "
              f"{r['accuracy']:5.3f} {r['balanced_acc']:5.3f} {r['miss_rate']:5.3f} "
              f"{r['false_alarm_rate']:5.3f} {r['says_malignant']:5.2f} {r['mean_cost']:6.3f}")
    print("  miss = malignant called benign, among answered. FA = benign called "
          "malignant, among answered benign.")
    print("  saysM = share of answers that name a malignant diagnosis "
          f"(true prevalence {np.isin(yt, MAL).mean():.2f}).")

    # ---------------- acquisition efficiency --------------------------------
    print()
    print("ACQUISITION EFFICIENCY  AE = (policy - random) / (oracle - random)")
    ae_h = acquisition_efficiency(benefit_t, hat_t, rate=max(rate, 0.05))
    ae_e = acquisition_efficiency(benefit_t, entA, rate=max(rate, 0.05))
    print(f"  at rate {max(rate, 0.05):.2f}:  h estimator AE {ae_h.efficiency:+.3f}   "
          f"entropy AE {ae_e.efficiency:+.3f}   (1 = oracle, 0 = random)")

    # ---------------- ranking metrics ---------------------------------------
    print()
    print("RANKING (AUGRC, lower is better; full view)")
    pf = test["full_prob"].argmax(1)
    au_u = augrc(yt, pf, 1 - test["full_u"].ravel())
    au_p = augrc(yt, pf, test["full_prob"].max(1))
    print(f"  1-u  {au_u:.4f}     max-softmax {au_p:.4f}")

    # ---------------- examples ----------------------------------------------
    if args.show:
        print()
        print("EXAMPLE MESSAGES")
        shown = set()
        for i, d in enumerate(decisions):
            key = d.action.value
            if key in shown:
                continue
            shown.add(key)
            print()
            print(f"--- lesion {test['lesion_id'][i]}  true "
                  f"{MILK_CLASS_NAMES[yt[i]]}  stage {stage_of[i]} ---")
            print("[doctor]")
            print(explain_doctor(d, MILK_CLASS_NAMES, MILK_CLASS_FULL_NAMES))
            print("[patient]")
            print(explain_patient(d, MILK_CLASS_NAMES, MILK_CLASS_FULL_NAMES))
            if len(shown) >= args.show:
                break

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "n_calibration": int(len(yv)), "n_test": int(n),
        "cost": {"missed_malignant": cost.missed_malignant,
                 "false_alarm": cost.false_alarm,
                 "acquire_dermoscopy": cost.acquire["dermoscopy_image"],
                 "defer": cost.defer},
        "calibration": {
            "stage_A": {"qhat": ccA.qhat, "conflict_p90": cthrA,
                        "ltt": lttA.to_dict()},
            "stage_B": {"qhat": ccB.qhat, "conflict_p90": cthrB,
                        "ltt": lttB.to_dict()},
        },
        "tiers": {
            "defer_mode": args.defer_mode,
            "answered_rate": float(answered.mean()),
            "high_rate": float(proceed.mean()),
            "low_rate": float(low.mean()),
            "defer_rate": float((act == "DEFER").mean()),
            "low_accuracy": float((pred[low] == yt[low]).mean()) if low.any() else None,
            "low_miss_rate": float(low_miss.mean()) if low.any() else None,
            "low_top3": float(np.mean([yt[i] in [c for c, _ in decisions[i].differential]
                                       for i in np.flatnonzero(low)])) if low.any() else None,
            "answered_accuracy": float((pred[answered] == yt[answered]).mean()) if answered.any() else None,
        },
        "dguard": {
            "proceed_rate": float(proceed.mean()),
            "acquire_rate": float(acquired.mean()),
            "defer_rate": float((act == "DEFER").mean()),
            "accuracy_proceeded": float((pred[proceed] == yt[proceed]).mean()) if proceed.any() else None,
            "miss_rate_proceeded": float(miss.mean()) if proceed.any() else None,
            "mean_cost": float(case_cost.mean()),
        },
        "baselines": rows,
        "acquisition_efficiency": {"h": ae_h.to_dict(), "entropy": ae_e.to_dict()},
        "augrc": {"one_minus_u": au_u, "max_softmax": au_p},
        "h_corr_test": float(np.corrcoef(hat_t, benefit_t)[0, 1]),
    }
    # -------- diagnosis-conditioned modality reliability ------------------
    # The old system's "diagnosis-adaptive prior" was a hand-typed 6x3 matrix
    # (e.g. MEL: image 0.96, metadata 0.75). This is the same idea MEASURED:
    # for lesions whose true diagnosis is c, how often does modality m on its
    # own name c? Fitted on validation; shown to clinicians, never used to
    # change a probability.
    rel = {}
    for m in ("image", "dermoscopy", "meta"):
        b = val[f"mod_{m}_belief"]
        hit = b.argmax(1) == yv
        rel[m] = [float(hit[yv == c].mean()) if (yv == c).any() else None
                  for c in range(K)]
    report["modality_reliability_by_class"] = {
        "classes": MILK_CLASS_NAMES, **rel,
        "n_per_class": [int((yv == c).sum()) for c in range(K)],
    }
    print()
    print("MODALITY RELIABILITY BY TRUE CLASS (validation; old 'diagnosis prior', measured)")
    print(f"  {'class':8} {'n':>4} {'clinical':>9} {'dermosc.':>9} {'metadata':>9}")
    for c in range(K):
        n_c = int((yv == c).sum())
        if n_c == 0:
            continue
        print(f"  {MILK_CLASS_NAMES[c]:8} {n_c:4d} {rel['image'][c]:9.2f} "
              f"{rel['dermoscopy'][c]:9.2f} {rel['meta'][c]:9.2f}")

    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))

    # -------- calibration bundle for the app / native engine ---------------
    import joblib
    bundle = {
        "version": 1,
        "classes": MILK_CLASS_NAMES,
        "malignant": MAL,
        "cost": cost,
        "label_rule": args.label_rule,
        "ambiguity_gate": args.gate,
        "defer_mode": args.defer_mode,
        "set_mode": args.set_mode,
        "stage_A": {"sets": ccA, "conflict_threshold": cthrA,
                    "lambda_hat": lttA.lambda_hat if not args.no_ltt else None},
        "stage_B": {"sets": ccB, "conflict_threshold": cthrB,
                    "lambda_hat": lttB.lambda_hat if not args.no_ltt else None},
        "acquisition_value_model": h,
        "modality_reliability_by_class": report["modality_reliability_by_class"],
        "test_summary": report["dguard"],
    }
    joblib.dump(bundle, out / "calibration.joblib")
    print(f"calibration bundle -> {out / 'calibration.joblib'}")
    print(f"\nsaved -> {out / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
