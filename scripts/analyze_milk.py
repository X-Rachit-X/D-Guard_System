"""
scripts/analyze_milk.py - Statistics the paper needs, from saved per-case results.

    1  bootstrap 95% CIs (2000 resamples) + paired tests:
         - D-GUARD vs max-softmax abstention, missed cancers at matched coverage
         - adaptive team fusion vs Dempster-Shafer heads, macro-F1
    2  fairness by skin tone (MILK10k skin_tone_class 0-5)
    3  a LEARNED-DEFERRAL baseline (L2D-style rejector): a classifier trained on
       validation to predict the model's own errors; abstains on the cases it
       expects to get wrong, at the same coverage as D-GUARD
    4  decision-curve net benefit for "refer as possibly malignant"

    python scripts/analyze_milk.py
Prerequisites: run_team.py and run_eade_milk.py --native outputs/milk_team.
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
from sklearn.metrics import f1_score  # noqa: E402

from src.constants import MILK_CLASS_NAMES, milk_malignant_indices  # noqa: E402

K = len(MILK_CLASS_NAMES)
MAL = milk_malignant_indices()
B = 2000
rng = np.random.default_rng(2026)


def load(p):
    d = np.load(p, allow_pickle=True)
    return {k: d[k] for k in d.files}


def missed(y, pred):
    return np.isin(y, MAL) & ~np.isin(pred, MAL)


def ci(vals):
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)]


def main() -> int:
    cases = load(ROOT / "outputs/eade_team/cost_5/cases.npz")
    team = {s: load(ROOT / f"outputs/milk_team/{s}.npz") for s in ("val", "test")}
    nat_t = load(ROOT / "outputs/milk_native/test.npz")
    y = cases["y"]
    n = len(y)
    act, tier, pred = cases["action"], cases["tier"], cases["pred"]
    high = (act == "PROCEED") & (tier != "LOW")
    answered = act == "PROCEED"
    cov = high.mean()
    out = {}

    # ---- baselines at matched coverage (stage-A probabilities) --------------
    pA = team["test"]["no_dermo_prob"]
    predA = pA.argmax(1)
    thr = np.quantile(pA.max(1), 1 - cov)
    b3 = pA.max(1) >= thr

    # learned rejector: predict own error on validation, keep the most-likely-right
    from sklearn.ensemble import GradientBoostingClassifier

    def feats(d):
        p = d["no_dermo_prob"]
        s = np.sort(p, 1)
        ent = -(p * np.log(np.clip(p, 1e-12, 1))).sum(1) / np.log(K)
        return np.column_stack([p.max(1), s[:, -1] - s[:, -2], ent, p[:, MAL].sum(1),
                                d["no_dermo_u"].ravel(), d["no_dermo_conflict"].ravel()])

    yv = team["val"]["y"]
    err_v = (team["val"]["no_dermo_prob"].argmax(1) != yv).astype(int)
    rej = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                     random_state=0).fit(feats(team["val"]), err_v)
    risk = rej.predict_proba(feats(team["test"]))[:, 1]
    b4 = risk <= np.quantile(risk, cov)

    print("=" * 76)
    print(f"1. BOOTSTRAP 95% CIs  ({B} resamples of the {n} test lesions)")
    print("=" * 76)

    def stats(idx):
        yy, hh, pp = y[idx], high[idx], pred[idx]
        r = {"high_cov": hh.mean(), "answered": answered[idx].mean(),
             "high_acc": (pp[hh] == yy[hh]).mean() if hh.any() else np.nan,
             "high_miss": missed(yy[hh], pp[hh]).mean() if hh.any() else np.nan}
        bb = b3[idx]
        r["b3_miss"] = missed(yy[bb], predA[idx][bb]).mean() if bb.any() else np.nan
        r["b3_acc"] = (predA[idx][bb] == yy[bb]).mean() if bb.any() else np.nan
        b4b = b4[idx]
        r["b4_miss"] = missed(yy[b4b], predA[idx][b4b]).mean() if b4b.any() else np.nan
        r["b4_acc"] = (predA[idx][b4b] == yy[b4b]).mean() if b4b.any() else np.nan
        r["f1_team"] = f1_score(yy, team["test"]["full_prob"][idx].argmax(1),
                                labels=list(range(K)), average="macro", zero_division=0)
        r["f1_ds"] = f1_score(yy, nat_t["full_prob"][idx].argmax(1),
                              labels=list(range(K)), average="macro", zero_division=0)
        return r

    point = stats(np.arange(n))
    boots = [stats(rng.integers(0, n, n)) for _ in range(B)]
    keys = ["high_cov", "answered", "high_acc", "high_miss", "b3_miss", "b3_acc",
            "b4_miss", "b4_acc", "f1_team", "f1_ds"]
    out["ci"] = {}
    for k in keys:
        vals = np.array([b[k] for b in boots])
        out["ci"][k] = {"point": float(point[k]), "ci95": ci(vals[~np.isnan(vals)])}
        print(f"  {k:10} {point[k]:.3f}  [{out['ci'][k]['ci95'][0]:.3f}, "
              f"{out['ci'][k]['ci95'][1]:.3f}]")

    for name, a, b_ in (("missed: B3 - D-GUARD", "b3_miss", "high_miss"),
                        ("missed: learned rejector - D-GUARD", "b4_miss", "high_miss"),
                        ("macro-F1: team - DS", "f1_team", "f1_ds")):
        d = np.array([bb[a] - bb[b_] for bb in boots])
        d = d[~np.isnan(d)]
        p_one = float((d <= 0).mean())
        out.setdefault("paired", {})[name] = {"diff": float(point[a] - point[b_]),
                                              "ci95": ci(d), "p_one_sided": p_one}
        print(f"  {name:36} diff {point[a] - point[b_]:+.3f}  CI [{ci(d)[0]:+.3f}, "
              f"{ci(d)[1]:+.3f}]  one-sided p = {p_one:.4f}")

    # ---- 2. fairness -----------------------------------------------------------
    print()
    print("=" * 76)
    print("2. BY SKIN TONE (MILK10k skin_tone_class)")
    print("=" * 76)
    tone = cases["skin_tone"].astype(str)
    out["skin_tone"] = {}
    print(f"  {'tone':5} {'n':>4} {'HIGH':>6} {'HIGH-acc':>9} {'HIGH-miss':>10} "
          f"{'answered-acc':>13}")
    for t in sorted(set(tone)):
        m = tone == t
        hh = high & m
        r = {"n": int(m.sum()), "high_cov": float(hh.sum() / max(m.sum(), 1)),
             "high_acc": float((pred[hh] == y[hh]).mean()) if hh.any() else None,
             "high_miss": float(missed(y[hh], pred[hh]).mean()) if hh.any() else None,
             "answered_acc": float((pred[answered & m] == y[answered & m]).mean())
             if (answered & m).any() else None}
        out["skin_tone"][t] = r
        fmt = lambda v: "   n/a" if v is None else f"{v:6.3f}"  # noqa: E731
        print(f"  {t:5} {r['n']:4d} {r['high_cov']:6.2f} {fmt(r['high_acc']):>9} "
              f"{fmt(r['high_miss']):>10} {fmt(r['answered_acc']):>13}"
              + ("   (too few to judge)" if r["n"] < 30 else ""))

    # ---- 3. learned deferral (summary) -----------------------------------------
    print()
    print("=" * 76)
    print(f"3. LEARNED-DEFERRAL BASELINE at the same {cov:.1%} coverage")
    print("=" * 76)
    for name, mask, pr in (("B3 max-softmax threshold", b3, predA),
                           ("B4 learned rejector (L2D-style)", b4, predA),
                           ("D-GUARD HIGH tier", high, pred)):
        print(f"  {name:32} acc {(pr[mask] == y[mask]).mean():.3f}  missed cancers "
              f"{missed(y[mask], pr[mask]).mean():.3f}")
    out["learned_deferral"] = {
        "b4_acc": float((predA[b4] == y[b4]).mean()),
        "b4_miss": float(missed(y[b4], predA[b4]).mean())}

    # ---- 4. net benefit -----------------------------------------------------------
    print()
    print("=" * 76)
    print("4. DECISION CURVE - net benefit of referring as 'possibly malignant'")
    print("=" * 76)
    ym = np.isin(y, MAL)
    p_mal = cases["p_malignant"]
    # D-GUARD refers everything except a HIGH-confidence benign answer
    dg_refer = ~(high & ~np.isin(pred, MAL))
    print(f"  {'threshold':>9} {'refer all':>10} {'p_mal>=t':>10} {'D-GUARD':>10}")
    out["net_benefit"] = {}
    for t in (0.1, 0.2, 0.3, 0.4, 0.5):
        w = t / (1 - t)

        def nb(refer):
            return (refer & ym).sum() / n - (refer & ~ym).sum() / n * w
        row = {"refer_all": nb(np.ones(n, bool)), "model": nb(p_mal >= t),
               "dguard": nb(dg_refer)}
        out["net_benefit"][str(t)] = {k: float(v) for k, v in row.items()}
        print(f"  {t:9.1f} {row['refer_all']:10.3f} {row['model']:10.3f} {row['dguard']:10.3f}")

    dest = ROOT / "outputs" / "analysis"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "analysis.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {dest / 'analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
