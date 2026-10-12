"""
scripts/run_shift_derm7pt.py - External test on Derm7pt, WITH dermoscopy.

PAD-UFES-20 (run_shift_pad.py) only tests stage A: it has no dermoscopy. Derm7pt
(Kawahara et al., IEEE JBHI 2019; images (c) Dr. G. Argenziano, research use,
not redistributable) has a clinical photo AND a dermoscopy image for every
lesion, so the whole D-GUARD loop can be run on a different population, clinic
and decade - including the "ask for dermoscopy" step. Nothing is refitted.

  * Diagnoses map onto MILK10k: every nevus subtype -> NV, every melanoma -> MEL,
    seborrheic keratosis / lentigo / melanosis -> BKL, BCC -> BCC,
    dermatofibroma -> DF, vascular lesion -> VASC; "miscellaneous" is dropped.
  * Metadata: sex and body site. Derm7pt has no age, no skin tone, no MONET.
  * Flow per lesion, as in the app:
        stage A  photo + details  -> HIGH / LOW / DEFER, or ACQUIRE dermoscopy
        stage B  (only if ACQUIRE) photo + dermoscopy + details -> final answer
    plus "always stage B" for reference.
  * Quality gate as in the app (severe photo issue -> no HIGH).

    python scripts/run_shift_derm7pt.py --data dataset/release_v0
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score  # noqa: E402


def to_milk(dx: str):
    dx = dx.lower()
    if "melanoma" in dx:
        return "MEL"
    if "nevus" in dx:
        return "NV"
    if dx in ("seborrheic keratosis", "lentigo", "melanosis"):
        return "BKL"
    return {"basal cell carcinoma": "BCC", "dermatofibroma": "DF",
            "vascular lesion": "VASC"}.get(dx)


SITE = {"back": "trunk", "abdomen": "trunk", "chest": "trunk", "buttocks": "trunk",
        "upper limbs": "upper_extremity", "lower limbs": "lower_extremity",
        "head neck": "head_neck_face"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "dataset" / "release_v0"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="outputs/shift_derm7pt/report.json")
    args = ap.parse_args()

    from src.agents.quality import ImageQualityAgent
    from src.native import NativeEngine

    eng = NativeEngine()
    if eng.team is None:
        print("ERROR: team bundle not active - run scripts/run_team.py first")
        return 1
    qa = ImageQualityAgent.fit([], cache=ROOT / "outputs/milk_rag/quality_thresholds.json")
    qd = ImageQualityAgent.fit([], cache=ROOT / "outputs/milk_rag/quality_thresholds_dermoscopy.json")

    data = Path(args.data)
    df = pd.read_csv(data / "meta" / "meta.csv")
    df["milk"] = df["diagnosis"].map(to_milk)
    df = df[df["milk"].notna()].reset_index(drop=True)
    if args.limit:
        df = df.sample(n=args.limit, random_state=0).reset_index(drop=True)
    mal = set(eng.malignant)
    print(f"Derm7pt: {len(df)} lesions -> " + ", ".join(
        f"{k} {v}" for k, v in df["milk"].value_counts().items()), flush=True)

    rows = []
    for i, r in enumerate(df.itertuples(index=False)):
        clin = Image.open(data / "images" / r.clinic).convert("RGB")
        derm = Image.open(data / "images" / r.derm).convert("RGB")
        meta = {"sex": r.sex if r.sex in ("male", "female") else None,
                "site": SITE.get(str(r.location)),
                "anatom_site_general": SITE.get(str(r.location))}
        y = eng.classes.index(r.milk)
        severe_c = bool(qa.check(clin).severe)
        severe_d = bool(qd.check(derm).severe)

        a = eng.assess(clin, None, meta)
        b = eng.assess(clin, derm, meta)

        def tier_of(res, severe):
            d = res.decision
            t = d.tier or ""
            if d.action.value == "PROCEED" and t != "LOW" and severe:
                t = "LOW"
            return d.action.value, t, d.tentative, np.array(res.probs)

        actA, tierA, predA, pA = tier_of(a, severe_c)
        actB, tierB, predB, pB = tier_of(b, severe_c or severe_d)
        # the D-GUARD flow: dermoscopy only when stage A asked for it
        if actA == "ACQUIRE":
            act, tier, pred, p, stage = actB, tierB, predB, pB, "B"
        else:
            act, tier, pred, p, stage = actA, tierA, predA, pA, "A"
        rows.append(dict(y=y, act=act, tier=tier, pred=pred, p=p, stage=stage,
                         actB=actB, tierB=tierB, predB=predB, pB=pB, pA=pA,
                         u=a.ignorance, ood=bool(a.team.get("ood")),
                         severe=severe_c, difficulty=r.level_of_diagnostic_difficulty))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(df)}", flush=True)

    y = np.array([r["y"] for r in rows])
    rep = {"n": len(y), "classes": sorted({eng.classes[c] for c in y}),
           "malignant_share": float(np.isin(y, list(mal)).mean())}

    def summarise(prefix, act, tier, pred, P):
        act, tier, pred = np.array(act), np.array(tier), np.array(pred)
        high = (act == "PROCEED") & (tier != "LOW")
        ans = act == "PROCEED"
        labels = sorted(set(y))
        miss = lambda m: (float((np.isin(y[m], list(mal)) & ~np.isin(pred[m], list(mal))).mean())  # noqa: E731
                          if m.any() else None)
        conf = P.max(1)
        cov = high.mean()
        b3 = conf >= np.quantile(conf, 1 - cov) if cov > 0 else np.zeros(len(y), bool)
        ptop = P.argmax(1)
        out = {
            "accuracy_all": float((ptop == y).mean()),
            "macro_f1": float(f1_score(y, ptop, labels=labels, average="macro", zero_division=0)),
            "mal_auroc": float(roc_auc_score(np.isin(y, list(mal)), P[:, list(mal)].sum(1))),
            "high": float(cov), "answered": float(ans.mean()),
            "defer": float((act == "DEFER").mean()), "acquire": float((act == "ACQUIRE").mean()),
            "high_accuracy": float((pred[high] == y[high]).mean()) if high.any() else None,
            "high_miss": miss(high),
            "b3_miss": (float((np.isin(y[b3], list(mal)) & ~np.isin(ptop[b3], list(mal))).mean())
                        if b3.any() else None),
            "b3_accuracy": float((ptop[b3] == y[b3]).mean()) if b3.any() else None,
        }
        rep[prefix] = out

    summarise("dguard_flow", [r["act"] for r in rows], [r["tier"] for r in rows],
              [r["pred"] for r in rows], np.stack([r["p"] for r in rows]))
    summarise("always_stage_B", [r["actB"] for r in rows], [r["tierB"] for r in rows],
              [r["predB"] for r in rows], np.stack([r["pB"] for r in rows]))
    rep["acquired_dermoscopy"] = float(np.mean([r["stage"] == "B" for r in rows]))
    rep["stage_A_accuracy"] = float((np.stack([r["pA"] for r in rows]).argmax(1) == y).mean())
    rep["mean_u_stage_A"] = float(np.mean([r["u"] for r in rows]))
    rep["ood_rate"] = float(np.mean([r["ood"] for r in rows]))
    rep["quality_severe_rate"] = float(np.mean([r["severe"] for r in rows]))
    per = {}
    for c in sorted(set(y)):
        m = y == c
        pf = np.array([r["pred"] for r in rows])
        per[eng.classes[c]] = {"n": int(m.sum()), "recall_flow": float((pf[m] == c).mean())}
    rep["per_class"] = per

    print()
    print("=" * 72)
    print("EXTERNAL TEST - MILK10k system on Derm7pt (no refitting)")
    print("=" * 72)
    print(json.dumps(rep, indent=2, default=float))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, default=float))
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
