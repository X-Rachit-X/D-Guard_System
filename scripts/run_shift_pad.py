"""
scripts/run_shift_pad.py - Distribution shift: the MILK10k system on PAD-UFES-20.

Does the calibrated system stay safe on a different population, camera and
country? Every PAD-UFES-20 image is a smartphone clinical photo from Brazil; the
system was trained and calibrated only on MILK10k. Nothing is refitted.

  * PAD classes map onto MILK10k: BCC->BCC, ACK->AKIEC, NEV->NV, SEK->BKL,
    SCC->SCCKA, MEL->MEL. Malignancy uses MILK10k's definition (AKIEC counts),
    with PAD-UFES's own (BCC, SCC, MEL only) reported alongside.
  * Metadata: age, sex and body site mapped; no skin tone, no MONET scores.
  * Specialist S4 (the original PAD-UFES Model A) is SWITCHED OFF: it was trained
    on these images, so including it would leak labels.
  * Stage A only - PAD-UFES has no dermoscopy.

The question is the one the guarantee cannot answer by itself: Learn-then-Test
assumes calibration and test data are exchangeable. Here they are not.

    python scripts/run_shift_pad.py --data ../HealthSphere_archive/data/raw
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

PAD_TO_MILK = {"BCC": "BCC", "ACK": "AKIEC", "NEV": "NV", "SEK": "BKL",
               "SCC": "SCCKA", "MEL": "MEL"}
SITE = {"FACE": "head_neck_face", "NOSE": "head_neck_face", "EAR": "head_neck_face",
        "LIP": "head_neck_face", "NECK": "head_neck_face", "SCALP": "head_neck_face",
        "FOREARM": "upper_extremity", "ARM": "upper_extremity", "HAND": "hand",
        "CHEST": "trunk", "BACK": "trunk", "ABDOMEN": "trunk",
        "THIGH": "lower_extremity", "FOOT": "foot"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT.parent / "HealthSphere_archive" / "data" / "raw"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="outputs/shift_pad/report.json")
    args = ap.parse_args()

    from src.agents.quality import ImageQualityAgent
    from src.native import NativeEngine

    eng = NativeEngine()
    if eng.team is None:
        print("ERROR: team bundle not active - run scripts/run_team.py and "
              "run_eade_milk.py --native outputs/milk_team first")
        return 1
    eng.team.pad.available = False                  # S4 trained on PAD-UFES: exclude
    qa = ImageQualityAgent.fit([], cache=ROOT / "outputs/milk_rag/quality_thresholds.json")

    data = Path(args.data)
    df = pd.read_csv(data / "metadata.csv")
    if args.limit:
        df = df.sample(n=args.limit, random_state=0)
    mal = set(eng.malignant)
    pad_cancer = {eng.classes.index(c) for c in ("BCC", "SCCKA", "MEL")}

    rows = []
    for i, r in enumerate(df.itertuples(index=False)):
        img = Image.open(data / "images" / r.img_id).convert("RGB")
        meta = {"age": None if pd.isna(r.age) else float(r.age),
                "sex": {"MALE": "male", "FEMALE": "female"}.get(str(r.gender)),
                "site": SITE.get(str(r.region))}
        res = eng.assess(img, None, meta)
        d = res.decision
        q = qa.check(img)
        tier = d.tier
        if d.action.value == "PROCEED" and tier != "LOW" and q.severe:
            tier = "LOW"                                  # same quality gate as the app
        y = eng.classes.index(PAD_TO_MILK[r.diagnostic])
        rows.append({"y": y, "action": d.action.value, "tier": tier or "",
                     "pred": d.tentative, "p": np.array(res.probs),
                     "u": res.ignorance, "ood": bool(res.team.get("ood")),
                     "quality_severe": bool(q.severe), "biopsed": bool(r.biopsed)})
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(df)}", flush=True)

    y = np.array([r["y"] for r in rows])
    pred = np.array([r["pred"] for r in rows])
    P = np.stack([r["p"] for r in rows])
    act = np.array([r["action"] for r in rows])
    tier = np.array([r["tier"] for r in rows])
    high = (act == "PROCEED") & (tier != "LOW")
    answered = act == "PROCEED"
    labels = sorted(set(y))

    def missed(mask, malset):
        return float((np.isin(y[mask], list(malset)) & ~np.isin(pred[mask], list(malset))).mean()) \
            if mask.any() else None

    cov = high.mean()
    conf = P.max(1)
    b3 = conf >= np.quantile(conf, 1 - cov) if cov > 0 else np.zeros(len(y), bool)
    rep = {
        "n": len(y), "classes": [eng.classes[c] for c in labels],
        "accuracy": float((pred == y).mean()),
        "macro_f1_6": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "mal_auroc": float(roc_auc_score(np.isin(y, list(mal)), P[:, list(mal)].sum(1))),
        "mean_u": float(np.mean([r["u"] for r in rows])),
        "ood_rate": float(np.mean([r["ood"] for r in rows])),
        "quality_severe_rate": float(np.mean([r["quality_severe"] for r in rows])),
        "answered": float(answered.mean()), "high": float(cov),
        "acquire": float((act == "ACQUIRE").mean()), "defer": float((act == "DEFER").mean()),
        "high_accuracy": float((pred[high] == y[high]).mean()) if high.any() else None,
        "high_miss_milk_def": missed(high, mal),
        "high_miss_pad_def": missed(high, pad_cancer),
        "b3_miss_milk_def": missed(b3, mal),
        "b3_accuracy": float((pred[b3] == y[b3]).mean()) if b3.any() else None,
        "target_alpha": 0.05,
    }
    rep["guarantee_violation"] = (None if rep["high_miss_milk_def"] is None
                                  else rep["high_miss_milk_def"] - 0.05)

    print()
    print("=" * 72)
    print("DISTRIBUTION SHIFT - MILK10k system on PAD-UFES-20 (no refitting)")
    print("=" * 72)
    for k, v in rep.items():
        if k != "classes":
            print(f"  {k:24} {v if not isinstance(v, float) else round(v, 4)}")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(f"\nsaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
