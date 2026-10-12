"""
scripts/run_stress_milk.py - DERMA-STRESS on the trained MILK10k model.

Does the system notice bad evidence and become MORE careful, instead of
confidently wrong? Each condition degrades the stage-A input (clinical photo +
details) of the same held-out test lesions; nothing is retrained.

    clean        unchanged
    blur         Gaussian blur, radius 4
    dark         brightness x0.35
    noise        Gaussian noise, sigma 0.12
    jpeg         JPEG quality 8
    occlusion    centre 35% of the photo blacked out (caught by the `covered` check)
    no_details   metadata withheld (vacuous opinion)

Per condition: mean ignorance u, HIGH-tier rate, answered rate, deferral rate,
accuracy and missed-cancer rate of HIGH-tier answers, and how often the
image-quality agent flagged the photo. "model only" columns are without the
pipeline's quality and missing-details gates.

    python scripts/run_stress_milk.py              # 300 lesions, ~3 min CPU
    python scripts/run_stress_milk.py --n 787       # whole test split
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter  # noqa: E402


from src.eval.corruptions import blur, dark, jpeg, noise, occlusion  # noqa: E402


CONDITIONS = {"clean": None, "blur": blur, "dark": dark, "noise": noise,
              "jpeg": jpeg, "occlusion": occlusion, "no_details": "no_meta"}
DERMO_CACHE = "quality_thresholds_dermoscopy.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="outputs/stress_milk/report.json")
    args = ap.parse_args()

    from src.agents.quality import ImageQualityAgent  # noqa: F401
    from src.native import NativeEngine
    eng = NativeEngine()
    train = eng.splits["train"]
    qa = ImageQualityAgent.fit([str(eng.img_dir / f) for f in train["clinical_file"].iloc[:300]],
                               cache=ROOT / "outputs" / "milk_rag" / "quality_thresholds.json")
    test = eng.splits["test"].iloc[: args.n]
    mal = set(eng.malignant)

    rows = {}
    for name, fn in CONDITIONS.items():
        u, tier, act, correct, missed, flagged, y_mal = [], [], [], [], [], [], []
        for r in test.itertuples(index=False):
            img = Image.open(eng.img_dir / r.clinical_file).convert("RGB")
            meta = {"age": None if np.isnan(r.age) else float(r.age), "sex": r.sex,
                    "site": r.site, "anatom_site_general": r.anatom_site_general,
                    "skin_tone_class": str(r.skin_tone_class),
                    **{c: float(getattr(r, c)) for c in eng.monet_median}}
            if fn == "no_meta":
                meta = {}
            elif fn is not None:
                img = fn(img)
            res = eng.assess(img, None, meta)
            d = res.decision
            q = qa.check(img)
            u.append(res.ignorance)
            act.append(d.action.value)
            tier.append(d.tier or "")
            flagged.append(bool(q.severe))
            pred = d.prediction
            correct.append(pred == int(r.label_idx) if pred is not None else None)
            is_mal = int(r.label_idx) in mal
            y_mal.append(is_mal)
            missed.append(is_mal and pred is not None and pred not in mal)
        act, tier = np.array(act), np.array(tier)
        high_model = (act == "PROCEED") & (tier != "LOW")
        # With the pipeline's gates: a flagged photo (blur, dark, covered, ...) or a
        # photo with no details and no dermoscopy cannot be HIGH.
        gated = np.array(flagged) | (np.array([fn == "no_meta"] * len(flagged)))
        high = high_model & ~gated
        answered = act == "PROCEED"
        corr = np.array([c if c is not None else False for c in correct])
        miss = np.array(missed)
        rows[name] = {
            "mean_u": float(np.mean(u)),
            "high_rate": float(high.mean()),
            "answered_rate": float(answered.mean()),
            "acquire_rate": float((act == "ACQUIRE").mean()),
            "defer_rate": float((act == "DEFER").mean()),
            "high_accuracy": float(corr[high].mean()) if high.any() else None,
            "high_miss_rate": float(miss[high].mean()) if high.any() else None,
            "answered_accuracy": float(corr[answered].mean()) if answered.any() else None,
            "quality_flagged": float(np.mean(flagged)),
            "high_rate_model_only": float(high_model.mean()),
            "high_miss_rate_model_only": (float(miss[high_model].mean())
                                          if high_model.any() else None),
            "high_accuracy_model_only": (float(corr[high_model].mean())
                                         if high_model.any() else None),
        }
        r_ = rows[name]
        print(f"{name:11} model-only HIGH {r_['high_rate_model_only']:.2f} "
              f"miss {(r_['high_miss_rate_model_only'] or 0):.3f} | with quality gate ->", end=" ")
        print(f"u {r_['mean_u']:.3f}  HIGH {r_['high_rate']:.2f}  "
              f"answered {r_['answered_rate']:.2f}  acquire {r_['acquire_rate']:.2f}  "
              f"defer {r_['defer_rate']:.2f}  HIGH-acc "
              f"{(r_['high_accuracy'] or 0):.3f}  HIGH-miss {(r_['high_miss_rate'] or 0):.3f}  "
              f"flagged {r_['quality_flagged']:.2f}", flush=True)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_lesions": len(test), "conditions": rows}, indent=2))
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
