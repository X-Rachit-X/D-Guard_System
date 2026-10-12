"""
scripts/analyze_pad_symptoms.py - Do the scribe's warning signs actually signal cancer?

The scribe (src/agents/scribe.py) raises triage urgency when the patient mentions
bleeding, growth, change, itching/pain, ... MILK10k has no symptom fields, so the rule
could not be checked on it. PAD-UFES-20 (the old project's dataset, in the archive)
records the patient's answers to exactly these questions next to the diagnosis -
so the rule can be checked on real data.

Per symptom: malignancy rate with vs without it, relative risk with a 95% CI, and the
same for "any warning sign". One row per LESION (photos of the same lesion share
answers), malignancy = BCC, SCC or MEL (PAD-UFES's own definition) and, separately,
MILK10k's definition (adds ACK as AKIEC).

    python scripts/analyze_pad_symptoms.py --data ../HealthSphere_archive/data/raw
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SYMPTOMS = {"bleed": "bleeding", "grew": "rapid growth", "changed": "change",
            "itch": "itching", "hurt": "pain", "elevation": "raised (not a scribe flag)"}
SCRIBE_FLAGS = ["bleed", "grew", "changed", "itch", "hurt"]


def rr_ci(a, n1, c, n0):
    """Relative risk (a/n1)/(c/n0) with a log-normal 95% CI."""
    if min(a, c) == 0 or n1 == 0 or n0 == 0:
        return float("nan"), [float("nan"), float("nan")]
    rr = (a / n1) / (c / n0)
    se = np.sqrt(1 / a - 1 / n1 + 1 / c - 1 / n0)
    return float(rr), [float(np.exp(np.log(rr) - 1.96 * se)), float(np.exp(np.log(rr) + 1.96 * se))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT.parent / "HealthSphere_archive" / "data" / "raw"))
    ap.add_argument("--out", default="outputs/analysis/pad_symptoms.json")
    args = ap.parse_args()
    df = pd.read_csv(Path(args.data) / "metadata.csv").drop_duplicates("lesion_id")
    defs = {"PAD-UFES (BCC, SCC, MEL)": {"BCC", "SCC", "MEL"},
            "MILK10k (adds ACK)": {"BCC", "SCC", "MEL", "ACK"}}
    out = {"n_lesions": int(len(df)), "definitions": {}}
    for dname, mal in defs.items():
        y = df["diagnostic"].isin(mal).values
        rows = {}
        print(f"\n=== malignant = {dname}: base rate {y.mean():.1%} over {len(df)} lesions ===")
        print(f"{'symptom':28} {'n yes':>6} {'mal|yes':>8} {'mal|no':>7} {'RR':>5}  95% CI")
        cols = list(SYMPTOMS)
        known_any = df[SCRIBE_FLAGS].isin(["TRUE", "FALSE"]).all(1).values
        for c in cols + ["ANY"]:
            if c == "ANY":
                yes = (df[SCRIBE_FLAGS] == "TRUE").any(axis=1).values
                known = known_any | yes
                label = "any scribe warning sign"
            else:
                yes = (df[c] == "TRUE").values
                known = df[c].isin(["TRUE", "FALSE"]).values
                label = SYMPTOMS[c]
            a, n1 = int((y & yes & known).sum()), int((yes & known).sum())
            cc, n0 = int((y & ~yes & known).sum()), int((~yes & known).sum())
            rr, ci = rr_ci(a, n1, cc, n0)
            rows[label] = {"n_yes": n1, "n_no": n0, "mal_if_yes": a / max(n1, 1),
                           "mal_if_no": cc / max(n0, 1), "relative_risk": rr, "ci95": ci}
            print(f"{label:28} {n1:6d} {a / max(n1, 1):8.1%} {cc / max(n0, 1):7.1%} "
                  f"{rr:5.2f}  [{ci[0]:.2f}, {ci[1]:.2f}]")
        out["definitions"][dname] = {"base_rate": float(y.mean()), "symptoms": rows}
    dest = ROOT / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
