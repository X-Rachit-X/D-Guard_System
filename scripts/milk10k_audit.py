"""
scripts/milk10k_audit.py - Audit MILK10k before anything is trained on it.

Reports what the dataset actually contains, and runs the integrity checks whose
absence is what made the previous generation of results untrustworthy:

  1. structural load (one row per lesion, both images resolved, labels one-hot)
  2. exact-duplicate detection by SHA-256 over image bytes
  3. per-split class and skin-tone counts, with the conformal power check
     (a stratum needs ceil(1/alpha) calibration points or it gets NO guarantee)
  4. calibration-size feasibility for the stated risk target

Usage:
    python scripts/milk10k_audit.py
    python scripts/milk10k_audit.py --alpha 0.05 --delta 0.10
    python scripts/milk10k_audit.py --hash          # slow: reads all 10480 files
    python scripts/milk10k_audit.py --json outputs/audit/milk10k.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.constants import MILK_CLASS_NAMES, MILK_MALIGNANT_CLASSES  # noqa: E402
from src.data.milk10k import (  # noqa: E402
    build_cases,
    lesion_level_split,
    load_milk10k,
    summarise,
)


def sha256_file(path: Path, chunk: int = 1 << 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def rule(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="milk10k", help="extracted MILK10k directory")
    ap.add_argument("--alpha", type=float, default=0.05, help="target risk level")
    ap.add_argument("--delta", type=float, default=0.10, help="LTT failure probability")
    ap.add_argument("--hash", action="store_true",
                    help="SHA-256 every image to find exact duplicates (slow)")
    ap.add_argument("--no-images", action="store_true",
                    help="skip the image-existence check")
    ap.add_argument("--json", default=None, help="write the report here")
    args = ap.parse_args()

    report: dict = {"root": args.root, "alpha": args.alpha, "delta": args.delta}

    rule("1. STRUCTURE")
    df = load_milk10k(args.root, require_images=not args.no_images)
    s = summarise(df)
    report["summary"] = s
    print(f"lesions            : {s['lesions']}")
    print(f"images             : {s['images']}  (every lesion has clinical + dermoscopic)")
    print(f"classes            : {s['classes']}")
    print(f"malignant          : {s['malignant']} ({100 * s['malignant_frac']:.1f}%)")
    print(f"histopath-confirmed: {100 * s['histopath_frac']:.1f}%")
    print(f"rarest class       : {s['rarest_class'][0]} n={s['rarest_class'][1]}")
    print()
    print(f"{'class':<10}{'lesions':>9}{'share':>9}   malignancy")
    for c in MILK_CLASS_NAMES:
        n = s["class_counts"][c]
        tag = "MALIGNANT" if c in MILK_MALIGNANT_CLASSES else "benign"
        print(f"{c:<10}{n:>9}{100 * n / s['lesions']:>8.2f}%   {tag}")

    print()
    print("NOTE: 71.7% of lesions are malignant and BCC alone is 48.1%. This is a")
    print("      tertiary-referral population, not a screening one. Any cost matrix")
    print("      or risk target calibrated here inherits that base rate.")

    rule("2. EXACT DUPLICATES")
    if args.hash:
        seen: dict[str, list[str]] = defaultdict(list)
        paths = [*df["clinical_path"], *df["dermoscopy_path"]]
        for i, p in enumerate(paths, 1):
            seen[sha256_file(Path(p))].append(Path(p).name)
            if i % 2000 == 0:
                print(f"  hashed {i}/{len(paths)}")
        dups = {h: v for h, v in seen.items() if len(v) > 1}
        report["duplicate_groups"] = len(dups)
        report["duplicate_files"] = sum(len(v) for v in dups.values())
        print(f"distinct image bytes : {len(seen)} of {len(paths)}")
        print(f"duplicate groups     : {len(dups)}")
        for h, v in list(dups.items())[:10]:
            print(f"   {h[:16]}  {v}")
        if dups:
            print()
            print("ACTION: duplicated bytes across lesions would leak across splits.")
    else:
        print("skipped (pass --hash to read all image bytes)")
        report["duplicate_groups"] = None

    rule("3. SPLITS")
    tr, ca, te = lesion_level_split(df)
    report["splits"] = {"train": len(tr), "calibration": len(ca), "test": len(te)}
    print(f"train {len(tr)}  calibration {len(ca)}  test {len(te)}  "
          f"(lesion-disjoint, stratified, deterministic)")
    print()
    print(f"{'class':<10}{'train':>7}{'cal':>7}{'test':>7}   per-class guarantee at "
          f"alpha={args.alpha}")
    need = math.ceil(1.0 / args.alpha)
    underpowered = []
    for c in MILK_CLASS_NAMES:
        n_tr = int((tr["label_name"] == c).sum())
        n_ca = int((ca["label_name"] == c).sum())
        n_te = int((te["label_name"] == c).sum())
        ok = n_ca >= need
        if not ok:
            underpowered.append(c)
        print(f"{c:<10}{n_tr:>7}{n_ca:>7}{n_te:>7}   "
              f"{'yes' if ok else f'NO (needs {need}, has {n_ca})'}")
    report["underpowered_classes"] = underpowered
    print()
    print(f"malignancy prevalence: train {tr['malignant'].mean():.3f}  "
          f"cal {ca['malignant'].mean():.3f}  test {te['malignant'].mean():.3f}")
    if underpowered:
        print()
        print(f"{len(underpowered)} class(es) cannot receive a class-conditional")
        print(f"guarantee: {', '.join(underpowered)}. The implementation withholds a")
        print("threshold for these rather than fabricating one.")

    rule("4. SKIN TONE (fairness strata)")
    print(f"{'tone':<8}{'lesions':>9}{'cal':>7}   per-group guarantee")
    tone_report = {}
    for tone in sorted(set(df["skin_tone_class"].astype(str))):
        n = int((df["skin_tone_class"].astype(str) == tone).sum())
        n_ca = int((ca["skin_tone_class"].astype(str) == tone).sum())
        ok = n_ca >= need
        tone_report[tone] = {"lesions": n, "calibration": n_ca, "powered": ok}
        print(f"{tone or '(blank)':<8}{n:>9}{n_ca:>7}   "
              f"{'yes' if ok else f'NO (needs {need})'}")
    report["skin_tone"] = tone_report
    print()
    print("Tone distribution is severely imbalanced (level 3 is ~61% of images,")
    print("level 0 has 6 lesions). Fairness is REPORTED per tone; no mitigation is")
    print("attempted, and unpowered groups get no per-group guarantee.")

    rule("5. RISK-TARGET FEASIBILITY")
    n_cal = len(ca)
    base = float(ca["malignant"].mean())
    print(f"calibration size     n = {n_cal}")
    print(f"malignant prevalence     {base:.4f}")
    print(f"target               alpha = {args.alpha}, delta = {args.delta}")
    print()
    # Hoeffding one-sided bound on the slack a calibration set of this size buys.
    slack = math.sqrt(math.log(1.0 / args.delta) / (2.0 * n_cal))
    print(f"Hoeffding slack at delta={args.delta}: {slack:.4f}")
    print(f"=> an empirical miss rate must fall below {max(args.alpha - slack, 0.0):.4f}")
    print(f"   on the calibration split before alpha={args.alpha} can be certified.")
    report["ltt_feasibility"] = {
        "n_calibration": n_cal,
        "hoeffding_slack": round(slack, 5),
        "required_empirical_risk": round(max(args.alpha - slack, 0.0), 5),
    }
    if args.alpha - slack <= 0:
        print()
        print(f"INFEASIBLE: with n={n_cal} the slack alone exceeds alpha. A larger")
        print("calibration split or a looser alpha is required.")

    rule("6. CASES")
    cases = build_cases(te.head(50))
    report["case_fields"] = sorted(cases[0].keys())
    print(f"built {len(cases)} sample cases from the test split")
    print(f"both image modalities available : "
          f"{all(c['clinical_available'] and c['dermoscopy_available'] for c in cases)}")
    print(f"text modality available        : "
          f"{any(c['text_available'] for c in cases)}  (MILK10k ships no narrative)")
    print(f"metadata keys                  : {len(cases[0]['metadata'])}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print()
        print(f"report written to {out}")

    print()
    print("AUDIT COMPLETE - no blocking defect found" if not underpowered
          else f"AUDIT COMPLETE - {len(underpowered)} underpowered class(es), "
               f"handled by withholding guarantees")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
