"""
scripts/baseline_probe.py - Honest baseline ladder via linear probes on frozen features.

Trains a class-weighted logistic regression on each modality subset, using the
reproducible feature set from `build_features.py` and a disjoint patient-level
split. No pretrained specialist weights are involved, so these numbers are free
of the leakage and metric-rewriting described in docs/evaluation_integrity.md.

This is the reference table every proposed method must beat.

Usage:
    python scripts/build_features.py --fit-metadata-on-train      # once
    python scripts/baseline_probe.py
    python scripts/baseline_probe.py --model gbm --seeds 42 1 2
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

SUBSETS = {
    "A": ("image only", ("img",)),
    "B": ("metadata only", ("meta",)),
    "C": ("text only", ("text",)),
    "D": ("image + metadata", ("img", "meta")),
    "E": ("image + text", ("img", "text")),
    "F": ("text + metadata", ("text", "meta")),
    "G": ("all three", ("img", "meta", "text")),
}


def main() -> int:
    import numpy as np
    import torch
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 classification_report, f1_score)
    from sklearn.preprocessing import StandardScaler

    from src.constants import CLASS_NAMES
    from src.data.case_loader import patient_level_split
    from src.eval.stats import bootstrap_ci

    ap = argparse.ArgumentParser(description="Honest baseline ladder (linear probes)")
    ap.add_argument("--features", default="data/processed/features_v2.pt")
    ap.add_argument("--model", choices=["logreg", "gbm"], default="logreg")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out", default="docs/results/baseline_probe.json")
    args = ap.parse_args()

    path = PROJECT_ROOT / args.features
    if not path.exists():
        print(f"Missing {path}. Run: python scripts/build_features.py --fit-metadata-on-train")
        return 1

    blob = torch.load(path, map_location="cpu", weights_only=False)
    man = blob["manifest"]
    ids, pats = blob["img_ids"], blob["patient_ids"]
    y = blob["labels"].numpy()
    F = {"img": blob["img_feats"].numpy(),
         "meta": blob["meta_feats"].numpy(),
         "text": blob["text_feats"].numpy()}

    print("=" * 72)
    print("Honest baseline ladder - PAD-UFES-20, linear probes on frozen features")
    print(f"features built {man['created']} | image sha {man['image_set_sha256'][:12]}")
    print(f"dims: image {F['img'].shape[1]} | metadata {F['meta'].shape[1]} | text {F['text'].shape[1]}")
    print("=" * 72)

    results = {"manifest": man, "model": args.model, "seeds": args.seeds, "runs": {}}

    for seed in args.seeds:
        stub = [{"case_id": i, "patient_id": p, "label": int(l)}
                for i, p, l in zip(ids, pats, y)]
        sp = patient_level_split(stub, seed=seed)
        pos = {c: i for i, c in enumerate(ids)}
        idx = {k: np.array([pos[c["case_id"]] for c in sp[k]]) for k in ("train", "val", "test")}

        print(f"\nseed {seed}: train={len(idx['train'])} val={len(idx['val'])} test={len(idx['test'])}")
        print("test class counts:",
              dict(zip(CLASS_NAMES, np.bincount(y[idx["test"]], minlength=len(CLASS_NAMES)).tolist())))
        print(f"\n{'':3}{'configuration':<22}{'acc':>7}{'bal-acc':>9}{'macroF1':>9}{'  acc 95% CI':>18}")
        print("-" * 72)

        seed_res = {}
        for code, (label, keys) in SUBSETS.items():
            X = np.hstack([F[k] for k in keys])
            sc = StandardScaler().fit(X[idx["train"]])
            Xs = sc.transform(X)

            if args.model == "logreg":
                from sklearn.linear_model import LogisticRegression
                clf = LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced")
            else:
                from sklearn.ensemble import HistGradientBoostingClassifier
                clf = HistGradientBoostingClassifier(max_iter=250, random_state=seed)
            clf.fit(Xs[idx["train"]], y[idx["train"]])

            yt = y[idx["test"]]
            p = clf.predict(Xs[idx["test"]])
            acc = accuracy_score(yt, p)
            bal = balanced_accuracy_score(yt, p)
            mf1 = f1_score(yt, p, average="macro", zero_division=0)
            ci = bootstrap_ci(lambda a, b: float((a == b).mean()), yt, p,
                              n_bootstrap=args.bootstrap, seed=seed)
            seed_res[code] = {"label": label, "modalities": list(keys),
                              "accuracy": acc, "balanced_accuracy": bal, "macro_f1": mf1,
                              "acc_ci": [ci["ci_lower"], ci["ci_upper"]]}
            print(f"{code:<3}{label:<22}{acc:>7.3f}{bal:>9.3f}{mf1:>9.3f}"
                  f"   [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]")

        best_pair = max(("D", "E", "F"), key=lambda c: seed_res[c]["macro_f1"])
        if seed_res[best_pair]["macro_f1"] > seed_res["G"]["macro_f1"]:
            print(f"\n  NOTE  best pair {best_pair} (macroF1 {seed_res[best_pair]['macro_f1']:.3f}) "
                  f"BEATS all-three G ({seed_res['G']['macro_f1']:.3f}).")
            print("        Naive concatenation cannot discount a weak modality - this is the")
            print("        motivating gap for evidence-aware fusion. See docs/07_experimental_protocol.md")

        results["runs"][str(seed)] = seed_res

    # Per-class detail for the strongest configuration of the last seed
    best = max(SUBSETS, key=lambda c: seed_res[c]["macro_f1"])
    keys = SUBSETS[best][1]
    X = np.hstack([F[k] for k in keys])
    sc = StandardScaler().fit(X[idx["train"]]); Xs = sc.transform(X)
    if args.model == "logreg":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced")
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(max_iter=250, random_state=args.seeds[-1])
    clf.fit(Xs[idx["train"]], y[idx["train"]])
    print(f"\nPer-class breakdown - {best} ({SUBSETS[best][0]}):")
    print(classification_report(y[idx["test"]], clf.predict(Xs[idx["test"]]),
                                labels=list(range(len(CLASS_NAMES))),
                                target_names=CLASS_NAMES, digits=3, zero_division=0))

    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
