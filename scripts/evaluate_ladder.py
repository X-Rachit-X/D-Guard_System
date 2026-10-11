"""
scripts/evaluate_ladder.py - Baseline ladder over the trained specialists A-G.

This is the PAD-UFES-20 ladder for this project. It replaces the B1-B6 ladder in
proj.md Part 2, which was defined over ResNet152 / Swin-Tiny / PanDerm - models
that have no weights here (docs/evaluation_integrity.md section 4).

    A  image only          <- unimodal
    B  metadata only       <- unimodal
    C  text only           <- unimodal
    D  image + metadata    <- bimodal
    E  image + text        <- bimodal
    F  text + metadata     <- bimodal
    G  image + meta + text <- full multimodal

Reports per-class precision/recall/F1 (proj.md rule 9), macro-F1, accuracy,
bootstrap 95% CIs, and pairwise McNemar tests against A.

READ THIS BEFORE USING THE NUMBERS
----------------------------------
The shipped checkpoints were trained on every case in this dataset, so NO split
of it is held out for them. `--split test` uses a clean patient-level partition,
but the models have still seen those patients. The output is therefore an upper
bound contaminated by memorisation, not a generalisation estimate. It is printed
so the ladder's *shape* (does adding a modality help?) can be inspected, and so
the machinery is ready for a legitimate run after retraining.

Usage:
    python scripts/evaluate_ladder.py --limit 300
    python scripts/evaluate_ladder.py --split test --bootstrap 1000
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LADDER = ["A", "B", "C", "D", "E", "F", "G"]
RUNG_LABEL = {
    "A": "image only", "B": "metadata only", "C": "text only",
    "D": "image + metadata", "E": "image + text", "F": "text + metadata",
    "G": "image + metadata + text",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the A-G baseline ladder")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of cases")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--texts-csv", default="Dummy/Project/data/generated_clinical_texts.csv")
    parser.add_argument("--output", default="outputs/ladder/ladder_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import numpy as np
    from sklearn.metrics import classification_report, f1_score

    from src.constants import CLASS_NAMES
    from src.data.case_loader import load_cases, patient_level_split
    from src.eval.stats import bootstrap_ci, fdr_correction, mcnemar_test
    from src.models.backends import get_backends

    print("=" * 78)
    print("DERMA-GUARD baseline ladder - PAD-UFES-20, specialists A-G")
    print("=" * 78)

    texts = args.texts_csv if Path(args.texts_csv).exists() else None
    cases = load_cases(texts_csv=texts, limit=args.limit, seed=args.seed)
    if args.split != "all":
        cases = patient_level_split(cases, seed=args.seed)[args.split]
    if not cases:
        print("No cases loaded.")
        return 1

    print(f"Split: {args.split} | cases: {len(cases)} | "
          f"patients: {len({c['patient_id'] for c in cases})}")
    print("NOTE: these checkpoints trained on every case here - see "
          "docs/evaluation_integrity.md\n")

    backends = get_backends()
    labels, preds = [], {c: [] for c in LADDER}

    for i, case in enumerate(cases, 1):
        from src.data.case_loader import image_for_pipeline

        enc = backends.encode_case(
            image=image_for_pipeline(case.get("clinical_image")),
            metadata=case.get("metadata") or None,
            text=case.get("patient_text") or None,
        )
        available = enc.available()
        labels.append(case["label"])
        for code in LADDER:
            try:
                logits = backends.run_specialist(code, enc)
                preds[code].append(int(logits.argmax(-1).item()))
            except Exception:
                preds[code].append(-1)   # specialist not applicable to this case
        if i % 25 == 0 or i == len(cases):
            print(f"  encoded {i}/{len(cases)}", flush=True)

    y = np.array(labels)
    results = {"split": args.split, "n_cases": len(cases), "rungs": {}}

    print("\n" + "=" * 78)
    print(f"{'':3}{'modalities':<24}{'n':>6}{'acc':>8}{'macroF1':>9}{'  95% CI (acc)':>20}")
    print("-" * 78)

    for code in LADDER:
        p = np.array(preds[code])
        mask = p >= 0                       # cases this specialist could read
        if mask.sum() == 0:
            continue
        yt, yp = y[mask], p[mask]
        acc = float((yt == yp).mean())
        macro = float(f1_score(yt, yp, average="macro", zero_division=0))
        ci = bootstrap_ci(lambda a, b: float((a == b).mean()), yt, yp,
                          n_bootstrap=args.bootstrap, seed=args.seed)
        results["rungs"][code] = {
            "modalities": RUNG_LABEL[code], "n": int(mask.sum()),
            "accuracy": acc, "macro_f1": macro,
            "ci_lower": ci["ci_lower"], "ci_upper": ci["ci_upper"],
        }
        print(f"{code:<3}{RUNG_LABEL[code]:<24}{int(mask.sum()):>6}{acc:>8.3f}{macro:>9.3f}"
              f"   [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]")

    # Pairwise McNemar against the image-only rung, FDR-corrected.
    print("\n" + "=" * 78)
    print("McNemar vs A (image only), Benjamini-Hochberg corrected")
    print("-" * 78)
    comparisons, raw_p = [], []
    base = np.array(preds["A"])
    for code in LADDER[1:]:
        p = np.array(preds[code])
        mask = (p >= 0) & (base >= 0)
        if mask.sum() == 0:
            continue
        out = mcnemar_test(y[mask], p[mask], base[mask])
        comparisons.append((code, out))
        raw_p.append(out["p_value"])

    if raw_p:
        adjusted = fdr_correction(raw_p, alpha=0.05)
        for (code, out), adj, sig in zip(comparisons, adjusted["adjusted_p_values"],
                                         adjusted["significant"]):
            print(f"  A vs {code}: b={out.get('correct_a_wrong_b', 0):>4} "
                  f"c={out.get('wrong_a_correct_b', 0):>4}  "
                  f"p={out['p_value']:.4g}  p_adj={adj:.4g}  "
                  f"{'significant' if sig else 'n.s.'}")
            results.setdefault("mcnemar_vs_A", {})[code] = {
                "p_value": out["p_value"], "p_adjusted": adj, "significant": bool(sig)}

    # Per-class breakdown for the full multimodal rung.
    p = np.array(preds["G"])
    mask = p >= 0
    if mask.sum():
        print("\n" + "=" * 78)
        print("Per-class breakdown - G (full multimodal)")
        print("-" * 78)
        print(classification_report(y[mask], p[mask], labels=list(range(len(CLASS_NAMES))),
                                    target_names=CLASS_NAMES, digits=3, zero_division=0))

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")
    print("\nThese are contaminated by train/test leakage in the shipped "
          "checkpoints. See docs/evaluation_integrity.md section 6.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
