"""
scripts/run_stress.py - DERMA-STRESS benchmark entry point.

Runs the corruption conditions defined in src/stress/corruption.py against the
full DERMA-GUARD pipeline and writes results to outputs/stress/.

Corruptions are generated ONCE, cached, and replayed identically for every run,
so results are comparable across model variants.

Usage:
    python scripts/run_stress.py --limit 50
    python scripts/run_stress.py --condition blur_severe --condition miss_text
    python scripts/run_stress.py --list-conditions

NOTE: absolute accuracy here is NOT a generalisation estimate - the specialists
were trained on every case in this dataset. See docs/evaluation_integrity.md.
What the benchmark measures honestly is how coverage, deferral and calibration
shift as evidence degrades.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.stress.corruption import CORRUPTION_CONDITIONS  # noqa: E402

CONDITION_IDS = [c["id"] for c in CORRUPTION_CONDITIONS]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the DERMA-STRESS benchmark")
    parser.add_argument("--output-dir", default="outputs/stress")
    parser.add_argument("--metadata-csv", default="data/raw/metadata.csv")
    parser.add_argument("--image-dir", default="data/raw/images")
    parser.add_argument("--texts-csv", default="Dummy/Project/data/generated_clinical_texts.csv",
                        help="CSV with img_id + a narrative column")
    parser.add_argument("--limit", type=int, default=50,
                        help="Number of cases (stratified by diagnosis). None = all 2298.")
    parser.add_argument("--condition", action="append", default=None,
                        help="Condition id; repeatable. Default: every condition.")
    parser.add_argument("--list-conditions", action="store_true")
    parser.add_argument("--model-name", default="derma-guard-v4")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.list_conditions:
        print(f"{len(CORRUPTION_CONDITIONS)} conditions:\n")
        for c in CORRUPTION_CONDITIONS:
            print(f"  {c['id']:<20} {c['category']:<13} {c['name']}")
        return 0

    unknown = [c for c in (args.condition or []) if c not in CONDITION_IDS]
    if unknown:
        print(f"Unknown condition(s): {unknown}")
        print(f"Available: {CONDITION_IDS}")
        return 1

    from src.agents.orchestrator import OrchestratorAgent
    from src.data.case_loader import load_cases
    from src.stress.benchmark import StressBenchmark

    texts_csv = args.texts_csv if Path(args.texts_csv).exists() else None
    if args.texts_csv and texts_csv is None:
        print(f"[run_stress] No narratives at {args.texts_csv}; running without text.")

    print("=" * 70)
    print("DERMA-STRESS Benchmark")
    print(f"Conditions: {len(args.condition or CONDITION_IDS)} | cases: {args.limit or 'all'}")
    print("=" * 70)

    cases = load_cases(metadata_csv=args.metadata_csv, image_dir=args.image_dir,
                       texts_csv=texts_csv, limit=args.limit, seed=args.seed)
    if not cases:
        print("[run_stress] No cases loaded - check --metadata-csv and --image-dir.")
        return 1
    print(f"[run_stress] Loaded {len(cases)} cases "
          f"({sum(c['clinical_available'] for c in cases)} with images).")

    benchmark = StressBenchmark(
        pipeline=OrchestratorAgent.build(),
        model_name=args.model_name,
        dataset="pad_ufes",
        seed=args.seed,
    )
    benchmark.run(cases, conditions=args.condition,
                  output_dir=str(PROJECT_ROOT / args.output_dir))

    print("\nReminder: absolute accuracy is inflated by train/test leakage in the "
          "shipped checkpoints. See docs/evaluation_integrity.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
