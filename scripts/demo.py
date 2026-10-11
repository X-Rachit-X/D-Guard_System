"""
scripts/demo.py - End-to-end smoke test of the DERMA-GUARD pipeline.

Runs one real case through every stage and prints the decision, the evidence
that drove it, and the full agent trace. Then re-runs the case with modalities
removed to show graceful degradation.

Usage:
    python scripts/demo.py
    python scripts/demo.py --image data/raw/images/PAT_1000_31_620.png
    python scripts/demo.py --trace
"""
import argparse
import glob
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_METADATA = {
    "age": 55, "gender": "FEMALE", "region": "NECK",
    "diameter_1": 6.0, "diameter_2": 5.0, "fitspatrick": 3,
    "itch": "TRUE", "grew": "TRUE", "hurt": "FALSE", "bleed": "TRUE",
    "elevation": "TRUE", "changed": "FALSE",
}
DEFAULT_TEXT = ("55-year-old female with a 6mm bleeding pigmented nodular lesion "
                "on the neck that has grown over three months and itches.")


def show(result, indent="") -> None:
    conf = f"{result.confidence:.3f}" if result.confidence is not None else "-"
    print(f"{indent}action      : {result.action}"
          f"{'' if result.consensus_level is None else f'  (level {result.consensus_level})'}")
    print(f"{indent}diagnosis   : {result.diagnosis_name or '-'}   confidence: {conf}")
    if result.acquire_target:
        print(f"{indent}request     : {result.acquire_target}")
    if result.escalate_reason:
        print(f"{indent}reason      : {result.escalate_reason}")
    active = {k: round(v, 3) for k, v in result.fusion_weights.items() if v > 0}
    if active:
        print(f"{indent}weights     : {active}")
    if result.specialist_votes:
        print(f"{indent}votes       : {result.specialist_votes}")
    print(f"{indent}evidence    : used={result.evidence_used} missing={result.evidence_missing}")
    if result.similar_cases:
        # Explanation only: these never enter the decision.
        top = ", ".join(f"{c.get('diagnosis')}@{c.get('distance', 0):.3f}"
                        for c in result.similar_cases[:5])
        agree = result.retrieval_agreement
        print(f"{indent}retrieval   : {len(result.similar_cases)} similar cases"
              + (f", agreement {agree:.3f}" if agree is not None else ""))
        print(f"{indent}              {top}")
    else:
        print(f"{indent}retrieval   : none "
              f"(build an index: python scripts/build_rag_index.py --verify)")
    print(f"{indent}elapsed     : {result.processing_time_s:.2f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description="DERMA-GUARD end-to-end demo")
    parser.add_argument("--image", default=None, help="Path to a lesion image")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--trace", action="store_true", help="Print the full agent trace")
    parser.add_argument("--no-degraded", action="store_true",
                        help="Skip the degraded-modality runs")
    args = parser.parse_args()

    from PIL import Image

    from src.agents.orchestrator import OrchestratorAgent, PatientCase
    from src.models.backends import default_weights_dir

    image_path = args.image
    if image_path is None:
        matches = sorted(glob.glob(str(PROJECT_ROOT / "data" / "raw" / "images" / "*.png")))
        image_path = matches[0] if matches else None

    print("=" * 74)
    print("DERMA-GUARD v4.0 - end-to-end demo")
    print(f"weights : {default_weights_dir()}")
    print(f"image   : {image_path or '(none found)'}")
    print("=" * 74)
    print("\nBuilding pipeline (first run loads ResNet50 + Bio_ClinicalBERT)...")

    orch = OrchestratorAgent.build()
    image = Image.open(image_path) if image_path and Path(image_path).exists() else None

    print("\n--- FULL CASE (image + metadata + text) " + "-" * 34)
    result = orch.process_case(PatientCase(
        case_id="demo-full", image=image, metadata=DEFAULT_METADATA, text=args.text))
    show(result)

    if result.report:
        print("\nreport:")
        for line in result.report.splitlines():
            print(f"  {line}")

    if args.trace:
        print("\ntrace:")
        for step in result.evidence_trace:
            print(f"  {step['t']:>7.3f}s  {step['step']:<20} {step['data'][:150]}")

    if not args.no_degraded:
        print("\n--- DEGRADED CASES " + "-" * 54)
        variants = [
            ("image only", dict(image=image)),
            ("metadata only", dict(metadata=DEFAULT_METADATA)),
            ("text only", dict(text=args.text)),
            ("image + metadata", dict(image=image, metadata=DEFAULT_METADATA)),
            ("nothing", dict()),
        ]
        for label, kwargs in variants:
            res = orch.process_case(PatientCase(case_id=f"demo-{label}", **kwargs))
            conf = f"{res.confidence:.3f}" if res.confidence is not None else "-"
            extra = res.acquire_target or (res.escalate_reason or "")[:46]
            print(f"  {label:<18} {res.action:<10} {res.diagnosis_name or '-':<5} "
                  f"conf={conf:<7} {extra}")

    print("\n" + "=" * 74)
    print("Research prototype. The bundled model metrics are NOT valid -")
    print("see docs/evaluation_integrity.md before citing any number.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
