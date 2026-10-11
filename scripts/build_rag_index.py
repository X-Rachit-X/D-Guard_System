"""
scripts/build_rag_index.py - Build the similar-case retrieval index.

This is the step that was missing: `RAGIndex` and `LibrarianAgent` were both
implemented and tested, but nothing ever populated an index, so the Librarian
returned "RAG index not loaded" on every real case and the retrieval channel was
inert in the running system.

THE SAFETY PROPERTY THAT MATTERS. Retrieval here is **explanation only**. It sits
outside the decision path on purpose: the conformal and Learn-then-Test
guarantees assume the calibration and test data are exchangeable, and admitting a
corpus that grows or varies at inference time breaks that assumption. The index
is therefore built from the **training split only**, the split is recorded inside
the artifact, and `--verify` asserts that no indexed case belongs to the
calibration or test patients. Indexing all 2298 cases would let a retrieved
neighbour carry a held-out label into the report - a leak that would be invisible
in the metrics, because it would flow through prose rather than through logits.

Embedding space: the raw reproducible backbone features from
scripts/build_features.py (image 2048 + text 768 + metadata 114 = 2930), not the
projected 256-d space. Those are the features that verify at cosine 1.000000;
the legacy projectors do not reproduce (docs/evaluation_integrity.md). The spec
is written into the artifact and re-checked on load, so an index built in one
space cannot be queried in another.

Usage:
    python scripts/build_rag_index.py
    python scripts/build_rag_index.py --verify
    python scripts/build_rag_index.py --k 8 --out outputs/rag/index_train
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.constants import CLASS_NAMES  # noqa: E402
from src.retrieval.rag_index import RAGIndex  # noqa: E402

# The embedding space this index is built over. Widths are asserted against the
# loaded artifact rather than trusted.
#
# WHY METADATA IS EXCLUDED BY DEFAULT. The metadata preprocessor emits **116**
# columns when fitted on the full metadata.csv but **114** when fitted on the
# training split only, which is how features_v2.pt was built: two one-hot columns
# correspond to categories that do not occur in the training subset. The running
# system's `encode_metadata` returns 116. So a 114-wide index could not be
# queried by the live pipeline at all, and padding it would silently misalign
# every column after the missing ones.
#
# `image_text` uses only the two widths that are provably identical on both
# sides - CaseEncoding.image_raw (2048) and CaseEncoding.text_raw (768) - so an
# index built offline is queryable by the orchestrator unchanged. `raw3` adds the
# 114-wide metadata for offline analysis; it is NOT runtime-queryable, and the
# script says so.
SPECS = {
    "image_text": {
        "clinical_embedding": 2048,   # ResNet50 penultimate, IMAGENET1K_V2
        "text_embedding": 768,        # Bio_ClinicalBERT [CLS]
    },
    "raw3": {
        "clinical_embedding": 2048,
        "text_embedding": 768,
        "metadata_embedding": 114,    # train-fitted MetadataPreprocessor
    },
}
RUNTIME_QUERYABLE = {"image_text"}


def patient_level_split(patient_ids, ratios=(0.6, 0.2, 0.2), seed: int = 42):
    """Deterministic, patient-disjoint 3-way split. Mirrors run_eade.py."""
    uniq = sorted(set(patient_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_tr = int(round(n * ratios[0]))
    n_ca = int(round(n * ratios[1]))
    train = set(uniq[:n_tr])
    calib = set(uniq[n_tr:n_tr + n_ca])
    test = set(uniq[n_tr + n_ca:])
    assert not (train & calib) and not (train & test) and not (calib & test)
    return train, calib, test


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="data/processed/features_v2.pt")
    ap.add_argument("--out", default="outputs/rag/index_train")
    ap.add_argument("--k", type=int, default=8, help="neighbours to probe during --verify")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--spec", choices=sorted(SPECS), default="image_text",
                    help="embedding space to index (default: image_text, the only "
                         "one the live pipeline can query)")
    ap.add_argument("--verify", action="store_true",
                    help="reload the artifact and assert the no-leakage property")
    args = ap.parse_args()
    spec = SPECS[args.spec]

    feat_path = Path(args.features)
    if not feat_path.exists():
        print(f"ERROR: {feat_path} not found. Run first:")
        print("  python scripts/build_features.py --fit-metadata-on-train")
        return 1

    print("=" * 74)
    print("BUILD RAG INDEX  (explanation-only, training split only)")
    print("=" * 74)

    d = torch.load(feat_path, map_location="cpu", weights_only=False)
    img, txt, meta = d["img_feats"], d["text_feats"], d["meta_feats"]
    labels = d["labels"]
    img_ids, patient_ids = d["img_ids"], d["patient_ids"]
    texts = d.get("texts", [""] * len(img_ids))

    available = {"clinical_embedding": img.shape[1],
                 "text_embedding": txt.shape[1],
                 "metadata_embedding": meta.shape[1]}
    mismatched = {k: (v, available.get(k)) for k, v in spec.items()
                  if available.get(k) != v}
    if mismatched:
        print(f"ERROR: feature widths do not match spec {args.spec!r}: "
              f"{ {k: f'expected {a}, got {b}' for k, (a, b) in mismatched.items()} }")
        print("Refusing to build an index whose embedding space is not what this")
        print("script documents. Edit SPECS deliberately if the pipeline changed.")
        return 1
    print(f"features   : {len(img_ids)} rows")
    print(f"spec       : {args.spec} = {spec} -> dim {sum(spec.values())}")
    if args.spec not in RUNTIME_QUERYABLE:
        print(f"             NOTE: {args.spec!r} is NOT runtime-queryable "
              f"(see the comment on SPECS); offline analysis only.")

    train, calib, test = patient_level_split(patient_ids, seed=args.seed)
    print(f"patients   : train {len(train)}  calibration {len(calib)}  test {len(test)}")

    source = {"clinical_embedding": img, "text_embedding": txt,
              "metadata_embedding": meta}
    cases = []
    for i, (iid, pid) in enumerate(zip(img_ids, patient_ids)):
        if pid not in train:
            continue
        lab = int(labels[i])
        case = {
            "case_id": iid,
            "patient_id": pid,
            "label": lab,
            "diagnosis": CLASS_NAMES[lab] if 0 <= lab < len(CLASS_NAMES) else str(lab),
            # Narrative is kept short: it goes into an LLM prompt, not a metric.
            "narrative": (texts[i] or "")[:400],
        }
        for key in spec:
            case[key] = source[key][i].numpy()
        cases.append(case)

    held_out = len(img_ids) - len(cases)
    print(f"indexing   : {len(cases)} training cases "
          f"({held_out} calibration/test cases EXCLUDED)")
    if not cases:
        print("ERROR: no training cases selected.")
        return 1

    index = RAGIndex(embedding_spec=spec)
    index.source_split = "train"
    index.add_training_cases(cases)

    counts = {}
    for c in cases:
        counts[c["diagnosis"]] = counts.get(c["diagnosis"], 0) + 1
    print("class mix  :", {k: counts.get(k, 0) for k in CLASS_NAMES})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    index.save(str(out))

    excluded = sorted({p for p in patient_ids if p not in train})
    side = out.with_name(out.name + "_excluded_patients.json")
    side.write_text(json.dumps({
        "source_split": "train",
        "n_indexed": len(cases),
        "n_excluded_cases": held_out,
        "excluded_patients": excluded,
        "seed": args.seed,
        "embedding_spec": spec,
        "spec_name": args.spec,
        "runtime_queryable": args.spec in RUNTIME_QUERYABLE,
    }, indent=2), encoding="utf-8")
    print(f"audit trail: {side}")

    if args.verify:
        print()
        print("=" * 74)
        print("VERIFY")
        print("=" * 74)
        reloaded = RAGIndex().load(str(out))

        assert reloaded.embedding_spec == spec, "embedding spec did not round-trip"
        print("spec round-trip                     : ok")
        assert reloaded.source_split == "train", "source split did not round-trip"
        print("source split recorded as 'train'    : ok")

        indexed_patients = {c["patient_id"] for c in reloaded._training_cases}
        leaked = indexed_patients & (calib | test)
        assert not leaked, f"LEAK: {len(leaked)} held-out patient(s) in the index"
        print(f"no calibration/test patient indexed : ok ({len(indexed_patients)} patients)")

        # A real query, using a held-out case - the actual inference situation.
        qi = next(i for i, p in enumerate(patient_ids) if p in test)
        probe = {k: source[k][qi].numpy() for k in spec}
        res = reloaded.query(probe, k=args.k, predicted_diagnosis=int(labels[qi]))
        got_n = len(res["similar_cases"])
        assert got_n == args.k, f"expected {args.k} neighbours, got {got_n}"
        print(f"query returns k={got_n} neighbours     : ok")
        sims = [c["distance"] for c in res["similar_cases"]]
        assert all(-1.01 <= x <= 1.01 for x in sims), f"cosine out of range: {sims}"
        assert sims == sorted(sims, reverse=True), "neighbours not sorted by similarity"
        print(f"cosine in [-1,1], descending        : ok "
              f"({sims[0]:.4f} .. {sims[-1]:.4f})")
        assert not any(c["patient_id"] in (calib | test)
                       for c in res["similar_cases"]), "retrieved a held-out case"
        print("retrieved cases all from train      : ok")
        assert all("clinical_embedding" not in c for c in res["similar_cases"]), \
            "raw embeddings leaked into the returned payload"
        print("no raw tensors in the payload       : ok")

        print()
        print(f"query case: {img_ids[qi]}  true={CLASS_NAMES[int(labels[qi])]}")
        print(f"agreement rate among {got_n} neighbours: {res['agreement_rate']:.3f}")
        for c in res["similar_cases"][:5]:
            print(f"   cos {c['distance']:.4f}  {c['diagnosis']:<4}  {c['case_id']}")
        print()
        print("VERIFY PASSED")

    print()
    print("Wire it in:")
    print(f"  LibrarianAgent(index_path='{out}')")
    print("Retrieval is EXPLANATION ONLY - it must not enter the decision path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
