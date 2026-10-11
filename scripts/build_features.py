"""
scripts/build_features.py - Deterministic, provenance-tracked feature extraction.

WHY THIS EXISTS
---------------
The shipped `cached_multimodal_features.pt` cannot be regenerated. Its image
features reach a cosine similarity of only ~0.07 against anything this codebase
can compute from the shipped images, under any of six plausible preprocessing
variants and with the exact backbone stored inside the checkpoints. The
consequence is severe: specialists A/D/E/G were fitted to features no inference
path can reproduce, so they score ~99% replayed against that cache and ~21-30%
on the same images loaded from disk. See docs/evaluation_integrity.md.

This script rebuilds the feature tensor from raw inputs with every choice
recorded, so the artifact can always be reproduced or shown to have drifted:

  * `img_ids` are stored IN the artifact. Row order is never implied by the
    order of a CSV read elsewhere.
  * a manifest records library versions, the exact backbone weight enum, the
    transform spec, and a SHA-256 over the image bytes actually consumed.
  * `verify` re-extracts a random sample and compares against the stored rows.

Usage:
    python scripts/build_features.py
    python scripts/build_features.py --out data/processed/features_v2.pt
    python scripts/build_features.py --verify data/processed/features_v2.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
RESNET_WEIGHTS = "IMAGENET1K_V2"   # pinned explicitly; `DEFAULT` silently moved
BERT_REPO = "emilyalsentzer/Bio_ClinicalBERT"
IMAGE_SIZE = 224
TEXT_MAX_LEN = 128


# ---------------------------------------------------------------------------
def build_transform():
    from torchvision import transforms as T

    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def transform_spec() -> dict:
    """A machine-readable record of the preprocessing, for the manifest."""
    return {
        "resize": [IMAGE_SIZE, IMAGE_SIZE],
        "interpolation": "bilinear (torchvision default)",
        "to_tensor": "scales uint8 -> [0,1]",
        "normalize": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "channel_order": "RGB",
    }


def sha256_of_files(paths, chunk=1 << 20) -> str:
    """One digest over the concatenated bytes of every file, in the given order."""
    h = hashlib.sha256()
    for p in paths:
        h.update(str(Path(p).name).encode())
        with open(p, "rb") as f:
            while True:
                block = f.read(chunk)
                if not block:
                    break
                h.update(block)
    return h.hexdigest()


def load_resnet():
    import torch
    import torchvision.models as tvm

    weights = getattr(tvm.ResNet50_Weights, RESNET_WEIGHTS)
    net = tvm.resnet50(weights=weights)
    net.fc = torch.nn.Identity()
    return net.eval()


# ---------------------------------------------------------------------------
def extract(args) -> int:
    import numpy as np
    import pandas as pd
    import torch
    import torchvision
    import transformers
    from PIL import Image

    from src.constants import CLASS_TO_INDEX

    meta_csv = PROJECT_ROOT / args.metadata_csv
    image_dir = PROJECT_ROOT / args.image_dir
    df = pd.read_csv(meta_csv)
    print(f"[build] metadata rows: {len(df)}  ({meta_csv})")

    # --- texts -------------------------------------------------------------
    texts_map = {}
    texts_csv = PROJECT_ROOT / args.texts_csv if args.texts_csv else None
    if texts_csv and texts_csv.exists():
        tdf = pd.read_csv(texts_csv)
        col = next((c for c in ("clinical_text", "filled_description",
                                "paraphrased_description", "text") if c in tdf.columns), None)
        if col:
            texts_map = dict(zip(tdf["img_id"], tdf[col].astype(str)))
            print(f"[build] narratives: {len(texts_map)} from {texts_csv.name} (column '{col}')")
    if not texts_map:
        print("[build] WARNING no narratives found; text features will be empty strings")

    # --- keep only rows whose image exists, and freeze the order ----------
    rows, paths = [], []
    for _, r in df.iterrows():
        p = image_dir / str(r["img_id"])
        if p.exists():
            rows.append(r)
            paths.append(p)
    if not rows:
        print(f"[build] no images found under {image_dir}")
        return 1
    if len(rows) != len(df):
        print(f"[build] {len(df) - len(rows)} rows dropped (image missing)")

    img_ids = [str(r["img_id"]) for r in rows]
    patient_ids = [str(r.get("patient_id", "")) for r in rows]
    labels = torch.tensor([CLASS_TO_INDEX[r["diagnostic"]] for r in rows], dtype=torch.long)
    texts = [texts_map.get(i, "") for i in img_ids]

    print(f"[build] hashing {len(paths)} image files ...")
    image_digest = sha256_of_files(paths)
    print(f"[build] image set sha256 = {image_digest[:16]}...")

    # --- image features ----------------------------------------------------
    net = load_resnet()
    tf = build_transform()
    print(f"[build] ResNet50 ({RESNET_WEIGHTS}) -> 2048-d, batch {args.batch_size}")

    feats, t0 = [], time.time()
    for start in range(0, len(paths), args.batch_size):
        batch_paths = paths[start:start + args.batch_size]
        xs = torch.stack([tf(Image.open(p).convert("RGB")) for p in batch_paths])
        with torch.no_grad():
            feats.append(net(xs))
        done = min(start + args.batch_size, len(paths))
        if done % (args.batch_size * 10) == 0 or done == len(paths):
            rate = done / max(time.time() - t0, 1e-6)
            print(f"   {done}/{len(paths)}  ({rate:.1f} img/s)", flush=True)
    img_feats = torch.cat(feats).float()
    print(f"[build] image features {tuple(img_feats.shape)} in {time.time()-t0:.1f}s")

    # --- text features -----------------------------------------------------
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BERT_REPO)
    bert = AutoModel.from_pretrained(BERT_REPO).eval()
    print(f"[build] Bio_ClinicalBERT -> 768-d [CLS], max_len {TEXT_MAX_LEN}")

    t_feats, t0 = [], time.time()
    for start in range(0, len(texts), args.batch_size):
        batch = texts[start:start + args.batch_size]
        enc = tok(batch, padding=True, truncation=True,
                  max_length=TEXT_MAX_LEN, return_tensors="pt")
        with torch.no_grad():
            out = bert(**enc)
        # [CLS], matching what the trained text projector consumes.
        t_feats.append(out.last_hidden_state[:, 0, :])
    text_feats = torch.cat(t_feats).float()
    print(f"[build] text features {tuple(text_feats.shape)} in {time.time()-t0:.1f}s")

    # --- metadata features -------------------------------------------------
    from src.data.metadata_preprocessor import MetadataPreprocessor

    df_rows = pd.DataFrame(rows).reset_index(drop=True)
    pre = MetadataPreprocessor()
    # NOTE fitted on ALL rows. The scaler/one-hot vocabulary is unsupervised, so
    # this leaks no labels, but a strict protocol refits on train only; see
    # --fit-metadata-on-train.
    if args.fit_metadata_on_train:
        from src.data.case_loader import patient_level_split

        stub = [{"case_id": i, "patient_id": p, "label": int(l)}
                for i, p, l in zip(img_ids, patient_ids, labels.tolist())]
        train_ids = {c["case_id"] for c in patient_level_split(stub, seed=args.seed)["train"]}
        mask = [i in train_ids for i in img_ids]
        pre.fit_transform(df_rows[mask])
        meta_feats = torch.tensor(pre.transform(df_rows), dtype=torch.float32)
        print(f"[build] metadata preprocessor FIT ON TRAIN ONLY ({sum(mask)} rows)")
    else:
        meta_feats = torch.tensor(pre.fit_transform(df_rows), dtype=torch.float32)
    print(f"[build] metadata features {tuple(meta_feats.shape)}")

    # --- quality priors ----------------------------------------------------
    from src.agents.vision_team import score_image_quality
    from src.agents.scribe import PTISv1

    ptis = PTISv1()
    q_img = torch.tensor([[score_image_quality(Image.open(p).convert("RGB"))] for p in paths],
                         dtype=torch.float32)
    q_text = torch.tensor([[ptis.score(t)] for t in texts], dtype=torch.float32)
    q_meta = torch.tensor([[1.0 - float(r.isnull().mean())] for r in rows], dtype=torch.float32)

    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_rows": len(img_ids),
        "image_set_sha256": image_digest,
        "metadata_csv": str(meta_csv.relative_to(PROJECT_ROOT)),
        "image_dir": str(image_dir.relative_to(PROJECT_ROOT)),
        "texts_csv": str(texts_csv.relative_to(PROJECT_ROOT)) if texts_csv else None,
        "image_backbone": {"arch": "resnet50", "weights": RESNET_WEIGHTS, "head": "fc=Identity"},
        "image_transform": transform_spec(),
        "text_encoder": {"repo": BERT_REPO, "pooling": "CLS", "max_length": TEXT_MAX_LEN},
        "metadata": {"preprocessor": "src.data.metadata_preprocessor.MetadataPreprocessor",
                     "dim": int(meta_feats.shape[1]),
                     "fit_on_train_only": bool(args.fit_metadata_on_train)},
        "class_to_index": CLASS_TO_INDEX,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "seed": args.seed,
    }

    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "img_ids": img_ids,            # row order is explicit, never inferred
        "patient_ids": patient_ids,
        "labels": labels,
        "img_feats": img_feats,
        "text_feats": text_feats,
        "meta_feats": meta_feats,
        "q_img": q_img, "q_text": q_text, "q_meta": q_meta,
        "texts": texts,
        "manifest": manifest,
    }, out)
    (out.parent / (out.stem + "_manifest.json")).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n[build] saved -> {out}")
    print(f"[build] manifest -> {out.parent / (out.stem + '_manifest.json')}")
    return 0


# ---------------------------------------------------------------------------
def verify(args) -> int:
    """Re-extract a random sample and confirm it matches the stored rows."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    path = PROJECT_ROOT / args.verify
    blob = torch.load(path, map_location="cpu", weights_only=False)
    man = blob["manifest"]
    print(f"[verify] {path.name}: {man['n_rows']} rows, built {man['created']}")
    print(f"[verify] backbone {man['image_backbone']['arch']} "
          f"({man['image_backbone']['weights']})")

    image_dir = PROJECT_ROOT / man["image_dir"]
    net = load_resnet()
    tf = build_transform()

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(man["n_rows"], size=min(args.verify_n, man["n_rows"]), replace=False)
    cosines = []
    for i in idx:
        img_id = blob["img_ids"][int(i)]
        x = tf(Image.open(image_dir / img_id).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            f = net(x).squeeze(0)
        cosines.append(F.cosine_similarity(
            f.unsqueeze(0), blob["img_feats"][int(i)].unsqueeze(0)).item())

    cosines = np.array(cosines)
    print(f"[verify] re-extracted {len(cosines)} rows: "
          f"cos min {cosines.min():.6f}  mean {cosines.mean():.6f}")
    ok = bool(cosines.min() > 0.999)
    print(f"[verify] {'PASS - features are reproducible' if ok else 'FAIL - features drifted'}")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description="Build reproducible multimodal features")
    p.add_argument("--metadata-csv", default="data/raw/metadata.csv")
    p.add_argument("--image-dir", default="data/raw/images")
    p.add_argument("--texts-csv", default="Dummy/Project/data/generated_clinical_texts.csv")
    p.add_argument("--out", default="data/processed/features_v2.pt")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fit-metadata-on-train", action="store_true",
                   help="Fit the metadata scaler/one-hot on the train split only")
    p.add_argument("--verify", metavar="FEATURES_PT", default=None,
                   help="Verify an existing feature file instead of building one")
    p.add_argument("--verify-n", type=int, default=24)
    args = p.parse_args()
    return verify(args) if args.verify else extract(args)


if __name__ == "__main__":
    sys.exit(main())
