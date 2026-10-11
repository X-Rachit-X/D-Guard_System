"""
scripts/build_milk_index.py - Similar-case index over MILK10k TRAINING lesions.

Embeds every training lesion's clinical photo and metadata with the trained
model's own 256-d projections and stores them in a FAISS cosine index, together
with each lesion's confirmed diagnosis and image paths (for display).

Only the training split is indexed, and the build asserts it: a neighbour from
the validation or test split would put a held-out label in front of the reader.

    python scripts/build_milk_index.py          # ~3-4 min on CPU
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.agents.librarian import SPEC  # noqa: E402
from src.native import NativeEngine  # noqa: E402
from src.retrieval.rag_index import RAGIndex  # noqa: E402

OUT = ROOT / "outputs" / "milk_rag" / "index_train"


@torch.no_grad()
def main() -> int:
    eng = NativeEngine()
    train = eng.splits["train"]
    held_out = set(eng.splits["val"]["lesion_id"]) | set(eng.splits["test"]["lesion_id"])
    assert not (set(train["lesion_id"]) & held_out), "train overlaps val/test"

    cases, t0 = [], time.time()
    B = 32
    rows = list(train.itertuples(index=False))
    for i in range(0, len(rows), B):
        chunk = rows[i:i + B]
        imgs = torch.stack([eng.tf(Image.open(eng.img_dir / r.clinical_file).convert("RGB"))
                            for r in chunk])
        meta = torch.from_numpy(np.concatenate(
            [eng.meta_encoder.transform(eng._meta_row({
                "age": None if np.isnan(r.age) else r.age, "sex": r.sex, "site": r.site,
                "anatom_site_general": r.anatom_site_general,
                "skin_tone_class": r.skin_tone_class,
                **{c: getattr(r, c) for c in eng.monet_median}})) for r in chunk]))
        z_img = eng.model.img_proj(eng.model.backbone(imgs))
        z_meta = eng.model.meta_proj(meta)
        for j, r in enumerate(chunk):
            cases.append({
                "lesion_id": str(r.lesion_id),
                "diagnosis": eng.classes[int(r.label_idx)],
                "label": int(r.label_idx),
                "clinical_path": str(eng.img_dir / r.clinical_file),
                "dermoscopy_path": str(eng.img_dir / r.dermoscopy_file),
                "age": None if np.isnan(r.age) else float(r.age),
                "site": r.site,
                "clinical_embedding": z_img[j].numpy(),
                "metadata_embedding": z_meta[j].numpy(),
            })
        if (i // B) % 20 == 0:
            print(f"  {len(cases)}/{len(rows)}  ({time.time() - t0:.0f}s)", flush=True)

    idx = RAGIndex(embedding_spec=SPEC)
    idx.source_split = "train"
    idx.add_training_cases(cases)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    idx.save(str(OUT))

    back = RAGIndex().load(str(OUT))
    assert back.source_split == "train"
    assert not ({c["lesion_id"] for c in back._training_cases} & held_out)
    print(f"index: {len(cases)} training lesions, no val/test lesion present -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
