"""
scripts/build_specialists.py - Run every specialist over MILK10k train/val/test once.

Saves, per split, everything the evidence manager and the adaptive fusion need:

    z_img, z_derm, z_meta    the trained model's 256-d embeddings (kNN, OOD)
    pad_prob                 the ORIGINAL project's Model A on the clinical photo
    q_clin, q_derm           image-quality measurements (sharpness, brightness, ...)

The MILK10k heads' own opinions are already in outputs/milk_native/{val,test}.npz.

    python scripts/build_specialists.py          # ~10 min on CPU
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

from src.agents.quality import measure  # noqa: E402
from src.native import NativeEngine  # noqa: E402
from src.specialists import PadLegacySpecialist  # noqa: E402

OUT = ROOT / "outputs" / "specialists"
Q_KEYS = ("sharpness", "brightness", "contrast", "glare")


@torch.no_grad()
def run_split(eng, pad, df, name, batch=24, with_pad=True):
    z_img, z_derm, z_meta, pad_p, q_c, q_d = [], [], [], [], [], []
    rows = list(df.itertuples(index=False))
    t0 = time.time()
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        clin = [Image.open(eng.img_dir / r.clinical_file).convert("RGB") for r in chunk]
        derm = [Image.open(eng.img_dir / r.dermoscopy_file).convert("RGB") for r in chunk]
        xi = torch.stack([eng.tf(im) for im in clin])
        xd = torch.stack([eng.tf(im) for im in derm])
        meta = torch.from_numpy(np.concatenate([eng.meta_encoder.transform(eng._meta_row({
            "age": None if np.isnan(r.age) else r.age, "sex": r.sex, "site": r.site,
            "anatom_site_general": r.anatom_site_general,
            "skin_tone_class": r.skin_tone_class,
            **{c: getattr(r, c) for c in eng.monet_median}})) for r in chunk]))
        m = eng.model
        z_img.append(m.img_proj(m.backbone(xi)).numpy())
        z_derm.append(m.img_proj(m.backbone(xd)).numpy())
        z_meta.append(m.meta_proj(meta).numpy())
        if with_pad and pad.available:
            pad_p.append(pad.predict_batch(clin))
        q_c.append([[measure(im)[k] for k in Q_KEYS] for im in clin])
        q_d.append([[measure(im)[k] for k in Q_KEYS] for im in derm])
        if (i // batch) % 20 == 0:
            print(f"  [{name}] {i + len(chunk)}/{len(rows)}  ({time.time() - t0:.0f}s)",
                  flush=True)
    out = {
        "lesion_id": df["lesion_id"].astype(str).to_numpy(),
        "y": df["label_idx"].to_numpy(),
        "z_img": np.concatenate(z_img), "z_derm": np.concatenate(z_derm),
        "z_meta": np.concatenate(z_meta),
        "q_clin": np.array(sum(q_c, [])), "q_derm": np.array(sum(q_d, [])),
    }
    if pad_p:
        out["pad_prob"] = np.concatenate(pad_p)
    return out


def main() -> int:
    eng = NativeEngine()
    pad = PadLegacySpecialist(eng.classes)
    print(f"PAD-UFES Model A available: {pad.available}")
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("val", "test", "train"):
        res = run_split(eng, pad, eng.splits[name], name, with_pad=(name != "train"))
        np.savez_compressed(OUT / f"{name}.npz", **res)
        print(f"[{name}] saved {OUT / (name + '.npz')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
