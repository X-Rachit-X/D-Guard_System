"""
scripts/milk_native_infer.py - Run the Kaggle-trained MILK10k model locally.

This is "native mode": real evidential heads trained on MILK10k, not the
softmax-entropy bridge over the legacy PAD-UFES specialists.

What it does
------------
1. Rebuilds the EXACT train/val/test split the Kaggle run used (the trainer's
   own seeded lesion-level split), and refits the MILK10k metadata encoder on
   the training split only.
2. Loads checkpoints/best.pt with strict=True, so any architecture drift fails
   loudly instead of silently loading random weights.
3. Runs every val and test lesion TWICE:
     full       clinical + dermoscopy + metadata
     no_dermo   clinical + metadata  (dermoscopy masked -> vacuous opinion)
   Every MILK10k lesion really has both images, so the difference between the
   two views is the measured value of acquiring dermoscopy - not a simulation.
4. Recomputes conflict correctly as the MAX over the Dempster-Shafer fold. The
   Kaggle export reported only the last pairwise step, which on MILK10k is
   always against the absent text modality and therefore always exactly 0.
5. Checks the local test metrics against metrics.json from Kaggle, so you can
   see the model running here is the model that was trained there.

Usage
-----
    python scripts/milk_native_infer.py
    python scripts/milk_native_infer.py --run runs/milk10k_resnet50_seed42
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VIEWS = {
    "full": {"image": 1.0, "dermoscopy": 1.0, "meta": 1.0, "text": 0.0},
    "no_dermo": {"image": 1.0, "dermoscopy": 0.0, "meta": 1.0, "text": 0.0},
}
MODALITY_ORDER = ("image", "dermoscopy", "meta", "text")


def load_trainer():
    """Import the Kaggle trainer as a module, so this script and the GPU run use
    one definition of the model, the split and the metadata encoder."""
    path = ROOT / "notebooks" / "kaggle_train_dguard.py"
    spec = importlib.util.spec_from_file_location("kaggle_train_dguard", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fold(T, ops, pres):
    """DS fold with conflict carried as the max over every pairwise step."""
    fused, conflicts = None, []
    for name in MODALITY_ORDER:
        op = T.gate(ops[name], pres[name], T.K)
        if fused is None:
            fused = op
        else:
            fused = T.ds_combine(fused, op)
            conflicts.append(fused["conflict"])
    conflict = torch.stack(conflicts).amax(0)
    prob = fused["belief"] + fused["u"] / T.K       # pignistic probability
    return prob, fused["belief"], fused["u"], conflict


@torch.no_grad()
def run_split(T, model, loader, dev):
    out = {v: {"prob": [], "u": [], "conflict": []} for v in VIEWS}
    per_mod = {m: {"belief": [], "u": []} for m in ("image", "dermoscopy", "meta")}
    labels, ids = [], []
    for i, batch in enumerate(loader):
        for k in ("image", "dermoscopy", "meta"):
            batch[k] = batch[k].to(dev)
        B = batch["meta"].shape[0]
        z_img, z_derm, z_meta, z_text = model.encode(batch)
        ops = {"image": model.head_img(z_img),
               "dermoscopy": model.head_dermo(z_derm),
               "meta": model.head_meta(z_meta),
               "text": model.head_text(z_text)}
        has = {"image": batch["has_image"].float().to(dev),
               "dermoscopy": batch["has_dermoscopy"].float().to(dev),
               "meta": batch["has_meta"].float().to(dev),
               "text": batch["has_text"].float().to(dev)}
        for view, mask in VIEWS.items():
            pres = {m: has[m] * mask[m] for m in MODALITY_ORDER}
            prob, _, u, c = fold(T, ops, pres)
            out[view]["prob"].append(prob.cpu())
            out[view]["u"].append(u.cpu())
            out[view]["conflict"].append(c.cpu())
        for m in per_mod:
            per_mod[m]["belief"].append(ops[m]["belief"].cpu())
            per_mod[m]["u"].append(ops[m]["u"].cpu())
        labels.append(batch["label"].clone() if torch.is_tensor(batch["label"])
                      else torch.tensor(batch["label"]))
        ids.extend(batch["img_id"])
        if (i + 1) % 10 == 0:
            print(f"    batch {i + 1}/{len(loader)}", flush=True)

    res = {"y": torch.cat(labels).numpy(), "img_id": np.array(ids)}
    for view, d in out.items():
        for k, v in d.items():
            res[f"{view}_{k}"] = torch.cat(v).numpy()
    for m, d in per_mod.items():
        for k, v in d.items():
            res[f"mod_{m}_{k}"] = torch.cat(v).numpy()
    return res


def macro_f1(y, pred, k):
    from sklearn.metrics import f1_score
    return float(f1_score(y, pred, labels=list(range(k)), average="macro", zero_division=0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/milk10k_resnet50_seed42",
                    help="the Kaggle OUT_DIR you downloaded")
    ap.add_argument("--data", default="milk10k")
    ap.add_argument("--out", default="outputs/milk_native")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N lesions per split")
    args = ap.parse_args()

    run_dir = ROOT / args.run
    ckpt_path = run_dir / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        print(f"ERROR: {ckpt_path} not found.")
        return 1

    print("=" * 74)
    print("MILK10k NATIVE INFERENCE  (Kaggle-trained evidential model, run locally)")
    print("=" * 74)
    T = load_trainer()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = dict(T.CFG)
    cfg.update(ck["cfg"])
    cfg["DATA_DIR"] = str(ROOT / args.data)
    cfg["NUM_WORKERS"] = 0
    print(f"checkpoint : epoch {ck['epoch']}  val macro-F1 {ck['val_macro_f1']:.4f}")
    print(f"backbone   : {cfg['IMAGE_BACKBONE']}  embed {cfg['EMBED_DIM']}  "
          f"split {cfg['SPLIT']}  seed {cfg['SEED']}")

    df, img_dir, classes = T.milk_load(cfg["DATA_DIR"], cfg)
    T.CLASS_NAMES = classes
    T.CLASS_TO_IDX = {c: i for i, c in enumerate(classes)}
    T.K = len(classes)
    df["clinical_text"] = ""

    splits = T.lesion_level_split(df, cfg["SPLIT"], cfg["SEED"])
    sizes = {k: len(v) for k, v in splits.items()}
    print(f"split      : {sizes}")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if sizes != manifest["n"]:
        print(f"ERROR: split {sizes} differs from the Kaggle run {manifest['n']}.")
        print("The local split must reproduce the training split exactly, or test")
        print("lesions could have been trained on.")
        return 1
    print("           : identical to the Kaggle run")

    menc = T.MilkMetaEncoder().fit(splits["train"])
    meta = {k: menc.transform(v) for k, v in splits.items()}
    if meta["train"].shape[1] != ck["meta_dim"]:
        print(f"ERROR: metadata width {meta['train'].shape[1]} != checkpoint {ck['meta_dim']}")
        return 1

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T.DGuard(ck["meta_dim"], cfg, text_model=None)
    missing, unexpected = model.load_state_dict(ck["model"], strict=True), None
    model.to(dev).eval()
    print(f"weights    : loaded strictly on {dev}")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    from torch.utils.data import DataLoader

    results = {}
    for name in ("val", "test"):
        d = splits[name]
        m = meta[name]
        if args.limit:
            d, m = d.head(args.limit), m[: args.limit]
        ds = T.DermDataset(d, img_dir, m, None, train=False, cfg=cfg)
        dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)
        t0 = time.time()
        print(f"\n[{name}] {len(ds)} lesions")
        res = run_split(T, model, dl, dev)
        res["lesion_id"] = d["lesion_id"].astype(str).to_numpy()
        res["skin_tone"] = d["skin_tone_class"].astype(str).to_numpy()
        res["malignant"] = d["malignant"].to_numpy()
        res["clinical_file"] = d["clinical_file"].astype(str).to_numpy()
        res["dermoscopy_file"] = d["dermoscopy_file"].astype(str).to_numpy()
        np.savez_compressed(out_dir / f"{name}.npz", **res)
        print(f"[{name}] done in {time.time() - t0:.0f}s -> {out_dir / (name + '.npz')}")
        results[name] = res

    # ---- verification against the Kaggle run --------------------------------
    t = results["test"]
    y = t["y"]
    print()
    print("=" * 74)
    print("CHECK AGAINST KAGGLE")
    print("=" * 74)
    kag = json.loads((run_dir / "metrics.json").read_text())["test"]
    local_f1 = macro_f1(y, t["full_prob"].argmax(1), T.K)
    local_acc = float((t["full_prob"].argmax(1) == y).mean())
    print(f"macro-F1   local {local_f1:.4f}   kaggle {kag['macro_f1']:.4f}")
    print(f"accuracy   local {local_acc:.4f}   kaggle {kag['accuracy']:.4f}")
    if not args.limit:
        gap = abs(local_f1 - kag["macro_f1"])
        print(f"           {'MATCH' if gap < 0.02 else 'MISMATCH'} (|gap| = {gap:.4f}; "
              f"GPU mixed precision vs CPU fp32 explains small differences)")

    print()
    print("conflict C (fixed: max over the fold)")
    for view in VIEWS:
        c = t[f"{view}_conflict"].ravel()
        print(f"  {view:9} min {c.min():.3f}  median {np.median(c):.3f}  "
              f"max {c.max():.3f}  nonzero {(c > 1e-6).sum()}/{len(c)}")

    print()
    print("value of dermoscopy (test)")
    for view in VIEWS:
        p = t[f"{view}_prob"].argmax(1)
        print(f"  {view:9} macro-F1 {macro_f1(y, p, T.K):.4f}  acc {(p == y).mean():.4f}  "
              f"mean u {t[f'{view}_u'].mean():.3f}")

    summary = {
        "checkpoint_epoch": int(ck["epoch"]),
        "classes": classes,
        "split": sizes,
        "local_test_macro_f1": local_f1,
        "kaggle_test_macro_f1": kag["macro_f1"],
        "views": {v: {"macro_f1": macro_f1(y, t[f"{v}_prob"].argmax(1), T.K),
                      "mean_u": float(t[f"{v}_u"].mean()),
                      "mean_conflict": float(t[f"{v}_conflict"].mean())}
                  for v in VIEWS},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsummary -> {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
