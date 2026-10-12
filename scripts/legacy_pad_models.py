"""
scripts/legacy_pad_models.py - The old system's seven models (A-G), re-scored honestly.

The original HealthSphere project trained seven PAD-UFES-20 classifiers:

    A image          B metadata          C free text
    D image+meta     E image+text        F text+meta       G image+text+meta

and reported 84-90% accuracy. Its split was random BY PHOTO, so photos of the same
lesion / patient sit on both sides. This script loads the saved weights and the
cached input features exactly as the old training script produced them, rebuilds the
old split, and scores every model on

    old test        the 345 photos the old project reported on
    lesion-unseen   only test photos whose LESION never appears in training
    patient-unseen  only test photos whose PATIENT never appears in training

It also fuses the seven models with D-GUARD's Devil's Advocate gate (cross-validated
on the patient-unseen photos) to show what the new fusion does with the old team.

Nothing is retrained. Needs ../HealthSphere_archive/Old_system/HealthSphere_project
(code, saved_models/) and ../HealthSphere_archive/data/raw/metadata.csv.

    python scripts/legacy_pad_models.py
Writes outputs/legacy_pad/report.json and probs.npz (used by the app).
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
ROOT = Path(__file__).resolve().parent.parent
OLD = ROOT.parent / "HealthSphere_archive" / "Old_system" / "HealthSphere_project"
RAW = ROOT.parent / "HealthSphere_archive" / "data" / "raw"
CLASSES = ["BCC", "ACK", "NEV", "SEK", "SCC", "MEL"]
NAMES = {"A": "image", "B": "metadata", "C": "free text", "D": "image + metadata",
         "E": "image + text", "F": "text + metadata", "G": "image + text + metadata"}
MALIGNANT = {"BCC", "SCC", "MEL"}


def load_models():
    import torch
    import torch.nn as nn
    sys.path.insert(0, str(OLD))
    from models.encoders import VisionTransformerImageEncoder
    from models.model_A import ModelA
    from models.model_B import ModelB
    from models.model_C import ModelC
    from models.model_D import ModelD
    from models.model_E import ModelE
    from models.model_F import ModelF
    from models.model_G import ModelG

    class TextResidualProjector(nn.Module):          # copied from the old train_all.py
        def __init__(self, in_d=768, out_d=256):
            super().__init__()
            self.fc_in = nn.Sequential(nn.Linear(in_d, out_d), nn.LayerNorm(out_d), nn.GELU(),
                                       nn.Dropout(0.1))
            self.res = nn.Sequential(nn.Linear(out_d, out_d), nn.LayerNorm(out_d), nn.GELU(),
                                     nn.Dropout(0.1), nn.Linear(out_d, out_d),
                                     nn.LayerNorm(out_d), nn.GELU())

        def forward(self, x):
            h = self.fc_in(x)
            return h + self.res(h)

    sm = OLD / "saved_models"
    cache = torch.load(sm / "cached_multimodal_features.pt", map_location="cpu", weights_only=False)
    img_proj = VisionTransformerImageEncoder(embed_dim=256, num_tokens=8, num_heads=8,
                                             num_layers=2, pretrained=False)
    img_proj.load_state_dict(torch.load(sm / "img_proj.pt", map_location="cpu"))
    text_proj = TextResidualProjector()
    text_proj.load_state_dict(torch.load(sm / "text_proj.pt", map_location="cpu"))
    md = cache["meta_input_dim"]
    ctor = {"A": lambda: ModelA(embed_dim=256, num_classes=6),
            "B": lambda: ModelB(meta_input_dim=md, embed_dim=256, num_classes=6),
            "C": lambda: ModelC(embed_dim=256, num_classes=6),
            "D": lambda: ModelD(meta_input_dim=md, embed_dim=256, num_classes=6),
            "E": lambda: ModelE(embed_dim=256, num_classes=6),
            "F": lambda: ModelF(meta_input_dim=md, embed_dim=256, num_classes=6),
            "G": lambda: ModelG(meta_input_dim=md, embed_dim=256, num_classes=6)}
    models, load_report = {}, {}
    for c, make in ctor.items():
        m = make()
        sd = torch.load(sm / f"model_{c}" / f"model_{c}.pt", map_location="cpu")
        res = m.load_state_dict(sd, strict=False)
        load_report[c] = {"missing": len(res.missing_keys), "unexpected": len(res.unexpected_keys)}
        models[c] = m.eval()
    return cache, img_proj.eval(), text_proj.eval(), models, load_report


def predict_all(cache, img_proj, text_proj, models):
    import torch
    import torch.nn.functional as F
    with torch.no_grad():
        zi = img_proj(cache["img_feats"])
        zt = text_proj(cache["text_feats"])
        zm = cache["meta_feats"]
        qi, qm, qt = cache["q_img"], cache["q_meta"], cache["q_text"]
        out = {}
        for c, m in models.items():
            if c == "A":
                lg = m(img_embed=zi)[0]
            elif c == "B":
                lg = m(meta_embed=zm)[0]
            elif c == "C":
                lg = m(text_embed=zt)[0]
            elif c == "D":
                lg = m(img_embed=zi, meta_embed=zm, ev_img=qi, ev_meta=qm)[0]
            elif c == "E":
                lg = m(img_embed=zi, text_embed=zt, ev_img=qi, ev_text=qt)[0]
            elif c == "F":
                lg = m(meta_embed=zm, text_embed=zt, ev_meta=qm, ev_text=qt)[0]
            else:
                lg = m(img_embed=zi, meta_embed=zm, text_embed=zt, ev_img=qi, ev_meta=qm,
                       ev_text=qt)[0]
            out[c] = F.softmax(lg, -1).numpy()
    return out


def main() -> int:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import balanced_accuracy_score, f1_score
    from sklearn.model_selection import train_test_split

    cache, ip, tp, models, load_report = load_models()
    y = cache["labels"].numpy()
    df = pd.read_csv(RAW / "metadata.csv")
    assert len(df) == len(y)
    order_ok = float((df["diagnostic"].map({c: i for i, c in enumerate(CLASSES)}).values == y).mean())
    print(f"label order matches metadata.csv: {order_ok:.1%}")
    P = predict_all(cache, ip, tp, models)

    # the old split, exactly as in the old train_all.py
    tr, te = train_test_split(np.arange(len(y)), train_size=0.85, test_size=0.15,
                              random_state=42, stratify=y)
    L, Pt = set(df.lesion_id.iloc[tr]), set(df.patient_id.iloc[tr])
    lesion_unseen = np.array([df.lesion_id.iloc[i] not in L for i in te])
    patient_unseen = np.array([df.patient_id.iloc[i] not in Pt for i in te])
    subsets = {"old test (as reported)": te, "lesion never in training": te[lesion_unseen],
               "patient never in training": te[patient_unseen]}

    def scores(idx, p):
        pred = p[idx].argmax(1)
        yt = y[idx]
        mal = np.isin(yt, [CLASSES.index(c) for c in MALIGNANT])
        miss = (mal & ~np.isin(pred, [CLASSES.index(c) for c in MALIGNANT])).sum() / max(mal.sum(), 1)
        return {"n": int(len(idx)), "accuracy": float((pred == yt).mean()),
                "macro_f1": float(f1_score(yt, pred, labels=list(range(6)), average="macro",
                                           zero_division=0)),
                "bal_acc": float(balanced_accuracy_score(yt, pred)),
                "missed_cancer": float(miss)}

    rep = {"load": load_report, "label_order_match": order_ok, "models": {},
           "old_reported_accuracy": {}, "subset_sizes": {k: int(len(v)) for k, v in subsets.items()}}
    bench = OLD / "saved_models" / "models_benchmark_summary.csv"
    if bench.exists():
        b = pd.read_csv(bench)
        rep["old_reported_accuracy"] = dict(zip(b["code"], b["accuracy"].round(4)))
    print(f"\n{'model':4} {'inputs':24} " + " ".join(f"{k[:24]:>26}" for k in subsets))
    for c in "ABCDEFG":
        rep["models"][c] = {"inputs": NAMES[c],
                            **{k: scores(idx, P[c]) for k, idx in subsets.items()}}
        print(f"{c:4} {NAMES[c]:24} " + " ".join(
            f"{rep['models'][c][k]['accuracy']:>13.3f} (F1 {rep['models'][c][k]['macro_f1']:.2f})"
            for k in subsets))

    # ---- D-GUARD's gate over the old team, cross-validated on patient-unseen photos ----
    from src.fusion.adaptive import AdaptiveFusion, EvidenceManager
    from src.fusion.gating import LearnedGate, evidence_features
    idx = te[patient_unseen]
    Pm = np.stack([P[c][idx] for c in "ABCDEFG"], 1)
    yy = y[idx]
    pres = np.ones(Pm.shape[:2], bool)
    Q = np.ones(Pm.shape[:2])
    oof = {k: np.zeros((len(idx), 6)) for k in ("average", "gate")}
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(idx))
    folds = 5
    for k in range(folds):
        tst = perm[k::folds]
        trn = np.setdiff1d(perm, tst)
        mgr = EvidenceManager(names=list("ABCDEFG")).fit(Pm[trn], Q[trn], pres[trn], yy[trn])
        pri = AdaptiveFusion(names=list("ABCDEFG")).fit_prior(Pm[trn], pres[trn], yy[trn], 6)
        F_tr = evidence_features(Pm[trn], Q[trn], mgr.reliability(Pm[trn], Q[trn], pres[trn]),
                                 pres[trn], pri.W_prior, None)
        F_te = evidence_features(Pm[tst], Q[tst], mgr.reliability(Pm[tst], Q[tst], pres[tst]),
                                 pres[tst], pri.W_prior, None)
        g = LearnedGate(names=list("ABCDEFG"), l2=0.1).fit(Pm[trn], F_tr, pres[trn], yy[trn])
        oof["gate"][tst] = g.fuse(Pm[tst], F_te, pres[tst])[0]
        oof["average"][tst] = Pm[tst].mean(1)
    rep["fusion_patient_unseen"] = {}
    for k, p in oof.items():
        pred = p.argmax(1)
        rep["fusion_patient_unseen"][k] = {
            "accuracy": float((pred == yy).mean()),
            "macro_f1": float(f1_score(yy, pred, labels=list(range(6)), average="macro",
                                       zero_division=0))}
    print("\nfusion of A-G on patient-unseen photos (5-fold CV):",
          {k: round(v["accuracy"], 3) for k, v in rep["fusion_patient_unseen"].items()})

    out = ROOT / "outputs" / "legacy_pad"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=2))
    np.savez_compressed(out / "probs.npz", y=y, test_idx=te, lesion_unseen=lesion_unseen,
                        patient_unseen=patient_unseen, img_id=df.img_id.values,
                        **{c: P[c].astype(np.float32) for c in "ABCDEFG"})
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
