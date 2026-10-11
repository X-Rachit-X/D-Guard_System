"""
D-GUARD — Kaggle training script (single file, paste into one notebook cell or
run as a utility script).

WHAT THIS TRAINS
----------------
1. A fine-tuned image encoder.  This is the biggest available win: with a FROZEN
   ImageNet ResNet50 the image modality reaches only ~0.48 macro-F1, while
   metadata alone reaches ~0.65.  Fine-tuning needs a GPU, which is why this
   runs on Kaggle rather than locally.
2. Per-modality EVIDENTIAL heads (Dirichlet evidence -> belief + ignorance).
3. Dempster-Shafer combination across modalities, with absent modalities
   represented as the vacuous opinion (the identity element of the DS rule).
4. Modality dropout, so absence is inside the training distribution.

It evaluates honestly: disjoint patient-level splits, no metric post-processing,
per-class breakdown, calibration, and a risk-coverage curve.

WHAT TO SEND BACK
-----------------
The whole `OUT_DIR` (default /kaggle/working/dguard_out).  It contains:
    checkpoints/best.pt            model weights
    metrics.json                   all test metrics
    predictions_test.csv           per-case predictions (lets any metric be recomputed)
    history.json                   real training curves
    manifest.json                  versions, seeds, split hashes, config

DATASET
-------
PAD-UFES-20 (Pacheco et al., Data in Brief 2020).
  Authoritative source : Mendeley Data, DOI 10.17632/zr7vgbcyr2.1
  Kaggle               : search "PAD-UFES-20"; several mirrors exist, and their
                         directory layouts differ.

RECOMMENDED: upload YOUR OWN local copy (data/raw/images + data/raw/metadata.csv)
as a private Kaggle dataset.  The previous generation of this project was
invalidated partly because cached features could not be matched to any available
image copy; using the identical bytes you already have removes that whole class
of problem.  Set DATA_DIR to point at it.

Expected layout:
    <DATA_DIR>/metadata.csv            # PAD-UFES-20 metadata, incl. img_id, patient_id, diagnostic
    <DATA_DIR>/images/<img_id>         # one file per row of metadata.csv
Optional:
    <DATA_DIR>/generated_clinical_texts.csv   # columns: img_id, clinical_text

Accelerator: GPU T4 x2 or P100.  Runtime ~40-70 min for 25 epochs at 224px.
"""

# =============================================================================
# CONFIG
# =============================================================================
CFG = dict(
    DATA_DIR="/kaggle/input/pad-ufes-20",       # <-- point at your dataset
    IMAGES_SUBDIR="images",
    METADATA_CSV="metadata.csv",
    TEXTS_CSV="generated_clinical_texts.csv",   # optional; skipped if absent
    OUT_DIR="/kaggle/working/dguard_out",

    # dermlip_panderm is RECOMMENDED (dermatology foundation model, public weights).
    # Requires: pip install open_clip_torch, and an HF token -- the repo is gated.
    IMAGE_BACKBONE="resnet50",   # resnet50 | efficientnet_b0 | convnext_tiny | dermlip_panderm
    IMAGE_SIZE=224,
    EMBED_DIM=256,

    TEXT_MODEL="emilyalsentzer/Bio_ClinicalBERT",
    TEXT_MAX_LEN=128,
    FREEZE_TEXT=True,            # BERT frozen; only its projection trains. Flip if time allows.

    EPOCHS=25,
    BATCH_SIZE=32,
    LR_HEAD=1e-3,
    LR_BACKBONE=1e-4,            # 0.1x -- fine-tuning, not training from scratch
    WEIGHT_DECAY=1e-4,
    WARMUP_FROZEN_EPOCHS=3,      # train heads first, then unfreeze the backbone
    GRAD_CLIP=1.0,
    MODALITY_DROPOUT=0.25,
    EDL_ANNEAL_EPOCHS=10,        # KL regulariser ramp
    LABEL_SMOOTH=0.0,

    SEED=42,
    SPLIT=(0.70, 0.15, 0.15),
    NUM_WORKERS=2,
    AMP=True,
    EARLY_STOP_PATIENCE=8,       # on val macro-F1, NOT loss
)

CLASS_NAMES = ["BCC", "ACK", "NEV", "SEK", "SCC", "MEL"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
K = len(CLASS_NAMES)

import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# =============================================================================
# DATA
# =============================================================================
NUMERIC = ["age", "diameter_1", "diameter_2"]
CATEGORICAL = ["smoke", "drink", "background_father", "background_mother", "pesticide",
               "gender", "skin_cancer_history", "cancer_history", "has_piped_water",
               "has_sewage_system", "fitspatrick", "region", "itch", "grew", "hurt",
               "changed", "bleed", "elevation", "biopsed"]
SUN_EXPOSED = {"FACE", "NOSE", "EAR", "NECK", "SCALP", "LIP", "FOREARM", "HAND", "ARM"}


def engineer(df):
    """Lesion geometry, symptom burden, sun exposure. Deterministic functions of the
    raw fields -- uses no labels."""
    df = df.copy()
    d1 = pd.to_numeric(df.get("diameter_1"), errors="coerce").fillna(0.0)
    d2 = pd.to_numeric(df.get("diameter_2"), errors="coerce").fillna(0.0)
    df["diameter_1"], df["diameter_2"] = d1, d2
    df["diameter_ratio"] = np.where(d2 > 0, d1 / (d2 + 1e-5), 1.0)
    df["lesion_area"] = np.pi * (d1 / 2) * (d2 / 2)
    df["max_diameter"] = np.maximum(d1, d2)
    df["log_area"] = np.log1p(df["lesion_area"])

    sym = ["bleed", "hurt", "grew", "changed", "elevation", "itch"]
    score = np.zeros(len(df))
    for c in sym:
        if c in df.columns:
            score += (df[c].astype(str).str.upper() == "TRUE").astype(float).values
    df["symptom_burden"] = score

    region = df["region"].astype(str).str.upper() if "region" in df else pd.Series("UNK", index=df.index)
    df["is_sun_exposed"] = region.isin(SUN_EXPOSED).astype(float)
    age = pd.to_numeric(df.get("age"), errors="coerce").fillna(50.0)
    df["age"] = age
    df["age_norm"] = np.clip(age / 90.0, 0, 1)
    df["missing_frac"] = df.isnull().mean(axis=1)
    return df


ENGINEERED = NUMERIC + ["diameter_ratio", "lesion_area", "max_diameter", "log_area",
                        "symptom_burden", "is_sun_exposed", "age_norm", "missing_frac"]


class MetaEncoder:
    """Fit on TRAIN ONLY. Standardise numerics, one-hot categoricals."""

    def __init__(self):
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        self.num_imp = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.cat_imp = SimpleImputer(strategy="constant", fill_value="Unknown")
        try:
            self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:                       # older sklearn
            self.ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        self.cats = None

    def _prep(self, df):
        df = engineer(df)
        self.cats = [c for c in CATEGORICAL if c in df.columns]
        for c in self.cats:
            df[c] = df[c].astype(str).replace({"nan": "Unknown", "UNK": "Unknown",
                                               "None": "Unknown", "": "Unknown"})
        return df

    def fit(self, df):
        d = self._prep(df)
        self.scaler.fit(self.num_imp.fit_transform(d[ENGINEERED]))
        self.ohe.fit(self.cat_imp.fit_transform(d[self.cats]))
        return self

    def transform(self, df):
        d = self._prep(df)
        num = self.scaler.transform(self.num_imp.transform(d[ENGINEERED]))
        cat = self.ohe.transform(self.cat_imp.transform(d[self.cats]))
        return np.hstack([num, cat]).astype(np.float32)


def patient_level_split(df, ratios, seed):
    """Disjoint by patient_id. Asserted, because an overlapping split is what
    invalidated the previous results."""
    pats = sorted(df["patient_id"].astype(str).unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(pats)
    n = len(pats)
    n_tr, n_va = int(n * ratios[0]), int(n * ratios[1])
    groups = {"train": set(pats[:n_tr]),
              "val": set(pats[n_tr:n_tr + n_va]),
              "test": set(pats[n_tr + n_va:])}
    out = {k: df[df["patient_id"].astype(str).isin(v)].reset_index(drop=True)
           for k, v in groups.items()}
    assert not (groups["train"] & groups["test"]), "train/test patient overlap"
    assert not (groups["train"] & groups["val"]), "train/val patient overlap"
    assert not (groups["val"] & groups["test"]), "val/test patient overlap"
    assert sum(len(v) for v in out.values()) == len(df)
    return out


class DermDataset(Dataset):
    def __init__(self, df, image_dir, meta_mat, tokenizer, train, cfg):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.meta = meta_mat
        self.tok = tokenizer
        self.train = train
        self.cfg = cfg
        self.labels = self.df["diagnostic"].map(CLASS_TO_IDX).values.astype(np.int64)
        size = cfg["IMAGE_SIZE"]
        norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.tf = (T.Compose([T.RandomResizedCrop(size, scale=(0.7, 1.0)),
                              T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                              T.RandomRotation(20),
                              T.ColorJitter(0.15, 0.15, 0.15, 0.03),
                              T.ToTensor(), norm])
                   if train else
                   T.Compose([T.Resize((size, size)), T.ToTensor(), norm]))
        self.texts = (self.df["clinical_text"].fillna("").astype(str).tolist()
                      if "clinical_text" in self.df.columns else [""] * len(self.df))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        p = self.image_dir / str(row["img_id"])
        try:
            img = self.tf(Image.open(p).convert("RGB"))
            has_img = 1.0
        except Exception:
            img = torch.zeros(3, self.cfg["IMAGE_SIZE"], self.cfg["IMAGE_SIZE"])
            has_img = 0.0

        txt = self.texts[i]
        has_txt = 1.0 if txt.strip() else 0.0
        if self.tok is not None:
            enc = self.tok(txt or "", padding="max_length", truncation=True,
                           max_length=self.cfg["TEXT_MAX_LEN"], return_tensors="pt")
            ids, mask = enc["input_ids"][0], enc["attention_mask"][0]
        else:
            ids = torch.zeros(self.cfg["TEXT_MAX_LEN"], dtype=torch.long)
            mask = torch.zeros(self.cfg["TEXT_MAX_LEN"], dtype=torch.long)

        return {"image": img, "meta": torch.from_numpy(self.meta[i]),
                "input_ids": ids, "attention_mask": mask,
                "label": int(self.labels[i]),
                "has_image": has_img, "has_meta": 1.0, "has_text": has_txt,
                "img_id": str(row["img_id"])}


# =============================================================================
# MODEL — evidential heads + Dempster-Shafer combination
# =============================================================================
class _DermLIPVisual(nn.Module):
    """Vision tower of DermLIP_PanDerm, exposed as a plain feature extractor."""

    def __init__(self, repo):
        super().__init__()
        import open_clip
        self.clip, _, self.preprocess = open_clip.create_model_and_transforms(f"hf-hub:{repo}")
        self.clip.eval()

    def forward(self, x):
        return self.clip.encode_image(x)


def build_image_backbone(name):
    """Returns (module, feature_dim).

    `dermlip_panderm` is a dermatology FOUNDATION model (PanDerm, Nature Medicine 2025),
    self-supervised on >2M skin images. Strongly preferred over ImageNet weights: frozen
    ImageNet ResNet50 features reach only ~0.48 macro-F1 on this task, vs ~0.65 for
    metadata alone, so the image pathway is encoder-limited.

    Three caveats, all real:
      * the HF repo is GATED -- accept the terms on the model page and provide an HF
        token (Kaggle: Add-ons -> Secrets -> HF_TOKEN). The original implementation of
        this project failed silently here and fell back to ImageNet.
      * licence is non-commercial and labelled inconsistently (cc-by-nc-4.0 in the repo
        metadata, CC-BY-NC-ND-4.0 in the model details). "ND" arguably forbids
        fine-tuning, so LINEAR PROBING is the safer reading -- set
        WARMUP_FROZEN_EPOCHS >= EPOCHS to keep the tower frozen throughout.
      * PanDerm's pretraining corpus spans 11 institutions. CHECK whether PAD-UFES-20 is
        in it before claiming a clean result -- otherwise a linear probe is partly
        train-on-test.
    """
    import torchvision.models as tvm
    if name == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        d = m.fc.in_features; m.fc = nn.Identity()
    elif name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT)
        d = m.classifier[1].in_features; m.classifier = nn.Identity()
    elif name == "convnext_tiny":
        m = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.DEFAULT)
        d = 768; m.classifier[2] = nn.Identity()
    elif name == "dermlip_panderm":
        # pip install open_clip_torch ; huggingface-cli login
        m = _DermLIPVisual("redlessone/DermLIP_PanDerm-base-w-PubMed-256")
        with torch.no_grad():
            d = m(torch.zeros(1, 3, 224, 224)).shape[-1]
        print(f"[backbone] DermLIP_PanDerm loaded, feature dim {d}")
    else:
        raise ValueError(f"unknown IMAGE_BACKBONE {name!r}")
    return m, d


class EvidentialHead(nn.Module):
    """z -> Dirichlet evidence -> (belief, ignorance).

    u = K/S is the per-case, per-modality ignorance. It replaces the hand-tuned
    reliability formula in the old system: a blurred image or an empty narrative
    produces little evidence, so S -> K and u -> 1.
    """

    def __init__(self, d_in, k, hidden=128, p_drop=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden),
                                 nn.GELU(), nn.Dropout(p_drop), nn.Linear(hidden, k))
        self.k = k

    def forward(self, z):
        e = F.softplus(self.net(z)).clamp(max=1e4)
        alpha = e + 1.0
        S = alpha.sum(-1, keepdim=True)
        return {"evidence": e, "alpha": alpha, "S": S,
                "belief": e / S, "u": self.k / S, "prob": alpha / S}


def vacuous(batch, k, device):
    """What an ABSENT modality contributes: b=0, u=1. Identity for DS combination."""
    return {"belief": torch.zeros(batch, k, device=device),
            "u": torch.ones(batch, 1, device=device)}


def ds_combine(a, b, eps=1e-8):
    """Dempster-Shafer combination. Commutative and associative -> fold for M>2."""
    ba, ua = a["belief"], a["u"]
    bb, ub = b["belief"], b["u"]
    sa, sb = ba.sum(-1, keepdim=True), bb.sum(-1, keepdim=True)
    # conflict: belief mass on mutually exclusive class pairs
    C = (sa * sb - (ba * bb).sum(-1, keepdim=True)).clamp(0.0, 1.0 - 1e-6)
    denom = (1.0 - C).clamp(min=eps)
    return {"belief": (ba * bb + ba * ub + bb * ua) / denom,
            "u": (ua * ub) / denom, "conflict": C}


def gate(op, present, k):
    """Replace an opinion by the vacuous one wherever `present` is 0."""
    m = present.view(-1, 1)
    return {"belief": op["belief"] * m,
            "u": op["u"] * m + (1.0 - m)}


class DGuard(nn.Module):
    def __init__(self, meta_dim, cfg, text_model=None):
        super().__init__()
        self.cfg = cfg
        d = cfg["EMBED_DIM"]
        self.backbone, d_img = build_image_backbone(cfg["IMAGE_BACKBONE"])
        self.img_proj = nn.Sequential(nn.Linear(d_img, d), nn.LayerNorm(d), nn.GELU())

        self.meta_proj = nn.Sequential(
            nn.Linear(meta_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(512, d), nn.LayerNorm(d), nn.GELU())

        self.bert = text_model
        if self.bert is not None and cfg["FREEZE_TEXT"]:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.text_proj = nn.Sequential(nn.Linear(768, d), nn.LayerNorm(d), nn.GELU())

        self.head_img = EvidentialHead(d, K)
        self.head_meta = EvidentialHead(d, K)
        self.head_text = EvidentialHead(d, K)

    def encode(self, batch):
        z_img = self.img_proj(self.backbone(batch["image"]))
        z_meta = self.meta_proj(batch["meta"])
        if self.bert is not None:
            with torch.set_grad_enabled(not self.cfg["FREEZE_TEXT"]):
                out = self.bert(input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"])
            cls = out.last_hidden_state[:, 0, :]
        else:
            cls = torch.zeros(batch["meta"].shape[0], 768, device=batch["meta"].device)
        return z_img, z_meta, self.text_proj(cls)

    def forward(self, batch, modality_dropout=0.0):
        z_img, z_meta, z_text = self.encode(batch)
        dev = z_meta.device
        B = z_meta.shape[0]

        pres = {"image": batch["has_image"].to(dev).float(),
                "meta": batch["has_meta"].to(dev).float(),
                "text": batch["has_text"].to(dev).float()}

        if modality_dropout > 0 and self.training:
            # Absence must be in the training distribution, but never drop everything.
            keep = torch.rand(B, 3, device=dev) >= modality_dropout
            keep[keep.sum(1) == 0, 0] = True
            for j, name in enumerate(("image", "meta", "text")):
                pres[name] = pres[name] * keep[:, j].float()

        ops = {"image": gate(self.head_img(z_img), pres["image"], K),
               "meta": gate(self.head_meta(z_meta), pres["meta"], K),
               "text": gate(self.head_text(z_text), pres["text"], K)}

        fused, conflicts = None, []
        for name in ("image", "meta", "text"):
            fused = ops[name] if fused is None else ds_combine(fused, ops[name])
            if "conflict" in (fused or {}):
                conflicts.append(fused["conflict"])

        # pignistic probability: distribute ignorance uniformly
        prob = fused["belief"] + fused["u"] / K
        return {"prob": prob, "belief": fused["belief"], "u": fused["u"],
                "conflict": conflicts[-1] if conflicts else torch.zeros(B, 1, device=dev),
                "per_modality": ops, "presence": pres}


# =============================================================================
# LOSS
# =============================================================================
def edl_loss(alpha, y_onehot, anneal, class_w=None):
    """Bayes risk of squared error under the Dirichlet + annealed KL to uniform."""
    S = alpha.sum(-1, keepdim=True)
    p = alpha / S
    err = (y_onehot - p) ** 2
    var = p * (1 - p) / (S + 1.0)
    per_class = err + var
    if class_w is not None:
        per_class = per_class * class_w.view(1, -1)
    loss = per_class.sum(-1).mean()

    # strip the true class, then push remaining evidence toward uniform
    a_t = y_onehot + (1 - y_onehot) * alpha
    k = alpha.shape[-1]
    S_t = a_t.sum(-1, keepdim=True)
    kl = (torch.lgamma(S_t).squeeze(-1) - torch.lgamma(a_t).sum(-1)
          - math.lgamma(k)
          + ((a_t - 1) * (torch.digamma(a_t) - torch.digamma(S_t))).sum(-1))
    return loss + anneal * kl.mean()


def total_loss(out, y, anneal, class_w):
    y1 = F.one_hot(y, K).float()
    L = 0.0
    # Supervise each modality head -- this is what makes u_m meaningful per modality.
    for name, op in out["per_modality"].items():
        present = out["presence"][name].view(-1, 1)
        if present.sum() < 1:
            continue
        # Recover alpha from the gated opinion:  S = K/u,  e = b*S,  alpha = e+1.
        # Going via (belief, u) rather than keeping alpha means an absent modality
        # arrives here as the vacuous opinion and contributes alpha = 1 (no evidence).
        S = K / op["u"].clamp(min=1e-6)
        alpha = op["belief"] * S + 1.0
        L = L + edl_loss(alpha, y1, anneal, class_w) * present.mean()
    # and the fused opinion
    S_f = (K / out["u"].clamp(min=1e-6))
    L = L + 2.0 * edl_loss(out["belief"] * S_f + 1.0, y1, anneal, class_w)
    return L


# =============================================================================
# METRICS
# =============================================================================
def compute_metrics(y, prob):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 classification_report, confusion_matrix, f1_score,
                                 roc_auc_score)
    pred = prob.argmax(1)
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(K))).tolist(),
        "per_class": classification_report(y, pred, labels=list(range(K)),
                                           target_names=CLASS_NAMES, output_dict=True,
                                           zero_division=0),
    }
    try:
        out["auroc_macro_ovr"] = float(roc_auc_score(y, prob, multi_class="ovr",
                                                     average="macro"))
    except Exception:
        out["auroc_macro_ovr"] = None

    conf = prob.max(1)
    correct = (pred == y).astype(float)
    # adaptive (equal-mass) ECE
    order = np.argsort(conf)
    bins = np.array_split(order, 15)
    ece = sum(len(b) / len(y) * abs(correct[b].mean() - conf[b].mean())
              for b in bins if len(b))
    out["adaptive_ece_15"] = float(ece)
    onehot = np.eye(K)[y]
    out["brier"] = float(((prob - onehot) ** 2).sum(1).mean())
    return out


def risk_coverage(y, prob, score):
    """Risk-coverage curve + AURC. The primary metric family for an abstaining system."""
    pred = prob.argmax(1)
    wrong = (pred != y).astype(float)
    order = np.argsort(-score)                 # most confident first
    w = wrong[order]
    cov = np.arange(1, len(y) + 1) / len(y)
    risk = np.cumsum(w) / np.arange(1, len(y) + 1)
    return {"coverage": cov.tolist(), "risk": risk.tolist(),
            "aurc": float(np.mean(risk)),
            "risk_at_cov": {f"{c:.1f}": float(risk[max(0, int(c * len(y)) - 1)])
                            for c in (0.2, 0.4, 0.6, 0.8, 1.0)}}


# =============================================================================
# TRAIN
# =============================================================================
def run(cfg):
    set_seed(cfg["SEED"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["OUT_DIR"]); (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    print(f"device={dev}  torch={torch.__version__}")

    data_dir = Path(cfg["DATA_DIR"])
    meta_path = data_dir / cfg["METADATA_CSV"]
    if not meta_path.exists():
        hits = list(data_dir.rglob(cfg["METADATA_CSV"]))
        if not hits:
            raise SystemExit(f"metadata.csv not found under {data_dir}. "
                             f"Contents: {[p.name for p in data_dir.iterdir()][:20]}")
        meta_path = hits[0]
    img_dir = meta_path.parent / cfg["IMAGES_SUBDIR"]
    if not img_dir.is_dir():
        cands = [p for p in meta_path.parent.rglob("*") if p.is_dir() and
                 any(f.suffix.lower() in (".png", ".jpg", ".jpeg") for f in list(p.glob("*"))[:5])]
        if not cands:
            raise SystemExit(f"no image directory found under {meta_path.parent}")
        img_dir = cands[0]
    print(f"metadata={meta_path}\nimages={img_dir}")

    df = pd.read_csv(meta_path)
    df = df[df["diagnostic"].isin(CLASS_TO_IDX)].reset_index(drop=True)
    print(f"cases={len(df)}  patients={df['patient_id'].nunique()}")
    print("class counts:", df["diagnostic"].value_counts().to_dict())

    tpath = meta_path.parent / cfg["TEXTS_CSV"]
    if tpath.exists():
        t = pd.read_csv(tpath)
        col = next((c for c in ("clinical_text", "filled_description", "text")
                    if c in t.columns), None)
        if col:
            df = df.merge(t[["img_id", col]].rename(columns={col: "clinical_text"}),
                          on="img_id", how="left")
            print(f"narratives merged from {tpath.name}")
    if "clinical_text" not in df.columns:
        df["clinical_text"] = ""
        print("no narratives -> text modality will be vacuous")

    splits = patient_level_split(df, cfg["SPLIT"], cfg["SEED"])
    for k, v in splits.items():
        print(f"  {k:5} n={len(v):5} patients={v['patient_id'].nunique():5}")

    menc = MetaEncoder().fit(splits["train"])
    meta_mats = {k: menc.transform(v) for k, v in splits.items()}
    meta_dim = meta_mats["train"].shape[1]
    print(f"metadata dim = {meta_dim} (fitted on train only)")

    tokenizer, text_model = None, None
    if df["clinical_text"].astype(str).str.strip().ne("").any():
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg["TEXT_MODEL"])
        text_model = AutoModel.from_pretrained(cfg["TEXT_MODEL"])

    loaders = {}
    for k in ("train", "val", "test"):
        ds = DermDataset(splits[k], img_dir, meta_mats[k], tokenizer, k == "train", cfg)
        loaders[k] = DataLoader(ds, batch_size=cfg["BATCH_SIZE"], shuffle=(k == "train"),
                                num_workers=cfg["NUM_WORKERS"], pin_memory=True,
                                drop_last=False)

    model = DGuard(meta_dim, cfg, text_model).to(dev)
    counts = np.bincount(splits["train"]["diagnostic"].map(CLASS_TO_IDX).values, minlength=K)
    class_w = torch.tensor(len(splits["train"]) / (K * np.maximum(counts, 1)),
                           dtype=torch.float32, device=dev)
    print("class weights:", class_w.tolist())

    head_params = [p for n, p in model.named_parameters()
                   if not n.startswith("backbone") and p.requires_grad]
    bb_params = [p for n, p in model.named_parameters()
                 if n.startswith("backbone") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": head_params, "lr": cfg["LR_HEAD"]},
                             {"params": bb_params, "lr": cfg["LR_BACKBONE"]}],
                            weight_decay=cfg["WEIGHT_DECAY"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["EPOCHS"], eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["AMP"] and dev.type == "cuda")

    for p in model.backbone.parameters():
        p.requires_grad = False
    print(f"backbone frozen for first {cfg['WARMUP_FROZEN_EPOCHS']} epochs")

    history = {"train_loss": [], "val_macro_f1": [], "val_acc": [], "lr": []}
    best_f1, best_epoch, bad = -1.0, -1, 0

    @torch.no_grad()
    def evaluate(split):
        model.eval()
        P, Y, U, C, IDS = [], [], [], [], []
        for b in loaders[split]:
            bb = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            with torch.cuda.amp.autocast(enabled=cfg["AMP"] and dev.type == "cuda"):
                o = model(bb)
            P.append(o["prob"].float().cpu()); Y.append(b["label"])
            U.append(o["u"].float().cpu()); C.append(o["conflict"].float().cpu())
            IDS.extend(b["img_id"])
        return (torch.cat(P).numpy(), torch.cat(Y).numpy(),
                torch.cat(U).numpy().ravel(), torch.cat(C).numpy().ravel(), IDS)

    for ep in range(1, cfg["EPOCHS"] + 1):
        if ep == cfg["WARMUP_FROZEN_EPOCHS"] + 1:
            for p in model.backbone.parameters():
                p.requires_grad = True
            print(f"[epoch {ep}] backbone unfrozen at {cfg['LR_BACKBONE']}")

        model.train()
        anneal = min(1.0, ep / max(cfg["EDL_ANNEAL_EPOCHS"], 1))
        tot, nb, t0 = 0.0, 0, time.time()
        for b in loaders["train"]:
            bb = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v)
                  for k, v in b.items()}
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg["AMP"] and dev.type == "cuda"):
                o = model(bb, modality_dropout=cfg["MODALITY_DROPOUT"])
                loss = total_loss(o, bb["label"], anneal, class_w)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["GRAD_CLIP"])
            scaler.step(opt); scaler.update()
            tot += loss.item(); nb += 1

        sched.step()
        prob, y, u, c, _ = evaluate("val")
        m = compute_metrics(y, prob)
        history["train_loss"].append(tot / max(nb, 1))
        history["val_macro_f1"].append(m["macro_f1"])
        history["val_acc"].append(m["accuracy"])
        history["lr"].append(opt.param_groups[0]["lr"])
        print(f"ep {ep:3d}/{cfg['EPOCHS']} loss {tot/max(nb,1):.4f} "
              f"val_acc {m['accuracy']:.4f} val_macroF1 {m['macro_f1']:.4f} "
              f"u {u.mean():.3f} conflict {c.mean():.3f} ({time.time()-t0:.0f}s)", flush=True)

        # EARLY STOPPING ON VAL MACRO-F1. Values are recorded as measured -- never clipped.
        if m["macro_f1"] > best_f1:
            best_f1, best_epoch, bad = m["macro_f1"], ep, 0
            torch.save({"model": model.state_dict(), "epoch": ep, "val_macro_f1": best_f1,
                        "meta_dim": meta_dim, "cfg": cfg},
                       out_dir / "checkpoints" / "best.pt")
        else:
            bad += 1
            if bad >= cfg["EARLY_STOP_PATIENCE"]:
                print(f"early stop at epoch {ep} (best {best_epoch}, F1 {best_f1:.4f})")
                break

    print(f"\nloading best epoch {best_epoch} (val macroF1 {best_f1:.4f})")
    model.load_state_dict(torch.load(out_dir / "checkpoints" / "best.pt",
                                     map_location=dev)["model"])

    results = {}
    for split in ("val", "test"):
        prob, y, u, c, ids = evaluate(split)
        met = compute_metrics(y, prob)
        met["risk_coverage"] = risk_coverage(y, prob, 1.0 - u)
        met["mean_ignorance"] = float(u.mean())
        met["mean_conflict"] = float(c.mean())
        results[split] = met
        if split == "test":
            pd.DataFrame({"img_id": ids, "y_true": y, "y_pred": prob.argmax(1),
                          "ignorance_u": u, "conflict_C": c,
                          **{f"p_{n}": prob[:, i] for i, n in enumerate(CLASS_NAMES)}}
                         ).to_csv(out_dir / "predictions_test.csv", index=False)

    # Availability ablation: the key test of H1 -- does it hold up when modalities vanish?
    print("\navailability ablation (test):")
    avail = {}
    for name, flags in [("image+meta+text", (1, 1, 1)), ("image+meta", (1, 1, 0)),
                        ("image+text", (1, 0, 1)), ("meta+text", (0, 1, 1)),
                        ("image", (1, 0, 0)), ("meta", (0, 1, 0)), ("text", (0, 0, 1))]:
        model.eval(); P, Y = [], []
        with torch.no_grad():
            for b in loaders["test"]:
                bb = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
                bb["has_image"] = bb["has_image"] * flags[0]
                bb["has_meta"] = bb["has_meta"] * flags[1]
                bb["has_text"] = bb["has_text"] * flags[2]
                with torch.cuda.amp.autocast(enabled=cfg["AMP"] and dev.type == "cuda"):
                    o = model(bb)
                P.append(o["prob"].float().cpu()); Y.append(b["label"])
        mm = compute_metrics(torch.cat(Y).numpy(), torch.cat(P).numpy())
        avail[name] = {"accuracy": mm["accuracy"], "macro_f1": mm["macro_f1"]}
        print(f"  {name:<18} acc {mm['accuracy']:.4f}  macroF1 {mm['macro_f1']:.4f}")
    results["availability_ablation"] = avail

    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": cfg, "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
        "meta_dim": meta_dim, "class_names": CLASS_NAMES,
        "n": {k: len(v) for k, v in splits.items()},
        "patients": {k: int(v["patient_id"].nunique()) for k, v in splits.items()},
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "notes": "Patient-level disjoint splits. Metrics as measured; no clipping, "
                 "no target-matching, no prediction rewriting.",
    }
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    t = results["test"]
    print("\n" + "=" * 62)
    print(f"TEST  acc {t['accuracy']:.4f} | bal-acc {t['balanced_accuracy']:.4f} | "
          f"macroF1 {t['macro_f1']:.4f}")
    print(f"      AUROC {t['auroc_macro_ovr']} | ECE {t['adaptive_ece_15']:.4f} | "
          f"Brier {t['brier']:.4f} | AURC {t['risk_coverage']['aurc']:.4f}")
    print("per-class F1:", {n: round(t['per_class'][n]['f1-score'], 3) for n in CLASS_NAMES})
    print("=" * 62)
    print(f"\nSend back the whole directory: {out_dir}")
    return results


if __name__ == "__main__":
    run(CFG)
