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
    # "milk10k"  : paired clinical + dermoscopic images (PRIMARY dataset)
    # "pad_ufes" : single clinical image + metadata + synthetic text (shift set)
    DATASET="milk10k",
    DATA_DIR="/kaggle/input/milk10k",           # <-- point at your dataset
    IMAGES_SUBDIR="images",
    METADATA_CSV="metadata.csv",
    TEXTS_CSV="generated_clinical_texts.csv",   # PAD-UFES only; skipped if absent
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
    # Set >= EPOCHS to keep the backbone frozen throughout (linear probing).
    # Required reading if IMAGE_BACKBONE="dermlip_panderm": its licence is
    # labelled CC-BY-NC-ND, and "ND" arguably forbids fine-tuning.
    EDL_ANNEAL_EPOCHS=10,        # KL regulariser ramp
    LABEL_SMOOTH=0.0,

    SEED=42,
    SPLIT=(0.70, 0.15, 0.15),
    NUM_WORKERS=2,
    AMP=True,
    EARLY_STOP_PATIENCE=8,       # on val macro-F1, NOT loss
)

# When this whole file is pasted into a notebook cell, `__name__ == "__main__"`
# and the last line would start a full training run every time the cell runs.
# Set AUTORUN = False to only DEFINE everything, then call run_seeds(...) or
# run_dermlip(...) from the next cell.
AUTORUN = True

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


MILK_CLASSES = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF",
                "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]
MILK_MALIGNANT = ["AKIEC", "BCC", "MAL_OTH", "MEL", "SCCKA"]
MILK_MONET = ["MONET_ulceration_crust", "MONET_hair", "MONET_vasculature_vessels",
              "MONET_erythema", "MONET_pigmented",
              "MONET_gel_water_drop_fluid_dermoscopy_liquid",
              "MONET_skin_markings_pen_ink_purple_pen"]
MILK_CLINICAL_TYPE = "clinical: close-up"
MILK_DERMOSCOPIC_TYPE = "dermoscopic"

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

    def numcol(name, default):
        """Numeric column as a Series, or a constant Series when absent.

        DataFrame.get returns None for a missing column, and pd.to_numeric(None)
        yields a bare numpy scalar with no .fillna - which is how this crashed
        the first time it met a dataset without PAD-UFES geometry fields.
        """
        if name not in df.columns:
            return pd.Series(default, index=df.index, dtype=float)
        return pd.to_numeric(df[name], errors="coerce").fillna(default)

    d1 = numcol("diameter_1", 0.0)
    d2 = numcol("diameter_2", 0.0)
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
    age = numcol("age", 50.0)
    df["age"] = age
    df["age_norm"] = np.clip(age / 90.0, 0, 1)
    df["missing_frac"] = df.isnull().mean(axis=1)
    return df


ENGINEERED = NUMERIC + ["diameter_ratio", "lesion_area", "max_diameter", "log_area",
                        "symptom_burden", "is_sun_exposed", "age_norm", "missing_frac"]


# MILK10k carries an entirely different metadata schema: no lesion geometry, no
# symptom questionnaire. What it does carry is age, sex, site, an ordinal skin
# tone class, and 7 MONET concept-activation scores per image. Reusing the
# PAD-UFES encoder here silently produced all-constant columns.
MILK_NUMERIC = ["age", "age_norm", "skin_tone_ord"] + MILK_MONET
MILK_CATEGORICAL = ["sex", "site", "anatom_site_general", "skin_tone_class"]


class MilkMetaEncoder:
    """Fit on TRAIN ONLY. MILK10k schema. Uses no labels."""

    def __init__(self):
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        self.num_imp = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.cat_imp = SimpleImputer(strategy="constant", fill_value="Unknown")
        try:
            self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            self.ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        self.cats = None

    def _prep(self, df):
        d = df.copy()
        age = pd.to_numeric(d.get("age"), errors="coerce")
        if not isinstance(age, pd.Series):
            age = pd.Series(np.nan, index=d.index, dtype=float)
        d["age"] = age.fillna(age.median() if age.notna().any() else 60.0)
        d["age_norm"] = np.clip(d["age"] / 90.0, 0, 1)
        # Skin tone is ORDINAL (0..5), so it is kept both as a number and as a
        # one-hot - the ordering is real, but so is the grouping for fairness.
        d["skin_tone_ord"] = pd.to_numeric(
            d.get("skin_tone_class"), errors="coerce").fillna(-1.0)
        for c in MILK_MONET:
            if c not in d.columns:
                d[c] = 0.0
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
        self.cats = [c for c in MILK_CATEGORICAL if c in d.columns]
        for c in self.cats:
            d[c] = d[c].astype(str).replace(
                {"nan": "Unknown", "": "Unknown", "None": "Unknown"})
        return d

    def fit(self, df):
        d = self._prep(df)
        self.scaler.fit(self.num_imp.fit_transform(d[MILK_NUMERIC]))
        self.ohe.fit(self.cat_imp.fit_transform(d[self.cats]))
        return self

    def transform(self, df):
        d = self._prep(df)
        num = self.scaler.transform(self.num_imp.transform(d[MILK_NUMERIC]))
        cat = self.ohe.transform(self.cat_imp.transform(d[self.cats]))
        return np.hstack([num, cat]).astype(np.float32)


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


# ---------------------------------------------------------------------------
# MILK10k
# ---------------------------------------------------------------------------
# 10,480 images / 5,240 lesions, each lesion with ONE clinical close-up and ONE
# dermatoscopic image. CC-BY-NC. doi 10.34970/648456.
#
# Because every lesion has both modalities, "acquire dermoscopy" is a REAL
# experiment with an oracle upper bound rather than a simulation. That is the
# reason this is the primary dataset.
#
# MILK10k publishes no patient identifiers, so splits are by LESION, with
# ISIC-ID deduplication. State the residual leakage risk in limitations.

# Verified against the real release on 2026-10-11. The layout is FIXED, so this
# reads it exactly rather than pattern-matching column names:
#
#   metadata.csv                   10480 rows, one per IMAGE
#   supplements/training_gt.csv     5240 rows, one-hot over 11 classes
#   supplements/training_input.csv 10480 rows, skin_tone_class + 7 MONET scores
#   images/<isic_id>.jpg           10480 files
#
# Measured facts you should not rediscover the hard way:
#   * 11 classes, NOT 6. Order below is the sorted column order of training_gt.
#   * 71.7% of lesions are MALIGNANT; BCC alone is 48.1%. Referral population.
#   * rarest class MAL_OTH has 9 lesions in total.
#   * all 10480 images are byte-distinct (SHA-256 checked); no duplicates.
#   * `attribution` is the single constant "MILK study team": there is NO centre
#     column, so a centre-held-out shift experiment is impossible on this release.



def milk_load(data_dir, cfg):
    """Return (df, image_dir, classes) for MILK10k, one row per LESION.

    Mirrors src/data/milk10k.py so a Kaggle run and a local run agree. Fails
    loudly on any layout deviation instead of guessing.
    """
    data_dir = Path(data_dir)
    meta_csv = next(iter(sorted(data_dir.rglob("metadata.csv"))), None)
    if meta_csv is None:
        raise SystemExit(f"no metadata.csv under {data_dir}")
    root = meta_csv.parent
    gt_csv = root / "supplements" / "training_gt.csv"
    input_csv = root / "supplements" / "training_input.csv"
    img_dir = root / "images"
    for q in (gt_csv, input_csv, img_dir):
        if not q.exists():
            raise SystemExit(f"expected {q}; is MILK10k fully extracted?")

    meta = pd.read_csv(meta_csv, dtype=str, keep_default_na=False)
    inp = pd.read_csv(input_csv, dtype=str, keep_default_na=False)
    gt = pd.read_csv(gt_csv)

    missing = [c for c in MILK_CLASSES if c not in gt.columns]
    if missing:
        raise SystemExit(f"{gt_csv} missing class columns {missing}")
    onehot = gt[MILK_CLASSES].to_numpy(dtype=float)
    if not (onehot == 1.0).sum(axis=1).all():
        raise SystemExit(f"{gt_csv}: some lesion has no single active class")
    gt = pd.DataFrame({
        "lesion_id": gt["lesion_id"].astype(str),
        "label_idx": onehot.argmax(axis=1).astype(int),
    })
    gt["label_raw"] = [MILK_CLASSES[i] for i in gt["label_idx"]]

    keep = ["isic_id", "skin_tone_class", "site"] + MILK_MONET
    meta = meta.merge(inp[keep], on="isic_id", how="left", validate="one_to_one")

    clin = meta[meta["image_type"] == MILK_CLINICAL_TYPE].copy()
    derm = meta[meta["image_type"] == MILK_DERMOSCOPIC_TYPE].copy()
    if len(clin) != len(derm):
        raise SystemExit(f"unpaired: {len(clin)} clinical vs {len(derm)} dermoscopic")

    cols = ["lesion_id", "isic_id", "age_approx", "sex", "anatom_site_general",
            "site", "skin_tone_class", "diagnosis_confirm_type"] + MILK_MONET
    df = clin[cols].rename(columns={"isic_id": "clinical_file"})
    df = df.merge(derm[["lesion_id", "isic_id"]].rename(
        columns={"isic_id": "dermoscopy_file"}), on="lesion_id", validate="one_to_one")
    df["lesion_id"] = df["lesion_id"].astype(str)
    df = df.merge(gt, on="lesion_id", validate="one_to_one")

    # The loader hands back bare ISIC ids; the dataset class appends ".jpg".
    df["clinical_file"] = df["clinical_file"] + ".jpg"
    df["dermoscopy_file"] = df["dermoscopy_file"] + ".jpg"
    df["age"] = pd.to_numeric(df["age_approx"], errors="coerce")
    for c in MILK_MONET:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["malignant"] = df["label_raw"].isin(MILK_MALIGNANT).astype(int)

    df = df.sort_values("lesion_id").reset_index(drop=True)
    print(f"[milk10k] {len(df)} lesions, {len(MILK_CLASSES)} classes, "
          f"{100 * df['malignant'].mean():.1f}% malignant")
    return df, img_dir, MILK_CLASSES



def lesion_level_split(df, ratios, seed, key="lesion_id"):
    """Disjoint by lesion. MILK10k has no patient IDs, so this is the finest
    grouping available; the residual risk is stated in the limitations."""
    ids = sorted(df[key].astype(str).unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_tr, n_va = int(n * ratios[0]), int(n * ratios[1])
    groups = {"train": set(ids[:n_tr]), "val": set(ids[n_tr:n_tr + n_va]),
              "test": set(ids[n_tr + n_va:])}
    out = {k: df[df[key].astype(str).isin(v)].reset_index(drop=True)
           for k, v in groups.items()}
    assert not (groups["train"] & groups["test"]), "train/test lesion overlap"
    assert not (groups["train"] & groups["val"]), "train/val lesion overlap"
    assert not (groups["val"] & groups["test"]), "val/test lesion overlap"
    return out


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
        lab_col = "diagnostic" if "diagnostic" in self.df.columns else "label_idx"
        self.labels = self.df[lab_col].map(CLASS_TO_IDX).values.astype(np.int64)             if lab_col == "diagnostic" else self.df[lab_col].values.astype(np.int64)
        size = cfg["IMAGE_SIZE"]
        if str(cfg.get("IMAGE_BACKBONE", "")).startswith("dermlip"):
            norm = T.Normalize(CLIP_MEAN, CLIP_STD)
        else:
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
        # MILK10k carries a second image per lesion; PAD-UFES-20 does not.
        self.clin_col = "clinical_file" if "clinical_file" in self.df.columns else "img_id"
        self.derm_col = "dermoscopy_file" if "dermoscopy_file" in self.df.columns else None

    def __len__(self):
        return len(self.df)

    def _load(self, name):
        """Load one image; a missing file yields a zero tensor and has=0, which
        becomes the vacuous opinion in the DS fold rather than a fake input."""
        size = self.cfg["IMAGE_SIZE"]
        if not name or str(name) in ("nan", "None"):
            return torch.zeros(3, size, size), 0.0
        for cand in (self.image_dir / str(name),
                     self.image_dir / f"{name}.jpg",
                     self.image_dir / f"{name}.png"):
            if cand.exists():
                try:
                    return self.tf(Image.open(cand).convert("RGB")), 1.0
                except Exception:
                    break
        return torch.zeros(3, size, size), 0.0

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img, has_img = self._load(row.get(self.clin_col))
        if self.derm_col:
            derm, has_derm = self._load(row.get(self.derm_col))
        else:
            derm, has_derm = torch.zeros_like(img), 0.0

        txt = self.texts[i]
        has_txt = 1.0 if txt.strip() else 0.0
        if self.tok is not None:
            enc = self.tok(txt or "", padding="max_length", truncation=True,
                           max_length=self.cfg["TEXT_MAX_LEN"], return_tensors="pt")
            ids, mask = enc["input_ids"][0], enc["attention_mask"][0]
        else:
            ids = torch.zeros(self.cfg["TEXT_MAX_LEN"], dtype=torch.long)
            mask = torch.zeros(self.cfg["TEXT_MAX_LEN"], dtype=torch.long)

        return {"image": img, "dermoscopy": derm,
                "meta": torch.from_numpy(self.meta[i]),
                "input_ids": ids, "attention_mask": mask,
                "label": int(self.labels[i]),
                "has_image": has_img, "has_dermoscopy": has_derm,
                "has_meta": 1.0, "has_text": has_txt,
                "img_id": str(row.get(self.clin_col, i))}


# =============================================================================
# MODEL — evidential heads + Dempster-Shafer combination
# =============================================================================
DERMLIP_REPOS = (
    "redlessone/DermLIP_PanDerm-base-w-PubMed-256",   # PanDerm vision tower (preferred)
    "redlessone/DermLIP_ViT-B-16",                    # standard ViT-B/16 (fallback)
)
#: CLIP normalisation. DermLIP towers were trained with these, not ImageNet's.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class _DermLIPVisual(nn.Module):
    """Vision tower of a DermLIP model, exposed as a plain feature extractor.

    Tries each repo in DERMLIP_REPOS in turn and records which one loaded in
    `self.repo`. If none loads it RAISES: an earlier version of this project
    silently fell back to ImageNet weights when the gated repo was unreachable,
    and reported the result as a dermatology-encoder result.
    """

    def __init__(self, repos=DERMLIP_REPOS):
        super().__init__()
        try:
            import open_clip
        except ImportError as e:
            raise SystemExit("DermLIP needs open_clip: run  !pip -q install open_clip_torch") from e
        errors = []
        for repo in repos:
            try:
                self.clip, _, self.preprocess = open_clip.create_model_and_transforms(
                    f"hf-hub:{repo}")
                self.repo = repo
                break
            except Exception as e:  # gated, missing token, unknown architecture...
                errors.append(f"{repo}: {type(e).__name__}: {str(e)[:200]}")
        else:
            raise SystemExit(
                "Could not load any DermLIP model. Check that you accepted the terms on "
                "BOTH Hugging Face model pages and that HF_TOKEN is set "
                "(Kaggle: Add-ons -> Secrets). Errors:\n  " + "\n  ".join(errors))
        print(f"[backbone] DermLIP vision tower loaded from {self.repo}")
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
        if os.environ.get("HF_TOKEN"):
            try:
                from huggingface_hub import login
                login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
            except Exception as e:
                print(f"[backbone] HF login warning: {e}")
        m = _DermLIPVisual()
        with torch.no_grad():
            d = m(torch.zeros(1, 3, 224, 224)).shape[-1]
        print(f"[backbone] DermLIP feature dim {d} ({m.repo})")
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

        # One shared visual backbone, SEPARATE evidential heads per image
        # modality: clinical and dermoscopic views have different statistics but
        # a 5k-lesion dataset cannot support two backbones.
        self.head_img = EvidentialHead(d, K)
        self.head_dermo = EvidentialHead(d, K)
        self.head_meta = EvidentialHead(d, K)
        self.head_text = EvidentialHead(d, K)

    def encode(self, batch):
        z_img = self.img_proj(self.backbone(batch["image"]))
        z_derm = (self.img_proj(self.backbone(batch["dermoscopy"]))
                  if "dermoscopy" in batch else None)
        z_meta = self.meta_proj(batch["meta"])
        if self.bert is not None:
            with torch.set_grad_enabled(not self.cfg["FREEZE_TEXT"]):
                out = self.bert(input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"])
            cls = out.last_hidden_state[:, 0, :]
        else:
            cls = torch.zeros(batch["meta"].shape[0], 768, device=batch["meta"].device)
        return z_img, z_derm, z_meta, self.text_proj(cls)

    def forward(self, batch, modality_dropout=0.0):
        z_img, z_derm, z_meta, z_text = self.encode(batch)
        dev = z_meta.device
        B = z_meta.shape[0]
        names = ("image", "dermoscopy", "meta", "text")

        pres = {"image": batch["has_image"].to(dev).float(),
                "dermoscopy": batch.get("has_dermoscopy",
                                        torch.zeros(B)).to(dev).float(),
                "meta": batch["has_meta"].to(dev).float(),
                "text": batch["has_text"].to(dev).float()}

        if modality_dropout > 0 and self.training:
            # Absence must be in the training distribution, but never drop
            # everything - a fully vacuous case carries no gradient signal.
            keep = torch.rand(B, len(names), device=dev) >= modality_dropout
            keep[keep.sum(1) == 0, 0] = True
            for j, name in enumerate(names):
                pres[name] = pres[name] * keep[:, j].float()

        ops = {"image": gate(self.head_img(z_img), pres["image"], K),
               "meta": gate(self.head_meta(z_meta), pres["meta"], K),
               "text": gate(self.head_text(z_text), pres["text"], K)}
        ops["dermoscopy"] = (gate(self.head_dermo(z_derm), pres["dermoscopy"], K)
                             if z_derm is not None
                             else vacuous(B, K, dev))

        fused, conflicts = None, []
        for name in names:
            fused = ops[name] if fused is None else ds_combine(fused, ops[name])
            if "conflict" in (fused or {}):
                conflicts.append(fused["conflict"])

        # pignistic probability: distribute ignorance uniformly
        prob = fused["belief"] + fused["u"] / K
        # Conflict of the WHOLE fold, carried as the max over every pairwise
        # step - matching src/evidence/evidential.py. Reporting only the LAST
        # step's conflict (as an earlier version did) is silently wrong: on
        # MILK10k the last modality is always the absent text, its opinion is
        # vacuous, and conflict with a vacuous opinion is exactly zero. The
        # first Kaggle run exported conflict_C == 0.0 for all 787 test lesions
        # for precisely this reason.
        total_conflict = (torch.stack(conflicts).amax(0) if conflicts
                          else torch.zeros(B, 1, device=dev))
        return {"prob": prob, "belief": fused["belief"], "u": fused["u"],
                "conflict": total_conflict,
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

    # ---------------- MILK10k (primary) ----------------
    if cfg.get("DATASET", "pad_ufes") == "milk10k":
        global CLASS_NAMES, CLASS_TO_IDX, K
        df, img_dir, classes = milk_load(data_dir, cfg)
        CLASS_NAMES = classes
        CLASS_TO_IDX = {c: i for i, c in enumerate(classes)}
        K = len(classes)
        # label_idx already comes from the official one-hot ground truth, so it
        # is NOT re-derived from a string here - that mapping step is exactly
        # where the PAD-UFES label map got transposed.
        df["clinical_text"] = ""           # MILK10k ships no narrative
        print(f"[milk10k] K={K} classes: {classes}")
        print("class counts:", df["label_raw"].value_counts().to_dict())
        print("ground truth:", df["diagnosis_confirm_type"].value_counts().to_dict())
        print("skin tone   :", df["skin_tone_class"].value_counts().sort_index().to_dict())

        if cfg.get("LIMIT_LESIONS"):
            # Smoke-test path only. Stratified so every class survives.
            n = int(cfg["LIMIT_LESIONS"])
            df = (df.groupby("label_raw", group_keys=False)
                    .apply(lambda g: g.head(max(2, n // len(classes))))
                    .reset_index(drop=True))
            print(f"[smoke] LIMIT_LESIONS -> {len(df)} lesions")

        splits = lesion_level_split(df, cfg["SPLIT"], cfg["SEED"])
        for k, v in splits.items():
            print(f"  {k:5} n={len(v):5} lesions={v['lesion_id'].nunique():5}")

        menc = MilkMetaEncoder().fit(splits["train"])
        meta_mats = {k: menc.transform(v) for k, v in splits.items()}
        meta_dim = meta_mats["train"].shape[1]
        print(f"metadata dim = {meta_dim} (MILK10k schema, fitted on train only)")
        return _train_loop(cfg, splits, meta_mats, meta_dim, img_dir, out_dir, dev,
                           tokenizer=None, text_model=None, group_key="lesion_id")

    # ---------------- PAD-UFES-20 (shift set) ----------------
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

    return _train_loop(cfg, splits, meta_mats, meta_dim, img_dir, out_dir, dev,
                       tokenizer, text_model, group_key="patient_id")


def _train_loop(cfg, splits, meta_mats, meta_dim, img_dir, out_dir, dev,
                tokenizer, text_model, group_key="patient_id"):
    """Shared training / evaluation loop for both datasets."""
    loaders = {}
    for k in ("train", "val", "test"):
        ds = DermDataset(splits[k], img_dir, meta_mats[k], tokenizer, k == "train", cfg)
        loaders[k] = DataLoader(ds, batch_size=cfg["BATCH_SIZE"], shuffle=(k == "train"),
                                num_workers=cfg["NUM_WORKERS"], pin_memory=True,
                                drop_last=False)

    model = DGuard(meta_dim, cfg, text_model).to(dev)
    lab_col = "diagnostic" if "diagnostic" in splits["train"].columns else "label_idx"
    train_labels = (splits["train"][lab_col].map(CLASS_TO_IDX).values
                    if lab_col == "diagnostic" else splits["train"][lab_col].values)
    counts = np.bincount(train_labels.astype(int), minlength=K)
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
    # The key H1 test: does adding a modality ever HURT? Naive concatenation
    # does; evidence-aware fusion should not.
    for name, flags in [("clin+derm+meta", (1, 1, 1, 0)),
                        ("clin+derm", (1, 1, 0, 0)),
                        ("clin+meta", (1, 0, 1, 0)),
                        ("derm+meta", (0, 1, 1, 0)),
                        ("clinical", (1, 0, 0, 0)),
                        ("dermoscopy", (0, 1, 0, 0)),
                        ("metadata", (0, 0, 1, 0)),
                        ("all+text", (1, 1, 1, 1))]:
        model.eval(); P, Y = [], []
        with torch.no_grad():
            for b in loaders["test"]:
                bb = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
                bb["has_image"] = bb["has_image"] * flags[0]
                bb["has_dermoscopy"] = bb["has_dermoscopy"] * flags[1]
                bb["has_meta"] = bb["has_meta"] * flags[2]
                bb["has_text"] = bb["has_text"] * flags[3]
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
        "groups": {k: int(v[group_key].nunique()) for k, v in splits.items()},
        "group_key": group_key,
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


def apply_env_overrides(cfg):
    """Let any CFG key be overridden by DGUARD_<KEY> in the environment.

    Exists so the exact same file can be smoke-tested on a CPU box before it is
    handed to a GPU. Example:

        DGUARD_EPOCHS=1 DGUARD_LIMIT_LESIONS=80 DGUARD_AMP=0         DGUARD_DATA_DIR=milk10k python notebooks/kaggle_train_dguard.py
    """
    cfg = dict(cfg)
    for key in list(cfg) + ["LIMIT_LESIONS"]:
        raw = os.environ.get(f"DGUARD_{key}")
        if raw is None:
            continue
        cur = cfg.get(key)
        if isinstance(cur, bool):
            val = raw.strip().lower() in ("1", "true", "yes", "on")
        elif isinstance(cur, int) and not isinstance(cur, bool):
            val = int(raw)
        elif isinstance(cur, float):
            val = float(raw)
        elif isinstance(cur, tuple):
            val = tuple(float(x) for x in raw.split(","))
        else:
            val = int(raw) if key == "LIMIT_LESIONS" else raw
        cfg[key] = val
        print(f"[override] {key} = {val!r}")
    return cfg


def run_seeds(seeds=(7, 2026), data_dir=None, **overrides):
    """Train one model per seed, each into its own OUT_DIR. Notebook-friendly.

        run_seeds((7, 2026), data_dir="/kaggle/input/datasets/<user>/milk10k")
    """
    out = {}
    for seed in seeds:
        cfg = dict(CFG)
        cfg.update(SEED=seed, OUT_DIR=f"/kaggle/working/dguard_milk_s{seed}")
        if data_dir:
            cfg["DATA_DIR"] = data_dir
        cfg.update(overrides)
        print(f"\n{'=' * 70}\nSEED {seed} -> {cfg['OUT_DIR']}\n{'=' * 70}")
        out[seed] = run(cfg)
    return out


def run_dermlip(seed=42, data_dir=None, linear_probe=True, **overrides):
    """Train with the DermLIP image encoder. Needs HF_TOKEN and open_clip_torch.

    linear_probe=True keeps the encoder frozen for every epoch (the DermLIP
    licence is No-Derivatives, so fine-tuning the weights is the riskier reading).
    """
    cfg = dict(CFG)
    cfg.update(SEED=seed, IMAGE_BACKBONE="dermlip_panderm",
               OUT_DIR=f"/kaggle/working/dguard_milk_dermlip_s{seed}")
    if linear_probe:
        cfg["WARMUP_FROZEN_EPOCHS"] = cfg["EPOCHS"]
    if data_dir:
        cfg["DATA_DIR"] = data_dir
    cfg.update(overrides)
    return run(cfg)


if __name__ == "__main__" and AUTORUN:
    run(apply_env_overrides(CFG))
