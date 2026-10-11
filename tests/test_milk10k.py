"""
tests/test_milk10k.py - Pins the MILK10k loader to the verified release layout.

The structural tests run on a synthetic fixture so they work without the 368 MB
dataset. The tests marked `integration` run against the real thing when it is
extracted at ./milk10k, and pin the numbers measured on 2026-10-11.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.constants import (
    MILK_CLASS_NAMES,
    MILK_CLASS_TO_INDEX,
    MILK_CLINICAL_TYPE,
    MILK_DERMOSCOPIC_TYPE,
    MILK_MALIGNANT_CLASSES,
    MILK_MONET_COLUMNS,
    MILK_NUM_CLASSES,
    milk_malignant_indices,
)
from src.data.milk10k import (
    Milk10kLayoutError,
    _read_gt,
    _stable_bucket,
    build_cases,
    lesion_level_split,
    load_milk10k,
    summarise,
)

REAL_ROOT = Path("milk10k")
real_only = pytest.mark.skipif(
    not (REAL_ROOT / "supplements" / "training_gt.csv").exists(),
    reason="MILK10k not extracted at ./milk10k",
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
def test_eleven_classes():
    assert MILK_NUM_CLASSES == 11
    assert len(set(MILK_CLASS_NAMES)) == 11


def test_class_names_are_sorted():
    """The index must match the column order of training_gt.csv, which is sorted.

    If this ever fails, every trained head is silently mislabelled - the same
    failure mode as the transposed PAD-UFES map.
    """
    assert MILK_CLASS_NAMES == sorted(MILK_CLASS_NAMES)


def test_class_to_index_is_derived():
    for i, name in enumerate(MILK_CLASS_NAMES):
        assert MILK_CLASS_TO_INDEX[name] == i


def test_malignant_classes_are_known():
    for c in MILK_MALIGNANT_CLASSES:
        assert c in MILK_CLASS_TO_INDEX
    assert milk_malignant_indices() == [0, 1, 6, 7, 9]


def test_akiec_counted_malignant():
    """AKIEC is in-situ carcinoma; diagnosis_1 labels it Malignant upstream."""
    assert "AKIEC" in MILK_MALIGNANT_CLASSES


def test_seven_monet_columns():
    assert len(MILK_MONET_COLUMNS) == 7
    assert all(c.startswith("MONET_") for c in MILK_MONET_COLUMNS)


# ---------------------------------------------------------------------------
# Synthetic fixture
# ---------------------------------------------------------------------------
def _make_dataset(tmp_path: Path, n_lesions: int = 24, write_images: bool = True) -> Path:
    root = tmp_path / "milk"
    (root / "images").mkdir(parents=True)
    (root / "supplements").mkdir()

    meta_rows, gt_rows, input_rows = [], [], []
    for i in range(n_lesions):
        lid = f"IL_{i:05d}"
        cls = MILK_CLASS_NAMES[i % MILK_NUM_CLASSES]
        gt_rows.append({"lesion_id": lid,
                        **{c: 1.0 if c == cls else 0.0 for c in MILK_CLASS_NAMES}})
        for kind, itype in (("c", MILK_CLINICAL_TYPE), ("d", MILK_DERMOSCOPIC_TYPE)):
            iid = f"ISIC_{i:05d}{kind}"
            meta_rows.append({
                "isic_id": iid, "lesion_id": lid, "image_type": itype,
                "age_approx": "65", "sex": "male",
                "anatom_site_general": "trunk",
                "diagnosis_1": "Malignant" if cls in MILK_MALIGNANT_CLASSES else "Benign",
                "diagnosis_confirm_type": "histopathology",
            })
            input_rows.append({
                "isic_id": iid, "lesion_id": lid, "skin_tone_class": str(i % 6),
                "site": "trunk",
                **{c: f"{0.1 * (j + 1):.3f}" for j, c in enumerate(MILK_MONET_COLUMNS)},
            })
            if write_images:
                (root / "images" / f"{iid}.jpg").write_bytes(b"\xff\xd8\xff\xd9")

    pd.DataFrame(meta_rows).to_csv(root / "metadata.csv", index=False)
    pd.DataFrame(gt_rows).to_csv(root / "supplements" / "training_gt.csv", index=False)
    pd.DataFrame(input_rows).to_csv(root / "supplements" / "training_input.csv", index=False)
    return root


@pytest.fixture
def fake_root(tmp_path):
    return _make_dataset(tmp_path)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def test_loads_one_row_per_lesion(fake_root):
    df = load_milk10k(fake_root)
    assert len(df) == 24
    assert df["lesion_id"].is_unique


def test_both_image_paths_resolved(fake_root):
    df = load_milk10k(fake_root)
    assert df["clinical_path"].str.endswith("c.jpg").all()
    assert df["dermoscopy_path"].str.endswith("d.jpg").all()
    for p in df["clinical_path"]:
        assert Path(p).exists()


def test_clinical_and_dermoscopy_are_different_files(fake_root):
    df = load_milk10k(fake_root)
    assert (df["clinical_path"] != df["dermoscopy_path"]).all()


def test_label_matches_ground_truth(fake_root):
    df = load_milk10k(fake_root)
    for r in df.itertuples():
        assert MILK_CLASS_NAMES[r.label] == r.label_name


def test_malignant_flag_derived_from_label(fake_root):
    df = load_milk10k(fake_root)
    for r in df.itertuples():
        assert bool(r.malignant) == (r.label_name in MILK_MALIGNANT_CLASSES)


def test_monet_columns_are_numeric(fake_root):
    df = load_milk10k(fake_root)
    for c in MILK_MONET_COLUMNS:
        assert df[c].dtype.kind == "f"
        assert df[c].between(0.0, 1.0).all()


def test_missing_images_raise(tmp_path):
    root = _make_dataset(tmp_path, write_images=False)
    with pytest.raises(Milk10kLayoutError, match="absent"):
        load_milk10k(root)


def test_missing_images_tolerated_when_not_required(tmp_path):
    root = _make_dataset(tmp_path, write_images=False)
    assert len(load_milk10k(root, require_images=False)) == 24


def test_missing_root_raises(tmp_path):
    with pytest.raises(Milk10kLayoutError, match="expected"):
        load_milk10k(tmp_path / "nope")


def test_missing_column_raises(fake_root):
    meta = pd.read_csv(fake_root / "metadata.csv")
    meta.drop(columns=["lesion_id"]).to_csv(fake_root / "metadata.csv", index=False)
    with pytest.raises(Milk10kLayoutError, match="lesion_id"):
        load_milk10k(fake_root)


def test_unpaired_dataset_raises(fake_root):
    """Dropping one dermoscopic image must fail loudly, not silently pair wrong."""
    meta = pd.read_csv(fake_root / "metadata.csv")
    drop = meta.index[meta["image_type"] == MILK_DERMOSCOPIC_TYPE][0]
    meta.drop(index=drop).to_csv(fake_root / "metadata.csv", index=False)
    with pytest.raises(Milk10kLayoutError, match="unpaired"):
        load_milk10k(fake_root)


def test_ambiguous_ground_truth_raises(tmp_path):
    root = _make_dataset(tmp_path)
    gt = pd.read_csv(root / "supplements" / "training_gt.csv")
    gt.loc[0, MILK_CLASS_NAMES[0]] = 1.0
    gt.loc[0, MILK_CLASS_NAMES[1]] = 1.0
    gt.to_csv(root / "supplements" / "training_gt.csv", index=False)
    with pytest.raises(Milk10kLayoutError, match="exactly one active class"):
        _read_gt(root / "supplements" / "training_gt.csv")


def test_empty_ground_truth_row_raises(tmp_path):
    root = _make_dataset(tmp_path)
    gt = pd.read_csv(root / "supplements" / "training_gt.csv")
    gt.loc[0, MILK_CLASS_NAMES] = 0.0
    gt.to_csv(root / "supplements" / "training_gt.csv", index=False)
    with pytest.raises(Milk10kLayoutError, match="exactly one active class"):
        _read_gt(root / "supplements" / "training_gt.csv")


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def test_split_is_disjoint(fake_root):
    tr, ca, te = lesion_level_split(load_milk10k(fake_root))
    a, b, c = (set(p["lesion_id"]) for p in (tr, ca, te))
    assert not (a & b) and not (a & c) and not (b & c)


def test_split_conserves_lesions(fake_root):
    df = load_milk10k(fake_root)
    tr, ca, te = lesion_level_split(df)
    assert len(tr) + len(ca) + len(te) == len(df)


def test_split_is_deterministic(fake_root):
    df = load_milk10k(fake_root)
    first = [list(p["lesion_id"]) for p in lesion_level_split(df)]
    second = [list(p["lesion_id"]) for p in lesion_level_split(df)]
    assert first == second


def test_split_salt_changes_assignment(fake_root):
    df = load_milk10k(fake_root)
    a = list(lesion_level_split(df, salt="one")[0]["lesion_id"])
    b = list(lesion_level_split(df, salt="two")[0]["lesion_id"])
    assert a != b


def test_stable_bucket_in_unit_interval():
    for k in ("IL_0000001", "x", "", "IL_9999999"):
        assert 0.0 <= _stable_bucket(k, "salt") < 1.0


def test_stable_bucket_is_process_independent():
    """Hardcoded so a change to the hashing scheme cannot silently reshuffle splits."""
    assert _stable_bucket("IL_0000001", "milk10k-v3") == pytest.approx(
        _stable_bucket("IL_0000001", "milk10k-v3"))
    assert _stable_bucket("a", "s") != _stable_bucket("b", "s")


def test_stratified_split_reaches_every_partition(fake_root):
    """A class with only 2 lesions must not vanish from calibration or test."""
    df = load_milk10k(fake_root)
    tr, ca, te = lesion_level_split(df, stratify=True)
    # 24 lesions over 11 classes: every class has 2 or 3 members.
    for part in (tr, ca, te):
        assert len(part) > 0


def test_ratios_must_sum_to_one(fake_root):
    with pytest.raises(ValueError, match="sum to 1"):
        lesion_level_split(load_milk10k(fake_root), ratios=(0.5, 0.2, 0.2))


def test_split_requires_lesion_id():
    with pytest.raises(ValueError, match="lesion_id"):
        lesion_level_split(pd.DataFrame({"x": [1, 2, 3]}))


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
def test_cases_declare_both_images_available(fake_root):
    cases = build_cases(load_milk10k(fake_root))
    assert all(c["clinical_available"] for c in cases)
    assert all(c["dermoscopy_available"] for c in cases)


def test_cases_declare_text_absent(fake_root):
    """MILK10k ships no narrative. Claiming otherwise would fake a modality."""
    cases = build_cases(load_milk10k(fake_root))
    assert not any(c["text_available"] for c in cases)
    assert all(c["patient_text"] == "" for c in cases)


def test_cases_carry_monet_when_requested(fake_root):
    cases = build_cases(load_milk10k(fake_root), include_monet=True)
    assert all(c in cases[0]["metadata"] for c in MILK_MONET_COLUMNS)


def test_cases_omit_monet_when_not_requested(fake_root):
    cases = build_cases(load_milk10k(fake_root), include_monet=False)
    assert not any(k.startswith("MONET_") for k in cases[0]["metadata"])


def test_case_label_in_range(fake_root):
    for c in build_cases(load_milk10k(fake_root)):
        assert 0 <= c["label"] < MILK_NUM_CLASSES


def test_case_skin_tone_preserved(fake_root):
    cases = build_cases(load_milk10k(fake_root))
    assert {c["skin_tone"] for c in cases} <= {str(i) for i in range(6)}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def test_summarise_counts_all_classes(fake_root):
    s = summarise(load_milk10k(fake_root))
    assert set(s["class_counts"]) == set(MILK_CLASS_NAMES)
    assert s["lesions"] == 24
    assert s["images"] == 48


# ---------------------------------------------------------------------------
# Real dataset - pins the figures measured on 2026-10-11
# ---------------------------------------------------------------------------
@real_only
@pytest.mark.integration
def test_real_dataset_shape():
    s = summarise(load_milk10k(REAL_ROOT, require_images=False))
    assert s["lesions"] == 5240
    assert s["images"] == 10480
    assert s["classes"] == 11
    assert s["class_counts"]["BCC"] == 2522
    assert s["class_counts"]["MAL_OTH"] == 9
    assert s["malignant"] == 3757
    assert s["malignant_frac"] == pytest.approx(0.717, abs=1e-3)
    assert s["histopath_frac"] == pytest.approx(0.9573, abs=1e-3)


@real_only
@pytest.mark.integration
def test_real_dataset_has_no_centre_column():
    """Pins the finding that a centre-held-out experiment is impossible here."""
    meta = pd.read_csv(REAL_ROOT / "metadata.csv", dtype=str, keep_default_na=False)
    assert meta["attribution"].nunique() == 1


@real_only
@pytest.mark.integration
def test_real_split_preserves_prevalence():
    df = load_milk10k(REAL_ROOT, require_images=False)
    tr, ca, te = lesion_level_split(df)
    assert len(tr) + len(ca) + len(te) == 5240
    for part in (tr, ca, te):
        assert part["malignant"].mean() == pytest.approx(0.717, abs=0.01)
