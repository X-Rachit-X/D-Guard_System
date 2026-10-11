"""
src/native/engine.py - Assess ONE patient with the trained MILK10k model.

Same model, same calibration, same policy as scripts/run_eade_milk.py - so what
the app shows a patient is exactly what the experiment measured.

Flow for one case:

    clinical photo + details  ──►  Stage A  ──►  PROCEED | ACQUIRE | DEFER
                                                   │
                         (patient uploads the      ▼
                          requested dermoscopy) Stage B ──► PROCEED | DEFER

What it returns, besides the action:

  * the tentative opinion (best guess, differential, P(malignant), urgency) -
    present even on DEFER, labelled unconfirmed
  * per-modality ignorance u_m and conflict C
  * DYNAMIC FUSION WEIGHTS: the share of evidence each modality contributed to
    THIS case, w_m = (1 - u_m) / sum over present modalities. A blurred photo
    produces little evidence, so its weight falls on its own - per case, with no
    hand-set rule. This is the evidential counterpart of the old system's
    per-case weights.
  * the measured diagnosis-conditioned reliability of each modality for the
    tentative diagnosis (the old "diagnosis-adaptive prior", now measured)

Artifacts it needs (both produced by scripts you run once):
    runs/milk10k_resnet50_seed42/checkpoints/best.pt   (Kaggle)
    outputs/eade_milk/cost_5/calibration.joblib    (scripts/run_eade_milk.py)
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from src.constants import MILK_CLASS_FULL_NAMES, MILK_MONET_COLUMNS
from src.decision.explain import explain_doctor, explain_patient
from src.decision.policy import Action, Decision, ThreeActionPolicy

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "runs" / "milk10k_resnet50_seed42"
DEFAULT_BUNDLE = ROOT / "outputs" / "eade_milk" / "cost_5" / "calibration.joblib"
DEFAULT_DATA = ROOT / "milk10k"
MODALITY_ORDER = ("image", "dermoscopy", "meta", "text")
DISPLAY = {"image": "clinical photo", "dermoscopy": "dermoscopy", "meta": "patient details"}


def _load_trainer():
    path = ROOT / "notebooks" / "kaggle_train_dguard.py"
    spec = importlib.util.spec_from_file_location("kaggle_train_dguard", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass
class NativeResult:
    stage: str                                   # "A" (no dermoscopy) | "B"
    decision: Decision
    probs: List[float]
    ignorance: float
    conflict: float
    modality_ignorance: Dict[str, float]
    fusion_weights: Dict[str, float]
    modality_votes: Dict[str, str]
    reliability_for_tentative: Dict[str, Optional[float]]
    patient_message: str
    doctor_message: str
    classes: List[str] = field(default_factory=list)

    @property
    def action(self) -> str:
        return self.decision.action.value

    def to_dict(self) -> Dict:
        return {
            "stage": self.stage, "action": self.action,
            "decision": self.decision.to_dict(),
            "probs": {c: round(float(p), 4) for c, p in zip(self.classes, self.probs)},
            "ignorance": self.ignorance, "conflict": self.conflict,
            "modality_ignorance": self.modality_ignorance,
            "fusion_weights": self.fusion_weights,
            "modality_votes": self.modality_votes,
            "reliability_for_tentative": self.reliability_for_tentative,
            "patient_message": self.patient_message,
            "doctor_message": self.doctor_message,
        }


class NativeEngine:
    """Loads once (~10 s on CPU), then assesses a case in well under a second."""

    def __init__(self, run_dir: Path | str = DEFAULT_RUN,
                 bundle_path: Path | str = DEFAULT_BUNDLE,
                 data_dir: Path | str = DEFAULT_DATA,
                 device: str = "cpu"):
        import joblib

        self.run_dir = Path(run_dir)
        self.data_dir = Path(data_dir)
        ckpt_path = self.run_dir / "checkpoints" / "best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"trained checkpoint not found: {ckpt_path}")
        if not Path(bundle_path).exists():
            raise FileNotFoundError(
                f"calibration bundle not found: {bundle_path}\n"
                "Run: python scripts/milk_native_infer.py && "
                "python scripts/run_eade_milk.py --missed-cost 5 --out outputs/eade_milk/cost_5")

        T = _load_trainer()
        self.T = T
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = dict(T.CFG)
        cfg.update(ck["cfg"])
        cfg["DATA_DIR"] = str(self.data_dir)
        self.cfg = cfg

        df, img_dir, classes = T.milk_load(str(self.data_dir), cfg)
        T.CLASS_NAMES, T.K = classes, len(classes)
        T.CLASS_TO_IDX = {c: i for i, c in enumerate(classes)}
        self.classes = list(classes)
        self.K = len(classes)
        self.img_dir = Path(img_dir)
        df["clinical_text"] = ""
        self.splits = T.lesion_level_split(df, cfg["SPLIT"], cfg["SEED"])
        train = self.splits["train"]
        self.meta_encoder = T.MilkMetaEncoder().fit(train)
        self.monet_median = {c: float(train[c].median()) for c in MILK_MONET_COLUMNS}

        self.model = T.DGuard(ck["meta_dim"], cfg, text_model=None)
        self.model.load_state_dict(ck["model"], strict=True)
        self.device = torch.device(device)
        self.model.to(self.device).eval()

        import torchvision.transforms as TV
        size = cfg["IMAGE_SIZE"]
        self.tf = TV.Compose([TV.Resize((size, size)), TV.ToTensor(),
                              TV.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.bundle = joblib.load(bundle_path)
        if list(self.bundle["classes"]) != self.classes:
            raise ValueError("calibration bundle was built for a different class list")
        self.malignant = list(self.bundle["malignant"])
        self.malignant_names = [self.classes[i] for i in self.malignant]
        self.reliability = self.bundle["modality_reliability_by_class"]

    # ------------------------------------------------------------------ data
    def sample_lesions(self, split: str = "test", n: int = 12, seed: int = 0) -> List[Dict]:
        """Real held-out lesions for the demo - never training lesions."""
        if split == "train":
            raise ValueError("demo samples must come from val or test, never train")
        d = self.splits[split]
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(d), size=min(n, len(d)), replace=False)
        out = []
        for i in idx:
            r = d.iloc[int(i)]
            out.append({
                "lesion_id": str(r["lesion_id"]),
                "true_label": self.classes[int(r["label_idx"])],
                "clinical_path": str(self.img_dir / r["clinical_file"]),
                "dermoscopy_path": str(self.img_dir / r["dermoscopy_file"]),
                "metadata": {
                    "age": None if np.isnan(r["age"]) else float(r["age"]),
                    "sex": r["sex"] or None,
                    "site": r["site"] or None,
                    "anatom_site_general": r["anatom_site_general"] or None,
                    "skin_tone_class": str(r["skin_tone_class"]) or None,
                    **{c: float(r[c]) for c in MILK_MONET_COLUMNS},
                },
            })
        return out

    def _meta_row(self, metadata: Optional[Dict]):
        import pandas as pd
        m = dict(metadata or {})
        row = {
            "age": m.get("age"),
            "sex": m.get("sex") or "",
            "site": m.get("site") or "",
            "anatom_site_general": m.get("anatom_site_general") or "",
            "skin_tone_class": "" if m.get("skin_tone_class") is None else str(m["skin_tone_class"]),
        }
        # MONET scores are produced from the image by an external concept model.
        # For an uploaded photo we do not have them, so the TRAINING median is
        # used - a neutral value - rather than 0.0, which is off-distribution.
        for c in MILK_MONET_COLUMNS:
            v = m.get(c)
            row[c] = self.monet_median[c] if v is None else float(v)
        return pd.DataFrame([row])

    # ---------------------------------------------------------------- assess
    @torch.no_grad()
    def assess(self, clinical: Optional[Image.Image], dermoscopy: Optional[Image.Image],
               metadata: Optional[Dict]) -> NativeResult:
        T = self.T
        size = self.cfg["IMAGE_SIZE"]
        has_clin = clinical is not None
        has_derm = dermoscopy is not None
        has_meta = bool(metadata) and any(v not in (None, "") for v in metadata.values())

        def tens(img):
            if img is None:
                return torch.zeros(1, 3, size, size)
            return self.tf(img.convert("RGB")).unsqueeze(0)

        meta_vec = self.meta_encoder.transform(self._meta_row(metadata))
        batch = {
            "image": tens(clinical).to(self.device),
            "dermoscopy": tens(dermoscopy).to(self.device),
            "meta": torch.from_numpy(meta_vec).to(self.device),
            "input_ids": torch.zeros(1, self.cfg["TEXT_MAX_LEN"], dtype=torch.long),
            "attention_mask": torch.zeros(1, self.cfg["TEXT_MAX_LEN"], dtype=torch.long),
        }
        z_img, z_derm, z_meta, z_text = self.model.encode(batch)
        ops = {"image": self.model.head_img(z_img),
               "dermoscopy": self.model.head_dermo(z_derm),
               "meta": self.model.head_meta(z_meta),
               "text": self.model.head_text(z_text)}
        pres = {"image": float(has_clin), "dermoscopy": float(has_derm),
                "meta": float(has_meta), "text": 0.0}

        fused, conflicts = None, []
        for name in MODALITY_ORDER:
            op = T.gate(ops[name], torch.tensor([pres[name]]), self.K)
            if fused is None:
                fused = op
            else:
                fused = T.ds_combine(fused, op)
                conflicts.append(fused["conflict"])
        conflict = float(torch.stack(conflicts).amax(0).item())
        u = float(fused["u"].item())
        prob = (fused["belief"] + fused["u"] / self.K)[0].cpu().numpy()

        mod_u = {m: (float(ops[m]["u"].item()) if pres[m] else 1.0)
                 for m in ("image", "dermoscopy", "meta")}
        votes = {DISPLAY[m]: self.classes[int(ops[m]["belief"].argmax().item())]
                 for m in ("image", "dermoscopy", "meta") if pres[m]}
        ev = {m: 1.0 - mod_u[m] for m in mod_u if pres[m]}
        tot = sum(ev.values()) or 1.0
        weights = {DISPLAY[m]: round(v / tot, 3) for m, v in ev.items()}

        stage = "B" if has_derm else "A"
        cal = self.bundle["stage_B" if has_derm else "stage_A"]
        policy = ThreeActionPolicy(
            cost=self.bundle["cost"], malignant_classes=self.malignant,
            num_classes=self.K, conflict_threshold=cal["conflict_threshold"],
            lambda_hat=cal["lambda_hat"], label_rule=self.bundle["label_rule"],
            ambiguity_gate=self.bundle["ambiguity_gate"])
        pset = cal["sets"].predict_sets(prob[None, :])[0]

        if has_derm:
            decision = policy.decide(prob, ignorance=u, conflict=conflict,
                                     prediction_set=pset, acquirable=[],
                                     n_acquisitions_so_far=1)
        else:
            feats = self._acq_features(prob, u, conflict, mod_u)
            benefit = float(self.bundle["acquisition_value_model"].predict(feats)[0])
            decision = policy.decide(prob, ignorance=u, conflict=conflict,
                                     prediction_set=pset,
                                     acquirable=["dermoscopy_image"] if has_clin else [],
                                     predicted_benefit={"dermoscopy_image": benefit})

        tent = decision.tentative
        rel = {}
        if tent is not None:
            for m, key in (("clinical photo", "image"), ("dermoscopy", "dermoscopy"),
                           ("patient details", "meta")):
                rel[m] = self.reliability[key][tent]

        return NativeResult(
            stage=stage, decision=decision, probs=[float(x) for x in prob],
            ignorance=u, conflict=conflict, modality_ignorance=
            {DISPLAY[m]: round(v, 3) for m, v in mod_u.items()},
            fusion_weights=weights, modality_votes=votes,
            reliability_for_tentative=rel,
            patient_message=explain_patient(decision, self.classes, MILK_CLASS_FULL_NAMES),
            doctor_message=explain_doctor(decision, self.classes, MILK_CLASS_FULL_NAMES),
            classes=self.classes,
        )

    def _acq_features(self, prob, u, conflict, mod_u):
        """Must match `features()` in scripts/run_eade_milk.py column for column."""
        K = self.K
        ent = float(-(prob * np.log(np.clip(prob, 1e-12, 1))).sum() / np.log(K))
        return np.array([[u, conflict, float(prob.max()), float(prob[self.malignant].sum()),
                          ent, mod_u["image"], mod_u["meta"]]])

    def assess_paths(self, clinical_path: Optional[str], dermoscopy_path: Optional[str],
                     metadata: Optional[Dict]) -> NativeResult:
        load = lambda p: Image.open(p).convert("RGB") if p else None  # noqa: E731
        return self.assess(load(clinical_path), load(dermoscopy_path), metadata)
