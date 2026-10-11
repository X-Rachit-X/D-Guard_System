"""
src/agents/quality.py - Image-quality agent.

Checks a photo BEFORE it is trusted: sharpness, exposure, contrast, glare and
resolution. Carried over from the old system's evidence manager, with one change:
the thresholds are not hand-picked. They are the 5th / 95th percentiles of the
same measurements on real MILK10k training photos, so "too blurry" means
"blurrier than 95% of the photos the model learned from".

The agent never changes a probability. It tells the patient what to fix, and
its findings appear in the report; the evidential model's ignorance u is what
actually down-weights a poor photo.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

_LAPLACE = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)


def _gray(img: Image.Image, size: int = 384) -> np.ndarray:
    g = img.convert("L")
    g.thumbnail((size, size))
    return np.asarray(g, dtype=np.float32) / 255.0


def measure(img: Image.Image) -> Dict[str, float]:
    """Raw measurements for one photo."""
    g = _gray(img)
    # variance of the Laplacian: the standard focus measure
    pad = np.pad(g, 1, mode="edge")
    lap = (_LAPLACE[0, 1] * pad[:-2, 1:-1] + _LAPLACE[1, 0] * pad[1:-1, :-2]
           + _LAPLACE[1, 1] * pad[1:-1, 1:-1] + _LAPLACE[1, 2] * pad[1:-1, 2:]
           + _LAPLACE[2, 1] * pad[2:, 1:-1])
    return {
        "sharpness": float(lap.var() * 1e3),
        "brightness": float(g.mean()),
        "contrast": float(g.std()),
        "glare": float((g > 0.97).mean()),
        "min_side": float(min(img.size)),
    }


@dataclass
class QualityReport:
    ok: bool
    score: float                         # 0..1, share of checks passed
    issues: List[str] = field(default_factory=list)
    advice: List[str] = field(default_factory=list)
    #: Issues beyond the 1st/99th percentile of the reference photos. Only these
    #: trigger the quality gate (a HIGH answer is downgraded to LOW); plain
    #: `issues` (5th/95th) only produce advice. At 5/95 the agent flags about a
    #: quarter of perfectly ordinary photos - right for tips, wrong for a gate.
    severe: List[str] = field(default_factory=list)
    measurements: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


ADVICE = {
    "blurry": "Hold the camera steady and tap the spot on screen to focus.",
    "too dark": "Take the photo in daylight or with the room lights on.",
    "too bright": "Avoid direct flash or sunlight on the skin.",
    "low contrast": "Make sure the spot fills more of the frame and is in focus.",
    "glare": "Tilt the camera slightly to avoid reflections on the skin.",
    "low resolution": "Move closer so the spot fills the middle of the photo.",
}


class ImageQualityAgent:
    """Thresholds fitted on reference photos (percentiles), then reused."""

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.t = thresholds or {
            "sharpness_low": 0.05, "brightness_low": 0.15, "brightness_high": 0.92,
            "contrast_low": 0.04, "glare_high": 0.08, "min_side": 224.0,
        }

    @classmethod
    def fit(cls, image_paths: Sequence[str], cache: Optional[Path] = None) -> "ImageQualityAgent":
        """Fit thresholds from reference photos; cached to `cache` if given."""
        if cache is not None and Path(cache).exists():
            return cls(json.loads(Path(cache).read_text()))
        ms = [measure(Image.open(p)) for p in image_paths]
        arr = {k: np.array([m[k] for m in ms]) for k in ms[0]}
        t = {
            "sharpness_low": float(np.percentile(arr["sharpness"], 5)),
            "brightness_low": float(np.percentile(arr["brightness"], 5)),
            "brightness_high": float(np.percentile(arr["brightness"], 95)),
            "contrast_low": float(np.percentile(arr["contrast"], 5)),
            "glare_high": float(max(np.percentile(arr["glare"], 95), 0.02)),
            "min_side": 224.0,
            # severe = outside the 1st/99th percentile
            "sharpness_severe": float(np.percentile(arr["sharpness"], 1)),
            "brightness_low_severe": float(np.percentile(arr["brightness"], 1)),
            "brightness_high_severe": float(np.percentile(arr["brightness"], 99)),
            "contrast_severe": float(np.percentile(arr["contrast"], 1)),
            "glare_severe": float(max(np.percentile(arr["glare"], 99), 0.05)),
            "n_reference": len(ms),
        }
        if cache is not None:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            Path(cache).write_text(json.dumps(t, indent=2))
        return cls(t)

    def check(self, img: Optional[Image.Image]) -> Optional[QualityReport]:
        if img is None:
            return None
        m = measure(img)
        t = self.t
        issues = []
        if m["sharpness"] < t["sharpness_low"]:
            issues.append("blurry")
        if m["brightness"] < t["brightness_low"]:
            issues.append("too dark")
        if m["brightness"] > t["brightness_high"]:
            issues.append("too bright")
        if m["contrast"] < t["contrast_low"]:
            issues.append("low contrast")
        if m["glare"] > t["glare_high"]:
            issues.append("glare")
        if m["min_side"] < t["min_side"]:
            issues.append("low resolution")
        severe = []
        if m["sharpness"] < t.get("sharpness_severe", t["sharpness_low"] / 2):
            severe.append("blurry")
        if m["brightness"] < t.get("brightness_low_severe", t["brightness_low"] * 0.7):
            severe.append("too dark")
        if m["brightness"] > t.get("brightness_high_severe", 0.97):
            severe.append("too bright")
        if m["contrast"] < t.get("contrast_severe", t["contrast_low"] * 0.7):
            severe.append("low contrast")
        if m["glare"] > t.get("glare_severe", 0.15):
            severe.append("glare")
        if m["min_side"] < t["min_side"]:
            severe.append("low resolution")
        n_checks = 6
        return QualityReport(ok=not issues, score=round(1 - len(issues) / n_checks, 3),
                             issues=issues, advice=[ADVICE[i] for i in issues],
                             measurements={k: round(v, 4) for k, v in m.items()},
                             severe=severe)
