"""
src/agents/pipeline.py - The final D-GUARD agent pipeline.

    photo(s) + details
          │
          ▼
    ImageQualityAgent ── checks each photo (thresholds fitted on MILK10k photos)
          │
          ▼
    NativeEngine ─────── evidential heads → Dempster–Shafer fusion → calibrated
          │              policy → PROCEED (HIGH / LOW) | ACQUIRE | DEFER
          ▼
    MilkLibrarian ────── similar confirmed TRAINING cases (explanation only)
          │
          ▼
    ReportAgent ──────── structured report; optional LLM rewrite, verified claim
                         by claim, rejected if it adds anything unsupported

Only the engine decides. The other agents explain, warn and write - none of them
can change a probability or an action. This is the old system's multi-agent
design kept where it helps (transparency, quality control, readable reports) and
removed where it hurt (agents voting on the diagnosis).

`DGuardPipeline.assess` has the same signature as `NativeEngine.assess` and
returns the same result type, so the care platform uses either interchangeably.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from PIL import Image

from src.agents.librarian import MilkLibrarian
from src.agents.quality import ImageQualityAgent
from src.agents.report import ReportAgent
from src.agents.scribe import ScribeAgent
from src.constants import MILK_CLASS_FULL_NAMES

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX = ROOT / "outputs" / "milk_rag" / "index_train"
QUALITY_CACHE = ROOT / "outputs" / "milk_rag" / "quality_thresholds.json"
DERMO_QUALITY_CACHE = ROOT / "outputs" / "milk_rag" / "quality_thresholds_dermoscopy.json"


class DGuardPipeline:
    def __init__(self, engine, librarian: Optional[MilkLibrarian] = None,
                 quality: Optional[ImageQualityAgent] = None,
                 reporter: Optional[ReportAgent] = None,
                 dermo_quality: Optional[ImageQualityAgent] = None):
        self.engine = engine
        self.librarian = librarian
        self.quality = quality or ImageQualityAgent()
        # Dermoscopy images are smoother and more uniformly lit than phone photos,
        # so they get thresholds fitted on dermoscopy images, not on photos.
        self.dermo_quality = dermo_quality or self.quality
        self.scribe = ScribeAgent()
        self.reporter = reporter or ReportAgent(engine.classes, MILK_CLASS_FULL_NAMES)

    @classmethod
    def build(cls, engine=None, use_llm: bool = True, index_path: Path = DEFAULT_INDEX):
        """Assemble everything that is available; missing pieces degrade, not fail."""
        if engine is None:
            from src.native import NativeEngine
            engine = NativeEngine()
        librarian = None
        if Path(str(index_path) + ".index").exists():
            librarian = MilkLibrarian.load(index_path)
        train = engine.splits["train"]
        refs = [str(engine.img_dir / f) for f in train["clinical_file"].iloc[:300]]
        quality = ImageQualityAgent.fit(refs, cache=QUALITY_CACHE)
        drefs = [str(engine.img_dir / f) for f in train["dermoscopy_file"].iloc[:300]]
        dermo_quality = ImageQualityAgent.fit(drefs, cache=DERMO_QUALITY_CACHE)
        llm = None
        if use_llm:
            try:
                from src.llm.client import OllamaClient
                c = OllamaClient(timeout=20)
                llm = c if c.is_available() else None
            except Exception:
                llm = None
        return cls(engine, librarian, quality,
                   ReportAgent(engine.classes, MILK_CLASS_FULL_NAMES, llm=llm),
                   dermo_quality=dermo_quality)

    # The platform and the app read these straight from the engine.
    def __getattr__(self, name):
        return getattr(self.engine, name)

    def assess(self, clinical: Optional[Image.Image], dermoscopy: Optional[Image.Image],
               metadata: Optional[Dict]):
        q = {"clinical photo": self.quality.check(clinical),
             "dermoscopy": self.dermo_quality.check(dermoscopy)}
        metadata = dict(metadata or {})
        symptoms = metadata.pop("symptoms_text", None)       # the model has no text input
        result = self.engine.assess(clinical, dermoscopy, metadata)
        result.quality = {k: (v.to_dict() if v else None) for k, v in q.items()}

        # QUALITY GATE. The calibrated guarantees assume this photo resembles the
        # calibration photos. One with a SEVERE quality issue is, by construction,
        # outside the 1-99% range of those photos, so it cannot keep a HIGH-
        # confidence label. Measured reason (scripts/run_stress_milk.py): on blurred
        # photos the model's ignorance FALLS and its HIGH-tier rate RISES, and on
        # dark photos HIGH answers missed 12.5% of cancers - while the quality agent
        # flagged 100% of both. The gate only ever adds caution.
        d = result.decision
        flagged = [k for k, v in q.items() if v is not None and v.severe]
        if flagged and d.action.value == "PROCEED" and d.tier != "LOW":
            from src.decision.explain import explain_doctor, explain_patient
            from src.decision.policy import DeferReason
            d.tier = "LOW"
            d.low_reason = DeferReason.QUALITY
            d.gates = {**(d.gates or {}), "photo_quality": True}
            result.patient_message = explain_patient(d, self.engine.classes,
                                                     MILK_CLASS_FULL_NAMES)
            result.doctor_message = explain_doctor(d, self.engine.classes,
                                                   MILK_CLASS_FULL_NAMES)

        # SCRIBE: warning signs in the patient's own words raise triage urgency;
        # they never change the diagnosis or the confidence tier.
        sr = self.scribe.read(symptoms, metadata.get("age"))
        if sr is not None:
            result.scribe = sr.to_dict()
            if sr.red_flags and result.decision.urgency != "URGENT":
                result.decision.urgency = "URGENT"
                from src.decision.explain import explain_doctor, explain_patient
                result.patient_message = explain_patient(result.decision, self.engine.classes,
                                                         MILK_CLASS_FULL_NAMES)
                result.doctor_message = explain_doctor(result.decision, self.engine.classes,
                                                       MILK_CLASS_FULL_NAMES)
            if sr.red_flags:
                result.patient_message += ("\nYou mentioned " + ", ".join(sr.red_flags)
                                           + " - please make sure a doctor sees this spot soon.")

        tent = result.decision.tentative
        tent_name = self.engine.classes[tent] if tent is not None else None
        if self.librarian is not None and result.embeddings.get("clinical") is not None:
            result.similar = self.librarian.similar(result.embeddings.get("clinical"),
                                                    result.embeddings.get("meta"), tent_name)
        result.report = self.reporter.write(result, result.quality, result.similar)

        advice = [a for v in q.values() if v for a in v.advice]
        if advice:
            result.patient_message += ("\nPhoto tip: " + " ".join(dict.fromkeys(advice)))
        return result

    def assess_paths(self, clinical_path: Optional[str], dermoscopy_path: Optional[str],
                     metadata: Optional[Dict]):
        """Like assess(), from file paths - goes through every agent and the quality gate."""
        load = lambda p: Image.open(p).convert("RGB") if p else None  # noqa: E731
        return self.assess(load(clinical_path), load(dermoscopy_path), metadata)
