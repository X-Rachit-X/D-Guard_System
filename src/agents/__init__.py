"""src/agents - the D-GUARD agent pipeline.

    quality.py    ImageQualityAgent  - checks each photo before it is trusted
    librarian.py  MilkLibrarian      - similar confirmed training cases (explanation only)
    report.py     ReportAgent        - structured report; optional LLM rewrite, verified
    pipeline.py   DGuardPipeline     - quality -> engine -> librarian -> report

Only the engine (src/native) decides; the agents explain, warn and write.
"""
from src.agents.librarian import MilkLibrarian
from src.agents.pipeline import DGuardPipeline
from src.agents.quality import ImageQualityAgent, QualityReport
from src.agents.report import ReportAgent

__all__ = ["DGuardPipeline", "ImageQualityAgent", "QualityReport", "MilkLibrarian",
           "ReportAgent"]
