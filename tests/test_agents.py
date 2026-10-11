"""tests/test_agents.py - the final agents: quality check, retrieval, report verifier."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageFilter

from src.agents.librarian import SPEC, MilkLibrarian
from src.agents.quality import ImageQualityAgent, measure
from src.agents.report import ReportAgent
from src.decision.policy import ThreeActionPolicy
from src.retrieval.rag_index import RAGIndex

CLASSES = ["BCC", "ACK", "NEV", "SEK", "SCC", "MEL"]


def textured(seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray((rng.random((256, 256, 3)) * 255).astype(np.uint8))


# ----------------------------------------------------------------- quality
def test_blur_lowers_sharpness():
    img = textured()
    assert measure(img.filter(ImageFilter.GaussianBlur(6)))["sharpness"] < measure(img)["sharpness"]


def test_quality_flags_dark_and_small():
    agent = ImageQualityAgent()
    rep = agent.check(Image.new("RGB", (100, 100), (5, 5, 5)))
    assert not rep.ok
    assert "too dark" in rep.issues and "low resolution" in rep.issues
    assert len(rep.advice) == len(rep.issues)


def test_quality_fit_uses_reference_percentiles(tmp_path):
    paths = []
    for i in range(12):
        p = tmp_path / f"r{i}.png"
        textured(i).save(p)
        paths.append(str(p))
    agent = ImageQualityAgent.fit(paths, cache=tmp_path / "t.json")
    assert (tmp_path / "t.json").exists()
    assert agent.check(textured(99)).ok


# --------------------------------------------------------------- librarian
def _index(tmp_path, split="train"):
    rng = np.random.default_rng(0)
    cases = [{"lesion_id": f"L{i}", "diagnosis": CLASSES[i % 3], "label": i % 3,
              "clinical_embedding": rng.normal(size=256).astype(np.float32),
              "metadata_embedding": rng.normal(size=256).astype(np.float32)} for i in range(30)]
    idx = RAGIndex(embedding_spec=SPEC)
    idx.source_split = split
    idx.add_training_cases(cases)
    idx.save(str(tmp_path / "idx"))
    return cases


def test_librarian_finds_identical_case_and_reports_agreement(tmp_path):
    cases = _index(tmp_path)
    lib = MilkLibrarian.load(tmp_path / "idx", k=4)
    out = lib.similar(cases[0]["clinical_embedding"], cases[0]["metadata_embedding"],
                      tentative=cases[0]["diagnosis"])
    assert out["similar_cases"][0]["lesion_id"] == "L0"
    assert 0.0 < out["agreement"] <= 1.0


def test_librarian_refuses_non_training_index(tmp_path):
    _index(tmp_path, split="test")
    with pytest.raises(ValueError, match="training split"):
        MilkLibrarian.load(tmp_path / "idx")


# ------------------------------------------------------------------ report
class _Res:
    def __init__(self, decision):
        self.decision = decision
        self.ignorance, self.conflict = 0.4, 0.02
        self.fusion_weights = {"clinical photo": 0.7, "patient details": 0.3}
        self.modality_votes = {"clinical photo": "BCC", "patient details": "BCC"}


def _decision(low=False):
    pol = ThreeActionPolicy(malignant_classes=[0, 4, 5], num_classes=6, label_rule="argmax",
                            defer_mode="minimal", lambda_hat=0.9 if low else None)
    p = np.array([0.70, 0.10, 0.08, 0.06, 0.04, 0.02])
    return pol.decide(p, ignorance=0.4 if low else 0.05, conflict=0.0)


def test_structured_report_mentions_tier_and_confirmation():
    agent = ReportAgent(CLASSES, {"BCC": "Basal Cell Carcinoma"})
    text = agent.structured(_Res(_decision(low=True)), {}, None)
    assert "LOW confidence" in text and "confirmation" in text


def test_verifier_rejects_invented_diagnosis_and_number():
    agent = ReportAgent(CLASSES, {})
    res = _Res(_decision())
    probs = agent.verify("Likely MEL, 93% certain. Confirm with a doctor.", res, None)
    assert any("MEL" in p for p in probs) or any("93" in p for p in probs)


class _FakeLLM:
    def __init__(self, text):
        self.text = text

    def generate(self, prompt, system_prompt=None):
        return self.text


def test_unsupported_llm_rewrite_falls_back_to_structured():
    res = _Res(_decision(low=True))
    agent = ReportAgent(CLASSES, {}, llm=_FakeLLM("Definitely melanoma (MEL), 99%."))
    out = agent.write(res, {}, None)
    assert out["source"].startswith("structured") and out["problems"]
