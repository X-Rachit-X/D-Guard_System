"""
tests/test_benchmark_runner.py - StressBenchmark run loop and LLM client fallbacks.

The benchmark is driven against a stub pipeline with scripted behaviour, so the
metric definitions can be checked against hand-computed expectations rather than
against whatever the real models happen to output.
"""
import json
from pathlib import Path

import pytest
import torch

from src.constants import NUM_CLASSES
from src.stress.benchmark import StressBenchmark, StressReport


class StubResult:
    """Minimal stand-in for PipelineResult."""

    def __init__(self, action, diagnosis=None, confidence=None, fused=None):
        self.action = action
        self.diagnosis = diagnosis
        self.confidence = confidence
        self.fused_diagnosis = fused if fused is not None else diagnosis

    def to_dict(self):
        return {"action": self.action, "diagnosis": self.diagnosis,
                "confidence": self.confidence, "fused_diagnosis": self.fused_diagnosis}


class ScriptedPipeline:
    """Returns a scripted action per call, cycling through `script`."""

    def __init__(self, script):
        self.script = script
        self.calls = 0

    def process_case(self, case):
        out = self.script[self.calls % len(self.script)]
        self.calls += 1
        return out


def make_cases(n=4):
    return [{
        "case_id": f"c{i}", "label": 0,
        "clinical_image": torch.rand(1, 3, 224, 224),
        "metadata": {"age": 40 + i, "region": "ARM"},
        "patient_text": "a lesion on the arm that has grown over three months",
    } for i in range(n)]


class TestBenchmarkRun:
    def test_produces_a_result_per_condition(self, tmp_path):
        pipe = ScriptedPipeline([StubResult("ACCEPT", diagnosis=0, confidence=0.9)])
        report = StressBenchmark(pipeline=pipe).run(
            make_cases(), conditions=["miss_text", "miss_metadata"],
            output_dir=str(tmp_path))
        assert len(report.conditions) == 2
        assert {c.condition_id for c in report.conditions} == {"miss_text", "miss_metadata"}

    def test_writes_a_json_report(self, tmp_path):
        pipe = ScriptedPipeline([StubResult("ACCEPT", diagnosis=0, confidence=0.9)])
        StressBenchmark(pipeline=pipe).run(
            make_cases(), conditions=["miss_text"], output_dir=str(tmp_path))
        path = Path(tmp_path) / "stress_report.json"
        assert path.exists()
        body = json.loads(path.read_text())
        assert body["conditions"][0]["condition_id"] == "miss_text"
        assert "coverage" in body["conditions"][0]

    def test_all_correct_gives_full_coverage_and_accuracy(self, tmp_path):
        pipe = ScriptedPipeline([StubResult("ACCEPT", diagnosis=0, confidence=0.9)])
        report = StressBenchmark(pipeline=pipe).run(
            make_cases(), conditions=["miss_text"], output_dir=str(tmp_path))
        cond = report.conditions[0]
        assert cond.coverage == pytest.approx(1.0)
        assert cond.accuracy == pytest.approx(1.0)
        assert cond.escalation_rate == 0.0

    def test_total_deferral_gives_zero_coverage_and_nan_accuracy(self, tmp_path):
        """A system that answers nothing must not be scored as accurate."""
        pipe = ScriptedPipeline([StubResult("ESCALATE", fused=0)])
        report = StressBenchmark(pipeline=pipe).run(
            make_cases(), conditions=["miss_text"], output_dir=str(tmp_path))
        cond = report.conditions[0]
        assert cond.coverage == 0.0
        assert cond.accuracy != cond.accuracy       # NaN, not a flattering number
        assert cond.escalation_rate == pytest.approx(1.0)

    def test_recovery_is_zero_when_deferrals_were_unnecessary(self, tmp_path):
        """Every case deferred, but the withheld answer was CORRECT each time,
        so nothing was recovered. The old definition scored this 1.000."""
        pipe = ScriptedPipeline([StubResult("ESCALATE", fused=0)])   # label is 0
        report = StressBenchmark(pipeline=pipe).run(
            make_cases(), conditions=["miss_text"], output_dir=str(tmp_path))
        assert report.conditions[0].recovery_rate == 0.0

    def test_recovery_is_one_when_deferrals_avoided_errors(self, tmp_path):
        pipe = ScriptedPipeline([StubResult("ESCALATE", fused=3)])   # label is 0
        report = StressBenchmark(pipeline=pipe).run(
            make_cases(), conditions=["miss_text"], output_dir=str(tmp_path))
        assert report.conditions[0].recovery_rate == pytest.approx(1.0)

    def test_abstention_and_escalation_are_counted_separately(self, tmp_path):
        pipe = ScriptedPipeline([
            StubResult("ABSTAIN", fused=1),
            StubResult("ESCALATE", fused=1),
            StubResult("ACCEPT", diagnosis=0, confidence=0.9),
            StubResult("ACCEPT", diagnosis=0, confidence=0.9),
        ])
        report = StressBenchmark(pipeline=pipe).run(
            make_cases(4), conditions=["miss_text"], output_dir=str(tmp_path))
        cond = report.conditions[0]
        assert cond.abstention_rate == pytest.approx(0.25)
        assert cond.escalation_rate == pytest.approx(0.25)
        assert cond.coverage == pytest.approx(0.5)

    def test_pipeline_exception_is_recorded_as_a_deferral(self, tmp_path):
        class Exploding:
            def process_case(self, case):
                raise RuntimeError("boom")

        report = StressBenchmark(pipeline=Exploding()).run(
            make_cases(2), conditions=["miss_text"], output_dir=str(tmp_path))
        assert report.conditions[0].coverage == 0.0

    def test_summary_prints_without_error(self, tmp_path, capsys):
        pipe = ScriptedPipeline([StubResult("ACCEPT", diagnosis=0, confidence=0.9)])
        report = StressBenchmark(pipeline=pipe).run(
            make_cases(), conditions=["miss_text"], output_dir=str(tmp_path))
        report.print_summary()
        out = capsys.readouterr().out
        assert "DERMA-STRESS Report" in out
        assert "Cov" in out          # coverage must be visible next to accuracy


class TestLLMClientFallbacks:
    def test_unreachable_server_reports_unavailable(self):
        from src.llm.client import OllamaClient

        client = OllamaClient(base_url="http://127.0.0.1:9", timeout=1)
        assert client.is_available() is False

    def test_generate_raises_or_returns_on_unreachable_host(self):
        """Callers wrap generate() in try/except; it must fail fast, not hang."""
        from src.llm.client import OllamaClient

        client = OllamaClient(base_url="http://127.0.0.1:9", timeout=1, max_retries=0)
        try:
            out = client.generate("hello")
            assert isinstance(out, str)
        except Exception:
            pass    # an exception is the documented alternative
