"""
tests/test_v2_components.py - the pieces added for v2: occlusion check, missing-
details detection, learned evidence gate, fusion-method API, explainer, and the
platform's data-deletion controls.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PIL import Image

from src.agents.explainer import EvidenceExplainer
from src.agents.pipeline import has_details
from src.agents.quality import ImageQualityAgent, measure
from src.care import CareService, CareStore
from src.eval.corruptions import occlusion
from src.fusion.gating import LearnedGate, cross_fit, evidence_features
from src.fusion.methods import METHODS, fit_context, fit_method, fuse_method, oracle_best_head
from tests.test_care_platform import CONFIDENT_NEV, ScriptedEngine


def _texture(seed=0, size=256):
    rng = np.random.default_rng(seed)
    a = (rng.normal(150, 25, (size, size, 3))).clip(0, 255).astype(np.uint8)
    return Image.fromarray(a)


# ------------------------------------------------------------------ quality
def test_occlusion_is_detected_and_clean_texture_is_not():
    clean = _texture()
    assert measure(clean)["flat"] < 0.01
    assert measure(occlusion(clean))["flat"] > 0.8
    qa = ImageQualityAgent({"sharpness_low": 0, "brightness_low": 0, "brightness_high": 1,
                            "contrast_low": 0, "glare_high": 1, "min_side": 10})
    assert "covered" in qa.check(occlusion(clean)).severe
    assert "covered" not in qa.check(clean).severe


def test_has_details():
    assert not has_details({})
    assert not has_details({"age": None, "sex": "", "site": None})
    assert not has_details({"age": float("nan")})
    assert has_details({"sex": "female"})
    assert has_details({"age": 61.0})


# ------------------------------------------------------------------ fusion
def _toy(N=300, M=3, K=4, seed=0):
    """Specialist 0 is informative, 1 is noise, 2 is informative only for class 0."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, K, N)
    P = np.full((N, M, K), 1.0 / K)
    P[np.arange(N), 0, y] += 1.5
    P[:, 1] = rng.dirichlet(np.ones(K), N)
    P[y == 0, 2, 0] += 2.0
    P = P / P.sum(2, keepdims=True)
    Q = np.ones((N, M))
    pres = np.ones((N, M), bool)
    return P, Q, pres, y, K


def test_learned_gate_trusts_the_informative_specialist():
    P, Q, pres, y, K = _toy()
    R = np.ones_like(Q) * 0.5
    F = evidence_features(P, Q, R, pres, None, None)
    g = LearnedGate(names=["good", "noise", "partial"]).fit(P, F, pres, y)
    p, w = g.fuse(P, F, pres)
    assert w[:, 0].mean() > w[:, 1].mean()
    assert (p.argmax(1) == y).mean() > (P[:, 1].argmax(1) == y).mean()
    imp = g.importance()
    assert set(imp) and all(v >= 0 for v in imp.values())


def test_cross_fit_returns_out_of_fold_probabilities():
    P, Q, pres, y, K = _toy(N=200)
    F = evidence_features(P, Q, np.ones_like(Q) * 0.5, pres, None, None)
    oof = cross_fit(lambda: LearnedGate(names=list("abc")), P, F, pres, y, folds=4)
    assert oof.shape == (200, K) and np.allclose(oof.sum(1), 1, atol=1e-6)


def test_every_fusion_method_returns_distributions():
    P, Q, pres, y, K = _toy(N=240)
    ctx = fit_context(["a", "b", "c"], K, P, Q, pres, y)
    for m in METHODS:
        fit_method(ctx, m, P, Q, pres, y)
        p, w = fuse_method(ctx, m, P, Q, pres)
        assert p.shape == (240, K), m
        assert np.allclose(p.sum(1), 1, atol=1e-6), m
        assert np.allclose(w.sum(1), 1, atol=1e-6), m
    orc = oracle_best_head(P, pres, y)
    assert (orc.argmax(1) == y).mean() >= max((P[:, m].argmax(1) == y).mean() for m in range(3))


# ------------------------------------------------------------------ explainer
def test_explainer_names_dissent_and_verifier_rejects_invented_numbers():
    classes = ["BCC", "MEL", "NV"]
    team = {"votes": {"dermoscopy head": "MEL", "patient-details head": "NV"},
            "probs": {"dermoscopy head": [0.1, 0.8, 0.1], "patient-details head": [0.2, 0.3, 0.5]},
            "weights": {"dermoscopy head": 0.85, "patient-details head": 0.15},
            "reliability": {"dermoscopy head": 0.8, "patient-details head": 0.3},
            "quality": {"dermoscopy head": 1.0, "patient-details head": 1.0},
            "uncertainty": {"dermoscopy head": 0.4, "patient-details head": 0.95},
            "utility": {"dermoscopy head": 0.7, "patient-details head": 0.2},
            "consistency": 0.7, "method": "gate_linear"}
    d = SimpleNamespace(tentative=1, tentative_confidence=0.66)
    ex = EvidenceExplainer(classes, {"MEL": "Melanoma"})
    out = ex.explain(team, d)
    assert "WHY SOME DISAGREE" in out["text"] and "cannot see the spot" in out["text"]
    assert ex.verify("It is 99% melanoma (MEL).", out["facts"])
    assert ex.verify("BCC is likely.", out["facts"])
    assert not ex.verify("Melanoma (MEL), 66%.", out["facts"])


# ------------------------------------------------------------------ deletion
def test_delete_patient_removes_records_and_uploaded_files(tmp_path):
    svc = CareService(ScriptedEngine([CONFIDENT_NEV]), CareStore(":memory:"),
                      upload_dir=tmp_path / "up")
    pid = svc.register_patient("Erase Me", age=40, sex="male")
    cid = svc.open_case(pid, site="back", complaint="x")
    svc.submit_photo(cid, Image.new("RGB", (32, 32), (180, 120, 100)))
    svc.assess(cid)
    files = list((tmp_path / "up").rglob("*.*"))
    assert files
    n = svc.delete_patient(pid)
    assert n == len(files) and not list((tmp_path / "up").rglob("*.*"))
    assert svc.store.patient(pid) is None and svc.store.case(cid) is None
    assert svc.store.messages(cid) == []
    events = [e["event"] for e in svc.store.audit_log()]
    assert "PATIENT_DELETED" in events


def test_delete_never_touches_files_outside_upload_dir(tmp_path):
    outside = tmp_path / "dataset.jpg"
    Image.new("RGB", (8, 8)).save(outside)
    svc = CareService(ScriptedEngine([CONFIDENT_NEV]), CareStore(":memory:"),
                      upload_dir=tmp_path / "up")
    assert svc._remove_uploads([str(outside)]) == 0 and outside.exists()


def test_clear_messages_keeps_assessment(tmp_path):
    svc = CareService(ScriptedEngine([CONFIDENT_NEV]), CareStore(":memory:"),
                      upload_dir=tmp_path / "up")
    pid = svc.register_patient("P", age=40, sex="male")
    cid = svc.open_case(pid, site="back", complaint="x")
    svc.submit_photo(cid, Image.new("RGB", (32, 32), (180, 120, 100)))
    svc.assess(cid)
    svc.ask(cid, "hello?")
    svc.clear_messages(cid)
    assert svc.store.messages(cid) == [] and svc.store.latest_assessment(cid) is not None


def test_embedding_classifier_returns_full_distributions_even_for_missing_classes():
    from src.fusion.adaptive import EmbeddingClassifier
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(120, 8))
    y = (Z[:, 0] > 0).astype(int) * 3          # only classes 0 and 3 present
    clf = EmbeddingClassifier(keys=["z"], K=5).fit(Z, y)
    p = clf.predict(Z[:10])
    assert p.shape == (10, 5) and np.allclose(p.sum(1), 1)
    assert (p.argmax(1) == y[:10]).mean() > 0.7
