"""
tests/test_evidential.py - Evidential opinions and Dempster-Shafer combination.

The three properties that justify choosing DS over masking are tested directly:
identity element for absent modalities, conflict separated from ignorance, and
associativity/commutativity so the fold order cannot matter.
"""
import numpy as np
import pytest
import torch

from src.evidence.evidential import (
    EvidentialHead,
    Opinion,
    conflict_and_ignorance,
    dirichlet_kl_to_uniform,
    ds_combine,
    edl_loss,
    fold_opinions,
    gate_opinion,
    opinion_from_evidence,
    opinion_from_probs,
    vacuous,
)

K = 6


def confident(cls, strength=50.0, batch=1):
    e = torch.zeros(batch, K)
    e[:, cls] = strength
    return opinion_from_evidence(e)


class TestOpinionInvariants:
    def test_belief_plus_ignorance_is_one(self):
        for e in (torch.zeros(4, K), torch.rand(4, K) * 10, torch.rand(4, K) * 1000):
            assert opinion_from_evidence(e).check()

    def test_no_evidence_is_total_ignorance(self):
        op = opinion_from_evidence(torch.zeros(3, K))
        assert torch.allclose(op.u, torch.ones(3, 1))
        assert torch.allclose(op.belief, torch.zeros(3, K))

    def test_more_evidence_lowers_ignorance(self):
        us = [confident(0, s).u.item() for s in (1.0, 10.0, 100.0, 1000.0)]
        assert us == sorted(us, reverse=True)

    def test_pignistic_probability_is_a_distribution(self):
        op = opinion_from_evidence(torch.rand(5, K) * 20)
        p = op.probability
        assert torch.allclose(p.sum(-1), torch.ones(5), atol=1e-5)
        assert (p >= 0).all()

    def test_vacuous_gives_uniform_probability(self):
        p = vacuous(2, K).probability
        assert torch.allclose(p, torch.full((2, K), 1.0 / K))

    def test_alpha_and_strength_round_trip(self):
        e = torch.rand(4, K) * 7
        op = opinion_from_evidence(e)
        assert torch.allclose(op.alpha, e + 1.0, atol=1e-4)

    def test_opinion_from_probs_respects_strength(self):
        probs = torch.full((2, K), 1.0 / K)
        weak = opinion_from_probs(probs, strength=K)
        assert weak.u.mean().item() == pytest.approx(1.0, abs=1e-3)


class TestDSIdentity:
    def test_vacuous_is_the_identity_element(self):
        """The property that makes a missing modality a no-op."""
        op = confident(2, 30.0, batch=3)
        out = ds_combine(op, vacuous(3, K))
        assert torch.allclose(out.belief, op.belief, atol=1e-5)
        assert torch.allclose(out.u, op.u, atol=1e-5)

    def test_identity_from_either_side(self):
        op = confident(1, 25.0)
        assert torch.allclose(ds_combine(vacuous(1, K), op).belief,
                              ds_combine(op, vacuous(1, K)).belief, atol=1e-6)

    def test_many_vacuous_change_nothing(self):
        op = confident(3, 40.0)
        folded = fold_opinions([op] + [vacuous(1, K) for _ in range(5)])
        assert torch.allclose(folded.belief, op.belief, atol=1e-5)

    def test_all_vacuous_stays_vacuous(self):
        out = fold_opinions([vacuous(2, K) for _ in range(3)])
        assert torch.allclose(out.u, torch.ones(2, 1), atol=1e-6)
        assert torch.allclose(out.probability, torch.full((2, K), 1.0 / K), atol=1e-6)

    def test_gate_opinion_replaces_with_vacuous(self):
        op = confident(0, 50.0, batch=3)
        gated = gate_opinion(op, torch.tensor([1.0, 0.0, 1.0]))
        assert torch.allclose(gated.u[1], torch.ones(1))
        assert torch.allclose(gated.belief[1], torch.zeros(K))
        assert torch.allclose(gated.belief[0], op.belief[0], atol=1e-6)


class TestDSAlgebra:
    def test_commutative(self):
        a, b = confident(0, 20.0), confident(1, 15.0)
        assert torch.allclose(ds_combine(a, b).belief, ds_combine(b, a).belief, atol=1e-6)

    def test_associative(self):
        a, b, c = confident(0, 20.0), confident(0, 12.0), confident(2, 8.0)
        left = ds_combine(ds_combine(a, b), c)
        right = ds_combine(a, ds_combine(b, c))
        assert torch.allclose(left.belief, right.belief, atol=1e-5)

    def test_fold_order_does_not_matter(self):
        ops = [confident(0, 20.0), confident(0, 10.0), confident(1, 5.0)]
        assert torch.allclose(fold_opinions(ops).belief,
                              fold_opinions(list(reversed(ops))).belief, atol=1e-5)

    def test_agreement_reduces_ignorance(self):
        a, b = confident(0, 10.0), confident(0, 10.0)
        assert ds_combine(a, b).u.item() < a.u.item()

    def test_output_is_a_valid_opinion(self):
        out = fold_opinions([confident(0, 9.0), confident(3, 4.0), vacuous(1, K)])
        assert out.check()


class TestConflictVsIgnorance:
    def test_conflict_high_when_confident_and_disagreeing(self):
        out = ds_combine(confident(0, 100.0), confident(5, 100.0))
        assert out.conflict.item() > 0.8

    def test_conflict_low_when_agreeing(self):
        out = ds_combine(confident(0, 100.0), confident(0, 100.0))
        assert out.conflict.item() < 0.1

    def test_conflict_low_when_both_ignorant(self):
        """The case JS divergence cannot distinguish from agreement."""
        out = ds_combine(vacuous(1, K), vacuous(1, K))
        assert out.conflict.item() < 1e-6
        assert out.u.item() == pytest.approx(1.0, abs=1e-6)

    def test_conflict_and_ignorance_are_independent_signals(self):
        """The load-bearing claim: the two failure modes are separable.

        Confident disagreement -> high conflict, LOW ignorance.
        Shared ignorance       -> low conflict, HIGH ignorance.
        A single scalar cannot express both, which is why the old
        `consistency = 1 - mean JS` score conflated them.
        """
        disagree = ds_combine(confident(0, 100.0), confident(5, 100.0))
        unknown = ds_combine(vacuous(1, K), vacuous(1, K))
        assert disagree.conflict.item() > 0.8 and disagree.u.item() < 0.2
        assert unknown.conflict.item() < 0.1 and unknown.u.item() > 0.8

    def test_conflict_inflates_ignorance(self):
        agree = ds_combine(confident(0, 20.0), confident(0, 20.0))
        clash = ds_combine(confident(0, 20.0), confident(5, 20.0))
        assert clash.u.item() > agree.u.item()

    def test_diagnostic_bundle_reports_per_modality_ignorance(self):
        ops = {"image": confident(0, 50.0), "text": vacuous(1, K),
               "meta": confident(0, 20.0)}
        out = conflict_and_ignorance(ops)
        assert out["per_modality_ignorance"]["text"].item() == pytest.approx(1.0, abs=1e-6)
        assert out["per_modality_ignorance"]["image"].item() < 0.2
        assert torch.allclose(out["probability"].sum(-1), torch.ones(1), atol=1e-5)


class TestEvidentialHead:
    def test_output_shape_and_validity(self):
        head = EvidentialHead(d_in=32, num_classes=K)
        op = head(torch.randn(8, 32))
        assert op.belief.shape == (8, K) and op.u.shape == (8, 1)
        assert op.check()

    def test_evidence_is_non_negative(self):
        head = EvidentialHead(d_in=16, num_classes=K)
        for _ in range(5):
            op = head(torch.randn(10, 16) * 10)
            assert (op.belief >= 0).all() and (op.u > 0).all()

    def test_gradients_flow(self):
        head = EvidentialHead(d_in=16, num_classes=K)
        op = head(torch.randn(4, 16))
        edl_loss(op, torch.randint(0, K, (4,))).backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in head.parameters())

    def test_linear_variant(self):
        head = EvidentialHead(d_in=16, num_classes=K, hidden=None)
        assert head(torch.randn(3, 16)).check()


class TestEDLLoss:
    def test_finite_and_non_negative(self):
        op = opinion_from_evidence(torch.rand(8, K) * 10)
        for anneal in (0.0, 0.5, 1.0):
            v = edl_loss(op, torch.randint(0, K, (8,)), anneal=anneal)
            assert torch.isfinite(v) and v.item() >= 0

    def test_correct_confident_beats_wrong_confident(self):
        y = torch.zeros(1, dtype=torch.long)
        good = edl_loss(confident(0, 100.0), y, anneal=0.0)
        bad = edl_loss(confident(5, 100.0), y, anneal=0.0)
        assert good.item() < bad.item()

    def test_kl_term_penalises_misplaced_evidence(self):
        """Evidence on a wrong class should be penalised once the KL is on."""
        y = torch.zeros(1, dtype=torch.long)
        wrong = confident(3, 50.0)
        assert edl_loss(wrong, y, anneal=1.0).item() > edl_loss(wrong, y, anneal=0.0).item()

    def test_kl_to_uniform_is_zero_for_uniform(self):
        alpha = torch.ones(2, K)
        assert torch.allclose(dirichlet_kl_to_uniform(alpha), torch.zeros(2), atol=1e-5)

    def test_class_weight_applies(self):
        op = opinion_from_evidence(torch.rand(6, K) * 5)
        y = torch.randint(0, K, (6,))
        w = torch.ones(K); w[5] = 10.0
        assert edl_loss(op, y, 0.0, w).item() != pytest.approx(
            edl_loss(op, y, 0.0).item(), abs=1e-6)


class TestShapeGuards:
    def test_mismatched_class_count_raises(self):
        a = opinion_from_evidence(torch.rand(2, 6))
        b = opinion_from_evidence(torch.rand(2, 4))
        with pytest.raises(ValueError, match="class count mismatch"):
            ds_combine(a, b)

    def test_empty_fold_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            fold_opinions([])
