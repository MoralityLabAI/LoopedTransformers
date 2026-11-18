"""Source-free tests of shared native geometry and derivative-state pairing."""
import math
import unittest
from unittest.mock import patch

import torch

from mechinterp.loopscope13b.source_feasibility import QuadraticObserver
from mechinterp.loopscope13b.spectral_match import stable_seed
from mechinterp.loopscope13d import geometry


IDENTITY = {"teacher_seed": 61064, "base_index": 0, "goal": 0,
            "original_class": 0, "donor_class": 1}


class _Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        value = torch.zeros((10, 48), dtype=torch.float64)
        value[:, 38:] = torch.eye(10, dtype=torch.float64)
        self.weight = torch.nn.Parameter(value, requires_grad=False)

    def forward(self, state):
        return state @ self.weight.T


class _FourthPowerBlock(torch.nn.Module):
    def forward(self, state):
        return torch.roll(state, 7, dims=-1).square()


class _MixedBlock(torch.nn.Module):
    def forward(self, state):
        value = torch.roll(state, 7, dims=-1)
        return value + .1 * value * torch.roll(value, 1, dims=-1)


class _Toy(torch.nn.Module):
    def __init__(self, mixed=False):
        super().__init__()
        self.head = _Head()
        self.block = _MixedBlock() if mixed else _FourthPowerBlock()


def fixture(mixed=False):
    model = _Toy(mixed=mixed).eval()
    h32 = torch.ones((1, 81, 48), dtype=torch.float32)
    donor = h32.clone()
    donor[0, 80, 1] += .25
    donor[0, 0, 25] += .2
    reference = h32.double()
    reference[0, 0, 24] = 1.25
    reference[0, 0, 25] = 1.5
    reference[0, 0, 26] = .8
    # This token/feature is outside the toy goal continuation. Its coarse
    # FP32 spacing makes using the derivative state as the addition base
    # observable while leaving the analytic goal J and H unchanged.
    reference[0, 80, 1] = 2. ** 24
    projector = torch.eye(48, dtype=torch.float64)
    projector[38:, 38:] = 0.
    parent = geometry.build_common_native_parent(
        model, QuadraticObserver(), h32, donor,
        torch.zeros(48, dtype=torch.float64), torch.ones(48, dtype=torch.float64),
        head_projector=projector, **IDENTITY)
    return model, h32, donor, reference, projector, parent


def build(model, derivative, h32, projector, parent):
    return geometry.build_geometry(model, derivative, h32, parent, projector, **IDENTITY)


class NativeGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.model, cls.h32, cls.donor, cls.reference, cls.P, cls.parent = fixture()
        cls.native = build(cls.model, cls.h32.double(), cls.h32, cls.P, cls.parent)
        cls.stale = build(cls.model, cls.reference, cls.h32, cls.P, cls.parent)

    def test_common_parent_lifts_before_natural_subtraction(self):
        model, h32, donor, _, projector, _ = fixture()
        h32[0, 80, 0] = 1.
        donor[0, 80, 0] = 2. ** -25
        before_h, before_d = h32.clone(), donor.clone()
        parent = geometry.build_common_native_parent(
            model, QuadraticObserver(), h32, donor,
            torch.zeros(48, dtype=torch.float64), torch.ones(48, dtype=torch.float64),
            head_projector=projector, **IDENTITY)
        exact = (donor.double() - h32.double())[0, [80, 0]]
        rounded_first = (donor - h32).double()[0, [80, 0]]
        self.assertFalse(torch.equal(exact, rounded_first))
        self.assertTrue(torch.equal(parent["fiber"].donor, exact))
        self.assertEqual(parent["radius"], float(torch.linalg.vector_norm(exact)))
        self.assertTrue(parent["native_source"]["retained_auxiliary_parent_eligibility"])
        self.assertTrue(torch.equal(h32, before_h) and torch.equal(donor, before_d))

    def test_both_arms_share_exact_native_donor_dose_and_k(self):
        for name in ("selected", "natural_pair", "energy_gradient", "k_projector", "application_pair64"):
            self.assertTrue(torch.equal(self.native.raw[name], self.stale.raw[name]), name)
        self.assertEqual(self.native.receipt["radius"], self.stale.receipt["radius"])
        natural = (self.donor.double() - self.h32.double())[0, [80, 0]].reshape(96)
        self.assertTrue(torch.equal(self.native.raw["natural_pair"], natural))
        self.assertAlmostEqual(float(self.native.raw["selected"].norm()), float(natural.norm()), places=14)
        self.assertTrue(torch.equal(self.native.raw["energy_gradient"], self.h32[0, [80, 0]].double().reshape(96)))
        self.assertTrue(self.native.receipt["derivative_state_equals_native_lift"])
        self.assertFalse(self.stale.receipt["derivative_state_equals_native_lift"])

    def test_full_j_and_dense_h_use_the_requested_derivative_state(self):
        # Two blocks map x to roll(x,14)^4; the ten head rows select
        # original goal features24..33. Its pair J and margin H are analytic.
        for result, state in ((self.native, self.h32.double()), (self.stale, self.reference)):
            self.assertIsNotNone(result.future, result.receipt)
            self.assertIsNotNone(result.hessian, result.receipt)
            expected_j = torch.zeros((10, 96), dtype=torch.float64)
            for label in range(10):
                expected_j[label, 48 + 24 + label] = 4 * state[0, 0, 24 + label] ** 3
            expected_h = torch.zeros((96, 96), dtype=torch.float64)
            expected_h[48 + 25, 48 + 25] = 12 * state[0, 0, 25] ** 2
            expected_h[48 + 24, 48 + 24] = -12 * state[0, 0, 24] ** 2
            torch.testing.assert_close(result.raw["future_jacobian"], expected_j, atol=1e-12, rtol=0.)
            torch.testing.assert_close(result.hessian_raw, expected_h, atol=1e-12, rtol=0.)
            torch.testing.assert_close(result.hessian, expected_h, atol=1e-12, rtol=0.)
        self.assertFalse(torch.equal(self.native.hessian, self.stale.hessian))

    def test_selection_applies_at_native_base_in_both_arms(self):
        native_base = self.h32[0, [80, 0]].reshape(96)
        for result in (self.native, self.stale):
            self.assertIsNotNone(result.plan, result.receipt)
            candidates = result.plan.raw["candidate_intended"]
            self.assertGreater(len(candidates), 0)
            expected = (native_base + candidates.float()).double() - native_base.double()
            self.assertTrue(torch.equal(result.plan.raw["candidate_actual32"], expected))
            self.assertTrue(torch.equal(result.raw["application_state32"], self.h32))
        stale_base = self.reference[0, [80, 0]].reshape(96).float()
        candidates = self.stale.plan.raw["candidate_intended"]
        wrong = (stale_base + candidates.float()).double() - stale_base.double()
        self.assertFalse(torch.equal(self.stale.plan.raw["candidate_actual32"], wrong))

    def test_own_construction_targets_and_completed_hvp_counts_are_explicit(self):
        for result in (self.native, self.stale):
            selected = result.raw["selected"]
            target = float(selected @ result.hessian @ selected)
            self.assertAlmostEqual(result.receipt["construction_target_curvature"], target, places=12)
            expected_tol = 1e-8 * max(1., abs(result.reference.receipt["selected_curvature"]))
            self.assertEqual(result.receipt["construction_curvature_tolerance"], expected_tol)
            self.assertEqual(result.receipt["hvp_counts"]["reference_construction"], 66)
            self.assertEqual(result.receipt["hvp_counts"]["dense_primary"], 96)
            self.assertEqual(result.receipt["hvp_counts"], result.receipt["hvp_attempt_counts"])
            before = result.receipt["hvp_counts"].get("test_fresh", 0)
            vector = torch.linspace(-1., 1., 96, dtype=torch.float64)
            fresh = result.hvp(vector, "test_fresh")
            torch.testing.assert_close(fresh, result.hessian @ vector, atol=1e-12, rtol=0.)
            self.assertEqual(result.receipt["hvp_counts"]["test_fresh"], before + 1)
            self.assertGreater(result.receipt["hvp_seconds"]["test_fresh"], 0.)
        self.assertNotEqual(self.native.receipt["construction_target_curvature"],
                            self.stale.receipt["construction_target_curvature"])

    def test_common_ambient_key_can_produce_distinct_geometry_proposals(self):
        model, h32, _, reference, projector, parent = fixture(mixed=True)
        native = build(model, h32.double(), h32, projector, parent)
        stale = build(model, reference, h32, projector, parent)
        self.assertIsNotNone(native.reference, native.receipt)
        self.assertIsNotNone(stale.reference, stale.receipt)
        key = (0, 0, 0)
        seed = stable_seed("ambient", IDENTITY["teacher_seed"], 0, 0, *key)
        draw = torch.randn(96, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))
        proposals = []
        for result in (native, stale):
            reference_plan = result.reference
            V = reference_plan.raw["complement_basis"]
            start = V.T @ draw
            start /= start.norm()
            r = reference_plan.receipt["null_radius"]
            proposal = (reference_plan.raw["z0"] - .85 * reference_plan.raw["q0"]
                        + r * math.sqrt(1 - .85**2) * (V @ start))
            proposals.append(proposal)
            if reference_plan.receipt["attempts"]:
                first = reference_plan.receipt["attempts"][0]
                self.assertEqual(first["seed"], stable_seed("ambient", IDENTITY["teacher_seed"], 0, 0,
                    first["sham_index"], first["slice_index"], first["attempt"]))
        self.assertGreater(float(torch.linalg.vector_norm(proposals[0] - proposals[1])), 1e-6)
        self.assertTrue(torch.equal(native.raw["selected"], stale.raw["selected"]))

    def test_finite_dense_skew_failure_preserves_reference_and_raw_hessian(self):
        real_dense = geometry.dense_margin_hessian

        def fail_skew(oracle):
            dense, receipt, raw = real_dense(oracle)
            raw = raw.clone()
            raw[0, 1] += .01
            receipt = dict(receipt, passed=False,
                           skew_frobenius_norm=float(torch.linalg.vector_norm(raw - raw.T)))
            return .5 * (raw + raw.T), receipt, raw

        with patch.object(geometry, "dense_margin_hessian", side_effect=fail_skew):
            result = build(self.model, self.reference, self.h32, self.P, self.parent)
        self.assertFalse(result.eligible)
        self.assertEqual(result.reason, "DENSE_HESSIAN_SYMMETRY_FAILURE")
        self.assertIsNotNone(result.reference)
        self.assertIsNone(result.plan)
        self.assertIsNotNone(result.hessian_raw)
        self.assertFalse(torch.equal(result.hessian_raw, result.hessian_raw.T))
        self.assertEqual(result.receipt["hvp_counts"]["dense_primary"], 96)

    def test_thrown_oracle_failure_keeps_completed_trace_and_attempt_count(self):
        real_factory = geometry.native_margin_hvp

        def faulty_factory(*args):
            real = real_factory(*args)
            count = 0

            def faulty(vector):
                nonlocal count
                count += 1
                if count == 3:
                    raise ArithmeticError("synthetic third HVP failure")
                return real(vector)
            return faulty

        with patch.object(geometry, "native_margin_hvp", side_effect=faulty_factory):
            result = build(self.model, self.reference, self.h32, self.P, self.parent)
        self.assertEqual(result.reason, "EXPLICIT_GEOMETRY_EXCEPTION")
        self.assertEqual(result.receipt["exception"]["stage"], "spectral_reference")
        self.assertEqual(result.receipt["hvp_counts"]["reference_construction"], 2)
        self.assertEqual(result.receipt["hvp_attempt_counts"]["reference_construction"], 3)
        self.assertEqual(result.raw["partial_hvp__reference_construction__directions"].shape, (3, 96))
        self.assertEqual(result.raw["partial_hvp__reference_construction__values"].shape, (2, 96))
        self.assertIsNotNone(result.future)

    def test_identity_or_native_base_mismatch_stops_before_derivatives(self):
        changed = self.h32.clone()
        changed[0, 0, 0] += .25
        with patch.object(geometry, "native_margin_hvp") as oracle:
            with self.assertRaises(ValueError):
                build(self.model, self.reference, changed, self.P, self.parent)
            with self.assertRaises(ValueError):
                geometry.build_geometry(self.model, self.reference, self.h32, self.parent, self.P,
                    **dict(IDENTITY, teacher_seed=18))
            oracle.assert_not_called()


if __name__ == "__main__":
    unittest.main()
