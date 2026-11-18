"""Source-free checks of paired proposals and admission after FP32 addition.

Every fixture is an explicit physical quadratic. These tests initialize no
model and generate no source cohort.
"""
import copy
import math
import unittest
from unittest.mock import patch

import torch

from mechinterp.loopscope13b.spectral_match import (
    COSINES, bracketed_crossing, make_spectral_shams, stable_seed,
)
from mechinterp.loopscope13c.roundtrip_match import (
    PhysicalGate, make_roundtrip_shams,
)


DTYPE = torch.float64
IDENTITY = {"teacher_seed": 17, "base_index": 2, "goal": 10}


def explicit_problem():
    K = torch.zeros((96, 96), dtype=DTYPE)
    K[:75, :75] = torch.eye(75, dtype=DTYPE)
    R = torch.eye(96, dtype=DTYPE)[:, :10].clone()
    selected = torch.zeros(96, dtype=DTYPE)
    selected[0] = selected[10] = 1.
    H = torch.zeros((96, 96), dtype=DTYPE)
    H[0, 0], H[10, 10] = .4, .2
    H[0, 10] = H[10, 0] = .07
    H[0, 11] = H[11, 0] = .12
    H[10, 12] = H[12, 10] = -.09
    H[11:75, 11:75] = torch.diag(torch.linspace(-2., 2., 64, dtype=DTYPE))
    # This scale makes all baseline proposals pass the real physical gate.
    # Controlled gate wrappers then isolate retry semantics.
    H *= 1e-3
    return K, R, selected, H


def make_reference(K, R, selected, H, **kwargs):
    return make_spectral_shams(
        selected, R, lambda value: K @ value, lambda value: H @ value,
        **IDENTITY, **kwargs)


def make_gate(K, R, selected, H, **overrides):
    values = {
        "selected": selected,
        "hpair": torch.zeros((2, 48), dtype=DTYPE),
        "hessian": H,
        "k_projector": K,
        "jacobian": R.T,
        "head": torch.zeros((10, 48), dtype=DTYPE),
        "egrad": torch.zeros(96, dtype=DTYPE),
        "row_basis": R,
        "q0": selected - R @ (R.T @ selected),
        "target_curvature": float(selected @ H @ selected),
    }
    values.update(overrides)
    return PhysicalGate(**values)


class ControlledRejection:
    """Preserve measured vectors while injecting a declared test rejection."""

    def __init__(self, gate, reject_first=0, reject_all=False):
        self.gate = gate
        self.hessian = gate.hessian
        self.reject_first = reject_first
        self.reject_all = reject_all
        self.calls = 0

    def roundtrip(self, value):
        return self.gate.roundtrip(value)

    def __call__(self, candidate, accepted_shams):
        receipt, raw = self.gate(candidate, accepted_shams)
        self.calls += 1
        if self.reject_all or self.calls <= self.reject_first:
            receipt = copy.deepcopy(receipt)
            receipt["passed"] = False
            receipt["failed_checks"] = sorted(set(receipt["failed_checks"] + ["curvature"]))
            receipt["max_errors"]["curvature"] = max(
                receipt["max_errors"]["curvature"], 2 * receipt["limits"]["curvature"])
        return receipt, raw


def attempt_key(record):
    return record["sham_index"], record["slice_index"], record["attempt"]


class RoundtripMatcherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_roundtrip_captures_addition_quantization_and_shape(self):
        K, R, selected, H = explicit_problem()
        gate = make_gate(K, R, selected, H,
                         hpair=torch.full((2, 48), 2. ** 24, dtype=DTYPE))
        intended = torch.zeros((2, 48), dtype=DTYPE)
        intended.reshape(-1)[:4] = torch.tensor([1., 3., -1., .5], dtype=DTYPE)
        actual = gate.roundtrip(intended)
        self.assertEqual(actual.shape, intended.shape)
        self.assertEqual(actual.dtype, DTYPE)
        torch.testing.assert_close(actual.reshape(-1)[:4],
                                   torch.tensor([0., 4., -1., 0.], dtype=DTYPE),
                                   rtol=0., atol=0.)
        self.assertFalse(torch.equal(actual, intended.float().double()))
        batch = torch.stack([intended.reshape(-1), intended.reshape(-1)])
        torch.testing.assert_close(gate.roundtrip(batch),
                                   torch.stack([actual.reshape(-1), actual.reshape(-1)]),
                                   rtol=0., atol=0.)

    def test_gate_metrics_match_independent_physical_quadratic(self):
        K, R, selected, H = explicit_problem()
        baseline = make_reference(K, R, selected, H)
        hpair = torch.linspace(-7., 9., 96, dtype=DTYPE).reshape(2, 48)
        gate = make_gate(K, R, selected, H, hpair=hpair)
        accepted, candidate = baseline.shams[0], baseline.shams[1]
        receipt, raw = gate(candidate, [accepted])
        intended = torch.stack([selected, accepted.reshape(-1), candidate.reshape(-1)])
        base32 = hpair.reshape(-1).float()
        actual = (base32 + intended.float()).double() - base32.double()
        direct_values = torch.einsum("ni,ij,nj->n", actual, H, actual)
        torch.testing.assert_close(raw["actual32"], actual[-1], rtol=0., atol=0.)
        torch.testing.assert_close(raw["intended"], intended[-1], rtol=0., atol=0.)
        torch.testing.assert_close(raw["dense_hvp"], H @ actual[-1], rtol=0., atol=2e-18)
        self.assertAlmostEqual(receipt["selected_curvature"], float(direct_values[0]), places=15)
        self.assertAlmostEqual(receipt["trial_curvature"], float(direct_values[-1]), places=15)
        expected_q_error = float((direct_values[1:] - direct_values[0]).abs().max())
        self.assertAlmostEqual(receipt["max_errors"]["curvature"], expected_q_error, places=15)
        expected_j = float(torch.linalg.vector_norm((actual - actual[0]) @ R, dim=1).max())
        self.assertAlmostEqual(receipt["max_errors"]["full_j"], expected_j, places=15)
        expected_k = float(torch.linalg.vector_norm(actual @ K - actual, dim=1).max())
        self.assertAlmostEqual(receipt["max_errors"]["k"], expected_k, places=15)
        expected_dose = float((actual.norm(dim=1) - selected.norm()).abs().max())
        self.assertAlmostEqual(receipt["max_errors"]["dose"], expected_dose, places=15)
        dark = actual @ (K - R @ R.T)
        units = dark / dark.norm(dim=1, keepdim=True)
        self.assertAlmostEqual(receipt["max_errors"]["source_abs_cosine"],
                               float((units[1:] @ units[0]).abs().max()), places=14)
        self.assertAlmostEqual(receipt["max_errors"]["pairwise_signed_cosine"],
                               float(units[1] @ units[2]), places=14)
        self.assertEqual(receipt["limits"]["curvature"],
                         1e-8 * max(1., abs(float(selected @ H @ selected))))

    def test_gate_uses_actual_donor_curvature(self):
        K, R, selected, H = explicit_problem()
        H = torch.zeros_like(H)
        H[0, 0] = 1.
        gate = make_gate(K, R, selected, H,
                         hpair=torch.full((2, 48), .1, dtype=DTYPE))
        candidate = selected.clone()
        candidate[10], candidate[11] = 0., 1.
        receipt, _ = gate(candidate, [])
        donor_actual = gate.roundtrip(selected)
        expected = float(donor_actual @ H @ donor_actual)
        self.assertEqual(receipt["selected_curvature"], expected)
        self.assertNotEqual(expected, float(selected @ H @ selected))
        self.assertEqual(receipt["max_errors"]["curvature"], 0.)

    def test_strict_curvature_threshold_rejects_without_dose_or_j_mismatch(self):
        K, R, selected, H = explicit_problem()
        H = torch.zeros_like(H)
        candidate = torch.zeros(96, dtype=DTYPE)
        candidate[0] = candidate[11] = 1.
        H[11, 11] = 2e-8
        receipt, _ = make_gate(K, R, selected, H)(candidate, [])
        self.assertFalse(receipt["passed"])
        self.assertEqual(receipt["failed_checks"], ["curvature"])
        self.assertEqual(receipt["max_errors"]["curvature"], 2e-8)
        H[11, 11] = .5e-8
        receipt, _ = make_gate(K, R, selected, H)(candidate, [])
        self.assertTrue(receipt["passed"], receipt["failed_checks"])

    def test_each_physical_constraint_has_a_detectable_failure(self):
        K, R, selected, H = explicit_problem()
        H = torch.zeros_like(H)
        standard = torch.zeros(96, dtype=DTYPE)
        standard[0] = standard[11] = 1.
        cases = []
        trial = standard.clone(); trial[95] = 1e-3
        cases.append(("k", trial, {}, []))
        trial = standard.clone(); trial[1] = 1e-3
        cases.append(("full_j", trial, {}, []))
        trial = standard.clone(); trial[11] = 1.01
        cases.append(("dose", trial, {}, []))
        head = torch.zeros((10, 48), dtype=DTYPE); head[0, 11] = 1.
        cases.append(("head", standard, {"head": head}, []))
        gradient = torch.zeros(96, dtype=DTYPE); gradient[11] = 1.
        cases.append(("energy", standard, {"egrad": gradient}, []))
        cases.append(("source_abs_cosine", selected, {}, []))
        cases.append(("pairwise_signed_cosine", standard, {}, [standard]))
        for failure, candidate, overrides, accepted in cases:
            with self.subTest(failure=failure):
                receipt, _ = make_gate(K, R, selected, H, **overrides)(candidate, accepted)
                self.assertFalse(receipt["passed"])
                self.assertIn(failure, receipt["failed_checks"])
                self.assertGreater(receipt["max_errors"][failure], receipt["limits"][failure])

    def test_actual_dark_angle_excludes_rounding_outside_k(self):
        K, R, selected, H = explicit_problem()
        H = torch.zeros_like(H)
        candidate = torch.zeros(96, dtype=DTYPE)
        candidate[0], candidate[10], candidate[11], candidate[95] = 1., .5, math.sqrt(.75), 3.
        gate = make_gate(K, R, selected, H)
        receipt, raw = gate(candidate, [])
        actual = raw["actual32"]
        actual_donor = gate.roundtrip(selected)
        N = K - R @ R.T
        dark, donor_dark = N @ actual, N @ actual_donor
        expected = float(dark @ donor_dark) / float(dark.norm() * donor_dark.norm())
        row_complement = actual - R @ (R.T @ actual)
        wrong_cosine = float(row_complement @ donor_dark) / float(row_complement.norm() * donor_dark.norm())
        self.assertAlmostEqual(receipt["trial_source_cosine"], expected, places=14)
        self.assertGreater(abs(receipt["trial_source_cosine"] - wrong_cosine), .2)
        self.assertIn("k", receipt["failed_checks"])

    def test_zero_applied_dark_radius_fails(self):
        K, R, selected, H = explicit_problem()
        gate = make_gate(K, R, selected, torch.zeros_like(H),
                         hpair=torch.full((2, 48), 2. ** 24, dtype=DTYPE))
        candidate = selected.clone()
        candidate[10], candidate[11] = 0., 1.
        receipt, _ = gate(candidate, [])
        self.assertFalse(receipt["passed"])
        self.assertTrue(receipt["zero_actual_dark_radius"])
        self.assertIn("zero_actual_dark_radius", receipt["failed_checks"])

    def test_unfiltered_pairing_is_exact_and_reference_stays_unchanged(self):
        problem = explicit_problem()
        baseline = make_reference(*problem)
        old_receipt = copy.deepcopy(baseline.receipt)
        old_raw = {key: value.clone() for key, value in baseline.raw.items()}
        gate = make_gate(*problem)
        result = make_roundtrip_shams(baseline, gate, **IDENTITY)
        self.assertTrue(result.eligible, result.reason)
        self.assertEqual(len(result.shams), 8)
        self.assertEqual(result.receipt["construction_hvp_count"], 0)
        self.assertEqual(result.receipt["seed_namespace"], "LGA10B")
        self.assertEqual([attempt_key(a) for a in result.receipt["attempts"]],
                         [attempt_key(a) for a in baseline.receipt["attempts"]])
        for original, paired in zip(baseline.shams, result.shams):
            torch.testing.assert_close(original, paired, rtol=0., atol=0.)
        self.assertEqual(baseline.receipt, old_receipt)
        for key, old in old_raw.items():
            torch.testing.assert_close(baseline.raw[key], old, rtol=0., atol=0.)
        self.assertEqual(result.raw["candidate_start"].shape, (8, 64))
        for key in ("candidate_intended", "candidate_actual32", "candidate_hvp_dense"):
            self.assertEqual(result.raw[key].shape, (8, 96))
        for record in result.receipt["attempts"]:
            index = record["candidate_index"]
            raw_actual = result.raw["candidate_actual32"][index]
            torch.testing.assert_close(raw_actual,
                                       gate.roundtrip(result.raw["candidate_intended"][index]),
                                       rtol=0., atol=0.)
            torch.testing.assert_close(result.raw["candidate_hvp_dense"][index],
                                       gate.hessian @ raw_actual, rtol=0., atol=2e-18)
            self.assertTrue(record["physical"]["passed"])

    def test_first_physical_rejection_consumes_attempt_and_preserves_proposal_keys(self):
        problem = explicit_problem()
        baseline = make_reference(*problem)
        gate = ControlledRejection(make_gate(*problem), reject_first=1)
        result = make_roundtrip_shams(baseline, gate, **IDENTITY, n_shams=2)
        self.assertTrue(result.eligible, result.reason)
        attempts = result.receipt["attempts"]
        self.assertEqual(attempt_key(attempts[0]), (0, 0, 0))
        self.assertEqual(attempt_key(attempts[1]), (0, 0, 1))
        self.assertFalse(attempts[0]["accepted"])
        self.assertEqual(attempts[0]["reason"], "FP32_CURVATURE_REJECTION")
        self.assertTrue(attempts[1]["accepted"])
        originals = {
            attempt_key(record): vector.reshape(-1)
            for record, vector in zip(
                (r for r in baseline.receipt["attempts"] if r["accepted"]), baseline.shams)
        }
        shared_count = 0
        V = baseline.raw["complement_basis"]
        for record in attempts:
            key, index = attempt_key(record), record["candidate_index"]
            expected_seed = stable_seed("ambient", 17, 2, 10, *key)
            self.assertEqual(record["seed"], expected_seed)
            draw = torch.randn(96, dtype=DTYPE,
                               generator=torch.Generator().manual_seed(expected_seed))
            expected_start = V.T @ draw
            expected_start /= expected_start.norm()
            torch.testing.assert_close(result.raw["candidate_start"][index],
                                       expected_start, rtol=0., atol=0.)
            if key in originals:
                shared_count += 1
                torch.testing.assert_close(result.raw["candidate_intended"][index],
                                           originals[key], rtol=0., atol=0.)
        self.assertGreaterEqual(shared_count, 2)
        self.assertEqual(gate.calls, result.receipt["candidate_evaluations"])

    def test_all_rejections_exhaust_each_frozen_slice_budget(self):
        problem = explicit_problem()
        baseline = make_reference(*problem)
        gate = ControlledRejection(make_gate(*problem), reject_all=True)
        result = make_roundtrip_shams(baseline, gate, **IDENTITY,
                                      n_shams=1, max_attempts=2)
        self.assertFalse(result.eligible)
        self.assertEqual(result.reason, "ROUNDED_CANDIDATE_BUDGET_EXHAUSTED")
        self.assertEqual(len(result.shams), 0)
        expected = [(0, slice_index, attempt)
                    for slice_index, row in enumerate(baseline.receipt["slices"])
                    if row["attainable"] for attempt in range(2)]
        attempts = result.receipt["attempts"]
        self.assertEqual([attempt_key(a) for a in attempts], expected)
        self.assertEqual(len(attempts), 2 * len(COSINES))
        self.assertEqual([a["candidate_index"] for a in attempts], list(range(len(attempts))))
        self.assertEqual(gate.calls, len(attempts))
        self.assertTrue(all(not a["accepted"] and not a["physical"]["passed"] for a in attempts))
        self.assertEqual(result.raw["candidate_intended"].shape[0], len(attempts))

    def test_missing_crossing_keeps_attempt_start_and_candidate_mapping(self):
        problem = explicit_problem()
        baseline = make_reference(*problem)
        gate = ControlledRejection(make_gate(*problem))
        calls = 0

        def controlled_crossing(*args, **kwargs):
            nonlocal calls
            calls += 1
            return (None, None) if calls == 1 else bracketed_crossing(*args, **kwargs)

        with patch("mechinterp.loopscope13c.roundtrip_match.bracketed_crossing",
                   side_effect=controlled_crossing):
            result = make_roundtrip_shams(baseline, gate, **IDENTITY, n_shams=1)
        self.assertTrue(result.eligible, result.reason)
        attempts = result.receipt["attempts"]
        self.assertEqual([attempt_key(record) for record in attempts], [(0, 0, 0), (0, 0, 1)])
        self.assertEqual([record["start_index"] for record in attempts], [0, 1])
        self.assertEqual(attempts[0]["reason"], "NO_NUMERICAL_PATH_CROSSING")
        self.assertIsNone(attempts[0]["candidate_index"])
        self.assertIsNone(attempts[0]["physical"])
        self.assertEqual(attempts[1]["candidate_index"], 0)
        self.assertEqual(gate.calls, 1)
        self.assertEqual(result.raw["attempt_start"].shape, (2, 64))
        self.assertEqual(result.raw["candidate_start"].shape, (1, 64))
        torch.testing.assert_close(result.raw["candidate_start"][0],
                                   result.raw["attempt_start"][1], rtol=0., atol=0.)
        V = baseline.raw["complement_basis"]
        for record in attempts:
            draw = torch.randn(96, dtype=DTYPE,
                               generator=torch.Generator().manual_seed(record["seed"]))
            expected = V.T @ draw
            expected /= expected.norm()
            torch.testing.assert_close(result.raw["attempt_start"][record["start_index"]],
                                       expected, rtol=0., atol=0.)

    def test_same_environment_replay_reproduces_rejections_and_captures(self):
        problem = explicit_problem()
        baseline = make_reference(*problem)
        first = make_roundtrip_shams(
            baseline, ControlledRejection(make_gate(*problem), reject_first=1), **IDENTITY)
        second = make_roundtrip_shams(
            baseline, ControlledRejection(make_gate(*problem), reject_first=1), **IDENTITY)
        self.assertTrue(first.eligible, first.reason)
        self.assertEqual(first.receipt["attempts"], second.receipt["attempts"])
        self.assertEqual(first.receipt["rejection_counts"], second.receipt["rejection_counts"])
        for key in ("candidate_start", "candidate_intended", "candidate_actual32",
                    "candidate_hvp_dense", "shams"):
            torch.testing.assert_close(first.raw[key], second.raw[key], rtol=0., atol=0.)

    def test_empty_grid_failure_is_shared_without_new_proposals(self):
        K, R, selected, H = explicit_problem()
        H = torch.zeros_like(H); H[10, 10] = 1.
        baseline = make_reference(K, R, selected, H)
        self.assertEqual(baseline.reason, "GRID_NO_ATTAINABLE_TARGET")
        gate = ControlledRejection(make_gate(K, R, selected, H))
        result = make_roundtrip_shams(baseline, gate, **IDENTITY)
        self.assertEqual(result.reason, baseline.reason)
        self.assertEqual(gate.calls, 0)
        self.assertEqual(result.receipt["attempts"], [])
        self.assertEqual(result.raw["candidate_start"].shape, (0, 64))

    def test_baseline_diversity_failure_keeps_geometry_available(self):
        K, R, selected, H = explicit_problem()
        H = torch.zeros_like(H)
        slope = math.sqrt((1 - .85) / (1 + .85))
        H[0, 10] = H[10, 0] = -1e-3
        H[0, 11] = H[11, 0] = -1e-3 * slope
        baseline = make_reference(K, R, selected, H, n_shams=2, max_attempts=1)
        self.assertEqual(baseline.reason, "INSUFFICIENT_DISTINCT_SHAMS")
        gate = ControlledRejection(make_gate(K, R, selected, H))
        result = make_roundtrip_shams(baseline, gate, **IDENTITY,
                                      n_shams=2, max_attempts=1)
        self.assertEqual(result.reason, "ROUNDED_CANDIDATE_BUDGET_EXHAUSTED")
        self.assertEqual(result.receipt["reference_reason"], baseline.reason)
        self.assertGreater(gate.calls, 0)
        self.assertEqual(len(result.shams), 1)
        self.assertEqual(result.receipt["attempts"][-1]["reason"], "PAIRWISE_DIVERSITY_REJECTION")
        self.assertIsNotNone(result.receipt["attempts"][-1]["physical"])

    def test_identity_mismatch_fails_before_candidate_evaluation(self):
        problem = explicit_problem()
        baseline = make_reference(*problem)
        gate = ControlledRejection(make_gate(*problem))
        with self.assertRaises(ValueError):
            make_roundtrip_shams(baseline, gate, teacher_seed=18, base_index=2, goal=10)
        self.assertEqual(gate.calls, 0)


if __name__ == "__main__":
    unittest.main()
