"""Source-free analytic tests; no model, source generator, or data imports."""

import math
import unittest

import numpy as np
import torch

from mechinterp.loopscope13b.sphere_solver import SECULAR_ITERATIONS, sphere_extrema


DTYPE = torch.float64


class SphereSolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_certified(self, A, b, result):
        """Check reported certificates against the original matrix directly."""
        identity = torch.eye(A.shape[0], dtype=DTYPE)
        for extremum in (result.minimum, result.maximum):
            self.assertTrue(extremum.certificate_ok, extremum)
            self.assertLess(extremum.norm_error, 2e-14)
            self.assertLess(abs(float(extremum.vector.norm()) - 1.0), 2e-14)
            if extremum.scale == 0.0:
                self.assertEqual(extremum.value, 0.0)
                self.assertEqual(extremum.dual_bound, 0.0)
                continue
            scale = extremum.scale
            direct_value = float(extremum.vector @ (A / scale) @ extremum.vector
                                 + 2.0 * ((b / scale) @ extremum.vector))
            self.assertAlmostEqual(extremum.value / scale, direct_value, delta=2e-13)
            direct_residual = torch.linalg.vector_norm(
                (A / scale) @ extremum.vector
                - (extremum.multiplier / scale) * extremum.vector + b / scale
            )
            self.assertLess(float(direct_residual), extremum.certificate_tolerance)
            oriented = (A / scale) - (extremum.multiplier / scale) * identity
            if extremum.sense == "maximum":
                oriented = -oriented
            actual_gap = float(torch.linalg.eigvalsh(oriented)[0])
            self.assertGreaterEqual(actual_gap, -extremum.certificate_tolerance)
            self.assertLessEqual(extremum.psd_lower_bound / scale,
                                 actual_gap + 2e-14)
            self.assertLess(extremum.normalized_bound_gap, 1e-8)
        self.assertLessEqual(result.minimum.dual_bound, result.minimum.value)
        self.assertGreaterEqual(result.maximum.dual_bound, result.maximum.value)

    def test_scalar_extrema_allow_equality_multipliers_of_either_sign(self):
        for a in (-5.0, 5.0):
            A = torch.tensor([[a]], dtype=DTYPE)
            b = torch.tensor([0.7], dtype=DTYPE)
            result = sphere_extrema(A, b)
            self.assert_certified(A, b, result)
            self.assertAlmostEqual(result.minimum.value, a - 1.4, places=14)
            self.assertAlmostEqual(result.maximum.value, a + 1.4, places=14)
            self.assertAlmostEqual(result.minimum.multiplier, a - 0.7, places=14)
            self.assertAlmostEqual(result.maximum.multiplier, a + 0.7, places=14)
            self.assertEqual(float(result.minimum.vector[0]), -1.0)
            self.assertEqual(float(result.maximum.vector[0]), 1.0)
            self.assertLessEqual(result.minimum.dual_bound, a - 1.4)
            self.assertGreaterEqual(result.maximum.dual_bound, a + 1.4)
            self.assertEqual(result.minimum.secular_iterations, SECULAR_ITERATIONS)
        self.assertEqual(SECULAR_ITERATIONS, 64)

    def test_zero_problem_has_an_exact_deterministic_certificate(self):
        A = torch.zeros((4, 4), dtype=DTYPE)
        b = torch.zeros(4, dtype=DTYPE)
        result = sphere_extrema(A, b)
        self.assert_certified(A, b, result)
        for extremum in (result.minimum, result.maximum):
            torch.testing.assert_close(extremum.vector, torch.tensor([1., 0., 0., 0.], dtype=DTYPE))
            self.assertTrue(extremum.hard_case)
            self.assertEqual(extremum.secular_iterations, 0)
            self.assertEqual(extremum.scale, 0.0)

    def test_zero_linear_term_and_repeated_extreme_eigenvalue(self):
        A = torch.diag(torch.tensor([-2., -2., 3., 3.], dtype=DTYPE))
        b = torch.zeros(4, dtype=DTYPE)
        result = sphere_extrema(A, b)
        self.assert_certified(A, b, result)
        self.assertEqual(result.minimum.value, -2.0)
        self.assertEqual(result.maximum.value, 3.0)
        torch.testing.assert_close(result.minimum.vector, torch.tensor([1., 0., 0., 0.], dtype=DTYPE))
        torch.testing.assert_close(result.maximum.vector, torch.tensor([0., 0., 1., 0.], dtype=DTYPE))
        self.assertTrue(result.minimum.hard_case)
        self.assertTrue(result.maximum.hard_case)

    def test_nonzero_linear_hard_case_at_both_ends(self):
        A = torch.diag(torch.tensor([-3., 1., 4.], dtype=DTYPE))
        b = torch.tensor([0., 0.25, 0.], dtype=DTYPE)
        result = sphere_extrema(A, b)
        self.assert_certified(A, b, result)
        minimum = -3.0 - 0.25 ** 2 / 4.0
        maximum = 4.0 + 0.25 ** 2 / 3.0
        self.assertAlmostEqual(result.minimum.value, minimum, places=14)
        self.assertAlmostEqual(result.maximum.value, maximum, places=14)
        self.assertEqual(result.minimum.multiplier, -3.0)
        self.assertEqual(result.maximum.multiplier, 4.0)
        self.assertTrue(result.minimum.hard_case)
        self.assertTrue(result.maximum.hard_case)
        self.assertAlmostEqual(float(result.minimum.vector[1]), -0.25 / 4.0, places=14)
        self.assertAlmostEqual(float(result.maximum.vector[1]), 0.25 / 3.0, places=14)
        self.assertGreater(float(result.minimum.vector[0]), 0.99)
        self.assertGreater(float(result.maximum.vector[2]), 0.99)
        self.assertLessEqual(result.minimum.dual_bound, minimum)
        self.assertGreaterEqual(result.maximum.dual_bound, maximum)

    def test_hard_case_boundary_and_zero_pole_regular_case(self):
        A = torch.diag(torch.tensor([-1., 1.], dtype=DTYPE))
        for coefficient, hard_case, value in ((2., True, -3.), (3., False, -5.)):
            b = torch.tensor([0., coefficient], dtype=DTYPE)
            result = sphere_extrema(A, b)
            self.assert_certified(A, b, result)
            self.assertAlmostEqual(result.minimum.value, value, places=14)
            self.assertEqual(result.minimum.hard_case, hard_case)
            self.assertEqual(result.minimum.secular_iterations, 0 if hard_case else 64)
            torch.testing.assert_close(result.minimum.vector, torch.tensor([0., -1.], dtype=DTYPE))

    def test_pure_linear_objective(self):
        A = torch.zeros((3, 3), dtype=DTYPE)
        b = torch.tensor([2., -3., 6.], dtype=DTYPE)
        result = sphere_extrema(A, b)
        self.assert_certified(A, b, result)
        self.assertAlmostEqual(result.minimum.value, -14.0, places=13)
        self.assertAlmostEqual(result.maximum.value, 14.0, places=13)
        torch.testing.assert_close(result.minimum.vector, -b / 7.0, rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(result.maximum.vector, b / 7.0, rtol=1e-14, atol=1e-14)

    def test_indefinite_two_dimensional_extrema_against_polynomial_roots(self):
        # The half-angle substitution t=tan(theta/2) gives this independent
        # quartic for all stationary points on the two-dimensional sphere.
        a, c, d, p, q = -3.0, 0.7, 1.2, 0.35, -0.6
        coefficients = [c + q, 2 * (d - a - p), -6 * c,
                        -2 * (d - a + p), c - q]
        roots = np.polynomial.polynomial.polyroots(coefficients)
        angles = [2 * math.atan(float(root.real)) for root in roots if abs(root.imag) < 1e-12]
        angles.append(math.pi)
        candidates = []
        for angle in angles:
            x, y = math.cos(angle), math.sin(angle)
            candidates.append(a * x * x + 2 * c * x * y + d * y * y + 2 * p * x + 2 * q * y)
        A = torch.tensor([[a, c], [c, d]], dtype=DTYPE)
        b = torch.tensor([p, q], dtype=DTYPE)
        result = sphere_extrema(A, b)
        self.assert_certified(A, b, result)
        self.assertAlmostEqual(result.minimum.value, min(candidates), delta=2e-13)
        self.assertAlmostEqual(result.maximum.value, max(candidates), delta=2e-13)
        self.assertLessEqual(result.minimum.dual_bound, min(candidates))
        self.assertGreaterEqual(result.maximum.dual_bound, max(candidates))

    def test_random_orthogonal_gauge_transports_unique_extremizers(self):
        generator = torch.Generator().manual_seed(61064)
        matrix = torch.randn((9, 9), dtype=DTYPE, generator=generator)
        A = 0.5 * (matrix + matrix.T)
        b = torch.randn(9, dtype=DTYPE, generator=generator)
        rotation, _ = torch.linalg.qr(torch.randn((9, 9), dtype=DTYPE, generator=generator))
        rotated_A, rotated_b = rotation @ A @ rotation.T, rotation @ b
        original = sphere_extrema(A, b)
        rotated = sphere_extrema(rotated_A, rotated_b)
        self.assert_certified(A, b, original)
        self.assert_certified(rotated_A, rotated_b, rotated)
        for before, after in ((original.minimum, rotated.minimum),
                              (original.maximum, rotated.maximum)):
            self.assertAlmostEqual(before.value, after.value, delta=2e-13)
            self.assertAlmostEqual(before.multiplier, after.multiplier, delta=2e-13)
            torch.testing.assert_close(rotation @ before.vector, after.vector, rtol=2e-12, atol=2e-12)
        repeated = sphere_extrema(A, b)
        torch.testing.assert_close(original.minimum.vector, repeated.minimum.vector, rtol=0, atol=0)
        torch.testing.assert_close(original.maximum.vector, repeated.maximum.vector, rtol=0, atol=0)

    def test_rotated_nonzero_linear_hard_case_retains_global_certificate(self):
        generator = torch.Generator().manual_seed(61064)
        rotation, _ = torch.linalg.qr(torch.randn((4, 4), dtype=DTYPE, generator=generator))
        A = torch.diag(torch.tensor([-3., -3., 1., 4.], dtype=DTYPE))
        b = torch.tensor([0., 0., 0.25, 0.], dtype=DTYPE)
        rotated_A, rotated_b = rotation @ A @ rotation.T, rotation @ b
        result = sphere_extrema(rotated_A, rotated_b)
        self.assert_certified(rotated_A, rotated_b, result)
        expected_minimum = -3.0 - 0.25 ** 2 / 4.0
        expected_maximum = 4.0 + 0.25 ** 2 / 3.0
        self.assertAlmostEqual(result.minimum.value, expected_minimum, delta=2e-13)
        self.assertAlmostEqual(result.maximum.value, expected_maximum, delta=2e-13)
        self.assertLessEqual(result.minimum.dual_bound, expected_minimum)
        self.assertGreaterEqual(result.maximum.dual_bound, expected_maximum)

    def test_near_hard_poles_with_nonzero_complementary_linear_term(self):
        A = torch.diag(torch.tensor([-2., 1., 4.], dtype=DTYPE))
        limiting_radius = math.sqrt(1.0 - (0.3 / 3.0) ** 2 - (0.2 / 6.0) ** 2)
        for pole_coefficient in (1e-12, 1e-16, 1e-20, 1e-280):
            with self.subTest(pole_coefficient=pole_coefficient):
                b = torch.tensor([pole_coefficient, 0.3, -0.2], dtype=DTYPE)
                result = sphere_extrema(A, b)
                self.assert_certified(A, b, result)
                minimum = result.minimum
                self.assertFalse(minimum.hard_case)
                self.assertEqual(minimum.secular_iterations, 64)
                self.assertLess(minimum.normalized_stationarity_residual, 2e-13)
                self.assertAlmostEqual(float(minimum.vector[0]), -limiting_radius, delta=2e-12)
                self.assertAlmostEqual(minimum.secular_pole_offset / pole_coefficient,
                                       1.0 / limiting_radius, delta=2e-11)

    def test_near_boundary_hard_pole_has_an_honest_residual_and_bound(self):
        A = torch.diag(torch.tensor([0., 1.], dtype=DTYPE))
        b = torch.tensor([1e-20, 1.], dtype=DTYPE)
        result = sphere_extrema(A, b)
        self.assert_certified(A, b, result)
        self.assertEqual(result.minimum.secular_iterations, 64)
        self.assertAlmostEqual(result.minimum.value, -1.0, delta=5e-15)
        self.assertLess(result.minimum.normalized_stationarity_residual, 1e-13)
        self.assertLessEqual(result.minimum.dual_bound, -1.0)

    def test_scale_covariance_at_tiny_and_large_nonzero_scales(self):
        A = torch.diag(torch.tensor([-3., 1., 4.], dtype=DTYPE))
        b = torch.tensor([0.05, 0.4, -0.2], dtype=DTYPE)
        baseline = sphere_extrema(A, b)
        for factor in (1e-250, 1e250):
            result = sphere_extrema(A * factor, b * factor)
            self.assert_certified(A * factor, b * factor, result)
            for reference, scaled in ((baseline.minimum, result.minimum),
                                      (baseline.maximum, result.maximum)):
                self.assertGreater(scaled.scale, 0.0)
                self.assertAlmostEqual(scaled.value / factor, reference.value, delta=2e-13)
                self.assertAlmostEqual(scaled.multiplier / factor, reference.multiplier, delta=2e-13)
                torch.testing.assert_close(scaled.vector, reference.vector, rtol=2e-13, atol=2e-13)

    def test_smallest_subnormal_matrix_is_not_reclassified_as_zero(self):
        smallest = math.nextafter(0.0, 1.0)
        A = torch.tensor([[smallest]], dtype=DTYPE)
        b = torch.zeros(1, dtype=DTYPE)
        result = sphere_extrema(A, b)
        self.assertEqual(result.minimum.scale, smallest)
        self.assertEqual(result.minimum.value, smallest)
        self.assertEqual(result.maximum.value, smallest)
        self.assertLessEqual(result.minimum.dual_bound, smallest)
        self.assertGreaterEqual(result.maximum.dual_bound, smallest)
        self.assertTrue(result.minimum.certificate_ok)
        self.assertTrue(result.maximum.certificate_ok)

    def test_invalid_inputs_and_unrepresentable_extrema_fail_explicitly(self):
        eye = torch.eye(2, dtype=DTYPE)
        zero = torch.zeros(2, dtype=DTYPE)
        bad_pairs = (
            (eye.float(), zero),
            (eye, zero.float()),
            (torch.zeros((0, 0), dtype=DTYPE), torch.zeros(0, dtype=DTYPE)),
            (torch.zeros((2, 3), dtype=DTYPE), zero),
            (eye, torch.zeros((2, 1), dtype=DTYPE)),
            (eye, torch.tensor([math.nan, 0.], dtype=DTYPE)),
            (torch.tensor([[1., math.inf], [math.inf, 2.]], dtype=DTYPE), zero),
            (torch.tensor([[1., 1.], [0., 2.]], dtype=DTYPE), zero),
        )
        for A, b in bad_pairs:
            with self.subTest(A=A, b=b), self.assertRaises(ValueError):
                sphere_extrema(A, b)
        with self.assertRaises(FloatingPointError):
            sphere_extrema(torch.tensor([[1e308]], dtype=DTYPE),
                           torch.tensor([1e308], dtype=DTYPE))


if __name__ == "__main__":
    unittest.main()
