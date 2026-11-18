"""Independent synthetic integration checks for the frozen LGA-10B matcher.

These tests construct explicit matrices only. They do not load a model or
generate any excluded, scientific, or confirmation source cohort.
"""
import math
import unittest

import torch

from mechinterp.loopscope13b.spectral_match import (
    COSINES,
    bracketed_crossing,
    make_spectral_shams,
    sphere_path,
    stable_seed,
)


DTYPE = torch.float64


def coordinate_geometry(k_dimension=75, row_rank=10):
    """An explicit Euclidean K and its future-response row space."""
    K = torch.zeros((96, 96), dtype=DTYPE)
    K[:k_dimension, :k_dimension] = torch.eye(k_dimension, dtype=DTYPE)
    R = torch.eye(96, dtype=DTYPE)[:, :row_rank].clone()
    z = torch.zeros(96, dtype=DTYPE)
    if row_rank:
        z[0] = 1.
    z[row_rank] = 1.
    return K, R, z


def feasible_problem():
    K, R, z = coordinate_geometry()
    H = torch.zeros((96, 96), dtype=DTYPE)
    H[0, 0], H[10, 10] = .4, .2
    H[0, 10] = H[10, 0] = .07
    H[0, 11] = H[11, 0] = .12
    H[10, 12] = H[12, 10] = -.09
    H[11:75, 11:75] = torch.diag(torch.linspace(-2., 2., 64, dtype=DTYPE))
    return K, R, z, H


class SpectralMatcherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def plan(self, K, R, z, H, **kwargs):
        return make_spectral_shams(
            z, R, lambda x: K @ x, lambda x: H @ x,
            teacher_seed=17, base_index=2, goal=10, **kwargs)

    def assert_physical_constraints(self, plan, K, R, z, H):
        self.assertTrue(plan.eligible, plan.reason)
        z0 = R @ (R.T @ z)
        q0 = z - z0
        radius = float(z.norm())
        null_radius = float(q0.norm())
        target = float(z @ H @ z)
        q_vectors = []
        for sham in plan.shams:
            vector = sham.reshape(-1)
            q = vector - z0
            self.assertLess(abs(float(vector.norm()) - radius), 2e-11)
            self.assertLess(float((K @ vector - vector).norm()), 2e-11)
            self.assertLess(float((R.T @ (vector - z)).norm()), 2e-11)
            self.assertLess(abs(float(q.norm()) - null_radius), 2e-11)
            self.assertLessEqual(
                abs(float(vector @ H @ vector) - target),
                1e-8 * max(1., abs(target)))
            cosine = float(q @ q0) / (null_radius * null_radius)
            self.assertLessEqual(abs(cosine), .85 + 1e-12)
            self.assertLess(min(abs(cosine - c) for c in COSINES), 2e-11)
            for previous in q_vectors:
                pairwise = float(q @ previous) / (null_radius * null_radius)
                self.assertLessEqual(pairwise, .98 + 1e-12)
                self.assertGreaterEqual(float((q - previous).norm()), .2 * null_radius - 1e-11)
            q_vectors.append(q)

    def test_feasible_quadratic_checks_original_physical_matrix(self):
        K, R, z, H = feasible_problem()
        calls = []

        def hvp(vector):
            calls.append(vector.clone())
            return H @ vector

        plan = make_spectral_shams(
            z, R, lambda x: K @ x, hvp,
            teacher_seed=17, base_index=2, goal=10)
        self.assert_physical_constraints(plan, K, R, z, H)
        self.assertEqual(len(plan.shams), 8)
        self.assertEqual(len(calls), 66)
        self.assertEqual(plan.receipt["construction_hvp_count"], 66)
        self.assertEqual(plan.receipt["complement_dimension"], 64)
        self.assertEqual(tuple(row["cosine"] for row in plan.receipt["slices"]), COSINES)
        self.assertFalse(plan.receipt["numerical_certificates_are_interval_proofs"])
        V = plan.raw["complement_basis"]
        for row, vmin, vmax in zip(
                plan.receipt["slices"], plan.raw["min_vectors"], plan.raw["max_vectors"]):
            c = row["cosine"]
            a = plan.raw["z0"] + c * plan.raw["q0"]
            scale = plan.receipt["null_radius"] * math.sqrt(1 - c * c)
            physical_min = a + scale * (V @ vmin)
            physical_max = a + scale * (V @ vmax)
            self.assertAlmostEqual(float(physical_min @ H @ physical_min), row["q_min"], places=11)
            self.assertAlmostEqual(float(physical_max @ H @ physical_max), row["q_max"], places=11)
            self.assertLessEqual(row["outer_lower_bound"], row["q_min"] + 1e-12)
            self.assertGreaterEqual(row["outer_upper_bound"], row["q_max"] - 1e-12)

    def test_seed_replay_keeps_vectors_and_decisions(self):
        problem = feasible_problem()
        first = self.plan(*problem)
        second = self.plan(*problem)
        self.assertTrue(first.eligible, first.reason)
        self.assertTrue(second.eligible, second.reason)
        self.assertEqual(first.receipt["attempts"], second.receipt["attempts"])
        self.assertEqual(first.receipt["slices"], second.receipt["slices"])
        for left, right in zip(first.shams, second.shams):
            torch.testing.assert_close(left, right, rtol=0., atol=0.)
        self.assertEqual(stable_seed("ambient", 17, 2, 10, 0, 0, 0),
                         stable_seed("ambient", 17, 2, 10, 0, 0, 0))
        self.assertNotEqual(stable_seed("ambient", 17, 2, 10, 0, 0, 0),
                            stable_seed("ambient", 17, 2, 10, 1, 0, 0))

    def test_exact_infeasible_separated_band_has_explicit_grid_status(self):
        K, R, z = coordinate_geometry(k_dimension=6, row_rank=1)
        H = torch.zeros((96, 96), dtype=DTYPE)
        H[1, 1] = 1.
        plan = self.plan(K, R, z, H, expected_complement_dim=None)
        self.assertFalse(plan.eligible)
        self.assertEqual(plan.reason, "GRID_NO_ATTAINABLE_TARGET")
        self.assertEqual(len(plan.shams), 0)
        for row in plan.receipt["slices"]:
            self.assertAlmostEqual(row["q_min"], row["cosine"] ** 2, places=12)
            self.assertAlmostEqual(row["q_max"], row["cosine"] ** 2, places=12)
            self.assertLess(row["outer_upper_bound"], row["target"])

    def test_failed_seven_slices_can_have_an_unsampled_feasible_latitude(self):
        K, R, z = coordinate_geometry(k_dimension=6, row_rank=1)
        H = torch.zeros((96, 96), dtype=DTYPE)
        H[0, 1] = H[1, 0] = -.6
        H[1, 1] = 1.
        plan = self.plan(K, R, z, H, expected_complement_dim=None)
        self.assertEqual(plan.reason, "GRID_NO_ATTAINABLE_TARGET")
        witness = torch.zeros(96, dtype=DTYPE)
        witness[0], witness[1], witness[2] = 1., .2, math.sqrt(.96)
        self.assertAlmostEqual(float(witness @ H @ witness), float(z @ H @ z), places=13)
        self.assertAlmostEqual(float(witness.norm()), float(z.norm()), places=13)
        self.assertLess(float((R.T @ (witness - z)).norm()), 1e-13)
        self.assertLess(float((K @ witness - witness).norm()), 1e-13)
        self.assertLess(abs(float((witness - R @ (R.T @ z)) @ (z - R @ (R.T @ z)))), .85)
        self.assertNotIn(.2, COSINES)

    def test_attainable_singleton_cannot_supply_diverse_shams(self):
        K, R, z = coordinate_geometry(k_dimension=6, row_rank=1)
        H = torch.zeros((96, 96), dtype=DTYPE)
        slope = math.sqrt((1 - .85) / (1 + .85))
        H[0, 1] = H[1, 0] = -1.
        H[0, 2] = H[2, 0] = -slope
        plan = self.plan(K, R, z, H, expected_complement_dim=None,
                         n_shams=2, max_attempts=2)
        self.assertFalse(plan.eligible)
        self.assertEqual(plan.reason, "INSUFFICIENT_DISTINCT_SHAMS")
        self.assertEqual(len(plan.shams), 1)
        self.assertTrue(any(row["attainable"] for row in plan.receipt["slices"]))
        self.assertTrue(any(row["reason"] == "PAIRWISE_DIVERSITY_REJECTION"
                            for row in plan.receipt["attempts"]))

    def test_sign_bisection_can_select_a_later_crossing(self):
        A = torch.ones((2, 2), dtype=DTYPE)
        b = -torch.ones(2, dtype=DTYPE)
        start = torch.tensor([math.cos(2.), math.sin(2.)], dtype=DTYPE)
        endpoint = torch.tensor([1., 0.], dtype=DTYPE)
        vector, fraction = bracketed_crossing(start, endpoint, A, b, 0., -.9, 1e-12)
        self.assertIsNotNone(vector)
        self.assertAlmostEqual(float(vector @ A @ vector + 2 * b @ vector), -.9, places=12)
        first_angle = math.pi / 4 + math.acos((1 - math.sqrt(.1)) / math.sqrt(2.))
        first_fraction = (2. - first_angle) / 2.
        first_point = sphere_path(start, endpoint, first_fraction)
        self.assertAlmostEqual(float(first_point @ A @ first_point + 2 * b @ first_point), -.9, places=12)
        self.assertLess(first_fraction, fraction)
        self.assertGreater(fraction, .75)
        self.assertLess(fraction, .85)

    def test_antipodal_path_is_finite_and_stays_on_sphere(self):
        start = torch.tensor([1., 0., 0.], dtype=DTYPE)
        endpoint = -start
        for fraction in (0., .1, .5, .9, 1.):
            point = sphere_path(start, endpoint, fraction)
            self.assertTrue(bool(torch.isfinite(point).all()))
            self.assertAlmostEqual(float(point.norm()), 1., places=14)
        torch.testing.assert_close(sphere_path(start, endpoint, 0.), start, rtol=0., atol=0.)
        torch.testing.assert_close(sphere_path(start, endpoint, 1.), endpoint, rtol=0., atol=0.)

    def test_orthogonal_gauge_preserves_extrema_and_transported_witnesses(self):
        K, R, z, H = feasible_problem()
        first = self.plan(K, R, z, H)
        self.assert_physical_constraints(first, K, R, z, H)
        generator = torch.Generator().manual_seed(90211)
        T, _ = torch.linalg.qr(torch.randn((96, 96), dtype=DTYPE, generator=generator))
        Kt, Rt, zt, Ht = T @ K @ T.T, T @ R, T @ z, T @ H @ T.T
        second = self.plan(Kt, Rt, zt, Ht)
        self.assert_physical_constraints(second, Kt, Rt, zt, Ht)
        for old, new in zip(first.receipt["slices"], second.receipt["slices"]):
            self.assertAlmostEqual(old["q_min"], new["q_min"], places=10)
            self.assertAlmostEqual(old["q_max"], new["q_max"], places=10)
        torch.testing.assert_close(first.raw["projected_hessian_eigenvalues"],
                                   second.raw["projected_hessian_eigenvalues"],
                                   rtol=1e-10, atol=1e-10)
        for sham in first.shams:
            transported = T @ sham.reshape(-1)
            self.assertLess(float((Kt @ transported - transported).norm()), 2e-11)
            self.assertLess(float((Rt.T @ (transported - zt)).norm()), 2e-11)
            self.assertAlmostEqual(float(transported @ Ht @ transported),
                                   float(sham.reshape(-1) @ H @ sham.reshape(-1)), places=11)

    def test_non_symmetric_oracle_fails_before_sampling(self):
        K, R, z, H = feasible_problem()
        H[11, 12] = 1.
        H[12, 11] = 0.
        plan = self.plan(K, R, z, H)
        self.assertFalse(plan.eligible)
        self.assertEqual(plan.reason, "HVP_SYMMETRY_FAILURE")
        self.assertEqual(plan.receipt["attempts"], [])

    def test_degenerate_dark_radius_and_invalid_row_basis_are_explicit(self):
        K, R, z, H = feasible_problem()
        row_only = R[:, 0].clone()
        plan = self.plan(K, R, row_only, H)
        self.assertEqual(plan.reason, "DEGENERATE_FIRST_ORDER_DARK_RADIUS")
        broken = R.clone()
        broken[:, 0] *= 2.
        plan = self.plan(K, broken, z, H)
        self.assertEqual(plan.reason, "INVALID_FIRST_ORDER_GEOMETRY")


if __name__ == "__main__":
    unittest.main()
