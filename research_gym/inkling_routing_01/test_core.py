import unittest

import numpy as np

from research_gym.inkling_routing_01.core import (
    ARMS, TARGET_COORD, WIDTH, patch_vector, reconstruct_reader,
    rms_normalize, sham_coordinate, vector_metrics,
)


class ReaderMathTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261122)
        self.b2 = self.rng.normal(size=WIDTH)
        self.b3 = self.rng.normal(size=WIDTH)
        self.b3[TARGET_COORD] = 32.4
        self.sham = sham_coordinate("dev-root-01")

    def test_reader_recovers_all_four_mixture_weights(self):
        a, b = rms_normalize(self.b2), rms_normalize(self.b3)
        betas = [0.12, 0.55, 0.89, 0.97]
        messages = np.asarray([(1 - beta) * a + beta * b for beta in betas])
        fits = reconstruct_reader(self.b2, self.b3, messages)
        for fit, beta in zip(fits, betas):
            self.assertAlmostEqual(fit.beta3, beta, places=10)
            self.assertLess(fit.reconstruction_relative_error, 1e-10)

    def test_reconstruction_rejects_degenerate_sources(self):
        with self.assertRaises(ValueError):
            reconstruct_reader(self.b2, self.b2.copy(), np.ones((4, WIDTH)))

    def test_equal_dose_sham_and_norm_preservation(self):
        zero = patch_vector(self.b3, "zero_2027", self.sham)
        sham = patch_vector(self.b3, "l2_sham", self.sham)
        held = patch_vector(self.b3, "norm_preserving_zero", self.sham)
        self.assertAlmostEqual(np.linalg.norm(zero - self.b3), np.linalg.norm(sham - self.b3), places=10)
        self.assertAlmostEqual(np.linalg.norm(held), np.linalg.norm(self.b3), places=10)
        self.assertEqual(held[TARGET_COORD], 0)
        self.assertAlmostEqual(np.linalg.norm(patch_vector(self.b3, "swap", self.sham)), np.linalg.norm(self.b3))

    def test_reader_rescue_arm_matches_ablation(self):
        self.assertTrue(np.array_equal(patch_vector(self.b3, "zero_2027", self.sham), patch_vector(self.b3, "reader_rescue", self.sham)))

    def test_sham_selection_is_deterministic_and_safe(self):
        self.assertEqual(self.sham, sham_coordinate("dev-root-01"))
        self.assertNotEqual(self.sham, TARGET_COORD)
        self.assertTrue(0 <= self.sham < WIDTH)

    def test_outlier_fraction_and_l2_metrics(self):
        v = np.ones(WIDTH)
        v[TARGET_COORD] = 32.4
        x = vector_metrics(v, patch_vector(v, "zero_2027", 100))
        self.assertGreater(x["coordinate_squared_norm_fraction"], 0.3)
        self.assertAlmostEqual(x["raw_l2_delta"], 32.4)

    def test_nonfinite_and_invalid_arms_are_rejected(self):
        bad = self.b3.copy()
        bad[0] = np.nan
        with self.assertRaises(ValueError):
            patch_vector(bad, ARMS[0], self.sham)
        with self.assertRaises(ValueError):
            patch_vector(self.b3, "clean", self.sham)
        with self.assertRaises(ValueError):
            rms_normalize(np.ones((2, 2)))


if __name__ == "__main__":
    unittest.main()
