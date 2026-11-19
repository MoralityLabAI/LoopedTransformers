"""Synthetic tests for the LGA-10E retrospective numerical replay."""
import unittest

import numpy as np

from mechinterp.loopscope13e.diagnose import (
    ReplayIntegrityError, assert_array_equal, decompose, paired_deltas,
    verify_metadata, RAW_SHA256,
)


class TestPrecisionDecomposition(unittest.TestCase):
    def test_signed_telescoping_and_parity(self):
        q64_actual = np.linspace(0.01, 0.02, 9)
        q64_probe = q64_actual + np.linspace(-1e-8, 3e-8, 9)
        q32_probe = q64_probe + np.linspace(7e-9, -2e-8, 9)
        q_native = q32_probe + np.linspace(-3e-9, 4e-9, 9)
        components = decompose(q_native, q32_probe, q64_probe, q64_actual)
        self.assertTrue(np.allclose(components["total"], q_native - q64_actual,
                                    atol=1e-18, rtol=0))
        reconstructed = sum(components[key] for key in (
            "left_contraction", "hvp_arithmetic", "probe_substitution"))
        self.assertTrue(np.allclose(paired_deltas(reconstructed),
                                    paired_deltas(components["total"]),
                                    atol=1e-17, rtol=0))

    def test_donor_subtraction_prevents_raw_scalar_proxy(self):
        x = np.array([5e-8] * 9)
        self.assertTrue(np.array_equal(paired_deltas(x), np.zeros(8)))
        self.assertGreater(x[0], 1e-8)

    def test_signed_removal_differs_from_absolute_component_subtraction(self):
        q64a = np.zeros(9)
        q64p = np.ones(9) * -1e-8
        q32p = np.ones(9) * 2e-8
        qnative = np.ones(9) * -2e-8
        qnative[1] = 1e-8
        c = decompose(qnative, q32p, q64p, q64a)
        expected = paired_deltas(c["total"] - c["hvp_arithmetic"])
        self.assertTrue(np.allclose(expected,
                                    paired_deltas(c["left_contraction"]) +
                                    paired_deltas(c["probe_substitution"]),
                                    atol=1e-17, rtol=0))

    def test_reject_incomplete_or_nonfinite(self):
        ok = np.zeros(9)
        for bad in [np.zeros(8), np.r_[np.zeros(8), np.nan],
                    np.r_[np.zeros(8), np.inf]]:
            with self.assertRaises(ReplayIntegrityError):
                decompose(bad, ok, ok, ok)

    def test_integrity_comparison_catches_manipulation(self):
        assert_array_equal(np.zeros(9), np.zeros(9), "same")
        with self.assertRaisesRegex(ReplayIntegrityError, "content mismatch"):
            assert_array_equal(np.array([0.0, 0.0]),
                               np.array([0.0, 1e-8]), "corrupt")
        with self.assertRaisesRegex(ReplayIntegrityError, "wrong shape"):
            assert_array_equal(np.zeros(8), np.zeros(9), "shape")

    def test_source_metadata_fails_closed(self):
        good = {
            "schema": "LGA10D/excluded-source-61067-common-native-parent-v1",
            "excluded_model_seed": 61067,
            "excluded_source_seed": 61067,
            "n_cases": 64,
            "n_excluded_bases": 8,
            "n_goals_per_base": 8,
            "n_untrained_model_initializations": 1,
            "n_new_trained_model_roots": 0,
            "trained_weights_loaded": False,
            "edited_task_outcomes_scored": False,
            "reserved_science_seeds_unopened": [31059, 31064, 31065],
            "raw_sha256": RAW_SHA256,
            "capture": {"schema": "LGA10D/aliased-npz-v1"},
            "cases": [{} for _ in range(64)],
        }
        verify_metadata(good)
        for key, value in (("excluded_model_seed", 61068),
                           ("trained_weights_loaded", True),
                           ("raw_sha256", "0"*64),
                           ("n_cases", 63),
                           ("reserved_science_seeds_unopened", [])):
            changed = dict(good)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ReplayIntegrityError):
                verify_metadata(changed)


if __name__ == "__main__":
    unittest.main()
