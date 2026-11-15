"""SPC-03 precision controlled test against the repository's native LoopedChase."""
import unittest
import numpy as np
from experiments.spectral_state_cost_v0.precision_dynamics import (
    TRAIN_PAIRS, TEST_PAIRS, heldout_scalar_damping, run_precision_sweep,
)

class PrecisionDynamicsTests(unittest.TestCase):
    def test_preregistered_temporal_windows(self):
        self.assertEqual(TRAIN_PAIRS,tuple(range(3,9)))
        self.assertEqual(TEST_PAIRS,tuple(range(9,15)))
        self.assertLess(max(TRAIN_PAIRS),min(TEST_PAIRS))

    def test_exact_scalar_affine_model(self):
        t=np.arange(42)
        x=np.column_stack((4+2*0.43**t, -3+.7*0.43**t))
        got=heldout_scalar_damping(x)
        self.assertAlmostEqual(got['train_lambda'],.43,places=8)
        self.assertLess(got['test_ar1_normalized_error'],.0001)

    def test_rotating_modes_are_rejected(self):
        t=np.arange(42)
        x=np.stack((np.cos(.6*t),np.sin(.6*t)),axis=1)
        got=heldout_scalar_damping(x)
        self.assertGreater(got['test_ar1_normalized_error'],.4)

    def test_native_dtype_replay(self):
        record=run_precision_sweep(seeds=(13012902,13012903),batch=2)
        self.assertEqual(record['total_trajectories'],4)
        self.assertIsNone(record['trained_checkpoint'])
        self.assertTrue(record['shared_initial_weight_tensor'])
        self.assertTrue(record['identical_prompt_tokens'])
        # Containment thresholds intentionally far from the reference values.
        self.assertLess(record['float64_summary']['test_ar1_normalized_error'],.2)
        self.assertLess(record['float64_summary']['delta_norm_at_41'],1e-11)
        self.assertGreater(record['float32_summary']['late_delta_median'],1e-8)
        self.assertGreater(record['float64_summary']['test_cosine_mean'],.98)
        self.assertLess(record['mean_relative_float32_float64_state_error'],1e-5)
        self.assertLess(record['dc32_summary']['dc_masking_identity_gap'],1e-9)


if __name__=='__main__':unittest.main()
