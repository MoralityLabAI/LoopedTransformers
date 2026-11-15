"""SPC-03 deterministic center-masking and held-out transient controls."""
import unittest
import numpy as np
from experiments.spectral_state_cost_v0.dynamic_controls import dc_mask_identity, heldout_transient_models, native_sweep


class DynamicControlsTests(unittest.TestCase):
    def test_static_anchor_cloaks_a_nonperiodic_transient(self):
        rng=np.random.default_rng(117)
        z=rng.normal(size=(42,5))
        center=dc_mask_identity(z, 21)
        hidden=dc_mask_identity(z+100000.0,21)
        self.assertAlmostEqual(center['centered_periodic_relative_residual'],
                               hidden['centered_periodic_relative_residual'], places=9)
        self.assertLess(hidden['raw_periodic_relative_residual'],1e-4)
        self.assertGreater(hidden['dc_energy_fraction'],.99999)
        self.assertLess(hidden['dc_masking_identity_gap'],1e-14)

    def test_ar1_ideal_contraction(self):
        t=np.arange(42)
        x=5.+2.*0.5**t
        s=heldout_transient_models(x[:,None],train_fraction=.5)
        # Teacher-forced held-out test eventually approaches machine precision.
        self.assertAlmostEqual(s['scalar_ar1_coefficient'],.5,places=9)
        self.assertLess(s['heldout_scalar_ar1_normalized_error'],.1)

    def test_periodic_control_not_contraction(self):
        t=np.arange(42)
        x=np.column_stack([np.sin(.4*t), np.cos(.4*t)])
        s=heldout_transient_models(x)
        self.assertGreater(s['heldout_scalar_ar1_normalized_error'],.25)
        self.assertGreater(s['difference_decay_ratio_final_over_initial'],.9)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):dc_mask_identity(np.ones(10),5)
        with self.assertRaises(ValueError):heldout_transient_models(np.ones((10,3)))
        with self.assertRaises(ValueError):dc_mask_identity(np.ones((14,2)),0)

    def test_native_16_item_oracle(self):
        x=native_sweep(batch=2,seeds=(13012902,13012903))
        self.assertEqual(len(x['rows']),4)
        self.assertFalse(x['training_performed'])
        self.assertGreater(x['summary_mean']['dc_energy_fraction'], .99)
        self.assertGreater(x['summary_mean']['centered_periodic_relative_residual'], .6)
        self.assertLess(x['summary_mean']['dc_masking_identity_gap'],1e-10)


if __name__=='__main__':unittest.main()
