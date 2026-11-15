"""Bounded native full-state Jacobian validation, with math SDPA backend."""
import math
import unittest
from experiments.spectral_state_cost_v0.jacobian_monitorability import run_jacobian_sweep

class JacobianMonitorabilityTests(unittest.TestCase):
    def test_native_full_state_secant_and_arnoldi(self):
        r=run_jacobian_sweep(seeds=(13012902,13012903))
        self.assertFalse(r['training_performed'])
        self.assertTrue(r['full_state_includes_all_tokens'])
        self.assertEqual(len(r['rows']),2)
        for item in r['rows']:
            self.assertEqual(item['full_state_dim'],9*64)
            self.assertLess(item['max_forward_gap'],1e-12)
            self.assertLess(item['fixed_point_relative_residual'],1e-11)
            self.assertTrue(.2 < item['estimated_spectral_radius'] < .8)
            self.assertEqual(len(item['leading_eigenvalues']),4)
            self.assertLess(item['secants'][-1]['relative_endpoint_jacobian_error'],1e-4)
            self.assertLess(item['secants'][-1]['relative_endpoint_jacobian_error'],
                            item['secants'][0]['relative_endpoint_jacobian_error'])
            self.assertTrue(all(math.isfinite(e['modulus']) for e in item['leading_eigenvalues']))

if __name__=='__main__':unittest.main()
