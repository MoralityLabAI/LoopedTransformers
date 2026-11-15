"""NRF-08: paired random objective and synthetic full-Jacobian regression."""
import math
import unittest

from research_gym.analysis.nrf08_frozen_suffix_intervention import complete_median

class NRF08Tests(unittest.TestCase):
    def test_median_uses_eight_complete_arms(self):
        arms={}
        for i in range(8):
            e,l=(0.,10.) if i<2 else ((0.,0.) if i<4 else (10.,0.))
            arms[f"random_{i}"]={"heldout_mse":e,"local_fiber_leakage":l}
        self.assertEqual(complete_median(arms,1),10.)
        self.assertEqual(complete_median(arms,0),5.)

    def test_missing_arms_and_negative_penalty(self):
        with self.assertRaises(ValueError):
            complete_median({},1)
        arms={f"random_{i}":{"heldout_mse":0.,"local_fiber_leakage":0.}
              for i in range(8)}
        with self.assertRaises(ValueError):
            complete_median(arms,-1)

    def test_jordan_shear_full_jacobian(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch needed for full-Jacobian fixture")
        from research_gym.analysis.nrf08_suffix_transport import transport_components
        torch.set_num_threads(1)
        w1=torch.zeros(128,128,dtype=torch.float64);w1[0,1]=1
        w2=torch.zeros(128,128,dtype=torch.float64);w2[0,0]=1
        b=torch.zeros(128,dtype=torch.float64)
        basis=torch.eye(128,dtype=torch.float64)[[0,2,3,4,5,6,7,8]]
        result=transport_components((w1,b,w2,b),b,basis)
        self.assertAlmostEqual(result["L8"],2/math.sqrt(8),places=11)
        self.assertAlmostEqual(result["L16"],4/math.sqrt(8),places=11)
        self.assertAlmostEqual(result["interference_2inner_over_k"],1.,places=11)

if __name__=="__main__":
    unittest.main()
