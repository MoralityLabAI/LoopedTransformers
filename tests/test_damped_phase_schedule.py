import unittest
import numpy as np
from experiments.spectral_state_cost_v0.damped_phase_schedule import (
    damped_phase_matrix, worst_damped_gain, damped_frontier, crossover_schedule,
)

class DampedScheduleTests(unittest.TestCase):
    def test_zero_delay_matches_undamped_matrix(self):
        from experiments.spectral_state_cost_v0.phase_stability import phase_matrix
        self.assertTrue(np.allclose(damped_phase_matrix(7,(0,1,2,4),1.0),
                                    phase_matrix(7,(0,1,2,4)),atol=1e-12))

    def test_delayed_gain_scales_exactly(self):
        for rho in (.25,.435,.8,1.):
            g=worst_damped_gain(7,(0,1,2,4),2,rho,0)
            later=worst_damped_gain(7,(0,1,2,4),2,rho,9)
            self.assertAlmostEqual(later, g*rho**9,delta=1e-10*max(g,1e-30))

    def test_four_phase_optimum_flips_with_damping(self):
        for rho,expected in ((.435,(0,1,2,3)),(1.,(0,1,2,4))):
            frontier=damped_frontier(7,4,2,rho)
            self.assertEqual(len(frontier),35)
            self.assertEqual(frontier[0]['offsets'],list(expected))
        result=crossover_schedule()
        self.assertTrue(.58<result['rho_crossover']<.60)

    def test_consecutive_analytic_bound(self):
        # Lower bound inherited from undamped Vandermonde and row attenuation.
        for rho in (.25,.435,.8,1.):
            q=4
            g0=worst_damped_gain(7,tuple(range(q)),2,1.)
            gain=worst_damped_gain(7,tuple(range(q)),2,rho)
            self.assertGreaterEqual(gain+1e-13, rho**(q-1)*g0)
            self.assertLessEqual(gain, g0+1e-13)

    def test_bad_inputs(self):
        for rho in (0.,-1.,2.,float('nan')):
            with self.assertRaises(ValueError):damped_phase_matrix(7,[0,1],rho)
        with self.assertRaises(ValueError):damped_phase_matrix(7,[0,1],.5,-2)
        with self.assertRaises(ValueError):worst_damped_gain(7,[0,1,2],2,.8)

if __name__=='__main__':unittest.main()
