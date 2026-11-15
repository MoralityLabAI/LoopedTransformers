"""NRF-02D soft-routing exact-rational regression tests."""
import unittest
from fractions import Fraction as Q
from itertools import product
from research_gym.core.numerical_range_certificate import sqdist
from research_gym.core.soft_numerical_range import SoftJordanMixer

class SoftJordanTests(unittest.TestCase):
    def setUp(self): self.m=SoftJordanMixer()

    def test_exact_rates_and_probability(self):
        self.assertEqual(self.m.expansive.gain,Q(3,2))
        self.assertEqual(self.m.repair.gain,Q(1,4))
        self.assertEqual(self.m.fixed_probability_rate,Q(9,16))
        self.assertEqual(self.m.soft_rate,Q(53,64))
        self.assertEqual(self.m.high_probability((Q(-1),Q(0))),Q(3,16))
        self.assertEqual(self.m.high_probability((Q(1),Q(0))),Q(5,16))

    def test_pairwise_soft_lipschitz_grid(self):
        vals=(Q(-1),Q(-1,2),Q(0),Q(1,2),Q(1))
        states=list(product(vals,repeat=2))
        for x,y in product(states,repeat=2):
            self.assertLessEqual(sqdist(self.m.step(x),self.m.step(y)),
                                 self.m.soft_rate**2*sqdist(x,y))

    def test_pairwise_static_lipschitz_grid(self):
        vals=(Q(-1),Q(0),Q(1))
        states=list(product(vals,repeat=2))
        for x,y in product(states,repeat=2):
            self.assertLessEqual(sqdist(self.m.fixed_probability_step(x),
                                         self.m.fixed_probability_step(y)),
                                 self.m.fixed_probability_rate**2*sqdist(x,y))

    def test_exact_iterated_contraction(self):
        x=(Q(1),Q(-1));y=(Q(-1),Q(1))
        initial=sqdist(x,y)
        for t in range(8):
            a=self.m.iterate(x,t)
            b=self.m.iterate(y,t)
            self.assertLessEqual(sqdist(a,b),self.m.soft_rate**(2*t)*initial)
            self.assertTrue(all(-1<=v<=1 for v in (*a,*b)))

    def test_posterior_fixed_point_radius(self):
        start=(Q(1,2),Q(1,2))
        for t in range(5):
            approx=self.m.iterate(start,t)
            future=self.m.iterate(start,t+4)
            self.assertLessEqual(sqdist(approx,future),
                                 self.m.fixed_point_tail_bound_sq(start,t))

    def test_invalid_state(self):
        with self.assertRaises(ValueError): self.m.step((Q(2),Q(0)))
        with self.assertRaises(ValueError): self.m.iterate((Q(0),Q(0)),-1)

if __name__=="__main__":
    unittest.main()
