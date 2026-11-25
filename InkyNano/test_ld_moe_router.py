import unittest
import numpy as np
from ld_moe_router import choose, substitution_bound


class LDMoETests(unittest.TestCase):
    def test_exact_demand_fetch(self):
        s = np.arange(1, 129)/128
        r = choose(s, np.zeros(128), resident=range(8), exact=True,
                   delta=np.arange(128)*-100., cost=np.ones(128), penalty=10.)
        self.assertEqual(r.ids, tuple(range(127,119,-1)))
        self.assertEqual(r.ids, r.native_ids)
        self.assertEqual(r.miss_ids, r.ids)
        self.assertAlmostEqual(sum(r.weights), 8.)

    def test_bias_affects_ranking_not_weights(self):
        r=choose([.9,.2,.1], [0,1,0], k=2, scale=2)
        self.assertEqual(r.ids,(1,0))
        np.testing.assert_allclose(r.weights,2*np.array([.2,.9])/1.1)

    def test_strict_margin(self):
        s=np.array([.9,.7,.4,.1])
        r=choose(s,np.zeros(4),k=2,delta=[0,-.14,.14,0])
        self.assertTrue(r.membership_certified)
        self.assertEqual(set(r.ids),{0,1})
        r=choose(s,np.zeros(4),k=2,delta=[0,-.16,.16,0])
        self.assertFalse(r.membership_certified)
        self.assertEqual(set(r.ids),{0,2})

    def test_tie_not_certified(self):
        r=choose([.8,.5,.5],[0,0,0],k=2)
        self.assertFalse(r.membership_certified)

    def test_cost_bias_and_miss(self):
        r=choose([.9,.8,.7],[0,0,0],k=2,resident={1,2},cost=[1,0,0],penalty=.3)
        self.assertEqual(set(r.ids),{1,2})
        self.assertEqual(r.miss_ids,())

    def test_output_bound(self):
        v=substitution_bound([[1,0],[-1,0]],[[0,0],[0,0]],[1,1])
        self.assertEqual(v["actual_l2"],0)
        self.assertEqual(v["triangle_bound"],2)

    def test_invalid(self):
        with self.assertRaises(ValueError): choose([.2,.1],[0,0],k=1,cost=[-1,0])
        with self.assertRaises(ValueError): choose([0,0],[0,0],k=1)


if __name__ == "__main__":
    unittest.main()
