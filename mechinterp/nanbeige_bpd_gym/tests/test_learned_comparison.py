import unittest
import numpy as np
from mechinterp.nanbeige_bpd_gym.compare_learned_fits import operator_comparison


class LearnedPhysicalOperatorTests(unittest.TestCase):
    def test_factor_gauge_and_dense_operator_agree(self):
        rng=np.random.default_rng(19);v=rng.normal(size=(11,3));u=rng.normal(size=(3,7));b=rng.normal(size=(11,3));c=rng.normal(size=(3,7))
        q=np.array([[2.,1.,0.],[0.,.5,1.],[0.,0.,3.]])
        actual=operator_comparison(v,u,b,c);gauged=operator_comparison(v@q,np.linalg.solve(q,u),b,c)
        expected=np.linalg.norm(b@c-v@u)/np.linalg.norm(v@u)
        self.assertAlmostEqual(actual['physical_component_operator_relative_l2'],expected,places=12)
        for name,value in actual.items():self.assertAlmostEqual(gauged[name],value,places=12)


if __name__=='__main__':unittest.main()
