import unittest
import numpy as np
from mechinterp.nanbeige_bpd_gym.fit_progress import factor_comparison


class PhysicalFactorComparisonTests(unittest.TestCase):
    def test_gauge_invariance_and_dense_operator_reference(self):
        rng=np.random.default_rng(2026112007)
        a=dict(V=rng.normal(size=(13,8)),U=rng.normal(size=(8,11)))
        scale=np.array([.5,2,4,.25,8,1,.5,2]);order=np.arange(7,-1,-1)
        gauge=dict(V=(a['V']*scale)[:,order],U=(a['U']/scale[:,None])[order])
        result=factor_comparison(a,gauge)
        self.assertLess(result['physical_pure_component_operator_relative_Frobenius'],1e-7)
        np.testing.assert_allclose(result['output_principal_cosines'],1,atol=1e-12)
        b=dict(V=rng.normal(size=(13,8)),U=rng.normal(size=(8,11)))
        result=factor_comparison(a,b);left=a['V']@a['U'];right=b['V']@b['U']
        self.assertAlmostEqual(result['physical_pure_component_operator_relative_Frobenius'],np.linalg.norm(left-right)/np.linalg.norm(left),places=12)

    def test_zero_directions_do_not_create_spurious_subspace_matches(self):
        a=dict(V=np.zeros((10,8)),U=np.zeros((8,9)))
        result=factor_comparison(a,a)
        self.assertEqual(result['output_subspace_ranks'],[0,0])
        self.assertEqual(result['output_principal_cosines'],[])
        self.assertIsNone(result['physical_pure_component_operator_cosine'])


if __name__=='__main__':unittest.main()
