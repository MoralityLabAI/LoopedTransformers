"""Root-level arithmetic and independent equations remain valid on new strata."""
import unittest
import numpy as np
from experiments.nan_xref22.cache import correlations,root_mean
from experiments.nan_xref22.audit import macro,matrix_correlations,routes
from experiments.nan_fit07.pointwise import masks


class Arithmetic(unittest.TestCase):
    def test_repeated_worlds_do_not_reweight_independent_roots(self):
        meta=[{'root':'a'}]*3+[{'root':'b'}];values=np.array([0.,0.,0.,10.])
        self.assertEqual(float(root_mean(values,range(4),meta)),5.)
        self.assertEqual(float(macro(values,range(4),meta)),5.)

    def test_small_and_constant_columns_remain_undefined(self):
        z=np.full((5,128),.1);energy=np.ones((5,8))
        self.assertTrue(np.isnan(correlations(z,energy)).all())
        z[:,0]=np.arange(5);z[:,1]=-np.arange(5);energy[:,0]=np.arange(5)
        np.testing.assert_allclose(correlations(z,energy),matrix_correlations(z,energy),equal_nan=True,atol=1e-12)
        self.assertTrue(np.isnan(correlations(z[:2],energy[:2])).all())

    def test_independent_state_controller_equations(self):
        rng=np.random.default_rng(2026112122);x=rng.normal(size=(6,10752))
        bank={'ci__proj_in.weight':rng.normal(size=(32,10752))/100,'ci__proj_in.bias':rng.normal(size=32),
            'ci__out_proj.weight':rng.normal(size=(8,32))/10,'ci__out_proj.bias':rng.normal(size=8)}
        np.testing.assert_allclose(routes(x,bank),masks(x,bank,{'variant':'state_only','n_blocks':0}),atol=1e-12)


if __name__=='__main__':
    unittest.main()
