import unittest
import numpy as np
from experiments.nan_xref17.cache import correlations, root_mean
from experiments.nan_xref17.audit import macro, matrix_correlations, routes
from experiments.nan_fit07.pointwise import masks


class Arithmetic(unittest.TestCase):
    def test_world_replication_does_not_reweight_roots(self):
        meta = [{'root': 'a'}] * 3 + [{'root': 'b'}]
        values = np.array([0., 0., 0., 10.])
        self.assertEqual(float(root_mean(values, range(4), meta)), 5.)
        self.assertEqual(float(macro(values, range(4), meta)), 5.)

    def test_small_and_exact_constant_correlations_are_undefined(self):
        z = np.full((5, 128), .1); e = np.ones((5, 8))
        self.assertTrue(np.isnan(correlations(z, e)).all())
        z[:, 0] = np.arange(5); z[:, 1] = -np.arange(5); e[:, 0] = np.arange(5)
        a = correlations(z, e); b = matrix_correlations(z, e)
        np.testing.assert_allclose(a, b, equal_nan=True, atol=1e-12)
        self.assertAlmostEqual(a[0, 0], 1.); self.assertAlmostEqual(a[1, 0], -1.)
        self.assertTrue(np.isnan(correlations(z[:2], e[:2])).all())

    def test_independent_controller_replay(self):
        rng = np.random.default_rng(2026112117); x = rng.normal(size=(6, 10752))
        checkpoint = {'ci__proj_in.weight': rng.normal(size=(32, 10752)) / 100,
            'ci__proj_in.bias': rng.normal(size=32), 'ci__out_proj.weight': rng.normal(size=(8, 32)) / 10,
            'ci__out_proj.bias': rng.normal(size=8)}
        np.testing.assert_allclose(routes(x, checkpoint), masks(x, checkpoint, {'variant': 'state_only', 'n_blocks': 0}), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
