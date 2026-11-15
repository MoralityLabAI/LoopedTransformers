"""NRF-04 synthetic fixture tests; no private checkpoint is needed."""
import unittest
import torch
from research_gym.analysis.nrf04_semantic_fiber import (
    TEACHER_PERM8, teacher, array_inputs, evaluate_one, random_basis,
    gaussian_stream, coordinate_basis, leakage_at, blocks,
    supervised_basis, ridge_head, heldout_mse, split_data,
)


class SemanticFiberTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_teacher_uses_original_task_seeds(self):
        for seed in (101, 103, 107):
            perm, signs = teacher(seed)
            self.assertEqual(tuple(perm.tolist()), TEACHER_PERM8[seed])
            self.assertTrue(torch.all(signs.abs() == 1))

    def test_disjoint_reproducible_folds(self):
        x = array_inputs(101, 1, 2)
        y = array_inputs(101, 3, 2)
        self.assertTrue(torch.equal(x, array_inputs(101, 1, 2)))
        self.assertFalse(torch.equal(x, y))
        self.assertTrue(torch.isfinite(x).all())
        self.assertNotEqual(next(gaussian_stream(12)), next(gaussian_stream(13)))

    def test_equal_rank_controls(self):
        for basis in (random_basis(401717), coordinate_basis()):
            self.assertEqual(basis.shape, (8, 128))
            self.assertTrue(torch.allclose(
                basis @ basis.T, torch.eye(8, dtype=torch.float64),
                atol=1e-12))

    def test_identity_has_zero_fiber_leakage(self):
        z = torch.zeros(128, 128, dtype=torch.float32)
        b = torch.zeros(128, dtype=torch.float32)
        params = blocks((z, b, z, b))
        self.assertLess(leakage_at(
            params, torch.ones(128, dtype=torch.float64), 4,
            random_basis(123)), 1e-12)

    def test_teacher_recovery_on_frozen_identity(self):
        z = torch.zeros(128, 128, dtype=torch.float32)
        b = torch.zeros(128, dtype=torch.float32)
        params = blocks((z, b, z, b))
        _, h, y = split_data(params, 101, 16, 1, 160)
        _, eh, ey = split_data(params, 101, 16, 3, 96)
        mh, my = h.mean(0), y.mean(0)
        supervised = supervised_basis(h, y, mh, my)
        random = random_basis(400000 + 17 * 101)
        a = heldout_mse(eh, ey, supervised, mh, my,
                        ridge_head(h, y, supervised, mh, my))
        b = heldout_mse(eh, ey, random, mh, my,
                        ridge_head(h, y, random, mh, my))
        self.assertLess(a, b)

    def test_full_synthetic_pipeline(self):
        g = torch.Generator().manual_seed(7)
        w1 = torch.randn((128, 128), generator=g) * .025
        w2 = torch.randn((128, 128), generator=g) * .025
        b = torch.zeros(128)
        result = evaluate_one((w1, b, w2, b), 101, 16)
        self.assertEqual(len(result), 11)
        self.assertLess(result["supervised"]["heldout_mse"],
                        result["first_eight_coordinates"]["heldout_mse"])
        self.assertTrue(all(row["local_fiber_leakage"] >= 0
                            for row in result.values()))


if __name__ == "__main__":
    unittest.main()
