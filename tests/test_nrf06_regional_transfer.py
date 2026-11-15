"""NRF-06 checks new-fold disjointness, objective comparator and toy pipeline."""
import unittest

try:
    import torch
except ImportError:
    torch = None

from research_gym.analysis.nrf06_regional_transfer import (
    FRESH_EVAL_FOLD, FRESH_SHADOW_FOLD,
    fresh_split, evaluate_transfer, _median,
)

@unittest.skipIf(torch is None, "PyTorch neural extra required")
class RegionalTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.zeros = (torch.zeros(128, 128), torch.zeros(128),
                     torch.zeros(128, 128), torch.zeros(128))

    def test_new_fold_is_disjoint_and_deterministic(self):
        from research_gym.analysis.nrf04_semantic_fiber import array_inputs
        from research_gym.analysis.nrf04_semantic_fiber import blocks
        params = blocks(self.zeros)
        x, h, _ = fresh_split(params, 101, 16, FRESH_EVAL_FOLD, 2)
        y, _, _ = fresh_split(params, 101, 16, FRESH_SHADOW_FOLD, 2)
        self.assertEqual(x.shape, (2, 128))
        self.assertTrue(torch.equal(x, h))
        self.assertFalse(torch.equal(x, y))
        self.assertFalse(torch.equal(x, array_inputs(101, 3, 2)))
        x_again, _, _ = fresh_split(params, 101, 16, FRESH_EVAL_FOLD, 2)
        self.assertTrue(torch.equal(x, x_again))

    def test_bad_fold_rejected(self):
        from research_gym.analysis.nrf04_semantic_fiber import blocks
        params = blocks(self.zeros)
        with self.assertRaises(ValueError):
            fresh_split(params, 101, 16, 3, 2)

    def test_median_over_full_arm_objectives(self):
        self.assertEqual(_median([1,2,3,4,5,6,7,8]), 4.5)
        with self.assertRaises(ValueError):
            _median([1,2,3])

    def test_identity_weight_transfer(self):
        receipt = evaluate_transfer(self.zeros, 101, 16)
        self.assertEqual(len(receipt["arms"]), 11)
        for arm in receipt["arms"].values():
            self.assertAlmostEqual(arm["local_fiber_leakage"], 0., places=12)
        self.assertTrue(receipt["supervised_wins"])

if __name__ == "__main__":
    unittest.main()
