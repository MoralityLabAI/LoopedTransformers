"""NRF-07 synthetic and independently trained-seed protocol checks."""
import unittest
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch required for this experiment")
class NRF07Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_generator_disjoint_folds(self):
        from research_gym.analysis.nrf07_independent_seed_train import gaussian_inputs
        a, y = gaussian_inputs(109, 1, 2)
        b, _ = gaussian_inputs(109, 5, 2)
        self.assertEqual(a.shape, (2, 128))
        self.assertEqual(y.shape, (2, 8))
        self.assertTrue(torch.equal(
            a, gaussian_inputs(109, 1, 2)[0]))
        self.assertFalse(torch.equal(a, b))

    def test_registered_cohort_only(self):
        from research_gym.analysis.nrf07_independent_seed_train import gaussian_inputs
        with self.assertRaises(ValueError):
            gaussian_inputs(101, 5, 2)
        with self.assertRaises(ValueError):
            gaussian_inputs(109, 3, 2)

    def test_native_cpu_training_smoke(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from research_gym.analysis.nrf07_independent_seed_train import run
        with TemporaryDirectory() as directory:
            receipt = run(109, 16, Path(directory))
            self.assertEqual(len(receipt["rows"]), 2)
            for row in receipt["rows"]:
                self.assertEqual(len(row["arms"]), 11)
                self.assertGreaterEqual(row["supervised_objective"], 0)
                self.assertGreaterEqual(row["random_median_objective"], 0)
                self.assertEqual(len(row["checkpoint_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
