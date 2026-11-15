"""NRF-05 pure finite-field sanity tests; saved weights are optional."""
import unittest
from research_gym.analysis.nrf04_exact_gate_rank import float32_bits
from research_gym.analysis.nrf05_full_support_modular import (
    PRIME, CROSS_PRIME, product_support, certify_matrix_pair,
)

try:
    import torch
except ImportError:
    torch = None


class FullLinearCollapseTests(unittest.TestCase):
    def test_prime_67108859(self):
        for d in range(2, int(PRIME**.5) + 1):
            self.assertNotEqual(PRIME % d, 0)
        self.assertEqual(PRIME, 2**26 - 5)
        self.assertLess(PRIME*PRIME, 2**53)

    def test_modular_product_requires_all_entries(self):
        a = [[1, 0], [0, 1]]
        b = [[1, 2], [3, 4]]
        result = product_support(a, b)
        self.assertEqual(result["zero_count"], 0)
        self.assertEqual(result["checked_entries"], 4)
        self.assertEqual(result["first_row_four_residues"], [1, 2])
        b[0][1] = 0
        self.assertEqual(product_support(a,b)["zero_count"], 1)

    @unittest.skipIf(torch is None, "torch optional")
    def test_binary32_pair_certificate(self):
        w1 = torch.eye(2, dtype=torch.float32)
        w2 = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float32)
        result = certify_matrix_pair(w1, w2)
        self.assertEqual(result["w1"]["rank"], 2)
        self.assertEqual(result["w2"]["rank"], 2)
        self.assertEqual(result["w2_crossprime"]["rank"], 2)
        self.assertTrue(result["every_nonzero_linear_observation_collapses"])
        self.assertEqual(result["product_support"]["zero_count"], 0)

    @unittest.skipIf(torch is None, "torch optional")
    def test_reducible_counterexample(self):
        eye = torch.eye(2, dtype=torch.float32)
        result = certify_matrix_pair(eye, eye)
        self.assertFalse(result["every_nonzero_linear_observation_collapses"])
        self.assertEqual(result["product_support"]["zero_count"], 2)

    @unittest.skipIf(torch is None, "torch optional")
    def test_singular_weight_rejected_by_certificate(self):
        w1 = torch.tensor([[1., 1.], [1., 1.]], dtype=torch.float32)
        w2 = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float32)
        self.assertFalse(certify_matrix_pair(w1, w2)[
            "every_nonzero_linear_observation_collapses"])

    def test_invalid_modular_shape(self):
        with self.assertRaises(ValueError):
            product_support([[1, 2], [1, 3]], [[1], [2]])


if __name__ == "__main__":
    unittest.main()
