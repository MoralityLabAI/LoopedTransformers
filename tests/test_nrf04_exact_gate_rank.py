"""Exact modular-arithmetic tests independent of private checkpoint availability."""
import unittest
from research_gym.analysis.nrf04_exact_gate_rank import (
    PRIMES, float32_bits, dyadic_integer_mod, finite_field_det, certificate,
)

try:
    import torch
except ImportError:
    torch = None


class ExactGateRankTests(unittest.TestCase):
    def test_149_bit_binary32_numerator(self):
        for p in PRIMES:
            for value, numerator, denominator in (
                (1., 1, 1), (.5, 1, 2), (-2., -2, 1),
                (2**-149, 1, 2**149),
            ):
                bits = float32_bits(value)
                self.assertEqual(
                    dyadic_integer_mod(bits, p),
                    numerator * pow(denominator, p - 2, p) * pow(2, 149, p) % p,
                )

    def test_determinant_identity(self):
        bits = [
            [float32_bits(2. if i == j and i == 0
                          else -.5 if i == j and i == 1
                          else 1. if i == j else 0.)
             for j in range(3)] for i in range(3)
        ]
        for p in PRIMES:
            cert = finite_field_det(bits, p)
            self.assertEqual(cert["rank"], 3)
            self.assertEqual(
                cert["determinant_mod_prime"], (-pow(2, 149*3, p)) % p)

    def test_singular_matrix(self):
        bits = [[float32_bits(x) for x in row]
                for row in ((1., 2., 3.), (1., 2., 3.), (0., 0., 1.))]
        for p in PRIMES:
            cert = finite_field_det(bits, p)
            self.assertEqual(cert["rank"], 2)
            self.assertEqual(cert["determinant_mod_prime"], 0)

    @unittest.skipIf(torch is None, "optional PyTorch neural extra")
    def test_coordinate_gate_requires_dense_w2(self):
        w1 = torch.eye(2, dtype=torch.float32)
        w2 = torch.ones(2, 2, dtype=torch.float32)
        self.assertEqual(certificate(w1, w2)["exact_gate_closure_dimension"], 2)
        w2[0, 1] = 0
        self.assertIsNone(certificate(w1, w2)["exact_gate_closure_dimension"])

    def test_nonfinite_rejection(self):
        for p in PRIMES:
            with self.assertRaises(ValueError):
                dyadic_integer_mod(float32_bits(float("inf")), p)


if __name__ == "__main__":
    unittest.main()
