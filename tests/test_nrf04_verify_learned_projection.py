"""Tests for independent exact learned-observation modular verifier."""
import struct
import unittest
from research_gym.analysis.nrf04_verify_learned_projection import (
    float64_integer_mod, pairing_residues,
)

try:
    import torch
except ImportError:
    torch = None


def bits(x):
    return format(struct.unpack(">Q", struct.pack(">d", x))[0], "016x")


class ExactLearnedGateTests(unittest.TestCase):
    def test_scaled_binary64(self):
        for prime in (65521, 65519):
            for value, numerator, denominator in (
                (1., 1, 1), (.5, 1, 2), (-2., -2, 1),
                (2**-1074, 1, 2**1074),
            ):
                self.assertEqual(
                    float64_integer_mod(bits(value), prime),
                    numerator * pow(denominator, prime-2, prime)
                    * pow(2, 1074, prime) % prime)

    def test_signed_zero(self):
        for prime in (65521, 65519):
            self.assertEqual(float64_integer_mod(bits(0.), prime), 0)
            self.assertEqual(float64_integer_mod(bits(-0.), prime), 0)

    def test_reject_noncanonical_and_infinite(self):
        with self.assertRaises(ValueError):
            float64_integer_mod("abc", 65521)
        with self.assertRaises(ValueError):
            float64_integer_mod(bits(float("inf")), 65521)

    @unittest.skipIf(torch is None, "PyTorch optional neural extra")
    def test_exact_pairing_and_cancellation(self):
        w2 = torch.ones(2, 2, dtype=torch.float32)
        for prime in (65521, 65519):
            self.assertEqual(
                pairing_residues([bits(1.), bits(0.)], w2, prime),
                [pow(2, 1223, prime)] * 2)
            self.assertEqual(
                pairing_residues([bits(1.), bits(-1.)], w2, prime), [0, 0])


if __name__ == "__main__":
    unittest.main()
