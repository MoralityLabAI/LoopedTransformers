"""Deterministic CPU tests for the SPC-01 spectral-state benchmark."""

import cmath
import math
import random
import unittest

from experiments.spectral_state_cost_v0 import spectral_state_cost as spc


class SpectralStateCostTests(unittest.TestCase):
    def test_fourier_roundtrip(self):
        rng = random.Random(120130)
        for n in (1, 2, 4, 6, 8, 12, 24):
            x = [complex(rng.randrange(-4, 5), rng.randrange(-4, 5))
                 for _ in range(n)]
            self.assertLess(max(abs(a - b) for a, b in zip(x, spc.idft(spc.dft(x)))), 1e-11)

    def test_alias_formula(self):
        x = [complex((t * t + 2 * t) % 7 - 3) for t in range(24)]
        full = spc.dft(x)
        for d in (1, 2, 3, 4, 6, 8, 12):
            for s in range(d):
                observed = spc.dft(spc.sample_phases(x, d, [s])[0])
                for j, value in enumerate(observed):
                    self.assertAlmostEqual(abs(value - spc.alias_formula(full, d, s, j)), 0, delta=1e-9)

    def test_rank_is_exact_phase_budget(self):
        for n, d in ((8, 2), (12, 3), (24, 6)):
            for phases in ([], [0], list(range(min(2, d))), list(range(d))):
                budget = spc.rank_budget(n, d, phases)
                self.assertEqual(spc.matrix_rank(spc.measurement_matrix(n, d, phases)),
                                 budget["observed_rank"])
                self.assertEqual(budget["blind_dimension"] + budget["observed_rank"], n)

    def test_hankel_rank_equals_periodic_spectral_support(self):
        rng = random.Random(1923)
        for m in (2, 3, 4, 6, 8):
            for _ in range(12):
                outputs = [[complex(rng.randrange(-2, 3)) for _ in range(m)]
                           for _ in range(2)]
                self.assertEqual(spc.matrix_rank(spc.block_hankel(outputs)),
                                 spc.spectral_state_order(outputs))

    def test_sparse_recovery_at_sharp_threshold(self):
        rng = random.Random(129130)
        for n, d, k in ((8, 4, 1), (16, 4, 2), (24, 6, 2), (20, 5, 3)):
            m = n // d
            spectrum = [0j] * n
            for j in range(m):
                for r in rng.sample(range(d), k):
                    spectrum[j + r * m] = complex(rng.randrange(1, 4),
                                                 rng.randrange(-3, 4))
            offsets = list(range(min(2 * k, d)))
            obs = spc.sample_phases(spc.idft(spectrum), d, offsets)
            result = spc.sparse_recover(n, d, offsets, obs, k)
            self.assertLess(max(abs(a - b) for a, b in zip(spectrum, result)), 1e-7)

    def test_underbudget_schedule_rejected(self):
        spectrum = [0j] * 8
        spectrum[1] = 1
        obs = spc.sample_phases(spc.idft(spectrum), 4, [0])
        with self.assertRaisesRegex(ValueError, "insufficient phases"):
            spc.sparse_recover(8, 4, [0], obs, 1)

    def test_full_state_can_be_invisible(self):
        n, d = 24, 6
        impulse = [0j] * n
        impulse[5] = 1
        obs = spc.sample_phases(impulse, d, [0, 1, 2, 3])
        self.assertEqual(spc.spectral_state_order([impulse]), n)
        self.assertEqual(spc.spectral_state_order(obs), 0)
        self.assertEqual(spc.rank_budget(n, d, [0, 1, 2, 3])["blind_dimension"], 8)

    def test_demo_metrics(self):
        result = spc.demo()
        self.assertEqual(result["phase_count"], 4)
        self.assertEqual(result["measurement_rank"], 16)
        self.assertEqual(result["measurement_nullity"], 8)
        self.assertEqual(result["true_spectral_state_order"], 5)
        self.assertEqual(result["invisible_true_state_order"], 24)
        self.assertEqual(result["invisible_probed_state_order"], 0)
        self.assertLess(result["max_fourier_recovery_error"], 1e-8)


if __name__ == "__main__":
    unittest.main()
