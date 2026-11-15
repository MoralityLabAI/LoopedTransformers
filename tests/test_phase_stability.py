"""SPC-02 robust phase probe and memory-frontier CPU tests."""
import itertools
import math
import unittest

import numpy as np

from experiments.spectral_state_cost_v0 import phase_stability as s


class StabilityTests(unittest.TestCase):
    def test_oracle_noiseless_phase_demodulation(self):
        rng = np.random.default_rng(12309)
        for d, m in ((2, 4), (4, 3), (7, 3)):
            a = rng.normal(size=(m, d)) + 1j * rng.normal(size=(m, d))
            offsets = tuple(range(d))
            x = np.fft.ifft(s.blocks_to_coefficients(a))
            measurement = s.alias_blocks_from_samples(
                s.sample_signal(x, d, offsets), m * d, d, offsets)
            expected = a @ s.phase_matrix(d, offsets).T
            np.testing.assert_allclose(measurement, expected, atol=1e-12)
            np.testing.assert_allclose(s.coefficients_to_blocks(s.blocks_to_coefficients(a), d), a)

    def test_analytic_gain_floor(self):
        for d, k in ((3, 1), (5, 2), (6, 2), (7, 2), (8, 1), (8, 4)):
            p = s.critical_support(d, k)
            measured, _ = s.restricted_min_gain(d, tuple(range(p)), p)
            self.assertGreaterEqual(measured + 1e-10, s.certified_gain_floor(d, k))

    def test_singleton_gain_sharp(self):
        for d in range(2, 11):
            numerical, _ = s.restricted_min_gain(d, (0, 1), 2)
            self.assertAlmostEqual(numerical, s.sharp_singleton_gain(d), places=10)

    def test_coprime_arithmetic_steps_preserve_restricted_gain(self):
        d, p = 7, 4
        base, _ = s.restricted_min_gain(d, (0, 1, 2, 3), p)
        for a in (1, 2, 3, 4, 5, 6):
            phases = tuple((2 + a * j) % d for j in range(4))
            observed, _ = s.restricted_min_gain(d, phases, p)
            self.assertAlmostEqual(base, observed, places=10)

    def test_non_coprime_phase_schedule_aliased(self):
        gain, _ = s.restricted_min_gain(6, (0, 2), 2)
        self.assertLess(gain, 1e-12)
        for bad in ((0, 0), (-1,), (6,)):
            with self.assertRaises(ValueError):
                s.phase_matrix(6, bad)

    def test_finite_seven_phase_schedule_has_robustness_gain(self):
        frontier = s.schedule_frontier(7, 4, 2)
        self.assertEqual(len(frontier), math.comb(7, 4))
        best = float(frontier[0]["worst_gain"])
        consecutive, _ = s.restricted_min_gain(7, (0, 1, 2, 3), 4)
        self.assertAlmostEqual(best, 0.6311266657, places=8)
        self.assertAlmostEqual(consecutive, 0.2961051161, places=8)
        self.assertGreater(best / consecutive, 2.1)
        self.assertIn(frontier[0]["offsets"],
                      [list(x) for x in itertools.combinations(range(7), 4)])

    def test_composite_phase_schedule_can_lose_full_spark(self):
        consecutive, _ = s.restricted_min_gain(6, (0, 1, 2, 3), 4)
        defective, support = s.restricted_min_gain(6, (0, 1, 2, 4), 4)
        self.assertGreater(consecutive, 0.5)
        self.assertLess(defective, 1e-12)
        self.assertEqual(len(support), 4)

    def test_brute_force_noisy_sparse_recovery_certified(self):
        m, d, k = 3, 7, 2
        phases = (0, 1, 2, 3)  # Use the analytically guaranteed schedule.
        original = np.zeros((m, d), dtype=np.complex128)
        original[0, [0, 5]] = [2 - 1j, 0.7 + 1.2j]
        original[1, [2, 3]] = [-0.4 + 0.8j, 1 + 0.2j]
        original[2, [3, 6]] = [1.4, 0.3 - 0.9j]
        x = np.fft.ifft(s.blocks_to_coefficients(original))
        clean = s.sample_signal(x, d, phases)
        rng = np.random.default_rng(30021)
        noise = rng.normal(size=clean.shape) + 1j * rng.normal(size=clean.shape)
        noise = noise / np.linalg.norm(noise) * 1e-6
        measured = s.alias_blocks_from_samples(clean + noise, d*m, d, phases)
        recovered, residual = s.robust_recover_blocks(measured, d, phases, k)
        error = np.linalg.norm(recovered - original)
        self.assertLess(error, 1e-4)
        self.assertLessEqual(error, 2 * d * math.sqrt(m) * np.linalg.norm(noise)
                             / s.certified_gain_floor(d, k) + 1e-9)
        self.assertLessEqual(residual, d * math.sqrt(m) * np.linalg.norm(noise) + 1e-9)

    def test_adversarial_pair_ambiguity_and_schedule_repair(self):
        d = 7
        consecutive = (0, 1, 2, 3)
        gain, cols = s.restricted_min_gain(d, consecutive, 4)
        w = s.phase_matrix(d, consecutive)[:, cols]
        _, _, vh = np.linalg.svd(w)
        delta = vh[-1].conj()  # right singular vector, norm 1
        x, other = np.zeros(d, complex), np.zeros(d, complex)
        x[list(cols[:2])] = delta[:2]
        other[list(cols[2:])] = -delta[2:]
        self.assertAlmostEqual(np.linalg.norm(x-other), 1)
        gap0 = np.linalg.norm(s.phase_matrix(d, consecutive) @ (x-other))
        gap1 = np.linalg.norm(s.phase_matrix(d, (2, 4, 5, 6)) @ (x-other))
        self.assertAlmostEqual(gap0, gain, places=10)
        self.assertLess(gap0 / 2, 0.2)  # midpoint is feasible at noise radius .2
        self.assertGreater(gap1 / 2, 0.3)

    def test_periodic_proxy_residual_flags_transients(self):
        period = 21
        rng = np.random.default_rng(329)
        base = rng.normal(size=(period, 3))
        exact = np.tile(base, (3, 1))
        fitted, residual = s.best_fit_periodic(exact, period)
        self.assertLess(residual, 1e-14)
        np.testing.assert_allclose(fitted, base)
        with_drift = exact + np.linspace(0, 1, len(exact))[:, None]
        _, residual = s.best_fit_periodic(with_drift, period)
        self.assertGreater(residual, 0.05)

    def test_real_trace_diagnostic_contract(self):
        n, d, m = 21, 7, 3
        blocks = np.zeros((m, d, 2), dtype=complex)
        blocks[0, [0, 4], 0] = 1, 2
        blocks[1, [2, 5], 1] = 0.5, -0.3
        spec = np.stack([s.blocks_to_coefficients(blocks[:, :, f])
                         for f in range(2)], axis=1)
        period = np.fft.ifft(spec, axis=0)
        trace = np.tile(period, (3, 1))
        stats = s.trace_spectral_diagnostics(trace, n, d, k=2)
        self.assertLess(stats["periodic_proxy_relative_residual"], 1e-12)
        self.assertLess(stats["spectral_tail_energy_fraction_after_k_terms"], 1e-12)
        self.assertEqual(stats["occupied_alias_feature_pairs"], 2)
        self.assertEqual(stats["fraction_occupied_alias_pairs_with_k_or_fewer_modes"], 1.0)
        not_periodic = trace.copy()
        not_periodic[:, 0] += np.linspace(0, 1, len(trace))
        noisy_stats = s.trace_spectral_diagnostics(not_periodic, n, d, k=2)
        self.assertGreater(noisy_stats["periodic_proxy_relative_residual"], 0.1)
        with self.assertRaises(ValueError):
            s.trace_spectral_diagnostics(trace, n, d=6, k=2)

    def test_dc_anchor_cannot_mask_centered_dynamics(self):
        n, d = 21, 7
        depth = 42
        variable = np.linspace(0, 1, depth, dtype=float)
        raw = 1000.0 + variable
        stats = s.trace_spectral_diagnostics(raw, n, d, k=2)
        self.assertLess(stats["periodic_proxy_relative_residual"], 0.001)
        self.assertGreater(stats["centered_periodic_proxy_relative_residual"], 0.1)
        self.assertGreater(stats["temporal_dc_energy_fraction"], 0.999)
        self.assertLess(stats["centered_periodic_proxy_relative_residual"], 1.00001)
        denom = math.sqrt(max(1e-30, 1 - stats["temporal_dc_energy_fraction"]))
        self.assertAlmostEqual(stats["centered_periodic_proxy_relative_residual"],
                               stats["periodic_proxy_relative_residual"] / denom,
                               delta=1e-5)

    def test_benchmark_receipt(self):
        record = s.demonstration()
        self.assertEqual(record["schedules_enumerated"], 35)
        self.assertGreater(record["conditioning_improvement_factor"], 2.1)
        self.assertLess(record["recovered_coefficient_error_l2"], 0.0001)
        self.assertGreater(record["nonperiodic_proxy_relative_residual"], 0)

    def test_insufficient_and_overlarge_search_rejected(self):
        with self.assertRaisesRegex(ValueError, "insufficient"):
            s.robust_recover_blocks(np.zeros((1, 1)), 7, (0,), 2)
        with self.assertRaisesRegex(ValueError, "schedule search"):
            s.schedule_frontier(12, 6, 3)
        with self.assertRaises(ValueError):
            s.alias_blocks_from_samples(np.zeros((2, 4)), 21, 7, (0, 1))


if __name__ == "__main__":
    unittest.main()
