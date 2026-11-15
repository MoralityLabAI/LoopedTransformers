"""Pure-Python exact-rational tests for the NRF-01 recurrent gym."""
import unittest
from fractions import Fraction as Q
from research_gym.core.numerical_range_certificate import (
    Phase, cycle, cycle_gain, cumulative_noise_bound, example_phases,
    iterate_cycles, noise_budget, orthogonal_gate, perturbed_cycle,
    project_box, require_contracting_cycle, sqdist, tail_error_radius_sq,
)


class NumericalRangeCertificateTests(unittest.TestCase):
    def setUp(self):
        self.phases = example_phases()

    def test_disk_bound_and_cycle(self):
        p, q = self.phases
        self.assertEqual(p.q, Q(13, 64))
        self.assertEqual(q.q, Q(101, 400))
        self.assertEqual(p.gain, Q(1) - Q(1, 3) + Q(2, 3) * Q(13, 64))
        self.assertLess(cycle_gain(self.phases), 1)

    def test_centered_jordan_polynomial(self):
        p = self.phases[0]
        self.assertEqual(p.polynomial((Q(2), Q(3))), (Q(5, 8), Q(3, 8)))
        self.assertEqual(orthogonal_gate("rot90", (Q(1), Q(2))), (Q(2), Q(-1)))

    def test_paired_trajectory_contraction(self):
        a, b = (Q(1, 2), Q(-3, 4)), (Q(-2, 3), Q(3, 5))
        initial = sqdist(a, b)
        gamma = cycle_gain(self.phases)
        for n in range(23):
            x = iterate_cycles(self.phases, a, n)
            y = iterate_cycles(self.phases, b, n)
            self.assertLessEqual(sqdist(x, y), gamma**(2 * n) * initial)
            self.assertTrue(all(-1 <= v <= 1 for v in (*x, *y)))

    def test_phase_residual_bound(self):
        x = (Q(1, 3), Q(-2, 5))
        for p in self.phases:
            self.assertLessEqual(p.unregularized_residual_sq(x), 8 * p.eta**2)
            x = p.step(x)

    def test_noise_budget(self):
        noise = (Q(1, 10000), Q(0))
        clean = noisy = (Q(1, 2), Q(-3, 4))
        for n in range(1, 26):
            clean = cycle(self.phases, clean)
            noisy = perturbed_cycle(self.phases, noisy, noise)
            bound = cumulative_noise_bound(self.phases, Q(1, 10000), n)
            self.assertLessEqual(sqdist(clean, noisy), bound**2)
        self.assertLess(noise_budget(self.phases, Q(1, 10000)), Q(1, 1000))

    def test_expansive_phase_compensation(self):
        a = Phase(Q(0), Q(2), (Q(1),), Q(1), "identity")
        b = Phase(Q(0), Q(2), (Q(0),), Q(3, 4), "negate")
        schedule = (a, b)
        self.assertGreater(a.gain, 1)
        self.assertEqual(require_contracting_cycle(schedule), Q(1, 2))
        x, y = (Q(1, 2), Q(-1, 2)), (Q(-1, 4), Q(3, 4))
        self.assertLessEqual(sqdist(cycle(schedule, x), cycle(schedule, y)),
                             Q(1, 4) * sqdist(x, y))

    def test_a_posteriori_cauchy_radius(self):
        start = (Q(1, 3), Q(2, 5))
        gamma = require_contracting_cycle(self.phases)
        for n in range(10):
            x = iterate_cycles(self.phases, start, n)
            for j in range(1, 5):
                y = iterate_cycles(self.phases, x, j)
                self.assertLessEqual(sqdist(x, y),
                                     tail_error_radius_sq(self.phases, start, n))
        self.assertLess(gamma, 1)

    def test_rejections(self):
        with self.assertRaises(ValueError):
            require_contracting_cycle(
                (Phase(Q(0), Q(2), (Q(3, 4),), Q(1, 2), "identity"),))
        with self.assertRaises(ValueError):
            Phase(Q(0), Q(1), (Q(0),), Q(0), "identity")
        with self.assertRaises(ValueError):
            Phase(Q(0), Q(1), (Q(0),), Q(1, 2), "gibberish")
        with self.assertRaises(ValueError):
            cycle_gain(())
        with self.assertRaises(ValueError):
            cumulative_noise_bound(self.phases, Q(0), -1)

    def test_spectrum_only_false_certificate(self):
        p = Phase(Q(0), Q(2), (Q(0), Q(1, 8)), Q(1, 2), "identity")
        self.assertEqual(p.q, Q(1, 8))
        self.assertEqual(p.polynomial((Q(0), Q(1))), (Q(1, 4), Q(0)))
        self.assertEqual(project_box((Q(2), Q(-2))), (Q(1), Q(-1)))


if __name__ == "__main__":
    unittest.main()
