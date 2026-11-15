"""NRF-05 exact piecewise-linear objective crossover regression tests."""
import unittest
from fractions import Fraction as Q
from research_gym.analysis.nrf05_pareto_crossings import (
    first_crossover, loss_line, median,
)


class ParetoCrossingsTests(unittest.TestCase):
    def test_identical_random_arms(self):
        s = {"heldout_mse": .2, "local_fiber_leakage": .5}
        r = {"heldout_mse": .4, "local_fiber_leakage": .3}
        self.assertEqual(first_crossover(s, [r] * 8), Q(5, 4))

    def test_componentwise_medians_are_not_valid_random_objective(self):
        hi = {"heldout_mse": ".1", "local_fiber_leakage": "1"}
        lo = {"heldout_mse": "1", "local_fiber_leakage": ".1"}
        lines = [loss_line(x) for x in [hi] * 4 + [lo] * 4]
        actual = median(lines, Q(1))
        naive = Q(11, 20) + Q(11, 20)**2
        self.assertNotEqual(actual, naive)

    def test_crossing_with_reordered_random_lines(self):
        low_error_high_leak = {"heldout_mse": "0", "local_fiber_leakage": "1"}
        high_error_low_leak = {"heldout_mse": "2", "local_fiber_leakage": "0"}
        controls = [low_error_high_leak] * 4 + [high_error_low_leak] * 4
        supervised = {"heldout_mse": "-.2", "local_fiber_leakage": "1.2"}
        # Negative MSE is explicitly rejected rather than silently accepted.
        with self.assertRaises(ValueError):
            first_crossover(supervised, controls)
        supervised["heldout_mse"] = ".1"
        cross = first_crossover(supervised, controls)
        self.assertIsNotNone(cross)
        self.assertGreater(cross, 0)

    def test_bad_control_count_rejected(self):
        s = {"heldout_mse": 0, "local_fiber_leakage": 0}
        with self.assertRaises(ValueError):
            first_crossover(s, [s] * 7)


if __name__ == "__main__":
    unittest.main()
