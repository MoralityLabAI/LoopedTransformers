"""Unit tests for learned gate, exact verifier and physical positive controls."""

from __future__ import annotations

import math
import unittest

try:
    from research_gym.inkling_positive_control.experiment import (
        ARMS, DEPTHS, INTERVENTION_DOSE, evaluate, exact_verifier,
        forward, generate_roots, sigmoid, train_gate,
    )
except ModuleNotFoundError:
    from experiment import (
        ARMS, DEPTHS, INTERVENTION_DOSE, evaluate, exact_verifier,
        forward, generate_roots, sigmoid, train_gate,
    )


class LearnedGatePositiveControl(unittest.TestCase):
    def test_roots_are_frozen_and_independent_of_split(self):
        first = generate_roots(23, "train", 3, 5)
        repeated = generate_roots(23, "train", 3, 5)
        other = generate_roots(23, "development", 3, 5)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, other)
        self.assertEqual(len({group[0].root_id for group in first}), 3)
        for group in first:
            self.assertEqual(sorted(group[0].permutation), list(range(17)))
            for task in group:
                self.assertTrue(exact_verifier(task, task.target))
                self.assertFalse(exact_verifier(task, (task.target + 1) % 17))

    def test_analytic_gate_gradient_matches_finite_difference(self):
        task = generate_roots(73, "train", 1, 3)[0][2]
        theta = 0.45
        state, derivative, _ = forward(task, theta, task.hops, derivative=True)
        epsilon = 0.00001
        up, _, _ = forward(task, theta + epsilon, task.hops)
        down, _, _ = forward(task, theta - epsilon, task.hops)
        numerical = (up[task.target] - down[task.target]) / (2 * epsilon)
        self.assertAlmostEqual(derivative[task.target], numerical, places=6)
        self.assertGreater(state[task.target], 0)

    def test_train_then_intervene_at_equal_l2_dose(self):
        theta, history = train_gate(seed=23, roots=8, epochs=24, learning_rate=0.75)
        self.assertGreater(sigmoid(theta), 0.9)
        self.assertLess(history[-1]["cross_entropy"], history[0]["cross_entropy"])
        result, depth_rows, causal_rows = evaluate(theta, seed=4242, roots=8, dose=INTERVENTION_DOSE)
        self.assertEqual(result["source_roots"], 8)
        self.assertEqual(len(depth_rows), 8 * 8 * len(DEPTHS))
        self.assertEqual(len(causal_rows), 8 * 5 * len(ARMS))
        self.assertEqual(result["depth_accuracy"]["1"]["correct"], 8)
        self.assertEqual(result["depth_accuracy"]["10"]["correct"], 64)
        self.assertEqual(result["intervention_accuracy"]["clean"]["correct"], 40)
        self.assertEqual(result["intervention_accuracy"]["zero"]["correct"], 40)
        self.assertEqual(result["intervention_accuracy"]["active_coordinate"]["correct"], 0)
        self.assertTrue(all(
            math.isclose(row["realized_raw_l2_displacement"], INTERVENTION_DOSE, abs_tol=1e-10)
            for row in causal_rows if row["arm"] not in {"clean", "zero"}
        ))
        self.assertTrue(all(row["split"] == "development" for row in causal_rows))
        self.assertIn("dose_was_selected_on_prior_development_pilot", result["limitations"])

    def test_invalid_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            generate_roots(1, "sealed", 4, 5)
        with self.assertRaises(ValueError):
            generate_roots(1, "train", 0, 5)
        with self.assertRaises(ValueError):
            forward(generate_roots(1, "train", 1, 1)[0][0], 0, 0)
        with self.assertRaises(ValueError):
            evaluate(0, roots=2, depths=(1, 1))


if __name__ == "__main__":
    unittest.main()
