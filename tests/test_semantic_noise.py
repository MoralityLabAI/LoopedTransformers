"""SUD-02 exact controls and data-processing tests; stdlib only."""
from __future__ import annotations

import itertools
import math
import random
import unittest

from research_gym.analysis.semantic_noise import (
    binary_entropy, evaluate_channel, reversible_echo_control,
)


class TestExactSemanticNoise(unittest.TestCase):
    def test_reversible_order_three_echo(self):
        run = reversible_echo_control(.1)
        self.assertEqual(run["status"], "CONTROL_ONLY")
        values = run["depths"]
        self.assertEqual(len(values), 7)
        dictator_info = 1 - binary_entropy(.1)
        parity_info = 1 - binary_entropy(.18)
        self.assertAlmostEqual(values[0]["source_information_bits"], dictator_info)
        self.assertAlmostEqual(values[1]["source_information_bits"], parity_info)
        self.assertAlmostEqual(values[2]["source_information_bits"], dictator_info)
        for t in range(4):
            self.assertAlmostEqual(values[t]["source_information_bits"],
                                   values[t + 3]["source_information_bits"])
        self.assertAlmostEqual(values[0]["spectral_mass_by_degree"][1], 1)
        self.assertAlmostEqual(values[1]["spectral_mass_by_degree"][2], 1)
        self.assertAlmostEqual(values[2]["spectral_mass_by_degree"][1], 1)

    def test_all_24_four_node_permutations_multiple_depths(self):
        p = .1
        for perm in itertools.permutations(range(4)):
            for t in range(7):
                row = evaluate_channel(perm, t, p)
                self.assertGreaterEqual(row["conditional_entropy_bits"] + 1e-10, binary_entropy(p))
                self.assertGreaterEqual(row["source_bayes_error"] + 1e-10, p)
                self.assertAlmostEqual(sum(row["spectral_mass_by_degree"]), 1)
                self.assertAlmostEqual(row["source_information_bits"] +
                                       row["conditional_entropy_bits"], 1)
                self.assertLessEqual(row["source_information_bits"],
                                     row["fourier_info_upper_bits"] + 1e-10)
                self.assertGreaterEqual(row["source_information_bits"] + 1e-10,
                                        row["fourier_info_lower_bits"])

    def test_random_sixteen_node_graphs_and_head_maps(self):
        rng = random.Random(940211)
        for p in (0, .01, .1, .25, .5):
            for t in (0, 1, 2, 8, 16):
                perm = list(range(16))
                rng.shuffle(perm)
                native_nodes = list(range(16))
                rng.shuffle(native_nodes)
                r = evaluate_channel(perm, t, p, model_nodes_by_y=native_nodes)
                self.assertGreaterEqual(r["native_head"]["additional_conditional_entropy_bits"],
                                        -1e-10)
                self.assertLessEqual(r["native_head"]["mutual_information_bits"],
                                     r["source_information_bits"] + 1e-10)

    def test_perfect_oracle_replay_and_constant_head(self):
        perm = (0, 3, 1, 2)
        for t in range(5):
            successors = list(range(4))
            for _ in range(t):
                successors = [perm[x] for x in successors]
            perfect = evaluate_channel(perm, t, .1, model_nodes_by_y=successors)
            self.assertEqual(perfect["native_head"]["clean_native_cursor_accuracy"], 1)
            self.assertAlmostEqual(
                perfect["native_head"]["accuracy_vs_clean_target"],
                perfect["raw_corrupted_oracle_bit"]["accuracy_vs_clean_target"])
            constant = evaluate_channel(perm, t, .1, model_nodes_by_y=[0] * 4)
            self.assertAlmostEqual(constant["native_head"]["mutual_information_bits"], 0)
            self.assertAlmostEqual(constant["native_head"]["conditional_entropy_bits"], 1)

    def test_bayes_bit_decision_has_optimal_accuracy(self):
        for t in range(7):
            result = evaluate_channel((0, 3, 1, 2), t, .1)
            bayes_acc = result["bayes_optimal_bit"]["accuracy_vs_clean_target"]
            self.assertAlmostEqual(bayes_acc, 1 - result["source_bayes_error"])
            self.assertGreaterEqual(bayes_acc + 1e-10,
                                    result["raw_corrupted_oracle_bit"]["accuracy_vs_clean_target"])

    def test_dimension_reference_is_for_independent_episodes(self):
        result = evaluate_channel((0, 3, 1, 2), 1, .1, ratio=1 / 3)
        self.assertAlmostEqual(result["iid_episode_dimension_reference"] *
                               math.log2(3), result["conditional_entropy_bits"])

    def test_rejects_invalid_graphs_and_noise(self):
        cases = [([0, 0, 1, 2], 0, .1, 0), ([0, 1, 2], 0, .1, 0),
                 ([0, 1, 2, 3], -1, .1, 0), ([0, 1, 2, 3], 0, .6, 0),
                 ([0, 1, 2, 3], 0, .1, 2)]
        for perm, t, p, bit in cases:
            with self.assertRaises(ValueError):
                evaluate_channel(perm, t, p, bit)

    def test_rejects_inaccurate_model_maps(self):
        with self.assertRaises(ValueError):
            evaluate_channel((0, 3, 1, 2), 2, .1, model_nodes_by_y=[0, 1])
        with self.assertRaises(ValueError):
            evaluate_channel((0, 3, 1, 2), 2, .1, model_nodes_by_y=[0, 1, 2, 12])


if __name__ == "__main__":
    unittest.main()
