"""Offline matching contracts for the pass-3 K development redesign."""

import unittest

import numpy as np

from research_gym.inkling_routing_02.matched_controls import coordinate_matches, matched_vector


class MatchedControlTests(unittest.TestCase):
    def test_coordinate_rank_uses_oriented_loading_and_realized_dose(self):
        bank = np.array([10, -10, 9, 10, 10], dtype=float)
        query = np.array([2, -2, 2, -2, 2], dtype=float)
        dose_with_sign = np.array([-10, 10, -10, -10, -8], dtype=float)
        order, magnitude, loading, joint = coordinate_matches(bank, query, dose_with_sign, target=0)
        self.assertEqual(order.tolist(), [1, 2, 3])
        self.assertEqual(magnitude[1], 0)
        self.assertEqual(loading[1], 0)
        self.assertAlmostEqual(joint[2], 0.10)
        self.assertEqual(loading[3], 2)

    def test_vector_matches_dose_and_projections_without_editing_target(self):
        bank = np.array([2, 3, 4, 5], dtype=float)
        query = np.array([1, 2, 0, 1], dtype=float)
        delta, ratio = matched_vector(bank, query, seed=91, target=0)
        self.assertLess(ratio, 1)
        self.assertEqual(delta[0], 0)
        self.assertAlmostEqual(np.linalg.norm(delta), 2)
        self.assertAlmostEqual(bank @ delta, -4)
        self.assertAlmostEqual(query @ delta, -2)
        self.assertAlmostEqual(np.linalg.norm(bank + delta), np.linalg.norm(bank[1:]))
        np.testing.assert_array_equal(delta, matched_vector(bank, query, seed=91, target=0)[0])

    def test_vector_reports_infeasible_projection_budget(self):
        bank = np.array([4, 1, 1, 1], dtype=float)
        query = np.array([1, 0, 1, -1], dtype=float)
        delta, ratio = matched_vector(bank, query, seed=91, target=0)
        self.assertIsNone(delta)
        self.assertGreater(ratio, 1)
