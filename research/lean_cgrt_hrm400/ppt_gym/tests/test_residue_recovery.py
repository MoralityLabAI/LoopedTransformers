import unittest

import torch

from configs.run_ppt_residue_recovery import distribution, joint_uniform_first, projection
from ppt_gym.failed_prefix import WORLDS
from ppt_gym.paired_recovery import STOP
from ppt_gym.residue_recovery import MODULUS, cases, compatibility, route_of, world_weights


class ResidueRecoveryTests(unittest.TestCase):
    def test_held_partitions_are_distinct_exhaustive_and_non_aliasing(self):
        for pair in cases("development", 6):
            first = pair["first"]
            second = pair["second"]
            for case in (first, second):
                sites = [residue for row in case["routes"] for residue in row["residues"]]
                self.assertEqual(sorted(sites), list(range(MODULUS)))
                self.assertEqual(sum(case["expected"]), 1)
            a, b = route_of(first), route_of(second)
            for arm in range(3):
                self.assertGreaterEqual(len({right for left, right in zip(a, b, strict=True) if left == arm}), 2)
            self.assertEqual(sorted(len(route["residues"]) for route in second["routes"]), [2, 3, 4])

    def test_paid_evidence_preserves_multiple_target_routes(self):
        pair = next(cases("source", 1))
        first, second = pair["first"], pair["second"]
        positive = compatibility(first, 0, True)
        negative = compatibility(first, 0, False)
        self.assertEqual(sum(positive), 3)
        self.assertTrue(all(a != b for a, b in zip(positive, negative, strict=True)))
        probabilities = distribution(torch.zeros(MODULUS), torch.tensor(positive))
        self.assertTrue(all(p == 0. for p, ok in zip(probabilities, positive, strict=True) if not ok))
        weights = world_weights(second, probabilities, WORLDS)
        self.assertGreaterEqual(sum(weight > 0 for weight in weights), 2)
        self.assertAlmostEqual(sum(weights), 1.)
        mapped = torch.tensor(probabilities) @ torch.tensor(projection(second))
        torch.testing.assert_close(mapped, torch.tensor(weights))

    def test_joint_first_choice_depends_on_public_costs_and_not_hidden_residue(self):
        pair = next(cases("source", 1))
        a, b = pair["first"], pair["second"]
        choices = []
        for residue in range(MODULUS):
            truth = [residue in route["residues"] for route in a["routes"]]
            trace, posterior = joint_uniform_first({**a, "truth": truth}, b, 5)
            choices.append(trace["trace"][0]["arm"] if trace["trace"] else STOP)
            self.assertGreater(posterior[residue], 0.)
        self.assertEqual(len(set(choices)), 1)
