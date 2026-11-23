import unittest

from configs.run_ppt_paired_recovery import combined_prior, manipulate, pairs
from ppt_gym.failed_prefix import CHARGE, PREFIX_COST, WORLDS, cases
from ppt_gym.paired_recovery import STOP, order, prospect, run_pair, target_prior


class PairedRecoveryTests(unittest.TestCase):
    def test_paid_observation_follows_guard_identity_instead_of_route_number(self):
        source = {"routes": [{"guard": name} for name in ("a", "b", "c")]}
        target = {"routes": [{"guard": name} for name in ("c", "a", "b")]}
        alias = order(source, target)
        adjusted = target_prior([1.] * len(WORLDS), alias, (0, False))
        self.assertEqual(alias, (2, 0, 1))
        self.assertTrue(all(not bits[1] for bits, weight in zip(WORLDS, adjusted, strict=True) if weight))
        source_prior = [float(bits == (True, False, True)) for bits in WORLDS]
        shifted = target_prior(source_prior, alias)
        self.assertEqual(combined_prior(source_prior, shifted, alias), source_prior)

    def test_exact_joint_plan_matches_expected_two_theorem_reward(self):
        left, right = list(cases("source", 1))
        alias = order(left, right)
        weights = [1 / len(WORLDS)] * len(WORLDS)
        budgets = (7, 5)
        joint, immediate = prospect(left, right, budgets, weights)
        self.assertEqual(immediate[STOP], 0.)
        for action, expected in joint.items():
            total = 0.
            for bits, weight in zip(WORLDS, weights, strict=True):
                source = {**left, "truth": bits}
                target = {**right, "truth": [bits[i] for i in alias]}
                total += weight * run_pair(source, target, budgets, weights, action)["utility"]
            self.assertAlmostEqual(total + 2 * PREFIX_COST * CHARGE, expected)

    def test_transfer_dose_and_bystander_rotation_are_matched(self):
        scores = {0: -.02, 1: .05, 2: .0, STOP: .01}
        swapped, dose = manipulate(scores, 0, 1)
        sham, sham_dose = manipulate(scores, 0, 1, sham_sign=-1)
        a = [swapped[key] - value for key, value in scores.items()]
        b = [sham[key] - value for key, value in scores.items()]
        self.assertAlmostEqual(sum(b), 0.)
        self.assertAlmostEqual(sum(x * y for x, y in zip(a, b, strict=True)), 0.)
        self.assertAlmostEqual(sum(x * x for x in b), dose ** 2)
        self.assertEqual(dose, sham_dose)

    def test_pair_filter_leaves_guard_collisions_out_without_reading_truth(self):
        rows = [{**case, "truth": case["expected"]} for case in cases("source", 1)]
        self.assertEqual(list(pairs(rows)), [(0, 1)])
        rows[0]["routes"][2]["guard"] = rows[0]["routes"][0]["guard"]
        self.assertEqual(list(pairs(rows)), [])

    def test_pair_can_stop_when_neither_proof_can_afford_a_route(self):
        left, right = list(cases("source", 1))
        self.assertEqual(prospect(left, right, (1, 1), [1.] * len(WORLDS)), ({STOP: 0.}, {STOP: 0.}))
