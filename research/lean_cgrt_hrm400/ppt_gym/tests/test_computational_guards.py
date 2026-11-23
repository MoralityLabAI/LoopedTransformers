import unittest

from ppt_gym.computational_guards import GUARD_CHECK, OPEN, PREFIX, cases, geom, mask_at, menu


class ComputationalGuardTests(unittest.TestCase):
    def test_source_semantics_anchors(self):
        self.assertEqual(geom(3, 4), 40)
        self.assertEqual(mask_at(63), 1)
        self.assertEqual(len({mask_at(i) for i in range(63)}), 63)

    def test_pairs_exchange_only_the_discriminating_condition(self):
        roster = list(cases("source", 8))
        for left, right in zip(roster[::2], roster[1::2], strict=True):
            self.assertEqual(left["pair"], right["pair"])
            self.assertEqual(left["test_claims"], right["test_claims"][::-1])
            self.assertEqual(left["kernel_winner"], 1 - right["kernel_winner"])
            self.assertEqual(len(left["reference"]), left["guards"] + 5)
            self.assertEqual(len(right["reference"]), len(left["reference"]))
        self.assertFalse({row["pair"] for row in roster} & {row["pair"] for row in cases("development", 8)})

    def test_same_public_menu_at_either_dependent_witness(self):
        self.assertEqual(menu({"goals": [{"target": {"raw": "∃ n, P n ∧ Q n"}}]}), PREFIX[:1])
        self.assertEqual(menu({"goals": [{"target": {"and": []}}]}), PREFIX[1:])
        self.assertEqual(menu({"goals": [{"target": {"raw": "P ?w"}}]}), OPEN)
        self.assertEqual(menu({"goals": [{"target": {"raw": "MathCan.FiniteHorizonGap.maskAt 33 % 11 = 3"}}]}), (GUARD_CHECK,))
