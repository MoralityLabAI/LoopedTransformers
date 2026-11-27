"""OLR-01C computational guard: source custody and public-feature unit tests."""
import unittest
from ppt_gym.olr_guards import (
    generate_pairs, read_raw_guards, numeric_features, GuardPair, RAW_SUM,
)


class OLRGuardUnitTests(unittest.TestCase):
    def test_pair_swaps_the_legal_root_action(self):
        p = generate_pairs("train", 20261123, 4, low=0, high=8)
        self.assertEqual(len({x.pair_id for x in p}), 4)
        for item in p:
            a, b = item.world(0), item.world(1)
            self.assertNotEqual(a.statement(), b.statement())
            self.assertEqual(a.reference_action(), "apply r0")
            self.assertEqual(b.reference_action(), "apply r1")
            self.assertEqual(a.targets, b.targets[::-1])
            self.assertEqual(item.true_sum, item.a + item.b)
            self.assertNotEqual(item.true_sum, item.false_sum)

    def test_numeric_features_take_only_uncomputed_public_operands(self):
        self.assertEqual(numeric_features("3 + 4 = 7"), (3/40,4/40,7/80))
        self.assertEqual(numeric_features("(3 + 4 = 7)"), (3/40,4/40,7/80))
        self.assertIsNone(RAW_SUM.fullmatch("True"))
        self.assertIsNone(RAW_SUM.fullmatch("3 * 4 = 12"))
        with self.assertRaises(ValueError):
            numeric_features("3 + 4 = 999")

    def test_native_public_json_required_and_no_reference_feature(self):
        goal = {"target": {"a": "P"}, "hypotheses": [
            {"name":"r0","type":{"imp":[{"raw":"3 + 4 = 7"},{"a":"P"}]}},
            {"name":"r1","type":{"imp":[{"raw":"3 + 4 = 8"},{"a":"P"}]}},
        ]}
        self.assertEqual(read_raw_guards({"goals":[goal]},("apply r0","apply r1")),
                         ("3 + 4 = 7","3 + 4 = 8"))
        goal["hypotheses"][0]["type"]={"imp":[{"true":None},{"a":"P"}]}
        with self.assertRaises(ValueError):
            read_raw_guards({"goals":[goal]},("apply r0","apply r1"))

    def test_generation_fails_for_unbounded_request(self):
        with self.assertRaises(ValueError):
            generate_pairs("train", 1, 0, low=0, high=8)
        with self.assertRaises(ValueError):
            generate_pairs("train", 1, 1, low=0, high=100)
        with self.assertRaises(ValueError):
            GuardPair("train","x",1,1,2,3,1).world(2)


if __name__ == "__main__":
    unittest.main()
