"""Exact differential and provenance checks for TFH-01."""
import itertools
import random
import unittest
from tactic_fiber_balance import (
    Fibre, audit, exact_count, from_compiler_probes, hall_deficit,
    legal_pair_swaps, maximum_assignment, published_fc12_capacity,
    sample_uniform, six_cycle_fixture,
)


class TacticFibreTests(unittest.TestCase):
    def test_six_cycle_two_matchings_both_pair_swap_absorbing(self):
        fibre = six_cycle_fixture()
        a = {"row0": "A", "row1": "B", "row2": "C"}
        b = {"row0": "B", "row1": "C", "row2": "A"}
        self.assertEqual(exact_count(fibre), 2)
        self.assertEqual(legal_pair_swaps(fibre, a), [])
        self.assertEqual(legal_pair_swaps(fibre, b), [])
        self.assertEqual({tuple(sample_uniform(fibre, seed).values()) for seed in range(100)},
                         {tuple(a.values()), tuple(b.values())})
        self.assertEqual(audit(fibre)["maximum_filled_slots"], 3)

    def test_512_three_by_three_masks_vs_brute_permutations(self):
        for mask in range(512):
            rows = [{"id": str(i), "allowed": [
                "ABC"[j] for j in range(3) if mask & (1 << (3*i+j))]
            } for i in range(3)]
            fibre = Fibre.from_dict({
                "split": "train", "quotas": {"A": 1, "B": 1, "C": 1},
                "states": rows,
            })
            brute = sum(all(p[i] in rows[i]["allowed"] for i in range(3))
                        for p in itertools.permutations("ABC"))
            self.assertEqual(exact_count(fibre), brute)
            deficit = hall_deficit(fibre)["minimum_unfilled_slots"]
            self.assertEqual(len(maximum_assignment(fibre)), 3 - deficit)
            self.assertEqual(deficit == 0, brute > 0)

    def test_structural_zero_shortfall_and_rejected_sample(self):
        fibre = Fibre.from_dict({"split": "train", "quotas": {"A": 2, "B": 1, "C": 1},
            "states": [
                {"id": "s0", "allowed": ["A"]},
                {"id": "s1", "allowed": ["B"]},
                {"id": "s2", "allowed": ["C"]},
                {"id": "s3", "allowed": []},
            ]})
        self.assertEqual(audit(fibre)["hall_certificate"]["minimum_unfilled_slots"], 1)
        self.assertEqual(exact_count(fibre), 0)
        with self.assertRaises(ValueError):
            sample_uniform(fibre, 17)

    def test_uniform_sample_on_fully_allowed_two_by_two(self):
        fibre = Fibre.from_dict({"split": "train", "quotas": {"A": 1, "B": 1},
            "states": [{"id": "s0", "allowed": ["A", "B"]},
                       {"id": "s1", "allowed": ["A", "B"]}]})
        self.assertEqual(exact_count(fibre), 2)
        tally = {"A": 0, "B": 0}
        for seed in range(500):
            assignment = sample_uniform(fibre, seed)
            tally[assignment["s0"]] += 1
            self.assertNotEqual(assignment["s0"], assignment["s1"])
        self.assertGreater(tally["A"], 180)
        self.assertLess(tally["A"], 320)

    def test_published_bound_is_aggregate_only(self):
        result = published_fc12_capacity()
        self.assertEqual(result["minimum_additional_successfully_labeled_distinct_states"], 24)
        self.assertEqual(result["tv_lower_bound_for_one_pass_assignment"],
                         {"numerator": 24, "denominator": 51})
        self.assertTrue(result["does_not_reconstruct_unreleased_training_incidence"])

    def test_only_real_train_compiler_positives_are_accepted(self):
        row = {"stage_id": "goal", "family": "exact_local", "split": "train",
               "source_kind": "original_lean", "compiler_accepted": True, "progress": True}
        self.assertEqual(exact_count(from_compiler_probes([row], {"exact_local": 1})), 1)
        for key, value in [("source_kind", "synthetic"), ("compiler_accepted", False),
                           ("progress", False)]:
            changed = dict(row, **{key: value})
            self.assertEqual(exact_count(from_compiler_probes([changed],
                                                             {"exact_local": 1})), 0)
        for split in ["heldout", "test", None]:
            with self.assertRaises(ValueError):
                from_compiler_probes([dict(row, split=split)], {"exact_local": 1})
        with self.assertRaises(ValueError):
            Fibre.from_dict({"split": "heldout", "quotas": {"A": 1},
                             "states": [{"id": "s", "allowed": ["A"]}]})

    def test_random_hall_matching_differential(self):
        rng = random.Random(2813)
        for n in range(2, 8):
            for _ in range(24):
                k = min(n, 4)
                quotas = [0] * k
                for _ in range(n):
                    quotas[rng.randrange(k)] += 1
                rows = [{"id": str(i), "allowed": [
                    f"f{j}" for j in range(k) if rng.getrandbits(1)]
                } for i in range(n)]
                fibre = Fibre.from_dict({"split": "train",
                    "quotas": {f"f{j}": q for j, q in enumerate(quotas)}, "states": rows})
                deficit = hall_deficit(fibre)["minimum_unfilled_slots"]
                self.assertEqual(len(maximum_assignment(fibre)), n - deficit)
                if deficit:
                    self.assertEqual(exact_count(fibre), 0)
                else:
                    self.assertGreater(exact_count(fibre), 0)


if __name__ == "__main__":
    unittest.main()
