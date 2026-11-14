"""Exact differential and provenance checks for TFH-01."""
import itertools
import json
from pathlib import Path
import random
import unittest
from tactic_fiber_balance import (
    Fibre, audit, critical_new_positive_edges, exact_count,
    from_compiler_probes, hall_deficit,
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

    def test_pinned_original_fc12_train_51_state_hall_certificate(self):
        fixture_path = (Path(__file__).parent / "data" /
                        "tfh01_fc12_verified_incidence.json")
        data = json.loads(fixture_path.read_text(encoding="utf-8"))
        provenance = data["provenance"]
        self.assertEqual(provenance["original_compiler_results_sha256"],
                         "c32ebf6d341a000c351b3724b817ec72b3284d82450d986835980b0f51aa626d")
        self.assertEqual(provenance["original_train_artifact_id"], 14095030666)
        self.assertEqual(provenance["original_probed_attempts"], 961)
        fibre = Fibre.from_dict(data)
        self.assertEqual(len(fibre.states), 51)
        self.assertEqual(
            {fibre.families[j]: sum(j in row for row in fibre.legal)
             for j in range(len(fibre.families))},
            {"exact_local": 1, "induction_list": 27,
             "rcases_local": 1, "rw_local": 10},
        )
        result = audit(fibre)
        self.assertEqual(result["input_sha256"],
                         "636ac1719600b97735d70eb4cdfcc0dcecc5c2c0d3885409dc3714b6f7b5b98c")
        hall = result["hall_certificate"]
        self.assertEqual(hall["minimum_unfilled_slots"], 26)
        self.assertEqual(hall["witness_families"],
                         ["exact_local", "rcases_local", "rw_local"])
        self.assertEqual(hall["witness_quota"], 38)
        self.assertEqual(len(hall["witness_neighbors"]), 12)
        self.assertEqual(hall["total_variation_lower_bound"],
                         {"numerator": 26, "denominator": 51})
        self.assertEqual(result["maximum_filled_slots"], 25)
        self.assertEqual(len(result["matching"]), 25)

    def test_pinned_fc12_exact_one_edge_repair_priorities(self):
        file = (Path(__file__).parent / "data" /
                "tfh01_fc12_verified_incidence.json")
        fibre = Fibre.from_dict(json.loads(file.read_text(encoding="utf-8")))
        proposal = critical_new_positive_edges(fibre)
        self.assertEqual(proposal["baseline_hall_deficit"], 26)
        self.assertEqual(proposal["maximum_cut_count"], 1)
        self.assertEqual(proposal["structurally_critical_missing_positive_edges"], 117)
        self.assertEqual(proposal["distinct_candidate_states"], 39)
        self.assertEqual(proposal["by_family"], {
            "exact_local": 39, "induction_list": 0,
            "rcases_local": 39, "rw_local": 39,
        })
        self.assertTrue(proposal["candidates_require_new_original_lean_compiler_success"])
        edges = proposal["candidate_edges"]
        self.assertTrue(all(x["hall_deficit_after_one_verified_edge"] == 25 for x in edges))
        # Verify the closed-form max-cut condition against the independent
        # exhaustive subset recalculation, for all 204 missing graph edges.
        for i, sid in enumerate(fibre.states):
            for j, fam in enumerate(fibre.families):
                if j in fibre.legal[i]:
                    continue
                altered = list(fibre.legal)
                altered[i] = frozenset(set(altered[i]) | {j})
                new = Fibre(fibre.states, fibre.families, fibre.quotas, tuple(altered))
                direct = hall_deficit(new)["minimum_unfilled_slots"]
                predicted = 25 if any(
                    row["state_id"] == sid and row["family"] == fam for row in edges
                ) else 26
                self.assertEqual(direct, predicted, (sid, fam))

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
