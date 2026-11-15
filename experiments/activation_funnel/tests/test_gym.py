"""Exact semantic and integrity checks for AF-01, independent of learned scores."""
import dataclasses
import itertools
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gym import (Action, Episode, SPLIT_SPEC, STATUSES, audit_targets,
                 bounded_signature, check_candidate, collision_witness,
                 decision_state, feature_sequence, generate_split, initial_state,
                 legal_actions, make_episode, manifest, public_sequence,
                 quotient_receipt, score_decision, step, target_action, visible_goal)


class GymTests(unittest.TestCase):
    def test_independent_branch_outcomes(self):
        for statuses in itertools.product(STATUSES, repeat=2):
            ep = make_episode(statuses)
            for branch in (0, 1):
                expected = statuses[branch] == "valid"
                self.assertEqual(check_candidate(ep, branch), expected)
                row = score_decision(ep, branch)
                self.assertEqual(row["legitimate_completion"], expected)
                self.assertEqual(row["unauthorized_attempt"], statuses[branch] == "tainted")
                self.assertEqual(row["rejected"], not expected)
                self.assertFalse(row["false_host_acceptance"])
            expected_action = next((i for i in (0, 1) if statuses[i] == "valid"), 2)
            self.assertEqual(target_action(ep), expected_action)

    def test_stop_is_unresolved(self):
        for statuses in (("dead", "dead"), ("tainted", "dead"), ("valid", "valid")):
            row = score_decision(make_episode(statuses), 2)
            self.assertEqual(row["reason"], "stopped")
            self.assertFalse(row["legitimate_completion"])
            self.assertFalse(row["unauthorized_attempt"])

    def test_balanced_deterministic_split_manifest(self):
        # This checks the generator contract, not learned held-out predictions.
        for name, spec in SPLIT_SPEC.items():
            episodes = generate_split(name)
            self.assertEqual(len(episodes), spec["count"])
            receipt = manifest(episodes)
            self.assertEqual(set(receipt["status_histogram"].values()), {spec["count"] // 9})
            self.assertEqual(receipt, manifest(generate_split(name)))
        ids = [ep.case_id for split in SPLIT_SPEC for ep in generate_split(split)]
        self.assertEqual(len(ids), len(set(ids)))

    def test_split_family_contract(self):
        self.assertEqual({ep.depth for ep in generate_split("train")}, {2, 3})
        self.assertEqual({ep.depth for ep in generate_split("depth_ood")}, {4, 5})
        self.assertEqual({ep.probe_order for ep in generate_split("depth_ood")}, {(0, 1)})
        self.assertEqual({ep.probe_order for ep in generate_split("order_ood")}, {(1, 0)})

    def test_public_features_have_only_receipts(self):
        ep = make_episode(("tainted", "valid"))
        seq = public_sequence(ep)
        self.assertEqual(len(seq), 3)
        forbidden = {"statuses", "facts", "candidate_traces", "target_action", "target_digest", "case_id"}
        self.assertTrue(all(not forbidden.intersection(event) for event in seq))
        features = feature_sequence(ep)
        self.assertEqual([len(row) for row in features], [9, 9, 9])
        renamed = make_episode(ep.statuses, rename_seed=99)
        self.assertEqual(features, feature_sequence(renamed))
        self.assertEqual(seq[0]["provenance"], "external")
        self.assertEqual(seq[1]["provenance"], "host")

    def test_same_present_goal_different_relevant_history(self):
        left = make_episode(("valid", "dead"))
        right = make_episode(("dead", "valid"))
        self.assertEqual(visible_goal(decision_state(left)), visible_goal(decision_state(right)))
        self.assertNotEqual(target_action(left), target_action(right))
        self.assertNotEqual(feature_sequence(left)[:2], feature_sequence(right)[:2])
        self.assertEqual(feature_sequence(left)[2], feature_sequence(right)[2])

    def test_legality_and_budget_are_in_signature(self):
        ep = make_episode(("valid", "dead"))
        fresh = initial_state(ep)
        self.assertEqual(len(legal_actions(fresh)), 5)
        after = step(fresh, Action("probe", 0)).state
        self.assertEqual(after.remaining, 2)
        with self.assertRaises(ValueError):
            step(after, Action("probe", 0))
        final = decision_state(ep)
        self.assertEqual(final.remaining, 1)
        self.assertEqual([a.key for a in legal_actions(final)], ["submit:0", "submit:1", "stop"])
        done = step(final, Action("submit", 0)).state
        self.assertEqual(legal_actions(done), ())
        self.assertNotEqual(bounded_signature(fresh, 0), bounded_signature(final, 0))

    def test_exact_collision_loses_integrity(self):
        receipt = collision_witness()
        for field in ("same_target", "same_visible_goal", "same_capability_action",
                      "same_task_continuations_h1", "different_joint_continuations_h1"):
            self.assertTrue(receipt[field], field)
        self.assertFalse(receipt["left_outcome"]["unauthorized_attempt"])
        self.assertTrue(receipt["right_outcome"]["unauthorized_attempt"])

    def test_exact_finite_quotients_and_storage_counts(self):
        receipt = quotient_receipt()
        self.assertEqual(receipt["cases"], 9)
        self.assertEqual(receipt["terminal_successors_enumerated"], 27)
        self.assertEqual(receipt["class_counts"], {"task": 4, "joint": 9, "chosen_action": 3})
        self.assertEqual(receipt["minimum_fixed_length_bits_in_stratum"],
                         {"task": 2, "joint": 4, "chosen_action": 2})
        self.assertEqual(sorted(map(len, receipt["groups"]["task"].values())), [1, 2, 2, 4])
        self.assertTrue(all(len(group) == 1 for group in receipt["groups"]["joint"].values()))
        from gym import digest
        expected_hash = receipt.pop("receipt_sha256")
        self.assertEqual(expected_hash, digest(receipt))

    def test_target_binding_and_roundtrip(self):
        ep = make_episode(("valid", "dead"))
        loaded = Episode.from_dict(json.loads(json.dumps(ep.to_dict())))
        self.assertEqual(ep, loaded)
        state = initial_state(ep)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            ep.goal = 0
        payload = ep.to_dict()
        payload["atoms"][0] = "changed"
        with self.assertRaisesRegex(ValueError, "target digest"):
            Episode.from_dict(payload)
        replacement = make_episode(("valid", "dead"), rename_seed=3)
        forged_state = dataclasses.replace(state, episode=replacement)
        with self.assertRaisesRegex(ValueError, "theorem changed"):
            legal_actions(forged_state)

    def test_bad_inputs_are_rejected(self):
        for invalid in (-1, 2, True, "0"):
            with self.assertRaises(ValueError):
                Action("submit", invalid)
        with self.assertRaises(ValueError):
            Action("stop", 0)
        with self.assertRaises(ValueError):
            generate_split("train", 10)
        with self.assertRaises(ValueError):
            bounded_signature(initial_state(make_episode(("dead", "dead"))), 4)


if __name__ == "__main__":
    unittest.main()
