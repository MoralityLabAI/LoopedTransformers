"""Exact AF-02 corpus, host, address and intervention checks without fitting."""
from collections import Counter, defaultdict
import dataclasses
import hashlib
import itertools
import json
from pathlib import Path
import unittest

from experiments.activation_funnel.certify import HostTheorem, check_branch
from experiments.activation_funnel_af02 import gym


def independent_hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class GymTests(unittest.TestCase):
    def test_registered_corpus_and_canonical_weighting(self):
        protocol_file = Path(gym.__file__).with_name("PROTOCOL.json")
        protocol = json.loads(protocol_file.read_text())
        self.assertEqual(hashlib.sha256(protocol_file.read_bytes()).hexdigest(),
                         gym.REGISTERED_PROTOCOL_SHA256)
        self.assertEqual(list(gym.STATUSES), protocol["data"]["statuses"])
        all_ids, all_targets = [], []
        for split, spec in protocol["data"]["splits"].items():
            hosts = gym.generate_split(split)
            canonical = gym.canonical_cases(split)
            self.assertEqual(len(hosts), spec["hosts"])
            self.assertEqual(len(canonical), spec["unique_inputs"])
            by_input = defaultdict(list)
            for host in hosts:
                by_input[independent_hash(gym.feature_sequence(host))].append(host)
            self.assertEqual(len(by_input), spec["unique_inputs"])
            self.assertEqual({len(rows) for rows in by_input.values()}, {2})
            for rows in by_input.values():
                self.assertEqual([row.rename_index for row in rows], [0, 1])
                self.assertNotEqual(rows[0].target_digest, rows[1].target_digest)
                self.assertEqual(gym.feature_sequence(rows[0]), gym.feature_sequence(rows[1]))
            expected = list(itertools.product(spec["depths"], spec["schedules"],
                                               ((0, 1), (1, 0)), protocol["data"]["statuses"],
                                               protocol["data"]["statuses"]))
            observed = [(c.depth, c.schedule, c.probe_order, *c.statuses) for c in canonical]
            self.assertEqual(observed, expected)
            for index, case in enumerate(canonical):
                self.assertEqual(case.case_id, f"af02_{split}_c{index:04d}_r0")
                self.assertEqual(case.canonical_id, f"af02_{split}_c{index:04d}")
            context_counts = Counter((c.depth, c.schedule, c.probe_order) for c in canonical)
            self.assertEqual(set(context_counts.values()), {9})
            all_ids.extend(case.case_id for case in hosts)
            all_targets.extend(case.target_digest for case in hosts)
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(len(all_targets), len(set(all_targets)))

    def test_pairwise_public_split_disjointness_and_manifest(self):
        sets = {split: {independent_hash(gym.feature_sequence(c)) for c in gym.canonical_cases(split)}
                for split in gym.SPLIT_SPEC}
        for left, right in itertools.combinations(sets, 2):
            self.assertFalse(sets[left] & sets[right], (left, right))
        receipt = gym.corpus_manifest()
        expected_digest = receipt.pop("manifest_sha256")
        self.assertEqual(independent_hash(receipt), expected_digest)
        self.assertTrue(receipt["all_public_input_sets_disjoint"])
        self.assertEqual(set(receipt["pairwise_public_input_intersections"].values()), {0})
        for split, item in receipt["splits"].items():
            self.assertEqual(item["host_cases_sha256"], independent_hash(
                [case.to_dict() for case in gym.generate_split(split)]))
            self.assertEqual(item["canonical_cases_sha256"], independent_hash(
                [case.to_dict() for case in gym.canonical_cases(split)]))
            self.assertEqual(item["copies_per_input"], [2])
        self.assertEqual(gym.corpus_manifest()["manifest_sha256"], expected_digest)

    def test_original_host_oracle_is_independent_of_preview_status(self):
        for statuses in itertools.product(gym.STATUSES, repeat=2):
            for schedule in gym.SCHEDULES:
                case = gym.make_case(statuses, schedule=schedule)
                host = HostTheorem.read(case.to_dict())
                accepted = [check_branch(host, branch).accepted for branch in (0, 1)]
                self.assertEqual(accepted, [status == "valid" for status in statuses])
                self.assertEqual(gym.target_action(case), next((b for b in (0, 1) if accepted[b]), 2))
                for action in (0, 1, 2):
                    verdict = gym.score_decision(case, action)
                    self.assertEqual(verdict["action_cost"], 3 + schedule.count("W"))
                    self.assertEqual(verdict["remaining_budget"], 0)
                    self.assertFalse(verdict["false_host_acceptance"])
                    if action < 2:
                        self.assertEqual(verdict["legitimate_completion"], accepted[action])
                        self.assertEqual(verdict["rejected"], not accepted[action])
                        self.assertEqual(verdict["unauthorized_attempt"], statuses[action] == "tainted")
                    else:
                        self.assertEqual(verdict["reason"], "stopped")
                        self.assertFalse(verdict["legitimate_completion"])
                        self.assertFalse(verdict["unauthorized_attempt"])

    def test_wait_preserves_host_and_leaves_one_decision_action(self):
        for schedule in gym.SCHEDULES:
            for order in gym.ORDERS:
                case = gym.make_case(("valid", "tainted"), schedule=schedule, probe_order=order)
                state = gym.initial_state(case)
                original_target = state.target_digest
                preview_index = 0
                for symbol in schedule[:-1]:
                    action = gym.Action("wait") if symbol == "W" else gym.Action("probe", order[preview_index])
                    if symbol == "P":
                        preview_index += 1
                    before = state
                    state = gym.step(state, action).state
                    self.assertEqual(state.remaining, before.remaining - 1)
                    self.assertEqual(state.target_digest, original_target)
                    self.assertFalse(state.completed)
                    self.assertFalse(state.terminal)
                    if symbol == "W":
                        self.assertEqual(state.case, before.case)
                        self.assertEqual(state.observations[-1].kind, "wait")
                self.assertEqual(state.remaining, 1)
                self.assertEqual([a.key for a in gym.legal_actions(state)], ["submit:0", "submit:1", "stop"])
                with self.assertRaisesRegex(ValueError, "illegal action"):
                    gym.step(state, gym.Action("wait"))
                done = gym.step(state, gym.Action("submit", 0)).state
                self.assertTrue(done.completed)
                self.assertEqual(gym.legal_actions(done), ())

    def test_public_features_and_variable_length_boundary(self):
        forbidden = {"case_id", "canonical_id", "statuses", "facts", "candidate_traces", "schedule",
                     "target_digest", "context_digest", "budget", "target_action"}
        for schedule in gym.SCHEDULES:
            case = gym.make_case(("tainted", "valid"), schedule=schedule, probe_order=(1, 0))
            events = gym.public_sequence(case)
            x = gym.feature_sequence(case)
            self.assertEqual(len(x), len(schedule))
            self.assertTrue(all(len(row) == 9 for row in x))
            self.assertTrue(all(not forbidden.intersection(event) for event in events))
            self.assertEqual(gym.first_probe_cut(case), schedule.index("P") + 1)
            for symbol, row in zip(schedule, x):
                if symbol == "W":
                    self.assertEqual(row, [0.] * 8 + [case.depth / 8.])
            self.assertEqual(x[-1], [0., 1., 0., 0., 0., 0., 0., 0., case.depth / 8.])
            public_audit = {}
            for row in x:
                if row[0] == 1:
                    branch = int(row[3])
                    public_audit[branch] = 0 if row[5] else 2 if row[7] else 1
            self.assertEqual(tuple(public_audit[b] for b in (0, 1)), gym.audit_targets(case))
            renamed = gym.make_case(case.statuses, schedule=schedule, probe_order=case.probe_order,
                                    rename_seed=99, case_id="renamed")
            self.assertEqual(x, gym.feature_sequence(renamed))

    def test_same_goal_requires_history(self):
        left = gym.make_case(("valid", "dead"), schedule="PPWWWWG")
        right = gym.make_case(("dead", "valid"), schedule="PPWWWWG")
        self.assertEqual(gym.visible_goal(gym.decision_state(left)),
                         gym.visible_goal(gym.decision_state(right)))
        self.assertNotEqual(gym.target_action(left), gym.target_action(right))
        self.assertNotEqual(gym.feature_sequence(left), gym.feature_sequence(right))

    def test_case_roundtrip_and_separate_context_integrity(self):
        for split in gym.SPLIT_SPEC:
            case = gym.generate_split(split)[1]
            data = json.loads(json.dumps(case.to_dict()))
            self.assertEqual(gym.Case.from_dict(data), case)
            self.assertEqual(HostTheorem.read(data).target_digest, case.target_digest)
            with self.assertRaises(dataclasses.FrozenInstanceError):
                case.schedule = "PPG"
        case = gym.make_case(("valid", "dead"), schedule="PPWG")
        data = case.to_dict()
        data["statuses"][1] = "tainted"  # same theorem, different protected context
        with self.assertRaisesRegex(ValueError, "context digest"):
            gym.Case.from_dict(data)
        data = case.to_dict()
        data["schedule"] = "PWPG"  # same budget, different history
        with self.assertRaisesRegex(ValueError, "context digest"):
            gym.Case.from_dict(data)
        data = case.to_dict()
        data["atoms"][0] = "changed"
        with self.assertRaisesRegex(ValueError, "target digest"):
            gym.Case.from_dict(data)
        state = gym.initial_state(case)
        changed = dataclasses.replace(case, schedule="PWPG")
        with self.assertRaisesRegex(ValueError, "context changed"):
            gym.legal_actions(dataclasses.replace(state, case=changed))

    def test_donor_prefix_deduplication_composition_and_coverage(self):
        expected = {"train": (324, 216), "dev": (108, 72), "depth_context": (324, 216),
                    "delay2": (108, 72), "delay4": (108, 72)}
        for split, counts in expected.items():
            cases = gym.canonical_cases(split)
            pairs = gym.donor_pairs(cases)
            self.assertEqual((len(pairs), sum(p.changed_prefix_status for p in pairs)), counts)
            self.assertEqual(len({p.pair_id for p in pairs}), len(pairs))
            self.assertEqual(Counter(p.receiver_index for p in pairs), Counter({i: 3 for i in range(len(cases))}))
            self.assertEqual(len({p.composed_input_sha256 for p in pairs}), len(cases))
            prefixes = defaultdict(set)
            for pair in pairs:
                receiver, donor, hybrid = (cases[i] for i in
                                            (pair.receiver_index, pair.donor_index, pair.hybrid_index))
                self.assertEqual(gym.donor_stratum(receiver), gym.donor_stratum(donor))
                cut = receiver.schedule.index("P") + 1
                self.assertEqual(pair.cut_event, cut)
                expected_sequence = gym.feature_sequence(donor)[:cut] + gym.feature_sequence(receiver)[cut:]
                self.assertEqual(expected_sequence, gym.feature_sequence(hybrid))
                expected_status = list(receiver.statuses)
                first = receiver.probe_order[0]
                expected_status[first] = donor.statuses[first]
                self.assertEqual(tuple(expected_status), hybrid.statuses)
                self.assertEqual(pair.audit_target, gym.audit_targets(hybrid))
                self.assertEqual(pair.expected_action, gym.target_action(hybrid))
                self.assertEqual(pair.changed_prefix_status, receiver.statuses[first] != donor.statuses[first])
                prefixes[pair.stratum].add(pair.donor_prefix_sha256)
            self.assertEqual({len(values) for values in prefixes.values()}, {3})
        with self.assertRaisesRegex(ValueError, "unique canonical"):
            gym.donor_pairs(gym.generate_split("dev"))

    def test_binding_companions_hold_payload_fixed(self):
        for split in gym.SPLIT_SPEC:
            cases = gym.canonical_cases(split)
            pairs = gym.binding_pairs(cases)
            self.assertEqual(len(pairs), len(cases) // 2)
            self.assertEqual(sum(p.task_nontrivial for p in pairs), len(cases) * 4 // 18)
            self.assertEqual(sum(p.audit_nontrivial for p in pairs), len(cases) * 6 // 18)
            for pair in pairs:
                left, right = cases[pair.left_index], cases[pair.right_index]
                self.assertEqual(gym.erased_feature_sequence(left), gym.erased_feature_sequence(right))
                self.assertNotEqual(gym.feature_sequence(left), gym.feature_sequence(right))
                self.assertEqual(left.statuses, tuple(reversed(right.statuses)))
                self.assertEqual(pair.expected_left_audit, tuple(reversed(pair.expected_right_audit)))
                if pair.task_nontrivial:
                    self.assertEqual({pair.expected_left_action, pair.expected_right_action}, {0, 1})
                    self.assertTrue(check_branch(HostTheorem.read(left), pair.expected_left_action).accepted)
                    self.assertTrue(check_branch(HostTheorem.read(right), pair.expected_right_action).accepted)

    def test_exact_address_erasure_ceilings(self):
        receipt = gym.address_erasure_receipt()
        self.assertEqual(receipt["cases"], 18)
        self.assertEqual(receipt["erased_input_classes"], 9)
        expected = {"preferred_action": (14, 18), "audit_exact": (12, 18),
                    "exactly_one_valid_action": (4, 8), "binding_pair_both_correct": (0, 4),
                    "audit_pair_both_correct": (0, 6)}
        for name, (correct, count) in expected.items():
            self.assertEqual(receipt["ceilings"][name], {"correct": correct, "total": count, "rate": correct / count})
        self.assertEqual(sum(max(g["task_counts"].values()) for g in receipt["groups"]), 14)
        self.assertEqual(sum(max(g["audit_counts"].values()) for g in receipt["groups"]), 12)
        expected_digest = receipt.pop("receipt_sha256")
        self.assertEqual(independent_hash(receipt), expected_digest)

    def test_all_graph_permutations_use_logical_to_physical_slots(self):
        layouts = gym.graph_permutations()
        self.assertEqual(len(layouts), 12)
        self.assertEqual(layouts[0], gym.GraphPermutation())
        self.assertEqual(len({layout.name for layout in layouts}), 12)
        for layout in layouts:
            atoms, rules = [None] * 3, [None] * 2
            for logical, physical in enumerate(layout.atom_slots):
                atoms[physical] = f"A{logical}"
            for logical, physical in enumerate(layout.rule_slots):
                rules[physical] = f"R{logical}"
            self.assertEqual([atoms[slot] for slot in layout.atom_slots], ["A0", "A1", "A2"])
            self.assertEqual([rules[slot] for slot in layout.rule_slots], ["R0", "R1"])
        with self.assertRaises(ValueError):
            gym.GraphPermutation((0, 0, 2), (0, 1))
        with self.assertRaises(ValueError):
            gym.GraphPermutation([0, 1, 2], (0, 1))

    def test_invalid_actions_and_contexts_fail_explicitly(self):
        for branch in (True, "0", -1, 2):
            with self.assertRaises(ValueError):
                gym.Action("submit", branch)
        with self.assertRaises(ValueError):
            gym.Action("wait", 0)
        with self.assertRaises(ValueError):
            gym.make_case(("valid", "dead"), schedule="PWWWPG")
        with self.assertRaises(ValueError):
            gym.generate_split("order_ood")
        case = gym.make_case(("valid", "dead"), probe_order=(1, 0))
        with self.assertRaisesRegex(ValueError, "illegal action"):
            gym.step(gym.initial_state(case), gym.Action("probe", 0))


if __name__ == "__main__":
    unittest.main()
