"""Checks for registered scoring, evidence references, and state-patch controls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.activation_funnel_af02 import evaluate as E, gym
from experiments.activation_funnel_af02.model import make_model, make_codec


def oracle_rows(cases):
    rows = []
    for case in cases:
        action, audit = gym.target_action(case), list(gym.audit_targets(case))
        task_logits = [-1., -1., -1.]
        task_logits[action] = 1.
        audit_logits = [[1. if i == target else -1. for i in range(3)] for target in audit]
        rows.append(E.score_row(case, action, audit, task_logits, audit_logits,
                                "native", "teacher", 17, "clean", gym.feature_sequence(case), "fixture.npz", 0))
    return rows


def gate_fixture():
    result = []
    for architecture in ("gru", "native"):
        for seed in (17, 29, 43):
            splits = {}
            for split in E.SPLITS:
                arms = {}
                for arm in ("teacher", "task", "joint"):
                    arms[arm] = {"correct_decision_rate": 1., "audit_exact_rate": .5 if arm == "task" else 1.,
                                 "unauthorized_attempt_rate": 0., "counterfactuals": {
                                     "addressed_pair": {"changed_pair_first_branch_target_correct": .9,
                                                        "changed_pair_other_branch_retained": .95},
                                     "other_pair": {"changed_pair_first_branch_target_correct": .2,
                                                    "changed_pair_other_branch_retained": .95}}}
                splits[split] = arms
            result.append({"architecture": architecture, "seed": seed, "source_qualified": True, "splits": splits})
    return result


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7102)

    def test_evaluation_inputs_bind_to_frozen_corpus_before_forward(self):
        receipt = json.loads(Path(E.__file__).with_name("corpus_manifest.json").read_text())
        canonical, hosts = {"dev": gym.canonical_cases("dev")}, {"dev": gym.generate_split("dev")}
        E.bind_evaluation_corpus(canonical, hosts, receipt)
        bad = copy.deepcopy(receipt)
        bad["splits"]["dev"]["host_cases_sha256"] = "0" * 64
        bad["manifest_sha256"] = gym.digest({k: v for k, v in bad.items() if k != "manifest_sha256"})
        with self.assertRaisesRegex(ValueError, "corpus differs"):
            E.bind_evaluation_corpus(canonical, hosts, bad)
        with self.assertRaisesRegex(ValueError, "corpus differs"):
            E.bind_evaluation_corpus({"dev": list(reversed(canonical["dev"]))}, hosts, receipt)

    def test_patches_address_atom_and_rule_in_code_coordinates(self):
        for token_width in (4, 16):
            width = 5 * token_width
            left = E.patch_mask(width, 0, "addressed_pair", True)
            right = E.patch_mask(width, 0, "other_pair", True)
            goal = E.patch_mask(width, 0, "goal_only", True)
            self.assertEqual(left.nonzero().flatten().tolist(),
                             list(range(token_width)) + list(range(3 * token_width, 4 * token_width)))
            self.assertTrue(torch.equal(right, E.patch_mask(width, 1, "addressed_pair", True)))
            self.assertFalse(bool((left & right).any()))
            self.assertTrue(bool((left | right | goal).all()))
        self.assertFalse(bool(E.patch_mask(4, 0, "self", False).any()))
        self.assertTrue(bool(E.patch_mask(4, 0, "whole", False).all()))
        with self.assertRaises(ValueError):
            E.patch_mask(4, 0, "addressed_pair", False)

    def test_host_renamings_share_one_scoring_weight(self):
        cases = gym.generate_split("train")
        rows = oracle_rows(cases)
        result = E.metric_summary(rows)
        self.assertEqual((result["host_cases"], result["canonical_inputs"]), (216, 108))
        self.assertEqual(result["positive_inputs"], 60)
        first = cases[0]
        # Both branches are valid in the first registered pattern: branch1
        # completes the theorem while differing from the preferred action.
        self.assertEqual(first.statuses, ("valid", "valid"))
        for i in (0, 1):
            rows[i]["prediction"], rows[i]["branch"] = 1, 1
            rows[i].update(gym.score_decision(cases[i], 1))
        changed = E.metric_summary(rows)
        self.assertAlmostEqual(changed["correct_decision_rate"], 107 / 108)
        self.assertEqual(changed["positive_completion_rate"], 1.)
        rows[1]["prediction"] = 0
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            E.metric_summary(rows)

    def test_failed_or_insensitive_sources_make_preservation_inconclusive(self):
        runs = gate_fixture()
        gates = E.registered_gates(runs)
        self.assertEqual(gates["by_architecture"]["native"]["addressed_causal_preservation"], "pass")
        runs[3]["source_qualified"] = False
        result = E.registered_gates(runs)["per_run"][3]
        self.assertEqual(result["protected_benefit"]["verdict"], "inconclusive")
        self.assertEqual(result["delayed_preservation_overall"], "inconclusive")
        self.assertEqual(result["addressed_causal_preservation"]["verdict"], "inconclusive")
        runs = gate_fixture()
        source = runs[3]["splits"]["depth_context"]["teacher"]["counterfactuals"]
        source["other_pair"]["changed_pair_first_branch_target_correct"] = .8
        self.assertEqual(E.registered_gates(runs)["per_run"][3]["addressed_causal_preservation"]["verdict"], "inconclusive")
        runs[0]["splits"]["delay4"]["teacher"]["audit_exact_rate"] = .8
        self.assertEqual(E.registered_gates(runs)["per_run"][0]["delayed_preservation"]["delay4"]["verdict"], "inconclusive")

    def test_declared_failures_are_retained_in_aggregate(self):
        runs = gate_fixture()
        joint = runs[1]["splits"]["depth_context"]["joint"]
        joint["unauthorized_attempt_rate"] = .1
        gates = E.registered_gates(runs)
        self.assertEqual(gates["by_architecture"]["gru"]["protected_benefit"], "fail")
        self.assertEqual(gates["by_architecture"]["gru"]["delayed_preservation"], "fail")
        self.assertEqual(gates["by_architecture"]["native"]["protected_benefit"], "pass")

    def test_prefix_wait_interventions_retain_actual_code_and_resume(self):
        # Untrained engineering fixture on the development grammar. These
        # predictions are never a held-out scientific result.
        cases = gym.canonical_cases("dev")
        model, codec = make_model("native").eval(), make_codec("native").eval()
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            manifest = []
            rows, stats = E.counterfactuals(model, codec, cases, "native", "joint", 17, out, manifest)
            self.assertEqual(len(rows), 540)
            self.assertTrue(all(row["cut_event"] == 2 for row in rows))
            self.assertEqual(stats["whole"]["direct_hybrid_agreement"], 1.)
            self.assertEqual(stats["self"]["self_replay_agreement"], 1.)
            for record in manifest:
                with np.load(out / record["file"], allow_pickle=False) as arrays:
                    self.assertEqual(arrays["patched_code"].shape[1], 20)
                    self.assertEqual(arrays["initial_decoded"].shape[1], 80)
                    np.testing.assert_array_equal(arrays["patched_code"], np.where(
                        arrays["patch_mask"], arrays["donor_code"], arrays["recipient_code"]))
            permutations = []
            result = E.permutations(model, codec, cases, "native", "joint", 17, out, permutations)
            self.assertEqual(result["layouts"], 12)
            self.assertEqual(len(result["per_input"]), 36)
            self.assertLessEqual(result["max_logit_error"], E.TRANSPORT_ATOL)

    def test_primary_runtime_rejects_unregistered_thread_count(self):
        with self.assertRaisesRegex(ValueError, "two threads"):
            E.evaluate(Path("unused"), Path("unused"), threads=1)


if __name__ == "__main__":
    unittest.main()
