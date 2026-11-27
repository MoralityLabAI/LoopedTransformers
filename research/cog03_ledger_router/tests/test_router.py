"""Boundary tests for labels, recurrence interventions and charged proof search."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from research.cog03_ledger_router.environment import (
    available_actions, canonical, check_step, load_tasks, public_observation,
)
from research.cog03_ledger_router.model import LedgerRouter, encode_observation
from research.cog03_ledger_router.run import rollout, train_model


def task_fixture():
    task = {"schema_version": "cogentic.ledger.v1", "root_id": "root-a", "task_id": "task-a",
            "split": "train", "family": "chain", "world": {
                "atoms": [0, 1, 2, 3, 4], "facts": [0], "goal": 3,
                "rules": [{"premises": [0], "conclusion": 1},
                          {"premises": [1], "conclusion": 2},
                          {"premises": [2], "conclusion": 3},
                          {"premises": [0], "conclusion": 4}]},
            "provenance": {"kind": "synthetic_horn_proof_dag"}}
    task["task_digest"] = hashlib.sha256(canonical(task).encode()).hexdigest()
    return task


class RouterBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_checker_charges_rejection_and_duplicate(self):
        world = task_fixture()["world"]
        missing = check_step(world, [0], 2)
        self.assertFalse(missing["accepted"])
        self.assertEqual(missing["next_known"], [0])
        self.assertEqual(missing["verifier_calls"], 1)
        good = check_step(world, [0], 0)
        duplicate = check_step(world, good["next_known"], 0)
        self.assertTrue(duplicate["accepted"])
        self.assertIsNone(duplicate["new_fact"])
        self.assertEqual(duplicate["verifier_calls"], 1)
        self.assertEqual(duplicate["code"], "duplicate_lemma")

    def test_future_rule_cannot_reuse_unproved_premise(self):
        world = task_fixture()["world"]
        for action in [True, -1, 40, 1, 2]:
            outcome = check_step(world, [0], action)
            self.assertFalse(outcome["accepted"])
            self.assertFalse(outcome["solved"])

    def test_known_conclusions_and_current_failures_are_filtered(self):
        world = task_fixture()["world"]
        self.assertEqual(available_actions(world, {0, 1}, {2}), [1, 3])

    def test_metadata_and_teacher_labels_do_not_change_features(self):
        task = task_fixture()
        plain = public_observation(task, [0], [])
        changed = copy.deepcopy(task)
        changed.update(root_id="secret", split="ood_test", future_verdict=[1, 0], labels=[999])
        changed["provenance"] = {"depth": 99, "teacher_path": [0, 1, 2]}
        observed = public_observation(changed, [0], [])
        self.assertEqual(plain, observed)
        a, am = encode_observation(plain, 24, 24)
        b, bm = encode_observation(observed, 24, 24)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(am, bm))

    def test_public_history_uses_explicit_allowlist(self):
        history = [{"action": 1, "accepted": False, "code": "missing_premise",
                    "new_fact": None, "future_solution": "private", "labels": [1]}]
        result = public_observation(task_fixture(), [0], history)
        self.assertEqual(set(result["history"][0]), {"action", "accepted", "code", "new_fact"})

    def test_renamed_atoms_preserve_encoder(self):
        task = task_fixture()
        changed = copy.deepcopy(task)
        perm = {0: 4, 1: 2, 2: 0, 3: 1, 4: 3}
        world = changed["world"]
        world["facts"] = [perm[x] for x in world["facts"]]
        world["goal"] = perm[world["goal"]]
        for rule in world["rules"]:
            rule["premises"] = [perm[x] for x in rule["premises"]]
            rule["conclusion"] = perm[rule["conclusion"]]
        a, am = encode_observation(public_observation(task, [0], []), 24, 24)
        b, bm = encode_observation(public_observation(changed, [perm[0]], []), 24, 24)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(am, bm))

    def test_candidate_permutation_is_equivariant(self):
        torch.manual_seed(8)
        model = LedgerRouter().eval()
        task = task_fixture()
        features, mask = encode_observation(public_observation(task, [0], []), 24, 24)
        permutation = [3, 1, 0, 2]
        changed = copy.deepcopy(task)
        changed["world"]["rules"] = [task["world"]["rules"][i] for i in permutation]
        other, other_mask = encode_observation(public_observation(changed, [0], []), 24, 24)
        with torch.no_grad():
            first = model(features, mask)["logits"][permutation]
            second = model(other, other_mask)["logits"]
        self.assertTrue(torch.allclose(first, second, atol=1e-6))

    def test_reset_inner_retains_four_calls_but_matches_one_pass_output(self):
        model = LedgerRouter().eval()
        features, mask = encode_observation(public_observation(task_fixture(), [0], []), 24, 24)
        calls = []
        handle = model.recurrent.register_forward_hook(lambda *_: calls.append(1))
        with torch.no_grad():
            reset = model(features, mask, depth=4, reset_inner=True)["logits"]
            self.assertEqual(len(calls), 4)
            single = model(features, mask, depth=1)["logits"]
        handle.remove()
        self.assertTrue(torch.allclose(reset, single, atol=1e-7))

    def test_budget_and_full_history_replay(self):
        task = task_fixture()
        rows, episode = rollout(task, arm="goal_ancestry", seed=17, model=None, budget=2)
        self.assertEqual(len(rows), 2)
        self.assertFalse(episode["solved"])
        self.assertEqual(episode["final_known"], [0, 1, 2])
        self.assertEqual([r["step"] for r in rows], [0, 1])
        self.assertEqual(len(rows[1]["observation"]["history"]), 1)
        self.assertEqual(sum(r["verifier_calls"] for r in rows), episode["verifier_calls"])

    def test_solved_episode_stops_and_new_root_clears_model_history(self):
        model = LedgerRouter().eval()
        task = task_fixture()
        a, ea = rollout(task, arm="looped", seed=17, model=model, budget=8)
        b, eb = rollout(task, arm="looped", seed=17, model=model, budget=8)
        self.assertEqual(a, b)
        self.assertEqual(ea["final_known"], eb["final_known"])
        solved, outcome = rollout(task, arm="goal_ancestry", seed=17, model=None, budget=8)
        self.assertEqual(len(solved), 3)
        self.assertTrue(outcome["solved"])

    def test_training_rejects_heldout_roots(self):
        task = task_fixture()
        task["split"] = "ordinary_test"
        with self.assertRaisesRegex(ValueError, "train roots"):
            train_model([task], seed=17, depth=4, updates=1, budget=8)

    def test_loader_rejects_tampered_task(self):
        task = task_fixture()
        task["world"]["goal"] = 4
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.jsonl"
            path.write_text(json.dumps(task) + "\n")
            with self.assertRaisesRegex(ValueError, "digest"):
                load_tasks(path)


if __name__ == "__main__":
    unittest.main()
