from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from configs.run_ppt_two_slot import SCHEMA, TRM_SCHEMA, calibrate, evaluate, fork_features
from ppt_gym.data import digest, sha256_file
from ppt_gym.two_slot import TwoSlotPolicy, TwoSlotTRM
from ppt_gym.two_slot_fixture import BUNDLES, CHECKS, FAMILY, GUARD, PROBES, fixtures, menu, registered_tactic


def observation(finish):
    hypotheses = [{"name": name, "type": {"raw": "Nat"}} for name in "abcuvw"]
    equalities = ("a = b", "a = c", "b = c", "b = u", "c = u", f"{finish} = v")
    hypotheses.extend({"name": f"h{i}", "type": {"raw": equality}} for i, equality in enumerate(equalities))
    focus, pending = ({"target": {"raw": f"a = {target}"}, "hypotheses": hypotheses} for target in "uv")
    initial = {"target": {"and": [focus["target"], pending["target"]]}, "hypotheses": hypotheses}
    return [initial], [focus, pending]


class TwoSlotTests(unittest.TestCase):
    def test_prefix_carry_changes_fork_scores_and_receives_gradient(self):
        initial, goals = observation("b")
        model = TwoSlotPolicy(width=16, passes=1)
        _, predicted = model(initial, menu(initial, 0), CHECKS)
        carry = model.advance(predicted[0], goals)
        scores, _ = model(goals, menu(goals, 1), CHECKS - 1, carry)
        reset, _ = model(goals, menu(goals, 1), CHECKS - 1)
        self.assertEqual(carry.shape, (2, 16))
        self.assertFalse(torch.allclose(scores, reset))
        (scores[0] - scores[1]).backward()
        self.assertGreater(float(model.write.weight.grad.abs().sum()), 0.)

    def test_calibration_fits_only_the_actor(self):
        records = []
        for number in range(2):
            for finish in ("b", "c"):
                initial, goals = observation(finish)
                events = [{"goals": initial, "menu": ("constructor",)}, {"goals": goals, "menu": BUNDLES}]
                rewards = (.9, -.1) if finish == "b" else (-.1, .9)
                records.append({"case": {"cluster": str(number)}, "trials": [
                    {"events": events, "reward": reward} for reward in rewards]})
        model = TwoSlotPolicy(width=16, passes=1)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        calibrate(model, records)
        self.assertFalse(torch.equal(model.actor.weight, before["actor.weight"]))
        for name, value in model.state_dict().items():
            if not name.startswith("actor."):
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertTrue(torch.isfinite(fork_features(model, records[0])).all())

    def test_registered_pairs_and_bundles(self):
        training = {case.root for case, _ in fixtures("train")}
        dev = list(fixtures("dev"))
        self.assertEqual({case.family for case, _ in dev}, set(PROBES))
        for family, probe in PROBES.items():
            self.assertEqual(tuple(winner for case, winner in dev if case.family == family), probe.fork)
        self.assertTrue(all(case.family == FAMILY for case, _ in fixtures("train")))
        self.assertFalse(training.intersection(case.root for case, _ in dev))
        self.assertTrue(all(registered_tactic(action) for action in BUNDLES))
        self.assertFalse(registered_tactic("all_goals rw [h9]"))

    def test_all_arms_score_the_same_structural_pairs(self):
        repl = SimpleNamespace(version="4.19.0")
        tree = SimpleNamespace(stats=dict)
        learned = TwoSlotPolicy().state_dict()
        calibrated = {**learned, "actor.bias": learned["actor.bias"] + 1}
        trm = TwoSlotTRM().state_dict()
        calls = []

        def score(_repl, _tree, case, *, policy, reset_slots):
            calls.append((case.root, float(policy.actor.bias.detach()), reset_slots))
            return {"success": not (case.family == GUARD and case.tid.endswith("_1"))}

        with TemporaryDirectory() as scratch:
            checkpoint = Path(scratch) / "fit.pt"
            torch.save({"schema": SCHEMA, "lean": repl.version, "learned": learned, "calibrated": calibrated,
                        "training": digest([asdict(case) for case, _ in fixtures("train")])}, checkpoint)
            control = Path(scratch) / "trm.pt"
            torch.save({"schema": TRM_SCHEMA, "parent_sha256": sha256_file(checkpoint), "weights": trm}, control)
            with patch("configs.run_ppt_two_slot.trajectory", side_effect=score):
                report = evaluate(repl, tree, checkpoint, trm_checkpoint=control)

        dev = [case.root for case, _ in fixtures("dev")]
        for arm in report["results"].values():
            self.assertEqual((arm["complete_pairs"], arm["proofs"]), (2, 5))
            self.assertEqual(len(arm["pairs"]), 3)
        for weights, reset in ((learned, False), (calibrated, False), (learned, True), (trm, True)):
            self.assertEqual(calls[:len(dev)], [(root, float(weights["actor.bias"]), reset) for root in dev])
            del calls[:len(dev)]
        self.assertFalse(calls)
        arms = report["results"]
        self.assertAlmostEqual(arms["trm"]["parameters"] / arms["persistent"]["parameters"], 1., delta=.03)
