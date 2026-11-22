import unittest

import torch

from configs.run_ppt_two_slot import calibrate, fork_features
from ppt_gym.two_slot import TwoSlotPolicy
from ppt_gym.two_slot_fixture import BUNDLES, CHECKS, fixtures, menu, registered_tactic


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
        self.assertEqual(tuple(winner for _, winner in dev), BUNDLES)
        self.assertFalse(training.intersection(case.root for case, _ in dev))
        self.assertTrue(all(registered_tactic(action) for action in BUNDLES))
        self.assertFalse(registered_tactic("all_goals rw [h9]"))
