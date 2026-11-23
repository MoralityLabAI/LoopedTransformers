import unittest

from ppt_gym.guard_recovery import initial, menu, transition
from ppt_gym.guard_recovery_learning import baked_policy, source_targets, trace_policy


class GuardRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.case = {"module": "toy-recovery", "truth": [True, False, True, True], "routes": {
            "1_0": {"proof": ["first-good"]}, "1_1": {"proof": ["other-good"]}}}

    def test_fast_rejection_preserves_a_complete_recovery(self):
        chosen = iter(("enter_0", "verify_1", "enter_1", "verify_0", "verify_1", "close"))
        result = trace_policy(self.case, 8, lambda *_: next(chosen))
        self.assertTrue(result["proved"])
        self.assertEqual(result["checks"], 8)
        self.assertEqual(result["proof"], ["first-good"])

    def test_slow_rejection_misses_the_cap(self):
        chosen = iter(("enter_0", "verify_0", "verify_1", "enter_1", "verify_0", "verify_1"))
        result = trace_policy(self.case, 8, lambda *_: next(chosen))
        self.assertFalse(result["proved"])
        self.assertEqual(result["checks"], 8)

    def test_rejected_guard_carries_across_rollback(self):
        state, _ = transition(initial(8), "enter_0", self.case["truth"])
        together, result = transition(state, "verify_1", self.case["truth"])
        self.assertEqual(result, 0)
        self.assertEqual(together.known, (-1, 0, -1, -1))
        self.assertNotIn("enter_0", menu(together))

    def test_planner_and_matched_targets_use_checked_actions(self):
        control = baked_policy([.9, .1, .9, .9])
        result = trace_policy(self.case, 6, lambda state, _: control(state))
        self.assertTrue(result["proved"])
        self.assertEqual(result["checks"], 6)
        examples = source_targets(self.case)
        self.assertTrue(examples)
        self.assertTrue(all(move in menu(state) for state, move, _, _ in examples))
