import unittest

from ppt_gym.computational_guards import OPEN
from ppt_gym.guard_fork_policy import features, guard_tokens, numeric_description
from ppt_gym.live_quote_policy import FEATURES


class GuardForkPolicyTests(unittest.TestCase):
    def test_observation_carries_operands_and_rhs_without_evaluating(self):
        self.assertEqual(guard_tokens("BPD.OptimalBasis.geom 2 9 % 7 = 3"), (2, 9, 7, 3, 1))
        self.assertEqual(guard_tokens("MathCan.FiniteHorizonGap.maskAt 36 % 11 = 8"), (0, 36, 11, 8, 0))
        self.assertEqual(len(numeric_description((2, 9, 7, 3, 1))), 20)

    def test_discriminating_condition_stays_tied_to_the_applied_hypothesis(self):
        guard0 = {"raw": "BPD.OptimalBasis.geom 2 9 % 7 = 3"}
        guard1 = {"raw": "BPD.OptimalBasis.geom 3 9 % 7 = 3"}
        context = [
            {"name": "h0", "type": {"imp": [guard0, {"raw": "P 11"}]}},
            {"name": "h1", "type": {"imp": [guard1, {"raw": "P 12"}]}},
        ]
        public = {"goals": [{"hypotheses": context, "target": {"raw": "P ?w"}}]}
        tokens, guards = features(public, OPEN, 6, {})
        self.assertEqual(tuple(tokens.shape), (2, 3, FEATURES))
        self.assertEqual(guards, [guard_tokens(guard0["raw"]), guard_tokens(guard1["raw"])])
