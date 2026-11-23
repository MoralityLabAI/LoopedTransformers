import unittest

from ppt_gym.private_quotes import COMMITS, REGIMES, public_view, rollout, scenario, structure, tariff_layout
from ppt_gym.quote_fragments import QuoteFragmentGraph, source_fragments


class PrivateQuotesTests(unittest.TestCase):
    def setUp(self):
        plans = (("constructor", "r0", "r1", "r2", "h0", "shared"),
                 ("constructor", "r3", "r4", "r5", "h1", "shared"))
        ledger, other = tariff_layout(plans)
        self.case = {"module": "toy-plan", "plans": plans, "ledger": ledger, "other": other, "quote_order": tuple(ledger)}

    def test_prefix_producers_keep_equal_prices(self):
        for world in range(4):
            prices = scenario(self.case["plans"], self.case["ledger"], self.case["other"], REGIMES["direct"], world)
            self.assertEqual([prices[step] for step in ("constructor", "r0", "r3")], [1, 1, 1])

    def test_only_purchased_observations_enter_history(self):
        def choose(menu, history, left, fees):
            if not history:
                self.assertEqual(left, 9)
                return "ledger_u"
            self.assertEqual(history, (("ledger_u", 1),))
            self.assertNotIn("ledger_u", menu)
            self.assertEqual(left, 8)
            return COMMITS[0]

        row = rollout(self.case, "direct", 3, choose)
        self.assertTrue(row["proof"])
        self.assertEqual(row["proof_effort"] + row["quote_effort"], row["budget"])
        self.assertEqual(row["trace"], [{"action": "ledger_u", "charge": 1, "actual": 1, "seen": 1}])

    def test_selective_information_requires_more_than_a_one_step_quote(self):
        expected = {
            "strict": (2, 0),
            "direct": (4, 4),
            "two_quotes": (4, 8),
            "too_expensive": (2, 0),
            "either_plan": (4, 0),
        }
        for regime, counts in expected.items():
            with self.subTest(regime=regime):
                policy = structure(self.case, regime)
                rows = [rollout(self.case, regime, world, policy) for world in range(4)]
                self.assertEqual((sum(row["proof"] for row in rows), sum(row["quote_effort"] for row in rows)), counts)
        one_step = structure(self.case, "two_quotes", one_step=True)
        self.assertEqual(sum(rollout(self.case, "two_quotes", world, one_step)["proof"] for world in range(4)), 2)

    def test_bit_interchange_preserves_actual_charge_and_reverses_direct_success(self):
        policy = structure(self.case, "direct")
        for world in range(4):
            row = rollout(self.case, "direct", world, policy, patch=lambda _, bit: 1 - bit)
            self.assertFalse(row["proof"])
            self.assertEqual(row["quote_effort"], 1)
            self.assertLess(row["proof_effort"], row["budget"])

    def test_fragment_backup_uses_both_observed_yes_and_no_outcomes(self):
        view = public_view(self.case, "two_quotes")
        fragments = source_fragments(self.case, "two_quotes")
        graph = QuoteFragmentGraph(view, fragments)
        for action in ("ledger_u", "ledger_v"):
            backed_up, empirical = graph.labels((), action)
            self.assertEqual(backed_up, .875)
            self.assertTrue(empirical is None or empirical < backed_up)
        short = [f for f in fragments if not f["history"] and len(f["steps"]) == 1]
        graph = QuoteFragmentGraph(view, short)
        self.assertEqual(graph.labels((), "ledger_u"), (None, None))

    def test_fragment_restarts_require_a_paid_anchor(self):
        view = public_view(self.case, "two_quotes")
        fragment = {"history": (("ledger_u", 1),),
                    "steps": [{"action": "commit_0", "proof": False, "effort": 6}]}
        with self.assertRaisesRegex(ValueError, "unvisited"):
            QuoteFragmentGraph(view, [fragment])
