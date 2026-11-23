from functools import lru_cache
import unittest

import torch

from ppt_gym.failed_prefix import ALLOWANCES, CHARGE, PREFIX_COST, WORLDS, cases, legal, planned_values, trace_case
from ppt_gym.failed_prefix_learning import STARTS, observations, policy_distribution, returns


class FailedPrefixTests(unittest.TestCase):
    def test_both_worlds_keep_native_costs_but_move_successful_routes(self):
        left, right = list(cases("source", 1))
        self.assertEqual(left["pair"], right["pair"])
        self.assertNotEqual(left["expected"], right["expected"])
        for a, b in zip(left["routes"], right["routes"], strict=True):
            self.assertEqual((a["pass_cost"], a["fail_cost"]), (b["pass_cost"], b["fail_cost"]))

    def test_failed_restart_keeps_the_retained_route_available_and_bills_checks(self):
        routes = [{"pass_cost": 2, "fail_cost": 2}, {"pass_cost": 4, "fail_cost": 3},
                  {"pass_cost": 5, "fail_cost": 3}]
        case = {"routes": routes, "truth": [True, False, False]}
        trace = trace_case(case, 5, lambda _, discarded, __: 0 if discarded else 1)
        self.assertTrue(trace["proved"])
        self.assertEqual(trace["checks"], PREFIX_COST + 5)
        self.assertEqual(trace["trace"][1]["before"], 2)
        scores = planned_values(routes, [float(bits == tuple(case["truth"])) for bits in WORLDS])
        self.assertAlmostEqual(scores(5, 0)[1], 1 - 5 * CHARGE)
        self.assertAlmostEqual(scores(2, 2)[0], 1 - 2 * CHARGE)

    def test_policy_returns_match_scalar_native_tactic_accounting(self):
        case = next(cases("source", 1))
        routes, truth = case["routes"], case["expected"]
        _, _, public_menu, next_state = observations(routes, [[0.] * 8] * 3)
        menu = torch.tensor([public_menu])
        scores = torch.zeros(menu.shape).masked_fill(~menu, -torch.inf).requires_grad_()
        pi = policy_distribution(scores)
        arrays = [torch.tensor([values]) for values in (next_state, public_menu)]
        costs = [torch.tensor([[float(route[key]) for route in routes]]) for key in ("pass_cost", "fail_cost")]
        got = returns(*arrays, *costs, torch.tensor([truth]), pi)

        @lru_cache(maxsize=None)
        def direct(left, failed):
            options = legal(routes, left, failed)
            values = []
            for arm in options:
                route = routes[arm]
                after = route["fail_cost"]
                values.append(1 - CHARGE * route["pass_cost"] if truth[arm]
                              else direct(left - after, failed | (1 << arm)) - CHARGE * after)
            return sum(values) / (1 + len(options))

        torch.testing.assert_close(got[0], torch.tensor([direct(budget, 0) for budget in ALLOWANCES]))
        self.assertTrue(torch.isfinite(torch.autograd.grad(got.sum(), scores)[0]).all())
        self.assertEqual(STARTS, tuple(budget * 8 for budget in ALLOWANCES))
