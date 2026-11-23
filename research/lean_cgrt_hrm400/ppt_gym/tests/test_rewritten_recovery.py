import unittest

import torch

from ppt_gym.failed_prefix import WORLDS, cases, planned_values
from ppt_gym.failed_prefix_learning import recovery_graph
from ppt_gym.rewritten_recovery import ALIASES, assignment_logits, mix_table, planned_batch, rewrite, unfold_guard


class RewrittenRecoveryTests(unittest.TestCase):
    def test_rewrite_changes_only_recovery_binders_and_preserves_route_recipes(self):
        case = next(cases("development", 1))
        original = {**case, "truth": case["expected"]}
        changed = rewrite(original, "development")
        self.assertNotEqual(changed["module"], original["module"])
        self.assertEqual(changed["statement"].split("(hr0", 1)[0], case["statement"].split("(hr0", 1)[0])
        self.assertEqual(changed["expected"], case["expected"])
        for a, b in zip(case["routes"], changed["routes"], strict=True):
            self.assertNotEqual(a["guard"], b["guard"])
            self.assertEqual(a["attempt"], b["attempt"])
            self.assertEqual(a["pass_cost"], b["pass_cost"])

    def test_assignment_normalizes_only_bijections_and_transfers_checked_bit(self):
        edge = torch.tensor([[0., 0., 10., 10., 0., 0., 0., 10., 0.]])
        selected = ALIASES[int(assignment_logits(edge).argmax())]
        self.assertEqual(selected, (2, 0, 1))
        table = mix_table([1.] * len(WORLDS), 0, False)
        self.assertTrue(all(not bits[1] for bits, weight in zip(WORLDS, table[ALIASES.index(selected)]) if weight))

    def test_unfolding_exposes_recurrence_and_shifts_depth(self):
        geom = unfold_guard("BPD.OptimalBasis.geom 4 12 % 7", 2)
        mask = unfold_guard("MathCan.FiniteHorizonGap.maskAt 50 % 13", 3)
        self.assertEqual(geom, "(1 + 4 * (1 + 4 * (BPD.OptimalBasis.geom 4 10))) % 7")
        self.assertEqual(mask.count("nextMask"), 3)
        self.assertIn("MathCan.FiniteHorizonGap.maskAt 47", mask)

    def test_differentiable_values_match_independent_scalar_planner(self):
        case = next(cases("source", 1))
        routes = case["routes"]
        visited, menu, links = recovery_graph(routes)
        p = torch.tensor([[1., 2., 1., 4., 3., 2.]], requires_grad=True)
        q = planned_batch(torch.tensor([links]), torch.tensor([menu]),
                          torch.tensor([[r["pass_cost"] for r in routes]], dtype=torch.float),
                          torch.tensor([[r["fail_cost"] for r in routes]], dtype=torch.float), p)
        scalar = planned_values(routes, p[0].detach().tolist())
        for budget, failed in visited:
            for route, value in scalar(budget, failed).items():
                self.assertAlmostEqual(q[0, 8 * budget + failed, route].item(), value, places=6)
        loss = q[q.isfinite()].sum()
        self.assertTrue(torch.isfinite(torch.autograd.grad(loss, p)[0]).all())
