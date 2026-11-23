from itertools import product
from unittest import TestCase

import torch

from ppt_gym.scope_policy import ScopeSlots, inputs
from ppt_gym.scope_recovery import EARLY, OPEN, RESOLVE, TENTATIVE, Layout, State, legal, rollout, step


def worlds(edited):
    for left, right, tentative, keep in product((0, 1), repeat=4):
        early = (left, right)
        outcome = early[1 - edited] ^ (tentative if keep else early[edited])
        yield {"edited": edited, "observations": [{"value": bit} for bit in (*early, tentative, keep)],
               "witness_trials": [{"passed": witness == outcome, "checks": 6} for witness in (0, 1)]}


class ScopeRecoveryTests(TestCase):
    def test_native_observation_values_expire_by_scope(self):
        before = step(State(3), EARLY[0], 1)
        pending = step(before, OPEN)
        self.assertNotIn(EARLY[1], legal(pending))
        proposed = step(pending, TENTATIVE, 0)
        restored = step(proposed, RESOLVE, 0)
        self.assertEqual(restored.early, (1, -1))
        self.assertEqual(restored.tentative, 0)
        self.assertEqual(legal(restored), (0, 1))
        with self.assertRaisesRegex(ValueError, "unavailable"):
            step(restored, TENTATIVE, 1)

    def test_optimal_values_include_expired_probes_and_mandatory_check(self):
        for edited in (0, 1):
            rows = list(worlds(edited))
            layout = Layout(rows)
            for budget, expected in zip((1, 2, 3), (.45, .60, .80), strict=True):
                def decide(state):
                    q = layout.optimal(state)
                    return max(q, key=q.__getitem__)

                outcomes = [rollout(row, budget, decide) for row in rows]
                self.assertAlmostEqual(sum(x["utility"] for x in outcomes) / len(outcomes), expected)
                self.assertTrue(all(x["diagnostics"] <= budget + 1 for x in outcomes))

    def test_scope_memory_replaces_only_the_addressed_slot_or_restores_both(self):
        model = ScopeSlots().eval()
        for edited in (0, 1):
            with torch.no_grad():
                states = [State(0, 2, (0, 1), 1 - edited, keep) for keep in (0, 1)]
                context, history = zip(*(inputs(state, edited) for state in states), strict=True)
                context, history = torch.tensor(context).float(), torch.tensor(history).float()
                active, pending = model.memory(context, history)
                torch.testing.assert_close(active[0], model.written(history[:, :2])[0])
                torch.testing.assert_close(active[1, 1 - edited], active[0, 1 - edited])
                self.assertFalse(torch.equal(active[0, edited], active[1, edited]))
                self.assertEqual(int(pending.count_nonzero()), 0)
