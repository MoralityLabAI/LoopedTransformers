import unittest

import torch

from configs.run_ppt_guard_probe import paired_intervention
from ppt_gym.guard_probe import ProbeGraph, revised_scores
from ppt_gym.guard_recovery import ALLOWANCES, PROBE, feasible_menu, initial
from ppt_gym.guard_recovery_learning import GUARD_WORLDS, baked_policy, trace_policy


class GuardProbeTests(unittest.TestCase):
    def setUp(self):
        self.graph = ProbeGraph()
        self.probability = [.42, .88, .76, .97]
        self.encoded = self.graph.encode([[0.] * 20] * 4, self.probability, [[0.] * 8] * 2)

    def test_zero_correction_keeps_native_probe_values(self):
        _, prior, _, offsets, payoffs, _, _, _ = self.encoded
        for i, support in enumerate(self.graph.support.tolist()):
            values = {probe: offset for probe, offset, available in
                      zip(PROBE, offsets[i], self.graph.available[i], strict=True) if available}
            actual = revised_scores(values, payoffs[i], prior, [0.] * 4, support)
            self.assertEqual(values, actual)

    def test_reduced_rollout_reproduces_frozen_native_traces(self):
        _, prior, enabled, offsets, payoffs, edges, rewards, baseline = (
            torch.tensor([part] * 4) for part in self.encoded)
        candidates = self.graph.candidate_values(prior, torch.zeros_like(payoffs[:, :, 0]), offsets, payoffs)
        policy = torch.nn.functional.one_hot(candidates.argmax(dim=-1), len(PROBE)).float()
        expected = self.graph.rollout(policy, torch.arange(4), enabled, edges, rewards, baseline)
        for index, truth in enumerate(GUARD_WORLDS):
            frozen = baked_policy(self.probability, action_space=feasible_menu)
            routes = {f"{branch}_{order}": {"proof": ["exact example"]} for branch in range(2) for order in range(2)}
            case = {"truth": truth, "routes": routes}
            for col, budget in enumerate(ALLOWANCES):
                row = trace_policy(case, budget, lambda state, _: frozen(state), action_space=feasible_menu)
                self.assertAlmostEqual(expected[index, col].item(), row["utility"], places=6)
        self.assertNotIn(initial(6), self.graph.index)

    def test_probe_and_bystander_interventions_have_equal_dose(self):
        addends = [0.125, 0.5, -0.25, 0.0]
        target, dose = paired_intervention(addends, PROBE[1], PROBE[0], False)
        sham, sham_dose = paired_intervention(addends, PROBE[1], PROBE[0], True)
        self.assertEqual(target, [0.5, 0.125, -0.25, 0.0])
        self.assertEqual(sham, [0.125, 0.5, -0.625, 0.375])
        self.assertEqual(dose, sham_dose)

    def test_paid_true_check_masks_that_false_guard_hypothesis(self):
        _, prior, _, offsets, payoffs, _, _, _ = (torch.tensor([part]) for part in self.encoded)
        ignored = ~self.graph.support
        self.assertTrue(ignored.any())
        zero = torch.zeros_like(ignored, dtype=torch.float32).unsqueeze(0)
        masked = ignored.float().unsqueeze(0) * 100
        expected = self.graph.candidate_values(prior, zero, offsets, payoffs)
        actual = self.graph.candidate_values(prior, masked, offsets, payoffs)
        torch.testing.assert_close(actual, expected)
