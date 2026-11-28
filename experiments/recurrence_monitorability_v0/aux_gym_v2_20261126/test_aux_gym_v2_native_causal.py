import unittest
import torch
from research_gym.interp.aux_gym_v2_native_causal import component_ids, dose_match, read_swap_delta
from research_gym.interp.aux_gym_v2_trm import AuxTokenTRM, trace_positions


class NativeCausalTests(unittest.TestCase):
    def test_exact_component_delta(self):
        torch.manual_seed(1)
        a, b = torch.randn(1, 5), torch.randn(1, 5)
        v, u = torch.randn(5, 3), torch.randn(3, 4)
        d = read_swap_delta(a, b, v, u)
        self.assertTrue(torch.allclose(d, (b @ v - a @ v) @ u, atol=1e-6))
        ctrl = dose_match(torch.randn_like(d), float(d.norm()))
        self.assertAlmostEqual(float(ctrl.norm()), float(d.norm()), places=5)
        self.assertIsNone(dose_match(torch.zeros_like(d), 1.0))

    def test_gauge_invariant_rank(self):
        torch.manual_seed(2)
        v, u = torch.randn(6, 4), torch.randn(4, 5)
        scale = torch.tensor([0.1, -4.0, 2.0, 9.0])
        self.assertEqual(component_ids(v, u, 3), component_ids(v * scale, u / scale[:, None], 3))

    def test_native_trm_budget_grad(self):
        torch.set_num_threads(1)
        m = AuxTokenTRM(width=32)
        x = torch.randint(2048, (2, 64))
        o = m(x, commits=4)
        self.assertEqual(o.block_calls, 8)
        self.assertEqual(len(o.low_states), 4)
        self.assertEqual(len(o.high_states), 4)
        self.assertEqual(o.logits.shape, (2, 2048))
        o.logits.square().mean().backward()
        self.assertGreater(float(m.block.down_proj.weight.grad.norm()), 0)
        self.assertEqual(trace_positions(2, 4), [1, 2, 2, 2])


if __name__ == "__main__":
    unittest.main()
