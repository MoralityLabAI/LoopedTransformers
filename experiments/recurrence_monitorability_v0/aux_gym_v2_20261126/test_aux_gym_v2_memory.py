import unittest

import torch

from research_gym.interp.aux_gym_v2_memory import (
    FactorMemory, candidate_edits, factor_loss, physical_basis,
)


class MemoryTests(unittest.TestCase):
    def test_cap_fifo_and_key(self):
        m = FactorMemory(4 * 2**20, 128, range(17))
        self.assertLessEqual(m.allocated_bytes, m.byte_cap)
        self.assertGreater(m.slots, 0)
        key, negative = m.keys(torch.tensor([1]), 2, torch.tensor([3]))
        self.assertNotEqual(int(key[0]), int(negative[0]))
        rng = torch.Generator().manual_seed(1)
        _, hit = m.retrieve(key, rng, 4)
        self.assertFalse(bool(hit[0]))
        for i in range(m.slots + 1):
            m.insert(key, torch.full((1, 128), float(i)))
        self.assertEqual(m.evictions, 1)
        self.assertEqual(int(m.counts[key]), m.slots)
        v, hit = m.retrieve(key, rng, 4)
        self.assertTrue(bool(hit[0]))
        self.assertGreaterEqual(float(v.min()), 1)
        z = FactorMemory(0, 128, range(17))
        self.assertEqual(z.allocated_bytes, 0)
        self.assertEqual(z.receipt()["entries"], 0)

    def test_rank_and_gradient(self):
        g = torch.Generator().manual_seed(2)
        w = torch.randn(8, 12, generator=g)
        u, energy = physical_basis(w, 8, "svd", g)
        self.assertTrue(torch.allclose(u @ u.T, torch.eye(8), atol=1e-5))
        self.assertAlmostEqual(energy, 1.0, places=5)
        q = torch.randn(3, 8, generator=g, requires_grad=True)
        p = torch.randn(3, 8, generator=g, requires_grad=True)
        n = torch.randn(3, 8, generator=g, requires_grad=True)
        loss = factor_loss(q, p, n, torch.tensor([True, False, True]), u)
        loss.backward()
        self.assertGreater(float(q.grad.norm()), 0)
        self.assertEqual(float(q.grad[1].norm()), 0)
        self.assertIsNone(p.grad)
        self.assertIsNone(n.grad)

    def test_counterfactual_edit_shape(self):
        import random
        a = {"start": 2, "operations": [[2, 3], [4, 5]]}
        b, edit = next(candidate_edits("modular_recurrence", a, random.Random(1)))
        self.assertEqual(sum(x != y for x, y in zip(a["operations"], b["operations"])), 1)
        self.assertEqual(edit["kind"], "op_change")
        a = {"start": 0, "mapping": [1, -1, 0, -1], "hops": 2}
        b, _ = next(candidate_edits("masked_pointer_chase", a, random.Random(2)))
        self.assertEqual(sorted(a["mapping"]), sorted(b["mapping"]))
        self.assertEqual(sum(x != y for x, y in zip(a["mapping"], b["mapping"])), 2)


if __name__ == "__main__":
    unittest.main()
