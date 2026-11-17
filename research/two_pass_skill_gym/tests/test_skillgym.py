import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gym import TwoPassSkill, generate, residual, residual_certificate


class TwoPassGymTests(unittest.TestCase):
    def test_shapes_and_grad(self):
        m, b, y, gamma = generate(4, 8, seed=12)
        net = TwoPassSkill(size=8, visits=2)
        z, trajectory = net(m, b)
        self.assertEqual(tuple(z.shape), (4, 8))
        self.assertEqual(len(trajectory), 2)
        (z-y).square().mean().backward()
        self.assertIsNotNone(net.readout.weight.grad)

    def test_certificate_bounds_error(self):
        m, b, y, _ = generate(32, 8, seed=15)
        z = b.clone()
        bound = residual_certificate(m, b, z)
        self.assertTrue(bool(torch.all((z-y).abs().amax(-1) <= bound + 1e-5)))

    def test_fixed_point_preservation(self):
        m, b, y, _ = generate(4, 8, seed=19)
        self.assertLess(float(residual(m, b, y).abs().max()), 1e-5)

    def test_repeat_deterministic(self):
        m, b, _, _ = generate(2, 8, seed=19)
        net = TwoPassSkill(size=8)
        net.eval()
        with torch.no_grad():
            a = net(m, b)[0]
            c = net(m, b)[0]
        self.assertTrue(torch.equal(a,c))


if __name__ == "__main__":
    unittest.main()
