"""Pre-pilot derivative and physical-rounding checks on exposed seed 61064."""
import unittest

import numpy as np
import torch

from mechinterp.loopscope04.gradient_flow import TinyRecursiveModel, fixed_data
from mechinterp.loopscope10.donor_source import legal_predecessor_donor
from mechinterp.loopscope13.curvature_match import native_margin_hvp
from mechinterp.loopscope13b.audit_receipt import _checks as numpy_checks
from mechinterp.loopscope13b.source_feasibility import _physical_checks


class NativeHessianTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(61064)
        cls.model = TinyRecursiveModel().double().eval().requires_grad_(False)
        cls.source = fixed_data(1, 61064)[0]
        with torch.no_grad():
            cls.hrec = cls.model.reason_states(cls.source)[2]

    def test_hvp_matches_central_gradient_difference(self):
        goal = 0
        _, metadata = legal_predecessor_donor(self.source[0], goal)
        donor = metadata["donor_goal_label"]
        original = metadata["original_goal_label"]
        generator = torch.Generator().manual_seed(61064)
        v = torch.randn(96, dtype=torch.float64, generator=generator)
        v /= v.norm()
        hvp = native_margin_hvp(self.model, self.hrec, goal, donor, original)
        exact = hvp(v)

        def gradient(edit):
            edit = edit.reshape(2, 48).detach().requires_grad_(True)
            state = self.hrec.clone()
            state[0, 80] = state[0, 80] + edit[0]
            state[0, goal] = state[0, goal] + edit[1]
            logits = self.model.head(self.model.block(self.model.block(state)))[0, goal]
            margin = logits[donor] - logits[original]
            return torch.autograd.grad(margin, edit)[0].detach().reshape(-1)

        errors = []
        for step in (1e-3, 5e-4):
            central = (gradient(step * v) - gradient(-step * v)) / (2 * step)
            errors.append(float((central - exact).norm()))
        self.assertLess(errors[-1], 1e-7 * max(1., float(exact.norm())))
        self.assertTrue(errors[-1] < .4 * errors[0] or errors[0] < 1e-11)

    def test_native_hessian_bilinear_symmetry(self):
        goal = 40
        _, metadata = legal_predecessor_donor(self.source[0], goal)
        hvp = native_margin_hvp(self.model, self.hrec, goal,
                                metadata["donor_goal_label"], metadata["original_goal_label"])
        generator = torch.Generator().manual_seed(61064)
        v = torch.randn(96, dtype=torch.float64, generator=generator)
        w = torch.randn(96, dtype=torch.float64, generator=generator)
        v, w = v / v.norm(), w / w.norm()
        self.assertLess(abs(float(v @ hvp(w) - w @ hvp(v))), 1e-11)


class PhysicalApplicationTests(unittest.TestCase):
    def test_numpy_audit_agrees_and_rounding_can_fail_curvature(self):
        dtype = torch.float64
        selected = torch.zeros(96, dtype=dtype)
        selected[0] = 1.
        sham = torch.zeros_like(selected)
        sham[0], sham[1] = .35, np.sqrt(1 - .35**2)
        intended = torch.stack((selected, sham))
        h = torch.linspace(-1., 1., 96, dtype=dtype)
        row_basis = torch.eye(96, dtype=dtype)[:, 10:11]
        k = torch.eye(96, dtype=dtype)
        jacobian = torch.zeros((10, 96), dtype=dtype)
        head = torch.zeros((10, 48), dtype=dtype)
        energy = torch.zeros(96, dtype=dtype)
        for fp32 in (False, True):
            if fp32:
                actual = (h.float()[None] + intended.float()).double() - h.float().double()[None]
            else:
                actual = (h[None] + intended) - h[None]
            # H=I supplies an exact independent quadratic and HVP.
            report = _physical_checks(actual, actual, selected, 1., k, jacobian,
                                      head, energy, row_basis, selected, fp32=fp32)
            passed, errors, limits = numpy_checks(
                actual.numpy(), actual.numpy(), selected.numpy(), 1., k.numpy(),
                jacobian.numpy(), head.numpy(), energy.numpy(), row_basis.numpy(), fp32=fp32)
            self.assertEqual(report["passed"], passed)
            for key in errors:
                self.assertAlmostEqual(errors[key], report["max_errors"][key], places=13)
                self.assertAlmostEqual(limits[key], report["limits"][key], places=13)
            if fp32:
                self.assertGreater(errors["curvature"], limits["curvature"])
                self.assertFalse(passed)
            else:
                self.assertTrue(passed)


if __name__ == "__main__":
    unittest.main()
