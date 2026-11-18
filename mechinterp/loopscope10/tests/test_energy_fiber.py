"""LGA-07: excluded synthetic source-only CPU geometry and native-continuation checks.

Never creates reserved 31059 test sources and never reads LGA-02/05 binary weights.
"""
import unittest
import torch
from mechinterp.loopscope10.energy_fiber import (
    token_energy_fiber, native_decomposition, plan_source_only,
    native_finite_edit, future_selfscore_gradient, stable_seed
)
from mechinterp.loopscope04.gradient_flow import TinyRecursiveModel, fixed_data


class QuadraticPotential(torch.nn.Module):
    def forward(self, h, k, mu, std):
        return 0.5 * ((h - mu) / std).square().sum(-1)


class LGA07Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(61059)
        cls.teacher = TinyRecursiveModel().eval().requires_grad_(False)
        x, _ = fixed_data(1, 61059)  # explicitly excluded test-only smoke
        with torch.no_grad():
            cls.h = cls.teacher.reason_states(x)[2].detach()
        cls.potential = QuadraticPotential().eval()
        cls.mu = torch.zeros(48)
        cls.std = torch.ones(48)

    def test_known_null_rank_and_future_capacity(self):
        head = torch.eye(5, dtype=torch.float64)[:2]
        energy_g = torch.tensor([0.,0.,1.,0.,0.],dtype=torch.float64)
        f = token_energy_fiber(head, energy_g)
        self.assertEqual((f.rank_current,f.rank_joint,f.null_dimension),(2,3,2))
        self.assertEqual(f.lower_bound, 2)
        self.assertLess(f.null_residual,1e-12)
        future = torch.tensor([0.,0.,0.,1.,0.],dtype=torch.float64)
        self.assertGreater(float((future @ f.projector).norm()),.99)
        self.assertLess(float((head @ f.projector).norm()),1e-12)

    def test_constraint_scale_invariance(self):
        head=torch.eye(5,dtype=torch.float64)[:2]
        g=torch.tensor([0.,0.,1.e13,0.,0.],dtype=torch.float64)
        f=token_energy_fiber(head,g)
        self.assertEqual((f.rank_current,f.rank_joint,f.null_dimension),(2,3,2))
        self.assertLess(float((head@f.projector).norm()),1.e-12)
        self.assertLess(float((g@f.projector).norm()),1.e-3)
        scaled=head.clone()
        scaled[0] *= 1.e12
        scaled[1] *= 1.e-6
        f2=token_energy_fiber(scaled,g)
        self.assertEqual(f2.null_dimension,f.null_dimension)
        torch.testing.assert_close(f.projector,f2.projector,rtol=0,atol=1.e-11)
        self.assertLess(float((scaled@f2.projector).norm()),1.e-10)

    def test_orthogonal_change_of_coordinates_preserves_physical_projector(self):
        torch.manual_seed(60812)
        a = torch.randn(3,8,dtype=torch.float64)
        g = torch.randn(8,dtype=torch.float64)
        q, _ = torch.linalg.qr(torch.randn(8,8,dtype=torch.float64))
        f = token_energy_fiber(a,g)
        moved = token_energy_fiber(a @ q,q.T @ g)
        self.assertEqual(f.null_dimension, moved.null_dimension)
        torch.testing.assert_close(q @ moved.projector @ q.T,
                                   f.projector,rtol=0,atol=1e-11)

    def test_field_decomposition_and_gradient_zero(self):
        g = torch.tensor([[1.,2.],[3.,4.]])
        v = torch.tensor([[1.,-3.],[5.,2.]])
        parts = native_decomposition(v,g)
        self.assertTrue(parts["identified"])
        torch.testing.assert_close(parts["normal"]+parts["tangent"],
                                   v.double(),atol=1e-12,rtol=0)
        self.assertLess(abs(parts["tangent_dot_gradient"]),1e-11)
        zero = native_decomposition(v,torch.zeros_like(g))
        self.assertFalse(zero["identified"])
        self.assertIsNone(zero["alpha"])

    def test_finite_curvature_reversal(self):
        h=torch.tensor([1.,0.],dtype=torch.float64)
        step=torch.tensor([-.1,5.],dtype=torch.float64)
        self.assertLess(float(h @ step),0.)
        energy=lambda x: float((x*x).sum()/2)
        self.assertGreater(energy(h+step)-energy(h),0.)

    def test_excluded_smoke_native_future_gradient_finite_difference(self):
        grad, selected, _ = future_selfscore_gradient(
            self.teacher, self.h, goal=10, remaining=2)
        self.assertEqual(grad.shape,(48,))
        step = torch.zeros_like(self.h)
        step[0,10,7]=1e-3
        with torch.no_grad():
            def score(s):
                for _ in range(2):
                    s = self.teacher.block(s)
                return float(self.teacher.head(s)[0,10,selected])
            empirical=(score(self.h+step)-score(self.h-step))/.002
        self.assertAlmostEqual(empirical,float(grad[7]),delta=.002)

    def test_plan_label_blind_eight_shams_and_finite_native_edit(self):
        plan=plan_source_only(self.teacher,self.potential,self.h,
                              self.mu,self.std,goal=10,remaining=2,
                              seed=stable_seed("smoke",61059))
        self.assertTrue(plan["eligible"])
        f=plan["fiber"]
        self.assertGreaterEqual(f.null_dimension,37)
        self.assertEqual(plan["directions"].shape,(9,48))
        self.assertLess(plan["max_linear_constraint_error"],1e-7)
        self.assertLess(plan["max_dose_error"],1e-7)
        # Actual native continuation is run without any target-label input.
        edited=native_finite_edit(self.teacher,self.potential,self.h,
                                  self.mu,self.std,goal=10,remaining=2,
                                  edit=plan["directions"][0])
        self.assertEqual(edited["goal_logits"].shape,(10,))
        self.assertEqual(edited["all_logits"].shape,(1,81,10))
        self.assertLess(edited["finite_current_head_l2"],1e-5)
        self.assertAlmostEqual(edited["realized_edit_norm"],plan["radius"],
                               delta=1e-5)
        repeated=plan_source_only(self.teacher,self.potential,self.h,
                                   self.mu,self.std,goal=10,remaining=2,
                                   seed=stable_seed("smoke",61059))
        torch.testing.assert_close(plan["directions"], repeated["directions"],
                                   rtol=0,atol=0)

    def test_nonfinite_and_bad_shape_rejected(self):
        with self.assertRaises(ValueError):
            token_energy_fiber(torch.eye(2),torch.zeros(3))
        with self.assertRaises(ValueError):
            token_energy_fiber(torch.ones(2,3),torch.tensor([float("nan"),0.,0.]))
        with self.assertRaises(ValueError):
            future_selfscore_gradient(self.teacher,self.h,10,0)


if __name__ == "__main__":
    unittest.main()
