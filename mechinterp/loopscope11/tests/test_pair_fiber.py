"""LGA-08 excluded 61062 source-only algebra/gradient test (no original weights)."""
import unittest
import torch

from mechinterp.loopscope04.gradient_flow import TinyRecursiveModel,fixed_data
from mechinterp.loopscope10.donor_source import legal_predecessor_donor
from mechinterp.loopscope10.global_energy_fiber import full_head_kernel
from mechinterp.loopscope11.pair_fiber import (
    GOALS,SHAMS,ARMS,pair_project,plan_pair,project_into_k,
    matched_future_margin_sham,stable_seed,EXCLUDED_SEED,
)


class QuadraticEnergy(torch.nn.Module):
    def forward(self,h,k,mu,std):
        return .5*((h-mu)/std).square().sum(-1)


class PairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(EXCLUDED_SEED)
        cls.teacher=TinyRecursiveModel().eval().requires_grad_(False)
        cls.energy=QuadraticEnergy().eval().requires_grad_(False)
        cls.head=full_head_kernel(cls.teacher.head.weight)
        cls.x,_=fixed_data(1,EXCLUDED_SEED)
        cls.mu=torch.zeros(48)
        cls.std=torch.ones(48)

    def test_pair_rank_and_exact_double_geometry(self):
        u=torch.arange(96,dtype=torch.float64).reshape(2,48)*.001
        g=torch.flip(u,dims=(1,)) + .03
        f=pair_project(u,g,self.head.projector)
        self.assertEqual(f.dimension,75)
        self.assertTrue(f.energy_active)
        self.assertLess(f.head_residual,1e-10)
        self.assertLess(abs(f.tangent_residual),1e-10)
        self.assertLess(float((f.k_donor @ self.teacher.head.weight.T.double()).norm()),1e-10)

    def test_known_exact_75_dim_and_zero_energy_row(self):
        head=torch.eye(10,48,dtype=torch.float64)
        p=full_head_kernel(head)
        delta=torch.randn(2,48,dtype=torch.float64)
        g=torch.zeros_like(delta);g[0,10]=1.
        f=pair_project(delta,g,p.projector)
        self.assertEqual((f.dimension,f.energy_active),(75,True))
        self.assertLess(abs(float((f.k_donor*g).sum())),1e-12)
        off=pair_project(delta,torch.zeros_like(g),p.projector)
        self.assertEqual((off.dimension,off.energy_active),(76,False))

    def test_legal_native_sourced_pair_all_23_arms(self):
        goal=10
        donor,info=legal_predecessor_donor(self.x[0],goal)
        with torch.no_grad():
            hrec=self.teacher.reason_states(self.x)[2]
            hdon=self.teacher.reason_states(donor[None])[2]
        plan=plan_pair(self.teacher,self.energy,hrec,hdon,self.mu,self.std,
                       goal=goal,
                       old_class=info["original_goal_label"],
                       donor_class=info["donor_goal_label"],
                       teacher_seed=103,base_index=0,
                       head_projector=self.head.projector)
        self.assertTrue(plan["eligible"])
        self.assertEqual(plan["fiber"].dimension,75)
        self.assertEqual(len(ARMS),23)
        self.assertEqual(set(plan["directions"]),set(ARMS))
        radius=plan["radius"]
        signed=float(plan["selected_signed_derivative"])
        for name,z in plan["directions"].items():
            self.assertEqual(z.shape,(2,48))
            if name in ("pair_energy_tangent","pair_head_only") or name.startswith(
                    "pair_random_k_") or name.startswith("pair_gradient_match_"):
                self.assertAlmostEqual(float(z.norm()),radius,delta=1e-7)
                self.assertLess(float((z@self.teacher.head.weight.T.double()).norm()),1e-7)
            if name=="pair_energy_tangent" or name.startswith(
                    "pair_random_k_") or name.startswith("pair_gradient_match_"):
                self.assertLess(abs(float((z*plan["fiber"].egrad).sum())),
                                1e-7*max(1.,radius*float(plan["fiber"].egrad.norm())))
            if name.startswith("pair_gradient_match_"):
                self.assertAlmostEqual(float((z*plan["future_k_gradient"]).sum()),
                                       signed,delta=1e-8*max(1.,abs(signed)))
        repeat=plan_pair(self.teacher,self.energy,hrec,hdon,self.mu,self.std,
                         goal=goal,old_class=info["original_goal_label"],
                         donor_class=info["donor_goal_label"],
                         teacher_seed=103,base_index=0,
                         head_projector=self.head.projector)
        for name in ARMS:
            torch.testing.assert_close(plan["directions"][name],
                                       repeat["directions"][name],rtol=0,atol=0)
        with torch.no_grad():
            true=hdon
            for _ in range(2):true=self.teacher.block(true)
            patched=hrec.clone()
            patched[0]=hdon[0]
            for _ in range(2):patched=self.teacher.block(patched)
            torch.testing.assert_close(self.teacher.head(patched),
                                       self.teacher.head(true),rtol=0,atol=0)

    def test_signed_gradient_sham_is_distinct_but_equal_first_order(self):
        g=torch.zeros((2,48),dtype=torch.float64);g[0,10]=1
        u=torch.zeros((2,48),dtype=torch.float64)
        u[0,10]=.4;u[0,11]=.8
        p=full_head_kernel(torch.eye(10,48,dtype=torch.float64)).projector
        e=torch.zeros_like(g);e[1,12]=1.
        f=pair_project(u,e,p)
        from mechinterp.loopscope11.pair_fiber import normalized
        selected=normalized(f.k_donor,f.natural_radius)
        sham=matched_future_margin_sham(
            selected,g,f,p,f.natural_radius,seed=stable_seed("smoke"))
        self.assertGreater(float(torch.norm(selected-sham)),.1)
        self.assertLess(abs(float((selected*g).sum()-(sham*g).sum())),1e-10)
        self.assertAlmostEqual(float(sham.norm()),f.natural_radius,delta=1e-10)

    def test_zero_sham_radius_rejected(self):
        p=full_head_kernel(torch.eye(10,48,dtype=torch.float64)).projector
        f=pair_project(torch.zeros((2,48)),torch.ones((2,48)),p)
        with self.assertRaises(ValueError):
            matched_future_margin_sham(torch.zeros((2,48)),torch.zeros((2,48)),f,p,
                                        radius=0.,seed=1)

    def test_orthogonal_channel_transport_pair_projection(self):
        gen=torch.Generator().manual_seed(61062)
        w=torch.randn(10,48,dtype=torch.float64,generator=gen)
        q,_=torch.linalg.qr(torch.randn(48,48,dtype=torch.float64,generator=gen))
        p=full_head_kernel(w).projector
        p2=full_head_kernel(w@q).projector
        u=torch.randn(2,48,dtype=torch.float64,generator=gen)
        g=torch.randn(2,48,dtype=torch.float64,generator=gen)
        f=pair_project(u,g,p)
        rotated=pair_project(u@q,g@q,p2)
        torch.testing.assert_close(f.k_donor@q,rotated.k_donor,rtol=0,atol=1e-9)
        self.assertEqual(f.dimension,rotated.dimension)


if __name__=="__main__":
    unittest.main()
