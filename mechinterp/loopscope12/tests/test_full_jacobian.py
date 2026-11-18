"""LGA-09 excluded 61063 full future digit-logit source-only controls."""
from __future__ import annotations
import unittest
import torch

from mechinterp.loopscope04.gradient_flow import TinyRecursiveModel,fixed_data
from mechinterp.loopscope10.donor_source import legal_predecessor_donor
from mechinterp.loopscope10.global_energy_fiber import full_head_kernel
from mechinterp.loopscope11.pair_fiber import pair_project,project_into_k,normalized
from mechinterp.loopscope12.full_jacobian import (
    ARMS,SHAMS,SMOKE_SEED,GOALS,plan_full_j,
    future_ten_jacobian,full_j_matched_sham,measure_full_response,stable_seed
)


class QuadraticPotential(torch.nn.Module):
    def forward(self,h,k,mu,std):
        return .5*((h-mu)/std).square().sum(-1)


class FullFutureJacobianTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(SMOKE_SEED)
        cls.model=TinyRecursiveModel().eval().requires_grad_(False)
        cls.energy=QuadraticPotential().eval().requires_grad_(False)
        cls.x,_=fixed_data(1,SMOKE_SEED)
        cls.mu=torch.zeros(48)
        cls.std=torch.ones(48)
        cls.head=full_head_kernel(cls.model.head.weight)

    def test_full_J_contains_ten_real_native_two_block_derivatives(self):
        goal=10
        with torch.no_grad():
            h=self.model.reason_states(self.x)[2].detach()
        j,logits=future_ten_jacobian(self.model,h,goal)
        self.assertEqual(j.shape,(10,2,48))
        self.assertEqual(logits.shape,(10,))
        # True centered finite differences for one current pair coordinate.
        z=torch.zeros_like(h);z[0,(goal-1)%81,11]=1e-3
        with torch.no_grad():
            def out(state):
                z=state
                for _ in range(2):z=self.model.block(z)
                return self.model.head(z)[0,goal].double()
            numerical=(out(h+z)-out(h-z))/.002
        torch.testing.assert_close(numerical,j[:,0,11],atol=3e-3,rtol=0)

    def test_manufactured_pair_full_J_has_65_future_dark_dimensions(self):
        head=full_head_kernel(torch.eye(10,48,dtype=torch.float64))
        u=torch.randn(2,48,dtype=torch.float64)
        e=torch.zeros_like(u);e[0,10]=1.
        f=pair_project(u,e,head.projector)
        self.assertEqual(f.dimension,75)
        j=torch.zeros(10,2,48,dtype=torch.float64)
        for c in range(10):j[c,0,11+c]=1.
        jr=torch.stack([project_into_k(j[c],f,head.projector).flatten()
                        for c in range(10)])
        self.assertEqual(int(torch.linalg.matrix_rank(jr)),10)
        self.assertGreaterEqual(f.dimension-int(torch.linalg.matrix_rank(jr)),65)

    def test_source_legal_native_full_J_plan_is_exact_and_deterministic(self):
        goal=10
        donor,info=legal_predecessor_donor(self.x[0],goal)
        with torch.no_grad():
            hrec=self.model.reason_states(self.x)[2].detach()
            hdon=self.model.reason_states(donor[None])[2].detach()
        p=plan_full_j(self.model,self.energy,hrec,hdon,self.mu,self.std,
                      goal=goal,
                      original_class=info["original_goal_label"],
                      donor_class=info["donor_goal_label"],
                      teacher_seed=103,base_id=0,head_projector=self.head.projector)
        self.assertTrue(p["eligible"],p["reason"])
        self.assertEqual(set(p["directions"]),set(ARMS))
        self.assertEqual(len(ARMS),31)
        self.assertEqual(p["pair_plan"]["fiber"].dimension,75)
        self.assertLessEqual(p["rank_full_j"],10)
        self.assertGreaterEqual(p["future_null_dimension"],65)
        radius=p["natural_pair_radius"]
        j=p["future"].restricted_jacobian
        selected=p["directions"]["pair_donor_k"].flatten()
        response=j@selected
        for arm in ("pair_donor_k",*(f"full_j_sham_{i}" for i in range(SHAMS))):
            z=p["directions"][arm]
            self.assertAlmostEqual(float(z.norm()),radius,delta=1e-7)
            self.assertLess(float((z@self.model.head.weight.T.double()).norm()),1e-7)
            self.assertLess(abs(float((z*p["pair_plan"]["fiber"].egrad).sum())),
                            1e-7*max(1.,radius))
            torch.testing.assert_close(j@z.flatten(),response,atol=1e-8,rtol=0)
        p2=plan_full_j(self.model,self.energy,hrec,hdon,self.mu,self.std,
                       goal=goal,
                       original_class=info["original_goal_label"],
                       donor_class=info["donor_goal_label"],
                       teacher_seed=103,base_id=0,head_projector=self.head.projector)
        for a in ARMS:
            torch.testing.assert_close(p["directions"][a],p2["directions"][a],
                                       atol=0,rtol=0)

    def test_zero_gradient_rejected_or_properly_recorded(self):
        with self.assertRaises(ValueError):
            future_ten_jacobian(self.model,torch.zeros(81,48),10)
        self.assertEqual(len(ARMS),31)


if __name__=="__main__":
    unittest.main()
