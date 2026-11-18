"""LGA-10 excluded seed 61064 only; no original teacher/artifact imports."""
import math
import unittest
import torch

from mechinterp.loopscope04.gradient_flow import TinyRecursiveModel, fixed_data
from mechinterp.loopscope10.global_energy_fiber import full_head_kernel
from mechinterp.loopscope11.pair_fiber import pair_project,project_into_k
from mechinterp.loopscope13.curvature_match import (
    make_curvature_shams,native_margin_hvp,NEW_SEED,EXCLUDED_SEED,
    ANGLE_COS_MAX,stable_seed
)


class CurvatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.gen=torch.Generator().manual_seed(EXCLUDED_SEED)
        cls.h=torch.eye(10,48,dtype=torch.float64)
        cls.p=full_head_kernel(cls.h).projector
        cls.g=torch.zeros((2,48),dtype=torch.float64)
        cls.g[0,10]=1.
        cls.g[1,11]=.15
        cls.f=pair_project(
            torch.randn((2,48),dtype=torch.float64,generator=cls.gen),
            cls.g,cls.p)
        # Head-null and E tangent rows in dimensions 11..20.
        cls.R=torch.zeros((96,10),dtype=torch.float64)
        for i in range(10):
            cls.R[11+i,i]=1.
        assert float(torch.linalg.vector_norm(
            cls.R.T@cls.R-torch.eye(10,dtype=torch.float64)))<1e-12
        cls.selected=project_into_k(
            torch.randn((2,48),dtype=torch.float64,generator=cls.gen),
            cls.f,cls.p).reshape(-1)
        cls.selected=cls.selected/cls.selected.norm()*3.
        m=torch.randn((96,96),dtype=torch.float64,generator=cls.gen)
        cls.H=(m+m.T)/2.
        cls.proj=staticmethod(lambda z:project_into_k(z.reshape(2,48),cls.f,cls.p).reshape(-1))

    def test_analytic_quadratic_curvature_exact_same_ten_j(self):
        res=make_curvature_shams(
            self.selected,self.R,self.proj,lambda v:self.H@v,
            teacher_seed=103,base_index=0,goal=10)
        self.assertTrue(res.eligible,res.reason)
        self.assertEqual(len(res.shams),8)
        selected=self.selected
        target=float(selected@self.H@selected)
        for v,angle in zip(res.shams,res.angles):
            z=v.reshape(-1)
            self.assertLess(abs(float(z.norm()-selected.norm())),1e-8)
            self.assertLess(float(torch.linalg.vector_norm(
                self.R.T@(z-selected))),1e-8)
            self.assertLess(float(torch.linalg.vector_norm(self.proj(z)-z)),1e-8)
            self.assertLess(abs(float(z@self.H@z)-target),1e-8*max(1.,abs(target)))
            self.assertLessEqual(abs(math.cos(angle)),ANGLE_COS_MAX)
            self.assertGreater(float(torch.linalg.vector_norm(z-selected)),.05)
        repeat=make_curvature_shams(
            self.selected,self.R,self.proj,lambda v:self.H@v,
            teacher_seed=103,base_index=0,goal=10)
        self.assertEqual(res.angles,repeat.angles)
        self.assertEqual(res.attempts_used,repeat.attempts_used)
        for a,b in zip(res.shams,repeat.shams):
            torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_no_bracket_is_explicit_itt_ineligibility(self):
        res=make_curvature_shams(
            self.selected,self.R,self.proj,lambda v:self.H@v,
            teacher_seed=103,base_index=0,goal=10,max_attempts=0)
        self.assertFalse(res.eligible)
        self.assertEqual(res.reason,"INELIGIBLE_HESSIAN_BRACKET")
        self.assertEqual(res.shams,[])

    def test_zero_future_null_radius(self):
        selected=self.R[:,0].clone()
        res=make_curvature_shams(
            selected,self.R,self.proj,lambda v:self.H@v,
            teacher_seed=103,base_index=0,goal=10)
        self.assertFalse(res.eligible)
        self.assertEqual(res.reason,"DEGENERATE_FIRST_ORDER_DARK_RADIUS")

    def test_invalid_pair_and_bad_row_basis(self):
        with self.assertRaises(ValueError):
            make_curvature_shams(torch.zeros(95),self.R,self.proj,
                                 lambda v:self.H@v,teacher_seed=103,
                                 base_index=0,goal=10)
        with self.assertRaises(ValueError):
            make_curvature_shams(self.selected,self.R[:,:8]*2,self.proj,
                                 lambda v:self.H@v,teacher_seed=103,
                                 base_index=0,goal=10)

    def test_native_two_loop_hvp_is_symmetric_and_matches_gradient_fd(self):
        torch.manual_seed(EXCLUDED_SEED)
        model=TinyRecursiveModel().double().eval().requires_grad_(False)
        x,_=fixed_data(1,EXCLUDED_SEED)
        with torch.no_grad():
            h=model.reason_states(x)[2].detach()
        goal=10
        hvp=native_margin_hvp(model,h,goal,donor_class=3,original_class=2)
        v=torch.randn((2,48),dtype=torch.float64,generator=self.gen)
        w=torch.randn((2,48),dtype=torch.float64,generator=self.gen)
        v/=v.norm();w/=w.norm()
        Hv=hvp(v);Hw=hvp(w)
        self.assertTrue(bool(torch.isfinite(Hv).all() and torch.isfinite(Hw).all()))
        self.assertLess(abs(float(v.reshape(-1)@Hw-w.reshape(-1)@Hv)),1e-7)
        p=(goal-1)%81
        def grad_at(t):
            d=t.clone().requires_grad_(True)
            state=h.clone()
            state[0,p]=state[0,p]+d[0]
            state[0,goal]=state[0,goal]+d[1]
            logits=model.head(model.block(model.block(state)))[0,goal]
            grad,=torch.autograd.grad(logits[3]-logits[2],d)
            return grad.detach().reshape(-1)
        eps=1e-4
        estimate=(grad_at(eps*v)-grad_at(-eps*v))/(2*eps)
        self.assertLess(float(torch.linalg.vector_norm(estimate-Hv)),2e-5)

    def test_excluded_native_pair_full_j_margin_hvp_feasibility(self):
        from mechinterp.loopscope10.donor_source import legal_predecessor_donor
        from mechinterp.loopscope10.global_energy_fiber import full_head_kernel
        from mechinterp.loopscope11.pair_fiber import plan_pair
        from mechinterp.loopscope12.full_jacobian import measure_full_response
        torch.manual_seed(EXCLUDED_SEED+1)
        model=TinyRecursiveModel().double().eval().requires_grad_(False)
        x,_=fixed_data(1,EXCLUDED_SEED)
        donor,info=legal_predecessor_donor(x[0],10)
        with torch.no_grad():
            hrec=model.reason_states(x)[2]
            hdon=model.reason_states(donor[None])[2]
        class SyntheticEnergy(torch.nn.Module):
            def forward(self,h,k,mu,std):
                return .5*((h-mu)/std).square().sum(-1)
        p=full_head_kernel(model.head.weight).projector
        m=torch.zeros(48,dtype=torch.float64)
        v=torch.ones(48,dtype=torch.float64)
        parent=plan_pair(
            model,SyntheticEnergy(),hrec,hdon,m,v,goal=10,
            old_class=info["original_goal_label"],
            donor_class=info["donor_goal_label"],
            teacher_seed=103,base_index=0,head_projector=p)
        self.assertTrue(parent["eligible"])
        future=measure_full_response(model,hrec,10,parent,p)
        self.assertEqual(future.future_null_dimension,75-future.rank)
        hvp=native_margin_hvp(
            model,hrec,10,info["donor_goal_label"],
            info["original_goal_label"])
        kproj=lambda z:project_into_k(z.reshape(2,48),
                                      parent["fiber"],p).reshape(-1)
        plan=make_curvature_shams(
            parent["directions"]["pair_energy_tangent"],
            future.row_basis,kproj,hvp,
            teacher_seed=103,base_index=0,goal=10)
        self.assertIn(plan.reason,(None,"INELIGIBLE_HESSIAN_BRACKET"))
        print("LGA10_EXCLUDED_NATIVE_CURVATURE_FEASIBILITY",
              plan.eligible,len(plan.shams),plan.attempts_used)
        if plan.eligible:
            self.assertLess(plan.max_curvature_error,1e-7)
            self.assertLess(plan.max_jacobian_error,1e-8)
            self.assertEqual(len(plan.shams),8)

    def test_reserved_science_seed_not_used(self):
        self.assertEqual(NEW_SEED,31064)
        self.assertEqual(EXCLUDED_SEED,61064)
        self.assertEqual(stable_seed("curvature",103,0,10,0,0),
                         stable_seed("curvature",103,0,10,0,0))


if __name__=="__main__":
    unittest.main()
