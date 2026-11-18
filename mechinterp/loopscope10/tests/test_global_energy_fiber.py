"""LGA-07C source-only test. Uses only 61061, untrained model, fake energy."""
import unittest

import torch

from mechinterp.loopscope04.gradient_flow import TinyRecursiveModel, fixed_data
from mechinterp.loopscope10.donor_source import legal_predecessor_donor
from mechinterp.loopscope10.global_energy_fiber import (
    ARMS, SHAMS, full_head_kernel, project_global_donor,
    label_blind_global_plan, unit_scale,
)


class QuadraticEnergy(torch.nn.Module):
    def forward(self,h,k,mu,std):
        return ((h-mu)/std).square().sum(-1)*.5


class GlobalFiberTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(61061)
        cls.model=TinyRecursiveModel().eval().requires_grad_(False)
        cls.energy=QuadraticEnergy()
        cls.x,_=fixed_data(1,61061)
        donor,_=legal_predecessor_donor(cls.x[0],10)
        cls.donor=donor[None]
        with torch.no_grad():
            cls.hrec=cls.model.reason_states(cls.x)[2].detach()
            cls.hdon=cls.model.reason_states(cls.donor)[2].detach()
        cls.mu=torch.zeros(48)
        cls.std=torch.ones(48)

    def test_known_global_rank_and_energy_silence(self):
        w=torch.tensor([[1.,0.,0.]],dtype=torch.float64)
        h=full_head_kernel(w)
        u=torch.tensor([[1.,2.,3.],[4.,5.,6.]],dtype=torch.float64)
        g=torch.tensor([[0.,1.,0.],[0.,0.,1.]],dtype=torch.float64)
        f=project_global_donor(u,g,h)
        self.assertEqual((f.rank_head,f.dimension,f.energy_row_active),(1,3,True))
        self.assertLess(float(torch.linalg.vector_norm(f.tangent@w.T)),1e-12)
        self.assertLess(abs(float((f.tangent*g).sum())),1e-12)
        self.assertLess(float(torch.linalg.vector_norm(h.projector@h.projector-h.projector)),1e-12)

    def test_head_scale_invariance(self):
        a=torch.eye(5,dtype=torch.float64)[:2]
        b=a.clone();b[0]*=1e12;b[1]*=1e-6
        k=full_head_kernel(a)
        k2=full_head_kernel(b)
        self.assertEqual(k.rank,k2.rank)
        torch.testing.assert_close(k.projector,k2.projector,rtol=0,atol=1e-12)

    def test_zero_energy_row_has_no_extra_causal_constraint(self):
        p=full_head_kernel(torch.eye(4,dtype=torch.float64)[:2])
        f=project_global_donor(torch.ones(3,4),torch.zeros(3,4),p)
        self.assertFalse(f.energy_row_active)
        self.assertEqual(f.dimension,6)
        self.assertLess(abs(f.tangent_derivative),1e-14)

    def test_label_blind_natural_whole_state_donor(self):
        hf=full_head_kernel(self.model.head.weight)
        plan=label_blind_global_plan(
            self.model,self.energy,self.hrec,self.hdon,self.mu,self.std,
            goal=10,head_fiber=hf,teacher_seed=103,base_index=0)
        self.assertTrue(plan["eligible"])
        self.assertEqual(set(plan["directions"]),set(ARMS))
        self.assertEqual(len(ARMS),14)
        self.assertEqual(plan["tangent_dimension"],3077)
        d=plan["directions"]["full_donor_state"]
        torch.testing.assert_close(self.hrec[0].double()+d,self.hdon[0].double(),
                                   atol=2e-6,rtol=0)
        g=plan["fiber"].gradient
        w=self.model.head.weight.detach().double()
        for arm in ("global_energy_tangent",*(f"same_k_sham_{i}" for i in range(SHAMS))):
            z=plan["directions"][arm]
            self.assertAlmostEqual(float(z.norm()),plan["radius"],delta=1e-7)
            self.assertLess(float((z@w.T).norm()),1e-7)
            self.assertLess(abs(float((z*g).sum())),1e-7*max(1.,plan["radius"]*float(g.norm())))
        clone=label_blind_global_plan(
            self.model,self.energy,self.hrec,self.hdon,self.mu,self.std,
            goal=10,head_fiber=hf,teacher_seed=103,base_index=0)
        for arm in ARMS:
            torch.testing.assert_close(plan["directions"][arm],clone["directions"][arm],
                                       atol=0,rtol=0)
        with torch.no_grad():
            copy=self.hdon.clone()
            for _ in range(2):copy=self.model.block(copy)
            oracle=self.model.head(copy)
            patch=self.hrec.clone()
            patch[0]=self.hdon[0]  # whole physical donor state positive control
            for _ in range(2):patch=self.model.block(patch)
            torch.testing.assert_close(self.model.head(patch),oracle,atol=0,rtol=0)

    def test_bad_shapes_and_zero_dose(self):
        with self.assertRaises(ValueError):
            full_head_kernel(torch.tensor([1.,2.,3.]))
        with self.assertRaises(ValueError):
            project_global_donor(torch.zeros(2,3),torch.ones(2,4),
                                 full_head_kernel(torch.ones(1,3)))
        with self.assertRaises(ValueError):
            unit_scale(torch.zeros(3),1.)
        with self.assertRaises(ValueError):
            full_head_kernel(torch.tensor([[float("nan"),0.]]))


if __name__=="__main__":
    unittest.main()
