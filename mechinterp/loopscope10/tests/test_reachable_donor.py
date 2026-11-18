"""LGA-07B excluded 61060 source-only donor tests; 31060 stays unopened."""
import unittest
import torch
from mechinterp.loopscope04.gradient_flow import TinyRecursiveModel, fixed_data
from mechinterp.loopscope10.donor_source import (
    GOALS, legal_predecessor_donor, label_blind_donor_plan,
    native_donor_arm_results, stable_seed
)


class QuadraticPotential(torch.nn.Module):
    def forward(self,h,k,mu,std):
        return .5*((h-mu)/std).square().sum(dim=-1)


class LGA07BTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(61060)
        cls.model = TinyRecursiveModel().eval().requires_grad_(False)
        cls.potential = QuadraticPotential().eval()
        cls.mu = torch.zeros(48)
        cls.std = torch.ones(48)
        cls.x, _ = fixed_data(8,61060)  # reserved 31060 is never generated

    def test_all_eight_donor_targets_are_single_goal_isolated(self):
        for row in self.x:
            for j in GOALS:
                donor, info = legal_predecessor_donor(row,j)
                self.assertEqual((row!=donor).sum().item(),1)
                self.assertNotEqual(info["old_digit"],info["new_digit"])
                self.assertEqual(info["changed_targets"],[j])
                self.assertEqual(info["predecessor_position"],(j-1)%81)
                self.assertNotEqual(info["original_goal_label"],info["donor_goal_label"])
                self.assertTrue(bool(((donor>=1)&(donor<=9)).all()))

    def test_negative_source_contracts(self):
        x=self.x[0]
        with self.assertRaises(ValueError):
            legal_predecessor_donor(x.int(),GOALS[0])
        with self.assertRaises(ValueError):
            legal_predecessor_donor(x,81)
        with self.assertRaises(ValueError):
            legal_predecessor_donor(x[:80],GOALS[0])
        bad=x.clone();bad[0]=0
        with self.assertRaises(ValueError):
            legal_predecessor_donor(bad,GOALS[0])

    def test_frozen_matched_tangent_directions_and_true_native_continuation(self):
        j=10
        donor,_=legal_predecessor_donor(self.x[0],j)
        with torch.no_grad():
            recipient_state=self.model.reason_states(self.x[:1])[2]
            donor_state=self.model.reason_states(donor.unsqueeze(0))[2]
        plan=label_blind_donor_plan(
            self.model,self.potential,recipient_state,donor_state,
            self.mu,self.std,goal=j,seed=stable_seed(61060,0,j))
        self.assertTrue(plan["eligible"])
        self.assertGreaterEqual(plan["fiber"].null_dimension,37)
        self.assertEqual(len(plan["directions"]),12)  # projected/inverse, unproj/full, 8 nulls
        radius=plan["radius"]
        for name,d in plan["directions"].items():
            if name=="full_donor_token":
                continue
            self.assertAlmostEqual(float(d.norm()),radius,delta=1e-7)
        scored=native_donor_arm_results(
            self.model,self.potential,recipient_state,self.mu,self.std,
            goal=j,remaining=2,plan=plan)
        self.assertEqual(len(scored),12)
        self.assertLess(scored["projected_donor"]["finite_current_head_l2"],1e-5)
        self.assertLess(scored["inverse_projected_donor"]["finite_current_head_l2"],1e-5)
        self.assertEqual(scored["same_k_sham_0"]["goal_logits"].shape,(10,))
        for arm in scored:
            self.assertTrue(torch.isfinite(scored[arm]["goal_logits"]).all())
        plan_repeat=label_blind_donor_plan(
            self.model,self.potential,recipient_state,donor_state,
            self.mu,self.std,goal=j,seed=stable_seed(61060,0,j))
        for key in plan["directions"]:
            torch.testing.assert_close(plan["directions"][key],
                                      plan_repeat["directions"][key],
                                      atol=0.,rtol=0.)

    def test_invalid_matched_states_rejected(self):
        h=torch.zeros(1,81,48)
        with self.assertRaises(ValueError):
            label_blind_donor_plan(self.model,self.potential,h,
                                   torch.zeros(2,81,48),self.mu,self.std,
                                   goal=0,seed=1)


if __name__=="__main__":
    unittest.main()
