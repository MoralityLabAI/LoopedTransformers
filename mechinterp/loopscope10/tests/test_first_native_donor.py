"""LGA-07B source-only regression: excluded 61060 and random untrained teacher.

Do not load original teacher or energy bundles. Never generate seed 31060.
"""
from __future__ import annotations

import unittest
import torch

from mechinterp.loopscope04.gradient_flow import TinyRecursiveModel, fixed_data
from mechinterp.loopscope10.donor_source import (
    GOALS, legal_predecessor_donor, label_blind_donor_plan, stable_seed
)
from mechinterp.loopscope10.native_donor_eval import (
    ARMS, native_states, score_batch, summarize, median_binary, edit_sha
)
from mechinterp.loopscope10.audit_donor_receipt import (
    verify_case, median_binary as audit_median
)


class ArtificialPotential(torch.nn.Module):
    def forward(self,h,k,mu,std):
        return ((h-mu)/std).square().sum(-1)*0.5


class SourceOnly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(61060)
        cls.model=TinyRecursiveModel().eval().requires_grad_(False)
        cls.energy=ArtificialPotential().eval().requires_grad_(False)
        cls.x,cls.y=fixed_data(2,61060)
        cls.mu=torch.zeros(48)
        cls.std=torch.ones(48)

    def test_legal_source_only_predecessor_mutation(self):
        for goal in GOALS:
            donor,info=legal_predecessor_donor(self.x[0],goal)
            self.assertEqual(info["changed_targets"],[goal])
            self.assertEqual((donor!=self.x[0]).sum().item(),1)
            self.assertNotEqual(info["original_goal_label"],
                                info["donor_goal_label"])
            self.assertEqual(int(donor.roll(1)[goal]),info["donor_goal_label"])

    def test_label_blind_plan_and_native_batch_with_12_arms(self):
        x=self.x[:1]
        donor_cases=[legal_predecessor_donor(x[0],g) for g in GOALS]
        donors=torch.stack([z[0] for z in donor_cases])
        with torch.no_grad():
            recipient=native_states(self.model,x)
            donor=native_states(self.model,donors)
        plans=[]
        for j,goal in enumerate(GOALS):
            plan=label_blind_donor_plan(
                self.model,self.energy,recipient[2],donor[2][j:j+1],
                self.mu,self.std,goal=goal,
                seed=stable_seed("smoke",61060,goal))
            if not plan["eligible"]:
                plan["_source_donor_token"]=(donor[2][j,goal]-recipient[2][0,goal]).detach()
            plans.append(plan)
        donor_truth=[v[1]["donor_goal_label"] for v in donor_cases]
        rows=score_batch(self.model,self.energy,self.mu,self.std,
                         recipient[2],plans,donor_truth,depth=2,horizon=4)
        self.assertEqual(len(rows),8)
        for j,row in enumerate(rows):
            self.assertEqual(set(row["arms"]),set(ARMS))
            self.assertEqual(len(row["energy_gradient_goal"]),48)
            self.assertEqual(row["donor_goal_target"],donor_truth[j])
            self.assertEqual(len(row["clean"]["predicted_classes"]),81)
            for arm in ARMS:
                edit=row["arms"][arm]
                self.assertEqual(len(edit["edit_vector_float32"]),48)
                self.assertEqual(len(edit["predicted_classes"]),81)
                self.assertEqual(len(edit["goal_logits"]),10)
                self.assertEqual(edit["edit_sha256"],
                                 edit_sha(torch.tensor(edit["edit_vector_float32"])))
            with torch.no_grad():
                real=self.model.head(recipient[4])[0,GOALS[j]]
            torch.testing.assert_close(torch.tensor(row["clean"]["goal_logits"]),
                                       real,atol=1e-5,rtol=1e-5)
            row.update({
                "base_index":0,"source_goal_slot":j,
                "original_goal_target":int(self.y[0,GOALS[j]]),
                "clean_recipient_correct":bool(int(real.argmax())==int(self.y[0,GOALS[j]])),
                "clean_donor_correct":bool(int(
                    self.model.head(donor[4])[j,GOALS[j]].argmax())==donor_truth[j]),
                "clean_recipient4_goal_logits":real.tolist(),
                "clean_donor4_goal_logits":
                    self.model.head(donor[4])[j,GOALS[j]].detach().tolist(),
                "source_donor_goal_delta_l2":float(
                    (donor[2][j,GOALS[j]]-recipient[2][0,GOALS[j]]).norm()),
            })
            verify_case(row,{
                "base_index":0,
                "y":self.y[0].tolist(),
                "donors":[{"donor_goal_label":z[1]["donor_goal_label"]}
                          for z in donor_cases],
            },self.model.head.weight.detach().double().tolist(),2,GOALS[j])

    def test_bootstrap_aggregate_fixed_unit(self):
        rows=[]
        for base in range(64):
            for j,goal in enumerate(GOALS):
                arms={name:{"donor_target_correct":name=="projected_donor" or
                              name=="full_donor_token"}
                      for name in ARMS}
                rows.append({
                    "base_index":base,"goal":goal,"admitted":True,
                    "clean_recipient_correct":True,"clean_donor_correct":True,
                    "arms":arms,
                })
        result=summarize(rows,103,2)
        self.assertEqual(result["n_original_goals"],512)
        self.assertEqual(result["n_independent_base_sources"],64)
        self.assertAlmostEqual(result["ITT_paired_effect"],1.)
        self.assertEqual(result["whole_base_bootstrap95"],[1.,1.])
        self.assertTrue(result["teacher_engineering_gate"])
        self.assertEqual(audit_median([0]*8),median_binary([0]*8))

    def test_finite_and_complete_edit_sha(self):
        z=torch.arange(48,dtype=torch.float32)/10.
        self.assertEqual(len(edit_sha(z)),64)
        self.assertNotEqual(edit_sha(z),edit_sha(-z))


if __name__=="__main__":
    unittest.main()
