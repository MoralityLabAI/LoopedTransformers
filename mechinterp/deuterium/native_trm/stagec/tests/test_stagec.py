"""Stage C source-only tests on EXCLUDED untrained 61074 examples.

Never generates or scores protected 731057, reserved LOOPSCOPE 31059/64/65,
or first calibration 731055.
"""
from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from mechinterp.loopscope04.gradient_flow import TinyRecursiveModel, fixed_data
from mechinterp.deuterium.native_trm.stagec import score_native as s
from mechinterp.deuterium.native_trm.stagec import audit_native as a
from mechinterp.deuterium.native_trm.stagec.aggregate import aggregate


class TestNativeFirstScoringInstrument(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(61074)
        self.model=TinyRecursiveModel(48,4).eval().requires_grad_(False)
        self.x,self.y=fixed_data(4,61074)
        weight=self.model.block.chan_mix[2].weight
        g=torch.Generator().manual_seed(41074)
        d=torch.outer(torch.randn(48,generator=g),torch.randn(128,generator=g))
        self.dose=.01*float(weight.norm())
        self.edit=d/d.norm()*self.dose
        self.edits={"vpd":self.edit,"random":-self.edit,
                    "task_gradient":self.edit,"sham":torch.zeros_like(weight)}

    def test_original_four_pass_and_per_depth_identity(self):
        with torch.no_grad():
            clean=self.model(self.x)
            for k in range(4):
                local=s.local_one_pass(self.model,self.x,
                    torch.zeros_like(self.edit),k)
                self.assertTrue(torch.equal(clean,local))
            self.assertEqual(self.model.n_steps,4)

    def test_all_17_native_arms_and_restoration(self):
        old=self.model.block.chan_mix[2].weight.detach().clone()
        with torch.no_grad():
            clean,m=s.logits_for_arm(self.model,self.x,"clean",self.edits,self.dose)
            sham,_=s.logits_for_arm(self.model,self.x,"sham",self.edits,self.dose)
            self.assertTrue(torch.equal(clean,sham))
            self.assertEqual(len(s.ARMS),17)
            vpd,physical=s.logits_for_arm(self.model,self.x,"vpd",
                                          self.edits,self.dose)
            local,ldose=s.logits_for_arm(self.model,self.x,"vpd_depth2",
                                         self.edits,self.dose)
            self.assertEqual(physical["native_weight_uses"],4)
            self.assertEqual(ldose["native_weight_uses"],1)
            self.assertEqual(vpd.shape,(4,81,10))
            self.assertEqual(local.shape,(4,81,10))
            self.assertFalse(torch.equal(vpd,clean))
            self.assertTrue(torch.equal(old,self.model.block.chan_mix[2].weight))
            self.assertEqual(len(s.per_source_metrics(vpd,self.y)["goal_nll"]),4)

    def test_binary_auditor_categorical_and_cross_entropy(self):
        raw=[-1.5,.8,2.1]+[-3.]*7
        ce,pred=a.basic_scores(raw,2)
        torch_ce=float(F.cross_entropy(
            torch.tensor(raw).view(1,10),torch.tensor([2])))
        self.assertLess(abs(ce-torch_ce),1e-6)
        self.assertEqual(pred,1)

    def test_bootstrap_and_source_constants(self):
        self.assertEqual(s.TEST_SEED,731057)
        self.assertNotIn(31059,s.OLD_SOURCES)
        self.assertNotIn(31064,s.OLD_SOURCES)
        self.assertNotIn(31065,s.OLD_SOURCES)
        vals=[float(i%7)/100 for i in range(256)]
        first=s.percentile_bootstrap(vals,s.seed_rng("bootstrap",103))
        second=a.bootstrap(vals,a.bootstrap_seed(103))
        for a_value,b_value in zip(first["ci95"],second):
            self.assertAlmostEqual(a_value,b_value,places=12)

    def test_aggregator_requires_all_old_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            paths=[]
            for seed,diff in ((103,.1),(211,.09),(307,.11)):
                obj={
                    "schema":"DEUTERIUM-NATIVE-01/first-stdlib-heldout-audit-v1",
                    "status":"PASS","teacher_seed":seed,"torch_imported":False,
                    "comparison_completed":True,"clean_task_gate_pass":True,
                    "random_minus_vpd_goal_nll":diff,
                    "vpd_offgoal_accuracy_minus_clean":0,
                }
                p=root/f"{seed}.json"
                p.write_text(json.dumps(obj))
                paths.append(p)
            result=aggregate(paths,root/"summary.json")
            self.assertEqual(result["status"],"DEVELOPMENT_POSITIVE")
            self.assertTrue(result["registered_positive_rule_pass"])


if __name__=="__main__":
    unittest.main()
