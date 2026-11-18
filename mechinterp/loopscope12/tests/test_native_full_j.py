"""LGA-09 end-to-end excluded 61063 native source-only run and independent audit.

No original pretrained teacher/potential weights, and NO reserved 31063 data.
"""
from __future__ import annotations
import gzip
import hashlib
from pathlib import Path
import tempfile
import unittest

import torch

from mechinterp.loopscope04.gradient_flow import TinyRecursiveModel,fixed_data
from mechinterp.loopscope10.donor_source import legal_predecessor_donor
from mechinterp.loopscope10.global_energy_fiber import full_head_kernel
from mechinterp.loopscope12.full_jacobian import ARMS,GOALS,SMOKE_SEED
from mechinterp.loopscope12.native_full_j import (
    EDIT_BYTES,SOURCE_BYTES,score_base,summary,median8
)
from mechinterp.loopscope12.audit_full_j import audit_case,median8 as audit_median


class QuadraticPotential(torch.nn.Module):
    def forward(self,h,k,mu,std):
        return .5*((h-mu)/std).square().sum(-1)


class FirstNativeFullJTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(SMOKE_SEED)
        cls.teacher=TinyRecursiveModel().eval().requires_grad_(False)
        cls.energy=QuadraticPotential().eval().requires_grad_(False)
        cls.head=full_head_kernel(cls.teacher.head.weight)
        cls.x,cls.y=fixed_data(1,SMOKE_SEED)
        cls.mu=torch.zeros(48)
        cls.std=torch.ones(48)

    def test_actual_native_31_arms_and_independent_raw_audit(self):
        donors=[legal_predecessor_donor(self.x[0],goal) for goal in GOALS]
        inputs=torch.stack([x for x,_ in donors])
        info=[{"goal":goal,"donor_x":x.tolist(),**meta}
              for goal,(x,meta) in zip(GOALS,donors)]
        with torch.no_grad():
            recipient=self.teacher.reason_states(self.x)
            donor_states=self.teacher.reason_states(inputs)
            rec4=self.teacher.head(recipient[4])
            don4=self.teacher.head(donor_states[4])
        with tempfile.TemporaryDirectory() as tmp:
            edir=Path(tmp)
            edits=edir/"edits.gz"
            sources=edir/"sources.gz"
            h_edit=hashlib.sha256();h_source=hashlib.sha256()
            with gzip.open(edits,"wb") as edit_f,gzip.open(sources,"wb") as source_f:
                cases,nedit,nsource=score_base(
                    self.teacher,self.energy,self.mu,self.std,self.head,
                    recipient[2],donor_states[2],info,rec4,don4,
                    sid=103,base=0,
                    edit_writer=edit_f,source_writer=source_f,
                    edit_hash=h_edit,source_hash=h_source,
                    edit_index=0,source_index=0)
            self.assertEqual((len(cases),nedit,nsource),(8,8*len(ARMS),8))
            self.assertEqual(len(ARMS),31)
            source_case={"base_index":0,"x":self.x[0].tolist(),
                         "y":self.y[0].tolist(),"donors":info}
            independent_edit=hashlib.sha256()
            independent_source=hashlib.sha256()
            with gzip.open(edits,"rb") as edit_f,gzip.open(sources,"rb") as source_f:
                idx=0
                for j,case in enumerate(cases):
                    self.assertEqual(case["pair_fiber_dimension"],75)
                    self.assertTrue(case["admitted"],case["reason"])
                    self.assertGreaterEqual(case["future_dark_dimension"],65)
                    self.assertLessEqual(case["full_future_jacobian_rank"],10)
                    self.assertEqual(len(case["arms"]),len(ARMS))
                    self.assertLess(case["full_donor_max_logit_error"],1e-5)
                    raw=source_f.read(SOURCE_BYTES)
                    independent_source.update(raw)
                    self.assertEqual(case["source_index"],j)
                    idx=audit_case(
                        case,source_case,
                        self.teacher.head.weight.detach().double().tolist(),
                        self.head.projector.tolist(),
                        raw,edit_f,independent_edit,idx)
                self.assertEqual(idx,nedit)
                self.assertEqual(edit_f.read(1),b"")
                self.assertEqual(source_f.read(1),b"")
            self.assertEqual(independent_edit.hexdigest(),h_edit.hexdigest())
            self.assertEqual(independent_source.hexdigest(),h_source.hexdigest())
            with gzip.open(edits,"rb") as h:
                self.assertEqual(len(h.read()),nedit*EDIT_BYTES)
            with gzip.open(sources,"rb") as h:
                self.assertEqual(len(h.read()),nsource*SOURCE_BYTES)

    def test_original_64_base_root_gate_is_clustered(self):
        cases=[]
        for base in range(64):
            for j,goal in enumerate(GOALS):
                arms={arm:{"donor_correct":arm in (
                    "pair_donor_k","natural_pair","full_donor_state")}
                    for arm in ARMS}
                cases.append({
                    "base_index":base,"goal_slot":j,"goal":goal,
                    "admitted":True,"joint_clean_competence":True,
                    "arms":arms,"full_donor_max_logit_error":0.,
                })
        s=summary(cases,103)
        self.assertEqual(s["n_original_source_goals"],512)
        self.assertEqual(s["n_admitted"],512)
        for key in ("full_j","margin","random"):
            self.assertEqual(s["contrasts"][key]["paired_ITT_effect"],1.)
            self.assertEqual(s["contrasts"][key]["whole_base_bootstrap95"],[1.,1.])
        self.assertTrue(s["teacher_strong_full_J_gate"])
        for c in cases:
            for i in range(5):
                c["arms"][f"full_j_sham_{i}"]["donor_correct"]=True
        null=summary(cases,103)
        self.assertEqual(null["contrasts"]["full_j"]["paired_ITT_effect"],0.)
        self.assertFalse(null["teacher_strong_full_J_gate"])
        self.assertEqual(median8([0,0,0,0,1,1,1,1]),audit_median([0,0,0,0,1,1,1,1]))


if __name__=="__main__":
    unittest.main()
