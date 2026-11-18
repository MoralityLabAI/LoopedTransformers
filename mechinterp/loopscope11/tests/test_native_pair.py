"""LGA-08 end-to-end excluded 61062 native and independent no-Torch audit tests."""
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
from mechinterp.loopscope11.pair_fiber import GOALS,ARMS,EXCLUDED_SEED
from mechinterp.loopscope11.native_pair import (
    EDIT_BYTES,SOURCE_BYTES,score_batch,summarize,median8
)
from mechinterp.loopscope11.audit_pair import (
    audit_case,median_binary as audit_median
)


class QuadraticEnergy(torch.nn.Module):
    def forward(self,h,k,mu,std):
        return .5*((h-mu)/std).square().sum(-1)


class NativePairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(EXCLUDED_SEED)
        cls.model=TinyRecursiveModel().eval().requires_grad_(False)
        cls.energy=QuadraticEnergy().eval().requires_grad_(False)
        cls.x,cls.y=fixed_data(1,EXCLUDED_SEED)
        cls.mu=torch.zeros(48)
        cls.std=torch.ones(48)
        cls.head=full_head_kernel(cls.model.head.weight)

    def test_excluded_first_source_native_instrument_and_stdlib_audit(self):
        donor_list=[legal_predecessor_donor(self.x[0],goal) for goal in GOALS]
        donors=torch.stack([x[0] for x in donor_list])
        info=[{"goal":goal,"donor_x":donor.tolist(),**meta}
              for goal,(donor,meta) in zip(GOALS,donor_list)]
        with torch.no_grad():
            states=self.model.reason_states(self.x)
            ds=self.model.reason_states(donors)
            rec4=self.model.head(states[4])
            don4=self.model.head(ds[4])
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)
            ev=directory/"edits.gz";es=directory/"sources.gz"
            edigest=hashlib.sha256()
            sdigest=hashlib.sha256()
            with gzip.open(ev,"wb") as edit_file,gzip.open(es,"wb") as source_file:
                cases,nedit,nsource=score_batch(
                    self.model,self.energy,self.mu,self.std,
                    self.head,states[2],ds[2],info,
                    sid=103,base_id=0,edit_file=edit_file,
                    source_file=source_file,edit_digest=edigest,source_digest=sdigest,
                    edit_index=0,source_index=0,clean_rec4=rec4,clean_don4=don4)
            self.assertEqual((len(cases),nedit,nsource),(8,8*len(ARMS),8))
            source_case={"base_index":0,"x":self.x[0].tolist(),
                         "y":self.y[0].tolist(),"donors":info}
            checked_hash=hashlib.sha256()
            with gzip.open(ev,"rb") as edit_file,gzip.open(es,"rb") as source_file:
                idx=0
                for c in cases:
                    self.assertEqual(c["fiber_dimension"],75)
                    self.assertTrue(c["admitted"])
                    self.assertEqual(set(c["arms"]),set(ARMS))
                    self.assertLess(c["full_donor_max_logit_error"],1e-5)
                    raw=source_file.read(SOURCE_BYTES)
                    idx=audit_case(c,source_case,
                                   self.head.projector.new_tensor(self.model.head.weight).double().tolist(),
                                   self.head.projector.tolist(),raw,
                                   edit_file,checked_hash,idx)
                self.assertEqual(idx,nedit)
                self.assertEqual(checked_hash.hexdigest(),edigest.hexdigest())
                self.assertEqual(edit_file.read(1),b"")
                self.assertEqual(source_file.read(1),b"")
            with gzip.open(ev,"rb") as handle:
                self.assertEqual(len(handle.read()),nedit*EDIT_BYTES)
            with gzip.open(es,"rb") as handle:
                self.assertEqual(len(handle.read()),nsource*SOURCE_BYTES)

    def test_matched_sham_is_within_correct_registered_family(self):
        self.assertEqual(len(ARMS),23)
        self.assertEqual(median8([0,0,0,0,1,1,1,1]),.5)
        self.assertEqual(audit_median([0,0,0,0,1,1,1,1]),.5)

    def test_root_cluster_aggregation_and_first_registered_strong_gate(self):
        cases=[]
        for i in range(64):
            for goal in GOALS:
                arms={name:{"donor_correct":name in (
                       "pair_energy_tangent","pair_natural","full_donor_state")}
                       for name in ARMS}
                cases.append({
                    "base_index":i,"goal":goal,"admitted":True,
                    "joint_clean_competence":True,"arms":arms,
                    "full_donor_max_logit_error":0.,
                })
        s=summarize(cases,103)
        self.assertEqual(s["n_original_source_goals"],512)
        self.assertEqual(s["n_jointly_competent"],512)
        self.assertEqual(s["contrasts"]["gradient"]["paired_ITT_effect"],1.)
        self.assertEqual(s["contrasts"]["random"]["paired_ITT_effect"],1.)
        self.assertTrue(s["registered_strong_teacher_gate"])
        for c in cases:
            c["arms"]["pair_gradient_match_0"]["donor_correct"]=True
            c["arms"]["pair_gradient_match_1"]["donor_correct"]=True
            c["arms"]["pair_gradient_match_2"]["donor_correct"]=True
            c["arms"]["pair_gradient_match_3"]["donor_correct"]=True
            c["arms"]["pair_gradient_match_4"]["donor_correct"]=True
        changed=summarize(cases,103)
        self.assertEqual(changed["contrasts"]["gradient"]["paired_ITT_effect"],0.)
        self.assertFalse(changed["registered_strong_teacher_gate"])


if __name__=="__main__":
    unittest.main()
