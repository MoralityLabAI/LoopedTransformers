"""Pre-source LGA-10E synthetic controls. No call to source seed 61068."""
from __future__ import annotations

import math
import unittest
import copy
import numpy as np
import torch

from mechinterp.loopscope04.gradient_flow import TinyRecursiveModel, fixed_data
from mechinterp.loopscope10.global_energy_fiber import full_head_kernel
from mechinterp.loopscope13b.source_feasibility import QuadraticObserver
from mechinterp.loopscope13c.audit_receipt import audit_native
from mechinterp.loopscope13e.source61068 import (
    ANGLES, DIRECTIONS, CurvatureSelectionFilter, donor_variations,
    delta_at, locked_source, norm, blank_arm, evaluate_case,
)


class SimplePhysical:
    def __init__(self, h32):
        self.h32=h32
    def roundtrip(self, intended):
        x=self.h32[0,[80,0]].reshape(96)
        return (x[None]+intended.float()).double()-x.double()[None]
    def __call__(self, trial, previous):
        return {"passed":True,"failed_checks":[]}, {"intended":trial.clone()}


class TestProspectiveEndpoint(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_dark_rotation_retains_margin_and_dose(self):
        R=torch.eye(96,dtype=torch.float64)[:,:10]
        V=torch.eye(96,dtype=torch.float64)[:,11:75]
        donor=torch.zeros(96,dtype=torch.float64)
        donor[0]=.1; donor[10]=.5
        variations=list(donor_variations(donor,R,V,61064,1,10))
        self.assertEqual(len(variations),1+DIRECTIONS*len(ANGLES))
        self.assertTrue(torch.equal(variations[0][1],donor))
        for idx,z,meta in variations:
            self.assertAlmostEqual(norm(z),norm(donor),delta=1e-13)
            self.assertAlmostEqual(norm(R.T@(z-donor)),0.,delta=1e-13)
            self.assertLessEqual(norm(z-donor),.2*norm(donor))
            if idx:
                self.assertIn(meta["angle"],ANGLES)

    def test_deterministic_candidate_order(self):
        R=torch.eye(96,dtype=torch.float64)[:,:10]
        V=torch.eye(96,dtype=torch.float64)[:,11:75]
        z=torch.zeros(96,dtype=torch.float64); z[1]=.1; z[10]=.6
        xs=list(donor_variations(z,R,V,61064,2,20))
        ys=list(donor_variations(z,R,V,61064,2,20))
        self.assertTrue(all(torch.equal(x[1],y[1]) for x,y in zip(xs,ys)))

    def test_cast_addition_is_physical_fp32(self):
        h32=torch.zeros((1,81,48),dtype=torch.float32)
        h32[0,80,0]=1.5
        z=torch.full((1,96),1e-7,dtype=torch.float64)
        actual=delta_at(h32,0,z)[0]
        expected=((h32[0,[80,0]].reshape(96)+z[0].float()).double()-
                  h32[0,[80,0]].reshape(96).double())
        self.assertTrue(torch.equal(actual,expected))
        self.assertNotEqual(float(actual[0]),float(z[0,0]))

    def test_fp32_filter_does_not_use_fp64_selection_oracle(self):
        h32=torch.zeros((1,81,48),dtype=torch.float32)
        src=torch.zeros(96,dtype=torch.float64);src[0]=0.01
        count={"native":0,"promoted":0}
        def native(v):
            count["native"]+=1
            return v*1.0
        def promoted(v):
            count["promoted"]+=1
            raise AssertionError("FP64 oracle accessed during fp32 filter")
        filt=CurvatureSelectionFilter(SimplePhysical(h32),native,promoted,
                        h32,0,src,0.0,"fp32")
        trial=src.clone()
        rec,_=filt(trial,[])
        self.assertTrue(rec["passed"])
        self.assertEqual(count["promoted"],0)
        trial*=2
        rec,_=filt(trial,[])
        self.assertFalse(rec["passed"])
        self.assertIn("FP32_NATIVE_PARITY",rec["failed_checks"])

    def test_repaired_filter_checks_promoted_resolution(self):
        h32=torch.zeros((1,81,48),dtype=torch.float32)
        src=torch.zeros(96,dtype=torch.float64);src[1]=0.01
        native=lambda v:v*1.0
        promoted=lambda v:v*1.00001
        filt=CurvatureSelectionFilter(SimplePhysical(h32),native,promoted,
                                     h32,0,src,0.0,"mixed")
        trial=src.clone()
        rec,_=filt(trial,[])
        self.assertTrue(rec["passed"])
        self.assertEqual(filt.calls["promoted_actual"],2)
        self.assertEqual(filt.calls["promoted_probe"],2)

    def test_exposed_61064_one_case_end_to_end(self):
        """All new selection and native gate calls are exercised on old data."""
        torch.manual_seed(61064)
        model32=TinyRecursiveModel().float().eval().requires_grad_(False)
        model64=copy.deepcopy(model32).double()
        data,labels=fixed_data(1,61064)
        with torch.no_grad():
            h32=model32.reason_states(data)[2]
            h64=model64.reason_states(data)[2]
        hp=full_head_kernel(model64.head.weight).projector
        arr={}
        case=evaluate_case(model32,model64,h32,h64,data,hp,
                           QuadraticObserver(),
                           torch.zeros(48,dtype=torch.float64),
                           torch.ones(48,dtype=torch.float64),0,0,arr)
        self.assertTrue(case["source_eligible"],case.get("exception"))
        self.assertIsNone(case.get("exception"),case)
        self.assertTrue(case.get("geometry_eligible"),case.get("geometry_reason"))
        self.assertEqual(set(case["arms"]),{
            "frozen_baseline","fp32_sham_filter","mixed_donor_repair"})
        self.assertIn("case_00_00__common__target_hvp64",arr)
        for name,arm in case["arms"].items():
            if arm["complete"]:
                self.assertEqual(arm["n_shams"],8)
                self.assertEqual(arm["native"]["n_shams"],8)
                self.assertEqual(len(arm["native"]["arms"]),9)
                self.assertIn("case_00_00__"+name+"__native__hvp32",arr)
                intended=np.concatenate([
                    np.asarray(arr["case_00_00__"+name+"__selected"]).reshape(1,96),
                    np.asarray(arr["case_00_00__"+name+"__shams"])],axis=0)
                target=case["target_curvature_original"]
                prefix="case_00_00__"+name+"__native__"
                get=lambda key: np.asarray(arr[prefix+key])
                def check(value,message):
                    self.assertTrue(value,message)
                audited=audit_native(
                    arm["native"],get,check,target=target,
                    intended=intended,h32=h32.detach().numpy(),
                    h64=h64.detach().numpy(),
                    reference_j=np.asarray(arr["case_00_00__common__reference_jacobian64"]),
                    head_projector=hp.detach().numpy(),
                    head=model64.head.weight.detach().numpy())
                self.assertEqual(audited["native_ready"],arm["native_fp32_ready"])

    def test_freeze_fails_closed(self):
        with self.assertRaisesRegex(ValueError,"full implementation SHA"):
            locked_source("61068")
        with self.assertRaisesRegex(ValueError,"full implementation SHA"):
            locked_source("not_a_commit")

    def test_partial_or_exception_never_sets_success(self):
        arm=blank_arm()
        self.assertFalse(arm["complete"])
        self.assertFalse(arm["native_fp32_ready"])
        self.assertFalse(arm["source_preserved"])


if __name__=="__main__":
    unittest.main()
