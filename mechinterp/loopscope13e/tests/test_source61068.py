"""Pre-source LGA-10E synthetic controls. No call to source seed 61068."""
from __future__ import annotations

import math
import unittest
import torch

from mechinterp.loopscope13e.source61068 import (
    ANGLES, DIRECTIONS, CurvatureSelectionFilter, donor_variations,
    delta_at, locked_source, norm, blank_arm,
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
        self.assertNotEqual(float(actual[0]),float(z[0]))

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
        promoted=lambda v:v*1.001
        filt=CurvatureSelectionFilter(SimplePhysical(h32),native,promoted,
                                     h32,0,src,0.0,"mixed")
        trial=src.clone()
        rec,_=filt(trial,[])
        self.assertTrue(rec["passed"])
        self.assertEqual(filt.calls["promoted_actual"],2)
        self.assertEqual(filt.calls["promoted_probe"],2)

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
