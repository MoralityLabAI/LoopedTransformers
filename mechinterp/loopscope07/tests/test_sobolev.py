"""LGA-04 CPU source-only tests, with smoke cases using reserve seed 61054."""
from __future__ import annotations
import math
import unittest
import torch
from mechinterp.loopscope04.gradient_flow import TinyRecursiveModel,fixed_data
from mechinterp.loopscope06.fit_value import HorizonValue
from mechinterp.loopscope07.train_sobolev import (
    calibration_cases,source_cache,fit_pair,protocol_sha,ARMS,STEPS
)
from mechinterp.loopscope07.eval_sobolev import (
    fresh_test,profiles,summarize_profiles,native_interventions,
    summarize_native,bootstrap,seed
)
from mechinterp.loopscope07.audit_receipt import (
    bootstrap as stdlib_bootstrap,stable,cyclic_source
)
from mechinterp.loopscope05.fit_and_freeze import TopKSAE


class TestsSobolevCritic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_protocol_and_source_calibration_disjointness(self):
        self.assertEqual(len(protocol_sha()),64)
        self.assertEqual(STEPS,100)
        (x,y),(v,z),m=calibration_cases(smoke=True)
        self.assertTrue(torch.equal(y,x.roll(1,1)))
        self.assertTrue(torch.equal(z,v.roll(1,1)))
        self.assertIs(m["new_test_generated"],False)
        self.assertEqual(m["reserved_test_seed"],31054)
        self.assertEqual(len(x),16)
        x2,y2,test=fresh_test(smoke=True)
        self.assertEqual(test["seed"],61054)
        self.assertTrue(torch.equal(y2,x2.roll(1,1)))

    def test_cached_terminal_gradient_has_full_sequence_geometry(self):
        teacher=TinyRecursiveModel().eval().requires_grad_(False)
        x,y=fixed_data(4,31052)
        cache=source_cache(teacher,x,y)
        self.assertEqual(tuple(cache["h"].shape),(4,4,81,48))
        self.assertEqual(tuple(cache["true_grad"].shape),(4,4,81,48))
        self.assertEqual(tuple(cache["target"].shape),(4,81))
        self.assertTrue(bool(torch.isfinite(cache["true_grad"]).all()))

    def test_paired_critic_source_training_same_init(self):
        torch.manual_seed(83)
        teacher=TinyRecursiveModel().eval().requires_grad_(False)
        original=HorizonValue().eval().requires_grad_(False)
        tx,ty=fixed_data(12,31052)
        vx,vy=fixed_data(8,31053)
        train=source_cache(teacher,tx,ty)
        val=source_cache(teacher,vx,vy)
        model,receipts,rms,start=fit_pair(
            teacher,original,torch.zeros(48),torch.ones(48),
            train,val,103,smoke=True)
        self.assertEqual(set(model),set(ARMS))
        self.assertEqual(len(receipts),2)
        self.assertEqual(receipts[0]["starting_weight_sha256"],
                         receipts[1]["starting_weight_sha256"])
        self.assertEqual(receipts[0]["optimizer_updates"],3)
        self.assertTrue(bool(torch.isfinite(rms)))
        self.assertEqual(len(start),len(original.state_dict()))
        for r in receipts:
            self.assertEqual(len(r["validation_checkpoints"]),2)
            self.assertTrue(math.isfinite(
                r["validation_checkpoints"][-1]["normalized_gradient_MSE"]))

    def test_cluster_bootstrap_matches_independent_standard_library(self):
        vals=[x*.013 for x in range(-20,20)]
        root=seed(44054,103,1,"gain_over_valueonly")
        self.assertEqual(root,stable(44054,103,1,"gain_over_valueonly"))
        self.assertEqual(bootstrap(vals,root),stdlib_bootstrap(vals,root))

    def test_critic_gradient_holdout_profile_and_registered_depth(self):
        teacher=TinyRecursiveModel().eval().requires_grad_(False)
        critic=HorizonValue().eval().requires_grad_(False)
        v=HorizonValue().eval().requires_grad_(False)
        prior=HorizonValue().eval().requires_grad_(False)
        x,y=fixed_data(4,61054)
        rows=profiles(teacher,{"value_plus_gradient":critic,
                               "value_only_continue":v,"original":prior},
                      torch.zeros(48),torch.ones(48),torch.tensor(1.),x,y)
        self.assertEqual(set(rows),{"0","1","2","3"})
        self.assertEqual(len(rows["2"]),4)
        results,passed=summarize_profiles(rows,103,smoke=True)
        self.assertEqual(len(results),4)
        self.assertTrue(0<=passed<=4)

    def test_native_matched_physical_controls_and_full_CE(self):
        teacher=TinyRecursiveModel().eval().requires_grad_(False)
        v=HorizonValue().eval().requires_grad_(False)
        networks={"value_plus_gradient":v,"value_only_continue":v,
                  "original":v}
        sae=TopKSAE().eval().requires_grad_(False)
        saes={1:(sae,torch.zeros(48),torch.ones(48)),
              2:(sae,torch.zeros(48),torch.ones(48))}
        x,y=fixed_data(2,61054)
        cases=native_interventions(teacher,networks,
                                  torch.zeros(48),torch.ones(48),saes,
                                  x,y,103,smoke=True)
        self.assertEqual(set(cases),{"1","2"})
        for depth in cases:
            for group in cases[depth]:
                self.assertEqual(len(group["goal_plan"]),8)
                for plan in group["goal_plan"]:
                    if plan["preoutcome_admitted"]:
                        self.assertTrue({"sobolev_down","valueonly_down","random"} <= set(plan["arms"]))
                        for z in plan["arms"].values():
                            self.assertLess(abs(z["dose"]-plan["radius"]),1e-5)
                            self.assertEqual(len(z["all_final_token_CE"]),81)
                            self.assertEqual(len(z["goal_logits"]),10)
        summaries=summarize_native(cases,103,smoke=True)
        self.assertEqual(summaries["1"]["n_original_goal_slots"],16)

    def test_stdlib_cyclic_source_checks_original_81_cells(self):
        x,y=fixed_data(3,31052)
        cyclic_source(x.tolist(),y.tolist(),3)
        corrupt=y.tolist()
        corrupt[0][0]=10
        with self.assertRaisesRegex(ValueError,"cyclic"):
            cyclic_source(x.tolist(),corrupt,3)


if __name__=="__main__":
    unittest.main()
