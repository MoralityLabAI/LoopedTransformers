"""LOOPSCOPE-06 independent CPU instrumentation and source-isolation tests."""
from __future__ import annotations

import math
import unittest

import torch
from torch.nn import functional as F

from mechinterp.loopscope04.gradient_flow import TinyRecursiveModel,fixed_data
from mechinterp.loopscope06.fit_value import (
    HorizonValue,calibration_data,fit_one,protocol_sha,TRAIN_SEED,
    VAL_SEED,TEST_SEED
)
from mechinterp.loopscope06.eval_value import (
    sample_test,gradient_value,profile,summarize,native_interventions,
    summarize_causal,bootstrap,stable_seed,CONTRASTS,CAUSAL_ARMS
)
from mechinterp.loopscope06.audit_receipt import p95,seed as new_seed
from mechinterp.loopscope05.fit_and_freeze import TopKSAE


class TestsHorizonValue(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_registered_datasets_and_no_new_test_used_in_calibration(self):
        self.assertEqual(len(protocol_sha()),64)
        (x,y),(v,w),manifest=calibration_data()
        self.assertEqual(len(x),768)
        self.assertEqual(len(v),192)
        self.assertEqual(manifest["train"]["seed"],TRAIN_SEED)
        self.assertEqual(manifest["validation"]["seed"],VAL_SEED)
        self.assertIs(manifest["new_test_generated"],False)
        self.assertTrue(torch.equal(y,x.roll(1,1)))
        self.assertTrue(torch.equal(w,v.roll(1,1)))
        smoke,targets,meta=sample_test(smoke=True)
        self.assertEqual(meta["seed"],61050)
        self.assertNotEqual(meta["seed"],TEST_SEED)
        self.assertEqual(len(smoke),8)
        self.assertTrue(torch.equal(targets,smoke.roll(1,1)))

    def test_horizon_head_is_smooth_and_gradient_is_full_sequence(self):
        torch.manual_seed(123)
        net=HorizonValue()
        net.eval().requires_grad_(False)
        h=torch.randn((2,81,48),dtype=torch.float32)
        mu=torch.zeros(48)
        std=torch.ones(48)
        p,g=gradient_value(net,h,2,mu,std)
        self.assertEqual(tuple(p.shape),(2,81))
        self.assertEqual(tuple(g.shape),(2,81,48))
        self.assertTrue(bool((p>0).all()))
        self.assertTrue(bool(torch.isfinite(g).all()))
        with self.assertRaisesRegex(ValueError,"horizon"):
            gradient_value(net,h,7,mu,std)

    def test_autodiff_value_gradient_matches_true_finite_difference(self):
        net=HorizonValue().eval().requires_grad_(False)
        h=torch.randn((1,81,48),dtype=torch.float32)
        delta=torch.randn_like(h)*.05
        mu=torch.zeros(48)
        std=torch.ones(48)
        _,g=gradient_value(net,h,1,mu,std)
        eps=1e-3
        finite=float((net(h+eps*delta,1,mu,std).sum()-
                      net(h-eps*delta,1,mu,std).sum())/(2*eps))
        predicted=float((g*delta).sum())
        self.assertAlmostEqual(finite,predicted,delta=.03)

    def test_old_value_fit_validation_is_train_only_and_frozen(self):
        teacher=TinyRecursiveModel().eval().requires_grad_(False)
        x,y=fixed_data(24,31048)
        v,w=fixed_data(12,31049)
        first,source=fit_one(teacher,x,y,v,w,103,smoke=True)
        self.assertEqual(source["seed"],103)
        self.assertIn(source["selected_update"],(0,4,8))
        self.assertEqual(len(source["validation_grid"]),3)
        self.assertEqual(tuple(first["mu"].shape),(48,))
        self.assertEqual(tuple(first["std"].shape),(48,))
        self.assertGreater(first["constant_ce"],0.)
        self.assertFalse(torch.equal(first["selected"]["c1.weight"],torch.zeros_like(
            first["selected"]["c1.weight"])))

    def test_whole_sequence_bootstrap_matches_independent_auditor(self):
        vals=[float(i)/17 for i in range(-16,16)]
        base=stable_seed(43050,103,2,"gain_value_over_initial")
        self.assertEqual(base,new_seed(43050,103,2,"gain_value_over_initial"))
        # Different implementations share 2000 draws and full-source clustering.
        self.assertEqual(bootstrap(vals,base),p95(vals,base))

    def test_sae_controls_and_frozen_value_are_source_only(self):
        teacher=TinyRecursiveModel().eval().requires_grad_(False)
        net=HorizonValue().eval().requires_grad_(False)
        sae=TopKSAE().eval().requires_grad_(False)
        normal=(torch.zeros(48),torch.ones(48))
        saes={1:(sae,*normal),2:(sae,*normal)}
        x,y=fixed_data(2,61050)
        raw=native_interventions(teacher,net,*normal,saes,x,y,103,smoke=True)
        self.assertEqual(set(raw),{"1","2"})
        self.assertEqual(len(raw["1"]),2)
        self.assertEqual(len(raw["1"][0]["original_goals"]),8)
        self.assertIn(("value_down","random"),CONTRASTS)
        for dep in ("1","2"):
            for seq in raw[dep]:
                for goal in seq["original_goals"]:
                    if goal["admitted"]:
                        self.assertTrue(set(CAUSAL_ARMS)-{"oracle_down"} <= set(goal["arms"]))
                        for arm in goal["arms"].values():
                            self.assertLess(abs(arm["dose"]-goal["radius"]),1e-5)
                            self.assertEqual(len(arm["full_token_CE"]),81)
                            self.assertEqual(len(arm["goal_logits"]),10)
                            self.assertAlmostEqual(
                                arm["global_CE_change"],
                                arm["native_final_sequence_CE"]-
                                arm["clean_final_sequence_CE"],delta=1e-5)
        stats=summarize_causal(raw,103,smoke=True)
        self.assertEqual(set(stats),{"1","2"})
        self.assertEqual(stats["1"]["original_goals"],16)

    def test_profiles_keep_source_independent_sequence_denominators(self):
        teacher=TinyRecursiveModel().eval().requires_grad_(False)
        critic=HorizonValue().eval().requires_grad_(False)
        initial=HorizonValue().eval().requires_grad_(False)
        x,y=fixed_data(4,61050)
        rows=profile(teacher,critic,initial,torch.zeros(48),torch.ones(48),
                     const_ce=2.0,x=x,y=y)
        self.assertEqual(set(rows),{"0","1","2","3"})
        for depth in rows.values():
            self.assertEqual(len(depth),4)
            self.assertTrue(all(math.isfinite(z["gain_value_over_local"])
                                for z in depth))
        test,positive=summarize(rows,103,smoke=True)
        self.assertEqual(len(test),4)
        self.assertTrue(0<=positive<=4)
        self.assertEqual(test["0"]["n_whole_sequences"],4)


if __name__=="__main__":
    unittest.main()
