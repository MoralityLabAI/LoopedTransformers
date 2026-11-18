"""LGA-05: original frozen field, same-energy monotonicity and native controls."""
from __future__ import annotations
import math
import unittest
import torch
from mechinterp.loopscope04.gradient_flow import TinyRecursiveModel,fixed_data
from mechinterp.loopscope05.fit_and_freeze import TopKSAE
from mechinterp.loopscope06.fit_value import HorizonValue
from mechinterp.loopscope08.fit_potential import (
    protocol_sha,old_calibration,trajectory_cache,potential_gradient,
    fit_one,ARMS as TRAIN_ARMS,UPDATES
)
from mechinterp.loopscope08.eval_energy import (
    new_test,profile,summarize_profiles,native,causal_summary,
    curl_witness,bootstrap,seed,ARMS as CAUSAL_ARMS,
    radial_gradient,radial_energy
)
from mechinterp.loopscope08.audit_receipt import (
    bootstrap as stdlib_bootstrap,stable_seed,cyclic,sha_tensor
)


class TestsNativeEnergy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_registered_source_and_reserved_test_seeds(self):
        self.assertEqual(len(protocol_sha()),64)
        self.assertEqual(UPDATES,200)
        (x,y),(v,w),source=old_calibration(smoke=True)
        self.assertIs(source["new_test_generated"],False)
        self.assertIs(source["new_test_labels_available"],False)
        self.assertEqual(source["new_test_seed_reserved"],31057)
        self.assertEqual(len(x),16)
        self.assertTrue(torch.equal(y,x.roll(1,1)))
        self.assertTrue(torch.equal(w,v.roll(1,1)))
        test_x,test_y,manifest=new_test(smoke=True)
        self.assertEqual(manifest["seed"],61057)
        self.assertEqual(len(test_x),8)
        self.assertTrue(torch.equal(test_y,test_x.roll(1,1)))
        self.assertNotEqual(manifest["x_sha256"],source["train"]["x_sha256"])

    def test_value_gradient_matches_finite_difference(self):
        torch.manual_seed(77)
        model=HorizonValue().eval().requires_grad_(False)
        h=torch.randn(2,81,48)
        mu=torch.zeros(48);std=torch.ones(48)
        out,g=potential_gradient(model,h,0,mu,std)
        self.assertEqual(out.shape,(2,81))
        self.assertEqual(g.shape,h.shape)
        direction=torch.randn_like(h)*.01
        eps=.003
        numeric=float((model(h+eps*direction,0,mu,std).sum()-
                       model(h-eps*direction,0,mu,std).sum())/(2*eps))
        prediction=float((g*direction).sum())
        self.assertAlmostEqual(numeric,prediction,delta=.015)

    def test_shared_scalar_potential_uses_same_horizon(self):
        torch.manual_seed(14)
        model=HorizonValue().eval().requires_grad_(False)
        h=torch.randn(2,81,48)
        v=torch.randn_like(h)*.07
        mu=torch.zeros(48);std=torch.ones(48)
        energy_before=model(h,0,mu,std).sum(-1)
        energy_after=model(h+v,0,mu,std).sum(-1)
        measured=(energy_before-energy_after)/81
        self.assertEqual(measured.shape,(2,))
        gradient=radial_gradient(h,mu,std)
        self.assertTrue(torch.allclose(gradient,h))
        self.assertTrue(torch.allclose(radial_energy(h,mu,std),h.square().sum((1,2))*.5))

    def test_source_rollout_has_all_five_states_and_fits_two_arms(self):
        teacher=TinyRecursiveModel().eval().requires_grad_(False)
        tx,_=fixed_data(10,31055)
        vx,_=fixed_data(5,31056)
        train=trajectory_cache(teacher,tx)
        val=trajectory_cache(teacher,vx)
        self.assertEqual(train.shape,(5,10,81,48))
        saved,receipt=fit_one(103,teacher,train,val,smoke=True)
        self.assertEqual(set(saved["models"]),set(TRAIN_ARMS))
        self.assertEqual(set(receipt["arms"]),set(TRAIN_ARMS))
        self.assertEqual(receipt["arms"]["shared"]["optimizer_steps"],3)
        self.assertEqual(receipt["arms"]["shared"]["same_initial_weights_sha256"],
                         receipt["arms"]["conditioned"]["same_initial_weights_sha256"])
        self.assertEqual([x["update"] for x in receipt["arms"]["shared"]["validation_grid"]],[0,3])
        self.assertGreater(receipt["native_RMS"],0)

    @staticmethod
    def fixture_models():
        torch.manual_seed(123)
        teacher=TinyRecursiveModel().eval().requires_grad_(False)
        a=HorizonValue().eval().requires_grad_(False)
        b=HorizonValue().eval().requires_grad_(False)
        initial=HorizonValue().eval().requires_grad_(False)
        normal=(torch.zeros(48),torch.ones(48))
        return teacher,{"shared":a,"conditioned":b},initial,normal

    def test_new_source_profile_uses_81_tokens_per_sequence(self):
        teacher,nets,init,(mu,std)=self.fixture_models()
        x,y=fixed_data(4,61057)
        result=profile(teacher,nets,init,mu,std,x,y)
        self.assertEqual(set(result),{"0","1","2","3"})
        for row in result.values():
            self.assertEqual(len(row),4)
            for case in row:
                self.assertAlmostEqual(
                    case["shared_energy_drop"],
                    (case["shared_energy_before"]-case["shared_energy_after"])/81,
                    places=6)
                self.assertIn(case["shared_positive_drop"],(0.,1.))
                self.assertTrue(math.isfinite(case["shared_cosine"]))
        summary,n,all_monotone=summarize_profiles(result,103,smoke=True)
        self.assertEqual(len(summary),4)
        self.assertTrue(0<=n<=4)
        self.assertTrue(0<=all_monotone<=1.)

    def test_full_sequence_bootstrap_matches_independent_stdlib(self):
        scores=[i*.019 for i in range(-25,25)]
        root=seed(45057,103,1,"gain_shared_over_initial")
        self.assertEqual(root,stable_seed(45057,103,1,"gain_shared_over_initial"))
        self.assertEqual(bootstrap(scores,root),stdlib_bootstrap(scores,root))

    def test_cyclic_source_byte_manifest_readable_without_torch(self):
        x,y=fixed_data(3,31055)
        manifest={"x":x.tolist(),"y":y.tolist(),
                  "x_sha256":sha_tensor(x.tolist()),"y_sha256":sha_tensor(y.tolist())}
        original=cyclic(manifest,3)
        self.assertEqual(len(original),3)
        manifest["y"][0][0]=10
        with self.assertRaisesRegex(ValueError,"cyclic"):
            cyclic(manifest,3)

    def test_native_source_arms_and_descriptive_curl_are_complete(self):
        teacher,nets,init,(mu,std)=self.fixture_models()
        sae=TopKSAE().eval().requires_grad_(False)
        old_sae={k:(sae,mu,std) for k in (1,2)}
        x,y=fixed_data(2,61057)
        examples=native(teacher,nets,init,mu,std,old_sae,x,y,103,smoke=True)
        self.assertEqual(set(examples),{"1","2"})
        for dep in ("1","2"):
            self.assertEqual(len(examples[dep]),2)
            for group in examples[dep]:
                self.assertEqual(len(group["goals"]),8)
                for cell in group["goals"]:
                    if cell["source_admitted"]:
                        self.assertTrue(set(CAUSAL_ARMS)-{"oracle_down"}<=set(cell["arms"]))
                        for arm in cell["arms"].values():
                            self.assertAlmostEqual(arm["edit_L2"],cell["dose"],delta=1e-5)
                            self.assertEqual(len(arm["all_final_token_CE"]),81)
                            self.assertEqual(len(arm["final_goal_logits"]),10)
                            self.assertAlmostEqual(
                                arm["physical_edit_E_decrease"],
                                (arm["shared_E_before"]-arm["shared_E_after"])/81,
                                delta=1e-5)
        summary=causal_summary(examples,103,smoke=True)
        self.assertEqual(summary["1"]["n_original_goals"],16)
        curl=curl_witness(teacher,x,103,smoke=True)
        self.assertEqual(set(curl),{"1","2"})
        self.assertEqual(curl["1"]["n_source_sequences"],2)
        self.assertTrue(all(math.isfinite(x["skew_ratio"])
                            for x in curl["1"]["per_sequence"]))


if __name__=="__main__":
    unittest.main()
