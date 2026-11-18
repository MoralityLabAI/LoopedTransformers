"""LOOPSCOPE-09 / LGA-06 CPU tests. Uses seed 61058, never test seed 31058."""
from __future__ import annotations
import math
import unittest
import torch
from mechinterp.loopscope04.gradient_flow import TinyRecursiveModel,fixed_data
from mechinterp.loopscope05.fit_and_freeze import TopKSAE
from mechinterp.loopscope06.fit_value import HorizonValue
from mechinterp.loopscope09.eval_shift import (
    protocol_sha,create_source,trajectories,summarize,native_edits,
    causal_summary,curl_witness,stable_seed,bootstrap,ARMS,N_LOOPS,
    GOALS,SMOKE_SEED
)
from mechinterp.loopscope09.audit_receipt import (
    stable_seed as independent_seed,bootstrap as independent_bootstrap,
    source_cyclic,sha_int64
)


class TestsFrozenRecurrentEnergyShift(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_registration_and_outcome_blind_fresh_source_reservation(self):
        self.assertEqual(len(protocol_sha()),64)
        self.assertEqual(N_LOOPS,12)
        self.assertEqual(ARMS,("iid","period9","block9","zero20"))
        base,families,manifest=create_source(seed=SMOKE_SEED,n=8)
        self.assertEqual(manifest["seed"],61058)
        self.assertEqual(tuple(base.shape),(8,81))
        self.assertNotEqual(manifest["seed"],31058)
        self.assertEqual(len(families),4)
        for name,(x,y) in families.items():
            self.assertTrue(torch.equal(y,x.roll(1,1)))
            self.assertEqual(tuple(x.shape),(8,81))
            self.assertEqual(manifest["families"][name]["n"],8)

    def test_deterministic_shift_families_and_label_zero_support(self):
        base,families,manifest=create_source(seed=SMOKE_SEED,n=8)
        iid=families["iid"][0]
        period=families["period9"][0]
        block=families["block9"][0]
        zeros=families["zero20"][0]
        self.assertTrue(torch.equal(iid,base))
        self.assertTrue(torch.equal(period,base[:,:9].repeat(1,9)))
        self.assertTrue(torch.equal(block,base[:,::9].repeat_interleave(9,dim=1)))
        self.assertTrue(bool((zeros[:,::5]==0).all()))
        self.assertEqual(int((zeros==0).sum()),8*17)
        self.assertEqual(manifest["families"]["zero20"]["n_zero_input_tokens"],136)
        self.assertTrue(bool((families["zero20"][1]==0).any()))

    def test_independent_standard_library_reconstructs_source_shas(self):
        _,families,manifest=create_source(seed=SMOKE_SEED,n=8)
        for name,(x,y) in families.items():
            source_cyclic(x.tolist(),y.tolist())
            self.assertEqual(sha_int64(x.tolist()),manifest["families"][name]["x_sha256"])
            self.assertEqual(sha_int64(y.tolist()),manifest["families"][name]["y_sha256"])

    def test_potential_energy_is_same_scalar_on_both_sides(self):
        torch.manual_seed(8001)
        m=TinyRecursiveModel().eval().requires_grad_(False)
        shared=HorizonValue().eval().requires_grad_(False)
        conditional=HorizonValue().eval().requires_grad_(False)
        initial=HorizonValue().eval().requires_grad_(False)
        normal=(torch.zeros(48),torch.ones(48))
        x,y=fixed_data(2,61058)
        rows=trajectories(m,{"shared":shared,"conditioned":conditional},
                          initial,*normal,x,y,chunk=2)
        self.assertEqual(set(rows),{str(k) for k in range(12)})
        for k in range(12):
            self.assertEqual(len(rows[str(k)]),2)
            for case in rows[str(k)]:
                self.assertAlmostEqual(
                    case["shared_energy_drop"],
                    (case["shared_E_before"]-case["shared_E_after"])/81,places=7)
                self.assertEqual(case["shared_positive_drop"],
                                 float(case["shared_E_before"]>case["shared_E_after"]))
        for k in range(11):
            for i in range(2):
                self.assertAlmostEqual(rows[str(k)][i]["shared_E_after"],
                                       rows[str(k+1)][i]["shared_E_before"],places=8)
        summary=summarize(rows,103,"iid",smoke=True)
        self.assertEqual(summary["n_original_steps"],12)
        self.assertEqual(summary["n_original_sequences"],2)
        self.assertTrue(0<=summary["fraction_all12_positive"]<=1)
        self.assertEqual(len(summary["curves"]),12)

    def test_whole_base_sequence_bootstrap_same_as_independent_replay(self):
        values=[(j-32)/100 for j in range(64)]
        key=stable_seed(46058,103,"block9",7,"shared_energy_drop")
        self.assertEqual(key,independent_seed(46058,103,"block9",7,
                                             "shared_energy_drop"))
        self.assertEqual(bootstrap(values,key,draws=2000),
                         independent_bootstrap(values,key))

    def test_native_eight_physical_edits_preserve_source_and_endpoint(self):
        torch.manual_seed(444)
        model=TinyRecursiveModel().eval().requires_grad_(False)
        shared=HorizonValue().eval().requires_grad_(False)
        initial=HorizonValue().eval().requires_grad_(False)
        sae=TopKSAE().eval().requires_grad_(False)
        x,y=fixed_data(2,61058)
        normal=(torch.zeros(48),torch.ones(48))
        for k in (2,6):
            raw=native_edits(model,shared,initial,*normal,(sae,*normal),
                             x,y,103,"iid",k,smoke=True)
            self.assertEqual(len(raw),2)
            self.assertEqual(raw[0]["native_endpoint"],4 if k==2 else 12)
            for sample in raw:
                self.assertEqual(len(sample["original_goals"]),8)
                for goal in sample["original_goals"]:
                    if goal["source_admitted"]:
                        self.assertEqual(set(goal["arms"]),{
                            "shared_down","init_down","radial_down","native",
                            "sae_active","sae_remainder","head_null","random"})
                        for entry in goal["arms"].values():
                            self.assertLess(abs(entry["edit_L2"]-goal["dose"]),1e-5)
                            self.assertEqual(len(entry["all_final_token_CE"]),81)
                            self.assertEqual(len(entry["goal_logits"]),10)
            stats=causal_summary(raw,103,"iid",k,smoke=True)
            self.assertEqual(stats["original_goals"],16)
            self.assertEqual(stats["source_sequences"],2)

    def test_native_jacobian_skew_has_independent_bilinear_receipt(self):
        model=TinyRecursiveModel().eval().requires_grad_(False)
        x,_=fixed_data(2,61058)
        for k in (2,6):
            result=curl_witness(model,x,103,"iid",k,smoke=True)
            self.assertEqual(result["n_source_sequences"],2)
            for row in result["per_sequence"]:
                a,b=row["u_dot_J_w"],row["w_dot_J_u"]
                self.assertAlmostEqual(row["skew"],a-b)
                self.assertAlmostEqual(row["skew_ratio"],
                    abs(a-b)/(abs(a)+abs(b)+1e-8),places=8)
                self.assertTrue(math.isfinite(row["skew_ratio"]))

    def test_source_zero_digit_uses_ten_class_head(self):
        model=TinyRecursiveModel().eval().requires_grad_(False)
        _,families,_=create_source(seed=SMOKE_SEED,n=8)
        x,y=families["zero20"]
        with torch.no_grad():
            logits=model.head(model.initial_state(x[:2]))
        self.assertEqual(tuple(logits.shape),(2,81,10))
        self.assertTrue(bool((y[:2]==0).any()))


if __name__=="__main__":
    unittest.main()
