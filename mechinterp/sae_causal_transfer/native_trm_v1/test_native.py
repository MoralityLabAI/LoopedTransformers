"""CPU-only unit regressions for LS-SAE02 source lock and recurrent derivatives."""
import math
import unittest
from unittest.mock import patch

import numpy as np
import torch

from mechinterp.loopscope02.scripts.trm_sudoku import PureSudokuTRM
from mechinterp.sae_causal_transfer.native_trm_v1 import audit as AUD
from mechinterp.sae_causal_transfer.native_trm_v1 import families as FAM
from mechinterp.sae_causal_transfer.native_trm_v1 import fit as FIT
from mechinterp.sae_causal_transfer.native_trm_v1 import run_native as RUN


class NativeSudokuTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_noncanonical_solved_and_unique_full_prefix(self):
        groups,sha=FAM.build_rosters(pilot=True)
        self.assertEqual({k:len(v) for k,v in groups.items()},FAM.PILOT)
        all_rows=[r for values in groups.values() for r in values]
        self.assertEqual(len(set(tuple(row["target"]) for row in all_rows)),12)
        for name,rows in groups.items():
            self.assertEqual(sha[name],FAM.fingerprint(rows))
            for row in rows:
                self.assertTrue(FAM.valid_solution(row["target"]))
                self.assertGreater(FAM.rectangles(row["target"]),0)
                self.assertEqual(FAM.count_solutions(row["x"]),1)
                self.assertEqual(len(FAM.visible_goals(row["x"])),2)
                self.assertEqual(row["x"].count(0),FAM.HOLES[name])

    def test_training_is_fit_only_and_physically_normalized(self):
        rng=np.random.default_rng(177)
        # A small source-only 4-puzzle fit has 324 hidden states (>48 rank).
        states=rng.normal(size=(4,81,512)).astype(np.float32)
        fitted=FIT.fit_only(states)
        self.assertEqual(fitted.basis.shape,(512,48))
        self.assertEqual(fitted.physical_rows.shape,(96,48))
        self.assertEqual(fitted.physical_decoder.shape,(96,48))
        self.assertTrue(np.all(fitted.scale>0))
        self.assertTrue(np.all(np.isfinite(fitted.physical_bias)))
        self.assertEqual(len(fitted.fingerprint),64)
        sample=states[0,0]
        y=fitted.encode(sample)
        self.assertEqual(y.shape,(48,))
        with torch.no_grad():
            normed=torch.tensor((y-fitted.mu)/fitted.scale,dtype=torch.float32)
            a=fitted.sae.encode(normed).numpy()
        from mechinterp.sae_causal_transfer.observer import encode_top4
        b,_=encode_top4(y,fitted.physical_rows,fitted.physical_bias)
        self.assertLess(float(np.max(np.abs(a-b))),1e-5)
        with self.assertRaises(ValueError):
            FIT.fit_only(states,steps=201)
        with self.assertRaises(ValueError):
            FIT.fit_only(states,seed=89)

    def toy(self):
        torch.manual_seed(21)
        model=PureSudokuTRM(hidden_size=32,num_heads=4,l_layers=1,
                            h_cycles=1,l_cycles=1,puzzle_emb_len=4,seq_len=81).float().eval()
        model.requires_grad_(False)
        fixture=FAM.full_solution(202619000000)
        x=FAM.dig_unique(fixture,202619000000+1000000,30)
        row={"x":x,"target":fixture}
        hidden,anchors=FIT.stage_one(model,[row],retain_zl=True)
        self.assertEqual(hidden.shape,(1,81,32))
        return model,row,anchors[0]

    def test_real_act_recurrent_autograd_agrees_with_finite_difference(self):
        model,row,anchor=self.toy()
        basis=np.eye(32,dtype=np.float64)[:,:8]
        fit=type("TestFit",(),{"basis":basis})()
        pos=FAM.visible_goals(row["x"])[0]
        truth=row["target"][pos]-1
        edit=np.array([.001,0,0,0,0,0,0,0])
        dummy={"kill":edit,"baseline":np.zeros(8),"best":np.zeros(8)}
        clean=RUN.four_branch_logits(model,anchor,pos,fit,dummy)
        native=RUN.native_gradients(model,anchor,pos,truth,clean,fit)
        self.assertEqual(tuple(sorted(native)),(0,1,2))
        for k,(gradient,rival) in native.items():
            observed=(clean[k][0,truth]-clean[k][0,rival]).item()
            perturbed=(clean[k][1,truth]-clean[k][1,rival]).item()
            # Changing zh by minus epsilon: (clean - edited) / epsilon ≈ grad·e0.
            predicted=float(gradient[0]*.001)
            self.assertLess(abs((observed-perturbed)-predicted),1e-3)
            self.assertTrue(np.all(np.isfinite(gradient)))

    def test_one_goal_native_128_observer_controls_and_margin_bookkeeping(self):
        model,row,anchor=self.toy()
        fit=type("TestFit",(),{"basis":np.eye(32,dtype=np.float64)[:,:8]})()
        pos=FAM.visible_goals(row["x"])[0]
        dummy={"kill":np.eye(8)[0]*.01,"baseline":np.eye(8)[1]*.01,
               "best":np.eye(8)[2]*.01}
        score=RUN.score_case(model,anchor,pos,row["target"][pos]-1,fit,dummy,
                            "test_30",row.get("seed",202619030000),0)
        for depth in ("0","1","2"):
            d=score[depth]
            v=d["margins"]
            self.assertAlmostEqual(d["kill_minus_best_margin_loss"],
                                   v["best"]-v["kill"],places=5)
            self.assertEqual(len(d["gradient_48"]),8)
            self.assertEqual(len(d["all_arm_logits_4x9"]),4)
            null=d["random_rank1_controls"]
            self.assertEqual(len(null["random_alignments"]),128)
            self.assertGreaterEqual(null["rank_p"],1/129)
            self.assertLessEqual(null["rank_p"],1)
            self.assertTrue(all(0<=x<=1 for x in null["random_alignments"]))

    def test_puzzle_clustered_mean_and_flip_gate(self):
        rows=[]
        for j in range(16):
            goals=[]
            for slot in range(2):
                eligible=j<12 and slot==0
                if eligible:
                    goals.append({"eligible":True,"depths":{
                      "0":{"kill_minus_best_margin_loss":0,
                           "predicted_first_order_gradient_effect":0,
                           "relative_nonlinear_residual":0,
                           "random_rank1_controls":{"rank_p":.5}},
                      "1":{"kill_minus_best_margin_loss":1,
                           "predicted_first_order_gradient_effect":1,
                           "relative_nonlinear_residual":0,
                           "random_rank1_controls":{"rank_p":.5}},
                      "2":{"kill_minus_best_margin_loss":2,
                           "predicted_first_order_gradient_effect":2,
                           "relative_nonlinear_residual":0,
                           "kill_clean_correct_to_wrong":True,
                           "best_clean_correct_to_wrong":False,
                           "random_rank1_controls":{"rank_p":.5}}
                    }})
                else:goals.append({"eligible":False})
            rows.append({"goals":goals})
        got=RUN.summarize_family(rows,"test_30")
        self.assertEqual(got["eligible_goals"],12)
        self.assertEqual(got["eligible_puzzles"],12)
        self.assertAlmostEqual(got["paired_mean_depth2"],2)
        self.assertTrue(got["registered_success"])
        self.assertEqual(got["puzzle_bootstrap95"],[2,2])
        for row in rows:
            for g in row["goals"]:
                if g["eligible"]:
                    g["depths"]["2"]["kill_clean_correct_to_wrong"]=False
        rejected=RUN.summarize_family(rows,"test_30")
        self.assertFalse(rejected["registered_success"])

    def test_audit_rejects_nonregistered_checkpoint(self):
        with self.assertRaises((ValueError,KeyError)):
            AUD.audit({"schema":"spoof","checkpoint_tag":"gaoxin492",
                       "checkpoint_file_sha256":"wrong","preregistration_commit":AUD.PROTOCOL_SHA})


if __name__=="__main__":
    unittest.main()
