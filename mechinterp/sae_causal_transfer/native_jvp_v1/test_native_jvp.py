"""LS-SAE03 source-lock, 512D JVP and scientific-gate regression tests."""
import math
import unittest

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

from mechinterp.loopscope02.scripts.trm_sudoku import PureSudokuTRM
from mechinterp.sae_causal_transfer.native_jvp_v1 import geometry as G
from mechinterp.sae_causal_transfer.native_jvp_v1 import families as FAM
from mechinterp.sae_causal_transfer.native_jvp_v1 import run_native as NATIVE
from mechinterp.sae_causal_transfer.native_jvp_v1 import audit as AUD


class NativeJacobianTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_fresh_puzzles_and_label_blind_median(self):
        rosters,sha=FAM.build_new(pilot=True)
        self.assertEqual([len(rosters[f]) for f in FAM.NEW_BASES],[2,2])
        seen=set()
        for family,rows in rosters.items():
            for r in rows:
                self.assertGreater(r["rectangles"],0)
                self.assertEqual(r["x"].count(0),FAM.NEW_HOLES[family])
                self.assertEqual(FAM.median_blank(r["x"]),
                                 [i for i,v in enumerate(r["x"]) if v==0][r["holes"]//2])
                self.assertEqual(len(r["target"]),81)
                self.assertNotIn(tuple(r["target"]),seen)
                seen.add(tuple(r["target"]))
        with self.assertRaises(ValueError):
            FAM.median_blank([0]*81)

    def toy(self):
        torch.manual_seed(233)
        model=PureSudokuTRM(hidden_size=32,num_heads=4,l_layers=1,
                            h_cycles=1,l_cycles=1,puzzle_emb_len=4,seq_len=81).float().eval()
        model.requires_grad_(False)
        x=torch.randint(1,10,(1,81),dtype=torch.long)
        with torch.no_grad():
            inp=model.encode_inputs(x)
            h,l=model.init_states(1)
            h,l=model.act_step(h,l,inp)
        return model,h.detach(),l.detach(),inp.detach()

    def test_native_exact_forward_ad_matches_reverse_vjp_and_secant(self):
        model,h,l,emb=self.toy()
        index=model.puzzle_emb_len+18
        direction=torch.zeros_like(h)
        direction[0,index,7]=1.0
        with torch.enable_grad(),sdpa_kernel(backends=[SDPBackend.MATH]):
            clean,js=torch.func.jvp(
                lambda anchor:NATIVE.clean_trajectory(model,anchor,l,emb),
                (h,),(direction,))
            selected=h.detach().clone().requires_grad_(True)
            states=NATIVE.clean_trajectory(model,selected,l,emb)
            objective=(model.lm_head(states[2])[0,index,2]-
                       model.lm_head(states[2])[0,index,3])
            grad=torch.autograd.grad(objective,selected)[0]
            vjp=float((grad*direction).sum())
            out0=js[2][0,index]
            head=model.lm_head.weight.detach()
            jvp=float((head[2]-head[3])@out0)
        self.assertAlmostEqual(jvp,vjp,places=4)
        with torch.no_grad(),sdpa_kernel(backends=[SDPBackend.MATH]):
            a=NATIVE.clean_trajectory(model,h+.002*direction,l,emb)[2]
            b=NATIVE.clean_trajectory(model,h-.002*direction,l,emb)[2]
            row=head[2]-head[3]
            sec=float(row@(a[0,index]-b[0,index])/.004)
        self.assertLess(abs(sec-jvp),.06)

    def test_full_97_token_JVP_energy_replay(self):
        sample=np.zeros((1,97,512),dtype=np.float32)
        sample[0,16+40,5]=3
        sample[0,17,12]=4
        w=np.zeros(512,dtype=np.float64);w[5]=2
        result=G.tangent_stats(sample,56,w)
        self.assertAlmostEqual(result["full_state_gain"],5)
        self.assertAlmostEqual(result["goal_state_gain"],3)
        self.assertAlmostEqual(result["other_tokens_gain"],4)
        self.assertAlmostEqual(result["task_margin_slope"],6)
        self.assertAlmostEqual(result["target_energy_fraction"],9/25)
        self.assertAlmostEqual(result["task_readout_abs_cosine"],1)

    def test_calibration_label_blind_controls_and_rank(self):
        W=np.zeros((96,48),dtype=np.float64)
        for j in range(4):W[j,j]=1
        b=np.full(96,-50.);b[:4]=0
        D=np.zeros((96,48));D[0,0]=1
        B=np.eye(512)[:,:48]
        h=np.zeros(48);h[:4]=[2,1.5,1,0.5]
        head=np.eye(512)[:9]
        logits=np.arange(9,dtype=np.float64)
        case=G.choose_directions(h,W,b,D,B,head,logits,"test_30",202622030000)
        self.assertEqual(case["meta"]["sae_status"],"eligible")
        self.assertEqual(case["meta"]["sae_feature"],0)
        self.assertEqual(case["meta"]["clean_winner"],8)
        self.assertEqual(len(case["directions"]),4)
        self.assertAlmostEqual(float(np.linalg.norm(case["directions"]["sae"])),1)
        self.assertIsNone(case["directions"]["head_null"])  # SAE e0 within nine-digit head
        rand=G.random_rank_baseline(np.ones(512),case["directions"]["sae"],
                                    "test_30",202622030000)
        self.assertEqual(len(rand["random_cosines"]),128)
        self.assertGreaterEqual(rand["rank_p"],1/129)
        W[0]*=0
        case=G.choose_directions(h,W,b,D,B,head,logits,"test_30",202622030000)
        self.assertNotEqual(case["meta"]["sae_status"],"eligible")

    def test_registered_scoring_preserves_negative_outcomes_and_puzzle_clustering(self):
        def cond(a,b,r):
            return {"0":{"task_margin_slope":a},
                    "2":{"task_margin_slope":b,"full_state_gain":r}}
        rows=[]
        for j in range(12):
            rows.append({"sae_admitted":True,"directions":{
                "sae":cond(10,.1,1) if j<10 else cond(10,10,1),
                "random":cond(10,10,1)}})
        score=G.family_score(rows,"test_30")
        self.assertTrue(score["registered_success"])
        self.assertEqual(score["sae_attenuation_with_survival_events"],10)
        self.assertEqual(score["random_attenuation_with_survival_events"],0)
        for x in rows:x["directions"]["random"]=cond(10,.1,1)
        score=G.family_score(rows,"test_30")
        self.assertFalse(score["registered_success"])
        self.assertEqual(score["random_attenuation_with_survival_events"],12)

    def test_stdlib_rejects_changed_checkpoint(self):
        with self.assertRaises((ValueError,KeyError)):
            AUD.audit({"schema":"LS-SAE03/full-native-autograd-JVP-512-v1",
                       "preregistration_commit":"changed",
                       "source":{"original_checkpoint_file_sha256":"changed"}})


if __name__=="__main__":
    unittest.main()
