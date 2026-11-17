"""LS-SAE11 CPU-only same-run margin, exact cover and numeric-audit tests."""
from __future__ import annotations
import unittest

import numpy as np
import torch

from mechinterp.sae_causal_transfer.native_trm_v1 import families as S2
from mechinterp.sae_causal_transfer.native_hard_task_v1 import scoring as S7S
from mechinterp.sae_causal_transfer.native_withinrun_v1 import families as F
from mechinterp.sae_causal_transfer.native_withinrun_v1 import run_native as RUN
from mechinterp.sae_causal_transfer.native_withinrun_v1 import audit as AUD


class TinyNative:
    puzzle_emb_len=16
    def __init__(self):
        self.lm_head=torch.nn.Linear(512,11,bias=False)
        with torch.no_grad():
            self.lm_head.weight.zero_()
            for i in range(9):self.lm_head.weight[i+2,i]=1.0
            self.lm_head.weight[2,0]=1.25
    def act_step(self,zh,zl,x):
        next_l=.9*zl+.02*zh
        return .82*zh+.03*next_l+.01*x,next_l
    def digit_logits(self,zh):
        return self.lm_head(zh)[:,16:,2:11].float()


class WithinRunTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_exact_cover_unique_solved_and_conflicting_and_two_solution(self):
        target=S2.full_solution(20261117)
        solved=F.solve_unique(target)
        self.assertEqual(solved["solutions"],1)
        self.assertEqual(solved["target"],target)
        conflicting=target.copy()
        conflicting[0]=conflicting[1]
        self.assertEqual(F.solve_unique(conflicting)["solutions"],0)
        almost=target.copy()
        almost[0]=0
        self.assertEqual(F.solve_unique(almost)["solutions"],1)
        self.assertEqual(F.solve_unique(almost)["target"],target)
        multiple=F.solve_unique([0]*81,max_nodes=5000)
        self.assertEqual(multiple["solutions"],2)

    def test_first20_filenames_and_github_gitblob_integrity(self):
        self.assertEqual(len(F.SOURCE_FILES),20)
        self.assertEqual([name for name,_ in F.SOURCE_FILES],
                         sorted(name for name,_ in F.SOURCE_FILES))
        self.assertEqual(F.SOURCE_GROUPS["pilot_champagne"],(0,4))
        self.assertEqual(F.SOURCE_GROUPS["test_champagne_a"],(4,12))
        self.assertEqual(F.SOURCE_GROUPS["test_champagne_b"],(12,20))
        with self.assertRaises(ValueError):
            F.admitted({"test_champagne_a":[]})

    def test_strict_global_margin_certificate_is_sufficient(self):
        clean=np.zeros((81,9),dtype=np.float64)
        clean[:,0]=10
        edit=clean.copy();edit[0,1]+=1.
        good=RUN.certificates(clean,edit,np.asarray([0,1,3]))
        independent=AUD.certificate(clean.tolist(),edit.tolist(),[0]*81)
        self.assertTrue(good["strict_global_margin_certificate"])
        self.assertTrue(good["strict_all_blank_pairwise_certificate"])
        self.assertEqual(good["actual_blank_digit_flips"],0)
        self.assertEqual(good["actual_blank_digit_flips"],independent["actual_blank_digit_flips"])
        edit[0,1]=12
        bad=RUN.certificates(clean,edit,np.asarray([0,1,3]))
        self.assertEqual(bad["actual_blank_digit_flips"],1)
        self.assertFalse(bad["strict_global_margin_certificate"])
        self.assertFalse(bad["strict_all_blank_pairwise_certificate"])

    def test_same_forward_toy_hidden_logits_and_independent_scores(self):
        model=TinyNative()
        a={"zh":torch.ones((1,97,512),dtype=torch.float32),
           "zl":torch.ones((1,97,512),dtype=torch.float32)*.25,
           "x_emb":torch.ones((1,97,512),dtype=torch.float32)*.05}
        solution=S2.full_solution(20261118)
        puzzle=solution.copy()
        puzzle[3]=0
        donor={"source_direction_status":{arm:True for arm in RUN.ARMS},
               "direction_metadata":{"geometry":{}}}
        for i,arm in enumerate(RUN.ARMS):
            v=np.zeros(512,dtype=np.float64);v[i]=1.0
            donor["direction_metadata"]["geometry"][arm]={"direction_512":v.tolist()}
        wc,q,singular=RUN.readout(model)
        source={"seed":123,"x":puzzle,"target":solution}
        actual=RUN.run_case(model,source,a,donor,"test_52",0,wc,q,"hard")
        self.assertEqual(set(actual["native_81x9_logits"]),set(RUN.HORIZONS))
        AUD.signed_grid(actual["native_81x9_logits"],
                        actual["same_run_transport"],actual["native_task_scores"],
                        source,3,donor["source_direction_status"],singular[-1],singular[0])
        for h in RUN.HORIZONS:
            for arm in RUN.ARMS:
                for r in RUN.RADII:
                    for sign in ("plus","minus"):
                        rec=actual["same_run_transport"][h]["arms"][arm][r][sign]
                        self.assertLess(rec["centered_readout_reconstruction_error"],RUN.TOL_IDENTITY)

    def test_registered_fixed_paired_bootstrap_logic(self):
        scores=[]
        for _ in range(8):
            clean={"blank_correct":30,"blank_total":54}
            scores.append({"native_task_scores":{"ACT6":{"clean":clean,"arms":{
                "state_power":{"8.00":{"plus":{"blank_correct":34},
                                          "minus":{"blank_correct":32}}},
                "random":{"8.00":{"plus":{"blank_correct":30},
                                     "minus":{"blank_correct":30}}},
            }}}})
        both=RUN.family_summary(scores,"test_champagne_a")
        independent=AUD.score_family(scores,"test_champagne_a")
        self.assertEqual(both,independent)
        self.assertTrue(both["registered_success"])
        for case in scores:
            case["native_task_scores"]["ACT6"]["arms"]["random"]["8.00"]["plus"]["blank_correct"]=36
        no=RUN.family_summary(scores,"test_champagne_a")
        self.assertFalse(no["registered_success"])


if __name__=="__main__":
    unittest.main()
