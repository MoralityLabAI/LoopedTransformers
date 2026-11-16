"""CPU regressions for LS-SAE06 task truth, gauge and native all-cell controls."""
import unittest

import numpy as np
import torch

from mechinterp.sae_causal_transfer.native_trm_v1 import families as SOURCE
from mechinterp.sae_causal_transfer.native_task_transfer_v1 import families as FAM
from mechinterp.sae_causal_transfer.native_task_transfer_v1 import scoring as SCORE
from mechinterp.sae_causal_transfer.native_task_transfer_v1 import audit as AUD
from mechinterp.sae_causal_transfer.native_task_transfer_v1 import run_native as NATIVE


class NativeToy:
    puzzle_emb_len=16

    def __init__(self):
        self.head=torch.zeros((9,512),dtype=torch.float32)
        for j in range(9):self.head[j,j]=2.5

    def act_step(self,zh,zl,xemb):
        # Differentiable all-token tied recurrence with a stable local gain.
        global_mean=zh.mean(1,keepdim=True)
        output=.91*zh+.12*global_mean
        return output,zl

    def digit_logits(self,hidden):
        return torch.matmul(hidden[:,16:,:],self.head.T)


class LS6Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_source_new_solution_uniqueness_and_preoutcome_median(self):
        sha=FAM.manifest()
        self.assertEqual(set(sha),set(FAM.SEEDS))
        all_rows=[row for f in FAM.SEEDS for row in FAM.roster(f)]
        self.assertEqual(len(all_rows),20)
        self.assertEqual(len({tuple(r["target"]) for r in all_rows}),20)
        for family in FAM.SEEDS:
            self.assertEqual(sha[family],SOURCE.fingerprint(FAM.roster(family)))
            for row in FAM.roster(family):
                goal=FAM.goal_from_visible_mask(row["x"])
                blank=[i for i,z in enumerate(row["x"]) if z==0]
                self.assertEqual(goal,blank[len(blank)//2])
                self.assertEqual(SOURCE.count_solutions(row["x"]),1)
        with self.assertRaises(ValueError):
            FAM.roster("train_expanded")

    def test_exact_Sudoku_solution_validity_and_conflict(self):
        row=FAM.roster("pilot_30")[0]
        target=row["target"]
        self.assertTrue(SCORE.valid_sudoku(target,row["x"]))
        self.assertEqual(SCORE.unit_conflicts(target),0)
        edit=target.copy()
        edit[0]=edit[1]
        self.assertFalse(SCORE.valid_sudoku(edit,row["x"]))
        self.assertGreater(SCORE.unit_conflicts(edit),0)

    def test_81x9_truth_and_additive_logit_gauge(self):
        row=FAM.roster("pilot_30")[0]
        target=np.asarray(row["target"])
        base=np.full((81,9),-5,dtype=np.float64)
        base[np.arange(81),target-1]=5
        clean=base.copy()
        plus=clean+100.0*np.arange(1,82)[:,None]
        minus=clean-12.3*np.arange(1,82)[:,None]
        goal=FAM.goal_from_visible_mask(row["x"])
        info=SCORE.signed_pair_effect(clean,plus,minus,row["x"],row["target"],goal)
        self.assertEqual(info["clean"]["blank_correct"],30)
        self.assertEqual(info["clean"]["unit_conflicts"],0)
        self.assertTrue(info["clean"]["board_exact"])
        self.assertTrue(info["plus"]["sudoku_valid_and_clue_consistent"])
        self.assertEqual(info["symmetric_blank_prediction_flip_score"],0)
        self.assertLess(info["mean_centered_logit_effect_unscaled"],1e-10)
        task=info["plus_margins"]
        self.assertEqual(len(task["clean_rival_zero_based"]),30)
        self.assertTrue(all(abs(x-y)<1e-10 for x,y in zip(
            task["clean_margins"],task["edited_margins"])))

    def test_predicted_digit_flip_and_verification_differs_from_logit_magnitude(self):
        row=FAM.roster("pilot_34")[0]
        truth=row["target"]
        goal=FAM.goal_from_visible_mask(row["x"])
        clean=np.full((81,9),-10.,dtype=np.float64)
        clean[np.arange(81),np.asarray(truth)-1]=10
        altered=clean.copy()
        wrong=(truth[goal]%9)+1
        altered[goal,wrong-1]=30
        info=SCORE.signed_pair_effect(clean,altered,clean,row["x"],truth,goal)
        self.assertEqual(info["plus"]["blank_correct"],33)
        self.assertEqual(info["plus"]["blank_correct_to_wrong"],1)
        self.assertEqual(info["plus"]["blank_prediction_changes"],1)
        self.assertFalse(info["plus"]["board_exact"])
        self.assertEqual(info["symmetric_blank_prediction_flip_score"],.5)
        self.assertEqual(info["symmetric_blank_correctness_gain"],-.5)

    def test_native_logit_power_JtJ_and_true_81logit_JVP(self):
        model=NativeToy()
        anchor={"zh":torch.zeros((1,97,512)),
                "zl":torch.zeros((1,97,512)),
                "x_emb":torch.zeros((1,97,512))}
        token=16+48
        u,history=NATIVE.logit_power_direction(model,anchor,token,
                                               "test_30",202626030000)
        self.assertEqual(len(history),4)
        self.assertAlmostEqual(float(np.linalg.norm(u)),1,places=8)
        for k,step in enumerate(history):
            self.assertEqual(step["status"],"eligible")
            self.assertAlmostEqual(step["jvp_dot_vjp"],step["squared_gain"],places=4)
            nxt=np.asarray(history[k+1]["input_direction_512"] if k<3 else u)
            w=np.asarray(step["reverse_gradient_512"])
            np.testing.assert_allclose(w/np.linalg.norm(w),nxt,atol=1e-8)
        geo=NATIVE.forward_geometry(model,anchor,token,u)
        self.assertEqual(len(geo["all97_hidden_tangent_norms"]),97)
        self.assertEqual(np.asarray(geo["centered_digit_logit_jvp_81x9"]).shape,(81,9))
        self.assertTrue(all(abs(sum(z))<1e-5 for z in geo["centered_digit_logit_jvp_81x9"]))
        variant=NATIVE.native_full_board_variants(model,anchor,48,
                 {"state_power":u,"logit_power":u,"sae":u,"random":u})
        for horizon in ("ACT3","ACT6"):
            self.assertEqual(np.asarray(variant[horizon]["clean"]).shape,(81,9))
            self.assertEqual(np.asarray(variant[horizon]["state_power"]["0.20"]["plus"]).shape,(81,9))
        with self.assertRaises(ValueError):
            AUD.audit({"schema":NATIVE.SCHEMA,"original_preregistration_commit":"changed"})

    def test_preregistered_pair_bootstrap_and_negative_flips(self):
        rows=[]
        for _ in range(8):
            pair=lambda flips,correct:{"symmetric_blank_prediction_flip_score":flips,
                                        "symmetric_blank_correctness_gain":correct}
            rows.append({"task_score":{"ACT6":{
                "state_power":{"0.20":pair(3,0)},
                "random":{"0.20":pair(.5,0)}
            }}})
        got=SCORE.score_families(rows,"test_30")
        self.assertTrue(got["registered_success"])
        self.assertEqual(got["power_flip_events_at_least_one"],8)
        self.assertAlmostEqual(got["bootstrap95"][0],2.5)
        for item in rows:
            item["task_score"]["ACT6"]["random"]["0.20"]["symmetric_blank_prediction_flip_score"]=3.5
        self.assertFalse(SCORE.score_families(rows,"test_30")["registered_success"])


if __name__=="__main__":
    unittest.main()
