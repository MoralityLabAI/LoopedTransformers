"""LS-SAE08 frozen first-admissible source, native dose and audit regressions."""
from __future__ import annotations
import unittest
import numpy as np
import torch

from mechinterp.sae_causal_transfer.native_hard_task_v2 import families as FAM
from mechinterp.sae_causal_transfer.native_hard_task_v2 import run_native as NATIVE
from mechinterp.sae_causal_transfer.native_hard_task_v2 import audit as AUD
from mechinterp.sae_causal_transfer.native_hard_task_v1 import scoring as SCORE
from mechinterp.sae_causal_transfer.native_trm_v1 import families as SOURCE


class TinyNative:
    puzzle_emb_len=16
    def act_step(self,zh,zl,x):
        m=zh.mean(dim=1,keepdim=True)
        return .91*zh+.04*m,zl
    def digit_logits(self,zh):
        # All native 97 tokens, board suffix 81, exactly nine digit outputs.
        return zh[:,16:,:9].float()


class HardFirstAdmittedTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_all_twenty_new_unique_source_puzzles_and_first_success(self):
        digest=FAM.full_manifest()
        self.assertEqual(set(digest),set(FAM.BASES))
        solutions=set();puzzles=set()
        for family in FAM.BASES:
            rows=FAM.roster(family)
            self.assertEqual(len(rows),FAM.SIZES[family])
            self.assertEqual(digest[family],SOURCE.fingerprint(rows))
            for j,row in enumerate(rows):
                self.assertEqual(row["slot"],j)
                self.assertEqual(row["seed"],row["grid_seed"])
                self.assertLess(row["candidate_attempt_index"],32)
                self.assertEqual(len(row["rejected_model_blind_candidates"]),
                                 row["candidate_attempt_index"])
                self.assertEqual(row["seed"],
                                 FAM.BASES[family]+j*100+row["candidate_attempt_index"])
                for k,refusal in enumerate(row["rejected_model_blind_candidates"]):
                    self.assertEqual(refusal["grid_seed"],FAM.BASES[family]+j*100+k)
                    self.assertIn("registered hard source admission failed",refusal["failure"])
                self.assertEqual(row["x"].count(0),FAM.HOLES[family])
                self.assertEqual(SOURCE.count_solutions(row["x"],2),1)
                self.assertIn(row["visible_structural_difficulty"]["propagation_rounds"],range(82))
                self.assertGreater(row["rectangles"],0)
                self.assertEqual(FAM.median_blank(row["x"]),
                   [i for i,v in enumerate(row["x"]) if not v][row["holes"]//2])
                self.assertNotIn(tuple(row["target"]),solutions)
                self.assertNotIn(tuple(row["x"]),puzzles)
                solutions.add(tuple(row["target"]));puzzles.add(tuple(row["x"]))
        self.assertEqual(len(solutions),20)

    def test_source_failure_preserved_in_prior_campaign(self):
        from mechinterp.sae_causal_transfer.native_hard_task_v1 import admission
        prior=admission.evaluate()
        self.assertFalse(prior["full_scientific_cohort_executable"])
        self.assertEqual(prior["per_family"]["test_54"]["admitted_slots"],7)

    def test_native_all_act_horizons_and_three_registered_radii(self):
        model=TinyNative()
        anchor={"zh":torch.zeros((1,97,512)),
                "zl":torch.zeros((1,97,512)),
                "x_emb":torch.zeros((1,97,512))}
        directions={
          "state_power":np.eye(512,dtype=np.float64)[0],
          "logit_power":np.eye(512,dtype=np.float64)[1],
          "sae":np.eye(512,dtype=np.float64)[2],
          "random":np.eye(512,dtype=np.float64)[3]
        }
        from mechinterp.sae_causal_transfer.native_hard_task_v1 import run_native as BASE
        out=BASE.native_all_outputs(model,anchor,39,directions)
        self.assertEqual(set(out),{"ACT3","ACT6","ACT12"})
        for horizon in out:
            self.assertEqual(np.asarray(out[horizon]["clean"]).shape,(81,9))
            for arm in SCORE.ARMS:
                self.assertEqual(set(out[horizon][arm]),{"0.20","2.00","8.00"})
                for radius in ("0.20","2.00","8.00"):
                    for sign in ("plus","minus"):
                        self.assertEqual(np.asarray(out[horizon][arm][radius][sign]).shape,(81,9))

    def test_registered_paired_flip_and_separate_accuracy_count(self):
        def blank_pair(n,correct):
            return {"symmetric_blank_prediction_flip_score":n,
                    "symmetric_blank_correctness_gain":correct,
                    "clean":{"blank_correct":51,"blank_total":52},
                    "plus":{"blank_correct":52,"clue_disagreements":0},
                    "minus":{"blank_correct":51,"clue_disagreements":0}}
        fake=[]
        for i in range(8):
            fake.append({"task_score":{"ACT6":{
                "state_power":{"8.00":blank_pair(3,.5)},
                "random":{"8.00":blank_pair(1,0)}}}})
        yes=SCORE.family_score(fake,"test_52")
        self.assertTrue(yes["registered_success"])
        self.assertEqual(yes["clean_inexact_puzzles_ACT6"],8)
        self.assertEqual(yes["power_best_sign_clue_consistent_improvement_on_inexact"],8)
        for row in fake:
            row["task_score"]["ACT6"]["random"]["8.00"]["symmetric_blank_prediction_flip_score"]=4
        no=SCORE.family_score(fake,"test_52")
        self.assertFalse(no["registered_success"])

    def test_auditor_rejects_changed_prospective_protocol(self):
        with self.assertRaises((KeyError,ValueError)):
            AUD.audit({"schema":NATIVE.SCHEMA,"preregistration_commit":"invalid"})


if __name__=="__main__":
    unittest.main()
