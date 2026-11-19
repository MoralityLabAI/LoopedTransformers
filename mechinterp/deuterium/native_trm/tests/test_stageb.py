"""Development-only smoke and source-boundary tests; no fresh evaluation."""
import tempfile
import unittest
from pathlib import Path
import torch

from mechinterp.loopscope04.gradient_flow import TinyRecursiveModel, fixed_data
from mechinterp.deuterium.native_trm import fit_and_select as s


class TestStageBSourceOnly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(s.SMOKE_SEED)
        self.model = TinyRecursiveModel(48,4).eval().requires_grad_(False)
        self.x,self.y=fixed_data(4,s.SMOKE_SEED)

    def test_config_targets_one_native_shared_block(self):
        cfg=s.config(s.SMOKE_SEED,smoke=True)
        self.assertEqual(cfg.steps,1)
        self.assertEqual(cfg.module_info[0].module_pattern,s.TARGET)
        self.assertEqual(cfg.module_info[0].C,16)
        self.assertEqual(len(cfg.loss_metric_configs),3)
        self.assertEqual(s.CALIBRATION_SEED,731055)
        self.assertEqual(s.RESERVED_TEST_SEED,731057)
        self.assertNotIn(31059,s.OLD_SEED_COUNTS)

    def test_real_weight_edit_closure(self):
        base=self.model.block.chan_mix[2].weight.detach().clone()
        g=torch.Generator().manual_seed(s.SMOKE_SEED)
        dose=float(base.norm())*.01
        direction=s.normalize_rank_one(torch.outer(
            torch.randn(48,generator=g),torch.randn(128,generator=g)),dose)
        result, info=s.score_edit(self.model,self.x,self.y,direction,dose)
        self.assertEqual(len(result["goal_nll"]),4)
        self.assertLessEqual(info["actual_relative_dose_error"],1e-4)
        self.assertTrue(torch.equal(self.model.block.chan_mix[2].weight,base))
        self.assertEqual(set(result),{
            "goal_nll","offgoal_nll","full_nll",
            "goal_accuracy","offgoal_accuracy","full_accuracy"
        })

    def test_target_gradient_is_rank_one_at_matched_dose(self):
        w=self.model.block.chan_mix[2].weight
        rho=float(w.norm())*.01
        d=s.task_gradient(self.model,self.x,self.y,rho)
        self.assertEqual(tuple(d.shape),(48,128))
        self.assertLess(abs(float(d.norm())-rho)/rho,1e-5)
        self.assertLess(float(torch.linalg.svdvals(d)[1]),1e-5)
        self.assertFalse(w.requires_grad)

    def test_candidate_banks_independent_exact_dose(self):
        g=torch.Generator().manual_seed(16073)
        u=torch.randn(16,48,generator=g)
        v=torch.randn(128,16,generator=g)
        b,rho=s.banks(u,v,self.model.block.chan_mix[2].weight,103)
        self.assertEqual(set(b),{"vpd","random"})
        for arm,vecs in b.items():
            self.assertEqual(len(vecs),16)
            for d in vecs:
                self.assertLess(abs(float(d.norm())-rho)/rho,1e-5)
                self.assertLess(float(torch.linalg.svdvals(d)[1]),1e-5)

    def test_no_scoring_seed_generated_in_source_only_test(self):
        self.assertEqual(s.SOURCE_TRAIN_SEED,31041)
        self.assertNotEqual(s.SMOKE_SEED,s.CALIBRATION_SEED)
        self.assertNotEqual(s.SMOKE_SEED,s.RESERVED_TEST_SEED)
        self.assertEqual(sum(p.numel() for p in self.model.parameters()),44001)


if __name__=="__main__":
    unittest.main()
