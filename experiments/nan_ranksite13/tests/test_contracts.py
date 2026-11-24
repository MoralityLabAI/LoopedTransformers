import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from torch import nn
from experiments.nan_recomp09.common import read,sha
from experiments.nan_ranksite13.gauge import gauge_diagnostic


class RankSiteContracts(unittest.TestCase):
    def test_all_six_missing_cells_and_physical_widths_are_explicit(self):
        cfg=read('experiments/nan_ranksite13/config.json')
        actual={(c['module'],c['rank'],c['passes']) for c in cfg['conditions']}
        expected={(m,r,t) for m,r in [('down_proj',16),('o_proj',8),('o_proj',16)] for t in (1,2)}
        self.assertEqual(actual,expected);self.assertEqual(len(cfg['conditions']),6)
        for c in cfg['conditions']:
            self.assertEqual(c['input_width'],10752 if c['module']=='down_proj' else 6144)
            self.assertEqual(c['output_width'],3072)
        self.assertFalse(cfg['BPD_donor_transfer_enabled']);self.assertEqual(cfg['fit_steps'],8)
        self.assertEqual(cfg['formal_inferential_comparisons'],0)

    def test_derived_fit_retains_source_and_uses_rank_for_both_schedule_shapes(self):
        root=Path('experiments/nan_ranksite13');r=read(root/'derivation_receipt.json')
        self.assertEqual(sha(root/'fit.py'),r['derived_fit_sha256'])
        self.assertEqual(sha('experiments/nan_recomp09/fit.py'),r['parent_fit_sha256'])
        text=(root/'fit.py').read_text();ast.parse(text)
        self.assertEqual(text.count("(1,passes,length,cfg['rank'])"),2)
        self.assertNotIn("initial['V8']",text);self.assertIn("initial['V']",text)
        self.assertIn('with owner.context():',text)
        self.assertIn('clean_masks',text);self.assertIn('exact_means(',text)
        self.assertIn("cfg['diagnostic_modes']",text)

    def test_finite_gauge_relabels_controller_and_restores_all_parameters_at_both_ranks(self):
        class Controller(nn.Module):
            def __init__(self,rank):
                super().__init__();self.out_proj=nn.Linear(5,rank)
            def forward(self,values):
                return SimpleNamespace(lower_leaky={k:torch.sigmoid(self.out_proj(x)) for k,x in values.items()})
        for rank in (8,16):
            torch.manual_seed(rank)
            core=SimpleNamespace(C=rank,d_in=5,V=nn.Parameter(torch.randn(5,rank)),U=nn.Parameter(torch.randn(rank,3)),mean_comp_acts=torch.randn(2,rank))
            controller=Controller(rank);trainer=SimpleNamespace(ci_net=controller,bpd_model=SimpleNamespace(module_order=['site']))
            before=[x.clone() for x in (core.V,core.U,core.mean_comp_acts,controller.out_proj.weight,controller.out_proj.bias)]
            result=gauge_diagnostic(core,trainer)
            self.assertTrue(result['stable_at_1e_minus5'])
            for x,y in zip(before,(core.V,core.U,core.mean_comp_acts,controller.out_proj.weight,controller.out_proj.bias)):
                torch.testing.assert_close(x,y,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
