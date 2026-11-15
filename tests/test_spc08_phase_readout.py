"""SPC-08 bounded CPU checks; trained checkpoints are not included in CI."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.spectral_state_cost_v0.spc04_trained_dynamics import new_model,weight_sha256
from experiments.spectral_state_cost_v0.spc05_phase_interventions import model_from_receipt
from experiments.spectral_state_cost_v0.spc06_readout_observability import (
    logit_margin_jacobian,exact_readout_fiber_rotate,
)
from experiments.spectral_state_cost_v0.spc08_phase_readout import (
    row_null, fit_causal_decomposition, heldout_local_sensitivity, causal_phase_batch,
    summarize, paired_bootstrap,
)

class SPC08PhaseReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = new_model(41522).eval().double()
        for p in cls.model.parameters():p.requires_grad_(False)

    def test_readout_null_first_order_orthogonality(self):
        g = torch.Generator().manual_seed(20261115)
        state=torch.randn((4,48),generator=g,dtype=torch.float64)
        delta=torch.randn((4,48),generator=g,dtype=torch.float64)
        J=logit_margin_jacobian(self.model,state)
        row,null=row_null(delta,J)
        self.assertLess(float(torch.linalg.vector_norm((J@null.unsqueeze(-1)).squeeze(-1))),1e-9)
        self.assertLess(float(torch.linalg.vector_norm(row+null-delta)),1e-12)
        self.assertLess(float(torch.max(torch.abs((row*null).sum(-1)))),1e-10)
        self.assertTrue(torch.allclose(delta.square().sum(-1),row.square().sum(-1)+null.square().sum(-1),atol=1e-10))

    def test_exact_fiber_preserves_all_logits(self):
        g=torch.Generator().manual_seed(19)
        h=torch.randn((3,48),generator=g,dtype=torch.float64)
        delta=torch.randn((3,48),generator=g,dtype=torch.float64)
        for scale in (0.5,1.0,8.0):
            shifted=exact_readout_fiber_rotate(self.model,h,delta,scale,'logits')
            self.assertLess(float((self.model.readout(h)-self.model.readout(shifted)).abs().max()),1e-10)
            self.assertGreater(float(torch.linalg.vector_norm(shifted-h)),1e-8)

    def test_fit_causal_decomposition_valid_psd_and_captures(self):
        basis,diagnostics,hashes=fit_causal_decomposition(self.model,n_prompts=2,horizon=2)
        self.assertEqual(len(hashes),2)
        for key in ('immediate','temporal','delayed_null','phase'):
            self.assertEqual(tuple(basis[key].shape),(48,48))
            identity=torch.eye(48,dtype=torch.float64)
            self.assertTrue(torch.allclose(basis[key]@basis[key].T,identity,atol=2e-7))
        for metric in ('immediate','delayed_null','temporal'):
            for _,fraction in diagnostics['captures'][metric].items():
                self.assertTrue(0<=fraction<=1+1e-7)
        self.assertLess(diagnostics['analytic_jacobian_max_relative_gap'],1e-7)

    def test_small_heldout_sensitivity_independent(self):
        basis,_,train_hashes=fit_causal_decomposition(self.model,n_prompts=2,horizon=2)
        result=heldout_local_sensitivity(self.model,basis,prompt_seed=910709,n_prompts=2,horizon=2)
        self.assertEqual(len(result['local_rank_rel_1e-3']),2)
        self.assertTrue(all(0<=r<=48 for r in result['local_rank_rel_1e-3']))
        self.assertLess(result['max_analytic_jacobian_disagreement'],1e-7)
        for key in ('phase','delayed_null'):
            for m in ('current','deferred_given_current_null'):
                self.assertTrue(0<=result['basis_capture'][key][m]['mean']<=1+1e-7)

    def test_smoke_with_untrained_trajectory_has_no_fabricated_phase(self):
        identity=torch.eye(48,dtype=torch.float64)
        bases={k:identity for k in ('phase','temporal','delayed_null')}
        row=causal_phase_batch(self.model,bases,seed=910701,batch=4,depth=16,follow=2)
        self.assertEqual(row['n_items'],4)
        self.assertEqual(row['registered'],0)
        for c in row['conditions'].values():
            self.assertEqual(c['shift'],0)
            self.assertEqual(c['shift_bits'],[])
            self.assertLess(c['max_immediate_abs_logit_change'],float('inf'))

    def test_strict_checkpoint_digest(self):
        # Build a trusted synthetic fixture without any expensive training.
        raw=new_model(41522).eval()
        content={'seed':41522,'hyperparameters':{'n_nodes':8,'d_model':48},
                 'model':raw.state_dict(), 'weight_sha256':weight_sha256(raw)}
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'model.pt'
            torch.save(content,file)
            model,digest=model_from_receipt(file,41522)
            self.assertEqual(digest,content['weight_sha256'])
            content['weight_sha256']='00'*32
            torch.save(content,file)
            with self.assertRaises(ValueError):model_from_receipt(file,41522)

    def test_paired_bootstrap_reproducibility(self):
        def item(x,y):
            return {'registered':len(x),'conditions':{
                'target_phase8':{'shift_bits':x},
                'target_delayed8':{'shift_bits':y},
            }}
        fixture={'heldout':{'a':[item([1,1,0],[0,1,0]),item([1,0],[0,0])]}}
        comp=(('target_phase8','target_delayed8'),)
        one=paired_bootstrap(fixture,comparisons=comp,n_repeat=100,seed=1984)
        two=paired_bootstrap(fixture,comparisons=comp,n_repeat=100,seed=1984)
        self.assertEqual(one,two)
        self.assertAlmostEqual(one['target_phase8_minus_target_delayed8']['difference_rate'],0.4)

if __name__=='__main__':unittest.main()
